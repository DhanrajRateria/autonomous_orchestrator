import os
import logging
import asyncio
import time
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# Adjust imports based on your actual project structure
from federated_learning_framework.config import FrameworkConfig, ModelConfig, RobustnessConfig
from federated_learning_framework.crypto_engine import CryptoEngine
from federated_learning_framework.models import create_model
from federated_learning_framework.data_handler import DataHandler


class FederatedClient:
    """
    Federated Learning Client.
    Manages local data, model training, and interaction with the server.
    Supports homomorphic encryption for parameter exchange.
    """

    def __init__(self, client_id: str, config: FrameworkConfig, data_path: str, is_noisy: bool = False, is_adversarial: bool = False):
        """
        Initialize the client.
        Args:
            client_id: Unique identifier for this client.
            config: Global framework configuration.
            data_path: Path to this client's local dataset file (e.g., CSV).
        """
        self.client_id = client_id
        self.config = config
        self.local_data_path = data_path
        self.logger = logging.getLogger(f"federated.client.{self.client_id}")

        self.is_noisy = is_noisy
        self.is_adversarial = is_adversarial
        if self.is_noisy:
            self.logger.warning(f"Client {self.client_id} configured as NOISY. Label flip prob: {self.config.robustness.label_flip_probability}")
        if self.is_adversarial:
            self.logger.warning(f"Client {self.client_id} configured as ADVERSARIAL. Attack scale: {self.config.robustness.attack_scale_factor}")

        # --- System Setup ---
        self.checkpoint_dir = Path(config.system.checkpoint_dir) / f"client_{self.client_id}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(config.system.seed)
        np.random.seed(config.system.seed)
        try:
            self.device = torch.device(config.system.device)
            # Test CUDA availability if specified
            if "cuda" in config.system.device and not torch.cuda.is_available():
                 self.logger.warning(f"Device set to '{config.system.device}' but CUDA not available. Falling back to CPU.")
                 self.device = torch.device("cpu")
            elif "mps" in config.system.device and not torch.backends.mps.is_available():
                 self.logger.warning(f"Device set to '{config.system.device}' but MPS not available. Falling back to CPU.")
                 self.device = torch.device("cpu")

        except Exception as e:
             self.logger.error(f"Error setting device '{config.system.device}': {e}. Using CPU.", exc_info=True)
             self.device = torch.device("cpu")
        self.logger.info(f"Using device: {self.device}")


        # --- Crypto Engine ---
        self.crypto_engine = CryptoEngine(config.crypto)
        # Client only needs the public context, loaded later if needed

        # --- Data Handling ---
        # Pass task_type to DataHandler for correct target processing
        self.data_handler = DataHandler(config)
        self.train_dataloader = None
        self.val_dataloader = None
        self.train_size = 0

        # --- Model ---
        # Model is created *after* data is loaded to get definitive shapes
        self.model: Optional[nn.Module] = None

        # --- State ---
        self.current_round = 0
        self.training_history = []
        self.is_training = False
        self.initialized = False

        self.initial_global_params_np: Optional[Dict[str, np.ndarray]] = None

        self.logger.info(f"Federated client '{client_id}' created. Data: {data_path}")

    def _get_model_params_numpy(self, model: nn.Module) -> Dict[str, np.ndarray]:
        """Helper to get model parameters as a dictionary of numpy arrays."""
        return {name: param.cpu().detach().numpy().copy() for name, param in model.named_parameters()}

    async def initialize(self, public_context_bytes: Optional[bytes] = None):
        """
        Initialize client resources: load data, create model, load crypto context.
        Args:
            public_context_bytes: Serialized public TenSEAL context from the server.
        """
        if self.initialized:
            self.logger.warning("Client already initialized.")
            return True

        self.logger.info("Initializing client...")
        try:
            # 1. Load Data and get definitive shapes
            self.logger.info("Loading local dataset...")
            # Use client's specific data path, only request train/val split
            # Server handles test split on its own data
            self.train_dataloader, self.val_dataloader, _ = await self.data_handler.load_data(
                data_override_path=self.local_data_path,
                val_split=self.config.data.val_split,
                test_split=0.0 # Clients typically don't have a test set
            )

            if not self.train_dataloader:
                 raise RuntimeError("Failed to create training dataloader.")

            self.train_size = len(self.train_dataloader.dataset)
            val_size = len(self.val_dataloader.dataset) if self.val_dataloader else 0
            self.logger.info(f"Dataset loaded: {self.train_size} training samples, {val_size} validation samples.")

            # Get final input/output shapes from the data handler's config (updated during load)
            final_input_shape = self.data_handler.data_config.input_shape
            final_output_shape = self.data_handler.data_config.output_shape
            self.logger.info(f"Data loading complete. Final Input Shape: {final_input_shape}, Final Output Shape: {final_output_shape}")


            # 2. Create Model using final shapes
            self.logger.info("Creating local model...")
            self.model = create_model(
                model_config=self.config.model,
                input_shape=final_input_shape,
                output_shape=final_output_shape
            )
            self.model.to(self.device)
            self.logger.info(f"Model '{self.config.model.type}' created and moved to {self.device}.")
            # Log model structure
            self.logger.debug(f"Model structure:\n{self.model}")


            # 3. Load Crypto Context if enabled
            if self.crypto_engine.config.enabled:
                 if public_context_bytes:
                     self.logger.info("Loading public crypto context from server...")
                     self.crypto_engine.load_public_context(public_context_bytes)
                     if not self.crypto_engine.is_enabled():
                          self.logger.warning("Crypto engine failed to load context, HE disabled for this client.")
                 else:
                     self.logger.warning("HE is enabled in config, but no public context provided by server.")
                     # Should the client proceed without HE? Depends on system design.
                     # Let's disable it for this client for safety.
                     self.crypto_engine.config.enabled = False


            self.initialized = True
            self.logger.info(f"Client '{self.client_id}' initialization successful.")
            return True

        except Exception as e:
            self.logger.error(f"Client initialization failed: {e}", exc_info=True)
            self.initialized = False
            return False

    async def train(self, round_id: int, parameters: Union[Dict[str, np.ndarray], Dict[str, Any]],
                  encrypted: bool, config_update: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform local training for one round.
        Args:
            round_id: Current federated round number.
            parameters: Model parameters from the server (plain numpy arrays or encrypted dicts).
            encrypted: Flag indicating if parameters are encrypted.
            config_update: Dictionary with round-specific config (e.g., epochs, lr).
        Returns:
            Dictionary with training results and updated parameters (possibly encrypted).
        """
        if not self.initialized:
             msg = "Client not initialized. Cannot start training."
             self.logger.error(msg)
             return {"status": "error", "message": msg}
        if self.is_training:
            msg = "Client is already training. Ignoring new request."
            self.logger.warning(msg)
            return {"status": "busy", "message": msg}

        self.is_training = True
        self.current_round = round_id
        start_time = time.time()
        self.logger.info(f"Starting local training for round {round_id}...")

        # Get training parameters from config or update
        local_epochs = self.config.federated.local_epochs
        client_lr = self.config.federated.client_learning_rate
        if config_update:
            local_epochs = config_update.get("local_epochs", local_epochs)
            client_lr = config_update.get("learning_rate", client_lr)
            # Handle other potential overrides like proximal_mu if implementing FedProx

        self.logger.info(f"Round Config: Epochs={local_epochs}, LR={client_lr}")

        try:
            # 1. Update local model with server parameters
            self.logger.debug("Updating local model with server parameters...")
            await self._update_local_model(parameters, encrypted)

            # 2. Set up Optimizer (always use parameters from the current model state)
            # Consider allowing optimizer choice from config (Adam, etc.)
            # optimizer = optim.SGD(self.model.parameters(), lr=client_lr, momentum=0.9)
            optimizer = optim.Adam(self.model.parameters(), lr=client_lr) # NEW Adam
            self.logger.info(f"Using Adam optimizer with LR={client_lr}")

            # 3. Train for local epochs
            self.model.train() # Set model to training mode
            training_results = await self._train_epochs(optimizer, local_epochs)

            # --- Calculate Model Delta ---
            current_local_params_np = self._get_model_params_numpy(self.model)
            model_delta_np = {}
            if self.initial_global_params_np is None:
                self.logger.error("Initial global parameters not stored. Sending full model parameters instead of delta.")
                model_delta_np = current_local_params_np # Fallback
            else:
                for name in current_local_params_np:
                    if name in self.initial_global_params_np:
                        model_delta_np[name] = current_local_params_np[name] - self.initial_global_params_np[name]
                    else: # Should not happen if model structures are consistent
                        model_delta_np[name] = current_local_params_np[name]
                        self.logger.warning(f"Parameter {name} not in initial_global_params, sending full param value for delta.")
            self.logger.debug("Calculated model delta.")

            # --- Adversarial Client Logic: Model Poisoning (scaling delta) ---
            if self.is_adversarial:
                attack_scale = self.config.robustness.attack_scale_factor
                self.logger.warning(f"Client {self.client_id} is ADVERSARIAL: scaling delta by {attack_scale}.")
                for name in model_delta_np:
                    model_delta_np[name] = model_delta_np[name] * attack_scale

            params_to_send: Union[Dict[str, Any], Dict[str, np.ndarray]]
            params_encrypted: bool
            if self.crypto_engine.is_enabled():
                self.logger.info("Encrypting model delta...")
                # Use the existing encrypt_model_params which takes a dict of numpy arrays
                encrypted_delta = self.crypto_engine.encrypt_model_params(model_delta_np)
                if not encrypted_delta:
                    raise RuntimeError("Model delta encryption failed.")
                params_to_send = encrypted_delta
                params_encrypted = True
            else:
                params_to_send = model_delta_np
                params_encrypted = False

            # --- Calculate Quality Score ---
            avg_train_loss = training_results.get('train_loss') # Use train_loss from the _train_epochs result
            quality_score: Optional[float] = None
            if avg_train_loss is not None and not (np.isnan(avg_train_loss) or np.isinf(avg_train_loss)):
                if avg_train_loss < 0: # Should not happen with BCE/MSE/CE
                    self.logger.warning(f"Average train loss {avg_train_loss} is negative. Setting quality score to a low value.")
                    quality_score = self.config.quality.score_epsilon 
                else:
                    quality_score = 1.0 / (avg_train_loss + self.config.quality.score_epsilon)
            else:
                self.logger.warning(f"Invalid average train loss ({avg_train_loss}) for quality score calculation. Setting to low value.")
                quality_score = self.config.quality.score_epsilon # Default to a very small score if loss is problematic

            # 5. Prepare result package
            end_time = time.time()
            result = {
                "status": "success",
                "client_id": self.client_id,
                "round_id": round_id,
                "parameters": params_to_send,
                "encrypted": params_encrypted,
                "sample_size": self.train_size,
                "metrics": training_results, # Contains train_loss, train_acc, val_loss, val_acc
                "quality_score": quality_score,
                "train_duration_sec": end_time - start_time,
                "timestamp": datetime.now().isoformat()
            }

            # Store result in history
            self.training_history.append({
                "round": round_id,
                "metrics": training_results,
                "quality_score": quality_score,
                "duration": end_time - start_time
            })

            # Save checkpoint periodically (optional)
            if round_id % self.config.system.checkpoint_frequency == 0:
                self._save_checkpoint(f"round_{round_id}")

            self.logger.info(f"Local training round {round_id} completed in {end_time - start_time:.2f}s.")
            self.logger.info(f"Metrics: Train Loss={training_results['train_loss']:.4f}, "
                           f"Train Acc={training_results.get('train_accuracy', -1):.4f}, "
                           f"Val Loss={training_results.get('val_loss', -1):.4f}, "
                           f"Val Acc={training_results.get('val_accuracy', -1):.4f}")
            self.logger.info(f"Final calculated quality score for round {round_id}: {quality_score:.6e} (from avg_train_loss: {avg_train_loss})")
            return result

        except Exception as e:
            self.logger.error(f"Error during local training round {round_id}: {e}", exc_info=True)
            return {"status": "error", "client_id": self.client_id, "round_id": round_id, "message": str(e)}
        finally:
            self.is_training = False

    async def _train_epochs(self, optimizer: optim.Optimizer, epochs: int) -> Dict[str, float]:
        """Internal method to run training loop for specified epochs."""

        task_type = self.config.model.task_type
        is_binary = task_type == "binary_classification"
        num_classes = self.config.data.output_shape[0] # From data handler config
        pos_w_config = self.config.data.pos_weight

        criterion_pos_weight = None
        if is_binary and pos_w_config is not None:
            criterion_pos_weight = torch.tensor([pos_w_config], device=self.device)
            self.logger.debug(f"Using pos_weight for BCEWithLogitsLoss: {criterion_pos_weight.item():.4f}")

        # --- Select Loss Function ---
        if is_binary:
             # Assumes model output is single raw logit
            criterion = nn.BCEWithLogitsLoss()
            self.logger.debug("Using BCEWithLogitsLoss for binary classification.")
        elif task_type == "classification":
             # Assumes model output is raw logits [batch, num_classes]
            criterion = nn.CrossEntropyLoss()
            self.logger.debug(f"Using CrossEntropyLoss for multi-class classification ({num_classes} classes).")
        elif task_type == "regression":
            criterion = nn.MSELoss() # Or MAELoss etc.
            self.logger.debug("Using MSELoss for regression.")
        else:
            raise ValueError(f"Unsupported task_type '{task_type}' for loss selection.")

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            self.model.train() # Ensure model is in training mode each epoch

            for batch_idx, (inputs, targets) in enumerate(self.train_dataloader):
                inputs = inputs.to(self.device)
                original_targets = targets.to(self.device) # Keep original for accuracy calculation
                
                # --- Noisy Client Logic: Label Flipping ---
                current_batch_targets = original_targets.clone() # Work on a copy for potential flipping
                if self.is_noisy:
                    if is_binary: # Assuming binary targets are 0 or 1
                        # Probability for each sample in the batch
                        flip_probabilities = torch.rand(current_batch_targets.shape, device=self.device)
                        # Mask where flipping occurs
                        flip_mask = flip_probabilities < self.config.robustness.label_flip_probability
                        # Flip: 1 -> 0, 0 -> 1. Ensure target dtype is maintained.
                        flipped_values = 1 - current_batch_targets[flip_mask].long() # if targets are long for CE
                        if current_batch_targets.dtype == torch.float32: # For BCE
                             flipped_values = 1.0 - current_batch_targets[flip_mask]

                        current_batch_targets[flip_mask] = flipped_values.type(current_batch_targets.dtype)
                        if flip_mask.any():
                             self.logger.debug(f"Epoch {epoch+1}, Batch {batch_idx+1}: Flipped {flip_mask.sum().item()} labels for noisy training.")
                    else:
                         self.logger.warning("Noisy client label flipping not implemented for non-binary tasks yet.")

                optimizer.zero_grad()
                outputs = self.model(inputs)

                # --- Loss Calculation ---
                if is_binary:
                     # BCEWithLogitsLoss expects output [N] or [N, 1], target [N] or [N, 1] float
                    loss = criterion(outputs.squeeze(), current_batch_targets.float()) # Ensure targets are float
                elif task_type == "classification":
                     # CrossEntropyLoss expects output [N, C], target [N] long
                    loss = criterion(outputs, current_batch_targets.long()) # Ensure targets are long
                else: # Regression
                     # MSELoss expects output and target of same shape
                    loss = criterion(outputs.squeeze(), current_batch_targets.float()) # Ensure targets are float


                # --- Safe Loss Handling & Backpropagation ---
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: NaN or Inf loss detected! Loss value: {loss.item()}. Skipping batch update.")
                    # Skip backward/step for this batch to avoid propagating NaN/Inf
                    # Alternatively, use a small fixed loss, but skipping is often safer.
                    # loss = torch.tensor(10.0, device=self.device, requires_grad=True) # Example fixed loss
                    continue # Skip optimizer step


                loss.backward()
                # Gradient Clipping (optional, add to config if needed)
                # clip_value = self.config.privacy.gradient_clipping
                # if clip_value > 0:
                #    nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
                clip_value = self.config.privacy.gradient_clipping # Get value from config
                if clip_value > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
                optimizer.step()

                # --- Statistics Update ---
                batch_loss = loss.item()
                epoch_loss += batch_loss * inputs.size(0)
                epoch_samples += inputs.size(0)

                # --- Accuracy Calculation (Classification only) ---
                with torch.no_grad(): # Use original_targets for calculating accuracy
                    if is_binary:
                        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                        correct = (predicted == original_targets.float()).sum().item()
                    elif task_type == "classification":
                        _, predicted = torch.max(outputs, 1)
                        correct = (predicted == original_targets.long()).sum().item()
                    else:
                        correct = 0
                    epoch_correct += correct

                if (batch_idx + 1) % 50 == 0: # Log progress every 50 batches
                     self.logger.debug(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(self.train_dataloader)}, Loss: {batch_loss:.4f}")


            # --- Epoch Summary ---
            epoch_avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0
            epoch_duration = time.time() - epoch_start_time
            log_msg = f"Epoch {epoch+1}/{epochs} finished in {epoch_duration:.2f}s. Avg Train Loss: {epoch_avg_loss:.4f}"

            if task_type != "regression" and epoch_samples > 0:
                epoch_accuracy = epoch_correct / epoch_samples
                log_msg += f", Train Accuracy: {epoch_accuracy:.4f}"
                total_train_correct += epoch_correct

            self.logger.info(log_msg)

            total_train_loss += epoch_loss
            total_train_samples += epoch_samples


        # --- Final Training Metrics ---
        final_avg_train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0
        final_train_accuracy = total_train_correct / total_train_samples if task_type != "regression" and total_train_samples > 0 else None

        # --- Evaluate on Validation Set (if available) ---
        val_metrics = {}
        if self.val_dataloader:
             self.logger.info("Evaluating model on local validation set...")
             val_metrics = await self._evaluate()
             self.logger.info(f"Validation Metrics: Loss={val_metrics.get('val_loss', -1):.4f}, Acc={val_metrics.get('val_accuracy', -1):.4f}")
        else:
             self.logger.info("No validation set available for evaluation.")


        results = {
            "train_loss": final_avg_train_loss,
        }
        if final_train_accuracy is not None:
             results["train_accuracy"] = final_train_accuracy
        if val_metrics:
             results.update(val_metrics) # Adds val_loss, val_accuracy

        return results


    async def _evaluate(self) -> Dict[str, float]:
        """Evaluate the current model on the local validation set."""
        if not self.val_dataloader:
            return {}
        if not self.model:
             self.logger.error("Cannot evaluate, model not initialized.")
             return {}

        self.model.eval() # Set model to evaluation mode
        task_type = self.config.model.task_type
        is_binary = task_type == "binary_classification"
        num_classes = self.config.data.output_shape[0]

        # Select Loss Function (consistent with training)
        if is_binary: criterion = nn.BCEWithLogitsLoss()
        elif task_type == "classification": criterion = nn.CrossEntropyLoss()
        else: criterion = nn.MSELoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in self.val_dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)

                # Loss Calculation
                if is_binary: loss = criterion(outputs.squeeze(), targets.float())
                elif task_type == "classification": loss = criterion(outputs, targets.long())
                else: loss = criterion(outputs.squeeze(), targets.float())

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                # Accuracy Calculation
                if is_binary:
                    predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                    correct = (predicted == targets.float()).sum().item()
                elif task_type == "classification":
                    _, predicted = torch.max(outputs, 1)
                    correct = (predicted == targets.long()).sum().item()
                else:
                    correct = 0 # No accuracy for regression

                total_correct += correct

        # Compute final metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        metrics = {"val_loss": avg_loss}
        if task_type != "regression" and total_samples > 0:
            accuracy = total_correct / total_samples
            metrics["val_accuracy"] = accuracy

        self.model.train() # Set model back to training mode
        return metrics


    async def _update_local_model(self, parameters: Union[Dict[str, np.ndarray], Dict[str, Any]], encrypted: bool):
        """Update local model weights with parameters received from the server."""
        self.logger.debug(f"Received parameters are {'encrypted' if encrypted else 'plain'}.")
        if encrypted:
            if not self.crypto_engine.is_enabled():
                 self.logger.error("Received encrypted parameters, but HE is disabled on client. Cannot update model.")
                 # Option: Fallback to current weights? Or raise error?
                 # Let's keep current weights and log error.
                 return
            try:
                self.logger.info("Decrypting parameters...")
                start_dec = time.time()
                # This method handles decryption and loading into the model
                self.crypto_engine.decrypt_to_torch_params(self.model, parameters)
                self.logger.info(f"Decryption and model update took {time.time() - start_dec:.2f}s")
            except Exception as e:
                self.logger.error(f"Failed to decrypt and update model parameters: {e}", exc_info=True)
                # Keep existing model parameters
                self.logger.warning("Using existing model parameters due to decryption/update failure.")

        else:
            # Parameters are plain numpy arrays
            self.logger.debug("Updating model with plain parameters...")
            try:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in parameters:
                             param_data = parameters[name]
                             if not isinstance(param_data, np.ndarray):
                                  self.logger.warning(f"Plain parameter '{name}' is not a numpy array (type: {type(param_data)}). Skipping.")
                                  continue

                             param_tensor = torch.from_numpy(param_data).to(param.device) # Move to correct device
                             if param.shape == param_tensor.shape:
                                 param.copy_(param_tensor)
                             else:
                                 # Critical mismatch - should not happen if server sends correct params
                                 self.logger.error(f"Shape mismatch for parameter '{name}': Model={param.shape}, Received={param_tensor.shape}. Skipping update for this parameter.")
                                 # DO NOT RESHAPE - indicates a fundamental issue.
                        else:
                             self.logger.warning(f"Parameter '{name}' not found in received plain parameters.")
            except Exception as e:
                self.logger.error(f"Error updating model with plain parameters: {e}", exc_info=True)
                self.logger.warning("Using existing model parameters due to plain update failure.")

        self.initial_global_params_np = self._get_model_params_numpy(self.model)
        self.logger.debug("Stored initial global parameters for delta calculation.")

    def _save_checkpoint(self, tag: str):
        """Save model checkpoint."""
        if not self.model:
             self.logger.warning("Cannot save checkpoint, model not initialized.")
             return
        try:
            checkpoint_path = self.checkpoint_dir / f"model_round_{tag}.pt"
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "round": self.current_round,
                "timestamp": datetime.now().isoformat(),
                # Optional: Add optimizer state if needed for resuming training locally
                # "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}", exc_info=True)


    def get_client_info(self) -> Dict[str, Any]:
        """Get client information for registration."""
        if not self.initialized:
             # Provide basic info even if not fully initialized
             return {
                "client_id": self.client_id,
                "status": "initializing",
                "timestamp": datetime.now().isoformat()
             }
        return {
            "client_id": self.client_id,
            "train_size": self.train_size,
            "device": str(self.device),
            "supports_encryption": self.crypto_engine.config.enabled, # Reflects if context was loaded okay
            "status": "idle", # Or "ready"
            "input_shape": self.config.data.input_shape,
            "output_shape": self.config.data.output_shape,
            "task_type": self.config.model.task_type,
            "timestamp": datetime.now().isoformat()
        }

    def load_checkpoint(self, path: str) -> bool:
        """Load model from checkpoint."""
        if not self.model:
             self.logger.error("Cannot load checkpoint, model not initialized.")
             return False
        try:
            checkpoint_path = Path(path)
            if not checkpoint_path.is_file():
                 self.logger.error(f"Checkpoint file not found: {path}")
                 return False

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.current_round = checkpoint.get("round", self.current_round) # Use current if not in ckpt
            # Optional: Load optimizer state
            # if "optimizer_state_dict" in checkpoint and optimizer:
            #    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info(f"Loaded model checkpoint from {path} (Round {self.current_round})")
            return True
        except Exception as e:
            self.logger.error(f"Error loading checkpoint from {path}: {e}", exc_info=True)
            return False