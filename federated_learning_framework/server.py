# server.py
import os
import logging
import asyncio
import time
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import tenseal as ts
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
import uuid
from datetime import datetime
import copy # For deep copying parameters during aggregation
import base64 # For encoding public context bytes

# Adjust imports based on your actual project structure
from federated_learning_framework.config import FrameworkConfig
from federated_learning_framework.crypto_engine import CryptoEngine
from federated_learning_framework.models import create_model
from federated_learning_framework.data_handler import DataHandler
from federated_learning_framework.privacy import DifferentialPrivacy


class FederatedServer:
    """
    Federated Learning Server.
    Coordinates client training, aggregates model updates securely (optional HE),
    and evaluates the global model.
    """

    def __init__(self, config: FrameworkConfig):
        self.logger = logging.getLogger("federated.server")
        self.config = config

        # --- System Setup ---
        self.checkpoint_dir = Path(config.system.checkpoint_dir) / "server"
        self.result_dir = Path(config.system.result_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
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
        # Server initializes the context and holds the secret key
        self.crypto_engine = CryptoEngine(config.crypto)
        self.public_context_bytes: Optional[bytes] = None
        if self.crypto_engine.is_enabled():
            self.public_context_bytes = self.crypto_engine.get_public_context()
            if self.public_context_bytes is None:
                 self.logger.error("Failed to get public context bytes, disabling HE.")
                 self.config.crypto.enabled = False # Update global config state
                 self.crypto_engine.config.enabled = False # Update engine's config state


        # --- Privacy Engine ---
        self.dp_engine: Optional[DifferentialPrivacy] = None
        if config.privacy.differential_privacy:
            self.logger.info("Initializing Differential Privacy engine...")
            self.dp_engine = DifferentialPrivacy(
                epsilon=config.privacy.dp_epsilon,
                delta=config.privacy.dp_delta,
                noise_multiplier=config.privacy.dp_noise_multiplier,
                clipping_norm=config.privacy.gradient_clipping # Note: Clipping usually done client-side per-sample grad
            )

        # --- Data Handling (for server-side evaluation) ---
        # Pass task_type for correct target processing
        self.data_handler = DataHandler(config)
        self.val_dataloader = None
        self.test_dataloader = None

        # --- Global Model ---
        # Create model using config shapes initially. DataHandler might update these.
        self.model: Optional[nn.Module] = None # Initialized in start() after data loading
        # Server-side optimizer (e.g., for FedAvgM, FedAdam - not fully used in basic FedAvg)
        self.optimizer: Optional[optim.Optimizer] = None


        # --- State ---
        self.current_round = 0
        self.max_rounds = config.federated.communication_rounds
        self.clients: Dict[str, Dict[str, Any]] = {} # client_id -> client_info, status, etc.
        self.selected_clients_this_round: List[str] = []
        self.client_updates_this_round: Dict[str, Dict[str, Any]] = {} # client_id -> result_dict
        self.training_history: List[Dict[str, Any]] = []
        self.global_eval_metrics: Dict[str, float] = {} # Latest eval metrics
        self.best_model_metric_value: float = -1.0 # Track best validation metric (e.g., accuracy)
        self.best_model_round: int = 0

        # Locks for concurrent access control
        self._model_lock = asyncio.Lock()
        self._client_registry_lock = asyncio.Lock()
        self._round_lock = asyncio.Lock() # To manage round state transitions

        self.is_running = False
        self.logger.info(f"Federated server initialized for project '{config.project_name}'.")


    async def start(self):
        """Start the federated learning server process."""
        if self.is_running:
            self.logger.warning("Server already running.")
            return

        self.is_running = True
        self.logger.info("Starting federated learning server...")

        try:
            # 1. Initialize server data (validation/test sets)
            await self._initialize_server_data()

            # 2. Create Global Model using final shapes from data handler
            self.logger.info("Creating global model...")
            # Shapes should be finalized after loading server data
            final_input_shape = self.data_handler.data_config.input_shape
            final_output_shape = self.data_handler.data_config.output_shape
            self.model = create_model(
                model_config=self.config.model,
                input_shape=final_input_shape,
                output_shape=final_output_shape
            )
            self.model.to(self.device)
            self.logger.info(f"Global model '{self.config.model.type}' created and moved to {self.device}.")
            self.logger.debug(f"Model structure:\n{self.model}")


            # 3. Initialize Server Optimizer
            # Use SGD for server optimizer (e.g., FedAvgM) or Adam if needed
            # Note: For basic FedAvg, this optimizer isn't strictly necessary for updating the model
            # but might be used if server learning rate (eta) is < 1.
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.federated.server_learning_rate)
            # self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.federated.server_learning_rate)


            # 4. Save initial model checkpoint
            self._save_checkpoint("initial")

            # 5. Save crypto context files if HE is enabled
            if self.crypto_engine.is_enabled():
                context_path = self.checkpoint_dir / "crypto_context.tenseal"
                self.crypto_engine.save_context_to_file(str(context_path))

            self.logger.info("Server startup complete. Waiting for clients...")

        except Exception as e:
             self.logger.error(f"Server startup failed: {e}", exc_info=True)
             self.is_running = False
             # Perform cleanup?
             raise # Re-raise exception to indicate startup failure


    async def stop(self):
        """Stop the federated learning server."""
        if not self.is_running:
            self.logger.warning("Server is not running.")
            return

        self.is_running = False
        self.logger.info("Stopping federated learning server...")

        # Save final model checkpoint
        if self.model:
            self._save_checkpoint("final")

        # Save training history
        self._save_history()

        # Potential cleanup (e.g., disconnect clients gracefully if using persistent connections)

        self.logger.info("Federated server stopped.")


    async def _initialize_server_data(self):
        """Load validation and test data used by the server."""
        self.logger.info("Initializing server evaluation data...")
        try:
            # Load using the main data path from config
            # We only need validation and test sets on the server
            _, self.val_dataloader, self.test_dataloader = await self.data_handler.load_data(
                data_override_path=None, # Use config.data.data_path
                val_split=self.config.data.val_split,
                test_split=self.config.data.test_split
            )

            if not self.val_dataloader and not self.test_dataloader:
                 self.logger.warning("No validation or test data loaded for server evaluation. Evaluation will be skipped.")
            else:
                 val_info = f"{len(self.val_dataloader.dataset)} samples" if self.val_dataloader else "None"
                 test_info = f"{len(self.test_dataloader.dataset)} samples" if self.test_dataloader else "None"
                 self.logger.info(f"Server evaluation data loaded: Validation={val_info}, Test={test_info}")

        except FileNotFoundError as e:
             self.logger.error(f"Server data loading failed: {e}. Ensure config.data.data_path points to the full dataset.")
             raise # Stop server startup if eval data is missing
        except Exception as e:
            self.logger.error(f"Error initializing server data: {e}", exc_info=True)
            raise

    # --- Client Management ---

    async def register_client(self, client_id: str, client_info: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new client."""
        async with self._client_registry_lock:
            if client_id in self.clients:
                self.logger.warning(f"Client {client_id} attempted to register again.")
                # Optionally update existing info?
                self.clients[client_id]["last_seen"] = datetime.now()
                self.clients[client_id]["status"] = "active" # Mark as active on re-register attempt
                response_status = "already_registered"
            else:
                self.logger.info(f"Registering new client: {client_id} - Info: {client_info}")
                self.clients[client_id] = {
                    "info": client_info,
                    "last_seen": datetime.now(),
                    "status": "registered", # Initial status
                    "rounds_participated": 0,
                    "last_error": None
                }
                response_status = "registered"

            # Provide necessary info back to client (e.g., public context)
            registration_response = {
                 "status": response_status,
                 "public_context": base64.b64encode(self.public_context_bytes).decode('ascii') if self.public_context_bytes else None,
                 "server_config": { # Send relevant parts of server config if needed
                     "task_type": self.config.model.task_type,
                     # Add other necessary info
                 }
            }
            return registration_response


    async def unregister_client(self, client_id: str):
        """Unregister a client."""
        async with self._client_registry_lock:
            if client_id in self.clients:
                self.logger.info(f"Unregistering client: {client_id}")
                del self.clients[client_id]
                return True
            else:
                self.logger.warning(f"Attempted to unregister unknown client: {client_id}")
                return False

    async def client_heartbeat(self, client_id: str, status_update: Optional[Dict[str, Any]] = None):
        """Process heartbeat from a client."""
        async with self._client_registry_lock:
            if client_id in self.clients:
                self.clients[client_id]["last_seen"] = datetime.now()
                if status_update:
                    current_status = status_update.get("status", "active")
                    self.clients[client_id]["status"] = current_status
                    if current_status == "error":
                         self.clients[client_id]["last_error"] = status_update.get("message", "Unknown error")
                else:
                     # If no status update, assume active if previously registered/active
                     if self.clients[client_id]["status"] != "error":
                          self.clients[client_id]["status"] = "active"

                # self.logger.debug(f"Heartbeat received from client: {client_id}, Status: {self.clients[client_id]['status']}")
                return True
            else:
                self.logger.warning(f"Heartbeat from unknown client: {client_id}")
                return False

    async def get_available_clients(self) -> List[str]:
        """Get clients considered available for selection (recently active, not error state)."""
        async with self._client_registry_lock:
            available_clients = []
            # Consider clients active if seen within last 5 minutes (adjust as needed)
            cutoff_time = datetime.now().timestamp() - 300
            for client_id, client_data in self.clients.items():
                 # Check last seen time and status is not 'error' or 'busy' (add 'busy' if client reports it)
                if client_data["last_seen"].timestamp() >= cutoff_time and \
                   client_data.get("status", "unknown") not in ["error", "busy", "training"]:
                    available_clients.append(client_id)
            return available_clients

    async def _prepare_round_start_config(self) -> Optional[Dict[str, Any]]:
        """Get current model parameters (ALWAYS PLAIN for clients) and package round config."""
        async with self._model_lock:
            if not self.model:
                self.logger.error("Global model not available for preparing round config.")
                return None
            try:
                self.logger.debug("Cloning model state for parameter extraction.")
                model_copy = copy.deepcopy(self.model).cpu()

                # --- CHANGE: Always send plain parameters to clients ---
                self.logger.info("Exporting PLAIN global model parameters for round start...")
                model_params = {name: param.cpu().detach().numpy()
                                for name, param in model_copy.named_parameters()}
                params_encrypted = False # Set flag to False
                # --- END CHANGE ---

                # Round-specific configuration for clients
                client_round_config = {
                    "local_epochs": self.config.federated.local_epochs,
                    "learning_rate": self.config.federated.client_learning_rate,
                    "batch_size": self.config.data.batch_size,
                    # "proximal_mu": self.config.federated.proximal_mu
                }

                round_config_package = {
                    "round_id": self.current_round,
                    "parameters": model_params, # Plain parameters
                    "encrypted": params_encrypted, # False
                    "client_config": client_round_config,
                    "timestamp": datetime.now().isoformat()
                }
                return round_config_package

            except Exception as e:
                self.logger.error(f"Error preparing round start config: {e}", exc_info=True)
                return None



    async def submit_update(self, client_id: str, round_id: int, update_data: Dict[str, Any]):
        """Receive and store a client's update for the current round."""
        async with self._round_lock: # Ensure consistent round check and update storage
             if round_id != self.current_round:
                 self.logger.warning(f"Received update from client {client_id} for wrong round (Expected {self.current_round}, Got {round_id}). Discarding.")
                 return False

             if client_id not in self.selected_clients_this_round:
                 self.logger.warning(f"Received update from non-selected client {client_id} for round {self.current_round}. Discarding.")
                 return False

             if client_id in self.client_updates_this_round:
                 self.logger.warning(f"Received duplicate update from client {client_id} for round {self.current_round}. Overwriting previous.")
                 # Or discard? Overwriting might be okay if client resends.

             # Validate update structure (basic check)
             if "status" not in update_data or update_data["status"] != "success":
                  error_msg = update_data.get('message', 'Unknown error')
                  self.logger.error(f"Received failed update from client {client_id} for round {self.current_round}. Error: {error_msg}")
                  # Mark client as having errored this round?
                  async with self._client_registry_lock:
                       if client_id in self.clients:
                           self.clients[client_id]["status"] = "error"
                           self.clients[client_id]["last_error"] = error_msg
                  # Don't store the failed update for aggregation
                  return False

             if "parameters" not in update_data or "sample_size" not in update_data:
                  self.logger.error(f"Received incomplete update from client {client_id} (missing params or sample size). Discarding.")
                  return False


             # Store the valid update
             self.client_updates_this_round[client_id] = update_data
             self.logger.info(f"Stored update from client {client_id} for round {self.current_round}. ({len(self.client_updates_this_round)}/{len(self.selected_clients_this_round)} received)")

             # Update client status (optional, heartbeat might handle this)
             async with self._client_registry_lock:
                  if client_id in self.clients:
                       self.clients[client_id]["status"] = "update_received"
                       self.clients[client_id]["rounds_participated"] += 1

             return True


    async def _aggregate_and_update(self):
        """Aggregate valid client updates and update the global model."""
        self.logger.info("Starting aggregation process...")
        aggregation_start_time = time.time()

        valid_updates = [upd for upd in self.client_updates_this_round.values() if upd.get("status") == "success"]
        if not valid_updates:
            self.logger.warning("No valid updates available for aggregation.")
            return

        # --- Extract Quality Scores and Sample Sizes ---
        client_quality_scores_raw = [update.get("quality_score") for update in valid_updates]
        client_sample_sizes = [update.get("sample_size", 0) for update in valid_updates]
        total_samples = sum(client_sample_sizes)

        self.logger.info(f"Raw quality scores received: {client_quality_scores_raw}")

        weights: List[float]

        if self.config.quality.enabled:
            self.logger.info("Using quality-aware aggregation.")
            # Filter out None, NaN, Inf scores before processing
            valid_scores_indices = [i for i, s in enumerate(client_quality_scores_raw)
                                    if s is not None and not np.isnan(s) and not np.isinf(s) and s > 0] # Ensure positive scores

            if not valid_scores_indices:
                self.logger.warning("No valid positive quality scores reported. Falling back to FedAvg (sample-size based weights).")
                weights = [s / total_samples if total_samples > 0 else 1.0 / len(valid_updates) for s in client_sample_sizes]
            else:
                processed_scores = np.array([client_quality_scores_raw[i] for i in valid_scores_indices])
                self.logger.info(f"Processed (valid positive) scores: {processed_scores}")

                if self.config.quality.robust_score_aggregation and len(processed_scores) > 1: # Percentile needs >1 score
                    lower_p = self.config.quality.score_clip_percentile_lower
                    upper_p = self.config.quality.score_clip_percentile_upper
                    lower_bound = np.percentile(processed_scores, lower_p)
                    upper_bound = np.percentile(processed_scores, upper_p)
                    
                    # Ensure lower_bound is not excessively large if all scores are tiny, and upper_bound not too small
                    # This can happen if all scores are identical or nearly identical.
                    if upper_bound <= lower_bound and len(processed_scores) > 0: # If all scores are same, or range is tiny
                        # If upper_bound is problematic (e.g. 0 or negative), use a small positive value or mean
                        if upper_bound <= self.config.quality.score_epsilon:
                            upper_bound = max(np.mean(processed_scores), self.config.quality.score_epsilon*2)
                        if lower_bound >= upper_bound: # if lower bound ended up higher
                            lower_bound = upper_bound / 2.0 # or some other sensible small value

                    self.logger.info(f"Quality scores: Min={processed_scores.min():.4e}, Max={processed_scores.max():.4e}, Mean={processed_scores.mean():.4e}. "
                                   f"Clipping to percentile range [{lower_p}th={lower_bound:.4e}, {upper_p}th={upper_bound:.4e}]")
                    clipped_scores_values = np.clip(processed_scores, lower_bound, upper_bound)
                else:
                    clipped_scores_values = processed_scores
                    if len(processed_scores) <=1 and self.config.quality.robust_score_aggregation:
                        self.logger.info("Robust score aggregation enabled, but too few scores (<2) to apply percentile clipping. Using raw valid scores.")
                    else:
                        self.logger.info("Robust score aggregation disabled or not applicable. Using raw valid scores.")

                # Normalize clipped scores to sum to 1 to be used as weights
                # Assign weights back to the original 'valid_updates' structure
                temp_weights = {} # client_id -> weight
                current_sum_clipped_scores = np.sum(clipped_scores_values)
                if current_sum_clipped_scores > 1e-9: # Avoid division by zero
                    normalized_clipped_scores = clipped_scores_values / current_sum_clipped_scores
                    for i, original_idx in enumerate(valid_scores_indices):
                        client_id = valid_updates[original_idx]["client_id"]
                        temp_weights[client_id] = normalized_clipped_scores[i]
                else:
                    self.logger.warning("Sum of (clipped) quality scores is zero or negligible. Falling back to FedAvg (sample-size based weights).")
                    # Fallback logic assigned later

                # Create final weights list matching 'valid_updates' order
                final_quality_weights = []
                sum_final_weights = 0
                for update in valid_updates:
                    w = temp_weights.get(update["client_id"], 0.0) # Default to 0 if score was invalid
                    final_quality_weights.append(w)
                    sum_final_weights += w
                
                if sum_final_weights > 1e-9:
                    weights = [w / sum_final_weights for w in final_quality_weights] # Re-normalize
                else: # Fallback if all quality based weights ended up zero
                    self.logger.warning("All quality-based weights are zero. Falling back to FedAvg (sample-size based weights).")
                    weights = [s / total_samples if total_samples > 0 else 1.0 / len(valid_updates) for s in client_sample_sizes]

        else: # Quality-aware aggregation disabled, use standard FedAvg
            self.logger.info("Quality-aware aggregation disabled. Using FedAvg (sample-size based weights).")
            if total_samples == 0:
                self.logger.warning("Total sample size is zero. Using equal weights for aggregation.")
                weights = [1.0 / len(valid_updates)] * len(valid_updates)
            else:
                weights = [s / total_samples for s in client_sample_sizes]

        self.logger.info(f"Aggregating {len(valid_updates)} delta updates. Aggregation weights: {[f'{w:.3f}' for w in weights]}")

        are_updates_encrypted = valid_updates[0].get("encrypted", False)
        self.logger.info(f"Updates (deltas) are {'encrypted' if are_updates_encrypted else 'plain'}.")

        async with self._model_lock:
            try:
                aggregated_delta_dict: Optional[Dict[str, Any]] = None # Stores processed delta (encrypted dict or plain numpy dict)

                if are_updates_encrypted:
                    if not self.crypto_engine.is_enabled():
                        self.logger.error("Received encrypted deltas but HE is disabled on server. Cannot aggregate.")
                        return
                    # Secure Aggregation of DELTAS using HE
                    # _aggregate_encrypted_updates needs list of encrypted delta dicts and weights
                    client_encrypted_deltas = [upd["parameters"] for upd in valid_updates] # These are Dict[str, EncryptedDataStruct]
                    aggregated_delta_dict = await self._aggregate_encrypted_deltas_he(client_encrypted_deltas, weights)
                    if aggregated_delta_dict is None:
                        raise RuntimeError("Encrypted delta aggregation failed.")
                else:
                    # Standard FedAvg Aggregation on plain DELTAS
                    client_plain_deltas = [upd["parameters"] for upd in valid_updates] # These are Dict[str, np.ndarray]
                    aggregated_delta_dict = self._aggregate_plain_deltas(client_plain_deltas, weights)
                    if aggregated_delta_dict is None:
                        raise RuntimeError("Plain delta aggregation failed.")

                # --- Decrypt aggregated delta if it was encrypted ---
                decrypted_final_delta_np: Dict[str, np.ndarray]
                if are_updates_encrypted:
                    if aggregated_delta_dict is None: # Should have been caught by RuntimeError above
                        self.logger.error("Encrypted aggregated delta is None before decryption.")
                        return
                    self.logger.info("Decrypting aggregated delta...")
                    start_dec = time.time()
                    decrypted_final_delta_np = self.crypto_engine.decrypt_model_params(aggregated_delta_dict)
                    self.logger.info(f"Decryption of aggregated delta took {time.time() - start_dec:.2f}s")
                    if not decrypted_final_delta_np : # If decryption returns empty dict due to errors
                        raise RuntimeError("Aggregated delta decryption failed or yielded no parameters.")
                else:
                    decrypted_final_delta_np = aggregated_delta_dict # It's already Dict[str, np.ndarray]

                # --- Apply Aggregated Delta to Global Model ---
                self.logger.info("Applying aggregated delta to global model...")
                num_params_updated = 0
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in decrypted_final_delta_np:
                            delta_val = decrypted_final_delta_np[name]
                            if not isinstance(delta_val, np.ndarray):
                                self.logger.error(f"Decrypted delta for '{name}' is not a numpy array (type: {type(delta_val)}). Skipping update for this param.")
                                continue
                            
                            delta_tensor = torch.from_numpy(delta_val).to(param.device)
                            if param.shape == delta_tensor.shape:
                                param.add_(delta_tensor) # GLOBAL_MODEL = GLOBAL_MODEL + AGGREGATED_DELTA
                                num_params_updated +=1
                            else:
                                self.logger.error(f"Shape mismatch applying delta for '{name}': ModelParam={param.shape}, AggDelta={delta_tensor.shape}. Skipping.")
                        else:
                            self.logger.warning(f"Aggregated delta for parameter '{name}' not found. Parameter unchanged.")
                
                if num_params_updated == 0 and len(list(self.model.parameters())) > 0:
                    self.logger.error("No parameters in the global model were updated with the aggregated delta. Check delta content and names.")
                else:
                    self.logger.info(f"Successfully applied aggregated delta to {num_params_updated} model parameter groups.")


                # Apply Server-Side DP Noise (Optional)
                if self.dp_engine and self.config.privacy.differential_privacy:
                    self.logger.info("Applying differential privacy noise to the global model...")
                    self.dp_engine.add_noise_to_model(self.model) # Applies to full model params

                aggregation_duration = time.time() - aggregation_start_time
                self.logger.info(f"Aggregation and model update (with deltas) completed in {aggregation_duration:.2f}s.")

            except Exception as e:
                self.logger.error(f"Error during aggregation and model update: {e}", exc_info=True)

        # # --- Prepare weights (FedAvg) ---
        # total_samples = sum(update["sample_size"] for update in valid_updates)
        # if total_samples == 0:
        #     self.logger.warning("Total sample size is zero. Using equal weights for aggregation.")
        #     weights = [1.0 / len(valid_updates)] * len(valid_updates)
        # else:
        #     weights = [update["sample_size"] / total_samples for update in valid_updates]

        # self.logger.info(f"Aggregating {len(valid_updates)} updates using FedAvg. Total samples: {total_samples}")
        # # self.logger.debug(f"Aggregation weights: {weights}")

        # # Determine if updates are encrypted (check the first valid update)
        # are_updates_encrypted = valid_updates[0].get("encrypted", False)
        # self.logger.info(f"Updates are {'encrypted' if are_updates_encrypted else 'plain'}.")


        # # --- Perform Aggregation ---
        # async with self._model_lock:
        #      try:
        #          if are_updates_encrypted:
        #              if not self.crypto_engine.is_enabled():
        #                  self.logger.error("Received encrypted updates but HE is disabled on server. Cannot aggregate.")
        #                  return
        #              # Secure Aggregation using HE
        #              aggregated_params = await self._aggregate_encrypted_updates(valid_updates, weights)
        #              if aggregated_params is None:
        #                   raise RuntimeError("Encrypted aggregation failed.")

        #              # Decrypt and update model
        #              self.logger.info("Decrypting aggregated parameters...")
        #              start_dec = time.time()
        #              self.crypto_engine.decrypt_to_torch_params(self.model, aggregated_params)
        #              self.logger.info(f"Decryption and model update took {time.time() - start_dec:.2f}s")

        #          else:
        #              # Standard FedAvg Aggregation
        #              aggregated_params = self._aggregate_plain_updates(valid_updates, weights)
        #              if aggregated_params is None:
        #                    raise RuntimeError("Plain aggregation failed.")

        #              # Update model with plain aggregated parameters
        #              self.logger.info("Updating global model with plain aggregated parameters...")
        #              with torch.no_grad():
        #                  for name, param in self.model.named_parameters():
        #                      if name in aggregated_params:
        #                          agg_tensor = torch.from_numpy(aggregated_params[name]).to(param.device)
        #                          if param.shape == agg_tensor.shape:
        #                              param.copy_(agg_tensor)
        #                          else:
        #                               self.logger.error(f"Shape mismatch during plain aggregation update for '{name}': Model={param.shape}, Aggregated={agg_tensor.shape}. Skipping.")
        #                      else:
        #                           self.logger.warning(f"Aggregated parameter '{name}' not found. Model parameter unchanged.")


        #          # --- Apply Server-Side DP Noise (Optional) ---
        #          # This adds noise *after* aggregation.
        #          if self.dp_engine:
        #              self.logger.info("Applying differential privacy noise to the global model...")
        #              # Note: This adds noise directly to parameters. Adding noise to the
        #              # *aggregated update delta* might be more standard.
        #              self.dp_engine.add_noise_to_model(self.model)


        #          aggregation_duration = time.time() - aggregation_start_time
        #          self.logger.info(f"Aggregation and model update completed in {aggregation_duration:.2f}s.")

        #      except Exception as e:
        #           self.logger.error(f"Error during aggregation and model update: {e}", exc_info=True)

    async def _aggregate_encrypted_deltas_he(self, client_deltas_list: List[Dict[str, Any]], weights: List[float]) -> Optional[Dict[str, Any]]:
        """
        Aggregates a list of client deltas, where each delta is a dictionary of
        TenSEAL-encrypted parameter structures.
        (This is very similar to your previous _aggregate_encrypted_updates, just acting on deltas)
        """
        self.logger.info("Performing secure aggregation of encrypted DELTAS with HE...")
        if not client_deltas_list: return None
        
        aggregated_encrypted_deltas = {}
        # Assume all client_deltas in the list have the same parameter names and structure
        # (e.g., {"type": "tensor", "shape": ..., "data": serialized_encrypted_vector_str})
        param_names = list(client_deltas_list[0].keys())

        for name in param_names:
            param_structs_for_this_name = []
            current_weights_for_this_param = []

            for i, client_delta_dict in enumerate(client_deltas_list):
                if name in client_delta_dict:
                    param_structs_for_this_name.append(client_delta_dict[name])
                    current_weights_for_this_param.append(weights[i])
                else:
                    self.logger.warning(f"Encrypted delta for param '{name}' missing from a client. Skipping that client for this param.")

            if not param_structs_for_this_name:
                self.logger.error(f"No encrypted deltas found for param '{name}' across all clients. Skipping.")
                continue
            
            # Renormalize weights if some clients were skipped for this parameter
            if len(current_weights_for_this_param) < len(weights):
                sum_current_weights = sum(current_weights_for_this_param)
                if sum_current_weights > 1e-9:
                    current_weights_for_this_param = [w / sum_current_weights for w in current_weights_for_this_param]
                else: # All remaining weights are zero, cannot proceed for this param
                    self.logger.error(f"All effective weights for param '{name}' are zero after filtering. Skipping.")
                    continue
            
            # All param_structs_for_this_name should be like:
            # {"type": "tensor", "shape": [..], "data": "serialized_ckks_vector_string"}
            param_type = param_structs_for_this_name[0].get("type")
            shape = param_structs_for_this_name[0].get("shape")
            serialized_data_list = [p_info["data"] for p_info in param_structs_for_this_name]

            try:
                encrypted_vectors = [self.crypto_engine.deserialize_vector(s_data) for s_data in serialized_data_list]
                valid_indices = [i for i, vec in enumerate(encrypted_vectors) if vec is not None and isinstance(vec, ts.CKKSVector)]

                if len(valid_indices) < len(encrypted_vectors):
                    self.logger.warning(f"Could not deserialize all encrypted vectors for param delta '{name}'. Using {len(valid_indices)} valid vectors.")
                    if not valid_indices:
                        self.logger.error(f"No valid encrypted delta vectors to aggregate for param '{name}'. Skipping.")
                        continue
                    encrypted_vectors = [encrypted_vectors[i] for i in valid_indices]
                    # Adjust weights again for only the successfully deserialized vectors
                    current_weights_for_this_param = [current_weights_for_this_param[i] for i in valid_indices]
                    sum_current_weights = sum(current_weights_for_this_param)
                    if sum_current_weights > 1e-9:
                         current_weights_for_this_param = [w / sum_current_weights for w in current_weights_for_this_param]
                    else:
                         self.logger.error(f"All effective weights for param '{name}' are zero after deserialization filtering. Skipping.")
                         continue
            except Exception as e:
                self.logger.error(f"Error deserializing delta vectors for param '{name}': {e}", exc_info=True)
                continue

            # Perform HE Aggregation (weighted sum of encrypted delta vectors)
            aggregated_enc_vector = self.crypto_engine.secure_aggregation(encrypted_vectors, current_weights_for_this_param)
            
            if aggregated_enc_vector:
                serialized_agg_data = self.crypto_engine.serialize_vector(aggregated_enc_vector)
                aggregated_encrypted_deltas[name] = {"type": param_type, "shape": shape, "data": serialized_agg_data}
            else:
                self.logger.error(f"Secure aggregation of deltas failed for param '{name}' (type {param_type}).")
                # This could mean the whole aggregation for this round is compromised for this param
        
        return aggregated_encrypted_deltas if aggregated_encrypted_deltas else None


    def _aggregate_plain_deltas(self, client_deltas_list: List[Dict[str, np.ndarray]], weights: List[float]) -> Optional[Dict[str, np.ndarray]]:
        """
        Aggregates a list of client deltas, where each delta is a dictionary of
        plain numpy arrays.
        (This is very similar to your previous _aggregate_plain_updates, just acting on deltas)
        """
        self.logger.info("Performing plain FedAvg aggregation of DELTAS...")
        if not client_deltas_list: return None

        aggregated_deltas_np = {}
        # Get param names from the first client's delta (assume consistency)
        # Also, server's current model can define the expected structure
        param_names = list(self.model.state_dict().keys()) # Use server model's param names as canonical list

        for name in param_names:
            # Initialize aggregated delta with zeros matching the global model shape
            # This ensures if a client doesn't send a delta for a param, it contributes zero
            # to that parameter's aggregated delta.
            if name not in self.model.state_dict(): # Should not happen if using model keys
                self.logger.warning(f"Parameter name '{name}' from first delta not in server model. Skipping.")
                continue

            template_param = self.model.state_dict()[name]
            current_aggregated_delta_np = np.zeros_like(template_param.cpu().numpy(), dtype=np.float32)
            param_contribution_count = 0

            for i, client_delta_dict in enumerate(client_deltas_list):
                if name in client_delta_dict:
                    client_param_delta = client_delta_dict[name]
                    if isinstance(client_param_delta, np.ndarray):
                        if client_param_delta.shape == current_aggregated_delta_np.shape:
                            current_aggregated_delta_np += client_param_delta * weights[i]
                            param_contribution_count +=1
                        else:
                            self.logger.warning(f"Shape mismatch for plain delta '{name}' from client update {i}: Expected={current_aggregated_delta_np.shape}, Got={client_param_delta.shape}. Skipping contribution.")
                    else:
                        self.logger.warning(f"Plain delta for '{name}' from client update {i} is not numpy array (type: {type(client_param_delta)}). Skipping.")
            
            if param_contribution_count > 0:
                aggregated_deltas_np[name] = current_aggregated_delta_np
            else:
                # If no client contributed a valid delta for this param, the aggregated_delta for it will be zeros.
                # This is correct as it means no change to that global model parameter from this round's deltas.
                aggregated_deltas_np[name] = current_aggregated_delta_np # Store the zeros
                self.logger.debug(f"No valid client deltas received for param '{name}'. Aggregated delta is zero.")


        return aggregated_deltas_np if aggregated_deltas_np else None

    async def _aggregate_encrypted_updates(self, updates: List[Dict[str, Any]], weights: List[float]) -> Optional[Dict[str, Any]]:
        """Helper to perform secure aggregation on encrypted parameters."""
        self.logger.info("Performing secure aggregation with HE...")
        aggregated_params = {}
        # Assume all updates have the same structure (parameter names and types)
        first_update_params = updates[0]["parameters"]
        param_names = list(first_update_params.keys())

        for name in param_names:
            param_info_list = [upd["parameters"][name] for upd in updates if name in upd["parameters"]]
            if len(param_info_list) != len(updates):
                self.logger.warning(f"Parameter '{name}' missing in some client updates. Aggregating with available updates.")
                # Adjust weights? For now, use weights corresponding to available updates.
                current_weights = [weights[i] for i, upd in enumerate(updates) if name in upd["parameters"]]
                if not current_weights: continue # Skip if param missing in all
                # Renormalize weights for this parameter
                current_weights = [w / sum(current_weights) for w in current_weights]
            else:
                current_weights = weights


            # --- Aggregate based on parameter type ---
            param_type = param_info_list[0].get("type", None)
            shape = param_info_list[0].get("shape", None)
            serialized_data_list = [p_info["data"] for p_info in param_info_list]

            # Deserialize encrypted vectors
            try:
                 encrypted_vectors = [self.crypto_engine.deserialize_vector(s_data) for s_data in serialized_data_list]
                 # Filter out None results from deserialization errors
                 valid_indices = [i for i, vec in enumerate(encrypted_vectors) if vec is not None and isinstance(vec, ts.CKKSVector)]
                 if len(valid_indices) < len(encrypted_vectors):
                      self.logger.warning(f"Could not deserialize all encrypted vectors for param '{name}'. Using {len(valid_indices)} valid vectors.")
                      if not valid_indices:
                           self.logger.error(f"No valid encrypted vectors to aggregate for param '{name}'. Skipping.")
                           continue
                      # Adjust weights for valid vectors
                      encrypted_vectors = [encrypted_vectors[i] for i in valid_indices]
                      current_weights = [current_weights[i] for i in valid_indices]
                      current_weights = [w / sum(current_weights) for w in current_weights] # Renormalize


            except Exception as e:
                 self.logger.error(f"Error deserializing vectors for param '{name}': {e}", exc_info=True)
                 continue # Skip this parameter


            # --- Perform HE Aggregation ---
            if param_type in ["vector", "scalar", "tensor"]: # Flattened tensors treated as vectors
                aggregated_enc_vector = self.crypto_engine.secure_aggregation(encrypted_vectors, current_weights)
                if aggregated_enc_vector:
                     serialized_agg_data = self.crypto_engine.serialize_vector(aggregated_enc_vector)
                     aggregated_params[name] = {"type": param_type, "shape": shape, "data": serialized_agg_data}
                else:
                     self.logger.error(f"Secure aggregation failed for param '{name}' (type {param_type}).")

            elif param_type == "matrix":
                 # Aggregate row by row (assuming data is List[serialized_row_vector])
                 self.logger.debug(f"Aggregating matrix param '{name}' row by row...")
                 num_rows = shape[0] if shape else 0
                 aggregated_rows_serialized = []
                 valid_matrix_aggregation = True
                 for r in range(num_rows):
                      row_vectors_serialized = [p_info["data"][r] for p_info in param_info_list if isinstance(p_info.get("data"), list) and len(p_info["data"]) > r]

                      if len(row_vectors_serialized) != len(param_info_list):
                            self.logger.warning(f"Row {r} missing in some matrix updates for '{name}'. Adjusting weights for row aggregation.")
                            # Need complex weight adjustment here based on which matrices had the row
                            # Simplified: Skip row if inconsistent for now
                            self.logger.error(f"Inconsistent row data for matrix '{name}', row {r}. Skipping matrix aggregation.")
                            valid_matrix_aggregation = False
                            break # Cannot aggregate matrix if rows are inconsistent

                      try:
                           row_enc_vectors = [self.crypto_engine.deserialize_vector(s_row) for s_row in row_vectors_serialized]
                           # Check for deserialization errors
                           if any(v is None or not isinstance(v, ts.CKKSVector) for v in row_enc_vectors):
                                self.logger.error(f"Failed to deserialize some row vectors for matrix '{name}', row {r}.")
                                valid_matrix_aggregation = False
                                break
                      except Exception as e:
                            self.logger.error(f"Error deserializing row {r} for matrix '{name}': {e}")
                            valid_matrix_aggregation = False
                            break

                      # Aggregate the row
                      agg_row_enc = self.crypto_engine.secure_aggregation(row_enc_vectors, current_weights)
                      if agg_row_enc:
                           serialized_agg_row = self.crypto_engine.serialize_vector(agg_row_enc)
                           aggregated_rows_serialized.append(serialized_agg_row)
                      else:
                           self.logger.error(f"Secure aggregation failed for matrix '{name}', row {r}.")
                           valid_matrix_aggregation = False
                           break # Stop aggregating this matrix

                 if valid_matrix_aggregation and len(aggregated_rows_serialized) == num_rows:
                      aggregated_params[name] = {"type": "matrix", "shape": shape, "data": aggregated_rows_serialized}
                 elif valid_matrix_aggregation: # Should not happen if loop completed correctly
                      self.logger.error(f"Matrix aggregation finished for '{name}' but row count mismatch ({len(aggregated_rows_serialized)} vs {num_rows}).")

            else:
                 self.logger.error(f"Unsupported encrypted parameter type '{param_type}' for param '{name}'. Skipping.")

        return aggregated_params if aggregated_params else None


    def _aggregate_plain_updates(self, updates: List[Dict[str, Any]], weights: List[float]) -> Optional[Dict[str, np.ndarray]]:
        """Helper to perform standard FedAvg on plain numpy parameters."""
        self.logger.info("Performing plain FedAvg aggregation...")
        aggregated_params = {}
        # Get the state dict of the current global model to initialize aggregation
        # Work on CPU to avoid potential GPU memory issues during aggregation loops
        current_global_params = {name: param.cpu().detach().numpy()
                                for name, param in self.model.named_parameters()}

        param_names = list(current_global_params.keys())

        for name in param_names:
            # Initialize aggregated param with zeros matching the global model shape
            aggregated_param_np = np.zeros_like(current_global_params[name], dtype=np.float32)
            param_aggregated = False

            # Accumulate weighted updates from clients
            for i, update in enumerate(updates):
                if name in update["parameters"]:
                    client_param = update["parameters"][name]
                    if isinstance(client_param, np.ndarray):
                         if client_param.shape == aggregated_param_np.shape:
                             aggregated_param_np += client_param * weights[i]
                             param_aggregated = True
                         else:
                              self.logger.warning(f"Shape mismatch for plain param '{name}' from client {update.get('client_id','unknown')}: Model={aggregated_param_np.shape}, Client={client_param.shape}. Skipping client's contribution.")
                    else:
                         self.logger.warning(f"Plain parameter '{name}' from client {update.get('client_id','unknown')} is not a numpy array (type: {type(client_param)}). Skipping.")
                else:
                    # This shouldn't happen if clients return full parameter sets
                    self.logger.warning(f"Parameter '{name}' missing in plain update from client {update.get('client_id','unknown')}.")


            if param_aggregated:
                aggregated_params[name] = aggregated_param_np
            else:
                 # If no client provided this param (or all were skipped), keep the old global value?
                 # This should be rare. Log a warning.
                 self.logger.warning(f"Parameter '{name}' was not aggregated from any client. Keeping previous global value might lead to inconsistencies.")
                 # Option: Keep old value: aggregated_params[name] = current_global_params[name]
                 # Option: Skip (as implemented now - model update loop will warn)


        return aggregated_params if aggregated_params else None

    # --- Evaluation ---

    async def _evaluate_model(self, dataloader: DataHandler) -> Dict[str, float]:
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        """Evaluate the current global model on the provided dataloader (val or test)."""
        if not dataloader:
            self.logger.warning("Evaluation requested but dataloader is None.")
            return {}
        if not self.model:
             self.logger.error("Cannot evaluate, global model not initialized.")
             return {}

        self.model.eval() # Set model to evaluation mode
        task_type = self.config.model.task_type
        is_binary = task_type == "binary_classification"
        num_classes = self.config.data.output_shape[0]

        # Select Loss Function (consistent with training)
        try:
            if is_binary: criterion = nn.BCEWithLogitsLoss()
            elif task_type == "classification": criterion = nn.CrossEntropyLoss()
            elif task_type == "regression": criterion = nn.MSELoss()
            else: raise ValueError(f"Unsupported task type for evaluation: {task_type}")
        except Exception as e:
             self.logger.error(f"Failed to create criterion for evaluation: {e}")
             return {"eval_error": 1.0}


        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_targets = []
        all_predictions = [] # Store raw outputs for more metrics later if needed

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device) # Dtype should be correct from DataHandler
                outputs = self.model(inputs)

                # Loss Calculation
                try:
                     if is_binary: loss = criterion(outputs.squeeze(), targets.float())
                     elif task_type == "classification": loss = criterion(outputs, targets.long())
                     else: loss = criterion(outputs.squeeze(), targets.float()) # Regression

                     # Check for NaN/Inf loss during evaluation
                     if torch.isnan(loss) or torch.isinf(loss):
                          self.logger.warning(f"NaN or Inf loss encountered during evaluation batch {batch_idx}. Loss: {loss.item()}. Skipping batch metrics.")
                          continue # Skip this batch's contribution to metrics

                     total_loss += loss.item() * inputs.size(0)

                except Exception as e:
                     self.logger.error(f"Error calculating loss during evaluation batch {batch_idx}: {e}", exc_info=True)
                     continue # Skip batch


                total_samples += inputs.size(0)
                all_targets.append(targets.cpu().numpy())

                # Accuracy Calculation & Storing Predictions
                try:
                     if is_binary:
                         probs = torch.sigmoid(outputs.squeeze())
                         predicted = (probs > 0.5).float()
                         correct = (predicted == targets.float()).sum().item()
                         all_predictions.append(probs.cpu().numpy()) # Store probabilities
                     elif task_type == "classification":
                         probs = torch.softmax(outputs, dim=1)
                         _, predicted = torch.max(probs, 1)
                         correct = (predicted == targets.long()).sum().item()
                         all_predictions.append(probs.cpu().numpy()) # Store probabilities
                     else:
                         correct = 0 # No accuracy for regression
                         all_predictions.append(outputs.squeeze().cpu().numpy()) # Store raw regression outputs

                     total_correct += correct

                except Exception as e:
                    self.logger.error(f"Error calculating accuracy/predictions during evaluation batch {batch_idx}: {e}", exc_info=True)
                    # Continue calculating loss if possible

        self.model.train() # Set model back to training mode

        # --- Compute Final Metrics ---
        metrics = {}
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            metrics["loss"] = avg_loss

            if task_type != "regression":
                accuracy = total_correct / total_samples
                metrics["accuracy"] = accuracy

                y_true_np = np.concatenate(all_targets)

                if is_binary:
                    # For binary, all_predictions might store probabilities or raw logits
                    # If probabilities:
                    y_pred_probs_np = np.concatenate(all_predictions)
                    y_pred_binary_np = (y_pred_probs_np > 0.5).astype(int)
                    
                    metrics["precision"] = precision_score(y_true_np, y_pred_binary_np, zero_division=0)
                    metrics["recall"] = recall_score(y_true_np, y_pred_binary_np, zero_division=0)
                    metrics["f1_score"] = f1_score(y_true_np, y_pred_binary_np, zero_division=0)
                    try:
                        metrics["auc"] = roc_auc_score(y_true_np, y_pred_probs_np)
                    except ValueError: # Handles cases where only one class is present in y_true_np
                        metrics["auc"] = 0.5 # Or None or np.nan
                    # tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_binary_np).ravel()
                    # metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0


                elif task_type == "classification": # Multi-class
                    # For multi-class, all_predictions might store probability distributions
                    y_pred_probs_np = np.concatenate(all_predictions) # Shape (n_samples, n_classes)
                    y_pred_labels_np = np.argmax(y_pred_probs_np, axis=1)

                    metrics["precision_macro"] = precision_score(y_true_np, y_pred_labels_np, average='macro', zero_division=0)
                    metrics["recall_macro"] = recall_score(y_true_np, y_pred_labels_np, average='macro', zero_division=0)
                    metrics["f1_score_macro"] = f1_score(y_true_np, y_pred_labels_np, average='macro', zero_division=0)
                    try:
                        # roc_auc_score for multi-class needs probabilities and 'ovr' or 'ovo' strategy
                        metrics["auc_macro_ovr"] = roc_auc_score(y_true_np, y_pred_probs_np, average='macro', multi_class='ovr')
                    except ValueError:
                        metrics["auc_macro_ovr"] = 0.5 # Or None
             # TODO: Add more metrics if needed (AUC, F1, Precision, Recall using all_targets/all_predictions)
             # from sklearn.metrics import roc_auc_score, f1_score etc.
             # Requires careful handling of shapes and averaging for multi-class.

        else:
             self.logger.warning("No samples processed during evaluation.")
             metrics["loss"] = -1.0
             if task_type != "regression": metrics["accuracy"] = -1.0


        return metrics


    async def run_final_evaluation(self):
        """Evaluate the final model on the test set."""
        self.logger.info("===== Running Final Evaluation on Test Set =====")
        if self.test_dataloader:
            test_metrics = await self._evaluate_model(self.test_dataloader)
            self.logger.info(f"Final Test Metrics: {test_metrics}")
            # Store test metrics in history or results file
            if self.training_history:
                 self.training_history[-1]["test_metrics"] = test_metrics # Add to last round history
            else:
                 # Create a dummy history entry if no rounds were run but eval is possible
                 self.training_history.append({"round": self.current_round, "test_metrics": test_metrics})
            self._save_history() # Save history including test metrics
        else:
            self.logger.warning("No test data loaded. Skipping final evaluation.")
        self.logger.info("==============================================")

    async def start_new_round(self) -> Optional[Dict[str, Any]]:
        """
        Selects clients for a new round and prepares the configuration package.
        Returns the package or None if round cannot be started.
        """
        async with self._round_lock: # Ensure only one round starts at a time
            if self.current_round >= self.max_rounds:
                self.logger.info("Maximum rounds reached.")
                return None

            # Increment round counter *before* selection and config prep
            self.current_round += 1
            self.logger.info(f"===== Starting Round {self.current_round}/{self.max_rounds} =====")

            available_clients = await self.get_available_clients()
            self.logger.info(f"Available clients for selection: {len(available_clients)}")

            if len(available_clients) < self.config.federated.min_clients:
                self.logger.warning(f"Not enough available clients ({len(available_clients)}) to meet minimum ({self.config.federated.min_clients}). Skipping round {self.current_round}.")
                # Decrement round counter as it's skipped immediately
                self.current_round -= 1
                await asyncio.sleep(5) # Wait a bit before next attempt
                return None # Signal that round was skipped

            num_to_select = min(self.config.federated.clients_per_round, len(available_clients))
            self.selected_clients_this_round = np.random.choice(
                available_clients, num_to_select, replace=False
            ).tolist()
            self.logger.info(f"Selected {len(self.selected_clients_this_round)} clients for round {self.current_round}: {self.selected_clients_this_round}")

            # Reset updates for the new round
            self.client_updates_this_round = {}

            # Prepare parameters and configuration
            round_config_package = await self._prepare_round_start_config() # Uses self.current_round internally
            if round_config_package is None:
                self.logger.error("Failed to prepare round configuration. Skipping round.")
                self.current_round -= 1 # Decrement round counter
                return None

            # Add selected clients list to the package for the orchestrator
            round_config_package["selected_clients"] = self.selected_clients_this_round
            return round_config_package
        
    async def finalize_round(self) -> bool:
        """
        Aggregates received updates, updates the global model, evaluates, and logs history.
        Should be called by the orchestrator after submitting client updates.
        Returns True if aggregation happened, False otherwise.
        """
        async with self._round_lock: # Protect aggregation and evaluation steps
            self.logger.info(f"Finalizing Round {self.current_round}...")
            round_start_time = time.time() # Placeholder time for logging duration

            num_received = len(self.client_updates_this_round)
            if num_received == 0:
                self.logger.warning("No client updates received for this round. Skipping aggregation and evaluation.")
                # Record history indicating skip
                self._record_history() # Record skip status
                self._log_round_summary(round_start_time)
                self.logger.info(f"===== Finished Round {self.current_round} (Skipped) =====")
                return False

            # --- Aggregate and Update ---
            await self._aggregate_and_update() # Handles HE/Plain aggregation, DP

            # --- Evaluate ---
            if self.val_dataloader:
                self.logger.info("Evaluating updated global model (Validation)...")
                self.global_eval_metrics = await self._evaluate_model(self.val_dataloader)
                self.logger.info(f"Round {self.current_round} Validation Metrics: {self.global_eval_metrics}")
                # Track best model
                metric_to_track = 'accuracy' if self.config.model.task_type != 'regression' else 'loss'
                current_metric = self.global_eval_metrics.get(metric_to_track)
                if current_metric is not None:
                    is_better = (metric_to_track == 'accuracy' and current_metric > self.best_model_metric_value) or \
                                (metric_to_track == 'loss' and current_metric < (self.best_model_metric_value if self.best_model_round > 0 else float('inf')))
                    if is_better:
                        self.logger.info(f"New best model found! Round {self.current_round}, {metric_to_track}: {current_metric:.4f}")
                        self.best_model_metric_value = current_metric
                        self.best_model_round = self.current_round
                        self._save_checkpoint("best")
            else:
                 self.logger.info("Skipping global model evaluation (no validation data).")

            # --- Log History & Summary ---
            self._record_history()
            self._log_round_summary(round_start_time)

            # --- Periodic Checkpoint ---
            if self.current_round % self.config.system.checkpoint_frequency == 0:
                 self._save_checkpoint(f"round_{self.current_round}")

            self.logger.info(f"===== Finished Round {self.current_round} =====")

            # --- Check if Training Finished ---
            if self.current_round >= self.max_rounds:
                 self.logger.info("Maximum rounds reached.")
                 await self.run_final_evaluation()
                 # Consider stopping the server? Or let the main script handle stop.

            return True # Indicate aggregation/evaluation happened

    # --- Persistence & Logging ---

    def _save_checkpoint(self, tag: str):
        """Save global model checkpoint."""
        if not self.model:
             self.logger.warning("Cannot save checkpoint, model not initialized.")
             return
        try:
            checkpoint_path = self.checkpoint_dir / f"global_model_{tag}.pt"
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "round": self.current_round,
                "timestamp": datetime.now().isoformat(),
                "best_metric_value": self.best_model_metric_value,
                "best_model_round": self.best_model_round,
                "global_eval_metrics": self.global_eval_metrics,
                # Save config hash or subset for verification?
                # "config_hash": hash(json.dumps(self.config.to_dict(), sort_keys=True))
            }
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved global model checkpoint to {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Error saving global checkpoint '{tag}': {e}", exc_info=True)

    def _record_history(self):
        """Append metrics from the completed round to the history."""
        client_metrics_summary = {}
        total_client_samples = 0
        avg_client_train_loss = 0
        avg_client_train_acc = 0 # If applicable
        num_clients_with_metrics = 0
        sum_quality_scores = 0
        num_valid_quality_scores = 0   

        for client_id, update in self.client_updates_this_round.items():
            metrics = update.get("metrics", {})
            samples = update.get("sample_size", 0)
            quality_score = update.get("quality_score")
            client_metrics_summary[client_id] = {
                "train_loss": metrics.get("train_loss"),
                "train_accuracy": metrics.get("train_accuracy"),
                "val_loss": metrics.get("val_loss"),
                "val_accuracy": metrics.get("val_accuracy"),
                "quality_score": quality_score,
                "samples": samples,
                "duration": update.get("train_duration_sec")
            }
            if metrics.get("train_loss") is not None and samples > 0:
                avg_client_train_loss += metrics["train_loss"] * samples
                if metrics.get("train_accuracy") is not None:
                    avg_client_train_acc += metrics["train_accuracy"] * samples
                total_client_samples += samples
                num_clients_with_metrics += 1
           
            if quality_score is not None and not (np.isnan(quality_score) or np.isinf(quality_score)):
                sum_quality_scores += quality_score
                num_valid_quality_scores +=1

        avg_quality_score = (sum_quality_scores / num_valid_quality_scores) if num_valid_quality_scores > 0 else None
                
        if total_client_samples > 0:
            avg_client_train_loss /= total_client_samples
            if self.config.model.task_type != 'regression':
                avg_client_train_acc /= total_client_samples
        else: # Avoid division by zero if no samples
             avg_client_train_loss = None
             avg_client_train_acc = None

        history_entry = {
            "round": self.current_round,
            "timestamp": datetime.now().isoformat(),
            "num_selected_clients": len(self.selected_clients_this_round),
            "num_updates_received": len(self.client_updates_this_round),
            "global_metrics": copy.deepcopy(self.global_eval_metrics), # Metrics after aggregation
            "avg_client_train_loss": avg_client_train_loss if total_client_samples > 0 else None,
            "avg_client_train_accuracy": avg_client_train_acc if total_client_samples > 0 and self.config.model.task_type != 'regression' else None,
            "avg_quality_score": avg_quality_score,
            # "client_metrics_detailed": client_metrics_summary # Optional: Store per-client details
        }
        self.training_history.append(history_entry)


    def _log_round_summary(self, round_start_time: float):
        """Log a summary of the completed round."""
        round_duration = time.time() - round_start_time
        summary = f"Round {self.current_round} Summary ({round_duration:.2f}s): "
        summary += f"Selected={len(self.selected_clients_this_round)}, Received={len(self.client_updates_this_round)}. "
        if self.global_eval_metrics:
             loss = self.global_eval_metrics.get('loss', -1)
             acc = self.global_eval_metrics.get('accuracy', -1)
             summary += f"Global Eval Loss={loss:.4f}"
             if acc != -1: summary += f", Acc={acc:.4f}"
        else:
             summary += "Global Eval Skipped."

        # Add avg client metrics if available from history
        if self.training_history:
             last_entry = self.training_history[-1]
             client_loss = last_entry.get('avg_client_train_loss')
             client_acc = last_entry.get('avg_client_train_accuracy')
             if client_loss is not None: summary += f" | Avg Client Train Loss={client_loss:.4f}"
             if client_acc is not None: summary += f", Acc={client_acc:.4f}"

        self.logger.info(summary)


    def _save_history(self):
        """Save training history log to a JSON file."""
        try:
            history_path = self.result_dir / f"{self.config.project_name}_training_history.json"
            history_data = {
                "project_name": self.config.project_name,
                "config_summary": { # Store key config parameters
                     "model_type": self.config.model.type,
                     "task_type": self.config.model.task_type,
                     "total_rounds": self.max_rounds,
                     "clients_per_round": self.config.federated.clients_per_round,
                     "local_epochs": self.config.federated.local_epochs,
                     "client_lr": self.config.federated.client_learning_rate,
                     "he_enabled": self.config.crypto.enabled,
                     "dp_enabled": self.config.privacy.differential_privacy,
                },
                "start_time": self.training_history[0]["timestamp"] if self.training_history else None,
                "end_time": datetime.now().isoformat(),
                "best_model_round": self.best_model_round,
                "best_model_metric_value": self.best_model_metric_value,
                "final_global_metrics": self.global_eval_metrics, # Store latest eval metrics
                "training_log": self.training_history
            }
            with open(history_path, "w") as f:
                json.dump(history_data, f, indent=2)
            self.logger.info(f"Training history saved to {history_path}")
        except Exception as e:
            self.logger.error(f"Error saving training history: {e}", exc_info=True)


    def load_checkpoint(self, path: str):
        """Load global model and server state from checkpoint."""
        if not self.model or not self.optimizer:
             self.logger.error("Cannot load checkpoint, model or optimizer not initialized. Run start() first.")
             return False
        try:
            checkpoint_path = Path(path)
            if not checkpoint_path.is_file():
                 self.logger.error(f"Checkpoint file not found: {path}")
                 return False

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            async def load_state():
                 async with self._model_lock:
                      self.model.load_state_dict(checkpoint["model_state_dict"])
                      if "optimizer_state_dict" in checkpoint and self.optimizer:
                          self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                      else:
                           self.logger.warning("Optimizer state not found or optimizer not initialized. Optimizer not loaded.")

                 # Load server state
                 self.current_round = checkpoint.get("round", 0)
                 self.best_model_metric_value = checkpoint.get("best_metric_value", -1.0)
                 self.best_model_round = checkpoint.get("best_model_round", 0)
                 self.global_eval_metrics = checkpoint.get("global_eval_metrics", {})
                 self.logger.info(f"Loaded global model checkpoint from {path} (Round {self.current_round})")
                 return True

            # Run async load within the async context if needed, or directly if called synchronously
            # Assuming this might be called from a synchronous context before starting the loop
            # If called from within async loop, just await load_state()
            return asyncio.run(load_state()) # Be careful if server loop is already running

        except Exception as e:
            self.logger.error(f"Error loading checkpoint from {path}: {e}", exc_info=True)
            return False