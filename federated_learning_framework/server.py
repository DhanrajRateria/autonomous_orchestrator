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
            _, self.val_dataloader, self.test_dataloader = await asyncio.to_thread(
                self.data_handler.load_data,
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

        valid_updates = list(self.client_updates_this_round.values())
        if not valid_updates:
            self.logger.warning("No valid updates available for aggregation.")
            return

        # --- Prepare weights (FedAvg) ---
        total_samples = sum(update["sample_size"] for update in valid_updates)
        if total_samples == 0:
            self.logger.warning("Total sample size is zero. Using equal weights for aggregation.")
            weights = [1.0 / len(valid_updates)] * len(valid_updates)
        else:
            weights = [update["sample_size"] / total_samples for update in valid_updates]

        self.logger.info(f"Aggregating {len(valid_updates)} updates using FedAvg. Total samples: {total_samples}")
        # self.logger.debug(f"Aggregation weights: {weights}")

        # Determine if updates are encrypted (check the first valid update)
        are_updates_encrypted = valid_updates[0].get("encrypted", False)
        self.logger.info(f"Updates are {'encrypted' if are_updates_encrypted else 'plain'}.")


        # --- Perform Aggregation ---
        async with self._model_lock:
             try:
                 if are_updates_encrypted:
                     if not self.crypto_engine.is_enabled():
                         self.logger.error("Received encrypted updates but HE is disabled on server. Cannot aggregate.")
                         return
                     # Secure Aggregation using HE
                     aggregated_params = await self._aggregate_encrypted_updates(valid_updates, weights)
                     if aggregated_params is None:
                          raise RuntimeError("Encrypted aggregation failed.")

                     # Decrypt and update model
                     self.logger.info("Decrypting aggregated parameters...")
                     start_dec = time.time()
                     self.crypto_engine.decrypt_to_torch_params(self.model, aggregated_params)
                     self.logger.info(f"Decryption and model update took {time.time() - start_dec:.2f}s")

                 else:
                     # Standard FedAvg Aggregation
                     aggregated_params = self._aggregate_plain_updates(valid_updates, weights)
                     if aggregated_params is None:
                           raise RuntimeError("Plain aggregation failed.")

                     # Update model with plain aggregated parameters
                     self.logger.info("Updating global model with plain aggregated parameters...")
                     with torch.no_grad():
                         for name, param in self.model.named_parameters():
                             if name in aggregated_params:
                                 agg_tensor = torch.from_numpy(aggregated_params[name]).to(param.device)
                                 if param.shape == agg_tensor.shape:
                                     param.copy_(agg_tensor)
                                 else:
                                      self.logger.error(f"Shape mismatch during plain aggregation update for '{name}': Model={param.shape}, Aggregated={agg_tensor.shape}. Skipping.")
                             else:
                                  self.logger.warning(f"Aggregated parameter '{name}' not found. Model parameter unchanged.")


                 # --- Apply Server-Side DP Noise (Optional) ---
                 # This adds noise *after* aggregation.
                 if self.dp_engine:
                     self.logger.info("Applying differential privacy noise to the global model...")
                     # Note: This adds noise directly to parameters. Adding noise to the
                     # *aggregated update delta* might be more standard.
                     self.dp_engine.add_noise_to_model(self.model)


                 aggregation_duration = time.time() - aggregation_start_time
                 self.logger.info(f"Aggregation and model update completed in {aggregation_duration:.2f}s.")

             except Exception as e:
                  self.logger.error(f"Error during aggregation and model update: {e}", exc_info=True)


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

         for client_id, update in self.client_updates_this_round.items():
             metrics = update.get("metrics", {})
             samples = update.get("sample_size", 0)
             client_metrics_summary[client_id] = {
                 "train_loss": metrics.get("train_loss"),
                 "train_accuracy": metrics.get("train_accuracy"),
                 "val_loss": metrics.get("val_loss"),
                 "val_accuracy": metrics.get("val_accuracy"),
                 "samples": samples,
                 "duration": update.get("train_duration_sec")
             }
             if metrics.get("train_loss") is not None and samples > 0:
                  avg_client_train_loss += metrics["train_loss"] * samples
                  if metrics.get("train_accuracy") is not None:
                       avg_client_train_acc += metrics["train_accuracy"] * samples
                  total_client_samples += samples
                  num_clients_with_metrics += 1


         if total_client_samples > 0:
             avg_client_train_loss /= total_client_samples
             if self.config.model.task_type != 'regression':
                  avg_client_train_acc /= total_client_samples

         history_entry = {
             "round": self.current_round,
             "timestamp": datetime.now().isoformat(),
             "num_selected_clients": len(self.selected_clients_this_round),
             "num_updates_received": len(self.client_updates_this_round),
             "global_metrics": copy.deepcopy(self.global_eval_metrics), # Metrics after aggregation
             "avg_client_train_loss": avg_client_train_loss if total_client_samples > 0 else None,
             "avg_client_train_accuracy": avg_client_train_acc if total_client_samples > 0 and self.config.model.task_type != 'regression' else None,
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