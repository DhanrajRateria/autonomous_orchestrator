# cancer_detection.py
import asyncio
import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
import time
import torch
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Optional, Any
from datetime import datetime
# --- Setup Project Root Path ---
# Assuming this script is in framework_root/examples/cancer_detection/
# Add framework_root to sys.path
try:
    # More robust way to find project root assuming a specific structure
    # Adjust based on your actual project layout
    project_root = Path(__file__).resolve().parent.parent.parent
    if project_root.name == 'federated_learning_framework': # Or check for a known file/dir
         sys.path.insert(0, str(project_root))
         print(f"Added project root to sys.path: {project_root}")
    else:
         # Fallback if structure is different
         script_dir = Path(__file__).resolve().parent
         sys.path.insert(0, str(script_dir.parent.parent)) # Go up two levels
         print(f"Added parent's parent directory to sys.path: {script_dir.parent.parent}")
except Exception as e:
     print(f"Warning: Could not automatically add project root to sys.path. Ensure framework modules are importable. Error: {e}")

# --- Framework Imports ---
try:
    from federated_learning_framework.config import FrameworkConfig
    from federated_learning_framework.server import FederatedServer
    from federated_learning_framework.client import FederatedClient
except ImportError as e:
    print(f"Error importing framework components: {e}")
    print("Please ensure the script is run from the correct directory or the framework path is in sys.path.")
    sys.exit(1)


# --- Logging Setup ---
# Moved config loading after argparse to set log level from config
logger = logging.getLogger("cancer_detection_main")


def setup_logging(log_level_str: str, log_dir: str, project_name: str): # Added log_dir and project_name
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # Ensure the log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Create a unique log file name, e.g., with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"{project_name}_run_{timestamp}.log"
    log_file_path = Path(log_dir) / log_file_name

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)-25s - %(levelname)-8s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),    # Keep logging to console
            logging.FileHandler(log_file_path)    # Add logging to a file
        ],
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set higher level for noisy libraries if needed
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    # You might want to control TenSEAL's verbosity too if it becomes too much
    # logging.getLogger("tenseal").setLevel(logging.INFO) # Or WARNING

    # Log where the log file is being saved for user convenience
    logging.info(f"Logging to console and to file: {log_file_path}")


async def prepare_data(num_clients: int, base_data_path: str, target_column: str, task_type: str, data_dir: Path = Path("client_data")) -> Dict[str, str]:
    """
    Prepares and splits data for clients (potentially non-IID).
    Saves client data to separate files.

    Args:
        num_clients: Number of client datasets to create.
        base_data_path: Path to the full dataset CSV file.
        target_column: Name of the target column in the CSV.
        task_type: The type of task ('binary_classification', 'classification', 'regression').
        data_dir: Directory to save individual client data files.

    Returns:
        Dictionary mapping client_id to its data file path.
    """
    logger.info(f"Preparing data for {num_clients} clients from {base_data_path}...")
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(base_data_path)
        if df.empty:
            raise ValueError("Base dataset is empty.")
        logger.info(f"Full dataset loaded: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load base dataset: {e}", exc_info=True)
        raise

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")


    # --- Handle Non-IID splitting for Binary Classification ---
    client_data_paths = {}
    if task_type == "binary_classification":
        logger.info("Applying non-IID data split for binary classification...")
        # Ensure target is numeric (0/1)
        if not pd.api.types.is_numeric_dtype(df[target_column]):
             logger.warning(f"Target column '{target_column}' is not numeric. Attempting LabelEncoding.")
             le = LabelEncoder()
             try:
                  df[target_column] = le.fit_transform(df[target_column])
                  if len(le.classes_) != 2:
                      raise ValueError(f"LabelEncoder found {len(le.classes_)} classes, expected 2 for binary.")
                  logger.info(f"Applied LabelEncoder. Mapped classes: {le.classes_} to [0, 1]")
             except Exception as e:
                   logger.error(f"Failed to encode non-numeric binary target: {e}", exc_info=True)
                   raise

        # Separate classes
        positive_df = df[df[target_column] == 1]
        negative_df = df[df[target_column] == 0]
        logger.info(f"Class distribution: {len(positive_df)} positive (1), {len(negative_df)} negative (0)")

        if len(positive_df) == 0 or len(negative_df) == 0:
             logger.warning("Dataset contains only one class. Non-IID split will be effectively IID.")
             # Fallback to random split
             task_type = "classification" # Treat as random split case

        else:
             # Distribute data non-IID
             # Example: Skew distribution across clients
             samples_per_client = len(df) // num_clients
             remaining_pos = len(positive_df)
             remaining_neg = len(negative_df)
             pos_indices = positive_df.index.tolist()
             neg_indices = negative_df.index.tolist()
             np.random.shuffle(pos_indices)
             np.random.shuffle(neg_indices)

             pos_idx_ptr = 0
             neg_idx_ptr = 0


             for i in range(num_clients):
                 client_id = f"client_{i+1}"
                 # Create skewed ratios (e.g., dirichlet distribution could be better)
                 # Simple linear skew for example:
                 target_pos_ratio = 0.1 + (i / (num_clients -1 if num_clients > 1 else 1)) * 0.8 # Skew from 10% to 90% positive

                 num_samples_client = samples_per_client if i < num_clients - 1 else len(df) - (samples_per_client * (num_clients - 1)) # Give remainder to last client

                 num_pos = int(num_samples_client * target_pos_ratio)
                 num_neg = num_samples_client - num_pos

                 # Take available samples without replacement first
                 client_pos_indices = pos_indices[pos_idx_ptr : pos_idx_ptr + num_pos]
                 client_neg_indices = neg_indices[neg_idx_ptr : neg_idx_ptr + num_neg]

                 pos_idx_ptr += len(client_pos_indices)
                 neg_idx_ptr += len(client_neg_indices)

                 # Handle cases where we run out of samples (oversampling with replacement) - less ideal
                 if len(client_pos_indices) < num_pos:
                      logger.warning(f"{client_id}: Not enough unique positive samples ({len(client_pos_indices)}/{num_pos}). Sampling with replacement.")
                      needed = num_pos - len(client_pos_indices)
                      client_pos_indices.extend(np.random.choice(positive_df.index, needed, replace=True).tolist()) # Sample from all positive indices
                 if len(client_neg_indices) < num_neg:
                      logger.warning(f"{client_id}: Not enough unique negative samples ({len(client_neg_indices)}/{num_neg}). Sampling with replacement.")
                      needed = num_neg - len(client_neg_indices)
                      client_neg_indices.extend(np.random.choice(negative_df.index, needed, replace=True).tolist()) # Sample from all negative indices

                 client_indices = client_pos_indices + client_neg_indices
                 np.random.shuffle(client_indices) # Shuffle indices within client data

                 client_df = df.loc[client_indices]

                 client_path = data_dir / f"{client_id}_data.csv"
                 client_df.to_csv(client_path, index=False)
                 client_data_paths[client_id] = str(client_path)
                 logger.info(f"Created dataset for {client_id}: {len(client_df)} samples ({len(client_pos_indices)} pos, {len(client_neg_indices)} neg). Saved to {client_path}")

             return client_data_paths # Exit after handling non-IID binary case

    # --- Handle IID splitting for Multi-Class or Regression (or binary fallback) ---
    logger.info("Applying random (IID-like) data split...")
    df_shuffled = df.sample(frac=1).reset_index(drop=True) # Shuffle dataframe
    samples_per_client = len(df_shuffled) // num_clients

    for i in range(num_clients):
        client_id = f"client_{i+1}"
        start_idx = i * samples_per_client
        # Assign remaining samples to the last client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(df_shuffled)

        client_df = df_shuffled.iloc[start_idx:end_idx]

        client_path = data_dir / f"{client_id}_data.csv"
        client_df.to_csv(client_path, index=False)
        client_data_paths[client_id] = str(client_path)
        logger.info(f"Created dataset for {client_id}: {len(client_df)} samples (IID split). Saved to {client_path}")

    return client_data_paths


async def run_federated_learning(config: FrameworkConfig, num_clients: int, client_data_paths: Dict[str, str]):
    """
    Runs the main federated learning simulation loop.
    Orchestrates server rounds and client training.
    """
    server = None # Initialize server variable
    clients = {} # Initialize clients dictionary
    try:
        # --- Initialize Server ---
        logger.info("Initializing Federated Server...")
        server = FederatedServer(config)
        await server.start() # Loads server data, creates model, etc.
        logger.info("Server initialized successfully.")

        # --- Initialize Clients ---
        logger.info(f"Initializing {num_clients} Federated Clients...")
        all_client_ids_list = list(client_data_paths.keys())
        num_total_clients = len(all_client_ids_list)
        # Shuffle client IDs to randomize role assignment if not assigning by index
        # If client_ids are like "client_1", "client_2", parsing index is okay.
        # For more general IDs, map to indices first.
        client_indices_for_roles = list(range(num_total_clients))
        np.random.shuffle(client_indices_for_roles) # Shuffle indices for random assignment

        num_noisy = int(config.robustness.fraction_noisy_clients * num_total_clients)
        noisy_client_role_indices = set(client_indices_for_roles[:num_noisy])

        num_adversarial = int(config.robustness.fraction_adversarial_clients * num_total_clients)

        # Ensure adversarial are distinct from noisy for this simple assignment
        # (can be more complex if overlap is allowed and handled)
        adversarial_client_role_indices = set(client_indices_for_roles[num_noisy : num_noisy + num_adversarial])
        
        logger.info(f"Assigning roles: {num_noisy} noisy clients, {num_adversarial} adversarial clients.")

        clients = {}

        initialization_tasks = []
        active_client_ids_after_init = list(client_data_paths.keys())

        for i, client_id_str in enumerate(all_client_ids_list):
            data_path = client_data_paths[client_id_str]
            
            # Determine role based on its original index in the shuffled list used for role assignment
            is_noisy_client = i in noisy_client_role_indices # Check if the *shuffled index* i falls into noisy set
            is_adversarial_client = i in adversarial_client_role_indices # Check if the *shuffled index* i falls into adversarial set
            
            # If using client_id to map to role index directly (e.g. client_1 maps to index 0):
            # parsed_client_num = int(client_id_str.split('_')[-1]) -1 # Assuming client_X format
            # is_noisy_client = parsed_client_num in noisy_client_role_indices (if role_indices refer to original client numbers)
            # For this example, let's assume the simple shuffled assignment based on enumeration order of client_data_paths.keys()

            client = FederatedClient(
                client_id=client_id_str,
                config=config, # Pass the full config (now includes pos_weight if set by server DH)
                data_path=data_path,
                is_noisy=is_noisy_client,
                is_adversarial=is_adversarial_client
            )
            clients[client_id_str] = client
            initialization_tasks.append(client.initialize(server.public_context_bytes))

        # for client_id in client_ids:
        #     data_path = client_data_paths[client_id]
        #     client = FederatedClient(client_id, config, data_path)
        #     clients[client_id] = client
        #     # Prepare initialization task (loads data, creates model, gets public context)
        #     initialization_tasks.append(client.initialize(server.public_context_bytes))

        # Run initializations concurrently
        init_results = await asyncio.gather(*initialization_tasks)

        successful_inits = 0
        active_client_ids = [] # Keep track of clients ready to participate

        for i, result_or_exc in enumerate(init_results):
            client_id = all_client_ids_list[i]
            if isinstance(result_or_exc, Exception):
                logger.error(f"Initialization for client {client_id} failed: {result_or_exc}", exc_info=result_or_exc)
            elif result_or_exc: # True
                logger.info(f"Client {client_id} initialized successfully.")
                client_info = clients[client_id].get_client_info()
                # Server registration might include initial quality metrics if available
                # For now, it's just basic info
                await server.register_client(client_id, client_info)
                successful_inits += 1
                active_client_ids_after_init.append(client_id)
            else: # False
                logger.error(f"Failed to initialize client {client_id}. It will not participate.")
                if client_id in clients: del clients[client_id] # Remove from active simulation dict

        if successful_inits < config.federated.min_clients:
            logger.error(f"Successfully initialized clients ({successful_inits}) < min_clients ({config.federated.min_clients}). Aborting.")
            if server and server.is_running: await server.stop()
            return

        logger.info(f"Initialized {successful_inits}/{num_clients} clients.")
        if successful_inits < config.federated.min_clients:
            logger.error(f"Number of successfully initialized clients ({successful_inits}) is less than min_clients ({config.federated.min_clients}). Aborting.")
            await server.stop()
            return

        # --- Run Federated Rounds ---
        logger.info(f"Starting federated learning process for {config.federated.communication_rounds} rounds...")
        for round_num in range(1, config.federated.communication_rounds + 1):
             if not server.is_running:
                  logger.warning("Server stopped unexpectedly. Halting simulation.")
                  break

             logger.info(f"----- Orchestrating Round {round_num} -----")

             # 1. Server starts round, selects clients, prepares config
             # Uses server.current_round internally
             round_package = await server.start_new_round()

             if round_package is None:
                 # Server logged the reason (max rounds or not enough clients)
                 # If max rounds reached, break the loop
                 if server.current_round >= server.max_rounds:
                      break
                 else: # Not enough clients, wait and continue loop
                      logger.info(f"Waiting before retrying round {round_num}...")
                      await asyncio.sleep(10) # Wait longer if clients are scarce
                      continue # Try starting this round number again

             selected_clients = round_package["selected_clients"]
             logger.info(f"Round {round_num}: Server selected clients: {selected_clients}")

             # 2. Trigger training on selected clients concurrently
             training_tasks = []
             for client_id in selected_clients:
                 if client_id in clients: # Check if client is still active in our simulation list
                      client = clients[client_id]
                      logger.debug(f"Dispatching training task to client {client_id} for round {round_num}")
                      # Pass the parameters and config from the round package
                      training_tasks.append(
                          client.train(
                              round_id=round_package["round_id"],
                              parameters=round_package["parameters"],
                              encrypted=round_package["encrypted"],
                              config_update=round_package["client_config"] # Pass client-specific config
                          )
                      )
                 else:
                      logger.warning(f"Selected client {client_id} not found in active client list. Skipping.")

             # 3. Wait for training results from clients
             # Add a timeout for client training? Maybe handled by server timeout eventually.
             if not training_tasks:
                  logger.warning(f"Round {round_num}: No valid clients to train. Skipping update submission.")
                  # Need to tell server to skip aggregation? Or finalize_round handles empty updates.
                  await server.finalize_round() # Finalize even if no tasks, logs skip
                  continue

             client_results = await asyncio.gather(*training_tasks, return_exceptions=True) # Catch exceptions during training

             # 4. Submit results to server
             num_success = 0
             num_exceptions = 0
             for i, result in enumerate(client_results):
                 client_id = selected_clients[i] # Get client_id based on order
                 if isinstance(result, Exception):
                      logger.error(f"Training task for client {client_id} failed with exception: {result}", exc_info=result)
                      num_exceptions += 1
                      # Optionally inform the server about the client error
                      # await server.client_heartbeat(client_id, {"status": "error", "message": str(result)})
                 elif result and result.get("status") == "success":
                      logger.debug(f"Submitting successful update from {client_id} for round {round_num}")
                      await server.submit_update(client_id, round_num, result)
                      num_success += 1
                 elif result: # Handle cases where client returns {"status": "error", ...}
                      error_msg = result.get("message", "unknown client error")
                      logger.error(f"Training failed on client {client_id} (reported status): {error_msg}")
                 else: # Should not happen if gather catches exceptions
                      logger.error(f"Received unexpected empty result from client {client_id}")


             logger.info(f"Round {round_num}: Submitted {num_success} successful updates to server.")
             if num_exceptions > 0:
                  logger.warning(f"Round {round_num}: Encountered exceptions in {num_exceptions} client training tasks.")


             # 5. Tell server to finalize the round (aggregate, evaluate)
             await server.finalize_round()

             # Optional: Add a small delay between rounds if needed
             # await asyncio.sleep(1)

        logger.info("Demonstrating prediction with the final global model...")
        if server.test_dataloader and len(server.test_dataloader.dataset) > 0:
            # Get a sample. Ensure data_handler used by server has preprocessed it.
            # The server.test_dataloader already provides preprocessed data.
            sample_input_batch, sample_target_batch = next(iter(server.test_dataloader))
            
            # Take the first sample from the batch
            sample_input_single = sample_input_batch[0:1].to(server.device) # Keep batch dim for model
            sample_target_single = sample_target_batch[0:1].item() # Get scalar target

            server.model.eval() # Ensure model is in eval mode
            with torch.no_grad():
                prediction_logit = server.model(sample_input_single)
                # For binary classification with BCEWithLogitsLoss, output is raw logit
                prediction_prob = torch.sigmoid(prediction_logit).item() 
                predicted_class = 1 if prediction_prob > 0.5 else 0
            
            logger.info(f"Sample for prediction: Actual Target={sample_target_single}, "
            f"Predicted Probability (Malignant)={prediction_prob:.4f}, "
            f"Predicted Class (0=Benign, 1=Malignant)={predicted_class}")
        else:
            logger.warning("No server test data available to demonstrate prediction.")

        logger.info("Federated learning simulation loop finished.")
        # Final evaluation is triggered inside finalize_round if max_rounds is reached

    except Exception as e:
        logger.error(f"An error occurred during the federated learning run: {e}", exc_info=True)
    finally:
        # --- Shutdown ---
        if server and server.is_running:
            logger.info("Shutting down server...")
            # Check if final eval already happened
            if not server.current_round >= server.max_rounds:
                await server.run_final_evaluation() # Run if loop exited early
            await server.stop()
        logger.info("Federated learning process terminated.")

async def start_round_simulation(self: FederatedServer) -> Optional[Dict[str, Any]]:
     """Simulation helper: Selects clients and prepares config package."""
     async with self._round_lock:
          if self.current_round >= self.max_rounds:
               self.logger.info("Maximum rounds reached.")
               return None

          self.current_round += 1
          self.logger.info(f"===== Starting Simulated Round {self.current_round}/{self.max_rounds} =====")

          available_clients = await self.get_available_clients()
          if len(available_clients) < self.config.federated.min_clients:
               self.logger.warning(f"Not enough available clients ({len(available_clients)}) < min ({self.config.federated.min_clients}). Round skipped.")
               self.current_round -= 1 # Decrement round counter as it was skipped
               return None

          num_to_select = min(self.config.federated.clients_per_round, len(available_clients))
          self.selected_clients_this_round = np.random.choice(available_clients, num_to_select, replace=False).tolist()
          self.logger.info(f"Selected {len(self.selected_clients_this_round)} clients: {self.selected_clients_this_round}")
          self.client_updates_this_round = {}

          round_config_package = await self._prepare_round_start_config()
          if round_config_package is None:
               self.logger.error("Failed to prepare round configuration.")
               self.current_round -=1 # Decrement round counter
               return None

          # Add selected clients to the package for the orchestrator
          round_config_package["selected_clients"] = self.selected_clients_this_round
          return round_config_package


async def end_round_simulation(self: FederatedServer):
        """Simulation helper: Performs aggregation, evaluation, logging after updates."""
        async with self._round_lock: # Ensure operations happen for the correct round
            num_received = len(self.client_updates_this_round)
            self.logger.info(f"End of Round {self.current_round}: Aggregating {num_received} updates.")

            round_start_time = time.time() # Placeholder for duration logging

            if num_received > 0:
                await self._aggregate_and_update()

                if self.val_dataloader:
                    self.logger.info("Evaluating updated global model (Validation)...")
                    self.global_eval_metrics = await self._evaluate_model(self.val_dataloader)
                    self.logger.info(f"Round {self.current_round} Validation Metrics: {self.global_eval_metrics}")

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

                self._record_history()
                self._log_round_summary(round_start_time) # Pass dummy start time

                if self.current_round % self.config.system.get('checkpoint_frequency', 10) == 0:
                    self._save_checkpoint(f"round_{self.current_round}")
            else:
                self.logger.warning("No valid updates received. Skipping aggregation and evaluation.")
                # Still record a history entry indicating skipped round?
                history_entry = { "round": self.current_round, "status": "skipped_no_updates", "timestamp": datetime.now().isoformat()}
                self.training_history.append(history_entry)


            self.logger.info(f"===== Finished Simulated Round {self.current_round} =====")

            # Check if max rounds reached after this round
            if self.current_round >= self.max_rounds:
                self.logger.info("Maximum rounds reached.")
                await self.run_final_evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Framework - Cancer Detection Example")
    parser.add_argument("--config", default="config.yaml", help="Path to the framework YAML configuration file.")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients to simulate.")
    # Optional: Specify data path directly, otherwise use config
    parser.add_argument("--data", type=str, help="Path to the *full* dataset CSV (overrides config path if provided). Clients will get splits.")
    # Optional: Specify where client data splits are stored/created
    parser.add_argument("--client-data-dir", type=str, default="client_data", help="Directory to store/create client data splits.")

    args = parser.parse_args()

    # --- Load Configuration ---
    try:
        config = FrameworkConfig.from_file(args.config)
    except FileNotFoundError:
        logging.basicConfig(level=logging.ERROR)
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logger.error(f"Error loading configuration from {args.config}: {e}", exc_info=True)
        sys.exit(1)

    # --- Setup Logging (using level from config) ---
    log_storage_dir = Path(config.system.result_dir) / "run_logs"
    setup_logging(config.system.log_level, str(log_storage_dir), config.project_name)
    logger.info(f"Loaded configuration from: {args.config}")
    logger.info(f"Project: {config.project_name}, Task: {config.model.task_type}")
    logger.info(f"HE Enabled: {config.crypto.enabled}, DP Enabled: {config.privacy.differential_privacy}")

    logger.info(f"Loaded configuration from: {args.config}")

    # --- Determine Base Data Path ---
    base_data_path = args.data if args.data else config.data.data_path
    if not Path(base_data_path).is_file():
         logger.error(f"Base data file not found or not a file: {base_data_path}")
         sys.exit(1)
    # Update config path if overridden by args.data - IMPORTANT for server eval
    config.data.data_path = os.path.abspath(base_data_path)
    logger.info(f"Using base dataset: {config.data.data_path}")


    # --- Prepare Client Data ---
    client_data_dir = Path(args.client_data_dir)
    try:
        client_paths = asyncio.run(prepare_data(
            num_clients=args.clients,
            base_data_path=config.data.data_path,
            target_column=config.data.target_column,
            task_type=config.model.task_type,
            data_dir=client_data_dir
        ))
    except Exception as e:
        logger.error(f"Failed to prepare client data: {e}", exc_info=True)
        sys.exit(1)

    # --- Add simulation methods to server ---
    # This feels a bit hacky, but avoids modifying the server class directly here
    # FederatedServer.start_round_simulation = start_round_simulation
    # FederatedServer.end_round_simulation = end_round_simulation

    # --- Run Main FL Process ---
    try:
        asyncio.run(run_federated_learning(config, args.clients, client_paths))
    except KeyboardInterrupt:
        logger.info("Federated learning interrupted by user.")
    except Exception as e:
        logger.critical(f"Federated learning terminated due to critical error: {e}", exc_info=True)

    logger.info("Main script finished.")