import os
import logging
import asyncio
import time
import json
import pickle
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from pathlib import Path
import uuid
from datetime import datetime

from federated_learning_framework.config import FrameworkConfig
from federated_learning_framework.crypto_engine import CryptoEngine
from federated_learning_framework.models import create_model
from federated_learning_framework.data_handler import DataHandler
from federated_learning_framework.privacy import DifferentialPrivacy

class FederatedServer:
    """
    Server for federated learning with homomorphic encryption support.
    Coordinates the training process across multiple clients and securely
    aggregates model updates.
    """
    
    def __init__(self, config: FrameworkConfig):
        """
        Initialize the federated learning server.
        
        Args:
            config: Framework configuration
        """
        self.logger = logging.getLogger("federated.server")
        self.config = config
        
        # Set up saving directories
        self.checkpoint_dir = Path(config.system.checkpoint_dir)
        self.result_dir = Path(config.system.result_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        torch.manual_seed(config.system.seed)
        np.random.seed(config.system.seed)
        
        # Initialize crypto engine
        self.crypto_engine = CryptoEngine(config.crypto)
        
        # Initialize privacy mechanism
        if config.privacy.differential_privacy:
            self.dp_engine = DifferentialPrivacy(
                epsilon=config.privacy.dp_epsilon,
                delta=config.privacy.dp_delta,
                noise_multiplier=config.privacy.dp_noise_multiplier,
                clipping_norm=config.privacy.gradient_clipping
            )
        else:
            self.dp_engine = None
        
        # Create global model
        self.device = torch.device(config.system.device)
        self.model = create_model(config.model, 
                                 input_shape=config.data.input_shape,
                                 output_shape=config.data.output_shape)
        self.model.to(self.device)
        
        # Initialize optimizer for server-side optimization (e.g., for FedAvgM)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                       lr=config.federated.server_learning_rate)
        
        # Server state
        self.current_round = 0
        self.max_rounds = config.federated.communication_rounds
        self.clients = {}  # Connected clients
        self.selected_clients = []  # Clients selected for current round
        self.round_results = {}  # Results from current round
        self.training_history = []  # Track metrics across rounds
        self.eval_metrics = {}  # Latest evaluation metrics
        self.best_model_metrics = {"accuracy": 0.0, "round": 0}  # Track best model
        
        # Data handler for validation/testing
        self.data_handler = DataHandler(config.data)
        
        # Locks for thread safety
        self._model_lock = asyncio.Lock()
        self._client_lock = asyncio.Lock()
        
        # Flag to track if server is running
        self.is_running = False
        
        self.logger.info(f"Federated server initialized for {config.project_name}")
    
    async def start(self):
        """Start the federated learning server"""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("Starting federated learning server")
        
        # Initialize server data (validation/test sets)
        await self._initialize_data()
        
        # Save initial model
        self._save_checkpoint("initial")
        
        # Save crypto context for client distribution
        if self.crypto_engine.is_enabled():
            self.crypto_engine.save_context(os.path.join(self.checkpoint_dir, "crypto_context"))
    
    async def stop(self):
        """Stop the federated learning server"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.logger.info("Stopping federated learning server")
        
        # Save final model
        self._save_checkpoint("final")
        
        # Save training history
        self._save_history()
    
    async def _initialize_data(self):
        """Initialize server data for validation/testing"""
        try:
            # Load validation and test data
            self.logger.info("Loading validation and test data")
            _, self.val_dataloader, self.test_dataloader = await asyncio.to_thread(
                self.data_handler.load_data,
                val_split=self.config.data.val_split,
                test_split=self.config.data.test_split
            )
            
            self.logger.info("Server data initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize data: {e}")
            raise
    
    async def register_client(self, client_id: str, client_info: Dict[str, Any]) -> bool:
        """
        Register a new client with the server.
        
        Args:
            client_id: Unique client identifier
            client_info: Client metadata and capabilities
            
        Returns:
            Whether registration was successful
        """
        async with self._client_lock:
            if client_id in self.clients:
                self.logger.warning(f"Client {client_id} already registered")
                return False
            
            self.clients[client_id] = {
                "info": client_info,
                "last_seen": datetime.now(),
                "status": "registered",
                "train_size": client_info.get("train_size", 0),
                "rounds_participated": 0
            }
            
            self.logger.info(f"Registered client {client_id} ({client_info.get('name', 'unnamed')})")
            return True
    
    async def unregister_client(self, client_id: str) -> bool:
        """
        Unregister a client from the server.
        
        Args:
            client_id: Client to unregister
            
        Returns:
            Whether unregistration was successful
        """
        async with self._client_lock:
            if client_id not in self.clients:
                self.logger.warning(f"Cannot unregister unknown client: {client_id}")
                return False
            
            del self.clients[client_id]
            self.logger.info(f"Unregistered client {client_id}")
            return True
    
    async def client_heartbeat(self, client_id: str, client_status: Dict[str, Any] = None) -> bool:
        """
        Process client heartbeat to track active clients.
        
        Args:
            client_id: Client sending the heartbeat
            client_status: Optional status update from client
            
        Returns:
            Whether heartbeat was processed
        """
        async with self._client_lock:
            if client_id not in self.clients:
                self.logger.warning(f"Heartbeat from unknown client: {client_id}")
                return False
            
            client = self.clients[client_id]
            client["last_seen"] = datetime.now()
            client["status"] = "active"
            
            # Update client status if provided
            if client_status:
                for key, value in client_status.items():
                    if key != "info":  # Don't overwrite client info
                        client[key] = value
            
            return True
    
    async def get_active_clients(self) -> List[str]:
        """
        Get a list of currently active clients.
        
        Returns:
            List of active client IDs
        """
        async with self._client_lock:
            # Client is considered active if seen in the last 5 minutes
            cutoff = datetime.now().timestamp() - 300  # 5 minutes in seconds
            
            active = []
            for client_id, client in self.clients.items():
                if client["last_seen"].timestamp() >= cutoff and client["status"] != "error":
                    active.append(client_id)
            
            return active
    
    async def select_clients(self, num_clients: int = None) -> List[str]:
        """
        Select clients for the next training round.
        
        Args:
            num_clients: Number of clients to select (default: from config)
            
        Returns:
            List of selected client IDs
        """
        if num_clients is None:
            num_clients = self.config.federated.clients_per_round
        
        # Get active clients
        active_clients = await self.get_active_clients()
        
        if len(active_clients) < self.config.federated.min_clients:
            self.logger.warning(f"Not enough active clients: {len(active_clients)} < {self.config.federated.min_clients}")
            return []
        
        # Select clients (could implement different selection strategies here)
        # For now, simple random selection
        selected = np.random.choice(
            active_clients,
            min(num_clients, len(active_clients)),
            replace=False
        ).tolist()
        
        self.logger.info(f"Selected {len(selected)} clients for round {self.current_round+1}")
        self.selected_clients = selected
        return selected
    
    async def start_round(self) -> Dict[str, Any]:
        """
        Start a new federated learning round.
        
        Returns:
            Round configuration for clients
        """
        # Check if we've reached the maximum rounds
        if self.current_round >= self.max_rounds:
            self.logger.info(f"Maximum rounds reached: {self.max_rounds}")
            return None
        
        # Select clients for this round
        selected_clients = await self.select_clients()
        if not selected_clients:
            self.logger.warning("No clients selected, skipping round")
            return None
        
        # Increment round counter
        self.current_round += 1
        self.round_results = {}
        
        self.logger.info(f"Starting federated round {self.current_round}")
        
        # Get current model parameters
        async with self._model_lock:
            # Export model parameters
            if self.crypto_engine.is_enabled():
                # Encrypt parameters for secure transmission
                model_params = self.crypto_engine.encrypt_torch_params(self.model)
                encrypted = True
            else:
                # Export plain parameters
                model_params = {name: param.cpu().detach().numpy() 
                               for name, param in self.model.named_parameters()}
                encrypted = False
        
        # Prepare round configuration
        round_config = {
            "round_id": self.current_round,
            "timestamp": datetime.now().isoformat(),
            "selected_clients": selected_clients,
            "parameters": model_params,
            "encrypted": encrypted,
            "config": {
                "local_epochs": self.config.federated.local_epochs,
                "batch_size": self.config.data.batch_size,
                "learning_rate": self.config.federated.client_learning_rate,
                "proximal_mu": self.config.federated.proximal_mu
            }
        }
        
        return round_config
    
    async def submit_update(self, client_id: str, round_id: int, 
                           update: Dict[str, Any]) -> bool:
        """
        Process model update from a client.
        
        Args:
            client_id: Client submitting the update
            round_id: Training round ID
            update: Model update including parameters and metrics
            
        Returns:
            Whether update was accepted
        """
        # Verify client is valid and selected for this round
        if client_id not in self.clients:
            self.logger.warning(f"Update from unknown client: {client_id}")
            return False
        
        if client_id not in self.selected_clients:
            self.logger.warning(f"Update from non-selected client: {client_id}")
            return False
        
        # Check round ID
        if round_id != self.current_round:
            self.logger.warning(f"Update for wrong round: got {round_id}, expected {self.current_round}")
            return False
        
        # Store update
        self.round_results[client_id] = update
        
        # Update client status
        async with self._client_lock:
            self.clients[client_id]["rounds_participated"] += 1
            self.clients[client_id]["last_update"] = datetime.now()
        
        self.logger.info(f"Received update from client {client_id} for round {round_id}")
        
        # Check if we have enough updates to aggregate
        if len(self.round_results) >= len(self.selected_clients):
            self.logger.info(f"Received all {len(self.selected_clients)} client updates for round {round_id}")
            
            # Trigger aggregation
            await self._aggregate_and_update()
        
        return True
    
    async def _aggregate_and_update(self):
        """Aggregate client updates and update the global model"""
        updates = []
        weights = []
        
        # Extract updates and weights
        for client_id, update in self.round_results.items():
            if "parameters" not in update:
                self.logger.warning(f"Invalid update from client {client_id}: missing 'parameters'")
                continue
            
            params = update["parameters"]
            sample_size = update.get("sample_size", self.clients[client_id]["train_size"])
            
            updates.append(params)
            weights.append(sample_size)
        
        if not updates:
            self.logger.warning("No valid updates to aggregate")
            return
        
        try:
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(weights)] * len(weights)
            
            # Aggregate updates (with or without encryption)
            if self.crypto_engine.is_enabled():
                # Perform secure aggregation
                self.logger.info("Performing secure aggregation with homomorphic encryption")
                
                # Aggregate each parameter separately
                aggregated_params = {}
                for param_name in updates[0].keys():
                    param_updates = [update[param_name] for update in updates]
                    aggregated_param = self._aggregate_encrypted_param(param_updates, weights)
                    aggregated_params[param_name] = aggregated_param
                
                # Decrypt and apply updates
                async with self._model_lock:
                    self.crypto_engine.decrypt_to_torch_params(self.model, aggregated_params)
                    
            else:
                # Perform standard aggregation
                self.logger.info("Performing standard FedAvg aggregation")
                
                # Apply federated averaging
                with torch.no_grad():
                    async with self._model_lock:
                        for param_name, param in self.model.named_parameters():
                            # Reset parameter
                            param_update = torch.zeros_like(param)
                            
                            # Add weighted contributions
                            for i, update in enumerate(updates):
                                if param_name in update:
                                    update_tensor = torch.tensor(update[param_name], device=self.device)
                                    param_update += update_tensor * weights[i]
                            
                            # Apply update
                            param.copy_(param_update)
            
            # Apply differential privacy if enabled
            if self.dp_engine is not None:
                self.logger.info("Applying differential privacy to aggregated model")
                async with self._model_lock:
                    self.dp_engine.add_noise_to_model(self.model)
            
            # Evaluate updated model
            eval_metrics = await self._evaluate_model()
            
            # Save training history
            history_entry = {
                "round": self.current_round,
                "timestamp": datetime.now().isoformat(),
                "metrics": eval_metrics,
                "num_clients": len(self.round_results),
                "client_metrics": {
                    client_id: {
                        "train_loss": update.get("train_loss"),
                        "train_accuracy": update.get("train_accuracy"),
                        "samples": update.get("sample_size", 0)
                    } for client_id, update in self.round_results.items() if "train_loss" in update
                }
            }
            self.training_history.append(history_entry)
            
            # Check and save if best model
            if eval_metrics.get("accuracy", 0) > self.best_model_metrics["accuracy"]:
                self.best_model_metrics = {
                    "accuracy": eval_metrics.get("accuracy", 0),
                    "round": self.current_round,
                    "metrics": eval_metrics
                }
                self._save_checkpoint("best")
            
            # Save periodic checkpoint
            if self.current_round % 10 == 0:
                self._save_checkpoint(f"round_{self.current_round}")
            
            self.logger.info(f"Round {self.current_round} completed. "
                           f"Validation accuracy: {eval_metrics.get('accuracy', 0):.4f}")
                
        except Exception as e:
            self.logger.error(f"Error during aggregation: {e}", exc_info=True)
    
    def _aggregate_encrypted_param(self, param_updates: List[Any], weights: List[float]):
        """
        Aggregate encrypted parameter updates.
        
        Args:
            param_updates: List of encrypted parameter updates
            weights: List of weights for aggregation
            
        Returns:
            Aggregated encrypted parameter
        """
        # Get the parameter type from the first update
        param_type = param_updates[0].get("type", "vector")
        shape = param_updates[0].get("shape")
        
        if param_type == "vector":
            # Extract encrypted vectors
            encrypted_vectors = [p["data"] for p in param_updates]
            # Perform secure aggregation
            aggregated = self.crypto_engine.secure_aggregation(encrypted_vectors, weights)
            return {"type": "vector", "shape": shape, "data": aggregated}
        
        elif param_type == "matrix":
            # For matrices, aggregate row by row
            encrypted_matrices = [p["data"] for p in param_updates]
            aggregated_rows = []
            
            # Determine number of rows from first matrix
            num_rows = len(encrypted_matrices[0])
            
            for row_idx in range(num_rows):
                # Extract corresponding row from each matrix
                row_updates = [matrix[row_idx] for matrix in encrypted_matrices]
                # Aggregate this row
                aggregated_row = self.crypto_engine.secure_aggregation(row_updates, weights)
                aggregated_rows.append(aggregated_row)
                
            return {"type": "matrix", "shape": shape, "data": aggregated_rows}
        
        elif param_type == "tensor":
            # For tensors, aggregate the flattened vectors
            encrypted_vectors = [p["data"] for p in param_updates]
            aggregated = self.crypto_engine.secure_aggregation(encrypted_vectors, weights)
            return {"type": "tensor", "shape": shape, "data": aggregated}
        
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    async def _evaluate_model(self) -> Dict[str, float]:
        """
        Evaluate the current global model on validation data.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating global model on validation data")
        
        # Switch model to eval mode
        self.model.eval()
        
        try:
            # Evaluate on validation data
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            # Classification task metrics
            if self.config.task_type == "classification":
                criterion = torch.nn.CrossEntropyLoss()
                
                with torch.no_grad():
                    for batch in self.val_dataloader:
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        
                        _, predicted = torch.max(outputs, 1)
                        correct = (predicted == targets).sum().item()
                        
                        total_loss += loss.item() * inputs.size(0)
                        total_correct += correct
                        total_samples += inputs.size(0)
            
            # Regression task metrics
            elif self.config.task_type == "regression":
                criterion = torch.nn.MSELoss()
                
                with torch.no_grad():
                    for batch in self.val_dataloader:
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        
                        total_loss += loss.item() * inputs.size(0)
                        total_samples += inputs.size(0)
            
            # Compute metrics
            avg_loss = total_loss / max(total_samples, 1)
            metrics = {"loss": avg_loss}
            
            # Add task-specific metrics
            if self.config.task_type == "classification":
                accuracy = total_correct / max(total_samples, 1)
                metrics["accuracy"] = accuracy
            
            self.eval_metrics = metrics
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {e}", exc_info=True)
            return {"error": str(e)}
        finally:
            # Switch back to training mode
            self.model.train()
    
    async def get_test_metrics(self) -> Dict[str, float]:
        """
        Evaluate the current global model on test data.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating global model on test data")
        
        # Switch model to eval mode
        self.model.eval()
        
        try:
            # Similar to validation but uses test data
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            # Classification task metrics
            if self.config.task_type == "classification":
                criterion = torch.nn.CrossEntropyLoss()
                
                with torch.no_grad():
                    for batch in self.test_dataloader:
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        
                        _, predicted = torch.max(outputs, 1)
                        correct = (predicted == targets).sum().item()
                        
                        total_loss += loss.item() * inputs.size(0)
                        total_correct += correct
                        total_samples += inputs.size(0)
            
            # Regression task metrics
            elif self.config.task_type == "regression":
                criterion = torch.nn.MSELoss()
                
                with torch.no_grad():
                    for batch in self.test_dataloader:
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        
                        total_loss += loss.item() * inputs.size(0)
                        total_samples += inputs.size(0)
            
            # Compute metrics
            avg_loss = total_loss / max(total_samples, 1)
            metrics = {"loss": avg_loss}
            
            # Add task-specific metrics
            if self.config.task_type == "classification":
                accuracy = total_correct / max(total_samples, 1)
                metrics["accuracy"] = accuracy
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during test evaluation: {e}", exc_info=True)
            return {"error": str(e)}
        finally:
            # Switch back to training mode
            self.model.train()
    
    def _save_checkpoint(self, tag: str = None):
        """
        Save model checkpoint.
        
        Args:
            tag: Optional tag for the checkpoint
        """
        try:
            checkpoint_path = self.checkpoint_dir / f"model_{tag if tag else self.current_round}.pt"
            
            # Save model state
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "round": self.current_round,
                "timestamp": datetime.now().isoformat(),
                "metrics": self.eval_metrics
            }
            
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved model checkpoint to {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
    
    def _save_history(self):
        """Save training history to file"""
        try:
            history_path = self.result_dir / "training_history.json"
            
            with open(history_path, "w") as f:
                json.dump({
                    "project": self.config.project_name,
                    "timestamp": datetime.now().isoformat(),
                    "rounds": self.current_round,
                    "best_model": self.best_model_metrics,
                    "history": self.training_history
                }, f, indent=2)
            
            self.logger.info(f"Saved training history to {history_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving history: {e}")
    
    def load_checkpoint(self, path: str):
        """
        Load model from checkpoint.
        
        Args:
            path: Path to the checkpoint file
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.current_round = checkpoint["round"]
            self.eval_metrics = checkpoint.get("metrics", {})
            
            self.logger.info(f"Loaded model checkpoint from {path} (round {self.current_round})")
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")