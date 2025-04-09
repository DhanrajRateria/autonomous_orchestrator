import os
import logging
import asyncio
import time
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from federated_learning_framework.config import FrameworkConfig
from federated_learning_framework.crypto_engine import CryptoEngine
from federated_learning_framework.models import create_model
from federated_learning_framework.data_handler import DataHandler

class FederatedClient:
    """
    Client for federated learning with homomorphic encryption support.
    Trains models locally on private data and communicates with the server
    for federated learning.
    """
    
    def __init__(self, client_id: str, config: FrameworkConfig, data_path: str = None):
        """
        Initialize the federated learning client.
        
        Args:
            client_id: Unique client identifier
            config: Framework configuration
            data_path: Path to client's dataset (overrides config)
        """
        self.logger = logging.getLogger(f"federated.client.{client_id}")
        self.client_id = client_id
        self.config = config
        
        # Override data path if provided
        if data_path:
            self.config.data.data_path = data_path
        
        # Set up directories
        self.checkpoint_dir = Path(config.system.checkpoint_dir) / f"client_{client_id}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        torch.manual_seed(config.system.seed)
        np.random.seed(config.system.seed)
        
        # Initialize crypto engine
        self.crypto_engine = CryptoEngine(config.crypto)
        
        # Set device
        self.device = torch.device(config.system.device)
        
        # Prepare local model
        self.model = create_model(config.model, 
                                 input_shape=config.data.input_shape,
                                 output_shape=config.data.output_shape)
        self.model.to(self.device)
        
        # Data handler
        self.data_handler = DataHandler(config.data)
        
        # Client state
        self.current_round = 0
        self.training_history = []
        self.is_training = False
        
        self.logger.info(f"Federated client {client_id} initialized")
    
    async def initialize(self):
        """Initialize client data and resources"""
        try:
            # Load dataset
            self.logger.info("Loading client dataset")
            self.train_dataloader, self.val_dataloader, _ = await asyncio.to_thread(
                self.data_handler.load_data,
                val_split=self.config.data.val_split,
                test_split=0.0  # No test set for clients
            )
            
            # Get data size for weighting
            self.train_size = len(self.train_dataloader.dataset)
            
            self.logger.info(f"Client dataset loaded with {self.train_size} training samples")
            
            # Load crypto context if needed
            if self.crypto_engine.is_enabled():
                try:
                    context_path = Path(self.config.system.checkpoint_dir) / "crypto_context"
                    if context_path.exists():
                        self.crypto_engine.load_context(str(context_path))
                        self.logger.info("Loaded crypto context from server")
                except Exception as e:
                    self.logger.warning(f"Could not load crypto context: {e}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing client: {e}")
            return False
    
    async def train(self, round_id: int, parameters: Dict[str, Any], 
                  encrypted: bool = False, epochs: int = None, 
                  learning_rate: float = None) -> Dict[str, Any]:
        """
        Train the local model using the provided parameters.
        
        Args:
            round_id: Current round number
            parameters: Model parameters from server
            encrypted: Whether parameters are encrypted
            epochs: Number of local epochs (overrides config)
            learning_rate: Learning rate (overrides config)
            
        Returns:
            Dictionary with training results and updated parameters
        """
        if self.is_training:
            self.logger.warning("Already training, cannot start new training job")
            return {"status": "error", "message": "Already training"}
        
        try:
            self.is_training = True
            self.current_round = round_id
            self.logger.info(f"Starting local training for round {round_id}")
            
            # Set training parameters
            epochs = epochs or self.config.federated.local_epochs
            learning_rate = learning_rate or self.config.federated.client_learning_rate
            batch_size = self.config.data.batch_size
            
            # Update local model with server parameters
            await self._update_local_model(parameters, encrypted)
            
            # Prepare optimizer
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9
            )
            
            # Set up loss function based on task
            if self.config.task_type == "classification":
                criterion = torch.nn.CrossEntropyLoss()
            elif self.config.task_type == "regression":
                criterion = torch.nn.MSELoss()
            else:
                criterion = torch.nn.CrossEntropyLoss()  # Default
            
            # Train for specified number of epochs
            self.model.train()
            training_results = await self._train_epochs(
                optimizer, criterion, epochs, batch_size
            )
            
            # Extract updated parameters (with or without encryption)
            if encrypted and self.crypto_engine.is_enabled():
                updated_params = self.crypto_engine.encrypt_torch_params(self.model)
            else:
                updated_params = {
                    name: param.cpu().detach().numpy()
                    for name, param in self.model.named_parameters()
                }
            
            # Prepare result
            result = {
                "status": "success",
                "round_id": round_id,
                "parameters": updated_params,
                "encrypted": encrypted,
                "sample_size": self.train_size,
                "train_loss": training_results["train_loss"],
                "train_accuracy": training_results.get("train_accuracy"),
                "val_loss": training_results["val_loss"],
                "val_accuracy": training_results.get("val_accuracy"),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store result in history
            self.training_history.append({
                "round": round_id,
                "metrics": {
                    "train_loss": training_results["train_loss"],
                    "train_accuracy": training_results.get("train_accuracy"),
                    "val_loss": training_results["val_loss"],
                    "val_accuracy": training_results.get("val_accuracy")
                }
            })
            
            # Save checkpoint periodically
            if round_id % 10 == 0:
                self._save_checkpoint(f"round_{round_id}")
            
            self.logger.info(f"Completed local training for round {round_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during local training: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
        finally:
            self.is_training = False
    
    async def _train_epochs(self, optimizer, criterion, epochs, batch_size) -> Dict[str, float]:
        """Run training for multiple epochs"""
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0
        is_classification = self.config.task_type == "classification"
        
        self.model.train()
        
        # Train for specified epochs
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item() * inputs.size(0)
                epoch_samples += inputs.size(0)
                
                # Classification metrics
                if is_classification:
                    _, predicted = torch.max(outputs, 1)
                    epoch_correct += (predicted == targets).sum().item()
                
                # Log progress
                if (batch_idx + 1) % 10 == 0:
                    self.logger.debug(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(self.train_dataloader)}, "
                                    f"Loss: {loss.item():.4f}")
            
            # Epoch statistics
            epoch_avg_loss = epoch_loss / epoch_samples
            
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_avg_loss:.4f}")
            
            if is_classification:
                epoch_accuracy = epoch_correct / epoch_samples
                self.logger.info(f"Epoch {epoch+1}/{epochs}, Accuracy: {epoch_accuracy:.4f}")
            
            # Update totals
            total_train_loss += epoch_loss
            total_train_samples += epoch_samples
            if is_classification:
                total_train_correct += epoch_correct
        
        # Calculate final training metrics
        avg_train_loss = total_train_loss / total_train_samples
        
        # Evaluate on validation set
        val_metrics = await self._evaluate()
        
        # Prepare results
        result = {
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"]
        }
        
        if is_classification:
            result["train_accuracy"] = total_train_correct / total_train_samples
            result["val_accuracy"] = val_metrics.get("accuracy")
        
        return result
    
    async def _evaluate(self) -> Dict[str, float]:
        """Evaluate the model on validation data"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        is_classification = self.config.task_type == "classification"
        
        if is_classification:
            criterion = torch.nn.CrossEntropyLoss()
        else:
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
                
                if is_classification:
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / total_samples
        metrics = {"loss": avg_loss}
        
        if is_classification:
            accuracy = total_correct / total_samples
            metrics["accuracy"] = accuracy
        
        return metrics
    
    async def _update_local_model(self, parameters: Dict[str, Any], encrypted: bool):
        """Update local model with server parameters"""
        if encrypted and self.crypto_engine.is_enabled():
            # Decrypt parameters and update model
            self.crypto_engine.decrypt_to_torch_params(self.model, parameters)
        else:
            # Directly update model with plain parameters
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in parameters:
                        param_tensor = torch.tensor(parameters[name], device=self.device)
                        param.copy_(param_tensor)
    
    def _save_checkpoint(self, tag: str = None):
        """Save model checkpoint"""
        try:
            checkpoint_path = self.checkpoint_dir / f"model_{tag if tag else self.current_round}.pt"
            
            # Save model state
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "round": self.current_round,
                "timestamp": datetime.now().isoformat()
            }
            
            torch.save(checkpoint, checkpoint_path)
            self.logger.debug(f"Saved model checkpoint to {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get client information for registration"""
        return {
            "id": self.client_id,
            "name": f"client_{self.client_id}",
            "train_size": getattr(self, "train_size", 0),
            "device": str(self.device),
            "supports_encryption": self.crypto_engine.is_enabled(),
            "timestamp": datetime.now().isoformat()
        }
    
    def load_checkpoint(self, path: str) -> bool:
        """
        Load model from checkpoint.
        
        Args:
            path: Path to the checkpoint file
            
        Returns:
            Whether loading was successful
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.current_round = checkpoint.get("round", 0)
            
            self.logger.info(f"Loaded model checkpoint from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return False