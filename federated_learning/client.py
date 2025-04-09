import asyncio
import logging
import time
import numpy as np
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from crypto.homomorphic_engine import HomomorphicEngine
from federated_learning.models.anomaly_detector import AnomalyDetectionModel
from federated_learning.models.network_ids import NetworkIDSModel

class FederatedClient:
    """
    Client for federated learning, operating within a cloud environment to
    train security models locally and participate in federated updates.
    """
    
    def __init__(self, client_id: str, info: Dict[str, Any], 
                 crypto_engine: HomomorphicEngine = None):
        """
        Initialize the federated learning client.
        
        Args:
            client_id: Unique identifier for this client
            info: Additional information about the client
            crypto_engine: Homomorphic encryption engine
        """
        self.client_id = client_id
        self.info = info
        self.logger = logging.getLogger(f"federated.client.{client_id}")
        self.crypto_engine = crypto_engine
        
        # Tracking client state
        self.last_heartbeat = datetime.now()
        self.training_in_progress = False
        self.current_round_id = None
        
        # Local data stats
        self.local_data_count = 0
        self.last_data_update = None
        
        # Local models (initialized on demand)
        self._models = {}
        
        # Model version tracking
        self.current_model_versions = {}
        
        self.logger.info(f"Federated client {client_id} initialized")
    
    async def train_model(self, round_id: str, model_name: str, model_version: int,
                         weights=None, encrypted_weights=None, 
                         epochs: int = 1, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train model locally with current data.
        
        Args:
            round_id: ID of the federated round
            model_name: Name of model to train
            model_version: Current version of the model
            weights: Model weights (plaintext)
            encrypted_weights: Encrypted model weights
            epochs: Number of local training epochs
            batch_size: Training batch size
            
        Returns:
            Dictionary with training results
        """
        if self.training_in_progress:
            self.logger.warning(f"Training already in progress for round: {self.current_round_id}")
            return {"status": "failed", "reason": "training_in_progress"}
        
        self.training_in_progress = True
        self.current_round_id = round_id
        
        try:
            # Initialize or update local model
            model = await self._get_or_create_model(model_name)
            
            # Update model with provided weights
            if weights is not None:
                model.set_weights(weights)
                using_encryption = False
            elif encrypted_weights is not None:
                # This would decrypt the weights using homomorphic encryption
                if self.crypto_engine:
                    decrypted_weights = self.crypto_engine.decrypt_model_parameters(encrypted_weights)
                    model.set_weights(decrypted_weights)
                    using_encryption = True
                else:
                    raise ValueError("Encrypted weights provided but crypto engine not available")
            else:
                self.logger.warning("Neither weights nor encrypted_weights provided, using current model state")
                using_encryption = False
            
            # Load local training data
            train_data, train_labels = await self._load_local_training_data(model_name)
            
            if len(train_data) == 0:
                self.logger.warning("No local training data available")
                return {
                    "status": "success",
                    "message": "no_local_data",
                    "sample_size": 0
                }
            
            # Train the model
            self.logger.info(f"Starting local training with {len(train_data)} samples for {epochs} epochs")
            training_result = await model.train(
                train_data, train_labels, 
                epochs=epochs, 
                batch_size=batch_size
            )
            
            # Get updated weights
            updated_weights = model.get_weights()
            
            # Update local model version
            self.current_model_versions[model_name] = model_version
            
            # Prepare result
            if using_encryption and self.crypto_engine:
                # Encrypt the updated weights
                encrypted_updated = self.crypto_engine.encrypt_model_parameters(updated_weights)
                result = {
                    "status": "success",
                    "encrypted_weights": encrypted_updated,
                    "sample_size": len(train_data),
                    "training_loss": training_result["loss"],
                    "training_accuracy": training_result["accuracy"]
                }
            else:
                result = {
                    "status": "success",
                    "weights": updated_weights,
                    "sample_size": len(train_data),
                    "training_loss": training_result["loss"],
                    "training_accuracy": training_result["accuracy"]
                }
            
            self.logger.info(f"Local training completed with {result['training_accuracy']:.4f} accuracy")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during local training: {e}", exc_info=True)
            return {"status": "failed", "reason": str(e)}
        finally:
            self.training_in_progress = False
    
    async def _get_or_create_model(self, model_name: str) -> Any:
        """
        Get or create a model instance by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model instance
        """
        if model_name in self._models:
            return self._models[model_name]
        
        # Create new model instance
        if model_name == "anomaly_detector":
            model = AnomalyDetectionModel(
                input_dim=32,
                hidden_dim=64,
                learning_rate=0.01
            )
        elif model_name == "network_ids":
            model = NetworkIDSModel(
                input_dim=128,
                learning_rate=0.01
            )
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        self._models[model_name] = model
        return model
    
    async def _load_local_training_data(self, model_name: str) -> tuple:
        """
        Load local data for model training.
        
        Args:
            model_name: Name of the model to load data for
            
        Returns:
            Tuple of (features, labels)
        """
        # In a real implementation, this would load and preprocess actual security data
        # For demonstration purposes, we generate synthetic data
        
        # Determine data shape based on model type
        if model_name == "anomaly_detector":
            feature_dim = 32
        elif model_name == "network_ids":
            feature_dim = 128
        else:
            feature_dim = 32  # Default
        
        # Generate some synthetic data
        data_size = np.random.randint(50, 200)  # Random number of samples
        self.local_data_count = data_size
        self.last_data_update = datetime.now()
        
        # Create synthetic features and labels
        features = np.random.normal(0, 1, (data_size, feature_dim)).astype(np.float32)
        
        # For anomaly detection: mostly normal samples with some anomalies
        if model_name == "anomaly_detector":
            # Generate mostly 0s (normal) with some 1s (anomalies)
            labels = np.zeros(data_size)
            anomaly_indices = np.random.choice(data_size, size=int(data_size * 0.05), replace=False)
            labels[anomaly_indices] = 1
        else:
            # For network IDS: multi-class classification
            # Generate random classes (normal, dos, probe, u2r, r2l)
            labels = np.random.randint(0, 5, data_size)
            
            # Ensure class imbalance similar to real security data
            normal_indices = np.random.choice(data_size, size=int(data_size * 0.8), replace=False)
            labels[normal_indices] = 0  # Mostly normal traffic
        
        return features, labels
    
    async def notify_new_model_available(self, model_version: int) -> bool:
        """
        Handle notification that a new model version is available.
        
        Args:
            model_version: New model version number
            
        Returns:
            bool: True if notification was acknowledged
        """
        # In a real implementation, this might trigger model fetching
        self.logger.info(f"Notified of new model version {model_version}")
        return True
    
    async def evaluate_model(self, model_name: str, weights=None, encrypted_weights=None) -> Dict[str, Any]:
        """
        Evaluate current model on local data.
        
        Args:
            model_name: Name of the model to evaluate
            weights: Model weights (optional)
            encrypted_weights: Encrypted model weights (optional)
            
        Returns:
            Evaluation metrics
        """
        try:
            # Get or create model
            model = await self._get_or_create_model(model_name)
            
            # Update model weights if provided
            if weights is not None:
                model.set_weights(weights)
            elif encrypted_weights is not None and self.crypto_engine:
                decrypted_weights = self.crypto_engine.decrypt_model_parameters(encrypted_weights)
                model.set_weights(decrypted_weights)
            
            # Load evaluation data
            eval_data, eval_labels = await self._load_local_training_data(model_name)
            
            if len(eval_data) == 0:
                return {
                    "status": "success",
                    "message": "no_evaluation_data",
                    "metrics": {}
                }
            
            # Evaluate the model
            metrics = await model.evaluate(eval_data, eval_labels)
            
            return {
                "status": "success",
                "metrics": metrics,
                "sample_size": len(eval_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {e}", exc_info=True)
            return {"status": "failed", "reason": str(e), "metrics": {}}