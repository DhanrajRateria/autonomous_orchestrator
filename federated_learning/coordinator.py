import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
import numpy as np
from datetime import datetime, timedelta
import uuid

from federated_learning.client import FederatedClient
from federated_learning.aggregator import ModelAggregator
from federated_learning.models.anomaly_detector import AnomalyDetectionModel
from federated_learning.models.network_ids import NetworkIDSModel
from crypto.homomorphic_engine import HomomorphicEngine
from core.config import FederatedLearningConfig

class FederatedCoordinator:
    """
    Coordinator for federated learning across multiple cloud environments.
    Manages the distributed learning process while preserving data privacy
    through homomorphic encryption.
    """
    
    def __init__(self, model_config: FederatedLearningConfig, 
                 crypto_engine: HomomorphicEngine, min_clients: int = 3):
        """
        Initialize the federated learning coordinator.
        
        Args:
            model_config: Configuration for federated learning
            crypto_engine: Homomorphic encryption engine
            min_clients: Minimum number of clients required for aggregation
        """
        self.logger = logging.getLogger("federated.coordinator")
        self.config = model_config
        self.crypto_engine = crypto_engine
        self.min_clients = min_clients
        
        # Set up the aggregator
        self.aggregator = ModelAggregator(
            method=self.config.aggregation_method,
            crypto_engine=self.crypto_engine
        )
        
        # Initialize model registry
        self.models = {}
        self._init_models()
        
        # Client management
        self.clients: Dict[str, FederatedClient] = {}
        self.active_clients: Set[str] = set()
        
        # Round management
        self.current_round_id: Optional[str] = None
        self.round_in_progress = False
        self.round_deadline: Optional[datetime] = None
        self.round_results: Dict[str, Any] = {}
        
        # Model versioning
        self.current_model_version = 0
        self.model_update_lock = asyncio.Lock()
        
        self.is_running = False
        self.logger.info("Federated coordinator initialized")
    
    def _init_models(self):
        """Initialize model architectures based on configuration"""
        if self.config.model_architecture == "lstm_anomaly_detector":
            self.models["anomaly_detector"] = AnomalyDetectionModel(
                input_dim=32,  # Log features dimensionality
                hidden_dim=64,
                learning_rate=self.config.learning_rate
            )
        elif self.config.model_architecture == "network_ids":
            self.models["network_ids"] = NetworkIDSModel(
                input_dim=128,  # Network flow features dimensionality 
                learning_rate=self.config.learning_rate
            )
        else:
            self.logger.error(f"Unknown model architecture: {self.config.model_architecture}")
            raise ValueError(f"Unknown model architecture: {self.config.model_architecture}")
        
        self.logger.info(f"Initialized {len(self.models)} model(s)")
    
    async def start(self):
        """Start the federated learning coordinator"""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("Starting federated learning coordinator")
        
        # Start maintenance task
        self._maintenance_task = asyncio.create_task(self._periodic_maintenance())
    
    async def stop(self):
        """Stop the federated learning coordinator"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.logger.info("Stopping federated learning coordinator")
        
        # Cancel the maintenance task
        if hasattr(self, '_maintenance_task'):
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        
        # If a round is in progress, abort it
        if self.round_in_progress:
            await self._abort_current_round("Coordinator shutting down")
    
    async def register_client(self, client_id: str, client_info: Dict[str, Any]) -> bool:
        """
        Register a new federated learning client.
        
        Args:
            client_id: Unique identifier for the client
            client_info: Information about the client
            
        Returns:
            bool: True if registration was successful
        """
        if client_id in self.clients:
            self.logger.warning(f"Client {client_id} already registered")
            return False
        
        # Create new client
        client = FederatedClient(
            client_id=client_id,
            info=client_info,
            crypto_engine=self.crypto_engine
        )
        
        self.clients[client_id] = client
        self.logger.info(f"Registered new client: {client_id}")
        return True
    
    async def deregister_client(self, client_id: str) -> bool:
        """
        Deregister a federated learning client.
        
        Args:
            client_id: Client ID to deregister
            
        Returns:
            bool: True if deregistration was successful
        """
        if client_id not in self.clients:
            self.logger.warning(f"Cannot deregister unknown client: {client_id}")
            return False
        
        # Remove client
        del self.clients[client_id]
        self.active_clients.discard(client_id)
        
        self.logger.info(f"Deregistered client: {client_id}")
        return True
    
    async def client_heartbeat(self, client_id: str) -> bool:
        """
        Handle client heartbeat to track active clients.
        
        Args:
            client_id: Client sending the heartbeat
            
        Returns:
            bool: True if heartbeat was processed
        """
        if client_id not in self.clients:
            self.logger.warning(f"Heartbeat from unknown client: {client_id}")
            return False
        
        # Update client's last seen timestamp
        client = self.clients[client_id]
        client.last_heartbeat = datetime.now()
        self.active_clients.add(client_id)
        
        return True
    
    async def trigger_update_round(self) -> bool:
        """
        Trigger a new federated learning round.
        
        Returns:
            bool: True if round was successfully started
        """
        async with self.model_update_lock:
            if self.round_in_progress:
                self.logger.warning("Cannot start new round: round already in progress")
                return False
            
            active_count = len(self.active_clients)
            if active_count < self.min_clients:
                self.logger.warning(
                    f"Cannot start round: insufficient active clients ({active_count} < {self.min_clients})"
                )
                return False
            
            # Start new round
            self.current_round_id = str(uuid.uuid4())
            self.round_in_progress = True
            self.round_deadline = datetime.now() + timedelta(minutes=15)  # 15-minute timeout
            self.round_results = {}
            
            self.logger.info(
                f"Started new federated learning round {self.current_round_id} "
                f"with {active_count} potential clients"
            )
            
            # Send model training requests to clients
            training_tasks = []
            for client_id in list(self.active_clients):
                task = asyncio.create_task(self._request_client_update(client_id))
                training_tasks.append(task)
            
            # Collect training results in background
            asyncio.create_task(self._collect_round_results(training_tasks))
            return True
    
    async def _request_client_update(self, client_id: str) -> bool:
        """
        Request a model update from a specific client.
        
        Args:
            client_id: Client to request update from
            
        Returns:
            bool: True if request was sent successfully
        """
        if client_id not in self.clients:
            return False
        
        client = self.clients[client_id]
        
        try:
            # Get current model weights to send to client
            model_name = list(self.models.keys())[0]  # Use first model for now
            current_weights = self.models[model_name].get_weights()
            
            # Encrypt weights if enabled
            if self.config.aggregation_method.startswith("secure_"):
                encrypted_weights = self.crypto_engine.encrypt_model_parameters(current_weights)
                result = await client.train_model(
                    round_id=self.current_round_id,
                    model_name=model_name,
                    model_version=self.current_model_version,
                    encrypted_weights=encrypted_weights,
                    epochs=self.config.local_epochs,
                    batch_size=self.config.batch_size
                )
            else:
                result = await client.train_model(
                    round_id=self.current_round_id,
                    model_name=model_name,
                    model_version=self.current_model_version,
                    weights=current_weights,
                    epochs=self.config.local_epochs,
                    batch_size=self.config.batch_size
                )
            
            return True
        except Exception as e:
            self.logger.error(f"Error requesting update from client {client_id}: {e}")
            return False
    
    async def submit_update(self, client_id: str, round_id: str, 
                           model_updates: Dict[str, Any]) -> bool:
        """
        Handle submission of model updates from clients.
        
        Args:
            client_id: Client submitting the update
            round_id: Round ID for the update
            model_updates: Model parameter updates
            
        Returns:
            bool: True if update was accepted
        """
        if not self.round_in_progress or round_id != self.current_round_id:
            self.logger.warning(
                f"Rejected update from client {client_id}: invalid round ID {round_id}"
            )
            return False
        
        if client_id not in self.active_clients:
            self.logger.warning(
                f"Rejected update from inactive client {client_id}"
            )
            return False
        
        # Store the update
        self.round_results[client_id] = model_updates
        self.logger.info(f"Received model update from client {client_id} for round {round_id}")
        
        # Check if we have enough updates to proceed with aggregation
        return True
    
    async def _collect_round_results(self, training_tasks: List[asyncio.Task]):
        """
        Wait for training results and perform model aggregation when ready.
        
        Args:
            training_tasks: List of tasks for client training requests
        """
        try:
            # Wait for all training tasks to complete or timeout
            timeout_sec = (self.round_deadline - datetime.now()).total_seconds()
            if timeout_sec > 0:
                await asyncio.wait(training_tasks, timeout=timeout_sec)
            
            # Check if we have enough results
            if len(self.round_results) < self.min_clients:
                self.logger.warning(
                    f"Insufficient client updates for round {self.current_round_id}: "
                    f"received {len(self.round_results)}, need at least {self.min_clients}"
                )
                await self._abort_current_round("Insufficient client participation")
                return
            
            # Perform model aggregation
            await self._aggregate_and_update_model()
            
        except Exception as e:
            self.logger.error(f"Error in round result collection: {e}")
            await self._abort_current_round(f"Error: {str(e)}")
    
    async def _aggregate_and_update_model(self):
        """Aggregate model updates and apply to global model"""
        try:
            self.logger.info(f"Aggregating {len(self.round_results)} model updates")
            
            model_name = list(self.models.keys())[0]  # Use first model for now
            model = self.models[model_name]
            
            # Extract updates from results
            updates = []
            for client_id, result in self.round_results.items():
                if "weights" in result:
                    updates.append((result["weights"], result.get("sample_size", 1)))
                elif "encrypted_weights" in result:
                    updates.append((result["encrypted_weights"], result.get("sample_size", 1)))
            
            # Aggregate updates
            if updates:
                new_weights = await self.aggregator.aggregate(updates, homomorphic=self.config.aggregation_method.startswith("secure_"))
                
                # Apply aggregated update to model
                model.set_weights(new_weights)
                
                # Update model version
                self.current_model_version += 1
                self.logger.info(f"Updated model to version {self.current_model_version}")
                
                # Notify clients of new model version
                asyncio.create_task(self._notify_clients_of_new_model())
            
            # Finalize round
            self.round_in_progress = False
            self.current_round_id = None
            self.round_results = {}
            
        except Exception as e:
            self.logger.error(f"Error in model aggregation: {e}")
            await self._abort_current_round(f"Aggregation error: {str(e)}")
    
    async def _abort_current_round(self, reason: str):
        """
        Abort the current federated learning round.
        
        Args:
            reason: Reason for aborting the round
        """
        self.logger.warning(f"Aborting round {self.current_round_id}: {reason}")
        self.round_in_progress = False
        self.current_round_id = None
        self.round_results = {}
    
    async def _notify_clients_of_new_model(self):
        """Notify all active clients of new model version"""
        for client_id in list(self.active_clients):
            try:
                client = self.clients.get(client_id)
                if client:
                    await client.notify_new_model_available(self.current_model_version)
            except Exception as e:
                self.logger.warning(f"Failed to notify client {client_id} of new model: {e}")
    
    async def _periodic_maintenance(self):
        """Periodically clean up inactive clients and check round status"""
        while self.is_running:
            try:
                # Clean up inactive clients
                now = datetime.now()
                inactive_threshold = now - timedelta(minutes=10)
                
                for client_id in list(self.clients.keys()):
                    client = self.clients[client_id]
                    if client.last_heartbeat < inactive_threshold:
                        self.logger.info(f"Removing inactive client {client_id}")
                        await self.deregister_client(client_id)
                
                # Check if current round has timed out
                if self.round_in_progress and self.round_deadline and now > self.round_deadline:
                    self.logger.warning(f"Round {self.current_round_id} timed out")
                    await self._abort_current_round("Round timed out")
                
                # Sleep for a while
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in maintenance task: {e}")
                await asyncio.sleep(60)  # Longer delay on error
    
    def get_current_model(self, model_name: str = None):
        """
        Get the current global model for inference.
        
        Args:
            model_name: Name of the model to retrieve (uses first model if None)
            
        Returns:
            The requested model
        """
        if model_name is None:
            model_name = list(self.models.keys())[0]
            
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
            
        return self.models[model_name]