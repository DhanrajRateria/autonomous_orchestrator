import unittest
import asyncio
import os
import sys
import tempfile
import shutil
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from federated_learning.coordinator import FederatedCoordinator
from federated_learning.client import FederatedClient
from federated_learning.aggregator import ModelAggregator
from crypto.homomorphic_engine import HomomorphicEngine
from core.config import FederatedLearningConfig

class TestFederatedLearningIntegration(unittest.TestCase):
    """
    Integration tests for federated learning components.
    Tests the interaction between coordinator, clients, and the aggregator.
    """
    
    def setUp(self):
        """Set up the test environment"""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a HomomorphicEngine with small parameters for tests
        self.crypto_engine = HomomorphicEngine(key_size=1024, security_level=128)
        
        # Create configuration
        self.config = FederatedLearningConfig(
            min_clients=2,
            aggregation_method="secure_fedavg",
            rounds_per_update=1,
            local_epochs=1,
            batch_size=16,
            learning_rate=0.01,
            model_architecture="lstm_anomaly_detector"
        )
        
        # Create coordinator
        self.coordinator = FederatedCoordinator(
            model_config=self.config,
            crypto_engine=self.crypto_engine,
            min_clients=2
        )
        
        # Create test clients
        self.client_info = {
            "provider_type": "test",
            "region": "us-west-1",
            "resources": 50
        }
        
        # We'll initialize clients in the tests
        self.clients = []
        
        # Get event loop
        self.loop = asyncio.get_event_loop()
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
        
        # Clean up any tasks
        tasks = asyncio.all_tasks(self.loop)
        for task in tasks:
            task.cancel()
    
    def test_client_registration(self):
        """Test client registration with the coordinator"""
        async def _test():
            # Start coordinator
            await self.coordinator.start()
            
            # Register clients
            client_id1 = "test-client-1"
            client_id2 = "test-client-2"
            
            success1 = await self.coordinator.register_client(client_id1, self.client_info)
            success2 = await self.coordinator.register_client(client_id2, self.client_info)
            
            # Verify registration
            self.assertTrue(success1)
            self.assertTrue(success2)
            self.assertEqual(len(self.coordinator.clients), 2)
            
            # Register same client again should fail
            success3 = await self.coordinator.register_client(client_id1, self.client_info)
            self.assertFalse(success3)
            
            # Try heartbeat for both clients
            heartbeat1 = await self.coordinator.client_heartbeat(client_id1)
            heartbeat2 = await self.coordinator.client_heartbeat(client_id2)
            self.assertTrue(heartbeat1)
            self.assertTrue(heartbeat2)
            
            # Verify active clients
            self.assertEqual(len(self.coordinator.active_clients), 2)
            
            # Stop coordinator
            await self.coordinator.stop()
        
        self.loop.run_until_complete(_test())
    
    def test_federated_round(self):
        """Test a complete federated learning round"""
        async def _test():
            # Start coordinator
            await self.coordinator.start()
            
            # Create and register clients
            client1 = FederatedClient(
                client_id="test-client-1",
                info=self.client_info,
                crypto_engine=self.crypto_engine
            )
            client2 = FederatedClient(
                client_id="test-client-2", 
                info=self.client_info,
                crypto_engine=self.crypto_engine
            )
            
            self.clients.append(client1)
            self.clients.append(client2)
            
            await self.coordinator.register_client(client1.client_id, self.client_info)
            await self.coordinator.register_client(client2.client_id, self.client_info)
            
            # Activate clients via heartbeat
            await self.coordinator.client_heartbeat(client1.client_id)
            await self.coordinator.client_heartbeat(client2.client_id)
            
            # Save the current model version
            old_version = self.coordinator.current_model_version
            
            # Start a federated round
            started = await self.coordinator.trigger_update_round()
            self.assertTrue(started)
            
            # Wait for the round to complete (in real code we'd have actual clients responding)
            # Here we simulate client updates directly
            current_round_id = self.coordinator.current_round_id
            self.assertIsNotNone(current_round_id)
            
            # Get model and perform "training"
            model_name = list(self.coordinator.models.keys())[0]
            model = self.coordinator.models[model_name]
            weights = model.get_weights()
            
            # Simulate client submissions
            dummy_update1 = {
                "weights": [w * 1.01 for w in weights],  # Slightly modified weights
                "sample_size": 50
            }
            
            dummy_update2 = {
                "weights": [w * 0.99 for w in weights],  # Slightly modified weights
                "sample_size": 75
            }
            
            await self.coordinator.submit_update(client1.client_id, current_round_id, dummy_update1)
            await self.coordinator.submit_update(client2.client_id, current_round_id, dummy_update2)
            
            # Wait for aggregation (would happen automatically, but we'll force it)
            await self.coordinator._aggregate_and_update_model()
            
            # Verify the model version was updated
            self.assertGreater(self.coordinator.current_model_version, old_version)
            
            # Stop coordinator
            await self.coordinator.stop()
        
        self.loop.run_until_complete(_test())
    
    def test_secure_aggregation(self):
        """Test secure aggregation with homomorphic encryption"""
        async def _test():
            # Start coordinator with secure aggregation
            await self.coordinator.start()
            
            # Create and register clients
            client1 = FederatedClient(
                client_id="secure-client-1",
                info=self.client_info,
                crypto_engine=self.crypto_engine
            )
            client2 = FederatedClient(
                client_id="secure-client-2", 
                info=self.client_info,
                crypto_engine=self.crypto_engine
            )
            
            await self.coordinator.register_client(client1.client_id, self.client_info)
            await self.coordinator.register_client(client2.client_id, self.client_info)
            
            # Activate clients
            await self.coordinator.client_heartbeat(client1.client_id)
            await self.coordinator.client_heartbeat(client2.client_id)
            
            # Start a federated round
            started = await self.coordinator.trigger_update_round()
            self.assertTrue(started)
            
            # Get round ID
            current_round_id = self.coordinator.current_round_id
            
            # Get model
            model_name = list(self.coordinator.models.keys())[0]
            model = self.coordinator.models[model_name]
            weights = model.get_weights()
            
            # Encrypt weights
            encrypted_weights1 = self.crypto_engine.encrypt_model_parameters(weights)
            encrypted_weights2 = self.crypto_engine.encrypt_model_parameters(weights)
            
            # Simulate client submissions with encrypted weights
            update1 = {
                "encrypted_weights": encrypted_weights1,
                "sample_size": 50
            }
            
            update2 = {
                "encrypted_weights": encrypted_weights2,
                "sample_size": 75
            }
            
            await self.coordinator.submit_update(client1.client_id, current_round_id, update1)
            await self.coordinator.submit_update(client2.client_id, current_round_id, update2)
            
            # Force aggregation
            await self.coordinator._aggregate_and_update_model()
            
            # Verify round completed
            self.assertFalse(self.coordinator.round_in_progress)
            
            # Stop coordinator
            await self.coordinator.stop()
        
        self.loop.run_until_complete(_test())

if __name__ == '__main__':
    unittest.main()