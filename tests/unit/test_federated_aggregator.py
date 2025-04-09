import unittest
import numpy as np
import os
import sys
import asyncio
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from federated_learning.aggregator import ModelAggregator
from crypto.homomorphic_engine import HomomorphicEngine

class TestModelAggregator(unittest.TestCase):
    """
    Unit tests for the ModelAggregator class.
    """
    
    def setUp(self):
        """Set up the test environment"""
        # Create a HomomorphicEngine with small parameters for tests
        self.crypto_engine = HomomorphicEngine(key_size=1024, security_level=128)
        
        # Create aggregator with the crypto engine
        self.aggregator = ModelAggregator(method="secure_fedavg", crypto_engine=self.crypto_engine)
        
        # Also create a plain aggregator for testing non-encrypted methods
        self.plain_aggregator = ModelAggregator(method="fedavg")
    
    def test_initialization(self):
        """Test that the aggregator initializes correctly"""
        self.assertEqual(self.aggregator.method, "secure_fedavg")
        self.assertIs(self.aggregator.crypto_engine, self.crypto_engine)
        
        # Test fallback to fedavg when no crypto engine
        no_crypto_aggregator = ModelAggregator(method="secure_fedavg")
        self.assertEqual(no_crypto_aggregator.method, "fedavg")
    
    def test_plaintext_aggregation(self):
        """Test aggregation of plaintext model updates"""
        # Create sample model updates
        updates = [
            ([np.array([1.0, 2.0]), np.array([[3.0, 4.0], [5.0, 6.0]])], 10),
            ([np.array([7.0, 8.0]), np.array([[9.0, 10.0], [11.0, 12.0]])], 5)
        ]
        
        # Expected result (weighted average)
        total_samples = 10 + 5
        weight_1 = 10 / total_samples
        weight_2 = 5 / total_samples
        
        expected = [
            np.array([1.0, 2.0]) * weight_1 + np.array([7.0, 8.0]) * weight_2,
            np.array([[3.0, 4.0], [5.0, 6.0]]) * weight_1 + np.array([[9.0, 10.0], [11.0, 12.0]]) * weight_2
        ]
        
        # Perform aggregation
        loop = asyncio.get_event_loop()
        aggregated = loop.run_until_complete(self.plain_aggregator.aggregate(updates))
        
        # Verify results
        self.assertEqual(len(aggregated), 2)
        np.testing.assert_allclose(aggregated[0], expected[0])
        np.testing.assert_allclose(aggregated[1], expected[1])
    
    @patch('federated_learning.aggregator.ModelAggregator._aggregate_encrypted')
    @patch('federated_learning.aggregator.ModelAggregator._aggregate_plaintext')
    def test_aggregate_method_selection(self, mock_plaintext, mock_encrypted):
        """Test that the correct aggregation method is selected based on parameters"""
        # Mock return values
        mock_plaintext.return_value = asyncio.Future()
        mock_plaintext.return_value.set_result("plaintext result")
        
        mock_encrypted.return_value = asyncio.Future()
        mock_encrypted.return_value.set_result("encrypted result")
        
        # Sample updates
        updates = [([np.array([1.0, 2.0])], 1)]
        
        # Test plaintext aggregation
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.plain_aggregator.aggregate(updates, homomorphic=False))
        mock_plaintext.assert_called_once()
        mock_encrypted.assert_not_called()
        self.assertEqual(result, "plaintext result")
        
        # Reset mocks
        mock_plaintext.reset_mock()
        mock_encrypted.reset_mock()
        
        # Test encrypted aggregation
        result = loop.run_until_complete(self.aggregator.aggregate(updates, homomorphic=True))
        mock_encrypted.assert_called_once()
        mock_plaintext.assert_not_called()
        self.assertEqual(result, "encrypted result")
    
    def test_differential_privacy(self):
        """Test applying differential privacy to aggregated model"""
        # Create sample model
        model = [
            np.random.normal(0, 1, (10,)).astype(np.float32),
            np.random.normal(0, 1, (5, 5)).astype(np.float32)
        ]
        
        # Apply differential privacy
        epsilon = 0.5
        delta = 1e-5
        
        # Create a copy to verify changes
        model_copy = [np.copy(layer) for layer in model]
        
        loop = asyncio.get_event_loop()
        noisy_model = loop.run_until_complete(
            self.plain_aggregator.differential_privacy(model, epsilon, delta)
        )
        
        # Verify shape and dtype preserved
        for orig, noisy in zip(model_copy, noisy_model):
            self.assertEqual(orig.shape, noisy.shape)
            self.assertEqual(orig.dtype, noisy.dtype)
            
            # Values should be different due to noise
            self.assertFalse(np.allclose(orig, noisy))
            
            # But not too different
            self.assertTrue(np.allclose(orig, noisy, atol=1.0))

if __name__ == '__main__':
    unittest.main()