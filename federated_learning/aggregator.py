import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional
import time

from crypto.homomorphic_engine import HomomorphicEngine

class ModelAggregator:
    """
    Aggregator for federated learning that securely combines model updates
    from multiple clients using homomorphic encryption.
    """
    
    def __init__(self, method: str = "secure_fedavg", 
                 crypto_engine: Optional[HomomorphicEngine] = None):
        """
        Initialize the model aggregator.
        
        Args:
            method: Aggregation method to use
            crypto_engine: Homomorphic encryption engine
        """
        self.logger = logging.getLogger("federated.aggregator")
        self.method = method
        self.crypto_engine = crypto_engine
        
        # Validate supported methods
        supported_methods = ["fedavg", "fedprox", "secure_fedavg", "secure_fedprox"]
        if method not in supported_methods:
            self.logger.warning(f"Unsupported aggregation method: {method}, falling back to fedavg")
            self.method = "fedavg"
        
        # Verify encryption engine if using secure methods
        if method.startswith("secure_") and crypto_engine is None:
            self.logger.warning(f"Secure aggregation method {method} requires crypto engine, falling back to fedavg")
            self.method = "fedavg"
        
        self.logger.info(f"Model aggregator initialized with method: {self.method}")
    
    async def aggregate(self, updates: List[Tuple[Any, Union[int, float]]], 
                      homomorphic: bool = False) -> List[np.ndarray]:
        """
        Aggregate model updates from multiple clients.
        
        Args:
            updates: List of (weights/encrypted_weights, sample_size) tuples from clients
            homomorphic: Whether the updates are homomorphically encrypted
            
        Returns:
            Aggregated model weights
        """
        if not updates:
            raise ValueError("No updates provided for aggregation")
        
        self.logger.info(f"Aggregating {len(updates)} model updates with method {self.method}")
        start_time = time.time()
        
        if homomorphic:
            if self.crypto_engine is None:
                raise ValueError("Homomorphic aggregation requires crypto_engine")
            aggregated_weights = await self._aggregate_encrypted(updates)
        else:
            aggregated_weights = await self._aggregate_plaintext(updates)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Aggregation completed in {elapsed:.2f} seconds")
        
        return aggregated_weights
    
    async def _aggregate_encrypted(self, updates: List[Tuple[Any, Union[int, float]]]) -> List[np.ndarray]:
        """
        Aggregate encrypted model updates using homomorphic encryption.
        
        Args:
            updates: List of (encrypted_weights, sample_size) tuples
            
        Returns:
            Decrypted aggregated weights
        """
        self.logger.info("Performing encrypted aggregation")
        
        # Calculate total sample size
        total_samples = sum(sample_size for _, sample_size in updates)
        if total_samples == 0:
            raise ValueError("Total sample size is zero")
        
        # Normalize weights to fractions of total
        weight_fractions = [sample_size / total_samples for _, sample_size in updates]
        
        # Get the number of layers/parameters from the first update
        first_update, _ = updates[0]
        num_layers = len(first_update)
        
        # Initialize result containers
        aggregated_encrypted = []
        
        # Process each layer separately
        for layer_idx in range(num_layers):
            layer_updates = []
            
            for (client_update, _), weight in zip(updates, weight_fractions):
                layer_update = client_update[layer_idx]
                layer_type = layer_update["type"]
                
                if layer_type == "vector":
                    # For vectors, perform weighted aggregation directly
                    encrypted_data = self.crypto_engine._deserialize_encrypted(layer_update["data"])
                    weighted_data = self.crypto_engine.homomorphic_multiply_plain(encrypted_data, weight)
                    layer_updates.append((weighted_data, weight))
                    
                elif layer_type == "matrix":
                    # For matrices, handle each row separately
                    encrypted_rows = []
                    for enc_row in layer_update["data"]:
                        encrypted_row = self.crypto_engine._deserialize_encrypted(enc_row)
                        weighted_row = self.crypto_engine.homomorphic_multiply_plain(encrypted_row, weight)
                        encrypted_rows.append(weighted_row)
                    
                    # Combine rows into aggregate structure
                    # This would be more complex in a real implementation
                    layer_updates.append((encrypted_rows, weight))
                    
                elif layer_type == "tensor":
                    # For tensors, similar to vectors but reshape after decryption
                    encrypted_data = self.crypto_engine._deserialize_encrypted(layer_update["data"])
                    weighted_data = self.crypto_engine.homomorphic_multiply_plain(encrypted_data, weight)
                    layer_updates.append((weighted_data, weight))
            
            # Aggregate this layer
            if layer_type == "vector":
                # Use the crypto engine for weighted aggregation
                aggregated_layer = self.crypto_engine.weighted_aggregation(layer_updates)
                # Decrypt the aggregated layer
                decrypted_vector = self.crypto_engine.decrypt_vector(aggregated_layer)
                # Convert to numpy array and save the shape
                shape = first_update[layer_idx]["shape"]
                aggregated_encrypted.append(np.array(decrypted_vector, dtype=np.float32))
                
            elif layer_type == "matrix":
                # Handle aggregation for matrices
                # In a full implementation, this would properly handle the matrix structure
                shape = first_update[layer_idx]["shape"]
                rows, cols = shape
                
                # Initialize result matrix
                result_matrix = np.zeros(shape, dtype=np.float32)
                
                # Process each row separately
                for row_idx in range(rows):
                    row_updates = []
                    for client_rows, weight in layer_updates:
                        row_updates.append((client_rows[row_idx], weight))
                    
                    # Aggregate this row
                    aggregated_row = self.crypto_engine.weighted_aggregation(row_updates)
                    decrypted_row = self.crypto_engine.decrypt_vector(aggregated_row)
                    result_matrix[row_idx] = np.array(decrypted_row, dtype=np.float32)
                
                aggregated_encrypted.append(result_matrix)
                
            elif layer_type == "tensor":
                # Handle higher dimensional tensors
                shape = first_update[layer_idx]["shape"]
                aggregated_layer = self.crypto_engine.weighted_aggregation(layer_updates)
                decrypted_tensor = self.crypto_engine.decrypt_vector(aggregated_layer)
                reshaped_tensor = np.array(decrypted_tensor, dtype=np.float32).reshape(shape)
                aggregated_encrypted.append(reshaped_tensor)
        
        return aggregated_encrypted
    
    async def _aggregate_plaintext(self, updates: List[Tuple[List[np.ndarray], Union[int, float]]]) -> List[np.ndarray]:
        """
        Aggregate plaintext model updates using weighted averaging.
        
        Args:
            updates: List of (weights, sample_size) tuples
            
        Returns:
            Aggregated weights
        """
        self.logger.info("Performing plaintext aggregation")
        
        # Calculate total sample size
        total_samples = sum(sample_size for _, sample_size in updates)
        if total_samples == 0:
            raise ValueError("Total sample size is zero")
        
        # Extract weights and calculate weighted fractions
        weights_list = [weights for weights, _ in updates]
        weight_fractions = [sample_size / total_samples for _, sample_size in updates]
        
        # Verify all weight lists have the same structure
        if not all(len(w) == len(weights_list[0]) for w in weights_list):
            raise ValueError("Inconsistent model structure across clients")
        
        # Initialize aggregated weights with zeros, matching the structure of the first client's weights
        aggregated_weights = [np.zeros_like(layer) for layer in weights_list[0]]
        
        # Perform weighted aggregation
        for client_idx, client_weight in enumerate(weight_fractions):
            client_weights = weights_list[client_idx]
            
            # Add weighted contribution from this client
            for layer_idx, layer_weights in enumerate(client_weights):
                aggregated_weights[layer_idx] += layer_weights * client_weight
        
        return aggregated_weights
    
    async def differential_privacy(self, aggregated_weights: List[np.ndarray], 
                                 epsilon: float, delta: float) -> List[np.ndarray]:
        """
        Apply differential privacy to the aggregated model.
        
        Args:
            aggregated_weights: Aggregated model weights
            epsilon: Privacy parameter (smaller = more privacy)
            delta: Privacy failure probability
            
        Returns:
            Model weights with differential privacy applied
        """
        self.logger.info(f"Applying differential privacy with ε={epsilon}, δ={delta}")
        
        # Apply Gaussian noise calibrated to sensitivity and privacy parameters
        noisy_weights = []
        
        for layer in aggregated_weights:
            # Calculate sensitivity based on clipping (would be more sophisticated in real implementation)
            sensitivity = 1.0
            
            # Calculate noise scale using the Gaussian mechanism
            noise_scale = np.sqrt(2 * np.log(1.25/delta)) * sensitivity / epsilon
            
            # Generate and add noise
            noise = np.random.normal(0, noise_scale, layer.shape)
            noisy_layer = layer + noise.astype(layer.dtype)
            noisy_weights.append(noisy_layer)
        
        return noisy_weights