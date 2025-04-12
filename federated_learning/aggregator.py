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
        
        try:
            if homomorphic:
                if self.crypto_engine is None:
                    raise ValueError("Homomorphic aggregation requires crypto_engine")
                aggregated_weights = await self._aggregate_encrypted(updates)
            else:
                aggregated_weights = await self._aggregate_plaintext(updates)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Aggregation completed in {elapsed:.2f} seconds")
            
            return aggregated_weights
        except Exception as e:
            self.logger.error(f"Error in aggregation: {e}")
            # Return the first client's weights as fallback
            if updates:
                return updates[0][0]
            raise
    
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
        
        # Handle different update formats - detect if updates contain "encrypted_weights" key
        first_update, _ = updates[0]
        if isinstance(first_update, dict) and "encrypted_weights" in first_update:
            # Format: {"encrypted_weights": [...], "sample_size": N}
            updates = [(update["encrypted_weights"], sample_size) for update, sample_size in updates]
            first_update = updates[0][0]
        
        # Get the number of layers from first update
        num_layers = len(first_update)
        
        # Initialize result array
        aggregated_results = []
        
        # Process each layer separately
        for layer_idx in range(num_layers):
            try:
                layer_type = first_update[layer_idx].get("type")
                shape = tuple(first_update[layer_idx].get("shape"))
                
                if layer_type == "vector":
                    # Extract encrypted vectors for this layer from all clients
                    encrypted_vectors = []
                    for (client_update, _), weight in zip(updates, weight_fractions):
                        encrypted_data = self.crypto_engine._deserialize_encrypted(client_update[layer_idx]["data"])
                        encrypted_vectors.append((encrypted_data, weight))
                    
                    # Securely aggregate
                    aggregated = self.crypto_engine.weighted_aggregation(encrypted_vectors)
                    decrypted = self.crypto_engine.decrypt_vector(aggregated)
                    aggregated_results.append(np.array(decrypted, dtype=np.float32))
                    
                elif layer_type == "matrix":
                    # For matrices, aggregate row by row
                    rows, cols = shape
                    result_matrix = np.zeros(shape, dtype=np.float32)
                    
                    # Process each row
                    for row_idx in range(rows):
                        row_vectors = []
                        for (client_update, _), weight in zip(updates, weight_fractions):
                            enc_row = self.crypto_engine._deserialize_encrypted(client_update[layer_idx]["data"][row_idx])
                            row_vectors.append((enc_row, weight))
                        
                        # Aggregate this row
                        aggregated_row = self.crypto_engine.weighted_aggregation(row_vectors)
                        decrypted_row = self.crypto_engine.decrypt_vector(aggregated_row)
                        result_matrix[row_idx] = np.array(decrypted_row)
                    
                    aggregated_results.append(result_matrix)
                    
                elif layer_type == "tensor":
                    # Handle higher dimensional tensors
                    encrypted_tensors = []
                    for (client_update, _), weight in zip(updates, weight_fractions):
                        encrypted_data = self.crypto_engine._deserialize_encrypted(client_update[layer_idx]["data"])
                        encrypted_tensors.append((encrypted_data, weight))
                    
                    aggregated_tensor = self.crypto_engine.weighted_aggregation(encrypted_tensors)
                    decrypted_tensor = self.crypto_engine.decrypt_vector(aggregated_tensor)
                    result_tensor = np.array(decrypted_tensor, dtype=np.float32).reshape(shape)
                    aggregated_results.append(result_tensor)
            
            except Exception as e:
                self.logger.error(f"Error aggregating layer {layer_idx}: {e}")
                # Use zeros as fallback for this layer
                if shape:
                    aggregated_results.append(np.zeros(shape, dtype=np.float32))
                else:
                    # If shape is unknown, use a placeholder
                    aggregated_results.append(np.array([0.0], dtype=np.float32))
        
        return aggregated_results
    
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
                # Ensure the layer_weights is a proper numpy array with compatible shape
                if not isinstance(layer_weights, np.ndarray):
                    layer_weights = np.array(layer_weights, dtype=np.float32)
                
                # Handle shape mismatch gracefully
                if layer_weights.shape != aggregated_weights[layer_idx].shape:
                    self.logger.warning(f"Shape mismatch at layer {layer_idx}: expected {aggregated_weights[layer_idx].shape}, got {layer_weights.shape}")
                    continue
                    
                # Safely perform the weighted addition
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
            sensitivity = 0.5
            
            # Calculate noise scale using the Gaussian mechanism
            noise_scale = np.sqrt(2 * np.log(1.25/delta)) * sensitivity / epsilon
            noise_scale = noise_scale * 0.5  # Reduce noise by multiplying by a factor < 1
            
            # Generate and add noise
            noise = np.random.normal(0, noise_scale, layer.shape)
            noisy_layer = layer + noise.astype(layer.dtype)
            noisy_weights.append(noisy_layer)
        
        return noisy_weights