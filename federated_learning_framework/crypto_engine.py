import numpy as np
import tenseal as ts
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
import base64
import json
import pickle
import time
from functools import lru_cache
import torch

from federated_learning_framework.config import CryptoConfig

class CryptoEngine:
    """
    Homomorphic encryption engine for privacy-preserving federated learning.
    Uses TenSEAL for CKKS and BFV homomorphic encryption schemes.
    """
    
    def __init__(self, config: CryptoConfig):
        """
        Initialize the crypto engine.
        
        Args:
            config: Cryptographic configuration
        """
        self.logger = logging.getLogger("crypto_engine")
        self.config = config
        
        # Skip initialization if encryption is disabled
        if not config.enabled:
            self.logger.info("Homomorphic encryption disabled")
            self.context = None
            return
        
        self.logger.info(f"Initializing {config.scheme} homomorphic encryption")
        
        self.scheme_type = ts.SCHEME_TYPE.CKKS if config.scheme == "CKKS" else ts.SCHEME_TYPE.BFV
        self.poly_modulus_degree = config.poly_modulus_degree
        self.security_level = config.security_level
        
        # Initialize context
        try:
            self._create_context()
            self.logger.info(f"Successfully initialized {config.scheme} encryption")
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")
            raise
    
    def _create_context(self):
        """Create the TenSEAL context with appropriate parameters"""
        start_time = time.time()
        
        if self.scheme_type == ts.SCHEME_TYPE.CKKS:
            # Create CKKS context
            self.context = ts.context(
                self.scheme_type,
                poly_modulus_degree=self.poly_modulus_degree,
                coeff_mod_bit_sizes=self.config.coeff_mod_bit_sizes
            )
            
            # Set up context parameters
            self.context.generate_galois_keys()
            self.context.global_scale = self.config.global_scale
        
        else:  # BFV scheme
            # Create BFV context
            self.context = ts.context(
                self.scheme_type,
                poly_modulus_degree=self.poly_modulus_degree,
                plain_modulus=1032193
            )
            
            # Generate evaluation keys
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()
        
        elapsed = time.time() - start_time
        self.logger.info(f"Context creation completed in {elapsed:.2f} seconds")
        
        # Extract keys for potential distribution in production systems
        self.secret_key = self.context.secret_key()
        self.context.make_context_public()
    
    def is_enabled(self) -> bool:
        """Check if encryption is enabled"""
        return self.config.enabled and self.context is not None
    
    def encrypt_vector(self, vector: List[float]) -> ts.CKKSVector:
        """
        Encrypt a vector using homomorphic encryption.
        
        Args:
            vector: List of floats to encrypt
            
        Returns:
            Encrypted vector
        """
        if not self.is_enabled():
            return vector  # Return as-is if encryption disabled
        
        try:
            if self.scheme_type == ts.SCHEME_TYPE.CKKS:
                encrypted = ts.ckks_vector(self.context, vector)
            else:
                # For BFV, we need to convert to integers
                int_vector = [int(round(x * 1000)) for x in vector]
                encrypted = ts.bfv_vector(self.context, int_vector)
            
            return encrypted
            
        except Exception as e:
            self.logger.error(f"Encryption error: {e}")
            raise
    
    def decrypt_vector(self, encrypted_vector) -> List[float]:
        """
        Decrypt an encrypted vector.
        
        Args:
            encrypted_vector: Encrypted vector
            
        Returns:
            Decrypted vector
        """
        if not self.is_enabled() or not isinstance(encrypted_vector, (ts.CKKSVector, ts.BFVVector)):
            return encrypted_vector  # Return as-is if encryption disabled or not encrypted
        
        try:
            # Create a context with the secret key for decryption
            decrypt_context = ts.context_from(self.context.serialize())
            decrypt_context.set_secret_key(self.secret_key)
            
            # Decrypt
            if isinstance(encrypted_vector, ts.CKKSVector):
                # For CKKS, get float values
                decrypted = encrypted_vector.decrypt(decrypt_context)
            else:
                # For BFV, convert back to floats
                decrypted = [x / 1000 for x in encrypted_vector.decrypt(decrypt_context)]
            
            return decrypted
            
        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            raise
    
    def encrypt_matrix(self, matrix: np.ndarray) -> List[Any]:
        """
        Encrypt a matrix by encrypting each row.
        
        Args:
            matrix: NumPy array to encrypt
            
        Returns:
            List of encrypted rows
        """
        if not self.is_enabled():
            return matrix  # Return as-is if encryption disabled
        
        encrypted_rows = []
        for row in matrix:
            encrypted_row = self.encrypt_vector(row.tolist())
            encrypted_rows.append(encrypted_row)
        
        return encrypted_rows
    
    def decrypt_matrix(self, encrypted_rows: List[Any]) -> np.ndarray:
        """
        Decrypt a matrix of encrypted rows.
        
        Args:
            encrypted_rows: List of encrypted rows
            
        Returns:
            Decrypted matrix as NumPy array
        """
        if not self.is_enabled():
            return encrypted_rows  # Return as-is if encryption disabled
        
        decrypted_rows = []
        for encrypted_row in encrypted_rows:
            decrypted_row = self.decrypt_vector(encrypted_row)
            decrypted_rows.append(decrypted_row)
        
        return np.array(decrypted_rows)
    
    def encrypt_model_params(self, model_params: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Encrypt model parameters.
        
        Args:
            model_params: Dictionary of parameter name to NumPy array
            
        Returns:
            Dictionary with encrypted parameters
        """
        if not self.is_enabled():
            return model_params  # Return as-is if encryption disabled
        
        encrypted_params = {}
        
        for name, param in model_params.items():
            # Handle different parameter shapes
            if param.ndim == 1:  # Vector
                encrypted_params[name] = {
                    "type": "vector",
                    "shape": param.shape,
                    "data": self.encrypt_vector(param.tolist())
                }
            elif param.ndim == 2:  # Matrix
                encrypted_params[name] = {
                    "type": "matrix",
                    "shape": param.shape,
                    "data": self.encrypt_matrix(param)
                }
            else:
                # For higher dimensions, flatten first
                flattened = param.reshape(-1).tolist()
                encrypted_params[name] = {
                    "type": "tensor",
                    "shape": param.shape,
                    "data": self.encrypt_vector(flattened)
                }
        
        return encrypted_params
    
    def decrypt_model_params(self, encrypted_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Decrypt model parameters.
        
        Args:
            encrypted_params: Dictionary with encrypted parameters
            
        Returns:
            Dictionary of parameter name to NumPy array
        """
        if not self.is_enabled():
            return encrypted_params  # Return as-is if encryption disabled
        
        decrypted_params = {}
        
        for name, param_data in encrypted_params.items():
            param_type = param_data.get("type", "vector")
            shape = param_data.get("shape")
            
            if param_type == "vector":
                decrypted = np.array(self.decrypt_vector(param_data["data"]))
                decrypted_params[name] = decrypted
            
            elif param_type == "matrix":
                decrypted = self.decrypt_matrix(param_data["data"])
                decrypted_params[name] = decrypted
            
            elif param_type == "tensor":
                # Decrypt flattened tensor
                decrypted_flat = np.array(self.decrypt_vector(param_data["data"]))
                # Reshape to original shape
                decrypted_params[name] = decrypted_flat.reshape(shape)
        
        return decrypted_params
    
    def homomorphic_add(self, a, b):
        """
        Perform homomorphic addition on encrypted data.
        
        Args:
            a: First encrypted object
            b: Second encrypted object
            
        Returns:
            Encrypted result
        """
        if not self.is_enabled():
            # If encryption is disabled, perform regular addition
            if isinstance(a, (list, np.ndarray)) and isinstance(b, (list, np.ndarray)):
                return np.array(a) + np.array(b)
            return a + b
        
        try:
            return a + b
        except Exception as e:
            self.logger.error(f"Homomorphic addition error: {e}")
            raise
    
    def homomorphic_multiply_scalar(self, enc_vector, scalar: float):
        """
        Multiply an encrypted vector by a scalar.
        
        Args:
            enc_vector: Encrypted vector
            scalar: Plain scalar value
            
        Returns:
            Encrypted result
        """
        if not self.is_enabled():
            # If encryption is disabled, perform regular multiplication
            if isinstance(enc_vector, (list, np.ndarray)):
                return np.array(enc_vector) * scalar
            return enc_vector * scalar
        
        try:
            return enc_vector * scalar
        except Exception as e:
            self.logger.error(f"Homomorphic multiplication error: {e}")
            raise
    
    def secure_aggregation(self, encrypted_values: List[Any], weights: List[float] = None) -> Any:
        """
        Securely aggregate encrypted values using weighted average.
        
        Args:
            encrypted_values: List of encrypted values to aggregate
            weights: Weights for weighted average
            
        Returns:
            Aggregated encrypted result
        """
        if not encrypted_values:
            raise ValueError("No values provided for aggregation")
        
        # Use equal weights if not specified
        if weights is None:
            weights = [1.0 / len(encrypted_values)] * len(encrypted_values)
        elif len(weights) != len(encrypted_values):
            raise ValueError("Number of weights must match number of values")
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if abs(total_weight) < 1e-6:
            raise ValueError("Weights sum to zero")
        weights = [w / total_weight for w in weights]
        
        if not self.is_enabled():
            # If encryption is disabled, perform regular weighted sum
            result = None
            for i, (value, weight) in enumerate(zip(encrypted_values, weights)):
                if i == 0:
                    if isinstance(value, (list, np.ndarray)):
                        result = np.array(value) * weight
                    else:
                        result = value * weight
                else:
                    if isinstance(value, (list, np.ndarray)):
                        result += np.array(value) * weight
                    else:
                        result += value * weight
            return result
        
        try:
            # Initialize with first weighted value
            result = self.homomorphic_multiply_scalar(encrypted_values[0], weights[0])
            
            # Add remaining weighted values
            for i in range(1, len(encrypted_values)):
                weighted_value = self.homomorphic_multiply_scalar(encrypted_values[i], weights[i])
                result = self.homomorphic_add(result, weighted_value)
            
            return result
        except Exception as e:
            self.logger.error(f"Secure aggregation error: {e}")
            raise
    
    def serialize_encrypted(self, encrypted_data):
        """
        Serialize encrypted data for transmission.
        
        Args:
            encrypted_data: Data to serialize
            
        Returns:
            Serialized data as base64 string
        """
        if not self.is_enabled() or not isinstance(encrypted_data, (ts.CKKSVector, ts.BFVVector)):
            # Use pickle for non-encrypted data
            return base64.b64encode(pickle.dumps(encrypted_data)).decode('ascii')
        
        try:
            serialized = encrypted_data.serialize()
            return base64.b64encode(serialized).decode('ascii')
        except Exception as e:
            self.logger.error(f"Serialization error: {e}")
            raise
    
    def deserialize_encrypted(self, serialized_data: str):
        """
        Deserialize encrypted data.
        
        Args:
            serialized_data: Serialized data as base64 string
            
        Returns:
            Deserialized encrypted data
        """
        if not self.is_enabled():
            # Use pickle for non-encrypted data
            return pickle.loads(base64.b64decode(serialized_data))
        
        try:
            binary_data = base64.b64decode(serialized_data)
            
            if self.scheme_type == ts.SCHEME_TYPE.CKKS:
                return ts.ckks_vector_from(self.context, binary_data)
            else:
                return ts.bfv_vector_from(self.context, binary_data)
        except Exception as e:
            self.logger.error(f"Deserialization error: {e}")
            raise
    
    def encrypt_torch_params(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Encrypt parameters of a PyTorch model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary of encrypted parameters
        """
        if not self.is_enabled():
            # If encryption disabled, just extract parameters
            return {name: param.cpu().numpy() for name, param in model.named_parameters()}
        
        try:
            encrypted_params = {}
            for name, param in model.named_parameters():
                param_np = param.detach().cpu().numpy()
                
                # Handle different parameter shapes
                if param_np.ndim == 1:  # Vector
                    encrypted_params[name] = {
                        "type": "vector",
                        "shape": param_np.shape,
                        "data": self.encrypt_vector(param_np.tolist())
                    }
                elif param_np.ndim == 2:  # Matrix
                    encrypted_params[name] = {
                        "type": "matrix",
                        "shape": param_np.shape,
                        "data": self.encrypt_matrix(param_np)
                    }
                else:
                    # For higher dimensions, flatten first
                    flattened = param_np.reshape(-1).tolist()
                    encrypted_params[name] = {
                        "type": "tensor",
                        "shape": param_np.shape,
                        "data": self.encrypt_vector(flattened)
                    }
            
            return encrypted_params
        except Exception as e:
            self.logger.error(f"PyTorch parameter encryption error: {e}")
            raise
    
    def decrypt_to_torch_params(self, model: torch.nn.Module, encrypted_params: Dict[str, Any]) -> None:
        """
        Decrypt parameters and load them into a PyTorch model.
        
        Args:
            model: PyTorch model to update
            encrypted_params: Encrypted parameters
        """
        if not self.is_enabled():
            # If encryption disabled, directly load parameters
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in encrypted_params:
                        param.copy_(torch.tensor(encrypted_params[name]))
            return
        
        try:
            decrypted_params = self.decrypt_model_params(encrypted_params)
            
            # Update model parameters
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in decrypted_params:
                        param.copy_(torch.tensor(decrypted_params[name]))
        except Exception as e:
            self.logger.error(f"PyTorch parameter decryption error: {e}")
            raise
    
    def save_context(self, path: str) -> None:
        """
        Save the encryption context to a file.
        
        Args:
            path: Path to save the context
        """
        if not self.is_enabled():
            return
        
        try:
            # Serialize context without secret key for client distribution
            public_context = self.context.serialize(save_secret_key=False)
            with open(path, "wb") as f:
                f.write(public_context)
            
            # Secret key is saved separately
            with open(f"{path}.secret", "wb") as f:
                f.write(self.secret_key)
            
            self.logger.info(f"Saved encryption context to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save context: {e}")
    
    def load_context(self, path: str, load_secret: bool = False) -> None:
        """
        Load an encryption context from a file.
        
        Args:
            path: Path to load the context from
            load_secret: Whether to load the secret key
        """
        if not self.config.enabled:
            return
        
        try:
            with open(path, "rb") as f:
                context_bytes = f.read()
            
            self.context = ts.context_from(context_bytes)
            
            # Load secret key if requested
            if load_secret:
                with open(f"{path}.secret", "rb") as f:
                    self.secret_key = f.read()
                    self.context.set_secret_key(self.secret_key)
            
            self.logger.info(f"Loaded encryption context from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load context: {e}")