import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Union
import time
import os
import json
import tenseal as ts  # TenSEAL for homomorphic encryption
from tenseal.tensors.ckksvector import CKKSVector
import base64
from pathlib import Path
from functools import lru_cache

class HomomorphicEngine:
    """
    Engine for homomorphic encryption operations, enabling secure computation
    on encrypted data without decryption.
    """
    
    def __init__(self, key_size: int = 2048, security_level: int = 128):
        """
        Initialize the homomorphic encryption engine.
        
        Args:
            key_size: Size of encryption keys
            security_level: Security level in bits
        """
        self.logger = logging.getLogger("crypto.homomorphic")
        self.key_size = key_size
        self.security_level = security_level
        
        # TenSEAL context parameters
        self.poly_modulus_degree = 8192
        self.context = None
        self.public_key = None
        self.secret_key = None
        self.relin_keys = None
        self.galois_keys = None
        
        # Paths for key storage (in production, use secure key management)
        self.keys_dir = Path("./keys")
        self.keys_dir.mkdir(exist_ok=True)
        
        self.logger.info("Initializing homomorphic encryption engine")
        self._initialize_context()
    
    def _initialize_context(self):
        """Initialize the CKKS encryption context"""
        try:
            # Check if keys already exist
            if self._keys_exist():
                self.logger.info("Loading existing encryption keys")
                self._load_keys()
            else:
                self.logger.info("Generating new encryption keys")
                self._generate_keys()
                
            self.logger.info("Homomorphic encryption engine initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize homomorphic encryption: {e}")
            raise
    
    def _generate_keys(self):
        """Generate new encryption keys"""
        # Create TenSEAL context for CKKS scheme
        self.logger.info("Generating TenSEAL CKKS context")
        start_time = time.time()
        
        # Create TenSEAL context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=[40, 20, 40]
        )
        
        # Set up the context
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40
        
        # Extract keys (in production, use proper key management)
        self.secret_key = self.context.secret_key()
        self.context.make_context_public()
        self.public_key = self.context
        self.galois_keys = self.context.galois_keys()
        self.relin_keys = self.context.relin_keys()
        
        # Save keys
        self._save_keys()
        
        elapsed = time.time() - start_time
        self.logger.info(f"Key generation completed in {elapsed:.2f} seconds")
    
    def _save_keys(self):
        """Save encryption keys (for demo purposes - use secure key mgmt in production)"""
        try:
            # Save context
            context_bytes = self.context.serialize(save_secret_key=False)
            with open(self.keys_dir / "context.bin", "wb") as f:
                f.write(context_bytes)
            
            # Save secret key (in real system, store securely!)
            secret_key_bytes = self.secret_key
            with open(self.keys_dir / "secret_key.bin", "wb") as f:
                f.write(secret_key_bytes)
            
            self.logger.info("Encryption keys saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save encryption keys: {e}")
            raise
    
    def _load_keys(self):
        """Load encryption keys from files"""
        try:
            # Load context (public key)
            with open(self.keys_dir / "context.bin", "rb") as f:
                context_bytes = f.read()
            self.context = ts.context_from(context_bytes)
            self.public_key = self.context
            
            # Load secret key
            with open(self.keys_dir / "secret_key.bin", "rb") as f:
                self.secret_key = f.read()
            
            self.logger.info("Encryption keys loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load encryption keys: {e}")
            raise
    
    def _keys_exist(self) -> bool:
        """Check if encryption keys already exist"""
        return (self.keys_dir / "context.bin").exists() and (self.keys_dir / "secret_key.bin").exists()
    
    def encrypt_vector(self, vector: List[float]) -> CKKSVector:
        """
        Encrypt a vector using CKKS homomorphic encryption.
        
        Args:
            vector: List of floating point values to encrypt
            
        Returns:
            Encrypted vector
        """
        try:
            encrypted = ts.ckks_vector(self.context, vector)
            return encrypted
        except Exception as e:
            self.logger.error(f"Error encrypting vector: {e}")
            raise
    
    def decrypt_vector(self, encrypted_vector: CKKSVector) -> List[float]:
        """
        Decrypt a CKKS encrypted vector.
        
        Args:
            encrypted_vector: Encrypted vector to decrypt
            
        Returns:
            List of decrypted floating point values
        """
        try:
            # Create a temporary context with the secret key
            temp_ctx = ts.context_from(self.context.serialize(save_secret_key=False))
            temp_ctx.make_context_public()
            temp_ctx.set_secret_key(self.secret_key)
            
            # Deserialize the encrypted vector with the temporary context
            decrypted = encrypted_vector.decrypt()
            return decrypted
        except Exception as e:
            self.logger.error(f"Error decrypting vector: {e}")
            raise
    
    def encrypt_model_parameters(self, parameters: List[np.ndarray]) -> List[Any]:
        """
        Encrypt model parameters for secure federated learning.
        
        Args:
            parameters: List of NumPy arrays representing model parameters
            
        Returns:
            List of encrypted parameter objects
        """
        encrypted_params = []
        
        for param in parameters:
            # Handle different parameter types/shapes
            if param.ndim == 1:  # Vector
                encrypted_param = self.encrypt_vector(param.tolist())
                encrypted_params.append({
                    "type": "vector",
                    "shape": param.shape,
                    "data": self._serialize_encrypted(encrypted_param)
                })
            elif param.ndim == 2:  # Matrix
                # For matrices, encrypt each row as a separate vector
                encrypted_rows = []
                for row in param:
                    enc_row = self.encrypt_vector(row.tolist())
                    encrypted_rows.append(self._serialize_encrypted(enc_row))
                
                encrypted_params.append({
                    "type": "matrix",
                    "shape": param.shape,
                    "data": encrypted_rows
                })
            else:
                # Handle higher dimensional tensors by flattening
                flattened = param.flatten().tolist()
                encrypted_flat = self.encrypt_vector(flattened)
                encrypted_params.append({
                    "type": "tensor",
                    "shape": param.shape,
                    "data": self._serialize_encrypted(encrypted_flat)
                })
        
        return encrypted_params
    
    def decrypt_model_parameters(self, encrypted_parameters: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Decrypt model parameters from encrypted format.
        
        Args:
            encrypted_parameters: List of encrypted parameter objects
            
        Returns:
            List of NumPy arrays with decrypted parameters
        """
        decrypted_params = []
        
        for enc_param in encrypted_parameters:
            param_type = enc_param["type"]
            shape = tuple(enc_param["shape"])
            
            if param_type == "vector":
                encrypted_vector = self._deserialize_encrypted(enc_param["data"])
                decrypted_data = self.decrypt_vector(encrypted_vector)
                decrypted_params.append(np.array(decrypted_data, dtype=np.float32))
                
            elif param_type == "matrix":
                decrypted_rows = []
                for enc_row in enc_param["data"]:
                    encrypted_row = self._deserialize_encrypted(enc_row)
                    decrypted_row = self.decrypt_vector(encrypted_row)
                    decrypted_rows.append(decrypted_row)
                decrypted_params.append(np.array(decrypted_rows, dtype=np.float32))
                
            elif param_type == "tensor":
                encrypted_tensor = self._deserialize_encrypted(enc_param["data"])
                decrypted_flat = self.decrypt_vector(encrypted_tensor)
                decrypted_tensor = np.array(decrypted_flat, dtype=np.float32).reshape(shape)
                decrypted_params.append(decrypted_tensor)
        
        return decrypted_params
    
    def homomorphic_add(self, enc_a: Any, enc_b: Any) -> Any:
        """
        Perform homomorphic addition on encrypted vectors.
        
        Args:
            enc_a: First encrypted vector
            enc_b: Second encrypted vector
            
        Returns:
            Encrypted result of addition
        """
        try:
            result = enc_a + enc_b
            return result
        except Exception as e:
            self.logger.error(f"Error in homomorphic addition: {e}")
            raise
    
    def homomorphic_multiply_plain(self, enc_a: Any, scalar: float) -> Any:
        """
        Multiply an encrypted vector by an unencrypted scalar.
        
        Args:
            enc_a: Encrypted vector
            scalar: Unencrypted scalar value
            
        Returns:
            Encrypted result of multiplication
        """
        try:
            result = enc_a * scalar
            return result
        except Exception as e:
            self.logger.error(f"Error in homomorphic plain multiplication: {e}")
            raise
    
    def weighted_aggregation(self, encrypted_vectors: List[Tuple[Any, float]]) -> Any:
        """
        Perform weighted aggregation of encrypted vectors.
        
        Args:
            encrypted_vectors: List of (encrypted_vector, weight) tuples
            
        Returns:
            Encrypted result of weighted aggregation
        """
        try:
            if not encrypted_vectors:
                raise ValueError("No vectors provided for aggregation")
            
            # Start with the first vector multiplied by its weight
            enc_vector, weight = encrypted_vectors[0]
            if isinstance(enc_vector, dict) and "data" in enc_vector:
                # Handle serialized format
                enc_vector = self._deserialize_encrypted(enc_vector["data"])
            
            result = self.homomorphic_multiply_plain(enc_vector, weight)
            
            # Add the remaining weighted vectors
            for enc_vector, weight in encrypted_vectors[1:]:
                if isinstance(enc_vector, dict) and "data" in enc_vector:
                    # Handle serialized format
                    enc_vector = self._deserialize_encrypted(enc_vector["data"])
                
                weighted = self.homomorphic_multiply_plain(enc_vector, weight)
                result = self.homomorphic_add(result, weighted)
            
            return result
        except Exception as e:
            self.logger.error(f"Error in weighted aggregation: {e}")
            raise
    
    def _serialize_encrypted(self, encrypted_data: CKKSVector) -> str:
        """
        Serialize encrypted data to string for transmission.
        
        Args:
            encrypted_data: Encrypted vector to serialize
            
        Returns:
            Base64 encoded serialized data
        """
        serialized = encrypted_data.serialize()
        return base64.b64encode(serialized).decode('ascii')
    
    def _deserialize_encrypted(self, serialized_data: str) -> CKKSVector:
        """
        Deserialize encrypted data from string.
        
        Args:
            serialized_data: Base64 encoded serialized data
            
        Returns:
            Deserialized encrypted vector
        """
        binary_data = base64.b64decode(serialized_data)
        return ts.ckks_vector_from(self.context, binary_data)
    
    @lru_cache(maxsize=128)
    def generate_random_mask(self, size: int) -> List[float]:
        """
        Generate a random mask for differential privacy.
        
        Args:
            size: Size of the mask to generate
            
        Returns:
            List of random values forming a mask
        """
        # Generate random noise from Laplacian distribution for differential privacy
        scale = 1.0 / self.security_level  # Scale based on security parameter
        mask = np.random.laplace(0, scale, size).tolist()
        return mask
    
    def add_differential_privacy(self, vector: List[float], epsilon: float) -> List[float]:
        """
        Add differentially private noise to a vector.
        
        Args:
            vector: Vector to add noise to
            epsilon: Privacy parameter (lower = more private, more noise)
            
        Returns:
            Vector with noise added
        """
        scale = 1.0 / epsilon
        noise = np.random.laplace(0, scale, len(vector))
        return [v + n for v, n in zip(vector, noise)]