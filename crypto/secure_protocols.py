import logging
import base64
import json
import os
import hashlib
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import numpy as np

from crypto.key_management import KeyManager

class SecureProtocols:
    """
    Implements secure communication protocols for the federated learning system,
    including secure aggregation, secure parameter exchange, and encrypted gradients.
    """
    
    def __init__(self, key_manager: KeyManager = None):
        """
        Initialize secure protocols manager.
        
        Args:
            key_manager: Key manager for cryptographic operations
        """
        self.logger = logging.getLogger("crypto.secure_protocols")
        self.key_manager = key_manager or KeyManager()
        
        self.logger.info("Secure protocols initialized")
    
    def secure_aggregate(self, encrypted_values: List[bytes], weights: List[float] = None) -> bytes:
        """
        Perform secure aggregation on encrypted values.
        
        Args:
            encrypted_values: List of encrypted values to aggregate
            weights: Optional weights for weighted average
            
        Returns:
            Encrypted aggregated result
        """
        if not encrypted_values:
            raise ValueError("No values provided for aggregation")
        
        # Use uniform weights if not specified
        if weights is None:
            weights = [1.0 / len(encrypted_values)] * len(encrypted_values)
        elif len(weights) != len(encrypted_values):
            raise ValueError("Number of weights must match number of values")
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Weights sum to zero")
        weights = [w / total_weight for w in weights]
        
        # Decrypt values
        decrypted_values = []
        for enc_value in encrypted_values:
            try:
                # Deserialize the encrypted value
                data = json.loads(self.key_manager.symmetric_decrypt(enc_value).decode('utf-8'))
                array_data = base64.b64decode(data['data'])
                shape = tuple(data['shape'])
                dtype = data.get('dtype', 'float32')
                
                # Reconstruct the numpy array
                value = np.frombuffer(array_data, dtype=dtype).reshape(shape)
                decrypted_values.append(value)
            except Exception as e:
                self.logger.error(f"Error decrypting value for aggregation: {e}")
                raise
        
        # Perform weighted average
        result = np.zeros_like(decrypted_values[0])
        for i, value in enumerate(decrypted_values):
            result += value * weights[i]
        
        # Encrypt the result
        return self.encrypt_numpy_array(result)
    
    def encrypt_numpy_array(self, array: np.ndarray) -> bytes:
        """
        Encrypt a numpy array for secure transmission.
        
        Args:
            array: Numpy array to encrypt
            
        Returns:
            Encrypted data
        """
        # Convert array to bytes
        array_bytes = array.tobytes()
        
        # Prepare metadata
        data = {
            'data': base64.b64encode(array_bytes).decode('utf-8'),
            'shape': array.shape,
            'dtype': str(array.dtype)
        }
        
        # Encrypt the serialized data
        return self.key_manager.symmetric_encrypt(json.dumps(data).encode('utf-8'))
    
    def decrypt_numpy_array(self, encrypted_data: bytes) -> np.ndarray:
        """
        Decrypt a numpy array from encrypted data.
        
        Args:
            encrypted_data: Encrypted numpy array
            
        Returns:
            Decrypted numpy array
        """
        # Decrypt the data
        decrypted_data = self.key_manager.symmetric_decrypt(encrypted_data)
        
        # Parse the JSON
        data = json.loads(decrypted_data.decode('utf-8'))
        
        # Reconstruct numpy array
        array_bytes = base64.b64decode(data['data'])
        shape = tuple(data['shape'])
        dtype = data.get('dtype', 'float32')
        
        return np.frombuffer(array_bytes, dtype=dtype).reshape(shape)
    
    def secure_parameter_exchange(self, parameters: Dict[str, Any], recipient_public_key: bytes) -> bytes:
        """
        Securely exchange model parameters with a recipient.
        
        Args:
            parameters: Dictionary of parameters to exchange
            recipient_public_key: Recipient's public key for encryption
            
        Returns:
            Encrypted parameter package
        """
        # Generate a random session key
        session_key = os.urandom(32)
        
        # Create a Fernet key from the session key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=os.urandom(16),
            iterations=100000
        )
        fernet_key = base64.urlsafe_b64encode(kdf.derive(session_key))
        fernet = Fernet(fernet_key)
        
        # Serialize and encrypt parameters
        serialized = json.dumps(parameters).encode('utf-8')
        encrypted_params = fernet.encrypt(serialized)
        
        # Encrypt session key with recipient's public key
        # (In production, deserialize the public key properly)
        recipient_key = serialization.load_pem_public_key(recipient_public_key)
        encrypted_key = recipient_key.encrypt(
            session_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Create the complete package
        package = {
            'encrypted_key': base64.b64encode(encrypted_key).decode('utf-8'),
            'encrypted_params': base64.b64encode(encrypted_params).decode('utf-8'),
            'id': str(uuid.uuid4()),
            'timestamp': int(time.time())
        }
        
        return json.dumps(package).encode('utf-8')
    
    def verify_client_identity(self, client_id: str, signature: bytes, message: bytes) -> bool:
        """
        Verify a client's identity using digital signature.
        
        Args:
            client_id: ID of the client
            signature: Client's signature
            message: Original message that was signed
            
        Returns:
            True if identity is verified
        """
        try:
            # In a real system, we'd have a registry of client public keys
            # For this example, assume we have a method to get the client's public key
            client_public_key = self._get_client_public_key(client_id)
            if not client_public_key:
                return False
            
            return self.key_manager.ec_verify(message, signature, client_public_key)
        except Exception as e:
            self.logger.error(f"Error verifying client identity: {e}")
            return False
    
    def _get_client_public_key(self, client_id: str) -> Optional[str]:
        """Get a client's public key from storage"""
        # This would be implemented to retrieve client keys from a secure store
        # For now, return None to indicate we don't have this functionality yet
        return None
    
    def create_secure_channel(self, peer_id: str, peer_public_key: bytes) -> Dict[str, Any]:
        """
        Create a secure communication channel with a peer.
        
        Args:
            peer_id: Identifier for the peer
            peer_public_key: Peer's public key
            
        Returns:
            Channel information including shared secret
        """
        # Generate an ephemeral EC key pair
        key_id = self.key_manager.generate_ec_key(label=f"channel-{peer_id}")
        _, _, public_key = self.key_manager.get_active_ec_key()
        
        # Serialize our public key
        serialized_public_key = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Generate a unique channel ID
        channel_id = f"channel-{uuid.uuid4()}"
        
        # In a real implementation, we would perform a proper key exchange
        # For this example, just derive a simple shared secret
        channel_secret = hashlib.sha256(
            serialized_public_key + peer_public_key + channel_id.encode('utf-8')
        ).digest()
        
        return {
            'channel_id': channel_id,
            'public_key': serialized_public_key,
            'secret': channel_secret,
            'created_at': int(time.time()),
            'peer_id': peer_id
        }
    
    def encrypt_gradients(self, gradients: List[np.ndarray]) -> List[bytes]:
        """
        Encrypt model gradients for secure transmission.
        
        Args:
            gradients: List of gradient arrays
            
        Returns:
            List of encrypted gradient data
        """
        encrypted_grads = []
        
        for grad in gradients:
            # Encrypt each gradient array
            encrypted = self.encrypt_numpy_array(grad)
            encrypted_grads.append(encrypted)
        
        return encrypted_grads
    
    def decrypt_gradients(self, encrypted_grads: List[bytes]) -> List[np.ndarray]:
        """
        Decrypt model gradients.
        
        Args:
            encrypted_grads: List of encrypted gradient data
            
        Returns:
            List of decrypted gradient arrays
        """
        decrypted_grads = []
        
        for enc_grad in encrypted_grads:
            # Decrypt each gradient array
            decrypted = self.decrypt_numpy_array(enc_grad)
            decrypted_grads.append(decrypted)
        
        return decrypted_grads
    
    def create_secure_token(self, data: Dict[str, Any], expiration_seconds: int = 3600) -> str:
        """
        Create a secure authentication token.
        
        Args:
            data: Data to include in the token
            expiration_seconds: Token validity period in seconds
            
        Returns:
            Secure token string
        """
        # Add expiration and issued time
        payload = {
            **data,
            'exp': int(time.time()) + expiration_seconds,
            'iat': int(time.time()),
            'jti': str(uuid.uuid4())
        }
        
        # Serialize and encrypt
        serialized = json.dumps(payload).encode('utf-8')
        
        # Sign the payload
        signature = self.key_manager.ec_sign(serialized)
        
        # Create the complete token
        token_data = {
            'payload': base64.b64encode(serialized).decode('utf-8'),
            'signature': base64.b64encode(signature).decode('utf-8')
        }
        
        # Encode the token
        return base64.urlsafe_b64encode(json.dumps(token_data).encode('utf-8')).decode('utf-8')
    
    def verify_secure_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify and decode a secure token.
        
        Args:
            token: Token to verify
            
        Returns:
            Tuple of (is_valid, token_data)
        """
        try:
            # Decode the token
            token_bytes = base64.urlsafe_b64decode(token)
            token_data = json.loads(token_bytes.decode('utf-8'))
            
            # Extract payload and signature
            payload_bytes = base64.b64decode(token_data['payload'])
            signature = base64.b64decode(token_data['signature'])
            
            # Verify signature
            _, _, public_key = self.key_manager.get_active_ec_key()
            valid = self.key_manager.ec_verify(payload_bytes, signature)
            
            if not valid:
                return False, None
            
            # Decode payload
            payload = json.loads(payload_bytes.decode('utf-8'))
            
            # Check expiration
            if int(time.time()) > payload.get('exp', 0):
                return False, None
            
            return True, payload
            
        except Exception as e:
            self.logger.error(f"Error verifying token: {e}")
            return False, None