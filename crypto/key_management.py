import os
import json
import logging
import base64
import time
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateKey
from cryptography.hazmat.primitives import serialization, hashes, hmac
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
from datetime import datetime, timedelta
import threading
import secrets

class KeyManager:
    """
    Secure key management system for the orchestrator.
    Handles key generation, rotation, storage, and secure distribution.
    """
    
    def __init__(self, keys_dir: str = "keys", 
                 rotation_days: int = 30,
                 key_size: int = 2048,
                 master_key_env: str = "ORCHESTRATOR_MASTER_KEY"):
        """
        Initialize the key manager.
        
        Args:
            keys_dir: Directory for key storage
            rotation_days: Days between key rotations
            key_size: Size for RSA keys
            master_key_env: Environment variable for master key
        """
        self.logger = logging.getLogger("crypto.key_manager")
        self.keys_dir = Path(keys_dir)
        self.rotation_days = rotation_days
        self.key_size = key_size
        self.master_key_env = master_key_env
        
        # Create keys directory if it doesn't exist
        self.keys_dir.mkdir(parents=True, exist_ok=True)
        
        # Key storage
        self._rsa_keys = {}  # {key_id: (private_key, public_key)}
        self._ec_keys = {}   # {key_id: (private_key, public_key)}
        self._sym_keys = {}  # {key_id: key}
        self._key_metadata = {}  # {key_id: metadata}
        
        # Key access lock
        self._key_lock = threading.RLock()
        
        # Master key for encrypting stored keys
        self._master_key = self._get_master_key()
        
        # Load or generate keys
        self._load_keys()
        
        # Key rotation task
        self._rotation_task = None
        
        self.logger.info("Key manager initialized")
    
    def _get_master_key(self) -> bytes:
        """Get or generate the master key for key encryption"""
        # Try to get from environment
        master_key = os.environ.get(self.master_key_env)
        
        if master_key:
            try:
                # Decode the base64 key
                return base64.urlsafe_b64decode(master_key)
            except:
                self.logger.warning("Invalid master key format in environment")
        
        # Check if master key exists in a file
        master_key_file = self.keys_dir / "master.key"
        if master_key_file.exists():
            try:
                with open(master_key_file, "rb") as f:
                    return base64.urlsafe_b64decode(f.read())
            except:
                self.logger.warning("Could not load master key from file")
        
        # Generate a new master key
        self.logger.warning("Generating new master key. Please secure this key for production use.")
        new_key = Fernet.generate_key()
        
        # Save to file (in production, use secure storage)
        with open(master_key_file, "wb") as f:
            f.write(base64.urlsafe_b64encode(new_key))
        
        return new_key
    
    def _load_keys(self):
        """Load keys from storage"""
        with self._key_lock:
            # Check for metadata file
            metadata_file = self.keys_dir / "key_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        self._key_metadata = json.load(f)
                except:
                    self.logger.error("Failed to load key metadata", exc_info=True)
                    self._key_metadata = {}
            
            # Load keys based on metadata
            for key_id, metadata in self._key_metadata.items():
                try:
                    key_type = metadata.get("type")
                    key_file = self.keys_dir / f"{key_id}.key"
                    
                    if not key_file.exists():
                        continue
                        
                    if key_type == "rsa":
                        self._load_rsa_key(key_id, key_file)
                    elif key_type == "ec":
                        self._load_ec_key(key_id, key_file)
                    elif key_type == "symmetric":
                        self._load_symmetric_key(key_id, key_file)
                except:
                    self.logger.error(f"Failed to load key {key_id}", exc_info=True)
            
            # If no keys exist, generate initial keys
            if not self._key_metadata:
                self._generate_initial_keys()
    
    def _load_rsa_key(self, key_id: str, key_file: Path):
        """Load an RSA key pair from storage"""
        try:
            # Load the encrypted key file
            with open(key_file, "rb") as f:
                encrypted_data = f.read()
            
            # Decrypt the key data
            fernet = Fernet(self._master_key)
            decrypted_data = fernet.decrypt(encrypted_data)
            key_data = json.loads(decrypted_data.decode('utf-8'))
            
            # Load private key
            private_key = serialization.load_pem_private_key(
                base64.b64decode(key_data["private_key"]),
                password=None,
                backend=default_backend()
            )
            
            # Extract public key
            public_key = private_key.public_key()
            
            # Store in memory
            self._rsa_keys[key_id] = (private_key, public_key)
            self.logger.debug(f"Loaded RSA key: {key_id}")
        except Exception as e:
            self.logger.error(f"Failed to load RSA key {key_id}: {e}")
            raise
    
    def _load_ec_key(self, key_id: str, key_file: Path):
        """Load an Elliptic Curve key pair from storage"""
        try:
            # Load the encrypted key file
            with open(key_file, "rb") as f:
                encrypted_data = f.read()
            
            # Decrypt the key data
            fernet = Fernet(self._master_key)
            decrypted_data = fernet.decrypt(encrypted_data)
            key_data = json.loads(decrypted_data.decode('utf-8'))
            
            # Load private key
            private_key = serialization.load_pem_private_key(
                base64.b64decode(key_data["private_key"]),
                password=None,
                backend=default_backend()
            )
            
            # Extract public key
            public_key = private_key.public_key()
            
            # Store in memory
            self._ec_keys[key_id] = (private_key, public_key)
            self.logger.debug(f"Loaded EC key: {key_id}")
        except Exception as e:
            self.logger.error(f"Failed to load EC key {key_id}: {e}")
            raise
    
    def _load_symmetric_key(self, key_id: str, key_file: Path):
        """Load a symmetric key from storage"""
        try:
            # Load the encrypted key file
            with open(key_file, "rb") as f:
                encrypted_data = f.read()
            
            # Decrypt the key data
            fernet = Fernet(self._master_key)
            decrypted_data = fernet.decrypt(encrypted_data)
            key_data = json.loads(decrypted_data.decode('utf-8'))
            
            # Decode the symmetric key
            symmetric_key = base64.b64decode(key_data["key"])
            
            # Store in memory
            self._sym_keys[key_id] = symmetric_key
            self.logger.debug(f"Loaded symmetric key: {key_id}")
        except Exception as e:
            self.logger.error(f"Failed to load symmetric key {key_id}: {e}")
            raise
    
    def _generate_initial_keys(self):
        """Generate initial set of cryptographic keys"""
        self.logger.info("Generating initial cryptographic keys")
        
        # Generate RSA key pair
        rsa_key_id = self.generate_rsa_key()
        
        # Generate EC key pair
        ec_key_id = self.generate_ec_key()
        
        # Generate symmetric key
        sym_key_id = self.generate_symmetric_key()
        
        # Set as current active keys
        with self._key_lock:
            self._key_metadata["active_rsa"] = rsa_key_id
            self._key_metadata["active_ec"] = ec_key_id
            self._key_metadata["active_symmetric"] = sym_key_id
            
            self._save_metadata()
    
    def generate_rsa_key(self, label: str = "default") -> str:
        """
        Generate a new RSA key pair.
        
        Args:
            label: Label for the key
            
        Returns:
            Key ID for the generated key
        """
        with self._key_lock:
            # Generate key ID
            key_id = f"rsa-{uuid.uuid4()}"
            
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.key_size,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            
            # Store in memory
            self._rsa_keys[key_id] = (private_key, public_key)
            
            # Update metadata
            self._key_metadata[key_id] = {
                "type": "rsa",
                "label": label,
                "created": datetime.utcnow().isoformat(),
                "expires": (datetime.utcnow() + timedelta(days=self.rotation_days)).isoformat(),
                "key_size": self.key_size
            }
            
            # Save to disk
            self._save_rsa_key(key_id, private_key)
            self._save_metadata()
            
            self.logger.info(f"Generated new RSA key: {key_id}")
            return key_id
    
    def generate_ec_key(self, curve: str = "secp256r1", label: str = "default") -> str:
        """
        Generate a new Elliptic Curve key pair.
        
        Args:
            curve: EC curve to use
            label: Label for the key
            
        Returns:
            Key ID for the generated key
        """
        with self._key_lock:
            # Generate key ID
            key_id = f"ec-{uuid.uuid4()}"
            
            # Map curve name to curve class
            curve_map = {
                "secp256r1": ec.SECP256R1,
                "secp384r1": ec.SECP384R1,
                "secp521r1": ec.SECP521R1
            }
            curve_class = curve_map.get(curve.lower(), ec.SECP256R1)
            
            # Generate EC key pair
            private_key = ec.generate_private_key(
                curve=curve_class(),
                backend=default_backend()
            )
            public_key = private_key.public_key()
            
            # Store in memory
            self._ec_keys[key_id] = (private_key, public_key)
            
            # Update metadata
            self._key_metadata[key_id] = {
                "type": "ec",
                "label": label,
                "created": datetime.utcnow().isoformat(),
                "expires": (datetime.utcnow() + timedelta(days=self.rotation_days)).isoformat(),
                "curve": curve
            }
            
            # Save to disk
            self._save_ec_key(key_id, private_key)
            self._save_metadata()
            
            self.logger.info(f"Generated new EC key: {key_id}")
            return key_id
    
    def generate_symmetric_key(self, key_size: int = 32, label: str = "default") -> str:
        """
        Generate a new symmetric key.
        
        Args:
            key_size: Size of the key in bytes
            label: Label for the key
            
        Returns:
            Key ID for the generated key
        """
        with self._key_lock:
            # Generate key ID
            key_id = f"sym-{uuid.uuid4()}"
            
            # Generate secure random key
            key = secrets.token_bytes(key_size)
            
            # Store in memory
            self._sym_keys[key_id] = key
            
            # Update metadata
            self._key_metadata[key_id] = {
                "type": "symmetric",
                "label": label,
                "created": datetime.utcnow().isoformat(),
                "expires": (datetime.utcnow() + timedelta(days=self.rotation_days)).isoformat(),
                "key_size": key_size
            }
            
            # Save to disk
            self._save_symmetric_key(key_id, key)
            self._save_metadata()
            
            self.logger.info(f"Generated new symmetric key: {key_id}")
            return key_id
    
    def _save_rsa_key(self, key_id: str, private_key):
        """Save an RSA key to disk"""
        # Serialize the private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Prepare key data
        key_data = {
            "private_key": base64.b64encode(private_pem).decode('utf-8')
        }
        
        # Encrypt with master key
        fernet = Fernet(self._master_key)
        encrypted_data = fernet.encrypt(json.dumps(key_data).encode('utf-8'))
        
        # Save to file
        key_file = self.keys_dir / f"{key_id}.key"
        with open(key_file, "wb") as f:
            f.write(encrypted_data)
    
    def _save_ec_key(self, key_id: str, private_key):
        """Save an EC key to disk"""
        # Serialize the private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Prepare key data
        key_data = {
            "private_key": base64.b64encode(private_pem).decode('utf-8')
        }
        
        # Encrypt with master key
        fernet = Fernet(self._master_key)
        encrypted_data = fernet.encrypt(json.dumps(key_data).encode('utf-8'))
        
        # Save to file
        key_file = self.keys_dir / f"{key_id}.key"
        with open(key_file, "wb") as f:
            f.write(encrypted_data)
    
    def _save_symmetric_key(self, key_id: str, key: bytes):
        """Save a symmetric key to disk"""
        # Prepare key data
        key_data = {
            "key": base64.b64encode(key).decode('utf-8')
        }
        
        # Encrypt with master key
        fernet = Fernet(self._master_key)
        encrypted_data = fernet.encrypt(json.dumps(key_data).encode('utf-8'))
        
        # Save to file
        key_file = self.keys_dir / f"{key_id}.key"
        with open(key_file, "wb") as f:
            f.write(encrypted_data)
    
    def _save_metadata(self):
        """Save key metadata to disk"""
        metadata_file = self.keys_dir / "key_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self._key_metadata, f, indent=2)
    
    def get_active_rsa_key(self) -> Tuple[str, Any, Any]:
        """
        Get the active RSA key pair.
        
        Returns:
            Tuple of (key_id, private_key, public_key)
        """
        with self._key_lock:
            active_id = self._key_metadata.get("active_rsa")
            if not active_id or active_id not in self._rsa_keys:
                # No active key, generate a new one
                active_id = self.generate_rsa_key()
                self._key_metadata["active_rsa"] = active_id
                self._save_metadata()
            
            private_key, public_key = self._rsa_keys[active_id]
            return active_id, private_key, public_key
    
    def get_active_ec_key(self) -> Tuple[str, Any, Any]:
        """
        Get the active EC key pair.
        
        Returns:
            Tuple of (key_id, private_key, public_key)
        """
        with self._key_lock:
            active_id = self._key_metadata.get("active_ec")
            if not active_id or active_id not in self._ec_keys:
                # No active key, generate a new one
                active_id = self.generate_ec_key()
                self._key_metadata["active_ec"] = active_id
                self._save_metadata()
            
            private_key, public_key = self._ec_keys[active_id]
            return active_id, private_key, public_key
    
    def get_active_symmetric_key(self) -> Tuple[str, bytes]:
        """
        Get the active symmetric key.
        
        Returns:
            Tuple of (key_id, key)
        """
        with self._key_lock:
            active_id = self._key_metadata.get("active_symmetric")
            if not active_id or active_id not in self._sym_keys:
                # No active key, generate a new one
                active_id = self.generate_symmetric_key()
                self._key_metadata["active_symmetric"] = active_id
                self._save_metadata()
            
            key = self._sym_keys[active_id]
            return active_id, key
    
    def get_key_by_id(self, key_id: str) -> Optional[Tuple[str, Any]]:
        """
        Get a key by its ID.
        
        Args:
            key_id: ID of the key to retrieve
            
        Returns:
            Tuple of (key_type, key) or None if not found
        """
        with self._key_lock:
            if key_id in self._rsa_keys:
                return "rsa", self._rsa_keys[key_id]
            elif key_id in self._ec_keys:
                return "ec", self._ec_keys[key_id]
            elif key_id in self._sym_keys:
                return "symmetric", self._sym_keys[key_id]
            else:
                return None
    
    def rsa_encrypt(self, data: bytes, key_id: str = None) -> bytes:
        """
        Encrypt data with RSA public key.
        
        Args:
            data: Data to encrypt
            key_id: Key ID to use (uses active key if None)
            
        Returns:
            Encrypted data
        """
        # Get the appropriate public key
        if key_id:
            with self._key_lock:
                if key_id not in self._rsa_keys:
                    raise ValueError(f"RSA key not found: {key_id}")
                _, public_key = self._rsa_keys[key_id]
        else:
            _, _, public_key = self.get_active_rsa_key()
        
        # Encrypt data
        return public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def rsa_decrypt(self, encrypted_data: bytes, key_id: str = None) -> bytes:
        """
        Decrypt data with RSA private key.
        
        Args:
            encrypted_data: Data to decrypt
            key_id: Key ID to use (uses active key if None)
            
        Returns:
            Decrypted data
        """
        # Get the appropriate private key
        if key_id:
            with self._key_lock:
                if key_id not in self._rsa_keys:
                    raise ValueError(f"RSA key not found: {key_id}")
                private_key, _ = self._rsa_keys[key_id]
        else:
            _, private_key, _ = self.get_active_rsa_key()
        
        # Decrypt data
        return private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    
    def ec_sign(self, data: bytes, key_id: str = None) -> bytes:
        """
        Sign data with EC private key.
        
        Args:
            data: Data to sign
            key_id: Key ID to use (uses active key if None)
            
        Returns:
            Signature bytes
        """
        # Get the appropriate private key
        if key_id:
            with self._key_lock:
                if key_id not in self._ec_keys:
                    raise ValueError(f"EC key not found: {key_id}")
                private_key, _ = self._ec_keys[key_id]
        else:
            _, private_key, _ = self.get_active_ec_key()
        
        # Sign the data
        return private_key.sign(
            data,
            ec.ECDSA(hashes.SHA256())
        )
    
    def ec_verify(self, data: bytes, signature: bytes, key_id: str = None) -> bool:
        """
        Verify signature with EC public key.
        
        Args:
            data: Original data
            signature: Signature to verify
            key_id: Key ID to use (uses active key if None)
            
        Returns:
            True if signature is valid
        """
        # Get the appropriate public key
        if key_id:
            with self._key_lock:
                if key_id not in self._ec_keys:
                    raise ValueError(f"EC key not found: {key_id}")
                _, public_key = self._ec_keys[key_id]
        else:
            _, _, public_key = self.get_active_ec_key()
        
        # Verify the signature
        try:
            public_key.verify(
                signature,
                data,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except:
            return False
    
    def symmetric_encrypt(self, data: bytes, key_id: str = None) -> bytes:
        """
        Encrypt data with symmetric key.
        
        Args:
            data: Data to encrypt
            key_id: Key ID to use (uses active key if None)
            
        Returns:
            Encrypted data
        """
        # Get the appropriate key
        if key_id:
            with self._key_lock:
                if key_id not in self._sym_keys:
                    raise ValueError(f"Symmetric key not found: {key_id}")
                key = self._sym_keys[key_id]
        else:
            _, key = self.get_active_symmetric_key()
        
        # Create Fernet with the key
        fernet_key = base64.urlsafe_b64encode(key[:32].ljust(32, b'\0'))
        fernet = Fernet(fernet_key)
        
        # Encrypt data
        return fernet.encrypt(data)
    
    def symmetric_decrypt(self, encrypted_data: bytes, key_id: str = None) -> bytes:
        """
        Decrypt data with symmetric key.
        
        Args:
            encrypted_data: Data to decrypt
            key_id: Key ID to use (uses active key if None)
            
        Returns:
            Decrypted data
        """
        # Get the appropriate key
        if key_id:
            with self._key_lock:
                if key_id not in self._sym_keys:
                    raise ValueError(f"Symmetric key not found: {key_id}")
                key = self._sym_keys[key_id]
        else:
            _, key = self.get_active_symmetric_key()
        
        # Create Fernet with the key
        fernet_key = base64.urlsafe_b64encode(key[:32].ljust(32, b'\0'))
        fernet = Fernet(fernet_key)
        
        # Decrypt data
        return fernet.decrypt(encrypted_data)
    
    def create_hmac(self, data: bytes, key_id: str = None) -> bytes:
        """
        Create HMAC for data using a symmetric key.
        
        Args:
            data: Data to authenticate
            key_id: Key ID to use (uses active key if None)
            
        Returns:
            HMAC tag
        """
        # Get the appropriate key
        if key_id:
            with self._key_lock:
                if key_id not in self._sym_keys:
                    raise ValueError(f"Symmetric key not found: {key_id}")
                key = self._sym_keys[key_id]
        else:
            _, key = self.get_active_symmetric_key()
        
        # Create HMAC
        h = hmac.HMAC(key, hashes.SHA256(), backend=default_backend())
        h.update(data)
        return h.finalize()
    
    def verify_hmac(self, data: bytes, tag: bytes, key_id: str = None) -> bool:
        """
        Verify HMAC for data.
        
        Args:
            data: Original data
            tag: HMAC tag to verify
            key_id: Key ID to use (uses active key if None)
            
        Returns:
            True if HMAC is valid
        """
        # Get the appropriate key
        if key_id:
            with self._key_lock:
                if key_id not in self._sym_keys:
                    raise ValueError(f"Symmetric key not found: {key_id}")
                key = self._sym_keys[key_id]
        else:
            _, key = self.get_active_symmetric_key()
        
        # Verify HMAC
        h = hmac.HMAC(key, hashes.SHA256(), backend=default_backend())
        h.update(data)
        try:
            h.verify(tag)
            return True
        except:
            return False
    
    async def start_rotation_task(self):
        """Start key rotation background task"""
        if self._rotation_task:
            return
        
        self._rotation_task = asyncio.create_task(self._rotation_loop())
        self.logger.info("Started key rotation task")
    
    async def stop_rotation_task(self):
        """Stop key rotation background task"""
        if not self._rotation_task:
            return
        
        self._rotation_task.cancel()
        try:
            await self._rotation_task
        except asyncio.CancelledError:
            pass
        
        self._rotation_task = None
        self.logger.info("Stopped key rotation task")
    
    async def _rotation_loop(self):
        """Background task for automatic key rotation"""
        while True:
            try:
                await self._check_and_rotate_keys()
                
                # Sleep until next check (daily)
                await asyncio.sleep(86400)  # 24 hours
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in key rotation task: {e}", exc_info=True)
                await asyncio.sleep(3600)  # Retry after 1 hour on error
    
    async def _check_and_rotate_keys(self):
        """Check for expired keys and rotate them"""
        with self._key_lock:
            now = datetime.utcnow()
            rotated_keys = []
            
            # Check each key for expiration
            for key_id, metadata in self._key_metadata.items():
                if key_id in ["active_rsa", "active_ec", "active_symmetric"]:
                    continue
                
                try:
                    expires_str = metadata.get("expires")
                    if not expires_str:
                        continue
                    
                    expires = datetime.fromisoformat(expires_str)
                    
                    # If key is expired or will expire soon (within 3 days)
                    if expires < now or (expires - now).days <= 3:
                        key_type = metadata.get("type")
                        label = metadata.get("label", "default")
                        
                        # Rotate the key
                        if key_type == "rsa":
                            new_key_id = self.generate_rsa_key(label=label)
                            if self._key_metadata.get("active_rsa") == key_id:
                                self._key_metadata["active_rsa"] = new_key_id
                        elif key_type == "ec":
                            curve = metadata.get("curve", "secp256r1")
                            new_key_id = self.generate_ec_key(curve=curve, label=label)
                            if self._key_metadata.get("active_ec") == key_id:
                                self._key_metadata["active_ec"] = new_key_id
                        elif key_type == "symmetric":
                            key_size = metadata.get("key_size", 32)
                            new_key_id = self.generate_symmetric_key(key_size=key_size, label=label)
                            if self._key_metadata.get("active_symmetric") == key_id:
                                self._key_metadata["active_symmetric"] = new_key_id
                        
                        rotated_keys.append(key_id)
                        
                        # Keep the old key for some time (1 week) before removing
                        metadata["to_remove"] = (now + timedelta(days=7)).isoformat()
                except Exception as e:
                    self.logger.error(f"Error rotating key {key_id}: {e}", exc_info=True)
            
            # Remove keys that were marked for removal and the time has passed
            to_remove = []
            for key_id, metadata in self._key_metadata.items():
                if key_id in ["active_rsa", "active_ec", "active_symmetric"]:
                    continue
                
                try:
                    remove_date_str = metadata.get("to_remove")
                    if not remove_date_str:
                        continue
                    
                    remove_date = datetime.fromisoformat(remove_date_str)
                    if remove_date < now:
                        to_remove.append(key_id)
                except:
                    pass
            
            # Remove old keys
            for key_id in to_remove:
                key_type = self._key_metadata.get(key_id, {}).get("type")
                if key_type == "rsa":
                    if key_id in self._rsa_keys:
                        del self._rsa_keys[key_id]
                elif key_type == "ec":
                    if key_id in self._ec_keys:
                        del self._ec_keys[key_id]
                elif key_type == "symmetric":
                    if key_id in self._sym_keys:
                        del self._sym_keys[key_id]
                
                # Remove from metadata
                if key_id in self._key_metadata:
                    del self._key_metadata[key_id]
                
                # Remove key file
                key_file = self.keys_dir / f"{key_id}.key"
                if key_file.exists():
                    try:
                        key_file.unlink()
                    except:
                        pass
            
            # Save metadata if any changes were made
            if rotated_keys or to_remove:
                self._save_metadata()
                
                if rotated_keys:
                    self.logger.info(f"Rotated {len(rotated_keys)} keys")
                if to_remove:
                    self.logger.info(f"Removed {len(to_remove)} old keys")