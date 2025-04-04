# client/crypto_manager.py
import phe as paillier
import json
import logging
import numpy as np
from .config import PRECISION_FACTOR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

public_key = None

def set_public_key(pub_key_json):
    """Sets the Paillier public key received from the server."""
    global public_key
    try:
        n = int(pub_key_json['n'])
        public_key = paillier.PaillierPublicKey(n)
        logger.info("Public key set successfully.")
    except Exception as e:
        logger.error(f"Failed to set public key from JSON: {pub_key_json}. Error: {e}")
        public_key = None # Ensure key is invalid if setting fails

def get_public_key():
     if public_key is None:
         raise ValueError("Public key not set. Register with the server first.")
     return public_key

def encrypt_vector(vector):
    """Encrypts a numpy vector using the Paillier public key."""
    pub_key = get_public_key()
    # Scale and convert to integer before encryption
    scaled_vector = (vector * PRECISION_FACTOR).astype(int)
    encrypted_vector = [pub_key.encrypt(int(x)) for x in scaled_vector] # Ensure x is Python int

    # Serialize for JSON transmission
    serialized_vector = [{'ciphertext': str(num.ciphertext(be_secure=False)), 'exponent': num.exponent}
                         for num in encrypted_vector]
    return json.dumps(serialized_vector)

def encrypt_value(value):
    """Encrypts a single scaled integer value."""
    pub_key = get_public_key()
    # Assume value is already scaled and converted to int if needed outside this func
    encrypted_value = pub_key.encrypt(int(value))
    # Serialize
    serialized_value = {'ciphertext': str(encrypted_value.ciphertext(be_secure=False)),
                        'exponent': encrypted_value.exponent}
    return json.dumps(serialized_value)