import phe as paillier
import pickle
import numpy as np
import logging
from typing import List, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HomomorphicEncryption:
    """
    Handles Paillier homomorphic encryption operations (key generation,
    encryption, decryption, serialization) for model parameters/updates.
    Uses fixed-point encoding for floats.
    """
    def __init__(self, key_length: int = 1024, precision_bits: int = 64):
        logger.info(f"Generating Paillier keys with length {key_length}...")
        self._public_key, self._private_key = paillier.generate_paillier_keypair(n_length=key_length)
        logger.info("Paillier keys generated.")
        # Precision factor for encoding/decoding floats
        # Using 2**precision_bits for better handling of range and precision
        self._precision_factor = 1 << precision_bits # Equivalent to 2**precision_bits
        self._precision_bits = precision_bits

    def get_public_key(self) -> paillier.PaillierPublicKey:
        return self._public_key

    def get_private_key(self) -> paillier.PaillierPrivateKey:
        return self._private_key

    def _encode(self, value: float) -> int:
        """Encodes a float to an integer using fixed-point representation."""
        # Clamp values to avoid overflow issues with Paillier's integer limits if necessary
        # This depends heavily on the range of your model parameters/updates
        # For simplicity, we're not clamping here, but it might be needed in practice.
        try:
            encoded = int(value * self._precision_factor)
            # Check if the encoded value fits within Paillier's plaintext space (roughly n/3 for Paillier)
            # This check is complex and depends on the specific Paillier implementation details.
            # A simpler practical check might be against a large portion of n.
            # max_plaintext = self._public_key.n // 3 # Approximate limit
            # if abs(encoded) > max_plaintext:
            #     logger.warning(f"Encoded value {encoded} might exceed Paillier plaintext space. Clamping or reducing precision might be needed.")
            return encoded
        except OverflowError:
            logger.error(f"OverflowError encoding float: {value}. Check precision or value range.")
            # Handle overflow - e.g., clamp to max/min representable int or raise error
            # Returning 0 or raising an exception are options. Raising is safer.
            raise OverflowError(f"Cannot encode float {value} with precision {self._precision_bits}")


    def _decode(self, value: int) -> float:
        """Decodes an integer back to a float."""
        return float(value) / self._precision_factor

    def encrypt_list(self, data: List[float]) -> List[paillier.EncryptedNumber]:
        """Encrypts a list of floats."""
        if not isinstance(data, list):
            raise TypeError("Input data must be a list of floats.")
        encoded_data = [self._encode(x) for x in data]
        encrypted_data = [self._public_key.encrypt(x) for x in encoded_data]
        return encrypted_data

    def decrypt_list(self, encrypted_data: List[paillier.EncryptedNumber]) -> List[float]:
        """Decrypts a list of Paillier EncryptedNumbers."""
        if not isinstance(encrypted_data, list) or not all(isinstance(x, paillier.EncryptedNumber) for x in encrypted_data):
             raise TypeError("Input must be a list of Paillier EncryptedNumbers.")
        decrypted_encoded = [self._private_key.decrypt(x) for x in encrypted_data]
        decrypted_data = [self._decode(x) for x in decrypted_encoded]
        return decrypted_data

    def decrypt_single(self, encrypted_value: paillier.EncryptedNumber) -> float:
        """Decrypts a single Paillier EncryptedNumber."""
        if not isinstance(encrypted_value, paillier.EncryptedNumber):
             raise TypeError("Input must be a Paillier EncryptedNumber.")
        decrypted_encoded = self._private_key.decrypt(encrypted_value)
        return self._decode(decrypted_encoded)

    @staticmethod
    def serialize_encrypted_list(encrypted_list: List[paillier.EncryptedNumber]) -> bytes:
        """Serializes a list of encrypted numbers using pickle."""
        # Storing ciphertext and exponent is often more robust than pickling objects
        # Pickling works here for simplicity with python-paillier objects
        serializable_list = [(n.ciphertext(), n.exponent) for n in encrypted_list]
        return pickle.dumps(serializable_list)

    @staticmethod
    def deserialize_encrypted_list(serialized_data: bytes, public_key: paillier.PaillierPublicKey) -> List[paillier.EncryptedNumber]:
        """Deserializes bytes back into a list of encrypted numbers."""
        serializable_list = pickle.loads(serialized_data)
        # Reconstruct EncryptedNumber objects
        reconstructed_list = [paillier.EncryptedNumber(public_key, c, e) for c, e in serializable_list]
        return reconstructed_list

    @staticmethod
    def aggregate_encrypted_lists(encrypted_lists: List[List[paillier.EncryptedNumber]]) -> List[paillier.EncryptedNumber]:
        """Performs element-wise addition of multiple lists of encrypted numbers."""
        if not encrypted_lists:
            return []

        # Ensure all lists have the same length
        list_len = len(encrypted_lists[0])
        if not all(len(lst) == list_len for lst in encrypted_lists):
            raise ValueError("All encrypted lists must have the same length for aggregation.")

        aggregated_result = [sum(encrypted_values) for encrypted_values in zip(*encrypted_lists)]
        return aggregated_result

# Example Usage (for testing)
if __name__ == "__main__":
    he = HomomorphicEncryption(key_length=1024, precision_bits=64)
    pk = he.get_public_key()

    # Test encryption/decryption
    data1 = [1.23, -0.5, 100.0]
    encrypted1 = he.encrypt_list(data1)
    decrypted1 = he.decrypt_list(encrypted1)
    print(f"Original 1: {data1}")
    print(f"Decrypted 1: {decrypted1}")
    assert np.allclose(data1, decrypted1, atol=1e-8), "Decryption failed!"

    # Test serialization/deserialization
    serialized = he.serialize_encrypted_list(encrypted1)
    deserialized = he.deserialize_encrypted_list(serialized, pk)
    decrypted_deserialized = he.decrypt_list(deserialized)
    print(f"Decrypted after serialization: {decrypted_deserialized}")
    assert np.allclose(data1, decrypted_deserialized, atol=1e-8), "Serialization/Deserialization failed!"

    # Test aggregation
    data2 = [0.77, 0.5, -50.0]
    encrypted2 = he.encrypt_list(data2)

    aggregated_encrypted = he.aggregate_encrypted_lists([encrypted1, encrypted2])
    decrypted_aggregated = he.decrypt_list(aggregated_encrypted)
    expected_aggregation = [d1 + d2 for d1, d2 in zip(data1, data2)]
    print(f"Original 2: {data2}")
    print(f"Expected Aggregation: {expected_aggregation}")
    print(f"Decrypted Aggregation: {decrypted_aggregated}")
    assert np.allclose(expected_aggregation, decrypted_aggregated, atol=1e-8), "Aggregation failed!"

    print("HE Utils tests passed.")