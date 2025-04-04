# server/crypto_manager.py
import phe as paillier
import json
import logging
from .config import HE_KEY_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

public_key, private_key = None, None

def generate_keys():
    """Generates Paillier key pair."""
    global public_key, private_key
    logger.info(f"Generating Paillier key pair with size {HE_KEY_SIZE} bits...")
    public_key, private_key = paillier.generate_paillier_keypair(n_length=HE_KEY_SIZE)
    logger.info("Key pair generated.")
    # In a real system, keys might be saved securely, e.g., using a KMS
    # with open("server_public_key.json", 'w') as f: json.dump({'n': public_key.n}, f)
    # with open("server_private_key.json", 'w') as f: json.dump({'p': private_key.p, 'q': private_key.q}, f)
    return public_key, private_key

def get_public_key():
    """Returns the public key."""
    if not public_key:
        generate_keys()
    return public_key

def get_private_key():
    """Returns the private key."""
    if not private_key:
        generate_keys()
    return private_key

def decrypt_value(encrypted_value_json):
    """Decrypts a single Paillier encrypted value."""
    priv_key = get_private_key()
    try:
        # Deserialize the EncryptedNumber object correctly
        encrypted_data = json.loads(encrypted_value_json)
        encrypted_number = paillier.EncryptedNumber(get_public_key(),
                                                    int(encrypted_data['ciphertext']),
                                                    int(encrypted_data['exponent']))
        decrypted_value = priv_key.decrypt(encrypted_number)
        return decrypted_value
    except Exception as e:
        logger.error(f"Decryption failed for value: {encrypted_value_json}. Error: {e}")
        raise

def decrypt_vector(encrypted_vector_json):
    """Decrypts a vector of Paillier encrypted values."""
    priv_key = get_private_key()
    pub_key = get_public_key()
    encrypted_vector_list = json.loads(encrypted_vector_json)
    decrypted_vector = []
    for item in encrypted_vector_list:
        try:
            encrypted_number = paillier.EncryptedNumber(pub_key,
                                                        int(item['ciphertext']),
                                                        int(item['exponent']))
            decrypted_value = priv_key.decrypt(encrypted_number)
            decrypted_vector.append(decrypted_value)
        except Exception as e:
            logger.error(f"Decryption failed for item: {item}. Error: {e}")
            # Decide how to handle partial failure - skip? error out?
            continue # Skip this element for now
    return decrypted_vector

def aggregate_encrypted_vectors(encrypted_vectors):
    """
    Aggregates multiple encrypted vectors using Paillier's additive homomorphism.
    Assumes all vectors are encrypted with the same public key.
    """
    if not encrypted_vectors:
        return None

    pub_key = get_public_key()
    aggregated_vector = None

    # Deserialize and sum
    for enc_vec_json in encrypted_vectors:
        enc_vec_list = json.loads(enc_vec_json)
        if aggregated_vector is None:
            # Initialize aggregated_vector with the first vector
            aggregated_vector = [paillier.EncryptedNumber(pub_key, int(item['ciphertext']), int(item['exponent']))
                                 for item in enc_vec_list]
        else:
            if len(aggregated_vector) != len(enc_vec_list):
                logger.warning(f"Skipping vector due to length mismatch. Expected {len(aggregated_vector)}, got {len(enc_vec_list)}")
                continue
            # Add the current vector element-wise
            for i, item in enumerate(enc_vec_list):
                try:
                    current_enc_num = paillier.EncryptedNumber(pub_key, int(item['ciphertext']), int(item['exponent']))
                    aggregated_vector[i] += current_enc_num
                except Exception as e:
                     logger.error(f"Error aggregating item at index {i}: {item}. Error: {e}")
                     # Handle element-wise aggregation error (e.g., skip element, use placeholder)
                     # For simplicity, we might continue, but this could skew results.
                     # A robust system might require re-submission or exclude the vector.
                     continue # Skip this element addition

    # Serialize the result back to JSON representation for storage/decryption
    if aggregated_vector:
         serialized_aggregate = json.dumps([{'ciphertext': str(num.ciphertext(be_secure=False)), 'exponent': num.exponent}
                                       for num in aggregated_vector])
         return serialized_aggregate
    else:
        return None