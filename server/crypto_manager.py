# server/crypto_manager.py
import phe as paillier
import numpy as np
import json
import logging
from server.config import HE_KEY_SIZE, QKD_KEY_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

public_key, private_key = None, None
# Store simulated QKD state per client if needed for more complex simulation
# For simple BB84 simulation, we can generate Bob's part on the fly
qkd_shared_keys = {} # Store derived keys per client_id {client_id: shared_key_string}

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

def get_serialized_public_key():
    """Returns the public key serialized for transmission."""
    pub_key = get_public_key()
    # Serialize public key for sending via JSON
    pub_key_json = {'n': str(pub_key.n)}
    return json.dumps(pub_key_json)

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
    
# --- QKD Simulation (BB84 - Server Side: Bob) ---

def simulate_qkd_server_protocol(client_id, alice_bases_str):
    """
    Simulates the server (Bob) side of BB84 key exchange.
    Receives Alice's bases, generates Bob's, compares, and derives a shared key.

    Args:
        client_id (str): Identifier for the client.
        alice_bases_str (str): Client's (Alice's) chosen bases as a comma-separated string.

    Returns:
        tuple: (bob_bases_str, shared_key_str) where
               bob_bases_str is Bob's chosen bases (comma-separated string).
               shared_key_str is the derived shared key (hex string) or None on error.
    """
    try:
        alice_bases = np.array([int(b) for b in alice_bases_str.split(',')])
        key_length = len(alice_bases) # Use the length sent by Alice

        # Bob generates his random bases for measurement
        bob_bases = np.random.randint(0, 2, key_length)

        # In a real protocol, Bob measures Alice's qubits using his bases.
        # Here, we simulate the *result* of the public comparison phase.
        # Indices where Alice and Bob used the same basis are kept.
        matched_indices = np.where(alice_bases == bob_bases)[0]

        # Simulate Alice sending her *bits* for the matched indices (or Bob deriving them)
        # In this simplified simulation, we just generate a key of appropriate length directly
        # A real simulation would involve Alice sending bits and checking a subset for errors.
        num_matched_bits = len(matched_indices)
        if num_matched_bits < QKD_KEY_LENGTH: # Check if enough bits survived for the target key length
            logger.warning(f"QKD for {client_id}: Not enough matched bases ({num_matched_bits}) to generate {QKD_KEY_LENGTH} bit key. Simulation yields fewer bits.")
            # Handle this case - perhaps request retry or use fewer bits
            if num_matched_bits == 0: return (','.join(map(str, bob_bases)), None) # No key possible
            effective_key_length = num_matched_bits
        else:
             # Select a subset of matched indices to form the key
             selected_indices = np.random.choice(matched_indices, QKD_KEY_LENGTH, replace=False)
             effective_key_length = QKD_KEY_LENGTH

        # Simulate the final shared key bits (these would come from Alice's original bits)
        # For simulation, we just generate random bits for the agreed length
        final_shared_key_bits = np.random.randint(0, 2, effective_key_length)

        # Store and return the key (e.g., as a hex string)
        # In a real system, use bytes: shared_key_bytes = bytes(final_shared_key_bits)
        shared_key_hex = hex(int("".join(map(str, final_shared_key_bits)), 2))[2:] # Convert bit array to hex string

        qkd_shared_keys[client_id] = shared_key_hex
        logger.info(f"QKD Simulation for {client_id}: Successfully derived simulated shared key of length {effective_key_length} bits.")

        bob_bases_str = ','.join(map(str, bob_bases))
        return bob_bases_str, shared_key_hex # In simulation, we return key directly for logging

    except Exception as e:
        logger.error(f"Error during QKD server simulation for {client_id}: {e}", exc_info=True)
        return (','.join(map(str, bob_bases)) if 'bob_bases' in locals() else "", None)