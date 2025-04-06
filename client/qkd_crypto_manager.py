# client/crypto_manager.py
import phe as paillier
import json
import logging
import numpy as np # Added for QKD simulation
from client.config import PRECISION_FACTOR, QKD_SIM_LENGTH # Import QKD config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

public_key = None
qkd_shared_key = None # Store derived key {client_id: shared_key_string}

def set_public_key(pub_key_data):
    """
    Sets the Paillier public key received from the server.
    Expects pub_key_data to be a dictionary (parsed from JSON).
    """
    global public_key
    try:
        n_str = pub_key_data.get('n')
        if n_str is None:
             raise ValueError("Public key data from server missing 'n' parameter.")
        n = int(n_str)
        public_key = paillier.PaillierPublicKey(n)
        logger.info("Paillier Public key set successfully.")
    except Exception as e:
        logger.error(f"Failed to set Paillier public key from data: {pub_key_data}. Error: {e}")
        public_key = None # Ensure key is invalid if setting fails

def get_public_key():
     if public_key is None:
         raise ValueError("Public key not set. Register with the server first.")
     return public_key

# --- QKD Simulation (BB84 - Client Side: Alice) ---

def simulate_qkd_client_protocol(bob_bases_str, alice_bits, alice_bases):
    """
    Simulates the client (Alice) side of BB84 reconciliation.
    Receives Bob's bases, compares with own, and derives the shared key.

    Args:
        bob_bases_str (str): Server's (Bob's) chosen bases as a comma-separated string.
        alice_bits (np.array): The initial random bits Alice generated.
        alice_bases (np.array): The initial random bases Alice generated.

    Returns:
        str: Derived shared key (hex string) or None on error.
    """
    global qkd_shared_key
    try:
        bob_bases = np.array([int(b) for b in bob_bases_str.split(',')])

        if len(bob_bases) != len(alice_bases):
            logger.error(f"QKD Error: Length mismatch between Alice's bases ({len(alice_bases)}) and Bob's bases ({len(bob_bases)}).")
            return None

        # Identify indices where bases match
        matched_indices = np.where(alice_bases == bob_bases)[0]

        # Extract the bits corresponding to matched bases
        final_shared_key_bits = alice_bits[matched_indices]

        # Simulate subset check for errors (skipped here)

        # Store and return the key
        effective_key_length = len(final_shared_key_bits)
        if effective_key_length == 0:
             logger.error("QKD Simulation Error: No matching bases found, cannot generate key.")
             return None

        # Convert bit array to hex string (matching server side for comparison)
        shared_key_hex = hex(int("".join(map(str, final_shared_key_bits)), 2))[2:]

        # Refine key length if needed (e.g., hash to fixed length, truncate)
        # For simulation, we assume the reconciliation yields the target length or fewer bits
        logger.info(f"QKD Simulation Client: Successfully derived simulated shared key of length {effective_key_length} bits.")
        qkd_shared_key = shared_key_hex
        return shared_key_hex

    except Exception as e:
        logger.error(f"Error during QKD client simulation: {e}", exc_info=True)
        return None

def generate_qkd_client_initial_state():
    """Generates Alice's initial random bits and bases."""
    bits = np.random.randint(0, 2, QKD_SIM_LENGTH)
    bases = np.random.randint(0, 2, QKD_SIM_LENGTH)
    bases_str = ','.join(map(str, bases))
    return bits, bases, bases_str


# --- HE Operations (Unchanged) ---

def encrypt_vector(vector):
    # ... (no changes needed)
    pub_key = get_public_key()
    scaled_vector = (vector * PRECISION_FACTOR).astype(int)
    encrypted_vector = [pub_key.encrypt(int(x)) for x in scaled_vector]
    serialized_vector = [{'ciphertext': str(num.ciphertext(be_secure=False)), 'exponent': num.exponent}
                         for num in encrypted_vector]
    return json.dumps(serialized_vector)


def encrypt_value(value):
    # ... (no changes needed)
    pub_key = get_public_key()
    encrypted_value = pub_key.encrypt(int(value))
    serialized_value = {'ciphertext': str(encrypted_value.ciphertext(be_secure=False)),
                        'exponent': encrypted_value.exponent}
    return json.dumps(serialized_value)