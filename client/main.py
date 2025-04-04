# client/main.py
import requests
import time
import logging
import numpy as np

from .config import SERVER_URL, CLIENT_ID
from . import crypto_manager
from . import data_simulator
from . import local_trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- State ---
current_round = -1
expected_feature_count = -1

def register_with_server():
    """Registers the client with the orchestrator server and gets the public key."""
    global expected_feature_count
    url = f"{SERVER_URL}/register"
    try:
        response = requests.post(url, json={'client_id': CLIENT_ID})
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        crypto_manager.set_public_key(data['public_key'])
        expected_feature_count = data.get('feature_count', -1) # Get expected feature count
        if expected_feature_count == -1 :
            logger.warning("Server did not provide expected feature count.")
        else:
            # Validate against local config (optional but good practice)
            from .config import FEATURE_COUNT
            if expected_feature_count != FEATURE_COUNT:
                 logger.error(f"FATAL: Feature count mismatch! Server expects {expected_feature_count}, client configured for {FEATURE_COUNT}.")
                 return False # Indicate registration failure due to mismatch
            logger.info(f"Successfully registered Client {CLIENT_ID}. Server expects {expected_feature_count} features.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to register with server at {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"An error occurred during registration: {e}")
        return False

def get_global_model(round_num):
    """Fetches the current global model weights from the server."""
    url = f"{SERVER_URL}/get_model"
    params = {'client_id': CLIENT_ID, 'round': round_num}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data['round'] != round_num:
             logger.warning(f"Received model for round {data['round']}, expected {round_num}. Skipping update.")
             return None
        logger.info(f"Received global model weights for round {round_num}.")
        return np.array(data['weights'])
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get global model from server: {e}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while fetching model: {e}")
        return None


def submit_encrypted_update(round_num, encrypted_update_json):
    """Submits the encrypted local model update to the server."""
    url = f"{SERVER_URL}/submit_update"
    payload = {
        'client_id': CLIENT_ID,
        'round': round_num,
        'update': encrypted_update_json # This is the JSON string from crypto_manager
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logger.info(f"Successfully submitted encrypted update for round {round_num}.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to submit update to server: {e}")
        return False
    except Exception as e:
        logger.error(f"An error occurred during update submission: {e}")
        return False


def run_client_round():
    """Executes one round of federated learning from the client's perspective."""
    global current_round

    # 1. Get Global Model for the current round
    logger.info(f"Attempting to fetch model for round {current_round}...")
    global_weights = get_global_model(current_round)
    if global_weights is None:
        logger.warning(f"Could not get global model for round {current_round}. Skipping this round.")
        return

    # Validate weight dimensions if possible (needs expected_feature_count)
    if expected_feature_count > 0:
        # Expected size = features + intercept (usually 1 for binary classification)
        # This check is simplistic as intercept shape can vary. Rely on server/client consistency.
        # expected_len = expected_feature_count + 1
        # if len(global_weights) != expected_len:
        #      logger.error(f"Received model weights have unexpected length {len(global_weights)}, expected around {expected_len}. Check model configuration.")
        #      return # Skip round due to potential incompatibility
        pass # Relaxed check for now

    # 2. Generate Local Data
    logger.info("Generating local security data...")
    X_local, y_local = data_simulator.generate_data(CLIENT_ID)
    if X_local.shape[1] != expected_feature_count:
         logger.error(f"FATAL: Generated data feature count {X_local.shape[1]} doesn't match expected {expected_feature_count}.")
         return # Stop if data shape is wrong

    # 3. Train Local Model
    logger.info("Training local model...")
    start_train_time = time.time()
    weight_difference = local_trainer.train_local_model(global_weights, X_local, y_local, CLIENT_ID)
    train_time = time.time() - start_train_time
    logger.info(f"Local training finished in {train_time:.4f} seconds.")

    # 4. Encrypt the Update (Weight Difference)
    logger.info("Encrypting model update...")
    start_encrypt_time = time.time()
    try:
        encrypted_update_json = crypto_manager.encrypt_vector(weight_difference)
        encrypt_time = time.time() - start_encrypt_time
        logger.info(f"Encryption finished in {encrypt_time:.4f} seconds.")
    except ValueError as e: # Catch errors like public key not set
        logger.error(f"Encryption failed: {e}. Cannot submit update.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during encryption: {e}")
        return

    # 5. Submit Encrypted Update
    logger.info("Submitting encrypted update to server...")
    submit_encrypted_update(current_round, encrypted_update_json)


def main_loop():
    """Main loop for the client."""
    global current_round
    if not register_with_server():
        logger.error("Client registration failed. Exiting.")
        return

    # Participate in rounds (in reality, driven by server coordination)
    # Here, we poll or assume rounds increment sequentially.
    server_rounds = 5 # Get this from config or server if possible
    from .config import SERVER_URL
    r_check_url = f"{SERVER_URL}/get_model" # Use get_model endpoint to check current round

    processed_rounds = set()

    while True:
        # Check current server round (simple polling mechanism)
        try:
            # Make a lightweight request just to check the round
            params = {'client_id': CLIENT_ID, 'round': -1} # Ask for invalid round
            response = requests.get(r_check_url, params=params, timeout=10)
            if response.status_code == 400 and "current is" in response.text:
                 # Extract current round from error message (hacky, needs better API)
                 try:
                     msg = response.json().get("error", "")
                     server_round = int(msg.split("current is ")[1].split(")")[0])
                 except:
                      logger.warning("Could not parse current round from server response.")
                      time.sleep(10)
                      continue
            elif response.status_code == 200: # Should not happen for round -1, but handle defensively
                 server_round = response.json().get("round", -1)
            else:
                 logger.warning(f"Unexpected response ({response.status_code}) when checking server round.")
                 time.sleep(10)
                 continue

        except requests.exceptions.Timeout:
             logger.warning("Timeout checking server round. Server might be busy or down.")
             time.sleep(15)
             continue
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to server to check round: {e}")
            time.sleep(15) # Wait longer if connection fails
            continue

        if server_round > current_round and server_round not in processed_rounds:
            logger.info(f"Detected new server round: {server_round}")
            current_round = server_round
            run_client_round()
            processed_rounds.add(current_round)
        elif server_round == -1 :
             logger.info("Server not yet started or finished rounds. Waiting.")
        else:
            # logger.info(f"Waiting for next round (current server round: {server_round})...")
            pass # Already processed or waiting for server to advance

        # Stop condition (example: after a certain number of rounds)
        if len(processed_rounds) >= server_rounds: # Match server's NUM_ROUNDS
             logger.info("Finished participating in all rounds.")
             break

        time.sleep(5) # Wait before checking again


if __name__ == '__main__':
    logger.info(f"Starting Client: {CLIENT_ID}")
    main_loop()
    logger.info(f"Client {CLIENT_ID} shutting down.")