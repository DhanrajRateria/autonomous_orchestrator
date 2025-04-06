# server/main.py
from flask import Flask, request, jsonify
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from server import crypto_manager
from server import model_manager
from server.config import SERVER_HOST, SERVER_PORT, NUM_ROUNDS, CLIENTS_PER_ROUND, MIN_CLIENTS_FOR_AGGREGATION

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# In-memory state (replace with database/persistent storage in production)
client_registry = {} # client_id -> last_seen
round_updates = {} # round_number -> {client_id: encrypted_update_json}
current_round = 0
global_model = None
server_lock = threading.Lock() # To protect shared state like round_updates

# Thread pool for handling concurrent tasks like decryption/aggregation
executor = ThreadPoolExecutor(max_workers=4) # Adjust worker count based on server resources

def initialize_server():
    """Initialize server components."""
    global global_model
    logger.info("Initializing orchestrator server...")
    crypto_manager.generate_keys()
    global_model = model_manager.load_model()
    logger.info("Server initialized.")

@app.route('/register', methods=['POST'])
def register_client():
    """Allows clients to register and get the public key."""
    client_id = request.json.get('client_id')
    if not client_id:
        return jsonify({"error": "Client ID required"}), 400

    with server_lock:
        client_registry[client_id] = time.time()

    pub_key = crypto_manager.get_public_key()
    # Serialize public key for sending via JSON
    pub_key_json = {'n': str(pub_key.n)} # Only 'n' is needed by clients for encryption usually

    logger.info(f"Client {client_id} registered.")
    return jsonify({
        "message": "Registered successfully",
        "public_key": pub_key_json,
        "feature_count": model_manager.MODEL_FEATURE_COUNT # Inform client of expected features
        })

@app.route('/get_model', methods=['GET'])
def get_model():
    """Provides the current global model parameters to clients."""
    client_id = request.args.get('client_id')
    request_round = request.args.get('round', type=int)

    if not client_id or client_id not in client_registry:
         return jsonify({"error": "Client not registered or invalid ID"}), 403
    if request_round != current_round:
         return jsonify({"error": f"Requesting model for wrong round ({request_round}), current is {current_round}"}), 400

    model_weights = model_manager.get_model_weights(global_model)
    logger.info(f"Sending model (round {current_round}) to client {client_id}")

    return jsonify({
        "round": current_round,
        "weights": model_weights.tolist() # Send weights as a list
        })

@app.route('/submit_update', methods=['POST'])
def submit_update():
    """Receives encrypted model updates from clients."""
    data = request.json
    client_id = data.get('client_id')
    round_num = data.get('round')
    encrypted_update_json = data.get('update') # This is already JSON string from client

    if not client_id or client_id not in client_registry:
        return jsonify({"error": "Client not registered or invalid ID"}), 403
    if round_num != current_round:
        return jsonify({"error": f"Update submitted for wrong round ({round_num}), current is {current_round}"}), 400
    if not encrypted_update_json:
         return jsonify({"error": "Encrypted update missing"}), 400

    with server_lock:
        if current_round not in round_updates:
            round_updates[current_round] = {}
        # Only accept one update per client per round
        if client_id in round_updates[current_round]:
            logger.warning(f"Client {client_id} already submitted update for round {current_round}. Ignoring.")
            return jsonify({"message": "Update already received for this round"}), 200

        round_updates[current_round][client_id] = encrypted_update_json
        logger.info(f"Received encrypted update from {client_id} for round {current_round}. Total updates this round: {len(round_updates[current_round])}")

        # Check if enough updates received to trigger aggregation
        if len(round_updates[current_round]) >= MIN_CLIENTS_FOR_AGGREGATION:
             # Optionally trigger aggregation immediately or wait for a timer/end of round
             logger.info(f"Minimum updates ({MIN_CLIENTS_FOR_AGGREGATION}) reached for round {current_round}. Aggregation can proceed.")
             # We will trigger aggregation explicitly in the main training loop

    return jsonify({"message": "Update received successfully"})

def run_federated_round(round_num):
    """Manages a single round of federated learning."""
    global global_model, current_round
    logger.info(f"--- Starting Federated Round {round_num} ---")

    with server_lock:
        current_round = round_num
        # In a real system, client selection would be more sophisticated
        # Here, we assume clients participating are those who submit updates
        round_updates[current_round] = {} # Clear updates for the new round

    # Wait for clients to fetch the model and submit updates
    # In a real system, use timeouts and potentially select specific clients
    logger.info(f"Waiting for client updates for round {round_num}...")
    # We'll rely on a timeout or a manual trigger in this simple loop.
    # Let's wait for a fixed time or until enough clients respond.
    round_start_time = time.time()
    wait_time_seconds = 60 # Wait up to 60 seconds for updates

    while time.time() - round_start_time < wait_time_seconds:
        with server_lock:
             num_received = len(round_updates.get(current_round, {}))
        # logger.info(f"Round {current_round}: Received {num_received} updates...")
        if num_received >= MIN_CLIENTS_FOR_AGGREGATION:
             logger.info(f"Round {current_round}: Reached minimum {num_received} updates. Proceeding early.")
             break
        time.sleep(5) # Check every 5 seconds

    # --- Aggregation and Update ---
    with server_lock:
        updates_to_process = round_updates.get(current_round, {})
        num_updates = len(updates_to_process)
        logger.info(f"Round {current_round} ended. Received {num_updates} updates.")

        if num_updates < MIN_CLIENTS_FOR_AGGREGATION:
            logger.warning(f"Round {current_round}: Not enough updates ({num_updates}) received. Skipping model update for this round.")
            return # Skip to next round

        # --- System Programming Aspect: Concurrent Aggregation/Decryption ---
        # Submit aggregation and decryption to the thread pool
        logger.info("Submitting aggregation and decryption tasks to executor...")
        future = executor.submit(aggregate_and_decrypt, list(updates_to_process.values()), num_updates)

        # Wait for the result
        try:
            aggregated_decrypted_updates = future.result(timeout=120) # Timeout for decryption

            if aggregated_decrypted_updates:
                logger.info("Aggregation and decryption complete. Updating global model.")
                global_model = model_manager.update_global_model(aggregated_decrypted_updates, num_updates)

                # --- Autonomous Action Trigger ---
                logger.info("Evaluating model and checking for autonomous actions...")
                model_manager.evaluate_model_and_trigger_action(global_model)

            else:
                logger.error("Aggregation/decryption failed or returned no result.")

        except Exception as e:
            logger.error(f"Error during aggregation/decryption task execution: {e}")


    logger.info(f"--- Federated Round {round_num} Complete ---")


def aggregate_and_decrypt(encrypted_updates_list, num_clients):
    """Function to run aggregation and decryption, potentially in parallel."""
    try:
        # 1. Aggregate Encrypted Updates
        logger.info(f"Aggregating {num_clients} encrypted vectors...")
        aggregated_encrypted_json = crypto_manager.aggregate_encrypted_vectors(encrypted_updates_list)
        if aggregated_encrypted_json is None:
            logger.error("Aggregation resulted in None.")
            return None

        # 2. Decrypt the Aggregated Result
        logger.info("Decrypting aggregated vector...")
        start_decrypt_time = time.time()
        aggregated_decrypted_updates = crypto_manager.decrypt_vector(aggregated_encrypted_json)
        decrypt_time = time.time() - start_decrypt_time
        logger.info(f"Decryption took {decrypt_time:.4f} seconds.")

        return aggregated_decrypted_updates
    except Exception as e:
         logger.error(f"Error in aggregate_and_decrypt: {e}", exc_info=True)
         return None


def run_server():
    initialize_server()
    # Start Flask server in a background thread
    flask_thread = threading.Thread(target=lambda: app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True), daemon=True)
    flask_thread.start()
    logger.info(f"Flask server running on http://{SERVER_HOST}:{SERVER_PORT}")

    # Run federated learning rounds
    for r in range(NUM_ROUNDS):
        run_federated_round(r)
        # Optional: Add delay between rounds
        time.sleep(5)

    logger.info("Federated learning process finished.")
    # Keep the Flask server running if needed, or add shutdown logic
    # flask_thread.join() # Uncomment if you want the main script to wait for Flask


if __name__ == '__main__':
    run_server()