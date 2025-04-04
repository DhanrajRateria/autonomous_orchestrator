# server/model_manager.py
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os
import logging
from .config import MODEL_FEATURE_COUNT, PRECISION_FACTOR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FILE = 'server/global_model.pkl'

def initialize_global_model():
    """Initializes a new global Logistic Regression model."""
    # Initialize with zeros or small random weights
    # Fitting with dummy data helps set the 'classes_' attribute
    model = LogisticRegression(warm_start=True, max_iter=1) # warm_start allows incremental updates
    # Fit with dummy data to initialize structure correctly
    dummy_X = np.zeros((2, MODEL_FEATURE_COUNT))
    dummy_y = np.array([0, 1]) # Ensure both classes are seen
    model.fit(dummy_X, dummy_y)
    logger.info(f"Initialized global model with {MODEL_FEATURE_COUNT} features.")
    save_model(model)
    return model

def load_model():
    """Loads the global model from disk."""
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            # Ensure model structure matches config (basic check)
            if hasattr(model, 'coef_') and model.coef_.shape[1] == MODEL_FEATURE_COUNT:
                 logger.info(f"Loaded global model from {MODEL_FILE}")
                 return model
            else:
                 logger.warning(f"Model structure in {MODEL_FILE} doesn't match config. Reinitializing.")
                 return initialize_global_model()
        except Exception as e:
            logger.error(f"Error loading model from {MODEL_FILE}: {e}. Reinitializing.")
            return initialize_global_model()
    else:
        logger.info(f"No existing model file found at {MODEL_FILE}. Initializing a new one.")
        return initialize_global_model()

def save_model(model):
    """Saves the global model to disk."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        joblib.dump(model, MODEL_FILE)
        logger.info(f"Saved global model to {MODEL_FILE}")
    except Exception as e:
        logger.error(f"Error saving model to {MODEL_FILE}: {e}")


def get_model_weights(model):
    """Extracts model weights (coefficients and intercept)."""
    # Combine coefficients and intercept into a single flat array
    weights = np.concatenate([model.coef_.flatten(), model.intercept_])
    return weights

def set_model_weights(model, weights):
    """Sets model weights from a flat array."""
    coef_shape = model.coef_.shape
    intercept_shape = model.intercept_.shape

    coef_size = np.prod(coef_shape)
    intercept_size = np.prod(intercept_shape)

    if len(weights) != coef_size + intercept_size:
        raise ValueError(f"Incorrect number of weights provided. Expected {coef_size + intercept_size}, got {len(weights)}")

    model.coef_ = weights[:coef_size].reshape(coef_shape)
    model.intercept_ = weights[coef_size:].reshape(intercept_shape)
    return model


def update_global_model(aggregated_decrypted_updates, num_clients):
    """Updates the global model using the averaged decrypted updates."""
    if num_clients == 0:
        logger.warning("Cannot update model with zero clients.")
        return

    global_model = load_model()
    current_weights = get_model_weights(global_model)

    # Convert aggregated integer updates back to scaled float updates
    # The aggregated_decrypted_updates is the SUM of scaled differences
    # We need the AVERAGE scaled difference
    average_scaled_update = np.array(aggregated_decrypted_updates) / num_clients

    # Convert back to float updates
    average_update = average_scaled_update / PRECISION_FACTOR

    # Apply the average update to the current weights
    new_weights = current_weights + average_update

    # Set the updated weights back to the model
    updated_model = set_model_weights(global_model, new_weights)

    save_model(updated_model)
    logger.info(f"Global model updated with aggregated results from {num_clients} clients.")

    return updated_model

def evaluate_model_and_trigger_action(model):
    """
    Placeholder for evaluating the global model's performance or threat detection
    rate and triggering autonomous actions if necessary.
    """
    # In a real system: Evaluate on a hold-out validation set
    # For simulation: Check if the model predicts high probability of threat for dummy data
    from .config import THREAT_THRESHOLD, MODEL_FEATURE_COUNT
    try:
        # Create some sample data potentially representing threats
        # These features would ideally represent suspicious patterns
        # Example: high login fails, large data egress, unusual port access, etc.
        # We use random data here for demonstration. Features need meaning in reality.
        num_samples = 10
        potential_threat_data = np.random.rand(num_samples, MODEL_FEATURE_COUNT) * 2 # Simulate slightly higher values

        # Increase certain features assumed to be threat indicators (e.g., last few features)
        potential_threat_data[:, -3:] *= 5

        probabilities = model.predict_proba(potential_threat_data)
        threat_probabilities = probabilities[:, 1] # Probability of class '1' (threat)
        avg_threat_prob = np.mean(threat_probabilities)

        logger.info(f"Simulated evaluation: Average threat probability on sample data: {avg_threat_prob:.4f}")

        if avg_threat_prob > THREAT_THRESHOLD:
            logger.warning(f"AUTONOMOUS ACTION TRIGGERED: Average threat probability {avg_threat_prob:.4f} exceeds threshold {THREAT_THRESHOLD}.")
            # --- Simulate Autonomous Action ---
            # Examples:
            # - Block suspicious IPs identified during data analysis (needs IP tracking)
            # - Increase logging levels on affected systems
            # - Alert administrators via webhook/email
            # - Adjust firewall rules (e.g., restrict egress on certain ports)
            # - Isolate a potentially compromised VM/container
            print("\n *** SIMULATING AUTONOMOUS ACTION: Blocking potentially malicious source (simulated) *** \n")
            # In a real implementation, this would call cloud provider APIs (AWS, Azure, GCP)
            # or interact with security tools (SIEM, SOAR platforms).
            pass # Placeholder for actual API calls or system interactions

    except Exception as e:
        logger.error(f"Error during model evaluation or action trigger: {e}")