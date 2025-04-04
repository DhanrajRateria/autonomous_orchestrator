# client/local_trainer.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
import logging
from .config import LOCAL_EPOCHS, BATCH_SIZE, FEATURE_COUNT # LEARNING_RATE (if using SGD)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_local_model(global_weights, X_train, y_train, client_id):
    """
    Trains a local model starting from the global model weights.
    Returns the *difference* between the new local weights and the initial global weights.
    """
    # Initialize local model
    local_model = LogisticRegression(warm_start=True, solver='liblinear', max_iter=100) # Use a solver that works well for smaller datasets
                                                                                       # warm_start helps, but we set weights directly

    # Set initial weights from global model
    # Need to create dummy structure first if model hasn't been fitted
    dummy_X = np.zeros((2, FEATURE_COUNT))
    dummy_y = np.array([0, 1])
    try:
        local_model.fit(dummy_X, dummy_y) # Fit with dummy data to initialize coefficients/intercept arrays
        set_model_weights(local_model, global_weights)
        logger.info(f"Client {client_id}: Initialized local model with global weights.")
    except Exception as e:
         logger.error(f"Client {client_id}: Error setting initial weights: {e}. Starting fresh.")
         # Fallback: train from scratch if setting weights fails
         local_model = LogisticRegression(solver='liblinear', max_iter=100)


    # Train the local model
    # LogisticRegression in scikit-learn doesn't directly support epochs in the same way NN frameworks do.
    # We can simulate epochs by calling fit multiple times or increasing max_iter.
    # For simplicity, we'll just fit once with the data.
    # If using SGDClassifier, could use partial_fit over epochs.
    logger.info(f"Client {client_id}: Starting local training for {LOCAL_EPOCHS} 'epochs' (simulated by max_iter or single fit).")
    try:
        local_model.fit(X_train, y_train) # Train on local data
        logger.info(f"Client {client_id}: Local training complete.")

        # Get the updated local weights
        new_local_weights = get_model_weights(local_model)

        # Calculate the difference (update)
        weight_difference = new_local_weights - global_weights

        return weight_difference

    except Exception as e:
        logger.error(f"Client {client_id}: Error during local training: {e}")
        # Return zero difference if training failed to avoid poisoning the global model
        return np.zeros_like(global_weights)


# Helper functions matching server's model_manager (could be in a shared module)
def get_model_weights(model):
    """Extracts model weights (coefficients and intercept)."""
    if not hasattr(model, 'coef_') or not hasattr(model, 'intercept_'):
         raise NotFittedError("Model weights not available. Model might not be fitted.")
    # Ensure coef_ is 2D
    coef = model.coef_
    if coef.ndim == 1:
        coef = coef.reshape(1, -1) # Reshape if it's flattened (e.g., binary classification)

    weights = np.concatenate([coef.flatten(), model.intercept_.flatten()])
    return weights

def set_model_weights(model, weights):
    """Sets model weights from a flat array."""
    # Check if model has the necessary attributes (coef_, intercept_)
    # This requires the model to be fitted at least once (e.g., with dummy data)
    if not hasattr(model, 'coef_') or not hasattr(model, 'intercept_'):
        raise NotFittedError("Cannot set weights on a model that hasn't been fitted yet.")

    coef_shape = model.coef_.shape
    intercept_shape = model.intercept_.shape

    coef_size = np.prod(coef_shape)
    intercept_size = np.prod(intercept_shape)

    expected_size = coef_size + intercept_size
    if len(weights) != expected_size:
        # Attempt to handle potential flattening issues for binary classification intercept
        if len(weights) == expected_size - intercept_size + 1 and intercept_size > 1:
             # Maybe intercept was flattened incorrectly somewhere, try common case
             expected_size = coef_size + 1
             if len(weights) != expected_size:
                  raise ValueError(f"Incorrect number of weights provided. Expected {expected_size} or {coef_size + intercept_size}, got {len(weights)}")
             else: # Assume intercept should be size 1
                 intercept_size = 1
                 intercept_shape = (1,)
        else:
            raise ValueError(f"Incorrect number of weights provided. Expected {expected_size}, got {len(weights)}")


    model.coef_ = weights[:coef_size].reshape(coef_shape)
    # Handle intercept shape carefully, especially for binary classification where it might be (1,)
    model.intercept_ = weights[coef_size:coef_size + intercept_size].reshape(intercept_shape)
    # If intercept_size > 1, classes_ needs to be set appropriately, usually handled by fit
    if intercept_size == 1:
        model.classes_ = np.array([0, 1]) # Assume binary if intercept is scalar


    return model