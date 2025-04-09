import numpy as np
from typing import List, Tuple

def parameters_to_weights(parameters: List[np.ndarray]) -> List[np.ndarray]:
    """Convert model parameters (list of ndarrays) to weights (list of ndarrays)."""
    # In this simple case, parameters are the weights
    return parameters

def weights_to_parameters(weights: List[np.ndarray]) -> List[np.ndarray]:
    """Convert weights (list of ndarrays) to model parameters (list of ndarrays)."""
    # In this simple case, weights are the parameters
    return weights

def flatten_parameters(parameters: List[np.ndarray]) -> np.ndarray:
    """Flattens a list of NumPy arrays (model parameters) into a single 1D array."""
    return np.concatenate([p.flatten() for p in parameters])

def unflatten_parameters(flat_params: np.ndarray, shapes: List[Tuple[int, ...]], dtypes: List[np.dtype]) -> List[np.ndarray]:
    """Unflattens a 1D array back into a list of NumPy arrays with original shapes and dtypes."""
    unflattened_params = []
    current_index = 0
    for shape, dtype in zip(shapes, dtypes):
        num_elements = np.prod(shape)
        param = flat_params[current_index : current_index + num_elements].reshape(shape).astype(dtype)
        unflattened_params.append(param)
        current_index += num_elements
    return unflattened_params

# Example Usage
if __name__ == "__main__":
    from model import CloudSecurityModel, get_model_parameters
    model = CloudSecurityModel(num_features=10, num_hidden_units=5, num_classes=2)
    original_params = get_model_parameters(model)
    original_shapes = [p.shape for p in original_params]
    original_dtypes = [p.dtype for p in original_params]

    flat = flatten_parameters(original_params)
    print(f"Original param list length: {len(original_params)}")
    print(f"Flattened params shape: {flat.shape}")

    unflattened = unflatten_parameters(flat, original_shapes, original_dtypes)
    print(f"Unflattened param list length: {len(unflattened)}")

    # Verify shapes and content
    assert len(original_params) == len(unflattened), "Length mismatch"
    for i in range(len(original_params)):
        assert original_params[i].shape == unflattened[i].shape, f"Shape mismatch at index {i}"
        assert original_params[i].dtype == unflattened[i].dtype, f"Dtype mismatch at index {i}"
        assert np.allclose(original_params[i], unflattened[i]), f"Value mismatch at index {i}"

    print("\nFlatten/Unflatten test passed.")