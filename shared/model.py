import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Tuple
import numpy as np

class CloudSecurityModel(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for cloud threat detection.
    """
    def __init__(self, num_features: int, num_hidden_units: int, num_classes: int):
        super().__init__()
        self.layer_1 = nn.Linear(num_features, num_hidden_units)
        self.layer_2 = nn.Linear(num_hidden_units, num_classes)

    def forward(self, x):
        x = x.float() # Ensure input is float
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x) # Raw scores (logits)
        # No softmax here, CrossEntropyLoss expects logits
        return x

def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Returns model parameters as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]):
    """Sets model parameters from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

# Example Usage
if __name__ == "__main__":
    model = CloudSecurityModel(num_features=20, num_hidden_units=64, num_classes=2)
    print("Model Architecture:")
    print(model)

    params = get_model_parameters(model)
    print(f"\nNumber of parameter tensors: {len(params)}")
    print(f"Shape of first parameter tensor (layer_1 weights): {params[0].shape}")

    # Modify parameters slightly and set them back
    params[0] += 0.01
    set_model_parameters(model, params)

    new_params = get_model_parameters(model)
    assert np.allclose(params[0], new_params[0]), "Setting parameters failed!"
    print("\nModel parameters get/set test passed.")