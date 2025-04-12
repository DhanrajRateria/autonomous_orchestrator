# models.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import logging
from typing import List, Dict, Any, Union, Optional

from federated_learning_framework.config import ModelConfig # Assuming config.py is in this path


class MLP(nn.Module):
    """Multi-layer perceptron for multi-class classification or regression."""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout_rate: float = 0.2, activation: str = "relu", task_type: str = "classification"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type # Primarily for info, logic driven by output_dim
        self.logger = logging.getLogger(f"models.{self.__class__.__name__}")
        self.logger.info(f"Creating MLP: input={input_dim}, hidden={hidden_dims}, output={output_dim}, activation={activation}, dropout={dropout_rate}, task={task_type}")

        if not hidden_dims:
             self.logger.warning("MLP created with no hidden layers.")
             # Handle case with no hidden layers: direct linear transformation
             layers = [nn.Linear(input_dim, output_dim)]
             if dropout_rate > 0 :
                  self.logger.warning("Dropout specified but no hidden layers; dropout ignored.")
             self.model = nn.Sequential(*layers)
             return


        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            # Add activation
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            else:
                self.logger.warning(f"Unknown activation '{activation}', defaulting to ReLU.")
                layers.append(nn.ReLU())
            # Add dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        # Final output layer
        layers.append(nn.Linear(current_dim, output_dim))

        # No final activation for CrossEntropyLoss (expects raw logits)
        # No final activation for BCEWithLogitsLoss (expects raw logits)
        # For regression, usually no final activation unless specific range needed.

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Input shape check - crucial! DataHandler should ensure this.
        if x.shape[1] != self.input_dim:
            # Log a critical error, as this indicates a problem upstream (DataHandler/Config)
            self.logger.error(f"CRITICAL: Input dimension mismatch! Expected {self.input_dim}, got {x.shape[1]}. Data pipeline or config error.")
            # Option 1: Raise error (stops execution)
            raise ValueError(f"Input dimension mismatch: Expected {self.input_dim}, got {x.shape[1]}")
            # Option 2: Try to proceed but log error (might lead to nonsensical results)
            # return torch.zeros((x.shape[0], self.output_dim), device=x.device) # Return zeros?

        return self.model(x)


# Example CNN (modify as needed for specific image tasks)
class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""
    def __init__(self, input_channels: int, num_classes: int, input_size_hw: tuple = (28, 28)):
        super().__init__()
        self.logger = logging.getLogger(f"models.{self.__class__.__name__}")
        self.logger.info(f"Creating SimpleCNN: channels={input_channels}, classes={num_classes}, size={input_size_hw}")

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2) # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2) # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2) # 14x14 -> 14x14
        # After pool: 14x14 -> 7x7
        # Calculate flattened size dynamically
        h, w = input_size_hw
        conv_output_size_h = (h // 2) // 2 # Two pools
        conv_output_size_w = (w // 2) // 2
        flattened_size = 64 * conv_output_size_h * conv_output_size_w

        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Raw logits for CrossEntropyLoss
        return x


# Add LSTM or other model types here if needed following similar patterns

def create_model(model_config: ModelConfig, input_shape: List[int], output_shape: List[int]):
    """
    Factory function to create a model based on configuration.

    Args:
        model_config: Model configuration object.
        input_shape: Actual input shape from DataHandler (e.g., [num_features] or [C, H, W]).
        output_shape: Actual output shape from DataHandler (e.g., [num_classes] or [1]).

    Returns:
        An instantiated PyTorch model.
    """
    logger = logging.getLogger("models.factory")

    model_type = model_config.type.lower()
    task_type = model_config.task_type.lower()

    # Determine dimensions from actual data shapes
    input_dim = int(np.prod(input_shape)) # Flatten if needed for MLP/LSTM
    output_dim = output_shape[0] # Usually interested in the first dimension (classes or regression output)

    logger.info(f"Attempting to create model '{model_type}' for task '{task_type}'")
    logger.info(f"Derived Dims: Input={input_dim} (from shape {input_shape}), Output={output_dim} (from shape {output_shape})")
    logger.info(f"Config Settings: Hidden={model_config.hidden_layers}, Activation={model_config.activation}, Dropout={model_config.dropout_rate}")


    if model_type == "mlp":
         # Use MLP for multi-class classification or regression
        if task_type == "binary_classification" and output_dim == 1:
             logger.warning(f"Model type is 'mlp' but task is 'binary_classification' and output_dim is 1. Consider using 'binary_classifier' type or ensure loss function matches.")
             # Proceeding with MLP, assuming BCEWithLogitsLoss will be used.
        elif task_type == "classification" and output_dim <= 1:
              logger.warning(f"Model type is 'mlp' for 'classification', but output_dim is {output_dim}. Expected > 1 classes for multi-class.")
              # Adjust output_dim maybe? Or rely on config/data being correct.
              if output_dim < 2: output_dim = 2 # Force at least 2 outputs for CE loss? Risky.

        return MLP(
            input_dim=input_dim,
            hidden_dims=model_config.hidden_layers,
            output_dim=output_dim, # Should match num_classes or 1 for regression
            dropout_rate=model_config.dropout_rate,
            activation=model_config.activation,
            task_type=task_type
        )
    elif model_type == "binary_classifier":
         # Explicitly for binary classification (typically outputs 1 logit)
         if task_type != "binary_classification":
              logger.warning(f"Model type is 'binary_classifier', but task_type is '{task_type}'. Ensure this is intended.")
         if output_dim != 1:
              logger.warning(f"Model type is 'binary_classifier', but output_dim is {output_dim}. Overriding output_dim to 1.")
              output_dim = 1 # Force output dim to 1

         # Use MLP structure but ensure output_dim is 1
         return MLP(
            input_dim=input_dim,
            hidden_dims=model_config.hidden_layers,
            output_dim=1, # Explicitly 1 for binary classifier using BCEWithLogitsLoss
            dropout_rate=model_config.dropout_rate,
            activation=model_config.activation,
            task_type="binary_classification" # Set task type for clarity
        )

    elif model_type == "cnn":
         if len(input_shape) != 3:
             raise ValueError(f"CNN model type requires 3D input shape [C, H, W], but got {input_shape}")
         input_channels = input_shape[0]
         input_size_hw = (input_shape[1], input_shape[2])
         num_classes = output_dim

         if task_type == "binary_classification" and num_classes != 1:
              logger.warning(f"CNN for binary classification, but output_dim is {num_classes}. Setting num_classes to 1 for BCEWithLogitsLoss.")
              num_classes = 1
         elif task_type == "classification" and num_classes <= 1:
              raise ValueError(f"CNN for multi-class classification requires num_classes > 1, but got {num_classes}")


         # Replace with your actual CNN class if different from SimpleCNN
         return SimpleCNN(
             input_channels=input_channels,
             num_classes=num_classes,
             input_size_hw=input_size_hw
         )

    # Add LSTM or other types here
    # elif model_type == "lstm":
    #     return LSTM(...)

    else:
        logger.error(f"Unknown model type: '{model_type}'. Check configuration.")
        raise ValueError(f"Unsupported model type: {model_type}")