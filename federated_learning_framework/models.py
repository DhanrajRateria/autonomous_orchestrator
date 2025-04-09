import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Union, Optional

from federated_learning_framework.config import ModelConfig

class MLP(nn.Module):
    """Multi-layer perceptron model for tabular data"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                dropout_rate: float = 0.2, activation: str = "relu"):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            
            # Add activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())  # Default to ReLU
            
            # Add dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = dim
        
        # Output layer
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x):
        x = self.layers(x)
        return self.output_layer(x)

class CNN(nn.Module):
    """Convolutional neural network for image data"""
    
    def __init__(self, input_channels: int, input_size: List[int], 
                 num_classes: int, dropout_rate: float = 0.2):
        super(CNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calculate size after convolutions and pooling
        h, w = input_size
        h = h // 8  # Three 2x2 max poolings
        w = w // 8
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * h * w, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class LSTM(nn.Module):
    """LSTM model for sequential data"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout_rate: float = 0.2):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        x = lstm_out[:, -1, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Final classification layer
        x = self.fc(x)
        
        return x

class BinaryClassifier(nn.Module):
    """Binary classifier for cancer detection use case"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float = 0.2):
        super(BinaryClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # Output layer for binary classification
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def create_model(model_config: ModelConfig, input_shape: List[int], output_shape: List[int]) -> nn.Module:
    """
    Factory function to create a model based on configuration.
    
    Args:
        model_config: Model configuration
        input_shape: Input shape
        output_shape: Output shape
        
    Returns:
        Model instance
    """
    model_type = model_config.type.lower()
    
    if model_type == "mlp":
        input_dim = input_shape[0]
        output_dim = output_shape[0]
        
        return MLP(
            input_dim=input_dim,
            hidden_dims=model_config.hidden_layers,
            output_dim=output_dim,
            dropout_rate=model_config.dropout_rate,
            activation=model_config.activation
        )
    
    elif model_type == "cnn":
        # For image data: [channels, height, width]
        if len(input_shape) == 3:
            input_channels = input_shape[0]
            input_size = input_shape[1:]
        else:
            input_channels = 1
            input_size = input_shape
        
        output_dim = output_shape[0]
        
        return CNN(
            input_channels=input_channels,
            input_size=input_size,
            num_classes=output_dim,
            dropout_rate=model_config.dropout_rate
        )
    
    elif model_type == "lstm":
        # For sequential data: [seq_len, features]
        input_dim = input_shape[-1]
        output_dim = output_shape[0]
        
        return LSTM(
            input_dim=input_dim,
            hidden_dim=model_config.hidden_layers[0] if model_config.hidden_layers else 64,
            output_dim=output_dim,
            num_layers=len(model_config.hidden_layers) if model_config.hidden_layers else 2,
            dropout_rate=model_config.dropout_rate
        )
    
    elif model_type == "binary_classifier":
        input_dim = input_shape[0]
        
        return BinaryClassifier(
            input_dim=input_dim,
            hidden_dims=model_config.hidden_layers,
            dropout_rate=model_config.dropout_rate
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")