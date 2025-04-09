import os
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import torch

def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/federated_learning.log"),
            logging.StreamHandler()
        ]
    )

def plot_training_history(history_path: str, output_dir: str = "plots"):
    """
    Plot training metrics from history file.
    
    Args:
        history_path: Path to training history JSON
        output_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load history data
    with open(history_path, 'r') as f:
        history_data = json.load(f)
    
    history = history_data["history"]
    
    # Extract metrics
    rounds = [entry["round"] for entry in history]
    accuracy = [entry["metrics"].get("accuracy", 0) for entry in history]
    loss = [entry["metrics"].get("loss", 0) for entry in history]
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracy, 'b-', label='Validation Accuracy')
    plt.title('Model Accuracy over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/accuracy.png")
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, loss, 'r-', label='Validation Loss')
    plt.title('Model Loss over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss.png")
    
    # Plot client metrics if available
    if history[0].get("client_metrics"):
        # Get unique client IDs
        client_ids = set()
        for entry in history:
            client_ids.update(entry.get("client_metrics", {}).keys())
        
        # Plot metrics for each client
        client_colors = plt.cm.tab10(np.linspace(0, 1, len(client_ids)))
        
        plt.figure(figsize=(12, 8))
        for idx, client_id in enumerate(sorted(client_ids)):
            client_acc = []
            client_rounds = []
            
            for entry in history:
                client_metrics = entry.get("client_metrics", {}).get(client_id)
                if client_metrics and "train_accuracy" in client_metrics:
                    client_acc.append(client_metrics["train_accuracy"])
                    client_rounds.append(entry["round"])
            
            if client_rounds:
                plt.plot(client_rounds, client_acc, 
                         color=client_colors[idx], 
                         label=f"{client_id} Accuracy",
                         marker='o', 
                         linestyle='-', 
                         alpha=0.7)
        
        plt.title('Client Training Accuracy')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/client_accuracy.png")

def compare_models(model_paths: List[str], dataset_path: str, output_file: str = None):
    """
    Compare multiple trained models on the same dataset.
    
    Args:
        model_paths: List of paths to model checkpoints
        dataset_path: Path to test dataset
        output_file: Path to save comparison results
    """
    from federated_learning_framework.data_handler import DataHandler
    from federated_learning_framework.config import DataConfig
    
    results = {}
    
    # Load dataset
    data_config = DataConfig(
        input_shape=[30],  # For cancer dataset
        output_shape=[2],
        data_path=dataset_path,
        normalize=True
    )
    
    data_handler = DataHandler(data_config)
    _, _, test_loader = data_handler.load_data(test_split=1.0)  # Use all data as test
    
    # Evaluate each model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for model_path in model_paths:
        try:
            # Load model
            checkpoint = torch.load(model_path, map_location=device)
            model_name = Path(model_path).stem
            
            # Create a new model and load weights
            from federated_learning_framework.models import create_model
            from federated_learning_framework.config import ModelConfig
            
            model_config = ModelConfig(
                type="binary_classifier",
                hidden_layers=[64, 32]
            )
            
            model = create_model(model_config, [30], [2])
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            
            # Evaluate model
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(inputs)
                    predicted = (outputs > 0.5).float()
                    
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            accuracy = correct / total
            
            results[model_name] = {
                "accuracy": accuracy,
                "round": checkpoint.get("round", "unknown"),
                "path": model_path
            }
            
            print(f"Model: {model_name}, Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error evaluating model {model_path}: {e}")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results