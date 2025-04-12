#!/usr/bin/env python
import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("generate_config")

def analyze_dataset(data_path: str):
    """Analyze dataset and generate a basic FL configuration."""
    logger.info(f"Analyzing dataset: {data_path}")
    if not Path(data_path).is_file():
        logger.error(f"Dataset path is not a file: {data_path}")
        return None
    try:
        df = pd.read_csv(data_path)
        if df.empty:
             logger.error("Dataset CSV is empty.")
             return None
    except Exception as e:
        logger.error(f"Failed to load or read CSV: {e}", exc_info=True)
        return None

    rows, cols = df.shape
    logger.info(f"Dataset dimensions: {rows} rows x {cols} columns")

    # --- Target Column Identification ---
    potential_targets = ['diagnosis', 'target', 'label', 'class', 'y']
    target_col = None
    for pt in potential_targets:
        if pt in df.columns:
            target_col = pt
            logger.info(f"Found potential target column: '{target_col}'")
            break
    if not target_col:
        target_col = df.columns[-1]
        logger.warning(f"No common target column name found. Assuming last column '{target_col}' is the target.")

    feature_cols = [col for col in df.columns if col != target_col]
    if not feature_cols:
         logger.error("Could not identify any feature columns.")
         return None

    logger.info(f"Identified {len(feature_cols)} feature columns.") # Feature names logged later if needed
    # Initial input dim based on original columns - DataHandler will adjust if one-hot encoding happens
    input_dim = len(feature_cols)

    # --- Task Type and Output Dimension Detection ---
    target_series = df[target_col]
    unique_targets = target_series.unique()
    num_unique = len(unique_targets)
    logger.info(f"Target column '{target_col}' unique values ({num_unique}): {unique_targets[:10]}...") # Show first 10

    task_type = "unknown"
    output_dim = 0

    if pd.api.types.is_numeric_dtype(target_series):
        # Could be regression or classification with numeric labels
        if num_unique == 2:
            task_type = "binary_classification"
            output_dim = 1 # Single output neuron for BCEWithLogitsLoss
            logger.info("Detected task type: Binary Classification (numeric labels)")
        elif 2 < num_unique <= 15: # Arbitrary threshold for multi-class
             # Check if they look like integer classes
             if np.all(np.equal(np.mod(target_series.dropna(), 1), 0)):
                 task_type = "classification"
                 output_dim = num_unique # N output neurons for CrossEntropyLoss
                 logger.info(f"Detected task type: Multi-Class Classification ({num_unique} classes)")
             else:
                 # Numeric but not integer-like, assume regression
                 task_type = "regression"
                 output_dim = 1 # Single output neuron
                 logger.info("Detected task type: Regression (non-integer numeric target)")
        else: # Many unique numeric values -> likely regression
            task_type = "regression"
            output_dim = 1
            logger.info("Detected task type: Regression (many unique numeric values)")
    elif pd.api.types.is_categorical_dtype(target_series) or target_series.dtype == 'object':
         # String/Object/Categorical types -> Classification
         if num_unique == 2:
            task_type = "binary_classification"
            output_dim = 1
            logger.info("Detected task type: Binary Classification (non-numeric labels)")
         elif 2 < num_unique <= 50: # Allow more classes for non-numeric
             task_type = "classification"
             output_dim = num_unique
             logger.info(f"Detected task type: Multi-Class Classification ({num_unique} classes)")
         else:
             logger.warning(f"Target column '{target_col}' has very high cardinality ({num_unique}). Treating as classification, but verify.")
             task_type = "classification"
             output_dim = num_unique # Risky, might need adjustment
    else:
         logger.error(f"Could not determine task type for target column '{target_col}' with dtype {target_series.dtype}")
         return None


    # --- Create Base Configuration ---
    # Use sensible defaults, disable advanced features initially
    project_name = f"fl_{Path(data_path).stem}"
    config = {
        "project_name": project_name,
        "system": {
            "seed": 42,
            "device": "cpu", # Default to CPU for broader compatibility
            "checkpoint_dir": f"checkpoints/{project_name}",
            "result_dir": f"results/{project_name}",
            "log_level": "INFO",
            "num_workers": 0,
            "checkpoint_frequency": 10 # Save checkpoint every 10 rounds
        },
        "data": {
            "data_path": os.path.abspath(data_path), # Store absolute path
            "feature_columns": feature_cols, # List original features
            "target_column": target_col,
            "input_shape": [input_dim], # Initial guess, may change
            "output_shape": [output_dim],
            "normalize": True,
            "batch_size": 32, # Adjusted batch size
            "val_split": 0.15,
            "test_split": 0.15,
            "preprocessing_steps": [], # Keep empty for now
        },
        "model": {
            "type": "mlp" if task_type != "binary_classification" else "binary_classifier", # Select model based on task
            "task_type": task_type, # Explicitly store detected task type
            "hidden_layers": [64, 32],
            "activation": "relu",
            "dropout_rate": 0.3, # Slightly increased dropout
            "optimizer": "adam", # Default, can be overridden
            "learning_rate": 0.001, # Default, can be overridden
            "loss": "cross_entropy" if task_type=="classification" else ("binary_cross_entropy" if task_type=="binary_classification" else "mse"), # Guess loss
            "metrics": ["accuracy"] if task_type != "regression" else [],
            "l2_regularization": 0.0
        },
        "federated": {
            "communication_rounds": 20, # Reduced default rounds
            "clients_per_round": 2,
            "min_clients": 2,
            "local_epochs": 3, # Increased local epochs slightly
            "client_learning_rate": 0.01,
            "server_learning_rate": 1.0, # Standard for FedAvg
            "aggregation_method": "fedavg",
            "proximal_mu": 0.0 # Keep FedProx term, but 0
        },
        "crypto": {
            "enabled": False, # DISABLED BY DEFAULT
            "scheme": "CKKS",
            "poly_modulus_degree": 8192,
            "coeff_mod_bit_sizes": [40, 20, 40],
            "global_scale": 2**40,
            "security_level": 128
        },
        "privacy": {
            "differential_privacy": False, # DISABLED BY DEFAULT
            "dp_epsilon": 3.0,
            "dp_delta": 1e-5,
            "dp_noise_multiplier": 1.1,
            "gradient_clipping": 1.0,
            "secure_aggregation": False # Placeholder, HE provides this implicitly
        }
    }

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a base Federated Learning config from a CSV dataset.")
    parser.add_argument("--data", required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--output", default="generated_config.yaml", help="Output config file name (YAML).")

    args = parser.parse_args()

    # Ensure output path exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated_config = analyze_dataset(args.data)

    if generated_config:
        try:
            with open(args.output, "w") as f:
                yaml.dump(generated_config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Configuration successfully generated and saved to {args.output}")
            print(f"\nConfiguration saved to: {args.output}")
            print("\nReview the generated configuration, especially:")
            print(" - data: input_shape, output_shape, feature_columns, target_column")
            print(" - model: type, task_type, hidden_layers")
            print("You may need to adjust parameters based on your specific needs.")
            print("\nTo run with this config (example):")
            print(f"python cancer_detection.py --config {args.output} --clients 5")
        except Exception as e:
            logger.error(f"Failed to save configuration to {args.output}: {e}", exc_info=True)

    else:
        logger.error("Configuration generation failed.")