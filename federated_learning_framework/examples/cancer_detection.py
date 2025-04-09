import asyncio
import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Add parent directory to path to import framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from federated_learning_framework.config import FrameworkConfig
from federated_learning_framework.server import FederatedServer
from federated_learning_framework.client import FederatedClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("cancer_detection")

async def prepare_data(num_clients: int = 5):
    """
    Prepare and split the Wisconsin Breast Cancer dataset for federated learning.
    
    Args:
        num_clients: Number of clients to simulate
        
    Returns:
        Dictionary of client data paths
    """
    # Create data directory if not exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Load breast cancer dataset
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    
    # Convert to dataframe
    df = pd.DataFrame(X, columns=cancer.feature_names)
    df['diagnosis'] = y
    
    # Save full dataset for server validation
    full_path = data_dir / "wisconsin_breast_cancer.csv"
    df.to_csv(full_path, index=False)
    logger.info(f"Saved complete dataset to {full_path}")
    
    # Split data for each client (non-IID)
    # For demonstration, we create data with different class distributions for each client
    client_data_paths = {}
    
    # Split patients with and without cancer
    cancer_df = df[df['diagnosis'] == 1]
    non_cancer_df = df[df['diagnosis'] == 0]
    
    for i in range(1, num_clients + 1):
        client_id = f"client_{i}"
        
        # Create different ratios of cancer/non-cancer data for each client
        # to simulate non-IID data distribution
        cancer_ratio = 0.2 + (i - 1) * 0.1  # Ranges from 0.2 to 0.6
        
        # Calculate samples of each class for this client
        total_samples = len(df) // num_clients
        cancer_samples = int(total_samples * cancer_ratio)
        non_cancer_samples = total_samples - cancer_samples
        
        # Sample from each class
        client_cancer = cancer_df.sample(cancer_samples)
        client_non_cancer = non_cancer_df.sample(non_cancer_samples)
        
        # Combine and shuffle
        client_df = pd.concat([client_cancer, client_non_cancer]).sample(frac=1)
        
        # Save client data
        client_path = data_dir / f"{client_id}_data.csv"
        client_df.to_csv(client_path, index=False)
        client_data_paths[client_id] = client_path
        
        logger.info(f"Created dataset for {client_id}: {len(client_df)} samples "
                   f"({cancer_samples} cancer, {non_cancer_samples} non-cancer)")
    
    return client_data_paths

async def run_federated_learning(config_path: str, num_clients: int = 5):
    """
    Run the federated learning process for cancer detection.
    
    Args:
        config_path: Path to configuration file
        num_clients: Number of clients to simulate
    """
    # Load configuration
    config = FrameworkConfig.from_file(config_path)
    
    # Create data for clients
    client_data_paths = await prepare_data(num_clients)
    
    # Initialize server
    server = FederatedServer(config)
    await server.start()
    
    # Initialize clients
    clients = {}
    for client_id, data_path in client_data_paths.items():
        client = FederatedClient(client_id, config, str(data_path))
        await client.initialize()
        clients[client_id] = client
        
        # Register with server
        await server.register_client(client_id, client.get_client_info())
    
    logger.info(f"Initialized {len(clients)} federated learning clients")
    
    # Run for specified number of rounds
    for round_id in range(1, config.federated.communication_rounds + 1):
        logger.info(f"Starting federated round {round_id}")
        
        # Start round on server
        round_config = await server.start_round()
        if round_config is None:
            logger.info("No more rounds to execute")
            break
        
        # Get selected clients
        selected_clients = round_config["selected_clients"]
        if not selected_clients:
            logger.warning("No clients selected for this round")
            continue
        
        logger.info(f"Selected clients for round {round_id}: {selected_clients}")
        
        # Train on each selected client
        for client_id in selected_clients:
            client = clients[client_id]
            
            # Train local model
            logger.info(f"Training on client {client_id}")
            result = await client.train(
                round_id=round_id,
                parameters=round_config["parameters"],
                encrypted=round_config["encrypted"],
                epochs=round_config["config"]["local_epochs"],
                learning_rate=round_config["config"]["learning_rate"]
            )
            
            # Submit update to server
            if result["status"] == "success":
                logger.info(f"Client {client_id} training complete: "
                           f"loss={result['train_loss']:.4f}, "
                           f"accuracy={result.get('train_accuracy', 0):.4f}")
                
                await server.submit_update(client_id, round_id, result)
            else:
                logger.error(f"Client {client_id} training failed: {result.get('message')}")
    
    # Evaluate final model
    test_metrics = await server.get_test_metrics()
    logger.info(f"Final model test metrics: {test_metrics}")
    
    # Stop server
    await server.stop()
    
    logger.info("Federated learning complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning for Cancer Detection")
    parser.add_argument("--config", default="cancer_detection_config.yaml", help="Path to configuration file")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients to simulate")
    
    args = parser.parse_args()
    
    # Run the federated learning process
    asyncio.run(run_federated_learning(args.config, args.clients))