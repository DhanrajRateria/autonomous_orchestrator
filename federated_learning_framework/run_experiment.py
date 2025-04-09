import asyncio
import argparse
import logging
import os
import json
from datetime import datetime
from pathlib import Path

from federated_learning_framework.config import FrameworkConfig
from federated_learning_framework.server import FederatedServer
from federated_learning_framework.client import FederatedClient
from federated_learning_framework.utils import setup_logging, plot_training_history

async def run_experiment(config_path: str, experiment_name: str = None, num_clients: int = 5):
    """
    Run a federated learning experiment with the given configuration.
    
    Args:
        config_path: Path to configuration file
        experiment_name: Name for this experiment run
        num_clients: Number of clients to simulate
    """
    # Create experiment name if not provided
    if experiment_name is None:
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set up directories
    base_dir = Path(experiment_name)
    base_dir.mkdir(exist_ok=True)
    
    # Set up logging
    setup_logging(log_dir=str(base_dir / "logs"))
    logger = logging.getLogger("experiment")
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Configuration: {config_path}")
    
    # Load configuration
    config = FrameworkConfig.from_file(config_path)
    
    # Override checkpoint and result directories
    config.system.checkpoint_dir = str(base_dir / "checkpoints")
    config.system.result_dir = str(base_dir / "results")
    
    # Save experiment configuration
    os.makedirs(config.system.checkpoint_dir, exist_ok=True)
    os.makedirs(config.system.result_dir, exist_ok=True)
    config.save(str(base_dir / "experiment_config.json"))
    
    # Initialize server
    server = FederatedServer(config)
    await server.start()
    
    # Generate synthetic client data if needed and initialize clients
    client_data_paths = await generate_client_data(config, num_clients)
    
    clients = {}
    for client_id, data_path in client_data_paths.items():
        client = FederatedClient(client_id, config, data_path)
        success = await client.initialize()
        if success:
            clients[client_id] = client
            
            # Register with server
            await server.register_client(client_id, client.get_client_info())
    
    logger.info(f"Initialized {len(clients)} federated learning clients")
    
    try:
        # Run for specified number of rounds
        for round_id in range(1, config.federated.communication_rounds + 1):
            logger.info(f"Starting federated round {round_id}/{config.federated.communication_rounds}")
            
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
        
        # Save experiment results summary
        with open(str(base_dir / "experiment_summary.json"), "w") as f:
            json.dump({
                "experiment_name": experiment_name,
                "date": datetime.now().isoformat(),
                "final_metrics": test_metrics,
                "config_path": config_path,
                "num_clients": len(clients),
                "rounds_completed": server.current_round
            }, f, indent=2)

    finally:
        # Stop server
        await server.stop()
        
        # Generate plots
        history_path = Path(config.system.result_dir) / "training_history.json"
        if history_path.exists():
            plot_training_history(str(history_path), str(base_dir / "plots"))
        
        logger.info("Experiment complete!")
        return str(base_dir)

async def generate_client_data(config: FrameworkConfig, num_clients: int) -> dict:
    """
    Generate synthetic client data.
    This is a placeholder that should be customized based on the dataset.
    
    Args:
        config: Framework configuration
        num_clients: Number of clients to generate data for
        
    Returns:
        Dictionary mapping client IDs to data paths
    """
    # This function should be customized based on the specific dataset
    # For now, we'll just split the data path from the config among clients
    
    base_data_path = config.data.data_path
    if not base_data_path or not os.path.exists(base_data_path):
        raise ValueError(f"Data path not found: {base_data_path}")
    
    # For this example, assume all clients use the same dataset
    # In a real application, we would split the data in a non-IID fashion
    client_data_paths = {}
    for i in range(1, num_clients + 1):
        client_id = f"client_{i}"
        client_data_paths[client_id] = base_data_path
    
    return client_data_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiment")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--name", help="Experiment name")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    
    args = parser.parse_args()
    
    # Run the experiment
    asyncio.run(run_experiment(args.config, args.name, args.clients))