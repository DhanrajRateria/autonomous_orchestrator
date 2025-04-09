#!/usr/bin/env python3
import asyncio
import argparse
import logging
import os
import sys
import signal
import yaml
from pathlib import Path

from core.orchestrator import SecurityOrchestrator
from core.config import OrchestrationConfig
from api.server import APIServer

def setup_logging(log_level: str, log_file: str = None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Path to log file (if None, log to console)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Configure file handler
        logging.basicConfig(
            level=numeric_level,
            format=log_format,
            filename=log_file,
            filemode='a'
        )
        
        # Add console handler as well
        console = logging.StreamHandler()
        console.setLevel(numeric_level)
        console.setFormatter(logging.Formatter(log_format))
        logging.getLogger('').addHandler(console)
    else:
        # Log to console only
        logging.basicConfig(
            level=numeric_level,
            format=log_format
        )
    
    # Set lower level for some noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    
    logger = logging.getLogger()
    logger.info(f"Logging initialized at level {log_level}")

async def run_orchestrator(config_path: str, api_enabled: bool = True):
    """
    Run the security orchestrator.
    
    Args:
        config_path: Path to configuration file
        api_enabled: Whether to enable the API server
    """
    logger = logging.getLogger("main")
    logger.info(f"Starting Cloud Security Orchestrator with config: {config_path}")
    
    # Create default config if not exists
    if not os.path.exists(config_path):
        logger.info(f"Config file {config_path} not found, creating default configuration")
        config_dir = os.path.dirname(config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir)
        
        # Create a minimal default config
        config = {
            "instance_id": "default-instance",
            "use_encryption_for_analysis": True,
            "event_polling_interval": 30,
            "model_update_interval": 3600,
            "detection_threshold": 0.7,
            "response_threshold": 0.85,
            "crypto_settings": {
                "key_size": 2048,
                "security_level": 128
            },
            "federated_learning": {
                "min_clients": 2,
                "aggregation_method": "secure_fedavg",
                "rounds_per_update": 5
            },
            "cloud_providers": [
                {
                    "provider_id": "default-aws",
                    "provider_type": "aws",
                    "credentials_path": "~/.aws/credentials",
                    "region": "us-west-1",
                    "enabled_services": ["cloudtrail", "guardduty", "securityhub"]
                }
            ]
        }
        
        # Write the config
        with open(config_path, 'w') as f:
            if config_path.endswith('.json'):
                import json
                json.dump(config, f, indent=2)
            else:
                yaml.dump(config, f)
    
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    orchestrator = None
    api_server = None
    
    try:
        # Set up signal handlers
        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
        for s in signals:
            loop.add_signal_handler(
                s, lambda s=s: asyncio.create_task(shutdown(loop, orchestrator, api_server, s))
            )
        
        if api_enabled:
            # Start with API server
            logger.info("Starting with API server enabled")
            api_server = APIServer(config_path)
            await api_server.start_server()
        else:
            # Start without API server
            logger.info("Starting in direct mode without API")
            config = OrchestrationConfig.from_file(config_path)
            orchestrator = SecurityOrchestrator(config)
            
            # Initialize and start orchestrator
            logger.info("Initializing cloud connectors")
            await orchestrator.initialize_cloud_connectors()
            
            logger.info("Starting orchestrator")
            await orchestrator.start()
            
            # Keep running until terminated
            logger.info("Orchestrator running, press Ctrl+C to stop")
            while True:
                await asyncio.sleep(3600)  # Sleep for an hour
    
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        # Make sure we always clean up
        await shutdown(loop, orchestrator, api_server)

async def shutdown(loop, orchestrator, api_server, signal=None):
    """
    Shutdown the orchestrator and API server gracefully.
    
    Args:
        loop: Event loop
        orchestrator: Security orchestrator instance
        api_server: API server instance
        signal: Signal that triggered the shutdown
    """
    logger = logging.getLogger("main")
    
    if signal:
        logger.info(f"Received exit signal {signal.name}")
    
    logger.info("Shutting down")
    
    # Stop orchestrator if running
    if orchestrator and orchestrator.is_running:
        logger.info("Stopping orchestrator")
        await orchestrator.stop()
    
    # Stop API server if running
    if api_server:
        logger.info("API server shutdown would happen here")
    
    # Finish all pending tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    
    if tasks:
        logger.info(f"Waiting for {len(tasks)} pending tasks to complete")
        await asyncio.gather(*tasks, return_exceptions=True)
    
    # Stop the event loop
    loop.stop()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Autonomous Cloud Security Orchestrator")
    parser.add_argument(
        "--config", 
        default="configs/orchestrator.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file (logs to console if not specified)"
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Disable API server"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    
    # Run the orchestrator
    try:
        asyncio.run(run_orchestrator(args.config, not args.no_api))
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        logging.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)