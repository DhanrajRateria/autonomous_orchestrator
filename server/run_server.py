import yaml
import logging
import argparse

from server.server import run_server
from shared.he_utils import HomomorphicEncryption

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Federated Learning Server with HE")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to the configuration file."
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded: {config}")


    # Initialize Homomorphic Encryption
    # IMPORTANT: The server holds the private key for decryption.
    # Key generation can be slow for large key sizes.
    logger.info("Initializing Homomorphic Encryption Manager (Server Side)...")
    he_manager = HomomorphicEncryption(
        key_length=config['homomorphic_encryption']['he_key_length'],
        precision_bits=config['homomorphic_encryption']['he_precision_bits']
    )
    logger.info("HE Manager initialized.")

    # Start the server
    try:
        run_server(config, he_manager)
    except Exception as e:
        logger.error(f"Server execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()