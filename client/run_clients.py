import flwr as fl
import torch
import yaml
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

from client.client import HomomorphicFederatedClient
from shared.model import CloudSecurityModel
from shared.data_utils import load_partitioned_data
from shared.he_utils import HomomorphicEncryption

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Client %(client_id)s] - [%(filename)s:%(lineno)d] - %(message)s'
)
# Use a separate logger instance to avoid conflicts if run in same process
client_logger = logging.getLogger("client_runner")

# Adapter to add client_id to log records
class ClientLogAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if 'client_id' not in self.extra:
            self.extra['client_id'] = 'N/A'
        return '[Client %s] %s' % (self.extra['client_id'], msg), kwargs

def get_client_logger(client_id):
     adapter = ClientLogAdapter(logging.getLogger(__name__), {'client_id': client_id})
     return adapter


def fit_client(client_id: int, config: dict, he_manager: HomomorphicEncryption, data_loaders: tuple):
    """Creates and runs a single Flower client."""
    client_id_str = str(client_id)
    logger = get_client_logger(client_id_str) # Use adapter for logging
    logger.info(f"Initializing client {client_id_str}")

    # Load data for this client
    trainloader, testloader = data_loaders

    # Instantiate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = CloudSecurityModel(
        num_features=config['model']['num_features'],
        num_hidden_units=config['model']['num_hidden_units'],
        num_classes=config['model']['num_classes']
    )

    # Instantiate the Flower client
    client = HomomorphicFederatedClient(
        cid=client_id_str,
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        he_manager=he_manager, # Client only needs public key (implicitly via manager)
        local_epochs=config['client']['local_epochs'],
        lr=config['client']['learning_rate'],
        device=device
    )

    # Start the client
    logger.info(f"Starting Flower client {client_id_str} connecting to {config['federated_learning']['server_address']}")
    try:
        fl.client.start_numpy_client(
            server_address=config['federated_learning']['server_address'],
            client=client,
            # Add root certificates if using TLS/SSL
            # root_certificates=Path(".cache/certificates/ca.crt").read_bytes()
        )
        logger.info(f"Client {client_id_str} finished.")
    except ConnectionRefusedError:
        logger.error(f"Client {client_id_str}: Connection refused by the server. Is the server running at {config['federated_learning']['server_address']}?")
    except Exception as e:
        logger.error(f"Client {client_id_str} failed: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Run Federated Learning Clients with HE")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to the configuration file."
    )
    parser.add_argument(
        "--num_clients", type=int, default=None, help="Number of clients to simulate (overrides config)."
    )
    parser.add_argument(
        "--executor", type=str, default="thread", choices=["thread", "process"], help="Executor type (thread or process)."
    )
    args = parser.parse_args()

    # Load configuration
    client_logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    num_clients = args.num_clients if args.num_clients is not None else config['federated_learning']['num_clients']
    client_logger.info(f"Simulating {num_clients} clients using {args.executor} executor.")

    # Initialize Homomorphic Encryption (only need one instance with public key for all clients)
    # NOTE: In a real deployment, the public key would be distributed securely,
    # and clients wouldn't generate keys. Here, we create one manager
    # and implicitly share the public key. The private key isn't needed by clients.
    client_logger.info("Initializing Homomorphic Encryption Manager (Client Side - Public Key)...")
    # We create the full manager, but clients only use encryption (public key ops)
    he_manager = HomomorphicEncryption(
        key_length=config['homomorphic_encryption']['he_key_length'],
        precision_bits=config['homomorphic_encryption']['he_precision_bits']
    )
    client_logger.info("HE Manager initialized for clients.")


    # Load and partition data for all clients
    client_logger.info("Loading and partitioning data...")
    all_client_loaders = load_partitioned_data(
        num_clients=num_clients,
        num_samples_per_client=config['data']['num_samples_per_client'],
        num_features=config['model']['num_features'],
        num_classes=config['model']['num_classes'],
        skewness=config['data']['data_skewness'],
        batch_size=config['client']['batch_size']
    )
    client_logger.info("Data loaded and partitioned.")


    # --- System Programming Aspect: Parallel Client Execution ---
    # Using ThreadPoolExecutor or ProcessPoolExecutor to run clients concurrently.
    # ProcessPoolExecutor provides true parallelism but has higher overhead due to
    # inter-process communication and pickling. Good if clients are CPU-bound (like HE).
    # ThreadPoolExecutor is lighter but subject to Python's Global Interpreter Lock (GIL),
    # better if clients spend time waiting (I/O, network).
    # For CPU-intensive HE, ProcessPoolExecutor might be better, but pickling
    # complex objects (like HE manager or model) can be problematic.
    # Let's stick with ThreadPoolExecutor for simplicity here, acknowledging the GIL limitation.
    # If using ProcessPoolExecutor, ensure all args to fit_client are pickleable.
    # The HE Manager object itself might not be easily pickleable across processes.
    # A common pattern is to initialize HE within the target function if using processes.
    # For demonstration, we pass the manager assuming threads or careful pickling.

    Executor = ThreadPoolExecutor if args.executor == "thread" else ProcessPoolExecutor

    with Executor(max_workers=num_clients) as executor:
        futures = []
        for i in range(num_clients):
            # Pass the necessary arguments for the client function
            # Ensure data loaders list matches the number of clients requested
            if i < len(all_client_loaders):
                 future = executor.submit(fit_client, i, config, he_manager, all_client_loaders[i])
                 futures.append(future)
            else:
                 client_logger.warning(f"Not enough data loaders generated for client {i}. Skipping.")
            time.sleep(1) # Small delay to avoid overwhelming the server immediately

        # Wait for all clients to complete (optional)
        for future in futures:
            try:
                future.result() # Retrieve result or raise exception if client failed
            except Exception as e:
                client_logger.error(f"A client process/thread raised an exception: {e}", exc_info=True)

    client_logger.info("All client simulations finished.")


if __name__ == "__main__":
    main()