import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Union

from shared.he_utils import HomomorphicEncryption
from shared.utils import flatten_parameters, unflatten_parameters, weights_to_parameters, parameters_to_weights
from shared.model import CloudSecurityModel # Needed for parameter shapes/dtypes

logger = logging.getLogger(__name__)

class SecureAggregationStrategy(FedAvg):
    """
    Custom Flower strategy that handles Homomorphically Encrypted updates.
    Inherits from FedAvg but overrides aggregation functions.
    """
    def __init__(
        self,
        he_manager: HomomorphicEncryption,
        model_template: CloudSecurityModel, # To get shapes/dtypes for unflattening
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.he_manager = he_manager # Contains private key for decryption
        # Store shapes and dtypes from the model template for unflattening aggregated results
        template_params = [p.cpu().numpy() for _, p in model_template.state_dict().items()]
        self.param_shapes = [p.shape for p in template_params]
        self.param_dtypes = [p.dtype for p in template_params]
        self.public_key = he_manager.get_public_key() # Needed for deserialization
        logger.info("SecureAggregationStrategy initialized with HE Manager.")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate encrypted model updates using Homomorphic Encryption.
        Overrides FedAvg.aggregate_fit.
        """
        if not results:
            logger.warning("aggregate_fit received no results. Skipping aggregation.")
            return None, {}
        if failures:
            logger.warning(f"aggregate_fit received {len(failures)} failures.")
            # Potentially handle failures differently here

        logger.info(f"Aggregating encrypted results from {len(results)} clients for round {server_round}...")

        # 1. Deserialize and Decrypt Updates from each client
        deserialized_encrypted_updates_lists = []
        total_num_examples = 0
        processed_client_count = 0

        for client_proxy, fit_res in results:
            # Check if client reported success (optional, based on client implementation)
            if fit_res.metrics.get("status") == "encryption_error":
                 logger.warning(f"Client {client_proxy.cid} reported encryption error: {fit_res.metrics.get('error')}. Skipping its results.")
                 continue
            if fit_res.status.message == "Success" and fit_res.parameters: # Check parameters exist
                try:
                    # Extract the single ndarray containing the bytes object
                    serialized_bytes_ndarray = parameters_to_ndarrays(fit_res.parameters)
                    if not serialized_bytes_ndarray or len(serialized_bytes_ndarray) != 1:
                         logger.warning(f"Received unexpected parameter format from client {client_proxy.cid}. Expected list with 1 ndarray. Skipping.")
                         continue

                    serialized_bytes = serialized_bytes_ndarray[0].item() # Extract bytes object
                    if not isinstance(serialized_bytes, bytes):
                         logger.warning(f"Extracted parameter from client {client_proxy.cid} is not bytes. Type: {type(serialized_bytes)}. Skipping.")
                         continue

                    # Deserialize bytes into list of EncryptedNumber objects
                    encrypted_update_list = self.he_manager.deserialize_encrypted_list(
                        serialized_bytes, self.public_key
                    )
                    deserialized_encrypted_updates_lists.append(encrypted_update_list)
                    total_num_examples += fit_res.num_examples
                    processed_client_count += 1
                    logger.debug(f"Successfully deserialized encrypted updates from client {client_proxy.cid} ({len(encrypted_update_list)} params).")

                except Exception as e:
                    logger.error(f"Error processing result from client {client_proxy.cid}: {e}", exc_info=True)
                    # Optionally add to failures list or just skip
            else:
                 logger.warning(f"Received potentially problematic FitRes from {client_proxy.cid}: Status={fit_res.status.code}, Message='{fit_res.status.message}', Params exist={bool(fit_res.parameters)}")


        if not deserialized_encrypted_updates_lists:
            logger.warning("No valid encrypted updates received after processing results. Cannot aggregate.")
            return None, {}

        logger.info(f"Successfully processed encrypted updates from {processed_client_count} clients.")

        # 2. Homomorphically Aggregate Encrypted Updates (Element-wise Sum)
        try:
            logger.info("Performing homomorphic aggregation (summation)...")
            aggregated_encrypted_updates = self.he_manager.aggregate_encrypted_lists(
                deserialized_encrypted_updates_lists
            )
            logger.info("Homomorphic aggregation successful.")
        except Exception as e:
             logger.error(f"Error during homomorphic aggregation: {e}", exc_info=True)
             return None, {} # Cannot proceed if aggregation fails

        # 3. Decrypt the Aggregated Result (Only the Sum)
        try:
            logger.info("Decrypting aggregated sum...")
            decrypted_aggregated_updates_sum_flat = self.he_manager.decrypt_list(
                aggregated_encrypted_updates
            )
            # Convert list of floats back to a NumPy array
            decrypted_sum_np = np.array(decrypted_aggregated_updates_sum_flat, dtype=np.float64) # Use float64 for precision
            logger.info("Decryption of aggregated sum successful.")
            logger.debug(f"Norm of decrypted sum: {np.linalg.norm(decrypted_sum_np):.4f}")

        except Exception as e:
            logger.error(f"Error during decryption of aggregated updates: {e}", exc_info=True)
            return None, {} # Cannot proceed if decryption fails

        # 4. Average the Decrypted Updates (Weighted averaging could also be done here if weights were collected)
        # Simple averaging: Divide the decrypted sum by the number of clients who contributed
        if processed_client_count > 0:
            averaged_updates_flat = decrypted_sum_np / float(processed_client_count)
            logger.info(f"Averaged decrypted updates over {processed_client_count} clients. Norm: {np.linalg.norm(averaged_updates_flat):.4f}")
        else:
            logger.warning("Processed client count is zero, cannot average. Returning None.")
            return None, {}

        # 5. Update the Global Model
        # Get current global model parameters (from the previous round or initial parameters)
        current_global_parameters_ndarrays = []
        if self.initial_parameters is not None:
             # This assumes self.initial_parameters holds the params from the *start* of the round
             # FedAvg logic updates the central model internally, but we need the parameters *before* this aggregation step.
             # A common pattern is to store the parameters from the *previous* round's aggregation.
             # Let's assume `self.current_weights` holds the latest aggregated weights (as ndarrays).
             # If it's the first round, use initial_parameters.
             if server_round == 1 and self.initial_parameters:
                 current_global_parameters_ndarrays = parameters_to_ndarrays(self.initial_parameters)
                 logger.info("Using initial parameters as base for update.")
             elif hasattr(self, 'current_weights') and self.current_weights is not None:
                 current_global_parameters_ndarrays = self.current_weights
                 logger.info("Using parameters from previous round as base for update.")
             else:
                 logger.error("Cannot find base parameters to apply updates to. Need initial_parameters or previous round's weights.")
                 return None, {} # Cannot update without a base model
        else:
            logger.error("Initial parameters not set in strategy. Cannot update model.")
            return None, {}

        current_global_flat = flatten_parameters(current_global_parameters_ndarrays)

        # Apply the averaged delta: New Global = Old Global + Averaged Delta
        new_global_flat = current_global_flat + averaged_updates_flat
        logger.debug(f"Norm of updated global flat params: {np.linalg.norm(new_global_flat):.4f}")


        # Unflatten the new global parameters
        try:
            new_global_parameters_ndarrays = unflatten_parameters(
                new_global_flat, self.param_shapes, self.param_dtypes
            )
        except Exception as e:
            logger.error(f"Error unflattening aggregated parameters: {e}", exc_info=True)
            return None, {}

        # Store the new weights for the next round (important!)
        self.current_weights = new_global_parameters_ndarrays

        # Convert back to Flower's Parameters format
        new_global_parameters = ndarrays_to_parameters(new_global_parameters_ndarrays)

        logger.info("Aggregation complete. Returning updated global parameters.")

        # Aggregate custom metrics if needed (e.g., average client accuracy from fit_res.metrics)
        metrics_aggregated = {}

        return new_global_parameters, metrics_aggregated

    # aggregate_evaluate can remain as FedAvg's implementation if evaluation doesn't need HE
    # (Clients evaluate normally, server averages standard metrics like loss/accuracy)

# --- Server Setup ---

def run_server(config: Dict, he_manager: HomomorphicEncryption):
    """Configures and runs the Flower server."""

    # Create a template model instance just to get parameter shapes/dtypes
    model_template = CloudSecurityModel(
        num_features=config['model']['num_features'],
        num_hidden_units=config['model']['num_hidden_units'],
        num_classes=config['model']['num_classes']
    )

    # Define the secure strategy
    strategy = SecureAggregationStrategy(
        he_manager=he_manager,
        model_template=model_template,
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0, # Sample 100% for evaluation
        min_fit_clients=config['federated_learning']['min_fit_clients'],
        min_evaluate_clients=config['federated_learning']['min_evaluate_clients'],
        min_available_clients=config['federated_learning']['min_available_clients'],
        # Pass initial parameters if desired (optional, FedAvg can handle it)
        # initial_parameters=ndarrays_to_parameters(get_model_parameters(model_template)),
    )

    logger.info("Starting Flower server...")
    fl.server.start_server(
        server_address=config['federated_learning']['server_address'],
        config=fl.server.ServerConfig(num_rounds=config['federated_learning']['num_rounds']),
        strategy=strategy,
    )
    logger.info("Flower server finished.")