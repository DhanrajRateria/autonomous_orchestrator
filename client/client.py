import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
import logging
from typing import List, Tuple, Dict, Any

from shared.model import CloudSecurityModel, get_model_parameters, set_model_parameters
from shared.he_utils import HomomorphicEncryption
from shared.utils import flatten_parameters, unflatten_parameters

logger = logging.getLogger(__name__)

class HomomorphicFederatedClient(fl.client.NumPyClient):
    """
    Flower client implementation using Homomorphic Encryption for updates.
    """
    def __init__(self,
                 cid: str,
                 model: CloudSecurityModel,
                 trainloader: DataLoader,
                 testloader: DataLoader,
                 he_manager: HomomorphicEncryption,
                 local_epochs: int,
                 lr: float,
                 device: torch.device):
        self.cid = cid
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.he_manager = he_manager # Contains public key
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.original_shapes = [p.shape for p in get_model_parameters(self.model)]
        self.original_dtypes = [p.dtype for p in get_model_parameters(self.model)]

        logger.info(f"[Client {self.cid}] Initialized.")

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Return the current local model parameters."""
        logger.debug(f"[Client {self.cid}] get_parameters called.")
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]):
        """Set the local model parameters."""
        logger.debug(f"[Client {self.cid}] set_parameters called.")
        set_model_parameters(self.model, parameters)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model locally, calculate updates, encrypt updates, and return them.
        """
        logger.info(f"[Client {self.cid}] Starting local training (fit)...")
        self.set_parameters(parameters) # Update model with global parameters

        # --- Standard Training Loop ---
        self.model.train()
        initial_params_flat = flatten_parameters(get_model_parameters(self.model))

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            for features, labels in self.trainloader:
                features, labels = features.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            logger.debug(f"[Client {self.cid}] Epoch {epoch+1}/{self.local_epochs}, Loss: {avg_epoch_loss:.4f}")

        # --- Calculate Parameter Updates (Delta) ---
        final_params = get_model_parameters(self.model)
        final_params_flat = flatten_parameters(final_params)
        # Delta = Final Parameters - Initial Parameters
        param_updates_flat = final_params_flat - initial_params_flat
        logger.info(f"[Client {self.cid}] Calculated parameter updates (delta). Norm: {np.linalg.norm(param_updates_flat):.4f}")

        # --- Homomorphic Encryption of Updates ---
        try:
            logger.info(f"[Client {self.cid}] Encrypting {len(param_updates_flat)} parameter updates...")
            # Convert updates to list of floats for encryption
            updates_list = param_updates_flat.tolist()
            encrypted_updates = self.he_manager.encrypt_list(updates_list)
            logger.info(f"[Client {self.cid}] Encryption successful.")

            # --- Serialization for Transport ---
            # Serialize the list of EncryptedNumber objects into bytes
            serialized_encrypted_updates = self.he_manager.serialize_encrypted_list(encrypted_updates)
            logger.info(f"[Client {self.cid}] Serialized encrypted updates (Size: {len(serialized_encrypted_updates)} bytes).")

            # Flower expects List[np.ndarray]. We wrap the serialized bytes in a NumPy array.
            # This is a workaround. Ideally, Flower would support custom serializers.
            # We send a list containing ONE ndarray, which holds the bytes object.
            parameters_serializable = [np.array(serialized_encrypted_updates, dtype=object)]

        except Exception as e:
            logger.error(f"[Client {self.cid}] Error during encryption or serialization: {e}", exc_info=True)
            # Return empty list or handle error appropriately
            # Returning the unencrypted updates here would violate privacy.
            # For robustness, we should signal an error. Flower doesn't have a built-in
            # way to signal failure other than raising an exception, which might halt the round.
            # Sending empty results might be handled by the server strategy.
            return [], len(self.trainloader.dataset), {"status": "encryption_error", "error": str(e)}


        num_examples_train = len(self.trainloader.dataset)
        # Return the *serialized encrypted updates* and the number of training examples
        return parameters_serializable, num_examples_train, {"status": "success"}


    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict]:
        """Evaluate the model locally."""
        logger.info(f"[Client {self.cid}] Starting local evaluation (evaluate)...")
        self.set_parameters(parameters) # Use the updated global model for evaluation

        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in self.testloader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        num_examples_test = len(self.testloader.dataset)
        if num_examples_test == 0:
             logger.warning(f"[Client {self.cid}] Test dataset is empty.")
             return 0.0, 0, {"accuracy": 0.0}

        average_loss = loss / len(self.testloader)
        accuracy = correct / total
        logger.info(f"[Client {self.cid}] Evaluation finished. Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

        return average_loss, num_examples_test, {"accuracy": accuracy}