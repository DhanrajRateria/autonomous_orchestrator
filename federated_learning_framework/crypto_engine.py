# crypto_engine.py
import numpy as np
import tenseal as ts
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
import base64
import pickle
import time
import torch

from federated_learning_framework.config import CryptoConfig # Assuming config.py is in this path

class CryptoEngine:
    """
    Homomorphic encryption engine using TenSEAL (CKKS scheme primarily).
    Handles encryption/decryption of model parameters for secure aggregation.
    """

    def __init__(self, config: CryptoConfig):
        self.logger = logging.getLogger("crypto_engine")
        self.config = config
        self.context = None
        self.secret_key = None # Only stored on the server instance typically

        if not config.enabled:
            self.logger.info("Homomorphic encryption is disabled.")
            return

        self.logger.info(f"Initializing {config.scheme} homomorphic encryption...")
        if config.scheme != "CKKS":
            self.logger.warning(f"Only CKKS scheme is fully tested. Using {config.scheme} might lead to issues.")
            # raise NotImplementedError(f"{config.scheme} not fully supported/tested.")

        try:
            self._create_context()
            # Generate necessary keys
            # Secret key should only be generated and kept by the server
            # self.secret_key = self.context.secret_key() # Generate secret key
            # self.context.make_context_public() # Remove secret key for sharing context
            self.logger.info(f"Successfully initialized {config.scheme} encryption context.")
        except Exception as e:
            self.logger.error(f"Failed to initialize TenSEAL context: {e}", exc_info=True)
            self.context = None # Ensure context is None if init fails
            self.config.enabled = False # Disable crypto if init fails
            self.logger.warning("Homomorphic encryption has been disabled due to initialization failure.")

    def _create_context(self):
        """Creates and configures the TenSEAL context."""
        start_time = time.time()
        scheme = ts.SCHEME_TYPE.CKKS if self.config.scheme == "CKKS" else ts.SCHEME_TYPE.BFV # Add other schemes if needed

        self.context = ts.context(
            scheme,
            poly_modulus_degree=self.config.poly_modulus_degree,
            coeff_mod_bit_sizes=self.config.coeff_mod_bit_sizes
        )
        # Set the global scale for CKKS
        self.context.global_scale = self.config.global_scale
        # Generate Galois keys needed for rotations (often required for operations like dot product)
        self.context.generate_galois_keys() # Generate necessary keys AFTER context setup
        self.context.generate_relin_keys() # Often needed for multiplications

        # IMPORTANT: Secret key generation and management
        # The entity performing decryption (usually the server) generates and holds the secret key.
        # The public context (without the secret key) is shared with clients.
        self.secret_key = self.context.secret_key() # Generate and store the secret key
        self.context.make_context_public() # Make the context suitable for sharing (removes secret key)

        elapsed = time.time() - start_time
        self.logger.info(f"TenSEAL context created in {elapsed:.2f} seconds.")

    def is_enabled(self) -> bool:
        """Check if encryption is enabled and context is valid."""
        return self.config.enabled and self.context is not None

    def encrypt_vector(self, vector: Union[List[float], np.ndarray]) -> Optional[ts.CKKSVector]:
        """Encrypts a list or numpy array."""
        if not self.is_enabled():
            self.logger.warning("Attempted to encrypt vector while HE is disabled.")
            return None # Or raise error? Should not happen in normal flow.
        if not isinstance(vector, (list, np.ndarray)):
            raise TypeError(f"Input must be a list or numpy array, got {type(vector)}")
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        try:
            return ts.ckks_vector(self.context, vector)
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}", exc_info=True)
            return None

    def decrypt_vector(self, encrypted_vector: ts.CKKSVector) -> Optional[List[float]]:
        """Decrypts an encrypted vector."""
        if not self.is_enabled():
             self.logger.warning("Attempted to decrypt vector while HE is disabled.")
             return None # Or raise error?
        if self.secret_key is None:
            self.logger.error("Decryption attempted without a secret key.")
            return None
        if not isinstance(encrypted_vector, ts.CKKSVector):
             # If we receive non-encrypted data unexpectedly, return it (might happen if encryption failed upstream)
            if isinstance(encrypted_vector, list):
                self.logger.warning("Decrypt called on a non-encrypted list, returning as is.")
                return encrypted_vector
            raise TypeError(f"Input must be a CKKSVector, got {type(encrypted_vector)}")

        try:
            # Decrypt using the secret key stored in this engine instance
            return encrypted_vector.decrypt(self.secret_key)
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}", exc_info=True)
            # Attempt to return zeros of appropriate size if possible
            try:
                # Getting size might fail on corrupted data
                size = len(encrypted_vector) # Or access underlying structure if possible
                return [0.0] * size
            except:
                 return [0.0] # Fallback

    # --- Serialization ---

    def serialize_vector(self, vector: Union[ts.CKKSVector, List, np.ndarray]) -> str:
        """Serializes an encrypted vector or plain data."""
        if isinstance(vector, ts.CKKSVector):
            try:
                serialized_bytes = vector.serialize()
                return base64.b64encode(serialized_bytes).decode('ascii')
            except Exception as e:
                self.logger.error(f"Failed to serialize CKKSVector: {e}", exc_info=True)
                # Fallback: try pickling the error state or return empty string?
                return "" # Indicate error
        elif isinstance(vector, (list, np.ndarray)):
            # Serialize plain data using pickle
            try:
                return base64.b64encode(pickle.dumps(vector)).decode('ascii')
            except Exception as e:
                self.logger.error(f"Failed to pickle/serialize plain data: {e}", exc_info=True)
                return ""
        else:
            raise TypeError(f"Unsupported type for serialization: {type(vector)}")

    def deserialize_vector(self, serialized_data: str) -> Union[ts.CKKSVector, List, np.ndarray, None]:
        """Deserializes data, determining if it's encrypted or plain."""
        if not serialized_data:
             self.logger.warning("Attempted to deserialize empty data.")
             return None
        try:
            decoded_bytes = base64.b64decode(serialized_data)
        except Exception as e:
            self.logger.error(f"Base64 decoding failed: {e}", exc_info=True)
            return None

        # Try deserializing as TenSEAL vector first
        if self.is_enabled():
            try:
                # Use context_from to handle potential context mismatches robustly?
                # Or assume the current context is the correct one.
                # CKKS is assumed based on current implementation focus
                return ts.ckks_vector_from(self.context, decoded_bytes)
            except (RuntimeError, ValueError, TypeError) as tenseal_error:
                # If TenSEAL deserialization fails, try pickle (it might be plain data)
                # self.logger.debug(f"TenSEAL deserialization failed ({tenseal_error}), trying pickle...")
                pass # Fall through to pickle attempt
            except Exception as e:
                 self.logger.error(f"Unexpected error during TenSEAL deserialization: {e}", exc_info=True)
                 # Fall through to pickle attempt

        # Try deserializing as pickled object
        try:
            obj = pickle.loads(decoded_bytes)
            if isinstance(obj, (list, np.ndarray)):
                # If encryption was expected but we got plain data, log a warning
                if self.is_enabled():
                    self.logger.warning("Deserialized plain data where encrypted data might have been expected.")
                return obj
            else:
                self.logger.warning(f"Pickle deserialization resulted in unexpected type: {type(obj)}")
                return None
        except pickle.UnpicklingError:
            self.logger.error("Deserialization failed: Data is neither a valid TenSEAL vector nor a pickled object.")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during pickle deserialization: {e}", exc_info=True)
            return None

    # --- Model Parameter Handling ---

    def _process_param(self, param: np.ndarray, encrypt: bool) -> Union[Dict[str, Any], np.ndarray]:
        """Helper to encrypt/package or just return a numpy parameter."""
        shape = param.shape
        if encrypt:
            if not self.is_enabled():
                 # Should not happen if called correctly, return plain data with warning
                 self.logger.warning("Encryption requested but HE disabled during parameter processing.")
                 return param

            if param.ndim == 0: # Scalar
                encrypted_vec = self.encrypt_vector([param.item()])
                data = self.serialize_vector(encrypted_vec) if encrypted_vec else None
                return {"type": "scalar", "shape": list(shape), "data": data} if data else None
            elif param.ndim == 1: # Vector
                encrypted_vec = self.encrypt_vector(param)
                data = self.serialize_vector(encrypted_vec) if encrypted_vec else None
                return {"type": "vector", "shape": list(shape), "data": data} if data else None
            elif param.ndim >= 2: # Matrix or Tensor -> Flatten
                flattened = param.reshape(-1)
                encrypted_vec = self.encrypt_vector(flattened)
                data = self.serialize_vector(encrypted_vec) if encrypted_vec else None
                return {"type": "tensor", "shape": list(shape), "data": data} if data else None
        else:
            # Return plain numpy array
            return param

    def _unprocess_param(self, processed_param: Union[Dict[str, Any], np.ndarray]) -> Optional[np.ndarray]:
        """Helper to decrypt/unpackage or just return a numpy parameter."""
        if isinstance(processed_param, np.ndarray):
            # It's already plain data
            return processed_param
        elif isinstance(processed_param, dict) and "type" in processed_param and "data" in processed_param:
            # It's packaged (potentially encrypted) data
            param_type = processed_param["type"]
            shape = tuple(processed_param["shape"])
            serialized_data = processed_param["data"]

            if serialized_data is None:
                self.logger.error(f"Cannot unprocess parameter, serialized data is None for shape {shape}")
                return None

            deserialized = self.deserialize_vector(serialized_data)

            if deserialized is None:
                 self.logger.error(f"Failed to deserialize parameter data for shape {shape}")
                 return None

            if isinstance(deserialized, ts.CKKSVector):
                 if not self.is_enabled():
                      self.logger.error("Cannot decrypt parameter: HE is disabled but received encrypted data.")
                      return None
                 # Decrypt
                 decrypted_list = self.decrypt_vector(deserialized)
                 if decrypted_list is None:
                      self.logger.error(f"Failed to decrypt parameter for shape {shape}")
                      return None
                 decrypted_np = np.array(decrypted_list, dtype=np.float32)

            elif isinstance(deserialized, (list, np.ndarray)):
                 # It was plain data serialized
                 decrypted_np = np.array(deserialized, dtype=np.float32)
            else:
                 self.logger.error(f"Deserialization yielded unexpected type: {type(deserialized)}")
                 return None

            # Reshape if needed
            if param_type == "tensor" or param_type == "matrix": # We always flatten tensors >= 2D
                 try:
                      return decrypted_np.reshape(shape)
                 except ValueError as e:
                      self.logger.error(f"Reshape error: Decrypted size {decrypted_np.size} != expected shape {shape} ({np.prod(shape)}). Error: {e}")
                      return None # Or try to return flattened?
            elif param_type == "vector" or param_type == "scalar":
                # Ensure shape matches (e.g., scalar needs to be extracted)
                if decrypted_np.shape == shape:
                    return decrypted_np
                elif param_type == "scalar" and decrypted_np.size == 1:
                    return decrypted_np.reshape(shape) # Reshape to ()
                elif param_type == "vector" and decrypted_np.size == np.prod(shape):
                     return decrypted_np.reshape(shape)
                else:
                    self.logger.error(f"Shape mismatch: Decrypted shape {decrypted_np.shape} != expected shape {shape}")
                    return None
            else:
                self.logger.error(f"Unknown parameter type '{param_type}' during unprocessing.")
                return None

        else:
            self.logger.error(f"Invalid format for processed parameter: {type(processed_param)}")
            return None

    def encrypt_model_params(self, model_params: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Encrypts and packages model parameters (numpy arrays)."""
        if not self.is_enabled():
            # If HE disabled, return plain numpy arrays (or pickled versions?)
            # Returning plain numpy arrays matches the non-HE flow better.
            return model_params

        self.logger.debug("Encrypting model parameters...")
        encrypted_packaged_params = {}
        for name, param in model_params.items():
            processed = self._process_param(param, encrypt=True)
            if processed is not None:
                encrypted_packaged_params[name] = processed
            else:
                self.logger.error(f"Failed to encrypt/package parameter '{name}'")
                # Decide handling: skip param, raise error, return partial?
                # Returning partial might be best for FL resilience.
        return encrypted_packaged_params

    def decrypt_model_params(self, processed_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Decrypts and unpackages parameters back into numpy arrays."""
        self.logger.debug("Decrypting model parameters...")
        decrypted_params = {}
        for name, processed_param in processed_params.items():
            unprocessed = self._unprocess_param(processed_param)
            if unprocessed is not None:
                decrypted_params[name] = unprocessed
            else:
                self.logger.error(f"Failed to decrypt/unpackage parameter '{name}'")
                # Skip param
        return decrypted_params

    def encrypt_torch_params(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Extracts, encrypts, and packages parameters from a PyTorch model."""
        model_params_np = {name: param.cpu().detach().numpy() for name, param in model.named_parameters()}
        return self.encrypt_model_params(model_params_np)

    def decrypt_to_torch_params(self, model: torch.nn.Module, processed_params: Dict[str, Any]):
        """Decrypts parameters and loads them into a PyTorch model."""
        decrypted_params_np = self.decrypt_model_params(processed_params)

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in decrypted_params_np:
                    decrypted_tensor = torch.from_numpy(decrypted_params_np[name])
                    if param.shape == decrypted_tensor.shape:
                        param.copy_(decrypted_tensor)
                    else:
                        self.logger.error(f"Shape mismatch loading decrypted param '{name}': Model={param.shape}, Decrypted={decrypted_tensor.shape}. Skipping update for this param.")
                else:
                    self.logger.warning(f"Parameter '{name}' not found in decrypted parameters. Keeping existing model parameter.")

    # --- Homomorphic Operations ---

    def homomorphic_add(self, enc_vec1: ts.CKKSVector, enc_vec2: ts.CKKSVector) -> Optional[ts.CKKSVector]:
        """Performs homomorphic addition."""
        if not self.is_enabled(): return None
        try:
            return enc_vec1 + enc_vec2
        except Exception as e:
            self.logger.error(f"Homomorphic addition failed: {e}", exc_info=True)
            return None

    def homomorphic_multiply_scalar(self, enc_vec: ts.CKKSVector, scalar: float) -> Optional[ts.CKKSVector]:
        """Performs homomorphic multiplication by a plain scalar."""
        if not self.is_enabled():
            self.logger.warning("HE disabled: skipping homomorphic multiply.")
            return None

        try:
            # In TenSEAL, when you multiply a CKKSVector by a plaintext scalar,
            # the library attempts to manage the scale of the resulting ciphertext.
            # It generally tries to keep the scale consistent with the input vector's scale
            # or the context's global_scale.

            # Direct multiplication is the standard way:
            result = enc_vec * scalar

            # After multiplication, the result vector will have a scale.
            # If this scale is drastically different from the context's global_scale,
            # or if "scale out of bounds" errors occur, it indicates that the
            # HE parameters (coeff_mod_bit_sizes, global_scale, poly_modulus_degree)
            # are not well-suited for the precision and depth of operations required.

            # No explicit rescaling loops are generally needed here for simple scalar multiplication.
            # If rescaling is necessary, it's usually because the parameters
            # don't provide enough "room" or "levels" in the modulus chain.

            return result

        except ValueError as ve:
            # This is where "scale out of bounds" or other TenSEAL value errors would be caught.
            self.logger.error(f"Homomorphic multiply by {scalar} failed with ValueError (likely scale or parameter issue): {ve}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Homomorphic multiply by {scalar} failed with other Exception: {e}", exc_info=True)
            return None

    def secure_aggregation(self, encrypted_vectors: List[ts.CKKSVector], weights: List[float]) -> Optional[ts.CKKSVector]:
        """
        Performs secure aggregation (weighted average) on encrypted vectors.
        Assumes vectors are CKKSVectors.
        """
        if not self.is_enabled():
            self.logger.error("Secure aggregation called when HE is disabled.")
            return None
        if not encrypted_vectors:
            self.logger.warning("No encrypted vectors provided for aggregation.")
            return None
        if len(encrypted_vectors) != len(weights):
            self.logger.error(f"Mismatch in number of vectors ({len(encrypted_vectors)}) and weights ({len(weights)}).")
            return None

        if abs(sum(weights) - 1.0) > 1e-6:
             self.logger.warning(f"Weights do not sum to 1 (sum={sum(weights)}). Normalizing.")
             total_weight = sum(weights)
             if total_weight == 0:
                 self.logger.error("Cannot aggregate with zero total weight.")
                 return None
             weights = [w / total_weight for w in weights]


        aggregated_vector = None
        try:
            # Multiply first vector by its weight
            weighted_first = self.homomorphic_multiply_scalar(encrypted_vectors[0], weights[0])
            if weighted_first is None:
                raise ValueError("Failed to multiply first vector by weight.")
            aggregated_vector = weighted_first

            # Add the rest weighted vectors
            for i in range(1, len(encrypted_vectors)):
                weighted_vec = self.homomorphic_multiply_scalar(encrypted_vectors[i], weights[i])
                if weighted_vec is None:
                     raise ValueError(f"Failed to multiply vector {i} by weight.")

                # Perform addition
                current_sum = self.homomorphic_add(aggregated_vector, weighted_vec)
                if current_sum is None:
                     raise ValueError(f"Failed to add weighted vector {i} to sum.")
                aggregated_vector = current_sum

            return aggregated_vector

        except Exception as e:
            self.logger.error(f"Secure aggregation failed: {e}", exc_info=True)
            return None

    # --- Context Management ---

    def get_public_context(self) -> Optional[bytes]:
        """Serializes the public context (without secret key) for distribution."""
        if not self.context:
            self.logger.error("Cannot get public context, context not initialized.")
            return None
        try:
            # Ensure the context is public before serializing
            if not self.context.is_public():
                 # This should not happen if _create_context worked correctly
                 self.context.make_context_public()
            return self.context.serialize()
        except Exception as e:
            self.logger.error(f"Failed to serialize public context: {e}", exc_info=True)
            return None

    def load_public_context(self, context_bytes: bytes):
        """Loads a public context from bytes (e.g., on the client)."""
        if self.config.enabled:
            try:
                self.context = ts.context_from(context_bytes)
                self.secret_key = None # Clients don't have the secret key
                self.logger.info("Successfully loaded public TenSEAL context.")
            except Exception as e:
                self.logger.error(f"Failed to load public context from bytes: {e}", exc_info=True)
                self.context = None
                self.config.enabled = False # Disable crypto if context loading fails
                self.logger.warning("Disabling HE due to public context loading failure.")
        else:
             self.logger.debug("Skipping public context loading as HE is disabled.")


    # Methods for saving/loading context to/from files (primarily for server persistence)
    def save_context_to_file(self, path: str):
        """Saves the public context and secret key (if present) to files."""
        if not self.context:
            self.logger.error("Cannot save context, not initialized.")
            return

        # Save public context
        public_context_bytes = self.get_public_context()
        if public_context_bytes:
            try:
                with open(path, "wb") as f:
                    f.write(public_context_bytes)
                self.logger.info(f"Public context saved to {path}")
            except Exception as e:
                self.logger.error(f"Failed to save public context to {path}: {e}", exc_info=True)

        # Save secret key separately (ONLY if it exists, i.e., on the server)
        if self.secret_key:
            secret_key_path = f"{path}.secret"
            try:
                 # SecretKey doesn't have a direct serialize, need to get it from context before make_public?
                 # Let's re-create context temporarily to get serializable secret key - this is awkward.
                 # OR - Store the secret key bytes when generated. Let's assume we stored it.
                 # TenSEAL SecretKey objects are directly serializable via pickle
                with open(secret_key_path, "wb") as f:
                    pickle.dump(self.secret_key, f)
                self.logger.info(f"Secret key saved to {secret_key_path}")
            except Exception as e:
                 self.logger.error(f"Failed to save secret key to {secret_key_path}: {e}", exc_info=True)


    def load_context_from_file(self, path: str, load_secret: bool = False):
        """Loads context from files. If load_secret is True, attempts to load secret key."""
        if not self.config.enabled:
            self.logger.debug("Skipping context loading from file as HE is disabled.")
            return

        # Load public context
        try:
            with open(path, "rb") as f:
                public_context_bytes = f.read()
            self.load_public_context(public_context_bytes)
            if not self.context: # Check if loading failed
                raise RuntimeError("Public context loading failed within load_public_context.")
        except FileNotFoundError:
             self.logger.error(f"Public context file not found: {path}")
             self.context = None
             self.config.enabled = False
             return
        except Exception as e:
            self.logger.error(f"Failed to load public context from {path}: {e}", exc_info=True)
            self.context = None
            self.config.enabled = False
            return

        # Load secret key if requested and possible
        if load_secret:
            secret_key_path = f"{path}.secret"
            if self.context: # Only load secret key if public context loaded okay
                try:
                    with open(secret_key_path, "rb") as f:
                        self.secret_key = pickle.load(f)
                    # Re-associate secret key with the context (is this needed? TenSEAL handles this internally)
                    # self.context.set_secret_key(self.secret_key) # May not be necessary/available API
                    self.logger.info(f"Secret key loaded from {secret_key_path}")
                except FileNotFoundError:
                    self.logger.warning(f"Secret key file not found: {secret_key_path}. Cannot perform decryption.")
                    self.secret_key = None
                except Exception as e:
                     self.logger.error(f"Failed to load secret key from {secret_key_path}: {e}", exc_info=True)
                     self.secret_key = None
            else:
                self.logger.warning("Skipping secret key loading as public context failed to load.")