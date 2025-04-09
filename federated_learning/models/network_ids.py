import numpy as np
import tensorflow as tf
import asyncio
from typing import Dict, List, Any, Tuple
import logging
import time
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class NetworkIDSModel:
    """
    Convolutional neural network model for network intrusion detection.
    Designed to detect various network attack patterns.
    """
    
    def __init__(self, input_dim: int = 128, learning_rate: float = 0.001):
        """
        Initialize the network IDS model.
        
        Args:
            input_dim: Dimensionality of input features
            learning_rate: Learning rate for optimizer
        """
        self.logger = logging.getLogger("model.network_ids")
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        
        # Number of attack classes (0=normal, 1-4=attack types)
        self.num_classes = 5
        
        # Create the model
        self._create_model()
        
        self.logger.info("Network IDS model initialized")
    
    def _create_model(self):
        """Create the TensorFlow model"""
        try:
            # Set memory growth to prevent TF from allocating all GPU memory
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            # Reshape input for 1D convolution
            inputs = Input(shape=(self.input_dim,))
            x = tf.reshape(inputs, [-1, self.input_dim, 1])
            
            # First convolutional block
            x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
            x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(0.2)(x)
            
            # Second convolutional block
            x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
            x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(0.2)(x)
            
            # Flatten and fully connected layers
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(64, activation='relu')(x)
            
            # Output layer with softmax for multi-class classification
            outputs = Dense(self.num_classes, activation='softmax')(x)
            
            # Create model
            self.model = Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Create TensorFlow function for predictions
            self.predict_fn = tf.function(
                self.model.predict,
                input_signature=[tf.TensorSpec(shape=(None, self.input_dim), dtype=tf.float32)]
            )
            
            self.logger.debug(f"Model summary: {self.model.summary()}")
        
        except Exception as e:
            self.logger.error(f"Error creating model: {e}", exc_info=True)
            raise
    
    async def train(self, features: np.ndarray, labels: np.ndarray,
                  epochs: int = 1, batch_size: int = 32) -> Dict[str, float]:
        """
        Train the model on the given data.
        
        Args:
            features: Input features
            labels: Target labels (class indices)
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training metrics
        """
        try:
            # Convert inputs to TF tensors
            features_tf = tf.convert_to_tensor(features, dtype=tf.float32)
            labels_tf = tf.convert_to_tensor(labels, dtype=tf.int32)
            
            # Make this function non-blocking by running in thread pool
            loop = asyncio.get_event_loop()
            start_time = time.time()
            
            # Train the model in a separate thread
            history = await loop.run_in_executor(
                None,
                lambda: self.model.fit(
                    features_tf, labels_tf,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )
            )
            
            elapsed = time.time() - start_time
            self.logger.info(f"Training completed in {elapsed:.2f} seconds")
            
            # Extract metrics
            metrics = {}
            for metric, values in history.history.items():
                metrics[metric] = float(values[-1])  # Last epoch value
            
            return {
                "loss": metrics.get("loss", 0.0),
                "accuracy": metrics.get("accuracy", 0.0),
                "training_time": elapsed
            }
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}", exc_info=True)
            raise
    
    async def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions on the input features.
        
        Args:
            features: Input features
            
        Returns:
            Predicted class probabilities
        """
        try:
            # Convert inputs to TF tensors
            features_tf = tf.convert_to_tensor(features, dtype=tf.float32)
            
            # Make this function non-blocking by running in thread pool
            loop = asyncio.get_event_loop()
            
            # Run prediction in a separate thread
            predictions = await loop.run_in_executor(
                None,
                lambda: self.predict_fn(features_tf).numpy()
            )
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}", exc_info=True)
            raise
    
    async def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on the given data.
        
        Args:
            features: Input features
            labels: True labels (class indices)
            
        Returns:
            Evaluation metrics
        """
        try:
            # Convert inputs to TF tensors
            features_tf = tf.convert_to_tensor(features, dtype=tf.float32)
            labels_tf = tf.convert_to_tensor(labels, dtype=tf.int32)
            
            # Make this function non-blocking by running in thread pool
            loop = asyncio.get_event_loop()
            
            # Evaluate the model in a separate thread
            results = await loop.run_in_executor(
                None,
                lambda: self.model.evaluate(features_tf, labels_tf, verbose=0)
            )
            
            # Create metrics dictionary
            metrics = {}
            for metric_name, value in zip(self.model.metrics_names, results):
                metrics[metric_name] = float(value)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}", exc_info=True)
            raise
    
    def get_weights(self) -> List[np.ndarray]:
        """
        Get the current model weights.
        
        Returns:
            List of weight arrays
        """
        return self.model.get_weights()
    
    def set_weights(self, weights: List[np.ndarray]):
        """
        Set the model weights.
        
        Args:
            weights: List of weight arrays
        """
        try:
            self.model.set_weights(weights)
        except Exception as e:
            self.logger.error(f"Error setting weights: {e}", exc_info=True)
            raise
    
    async def detect_attack_type(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect and classify network attack types.
        
        Args:
            features: Input features
            
        Returns:
            List of dictionaries with attack predictions
        """
        # Define attack classes
        attack_classes = {
            0: "normal",
            1: "dos",      # Denial of Service
            2: "probe",    # Scanning and probing
            3: "u2r",      # User to Root
            4: "r2l"       # Remote to Local
        }
        
        # Get predictions
        predictions = await self.predict(features)
        
        # Process predictions
        results = []
        for i, pred in enumerate(predictions):
            # Get most likely class and its probability
            most_likely_class = int(np.argmax(pred))
            confidence = float(pred[most_likely_class])
            
            # Create result object
            result = {
                "sample_id": i,
                "attack_type": attack_classes[most_likely_class],
                "is_attack": most_likely_class > 0,  # Class 0 is normal
                "confidence": confidence,
                "class_probabilities": {
                    attack_classes[j]: float(p) 
                    for j, p in enumerate(pred)
                }
            }
            results.append(result)
        
        return results
    
    def save_model(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        try:
            self.model.save(path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}", exc_info=True)
            raise
    
    def load_model(self, path: str):
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        try:
            self.model = tf.keras.models.load_model(path)
            self.logger.info(f"Model loaded from {path}")
            
            # Recreate prediction function
            self.predict_fn = tf.function(
                self.model.predict,
                input_signature=[tf.TensorSpec(shape=(None, self.input_dim), dtype=tf.float32)]
            )
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            raise