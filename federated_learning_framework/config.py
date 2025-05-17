# config.py
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
import yaml
import os

@dataclass
class DataConfig:
    """Configuration for dataset handling"""
    data_path: str  # Path to the main dataset file (used by server for eval) or dir
    input_shape: List[int] # Expected input shape (e.g., [num_features] or [C, H, W])
    output_shape: List[int] # Expected output shape (e.g., [num_classes] or [1])
    feature_columns: Optional[List[str]] = None # Specify feature columns for tabular data
    target_column: Optional[str] = None # Specify target column name
    normalize: bool = True
    val_split: float = 0.2
    test_split: float = 0.1
    batch_size: int = 32
    preprocessing_steps: List[str] = field(default_factory=list) # Future use
    pos_weight: Optional[float] = None

@dataclass
class QualityConfig:
    enabled: bool = True # Enable quality-aware aggregation
    score_epsilon: float = 1e-6 # For 1 / (loss + epsilon)
    robust_score_aggregation: bool = True # Enable robust clipping of scores
    # Percentile clipping is specified, so let's use that
    score_clip_percentile_lower: float = 5.0 # e.g., 5th percentile
    score_clip_percentile_upper: float = 95.0 # e.g., 95th percentile
    # MAD-based clipping can be an alternative if you want to add it later
    # use_mad_clipping: bool = False
    # mad_clipping_threshold_factor: float = 2.0 # e.g., median +/- 2*MAD

@dataclass
class RobustnessConfig:
    fraction_noisy_clients: float = 0.0 # 0.0 to 1.0
    label_flip_probability: float = 0.1 # Probability of flipping a label for noisy clients
    fraction_adversarial_clients: float = 0.0 # 0.0 to 1.0
    # For model poisoning, the delta is scaled.
    attack_scale_factor: float = -1.5 # e.g., invert and amplify delta

@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    type: str  # e.g., "mlp", "cnn", "lstm", "binary_classifier"
    task_type: str # Explicitly define: "classification", "binary_classification", "regression"
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    activation: str = "relu"
    dropout_rate: float = 0.2
    optimizer: str = "adam" # Note: Client currently uses SGD, Server uses SGD. Consider alignment.
    learning_rate: float = 0.001 # Global model learning rate (if applicable)
    loss: str = "cross_entropy" # Default, client/server might override based on task_type
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    l2_regularization: float = 0.0

@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    communication_rounds: int = 10 # Renamed from 'rounds' for clarity
    min_clients: int = 2 # Min clients required to start a round
    clients_per_round: int = 2 # Number of clients selected per round
    local_epochs: int = 1 # Epochs each client trains locally
    aggregation_method: str = "fedavg"  # fedavg, fedprox, etc. (Currently only FedAvg implemented)
    client_learning_rate: float = 0.01
    server_learning_rate: float = 1.0 # Learning rate for server-side updates (e.g., FedAvgM)
    proximal_mu: float = 0.01  # For FedProx (Requires implementation in client training)

@dataclass
class CryptoConfig:
    """Configuration for homomorphic encryption"""
    enabled: bool = False # Default to False for easier debugging
    scheme: str = "CKKS"  # CKKS or BFV
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = field(default_factory=lambda: [40, 20, 40])
    global_scale: int = 2**40
    security_level: int = 128 # Standard security level
    # use_bootstrapping: bool = False # Bootstrapping is advanced, not included here

@dataclass
class PrivacyConfig:
    """Configuration for privacy mechanisms"""
    differential_privacy: bool = False
    dp_epsilon: float = 3.0
    dp_delta: float = 1e-5
    dp_noise_multiplier: float = 1.1 # Noise level relative to sensitivity
    gradient_clipping: float = 1.0 # Max L2 norm for gradients (needs client-side implementation if per-sample) or updates
    secure_aggregation: bool = True # Placeholder - HE provides a form of secure aggregation

@dataclass
class SystemConfig:
    """System configuration settings"""
    device: str = "cpu"  # cpu, cuda, mps
    num_workers: int = 0 # DataLoader workers (0 often safest for debugging)
    log_level: str = "INFO"
    checkpoint_dir: str = "checkpoints"
    result_dir: str = "results"
    seed: int = 42
    checkpoint_frequency: int = 10

@dataclass
class FrameworkConfig:
    """Main configuration for the federated learning framework"""
    project_name: str
    data: DataConfig
    model: ModelConfig
    federated: FederatedConfig
    crypto: CryptoConfig
    privacy: PrivacyConfig
    system: SystemConfig
    quality: QualityConfig = field(default_factory=QualityConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)

    # Keep from_file, save, to_dict, _dataclass_to_dict methods as they were

    @classmethod
    def from_file(cls, config_path: str) -> "FrameworkConfig":
        """Load configuration from a file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                config_data = json.load(f)
            elif config_path.endswith(('.yaml', '.yml')):
                config_data = yaml.safe_load(f)
            else:
                raise ValueError("Configuration file must be JSON or YAML")

        # Create main config components, handling potential missing keys with defaults
        data_config = DataConfig(**config_data.get('data', {}))
        model_config = ModelConfig(**config_data.get('model', {}))
        federated_config = FederatedConfig(**config_data.get('federated', {}))
        crypto_config = CryptoConfig(**config_data.get('crypto', {}))
        privacy_config = PrivacyConfig(**config_data.get('privacy', {}))
        system_config = SystemConfig(**config_data.get('system', {}))
        quality_config = QualityConfig(**config_data.get('quality', {}))
        robustness_config = RobustnessConfig(**config_data.get('robustness', {}))

        # Create and return main config
        return cls(
            project_name=config_data.get('project_name', 'federated_project'),
            data=data_config,
            model=model_config,
            federated=federated_config,
            crypto=crypto_config,
            privacy=privacy_config,
            system=system_config,
            quality=quality_config,
            robustness=robustness_config 
        )

    def save(self, config_path: str):
        """Save configuration to a file"""
        config_dict = self.to_dict()
        os.makedirs(os.path.dirname(config_path), exist_ok=True) # Ensure dir exists
        with open(config_path, 'w') as f:
            if config_path.endswith('.json'):
                json.dump(config_dict, f, indent=2)
            elif config_path.endswith(('.yaml', '.yml')):
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                raise ValueError("Configuration file must be JSON or YAML")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        # Basic implementation, can be made more robust if needed
        return {
            'project_name': self.project_name,
            'data': self._dataclass_to_dict(self.data),
            'model': self._dataclass_to_dict(self.model),
            'federated': self._dataclass_to_dict(self.federated),
            'crypto': self._dataclass_to_dict(self.crypto),
            'privacy': self._dataclass_to_dict(self.privacy),
            'system': self._dataclass_to_dict(self.system),
            'quality': self._dataclass_to_dict(self.quality),
            'robustness': self._dataclass_to_dict(self.robustness)
        }

    @staticmethod
    def _dataclass_to_dict(obj):
        """Convert a dataclass instance to a dictionary"""
        if hasattr(obj, '__dataclass_fields__'):
            return {k: FrameworkConfig._dataclass_to_dict(getattr(obj, k))
                    for k in obj.__dataclass_fields__}
        elif isinstance(obj, list):
            return [FrameworkConfig._dataclass_to_dict(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: FrameworkConfig._dataclass_to_dict(v) for k, v in obj.items()}
        else:
            return obj