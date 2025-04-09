from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
import yaml
import os

@dataclass
class DataConfig:
    """Configuration for dataset handling"""
    input_shape: List[int]
    output_shape: List[int]
    feature_columns: List[str] = None
    target_column: str = None
    categorical_columns: List[str] = None
    normalize: bool = True
    val_split: float = 0.2
    test_split: float = 0.1
    batch_size: int = 32
    data_path: str = None
    preprocessing_steps: List[str] = field(default_factory=list)

@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    type: str  # e.g., "cnn", "mlp", "rnn"
    hidden_layers: List[int] = field(default_factory=list)
    activation: str = "relu"
    dropout_rate: float = 0.2
    optimizer: str = "adam"
    learning_rate: float = 0.001
    loss: str = "categorical_crossentropy"
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    l2_regularization: float = 0.0

@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    rounds: int = 10
    min_clients: int = 2
    clients_per_round: int = 2
    local_epochs: int = 1
    aggregation_method: str = "fedavg"  # fedavg, fedprox, etc.
    client_learning_rate: float = 0.01
    server_learning_rate: float = 1.0
    proximal_mu: float = 0.01  # For FedProx
    communication_rounds: int = 100

@dataclass
class CryptoConfig:
    """Configuration for homomorphic encryption"""
    enabled: bool = True
    scheme: str = "CKKS"  # CKKS or BFV
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = field(default_factory=lambda: [40, 20, 40])
    global_scale: int = 2**40
    security_level: int = 128
    use_bootstrapping: bool = False

@dataclass
class PrivacyConfig:
    """Configuration for privacy mechanisms"""
    differential_privacy: bool = False
    dp_epsilon: float = 3.0
    dp_delta: float = 1e-5
    dp_noise_multiplier: float = 1.1
    secure_aggregation: bool = True
    gradient_clipping: float = 1.0

@dataclass
class SystemConfig:
    """System configuration settings"""
    device: str = "cpu"  # cpu, gpu, tpu
    num_workers: int = 4
    log_level: str = "INFO"
    checkpoint_dir: str = "checkpoints"
    result_dir: str = "results"
    seed: int = 42

@dataclass
class FrameworkConfig:
    """Main configuration for the federated learning framework"""
    project_name: str
    task_type: str  # classification, regression, etc.
    data: DataConfig
    model: ModelConfig
    federated: FederatedConfig
    crypto: CryptoConfig
    privacy: PrivacyConfig
    system: SystemConfig
    
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
        
        # Create main config components
        data_config = DataConfig(**config_data.get('data', {}))
        model_config = ModelConfig(**config_data.get('model', {}))
        federated_config = FederatedConfig(**config_data.get('federated', {}))
        crypto_config = CryptoConfig(**config_data.get('crypto', {}))
        privacy_config = PrivacyConfig(**config_data.get('privacy', {}))
        system_config = SystemConfig(**config_data.get('system', {}))
        
        # Create and return main config
        return cls(
            project_name=config_data.get('project_name', 'federated_project'),
            task_type=config_data.get('task_type', 'classification'),
            data=data_config,
            model=model_config,
            federated=federated_config,
            crypto=crypto_config,
            privacy=privacy_config,
            system=system_config
        )
    
    def save(self, config_path: str):
        """Save configuration to a file"""
        # Convert to dictionary
        config_dict = self.to_dict()
        
        # Save to file
        with open(config_path, 'w') as f:
            if config_path.endswith('.json'):
                json.dump(config_dict, f, indent=2)
            elif config_path.endswith(('.yaml', '.yml')):
                yaml.dump(config_dict, f)
            else:
                raise ValueError("Configuration file must be JSON or YAML")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'project_name': self.project_name,
            'task_type': self.task_type,
            'data': self._dataclass_to_dict(self.data),
            'model': self._dataclass_to_dict(self.model),
            'federated': self._dataclass_to_dict(self.federated),
            'crypto': self._dataclass_to_dict(self.crypto),
            'privacy': self._dataclass_to_dict(self.privacy),
            'system': self._dataclass_to_dict(self.system)
        }
    
    @staticmethod
    def _dataclass_to_dict(obj):
        """Convert a dataclass instance to a dictionary"""
        if hasattr(obj, '__dataclass_fields__'):
            return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
        return obj