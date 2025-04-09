import os
import json
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class CryptoSettings:
    """Configuration settings for homomorphic encryption"""
    key_size: int = 2048
    security_level: int = 128
    scheme_type: str = "BFV"  # BFV, CKKS, BGV
    polynomial_modulus_degree: int = 8192
    use_secret_sharing: bool = True
    bootstrapping_enabled: bool = False  # Advanced feature - enables longer computation chains

@dataclass
class FederatedLearningConfig:
    """Configuration settings for federated learning"""
    min_clients: int = 3
    aggregation_method: str = "secure_fedavg"  # secure_fedavg, secure_fedprox
    rounds_per_update: int = 5
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    model_architecture: str = "lstm_anomaly_detector"
    diff_privacy_enabled: bool = True
    diff_privacy_epsilon: float = 3.0
    diff_privacy_delta: float = 1e-5

@dataclass
class CloudProviderConfig:
    """Configuration for a cloud provider connection"""
    provider_id: str
    provider_type: str  # aws, azure, gcp
    credentials_path: str
    enabled_services: List[str] = field(default_factory=list)
    region: str = "us-west-1"
    polling_interval: int = 60  # seconds
    
    def create_connector(self):
        """Factory method to create the appropriate cloud connector"""
        from cloud_providers.aws.connector import AWSConnector
        from cloud_providers.azure.connector import AzureConnector
        from cloud_providers.gcp.connector import GCPConnector
        
        if self.provider_type.lower() == "aws":
            return AWSConnector(self)
        elif self.provider_type.lower() == "azure":
            return AzureConnector(self)
        elif self.provider_type.lower() == "gcp":
            return GCPConnector(self)
        else:
            raise ValueError(f"Unsupported cloud provider type: {self.provider_type}")

@dataclass
class SystemConfig:
    """Configuration for system optimization settings"""
    max_workers: int = 8
    task_queue_size: int = 1000
    memory_limit_mb: int = 8192  # 8GB
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    optimization_level: str = "aggressive"  # conservative, balanced, aggressive

@dataclass
class OrchestrationConfig:
    """Main configuration for the security orchestrator"""
    # Core settings
    instance_id: str
    use_encryption_for_analysis: bool = True
    event_polling_interval: int = 30  # seconds
    model_update_interval: int = 3600  # seconds (1 hour)
    analysis_batch_size: int = 100
    detection_threshold: float = 0.7
    response_threshold: float = 0.85
    
    # Component configurations
    crypto_settings: CryptoSettings = field(default_factory=CryptoSettings)
    federated_learning: FederatedLearningConfig = field(default_factory=FederatedLearningConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Cloud provider configurations
    cloud_providers: List[CloudProviderConfig] = field(default_factory=list)
    
    # Paths to resources
    response_policies_path: str = "configs/response_policies.yaml"
    threat_definitions_path: str = "configs/threat_definitions.yaml"
    
    @classmethod
    def from_file(cls, config_path: str) -> "OrchestrationConfig":
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
        
        # Create main config
        config = cls(
            instance_id=config_data.get('instance_id', ''),
            use_encryption_for_analysis=config_data.get('use_encryption_for_analysis', True),
            event_polling_interval=config_data.get('event_polling_interval', 30),
            model_update_interval=config_data.get('model_update_interval', 3600),
            analysis_batch_size=config_data.get('analysis_batch_size', 100),
            detection_threshold=config_data.get('detection_threshold', 0.7),
            response_threshold=config_data.get('response_threshold', 0.85),
            response_policies_path=config_data.get('response_policies_path', 'configs/response_policies.yaml'),
            threat_definitions_path=config_data.get('threat_definitions_path', 'configs/threat_definitions.yaml')
        )
        
        # Parse crypto settings
        if 'crypto_settings' in config_data:
            cs = config_data['crypto_settings']
            config.crypto_settings = CryptoSettings(
                key_size=cs.get('key_size', 2048),
                security_level=cs.get('security_level', 128),
                scheme_type=cs.get('scheme_type', 'BFV'),
                polynomial_modulus_degree=cs.get('polynomial_modulus_degree', 8192),
                use_secret_sharing=cs.get('use_secret_sharing', True),
                bootstrapping_enabled=cs.get('bootstrapping_enabled', False)
            )
        
        # Parse federated learning config
        if 'federated_learning' in config_data:
            fl = config_data['federated_learning']
            config.federated_learning = FederatedLearningConfig(
                min_clients=fl.get('min_clients', 3),
                aggregation_method=fl.get('aggregation_method', 'secure_fedavg'),
                rounds_per_update=fl.get('rounds_per_update', 5),
                local_epochs=fl.get('local_epochs', 1),
                batch_size=fl.get('batch_size', 32),
                learning_rate=fl.get('learning_rate', 0.01),
                model_architecture=fl.get('model_architecture', 'lstm_anomaly_detector'),
                diff_privacy_enabled=fl.get('diff_privacy_enabled', True),
                diff_privacy_epsilon=fl.get('diff_privacy_epsilon', 3.0),
                diff_privacy_delta=fl.get('diff_privacy_delta', 1e-5)
            )
        
        # Parse system config
        if 'system' in config_data:
            sys = config_data['system']
            config.system = SystemConfig(
                max_workers=sys.get('max_workers', 8),
                task_queue_size=sys.get('task_queue_size', 1000),
                memory_limit_mb=sys.get('memory_limit_mb', 8192),
                use_gpu=sys.get('use_gpu', True),
                gpu_memory_fraction=sys.get('gpu_memory_fraction', 0.8),
                optimization_level=sys.get('optimization_level', 'aggressive')
            )
        
        # Parse cloud providers
        if 'cloud_providers' in config_data:
            for provider_data in config_data['cloud_providers']:
                provider_config = CloudProviderConfig(
                    provider_id=provider_data['provider_id'],
                    provider_type=provider_data['provider_type'],
                    credentials_path=provider_data['credentials_path'],
                    enabled_services=provider_data.get('enabled_services', []),
                    region=provider_data.get('region', 'us-west-1'),
                    polling_interval=provider_data.get('polling_interval', 60)
                )
                config.cloud_providers.append(provider_config)
        
        return config

    def save_to_file(self, config_path: str):
        """Save configuration to a file"""
        # Convert config to dictionary
        config_dict = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.json'):
                json.dump(config_dict, f, indent=2)
            elif config_path.endswith(('.yaml', '.yml')):
                yaml.dump(config_dict, f)
            else:
                raise ValueError("Configuration file must be JSON or YAML")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        # Implementation left as an exercise
        pass  # In a real implementation, this would convert all dataclasses to dictionaries