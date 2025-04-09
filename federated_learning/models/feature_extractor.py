import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Union, Optional, Tuple
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.signal as signal
from scipy.stats import skew, kurtosis

class FeatureExtractor:
    """
    Feature extraction for security data to prepare inputs for 
    machine learning models used in federated learning.
    """
    
    def __init__(self, input_dim: int = 128, output_dim: int = 32,
                 use_pca: bool = False, use_normalization: bool = True):
        """
        Initialize the feature extractor.
        
        Args:
            input_dim: Original input dimensionality
            output_dim: Target output dimensionality (after extraction/reduction)
            use_pca: Whether to use PCA for dimensionality reduction
            use_normalization: Whether to normalize features
        """
        self.logger = logging.getLogger("federated.feature_extractor")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_pca = use_pca
        self.use_normalization = use_normalization
        
        # Initialize components
        self.scaler = StandardScaler() if use_normalization else None
        self.pca = PCA(n_components=output_dim) if use_pca else None
        
        # Feature cache for efficiency
        self._feature_cache = {}
        
        self.logger.info(
            f"Feature extractor initialized with output_dim={output_dim}, "
            f"use_pca={use_pca}, use_normalization={use_normalization}"
        )
    
    def extract_features(self, data: Union[List[Dict[str, Any]], np.ndarray], 
                        data_type: str = "network") -> np.ndarray:
        """
        Extract features from raw data.
        
        Args:
            data: Input data (either raw events or numerical array)
            data_type: Type of data ('network', 'auth', 'api', 'resource')
            
        Returns:
            Numpy array of extracted features
        """
        if isinstance(data, np.ndarray):
            # If already numerical, just transform
            return self._transform_features(data)
        
        # Extract based on data type
        if data_type == "network":
            features = self._extract_network_features(data)
        elif data_type == "auth":
            features = self._extract_auth_features(data)
        elif data_type == "api":
            features = self._extract_api_features(data)
        elif data_type == "resource":
            features = self._extract_resource_features(data)
        else:
            self.logger.warning(f"Unknown data type: {data_type}, using generic extraction")
            features = self._extract_generic_features(data)
        
        # Transform the extracted features
        return self._transform_features(features)
    
    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        """Apply transformations like scaling and dimensionality reduction"""
        if features.shape[0] == 0:
            # Empty input, return empty output
            return np.zeros((0, self.output_dim))
        
        # Apply scaling if enabled
        if self.use_normalization and self.scaler:
            try:
                features = self.scaler.fit_transform(features)
            except Exception as e:
                self.logger.warning(f"Error in feature normalization: {e}, skipping")
        
        # Apply PCA if enabled
        if self.use_pca and self.pca:
            try:
                features = self.pca.fit_transform(features)
            except Exception as e:
                self.logger.warning(f"Error in PCA: {e}, using first {self.output_dim} features")
                # Fallback to simple slicing
                features = features[:, :min(features.shape[1], self.output_dim)]
        
        # If not using PCA, ensure output dimensionality
        if not self.use_pca:
            if features.shape[1] > self.output_dim:
                # Too many features, reduce
                features = features[:, :self.output_dim]
            elif features.shape[1] < self.output_dim:
                # Too few features, pad with zeros
                padding = np.zeros((features.shape[0], self.output_dim - features.shape[1]))
                features = np.hstack((features, padding))
        
        return features
    
    def _extract_network_features(self, events: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from network events"""
        features = np.zeros((len(events), self.input_dim))
        
        for i, event in enumerate(events):
            if "network" not in event:
                continue
            
            net = event["network"]
            
            # Basic traffic features
            idx = 0
            features[i, idx] = net.get("bytes_in", 0); idx += 1
            features[i, idx] = net.get("bytes_out", 0); idx += 1
            features[i, idx] = net.get("packets_in", 0); idx += 1
            features[i, idx] = net.get("packets_out", 0); idx += 1
            
            # Flow duration and timing
            features[i, idx] = net.get("duration_ms", 0) / 1000.0; idx += 1
            features[i, idx] = net.get("flow_idle_timeout", 0); idx += 1
            
            # Protocol indicators (one-hot)
            proto = net.get("protocol", "").lower()
            if proto == "tcp":
                features[i, idx] = 1; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
            elif proto == "udp":
                features[i, idx] = 0; idx += 1
                features[i, idx] = 1; idx += 1
                features[i, idx] = 0; idx += 1
            elif proto == "icmp":
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 1; idx += 1
            else:
                idx += 3  # Skip all protocol indicators
            
            # Port ranges
            dst_port = net.get("dst_port", 0)
            if 0 <= dst_port <= 1023:  # Well-known ports
                features[i, idx] = 1; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
            elif 1024 <= dst_port <= 49151:  # Registered ports
                features[i, idx] = 0; idx += 1
                features[i, idx] = 1; idx += 1
                features[i, idx] = 0; idx += 1
            else:  # Dynamic ports
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 1; idx += 1
            
            # TCP flags
            if "tcp_flags" in net:
                flags = net["tcp_flags"]
                features[i, idx] = int(flags.get("syn", False)); idx += 1
                features[i, idx] = int(flags.get("ack", False)); idx += 1
                features[i, idx] = int(flags.get("fin", False)); idx += 1
                features[i, idx] = int(flags.get("rst", False)); idx += 1
                features[i, idx] = int(flags.get("psh", False)); idx += 1
                features[i, idx] = int(flags.get("urg", False)); idx += 1
            else:
                idx += 6  # Skip all TCP flag features
            
            # Connection features
            features[i, idx] = net.get("connections_per_second", 0); idx += 1
            features[i, idx] = len(net.get("dest_ips", [])); idx += 1
            features[i, idx] = int(net.get("is_outbound", False)); idx += 1
            
            # Advanced statistical features
            if "packet_lengths" in net:
                lengths = net["packet_lengths"]
                if lengths:
                    features[i, idx] = np.mean(lengths); idx += 1
                    features[i, idx] = np.std(lengths); idx += 1
                    features[i, idx] = skew(lengths) if len(lengths) > 2 else 0; idx += 1
                    features[i, idx] = kurtosis(lengths) if len(lengths) > 2 else 0; idx += 1
                else:
                    idx += 4  # Skip statistical features if no data
            else:
                idx += 4  # Skip statistical features if no data
            
            # Leave remaining features as zeros
        
        return features
    
    def _extract_auth_features(self, events: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from authentication events"""
        features = np.zeros((len(events), self.input_dim))
        
        for i, event in enumerate(events):
            if "authentication" not in event:
                continue
            
            auth = event["authentication"]
            
            # Basic auth features
            idx = 0
            features[i, idx] = int(auth.get("success", True)); idx += 1
            features[i, idx] = int(auth.get("mfa_used", False)); idx += 1
            features[i, idx] = auth.get("attempts", 1); idx += 1
            
            # Account type
            features[i, idx] = int(auth.get("is_admin_account", False)); idx += 1
            features[i, idx] = int(auth.get("is_service_account", False)); idx += 1
            features[i, idx] = int(auth.get("is_privileged", False)); idx += 1
            
            # Auth type (one-hot)
            auth_type = auth.get("type", "").lower()
            if auth_type == "password":
                features[i, idx] = 1; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
            elif auth_type == "key":
                features[i, idx] = 0; idx += 1
                features[i, idx] = 1; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
            elif auth_type == "token":
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 1; idx += 1
                features[i, idx] = 0; idx += 1
            elif auth_type == "certificate":
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 1; idx += 1
            else:
                idx += 4  # Skip all auth type indicators
            
            # Temporal features
            features[i, idx] = int(auth.get("time_is_unusual", False)); idx += 1
            features[i, idx] = int(auth.get("source_is_new", False)); idx += 1
            
            # Source info
            src = event.get("source", {})
            features[i, idx] = int(src.get("is_known_location", True)); idx += 1
            features[i, idx] = int(src.get("is_tor_exit", False)); idx += 1
            features[i, idx] = int(src.get("is_proxy", False)); idx += 1
            features[i, idx] = int(src.get("is_cloud_provider", False)); idx += 1
            
            # Leave remaining features as zeros
        
        return features
    
    def _extract_api_features(self, events: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from API call events"""
        features = np.zeros((len(events), self.input_dim))
        
        for i, event in enumerate(events):
            if "api" not in event:
                continue
            
            api = event["api"]
            
            # API call features
            idx = 0
            features[i, idx] = int(api.get("is_admin_api", False)); idx += 1
            features[i, idx] = int(api.get("is_data_access", False)); idx += 1
            features[i, idx] = int(api.get("is_security_config", False)); idx += 1
            features[i, idx] = int(api.get("is_network_config", False)); idx += 1
            features[i, idx] = int(api.get("success", True)); idx += 1
            features[i, idx] = api.get("rate", 0); idx += 1  # calls per minute
            features[i, idx] = api.get("data_volume", 0); idx += 1  # in KB
            
            # API operation type (one-hot)
            op_type = api.get("operation_type", "").lower()
            if op_type == "create":
                features[i, idx] = 1; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
            elif op_type == "read":
                features[i, idx] = 0; idx += 1
                features[i, idx] = 1; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
            elif op_type == "update":
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 1; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
            elif op_type == "delete":
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 1; idx += 1
                features[i, idx] = 0; idx += 1
            elif op_type == "list":
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 1; idx += 1
            else:
                idx += 5  # Skip all operation type indicators
            
            # Error codes
            if not api.get("success", True):
                error_code = str(api.get("error_code", ""))
                if error_code.startswith("4"):  # 4xx client errors
                    features[i, idx] = 1; idx += 1
                    features[i, idx] = 0; idx += 1
                elif error_code.startswith("5"):  # 5xx server errors
                    features[i, idx] = 0; idx += 1
                    features[i, idx] = 1; idx += 1
                else:
                    idx += 2  # Skip error code indicators
            else:
                idx += 2  # Skip error code indicators
            
            # Leave remaining features as zeros
        
        return features
    
    def _extract_resource_features(self, events: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from resource usage events"""
        features = np.zeros((len(events), self.input_dim))
        
        for i, event in enumerate(events):
            if "resource" not in event:
                continue
            
            res = event["resource"]
            
            # Resource usage metrics
            idx = 0
            features[i, idx] = res.get("cpu_utilization", 0); idx += 1
            features[i, idx] = res.get("memory_utilization", 0); idx += 1
            features[i, idx] = res.get("disk_io_operations", 0); idx += 1
            features[i, idx] = res.get("network_in", 0); idx += 1
            features[i, idx] = res.get("network_out", 0); idx += 1
            features[i, idx] = res.get("gpu_utilization", 0); idx += 1
            features[i, idx] = res.get("container_count", 0); idx += 1
            features[i, idx] = res.get("instance_count", 0); idx += 1
            
            # Resource type (one-hot)
            res_type = res.get("type", "").lower()
            if res_type == "vm":
                features[i, idx] = 1; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
            elif res_type == "container":
                features[i, idx] = 0; idx += 1
                features[i, idx] = 1; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
            elif res_type == "serverless":
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 1; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
            elif res_type == "storage":
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 1; idx += 1
                features[i, idx] = 0; idx += 1
            elif res_type == "database":
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 0; idx += 1
                features[i, idx] = 1; idx += 1
            else:
                idx += 5  # Skip all resource type indicators
            
            # Statistical features over time series data
            if "metrics_history" in res and res["metrics_history"]:
                history = res["metrics_history"]
                
                if "cpu" in history:
                    cpu_history = history["cpu"]
                    features[i, idx] = np.mean(cpu_history); idx += 1
                    features[i, idx] = np.std(cpu_history); idx += 1
                    features[i, idx] = np.max(cpu_history); idx += 1
                    if len(cpu_history) > 1:
                        features[i, idx] = cpu_history[-1] - cpu_history[0]; idx += 1  # trend
                    else:
                        idx += 1  # Skip trend feature
                else:
                    idx += 4  # Skip CPU history features
                
                if "memory" in history:
                    mem_history = history["memory"]
                    features[i, idx] = np.mean(mem_history); idx += 1
                    features[i, idx] = np.std(mem_history); idx += 1
                    features[i, idx] = np.max(mem_history); idx += 1
                    if len(mem_history) > 1:
                        features[i, idx] = mem_history[-1] - mem_history[0]; idx += 1  # trend
                    else:
                        idx += 1  # Skip trend feature
                else:
                    idx += 4  # Skip memory history features
            else:
                idx += 8  # Skip all history features
            
            # Leave remaining features as zeros
        
        return features
    
    def _extract_generic_features(self, events: List[Dict[str, Any]]) -> np.ndarray:
        """Extract generic features when specific type is unknown"""
        # Create a simple numerical representation of each event
        features = np.zeros((len(events), self.input_dim))
        
        for i, event in enumerate(events):
            # Flatten the event dictionary into a feature vector
            idx = 0
            for key, value in self._flatten_dict(event).items():
                if idx >= self.input_dim:
                    break
                    
                if isinstance(value, (int, float)):
                    features[i, idx] = value
                elif isinstance(value, bool):
                    features[i, idx] = int(value)
                elif isinstance(value, str):
                    # Simple hash-based encoding of strings
                    features[i, idx] = hash(value) % 100  # Modulo to keep values reasonable
                
                idx += 1
        
        return features
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """Flatten a nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def compute_statistical_features(self, time_series_data: np.ndarray) -> np.ndarray:
        """
        Compute statistical features from time series data.
        
        Args:
            time_series_data: Array of time series data [n_samples, n_timesteps]
            
        Returns:
            Array of statistical features [n_samples, n_features]
        """
        if time_series_data.shape[0] == 0:
            return np.zeros((0, 8))  # Empty input
        
        n_samples = time_series_data.shape[0]
        features = np.zeros((n_samples, 8))
        
        for i in range(n_samples):
            ts = time_series_data[i]
            
            # Basic statistics
            features[i, 0] = np.mean(ts)
            features[i, 1] = np.std(ts)
            features[i, 2] = np.min(ts)
            features[i, 3] = np.max(ts)
            
            # More advanced statistics
            if len(ts) > 1:
                features[i, 4] = np.median(ts)
                features[i, 5] = skew(ts) if len(ts) > 2 else 0
                features[i, 6] = kurtosis(ts) if len(ts) > 2 else 0
                
                # Trend: simple linear coefficient
                x = np.arange(len(ts))
                try:
                    slope, _, _, _, _ = np.polyfit(x, ts, 1, full=True)
                    features[i, 7] = slope
                except:
                    features[i, 7] = 0
        
        return features
    
    def save(self, path: str):
        """
        Save the feature extractor to file.
        
        Args:
            path: Path to save the model
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'pca': self.pca,
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'use_pca': self.use_pca,
                'use_normalization': self.use_normalization
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'FeatureExtractor':
        """
        Load a feature extractor from file.
        
        Args:
            path: Path to load from
            
        Returns:
            Loaded FeatureExtractor
        """
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        extractor = cls(
            input_dim=data['input_dim'],
            output_dim=data['output_dim'],
            use_pca=data['use_pca'],
            use_normalization=data['use_normalization']
        )
        
        extractor.scaler = data['scaler']
        extractor.pca = data['pca']
        
        return extractor