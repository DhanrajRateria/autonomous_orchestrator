import logging
import numpy as np
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import json
from enum import Enum
from collections import defaultdict
import time

class ThreatSeverity(Enum):
    """Enumeration of threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatCategory(Enum):
    """Enumeration of threat categories"""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    MALWARE = "malware"
    DDOS = "ddos"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    CREDENTIAL_COMPROMISE = "credential_compromise"
    API_ABUSE = "api_abuse"
    RESOURCE_HIJACKING = "resource_hijacking"
    SUPPLY_CHAIN = "supply_chain"
    UNKNOWN = "unknown"

class ThreatDetection:
    """Class representing a detected security threat"""
    
    def __init__(self, id: str, provider_id: str,
                 timestamp: datetime, category: ThreatCategory,
                 severity: ThreatSeverity, confidence: float,
                 description: str, affected_resources: List[str],
                 raw_data: Dict[str, Any] = None):
        """
        Initialize a threat detection.
        
        Args:
            id: Unique threat identifier
            provider_id: Cloud provider where threat was detected
            timestamp: Time when threat was detected
            category: Threat category
            severity: Threat severity
            confidence: Confidence level (0.0-1.0)
            description: Human-readable description
            affected_resources: List of affected cloud resources
            raw_data: Raw detection data for reference
        """
        self.id = id
        self.provider_id = provider_id
        self.timestamp = timestamp
        self.category = category
        self.severity = severity
        self.confidence = confidence
        self.description = description
        self.affected_resources = affected_resources
        self.raw_data = raw_data
        self.response_actions = []
        self.status = "detected"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "provider_id": self.provider_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "description": self.description,
            "affected_resources": self.affected_resources,
            "status": self.status,
            "response_actions": self.response_actions
        }

class ThreatAnalyzer:
    """
    Analyzer for detecting security threats across cloud environments,
    leveraging federated learning models and rule-based detection.
    """
    
    def __init__(self, detection_threshold: float = 0.7, model_provider: Any = None):
        """
        Initialize the threat analyzer.
        
        Args:
            detection_threshold: Confidence threshold for threat detection
            model_provider: Provider of ML models for threat detection
        """
        self.logger = logging.getLogger("threat.analyzer")
        self.detection_threshold = detection_threshold
        self.model_provider = model_provider
        
        # Thread-safe storage for active threats
        self._active_threats = {}
        self._threat_lock = asyncio.Lock()
        
        # Counters for detection statistics
        self._detection_stats = defaultdict(int)
        
        # Load detection rules
        self.rules = self._load_detection_rules()
        
        # Feature extraction parameters
        self.feature_extractors = {
            "network_flow": self._extract_network_features,
            "auth_events": self._extract_auth_features,
            "api_calls": self._extract_api_features,
            "resource_usage": self._extract_resource_features
        }
        
        self.logger.info(f"Threat analyzer initialized with {len(self.rules)} detection rules")
    
    def _load_detection_rules(self) -> List[Dict[str, Any]]:
        """Load threat detection rules from configuration"""
        # In production, load from file or database
        # For demonstration, return sample rules
        return [
            {
                "id": "rule-001",
                "name": "Multiple Failed Login Attempts",
                "description": "Detection of multiple failed authentication attempts",
                "condition": {
                    "type": "threshold",
                    "field": "authentication.failed",
                    "threshold": 5,
                    "window_seconds": 300,
                    "per_entity": "user_id"
                },
                "category": ThreatCategory.UNAUTHORIZED_ACCESS,
                "severity": ThreatSeverity.MEDIUM
            },
            {
                "id": "rule-002",
                "name": "Suspicious API Usage",
                "description": "Detection of API calls associated with data exfiltration",
                "condition": {
                    "type": "sequence",
                    "events": [
                        {"api": "list_buckets"},
                        {"api": "get_object", "count_min": 10, "window_seconds": 60}
                    ]
                },
                "category": ThreatCategory.DATA_EXFILTRATION,
                "severity": ThreatSeverity.HIGH
            },
            {
                "id": "rule-003",
                "name": "Resource Abuse",
                "description": "Detection of abnormal resource usage patterns",
                "condition": {
                    "type": "anomaly",
                    "metric": "cpu_utilization",
                    "sensitivity": 3.0  # Standard deviations from normal
                },
                "category": ThreatCategory.RESOURCE_HIJACKING,
                "severity": ThreatSeverity.MEDIUM
            }
        ]
    
    def _extract_network_features(self, events: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract network flow features for machine learning models.
        
        Args:
            events: List of network events to extract features from
            
        Returns:
            Feature array for ML models
        """
        # Feature extraction logic - would be more complex in real implementation
        features = np.zeros((len(events), 32))
        
        for i, event in enumerate(events):
            if "network" not in event:
                continue
            
            net = event["network"]
            
            # Basic network features
            features[i, 0] = net.get("bytes_in", 0)
            features[i, 1] = net.get("bytes_out", 0)
            features[i, 2] = net.get("packets_in", 0)
            features[i, 3] = net.get("packets_out", 0)
            features[i, 4] = len(net.get("dest_ips", []))
            features[i, 5] = net.get("connections_per_second", 0)
            
            # Protocol indicators (one-hot)
            proto = net.get("protocol", "").lower()
            if proto == "tcp":
                features[i, 6] = 1
            elif proto == "udp":
                features[i, 7] = 1
            elif proto == "icmp":
                features[i, 8] = 1
                
            # Port ranges
            dst_port = net.get("dst_port", 0)
            if 0 <= dst_port <= 1023:  # Well-known ports
                features[i, 9] = 1
            elif 1024 <= dst_port <= 49151:  # Registered ports
                features[i, 10] = 1
            else:  # Dynamic ports
                features[i, 11] = 1
            
            # Timing features
            features[i, 12] = net.get("duration_ms", 0) / 1000.0  # Convert to seconds
            features[i, 13] = net.get("flow_idle_timeout", 0)
            
            # Flow direction and flags
            features[i, 14] = int(net.get("is_outbound", False))
            
            if "tcp_flags" in net:
                flags = net["tcp_flags"]
                features[i, 15] = int(flags.get("syn", False))
                features[i, 16] = int(flags.get("ack", False))
                features[i, 17] = int(flags.get("fin", False))
                features[i, 18] = int(flags.get("rst", False))
            
            # Additional features would be extracted here
        
        return features
    
    def _extract_auth_features(self, events: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from authentication events"""
        features = np.zeros((len(events), 32))
        
        for i, event in enumerate(events):
            if "authentication" not in event:
                continue
            
            auth = event["authentication"]
            
            # Authentication features
            features[i, 0] = int(auth.get("success", True))
            features[i, 1] = int(auth.get("mfa_used", False))
            features[i, 2] = auth.get("attempts", 1)
            features[i, 3] = int(auth.get("is_admin_account", False))
            features[i, 4] = int(auth.get("is_service_account", False))
            features[i, 5] = int(auth.get("source_is_new", False))
            features[i, 6] = int(auth.get("time_is_unusual", False))
            
            # Auth type (one-hot)
            auth_type = auth.get("type", "").lower()
            if auth_type == "password":
                features[i, 7] = 1
            elif auth_type == "key":
                features[i, 8] = 1
            elif auth_type == "token":
                features[i, 9] = 1
            elif auth_type == "certificate":
                features[i, 10] = 1
            
            # Source info
            src = event.get("source", {})
            features[i, 11] = int(src.get("is_known_location", True))
            features[i, 12] = int(src.get("is_tor_exit", False))
            features[i, 13] = int(src.get("is_proxy", False))
            features[i, 14] = int(src.get("is_cloud_provider", False))
            
            # Additional features would be extracted here
        
        return features
    
    def _extract_api_features(self, events: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from API call events"""
        features = np.zeros((len(events), 32))
        
        for i, event in enumerate(events):
            if "api" not in event:
                continue
            
            api = event["api"]
            
            # API call features
            features[i, 0] = int(api.get("is_admin_api", False))
            features[i, 1] = int(api.get("is_data_access", False))
            features[i, 2] = int(api.get("is_security_config", False))
            features[i, 3] = int(api.get("is_network_config", False))
            features[i, 4] = int(api.get("success", True))
            features[i, 5] = api.get("rate", 0)  # calls per minute
            features[i, 6] = api.get("data_volume", 0)  # in KB
            
            # API operation type (one-hot)
            op_type = api.get("operation_type", "").lower()
            if op_type == "create":
                features[i, 7] = 1
            elif op_type == "read":
                features[i, 8] = 1
            elif op_type == "update":
                features[i, 9] = 1
            elif op_type == "delete":
                features[i, 10] = 1
            elif op_type == "list":
                features[i, 11] = 1
            
            # Error codes
            if not api.get("success", True):
                error_code = str(api.get("error_code", ""))
                if error_code.startswith("4"):  # 4xx client errors
                    features[i, 12] = 1
                elif error_code.startswith("5"):  # 5xx server errors
                    features[i, 13] = 1
            
            # Additional features would be extracted here
        
        return features
    
    def _extract_resource_features(self, events: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from resource usage events"""
        features = np.zeros((len(events), 32))
        
        for i, event in enumerate(events):
            if "resource" not in event:
                continue
            
            res = event["resource"]
            
            # Resource usage features
            features[i, 0] = res.get("cpu_utilization", 0)
            features[i, 1] = res.get("memory_utilization", 0)
            features[i, 2] = res.get("disk_io_operations", 0)
            features[i, 3] = res.get("network_in", 0)
            features[i, 4] = res.get("network_out", 0)
            features[i, 5] = res.get("gpu_utilization", 0)
            features[i, 6] = res.get("container_count", 0)
            features[i, 7] = res.get("instance_count", 0)
            
            # Resource type (one-hot)
            res_type = res.get("type", "").lower()
            if res_type == "vm":
                features[i, 8] = 1
            elif res_type == "container":
                features[i, 9] = 1
            elif res_type == "serverless":
                features[i, 10] = 1
            elif res_type == "storage":
                features[i, 11] = 1
            elif res_type == "database":
                features[i, 12] = 1
            
            # Additional features would be extracted here
        
        return features
    
    async def analyze_events(self, provider_id: str, events: List[Dict[str, Any]], 
                            use_encryption: bool = False) -> List[ThreatDetection]:
        """
        Analyze security events to detect threats.
        
        Args:
            provider_id: Cloud provider ID source of the events
            events: List of security events to analyze
            use_encryption: Whether to use homomorphic encryption for analysis
            
        Returns:
            List of detected threats
        """
        if not events:
            return []
        
        detected_threats = []
        
        # Process events with rules
        rule_threats = await self._apply_detection_rules(provider_id, events)
        detected_threats.extend(rule_threats)
        
        # Process events with ML models if available
        if self.model_provider:
            ml_threats = await self._apply_ml_detection(provider_id, events, use_encryption)
            detected_threats.extend(ml_threats)
        
        # Update detection statistics
        for threat in detected_threats:
            self._detection_stats[threat.category.value] += 1
        
        # Store active threats
        async with self._threat_lock:
            for threat in detected_threats:
                self._active_threats[threat.id] = threat
        
        return detected_threats
    
    async def _apply_detection_rules(self, provider_id: str, 
                                    events: List[Dict[str, Any]]) -> List[ThreatDetection]:
        """Apply rule-based detection to events"""
        detected_threats = []
        
        # Group events by entity (user, resource, etc.) for stateful analysis
        events_by_user = defaultdict(list)
        events_by_resource = defaultdict(list)
        
        for event in events:
            if "user_id" in event:
                events_by_user[event["user_id"]].append(event)
            if "resource_id" in event:
                events_by_resource[event["resource_id"]].append(event)
        
        # Apply threshold-based rules
        for rule in self.rules:
            if rule["condition"]["type"] == "threshold":
                cond = rule["condition"]
                field_path = cond["field"].split(".")
                window_sec = cond["window_seconds"]
                threshold = cond["threshold"]
                
                # Determine which events to analyze based on per_entity setting
                if cond.get("per_entity") == "user_id":
                    targets = events_by_user
                elif cond.get("per_entity") == "resource_id":
                    targets = events_by_resource
                else:
                    targets = {"all": events}
                
                # Check each entity against the threshold
                for entity_id, entity_events in targets.items():
                    # Filter events within time window
                    now = datetime.now()
                    window_start = now - timedelta(seconds=window_sec)
                    
                    count = 0
                    for event in entity_events:
                        # Navigate to the field using the path
                        value = event
                        valid = True
                        for key in field_path:
                            if key not in value:
                                valid = False
                                break
                            value = value[key]
                        
                        # Count if the value exists and is within the time window
                        if valid:
                            event_time = datetime.fromisoformat(event["timestamp"]) if "timestamp" in event else now
                            if event_time >= window_start:
                                count += 1
                    
                    # Check if threshold was exceeded
                    if count >= threshold:
                        # Create a threat detection
                        threat_id = str(uuid.uuid4())
                        threat = ThreatDetection(
                            id=threat_id,
                            provider_id=provider_id,
                            timestamp=datetime.now(),
                            category=rule["category"],
                            severity=rule["severity"],
                            confidence=0.85,  # Rule-based detections have high confidence
                            description=f"{rule['name']}: {count} occurrences detected for {entity_id}",
                            affected_resources=[entity_id] if entity_id != "all" else [],
                            raw_data={"rule_id": rule["id"], "events": entity_events[:10]}  # Include first 10 events
                        )
                        detected_threats.append(threat)
        
        # Apply sequence-based rules
        # (Implementation would be more complex in a real system)
        
        return detected_threats
    
    async def _apply_ml_detection(self, provider_id: str, events: List[Dict[str, Any]], 
                                 use_encryption: bool = False) -> List[ThreatDetection]:
        """Apply machine learning models to detect threats"""
        detected_threats = []
        
        if not self.model_provider:
            return []
        
        try:
            # Group events by type for appropriate feature extraction
            events_by_type = defaultdict(list)
            for event in events:
                event_type = self._determine_event_type(event)
                events_by_type[event_type].append(event)
            
            # Process each event type with the appropriate model
            for event_type, type_events in events_by_type.items():
                if not type_events:
                    continue
                
                # Extract features for this event type
                extractor = self.feature_extractors.get(event_type)
                if not extractor:
                    continue
                
                features = extractor(type_events)
                if features.shape[0] == 0:
                    continue
                
                # Get the appropriate model
                model_name = self._get_model_for_event_type(event_type)
                model = self.model_provider.get_current_model(model_name)
                
                # Perform inference (with or without encryption)
                if use_encryption:
                    # This would use homomorphic encryption for inference
                    # In practice, this would require special HE-friendly models
                    results = await model.encrypted_predict(features)
                else:
                    results = await model.predict(features)
                
                # Process prediction results
                for i, score in enumerate(results):
                    if score >= self.detection_threshold:
                        event = type_events[i]
                        threat_id = str(uuid.uuid4())
                        
                        # Determine category and severity based on model output
                        category, severity = self._categorize_ml_detection(event_type, score)
                        
                        # Create threat detection
                        threat = ThreatDetection(
                            id=threat_id,
                            provider_id=provider_id,
                            timestamp=datetime.now(),
                            category=category,
                            severity=severity,
                            confidence=float(score),
                            description=f"ML model detected abnormal {event_type} pattern",
                            affected_resources=self._extract_affected_resources(event),
                            raw_data={"event": event, "score": float(score)}
                        )
                        detected_threats.append(threat)
        
        except Exception as e:
            self.logger.error(f"Error in ML threat detection: {e}", exc_info=True)
        
        return detected_threats
    
    def _determine_event_type(self, event: Dict[str, Any]) -> str:
        """Determine the type of security event"""
        if "network" in event:
            return "network_flow"
        elif "authentication" in event:
            return "auth_events"
        elif "api" in event:
            return "api_calls"
        elif "resource" in event:
            return "resource_usage"
        else:
            return "unknown"
    
    def _get_model_for_event_type(self, event_type: str) -> str:
        """Get the appropriate model name for an event type"""
        if event_type == "network_flow":
            return "network_ids"
        elif event_type in ["auth_events", "api_calls", "resource_usage"]:
            return "anomaly_detector"
        else:
            return "anomaly_detector"  # Default
    
    def _categorize_ml_detection(self, event_type: str, score: float) -> Tuple[ThreatCategory, ThreatSeverity]:
        """Determine threat category and severity for ML detection"""
        # Map event types to threat categories
        category_map = {
            "network_flow": ThreatCategory.UNAUTHORIZED_ACCESS,
            "auth_events": ThreatCategory.CREDENTIAL_COMPROMISE,
            "api_calls": ThreatCategory.API_ABUSE,
            "resource_usage": ThreatCategory.RESOURCE_HIJACKING
        }
        
        category = category_map.get(event_type, ThreatCategory.UNKNOWN)
        
        # Determine severity based on confidence score
        if score >= 0.9:
            severity = ThreatSeverity.CRITICAL
        elif score >= 0.8:
            severity = ThreatSeverity.HIGH
        elif score >= 0.7:
            severity = ThreatSeverity.MEDIUM
        else:
            severity = ThreatSeverity.LOW
        
        return category, severity
    
    def _extract_affected_resources(self, event: Dict[str, Any]) -> List[str]:
        """Extract affected resources from an event"""
        resources = []
        
        if "resource_id" in event:
            resources.append(event["resource_id"])
        
        if "target" in event and "id" in event["target"]:
            resources.append(event["target"]["id"])
        
        return resources
    
    async def update_threat_state(self, threat_id: str, action_taken: str, action_result: str) -> bool:
        """
        Update the state of a threat with response action information.
        
        Args:
            threat_id: ID of the threat to update
            action_taken: Action that was taken
            action_result: Result of the action
            
        Returns:
            bool: True if the update was successful
        """
        async with self._threat_lock:
            if threat_id not in self._active_threats:
                return False
            
            threat = self._active_threats[threat_id]
            threat.response_actions.append({
                "action": action_taken,
                "result": action_result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update status based on action result
            if action_result == "successful":
                threat.status = "mitigated"
            elif action_result == "failed":
                threat.status = "active"
            else:
                threat.status = "in_progress"
        
        return True
    
    async def get_active_threats(self) -> List[ThreatDetection]:
        """Get list of currently active threats"""
        async with self._threat_lock:
            return list(self._active_threats.values())
    
    async def get_threat(self, threat_id: str) -> Optional[ThreatDetection]:
        """Get a specific threat by ID"""
        async with self._threat_lock:
            return self._active_threats.get(threat_id)
    
    async def clear_resolved_threats(self) -> int:
        """
        Clear threats that have been successfully mitigated.
        
        Returns:
            int: Number of threats removed
        """
        cleared_count = 0
        
        async with self._threat_lock:
            threat_ids = list(self._active_threats.keys())
            for threat_id in threat_ids:
                threat = self._active_threats[threat_id]
                if threat.status == "mitigated":
                    del self._active_threats[threat_id]
                    cleared_count += 1
        
        return cleared_count
    
    async def get_detection_statistics(self) -> Dict[str, int]:
        """Get statistics about detected threats"""
        return dict(self._detection_stats)