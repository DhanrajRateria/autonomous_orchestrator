import asyncio
import logging
import yaml
import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid
import os

class SecurityAction:
    """Represents a security action to be taken in response to a threat"""
    
    def __init__(self, action_id: str, type: str, target: Dict[str, Any], 
                 parameters: Dict[str, Any] = None):
        """
        Initialize a security action.
        
        Args:
            action_id: Unique identifier for this action
            type: Type of action to take
            target: Target resource information
            parameters: Additional parameters for the action
        """
        self.action_id = action_id
        self.type = type
        self.target = target
        self.parameters = parameters or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "action_id": self.action_id,
            "type": self.type,
            "target": self.target,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityAction":
        """Create action from dictionary"""
        return cls(
            action_id=data["action_id"],
            type=data["type"],
            target=data["target"],
            parameters=data.get("parameters", {})
        )

class ActionResult:
    """Result of a security action execution"""
    
    def __init__(self, action_id: str, status: str, details: Dict[str, Any] = None):
        """
        Initialize an action result.
        
        Args:
            action_id: ID of the action
            status: Status of the execution (success, failed, etc)
            details: Additional details about the result
        """
        self.action_id = action_id
        self.status = status
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "action_id": self.action_id,
            "status": self.status,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }

class ResponsePlan:
    """A plan of security actions to be executed in response to a threat"""
    
    def __init__(self, plan_id: str, threat_id: str, actions: List[SecurityAction] = None):
        """
        Initialize a response plan.
        
        Args:
            plan_id: Unique identifier for this plan
            threat_id: ID of the threat this plan responds to
            actions: List of security actions to execute
        """
        self.plan_id = plan_id
        self.threat_id = threat_id
        self.actions = actions or []
        self.created_at = datetime.now()
    
    def add_action(self, action: SecurityAction):
        """Add an action to the plan"""
        self.actions.append(action)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "plan_id": self.plan_id,
            "threat_id": self.threat_id,
            "created_at": self.created_at.isoformat(),
            "actions": [action.to_dict() for action in self.actions]
        }

class ActionExecutor:
    """
    Executor for security actions in response to detected threats.
    Determines appropriate responses and executes them on cloud resources.
    """
    
    def __init__(self, policy_path: str):
        """
        Initialize the action executor.
        
        Args:
            policy_path: Path to response policy configuration
        """
        self.logger = logging.getLogger("response.executor")
        self.policy_path = policy_path
        
        # Load response policies
        self.policies = self._load_policies()
        
        # Track action history
        self.action_history = []
        self._history_lock = asyncio.Lock()
        
        self.logger.info(f"Action executor initialized with {len(self.policies)} policies")
    
    def _load_policies(self) -> List[Dict[str, Any]]:
        """Load response policies from configuration file"""
        # Create default policy file if it doesn't exist
        if not os.path.exists(self.policy_path):
            self.logger.warning(f"Policy file not found at {self.policy_path}, creating default")
            os.makedirs(os.path.dirname(self.policy_path), exist_ok=True)
            with open(self.policy_path, 'w') as f:
                yaml.safe_dump(self._get_default_policies(), f)
        
        try:
            with open(self.policy_path, 'r') as f:
                if self.policy_path.endswith('.yaml') or self.policy_path.endswith('.yml'):
                    policies = yaml.safe_load(f)
                elif self.policy_path.endswith('.json'):
                    policies = json.load(f)
                else:
                    raise ValueError("Policy file must be YAML or JSON")
            
            self.logger.info(f"Loaded {len(policies)} response policies")
            return policies
        except Exception as e:
            self.logger.error(f"Error loading policies: {e}", exc_info=True)
            return self._get_default_policies()
    
    def _get_default_policies(self) -> List[Dict[str, Any]]:
        """Get default response policies if none are configured"""
        return [
            {
                "id": "policy-001",
                "name": "Unauthorized Access Response",
                "description": "Response actions for unauthorized access attempts",
                "triggers": {
                    "categories": ["unauthorized_access", "credential_compromise"],
                    "severity_min": "medium"
                },
                "actions": [
                    {
                        "type": "block_ip",
                        "description": "Block source IP address",
                        "parameters": {
                            "duration": 3600  # 1 hour in seconds
                        }
                    },
                    {
                        "type": "disable_credentials",
                        "description": "Disable compromised credentials",
                        "parameters": {
                            "revoke_sessions": True
                        }
                    },
                    {
                        "type": "notify_security_team",
                        "description": "Notify security team of the incident",
                        "parameters": {
                            "priority": "high",
                            "include_details": True
                        }
                    }
                ]
            },
            {
                "id": "policy-002",
                "name": "Data Exfiltration Response",
                "description": "Response actions for potential data exfiltration",
                "triggers": {
                    "categories": ["data_exfiltration", "api_abuse"],
                    "severity_min": "high"
                },
                "actions": [
                    {
                        "type": "restrict_permissions",
                        "description": "Remove data access permissions",
                        "parameters": {
                            "scope": "data_resources"
                        }
                    },
                    {
                        "type": "snapshot_resources",
                        "description": "Create snapshots of affected resources for forensics",
                        "parameters": {
                            "include_logs": True
                        }
                    },
                    {
                        "type": "quarantine_instance",
                        "description": "Isolate instance from network",
                        "parameters": {
                            "maintain_service": False
                        }
                    }
                ]
            },
            {
                "id": "policy-003",
                "name": "Resource Abuse Response",
                "description": "Response actions for resource abuse or crypto mining",
                "triggers": {
                    "categories": ["resource_hijacking", "malware"],
                    "severity_min": "medium"
                },
                "actions": [
                    {
                        "type": "terminate_processes",
                        "description": "Terminate suspicious processes",
                        "parameters": {
                            "process_pattern": ["crypto", "miner"]
                        }
                    },
                    {
                        "type": "quarantine_instance",
                        "description": "Isolate instance from network",
                        "parameters": {
                            "maintain_service": True
                        }
                    },
                    {
                        "type": "restore_from_backup",
                        "description": "Restore instance from clean backup",
                        "parameters": {
                            "verify_first": True
                        }
                    }
                ]
            }
        ]
    
    async def create_response_plan(self, threat: Any, provider_id: str, 
                                 provider_type: str) -> ResponsePlan:
        """
        Create a response plan for a detected threat.
        
        Args:
            threat: The detected threat to respond to
            provider_id: ID of the cloud provider
            provider_type: Type of cloud provider (aws, azure, gcp)
            
        Returns:
            A response plan with actions to execute
        """
        plan_id = str(uuid.uuid4())
        plan = ResponsePlan(plan_id=plan_id, threat_id=threat.id)
        
        try:
            # Find matching policies
            matching_policies = []
            for policy in self.policies:
                if self._policy_matches_threat(policy, threat):
                    matching_policies.append(policy)
            
            if not matching_policies:
                self.logger.info(f"No matching policies for threat {threat.id} ({threat.category.value})")
                return plan
            
            self.logger.info(f"Found {len(matching_policies)} matching policies for threat {threat.id}")
            
            # Add actions from matching policies
            for policy in matching_policies:
                for action_spec in policy["actions"]:
                    action_id = str(uuid.uuid4())
                    action = SecurityAction(
                        action_id=action_id,
                        type=action_spec["type"],
                        target=self._get_action_target(threat, action_spec, provider_id),
                        parameters={
                            **action_spec.get("parameters", {}),
                            "provider_type": provider_type,
                            "provider_id": provider_id,
                            "policy_id": policy["id"]
                        }
                    )
                    plan.add_action(action)
            
            self.logger.info(f"Created response plan with {len(plan.actions)} actions for threat {threat.id}")
            
        except Exception as e:
            self.logger.error(f"Error creating response plan: {e}", exc_info=True)
        
        return plan
    
    def _policy_matches_threat(self, policy: Dict[str, Any], threat: Any) -> bool:
        """Check if a policy applies to a given threat"""
        triggers = policy.get("triggers", {})
        
        # Check category
        if "categories" in triggers:
            if threat.category.value not in triggers["categories"]:
                return False
        
        # Check severity
        if "severity_min" in triggers:
            severity_levels = ["low", "medium", "high", "critical"]
            threat_sev_idx = severity_levels.index(threat.severity.value)
            min_sev_idx = severity_levels.index(triggers["severity_min"])
            
            if threat_sev_idx < min_sev_idx:
                return False
        
        # Check confidence threshold
        if "confidence_min" in triggers:
            if threat.confidence < triggers["confidence_min"]:
                return False
        
        return True
    
    def _get_action_target(self, threat: Any, action_spec: Dict[str, Any], 
                         provider_id: str) -> Dict[str, Any]:
        """Determine the target for an action based on the threat"""
        # Start with affected resources from the threat
        target = {
            "provider_id": provider_id,
            "resources": threat.affected_resources
        }
        
        # Add source information if available
        if hasattr(threat, "source_info") and threat.source_info:
            if "ip" in threat.source_info:
                target["source_ip"] = threat.source_info["ip"]
            if "user_id" in threat.source_info:
                target["user_id"] = threat.source_info["user_id"]
        
        # Add additional targeting from raw data
        if hasattr(threat, "raw_data") and threat.raw_data:
            if "event" in threat.raw_data:
                event = threat.raw_data["event"]
                if "user_id" in event:
                    target["user_id"] = event["user_id"]
                if "resource_id" in event:
                    if "resource_id" not in target:
                        target["resource_id"] = event["resource_id"]
        
        return target
    
    async def record_action_result(self, action: SecurityAction, 
                                 result: ActionResult):
        """
        Record the result of an executed action.
        
        Args:
            action: The security action that was executed
            result: Result of the action execution
        """
        async with self._history_lock:
            self.action_history.append({
                "action": action.to_dict(),
                "result": result.to_dict(),
                "timestamp": datetime.now().isoformat()
            })
            
            # Trim history if it gets too large
            if len(self.action_history) > 1000:
                self.action_history = self.action_history[-1000:]
    
    async def get_action_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent action execution history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of action history items
        """
        async with self._history_lock:
            # Return most recent items up to the limit
            return self.action_history[-limit:]