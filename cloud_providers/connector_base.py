import asyncio
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime

from response_engine.action_executor import SecurityAction, ActionResult

class CloudProviderConnector(ABC):
    """
    Base class for cloud provider connectors.
    Defines the interface for interacting with different cloud providers.
    """
    
    def __init__(self, provider_config: Any):
        """
        Initialize the cloud provider connector.
        
        Args:
            provider_config: Configuration for this cloud provider
        """
        self.provider_id = provider_config.provider_id
        self.provider_type = provider_config.provider_type
        self.credentials_path = provider_config.credentials_path
        self.enabled_services = provider_config.enabled_services
        self.region = provider_config.region
        self.polling_interval = provider_config.polling_interval
        
        self.logger = logging.getLogger(f"cloud.{self.provider_type}.{self.provider_id}")
        self.is_connected = False
        self.last_event_timestamp = datetime.now()
        
        # Stats tracking
        self.stats = {
            "events_collected": 0,
            "actions_executed": 0,
            "connection_errors": 0,
            "last_connection_time": None
        }
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the connection to the cloud provider.
        
        Returns:
            bool: True if initialization was successful
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the cloud provider.
        
        Returns:
            bool: True if disconnection was successful
        """
        pass
    
    @abstractmethod
    async def collect_security_events(self) -> List[Dict[str, Any]]:
        """
        Collect security events from the cloud provider.
        
        Returns:
            List of security events
        """
        pass
    
    @abstractmethod
    async def execute_security_action(self, action: SecurityAction) -> ActionResult:
        """
        Execute a security action on the cloud provider.
        
        Args:
            action: Security action to execute
            
        Returns:
            Result of the action execution
        """
        pass
    
    @abstractmethod
    async def get_resource_info(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific cloud resource.
        
        Args:
            resource_id: ID of the resource to query
            
        Returns:
            Resource information or None if not found
        """
        pass
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """
        Get the current connection status of this provider.
        
        Returns:
            Status information dictionary
        """
        return {
            "provider_id": self.provider_id,
            "provider_type": self.provider_type,
            "connected": self.is_connected,
            "region": self.region,
            "enabled_services": self.enabled_services,
            "stats": self.stats
        }
    
    def _update_stats(self, key: str, increment: int = 1):
        """Update connector statistics"""
        if key in self.stats:
            if isinstance(self.stats[key], int):
                self.stats[key] += increment
            else:
                self.stats[key] = increment