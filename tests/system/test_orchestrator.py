import unittest
import asyncio
import os
import sys
import tempfile
import shutil
import yaml
import json
from unittest.mock import MagicMock, patch
import logging
import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.orchestrator import SecurityOrchestrator
from core.config import OrchestrationConfig
from cloud_providers.connector_base import CloudProviderConnector

# Disable logs during tests
logging.getLogger().setLevel(logging.CRITICAL)

class MockCloudConnector(CloudProviderConnector):
    """Mock implementation of cloud connector for testing"""
    
    def __init__(self, provider_config):
        super().__init__(provider_config)
        self.events_to_return = []
        self.actions_executed = []
        self.should_fail = False
    
    async def initialize(self) -> bool:
        if self.should_fail:
            return False
        self.is_connected = True
        return True
    
    async def disconnect(self) -> bool:
        self.is_connected = False
        return True
    
    async def collect_security_events(self) -> list:
        self._update_stats("events_collected", len(self.events_to_return))
        return self.events_to_return
    
    async def execute_security_action(self, action):
        self.actions_executed.append(action)
        self._update_stats("actions_executed")
        
        from response_engine.action_executor import ActionResult
        return ActionResult(
            action.action_id,
            "successful" if not self.should_fail else "failed",
            {"test": True}
        )
    
    async def get_resource_info(self, resource_id):
        return {"id": resource_id, "type": "test_resource"}

class TestOrchestratorSystem(unittest.TestCase):
    """
    System tests for the security orchestrator.
    Tests the complete system functioning with mock cloud connectors.
    """
    
    def setUp(self):
        """Set up the test environment"""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        config_data = {
            "instance_id": "test-instance",
            "use_encryption_for_analysis": True,
            "event_polling_interval": 1,  # 1 second for faster tests
            "model_update_interval": 10,  # 10 seconds for faster tests
            "detection_threshold": 0.7,
            "response_threshold": 0.8,
            "crypto_settings": {
                "key_size": 1024,  # Smaller for faster tests
                "security_level": 128
            },
            "federated_learning": {
                "min_clients": 1,  # Only 1 for testing
                "aggregation_method": "fedavg",
                "rounds_per_update": 1,
                "local_epochs": 1
            },
            "system": {
                "max_workers": 2,
                "task_queue_size": 100
            },
            "cloud_providers": [
                {
                    "provider_id": "test-aws",
                    "provider_type": "aws",
                    "credentials_path": os.path.join(self.temp_dir, "fake_creds"),
                    "enabled_services": ["test"],
                    "region": "test-region"
                },
                {
                    "provider_id": "test-azure",
                    "provider_type": "azure",
                    "credentials_path": os.path.join(self.temp_dir, "fake_creds"),
                    "enabled_services": ["test"],
                    "region": "test-region"
                }
            ],
            "response_policies_path": os.path.join(self.temp_dir, "policies.yaml")
        }
        
        # Write config to file
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Create simple policy file
        with open(os.path.join(self.temp_dir, "policies.yaml"), 'w') as f:
            yaml.dump([{
                "id": "policy-test",
                "name": "Test Policy",
                "triggers": {
                    "categories": ["unauthorized_access"],
                    "severity_min": "medium"
                },
                "actions": [{
                    "type": "test_action",
                    "parameters": {"test": True}
                }]
            }], f)
        
        # Create a fake credentials file
        with open(os.path.join(self.temp_dir, "fake_creds"), 'w') as f:
            f.write("test_credentials")
        
        # Load config
        self.config = OrchestrationConfig.from_file(self.config_path)
        
        # Set up event loop
        self.loop = asyncio.get_event_loop()
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
        
        # Clean up any remaining tasks
        tasks = asyncio.all_tasks(self.loop)
        for task in tasks:
            task.cancel()
    
    @patch('cloud_providers.connector_base.CloudProviderConnector')
    @patch('cloud_providers.aws.connector.AWSConnector', new=MockCloudConnector)
    @patch('cloud_providers.azure.connector.AzureConnector', new=MockCloudConnector)
    def test_orchestrator_initialization(self, mock_connector):
        """Test orchestrator initialization with mock cloud providers"""
        async def _test():
            # Create orchestrator
            orchestrator = SecurityOrchestrator(self.config)
            
            # Initialize cloud connectors
            await orchestrator.initialize_cloud_connectors()
            
            # Verify connectors were initialized
            self.assertEqual(len(orchestrator.cloud_connectors), 2)
            self.assertIn("test-aws", orchestrator.cloud_connectors)
            self.assertIn("test-azure", orchestrator.cloud_connectors)
            
            # Verify each connector is connected
            for connector in orchestrator.cloud_connectors.values():
                self.assertTrue(connector.is_connected)
                
        self.loop.run_until_complete(_test())
    
    @patch('cloud_providers.connector_base.CloudProviderConnector')
    @patch('cloud_providers.aws.connector.AWSConnector', new=MockCloudConnector)
    @patch('cloud_providers.azure.connector.AzureConnector', new=MockCloudConnector)
    def test_event_processing(self, mock_connector):
        """Test collecting and processing security events"""
        async def _test():
            # Create orchestrator
            orchestrator = SecurityOrchestrator(self.config)
            
            # Initialize cloud connectors
            await orchestrator.initialize_cloud_connectors()
            
            # Create test events
            test_events = [
                {
                    "event_type": "test_event",
                    "timestamp": "2025-04-09T12:00:00Z",
                    "severity": 0.8,
                    "authentication": {
                        "success": False,
                        "attempts": 5
                    },
                    "source": {
                        "ip": "192.168.1.100"
                    },
                    "resource_id": "test-resource"
                }
            ]
            
            # Set up connector to return test events
            for connector in orchestrator.cloud_connectors.values():
                connector.events_to_return = test_events.copy()
            
            # Process events manually
            await orchestrator._process_events()
            
            # Verify events were collected
            for connector in orchestrator.cloud_connectors.values():
                self.assertEqual(connector.stats["events_collected"], 1)
            
            # Stop orchestrator
            await orchestrator.stop()
                
        self.loop.run_until_complete(_test())
    
    @patch('cloud_providers.connector_base.CloudProviderConnector')
    @patch('cloud_providers.aws.connector.AWSConnector', new=MockCloudConnector)
    @patch('cloud_providers.azure.connector.AzureConnector', new=MockCloudConnector)
    @patch('threat_detection.analyzer.ThreatAnalyzer.analyze_events')
    def test_threat_response_flow(self, mock_analyze, mock_connector):
        """Test threat detection and response flow"""
        async def _test():
            # Create orchestrator
            orchestrator = SecurityOrchestrator(self.config)
            
            # Initialize cloud connectors
            await orchestrator.initialize_cloud_connectors()
            
            # Mock threat detection
            from threat_detection.analyzer import ThreatDetection, ThreatCategory, ThreatSeverity
            
            # Create a mock threat
            mock_threat = ThreatDetection(
                id="test-threat-1",
                provider_id="test-aws",
                timestamp=datetime.now(),
                category=ThreatCategory.UNAUTHORIZED_ACCESS,
                severity=ThreatSeverity.HIGH,
                confidence=0.9,
                description="Test threat",
                affected_resources=["test-resource"],
                raw_data={"source_info": {"ip": "192.168.1.100"}}
            )
            
            # Set up threat analyzer mock to return our threat
            mock_analyze.return_value = [mock_threat]
            
            # Create test events
            test_events = [
                {
                    "event_type": "test_event",
                    "timestamp": "2025-04-09T12:00:00Z",
                    "severity": 0.8,
                    "authentication": {
                        "success": False,
                        "attempts": 5
                    },
                    "source": {
                        "ip": "192.168.1.100"
                    },
                    "resource_id": "test-resource"
                }
            ]
            
            # Set up connector to return test events
            for connector in orchestrator.cloud_connectors.values():
                connector.events_to_return = test_events.copy()
            
            # Process events manually
            await orchestrator._process_events()
            
            # Verify threat analyzer was called
            mock_analyze.assert_called()
            
            # Verify actions were executed on the connector
            aws_connector = orchestrator.cloud_connectors["test-aws"]
            self.assertEqual(len(aws_connector.actions_executed), 1)
            
            # Verify action details
            action = aws_connector.actions_executed[0]
            self.assertEqual(action.type, "test_action")
            self.assertTrue(action.parameters["test"])
            
            # Stop orchestrator
            await orchestrator.stop()
                
        self.loop.run_until_complete(_test())
    
    @patch('cloud_providers.connector_base.CloudProviderConnector')
    @patch('cloud_providers.aws.connector.AWSConnector', new=MockCloudConnector)
    @patch('cloud_providers.azure.connector.AzureConnector', new=MockCloudConnector)
    def test_full_orchestration_cycle(self, mock_connector):
        """Test a complete orchestration cycle"""
        async def _test():
            # Create orchestrator
            orchestrator = SecurityOrchestrator(self.config)
            
            # Start orchestrator
            await orchestrator.start()
            
            # Verify orchestrator is running
            self.assertTrue(orchestrator.is_running)
            
            # Let it run for a short time
            await asyncio.sleep(3)
            
            # Verify the federated coordinator is running
            self.assertTrue(orchestrator.federated_coordinator.is_running)
            
            # Stop orchestrator
            await orchestrator.stop()
            
            # Verify orchestrator stopped
            self.assertFalse(orchestrator.is_running)
            self.assertFalse(orchestrator.federated_coordinator.is_running)
                
        self.loop.run_until_complete(_test())

if __name__ == '__main__':
    unittest.main()