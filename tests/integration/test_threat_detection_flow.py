import unittest
import asyncio
import os
import sys
import tempfile
import shutil
import json
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from threat_detection.analyzer import ThreatAnalyzer, ThreatCategory, ThreatSeverity
from response_engine.action_executor import ActionExecutor, SecurityAction, ResponsePlan
from federated_learning.models.anomaly_detector import AnomalyDetectionModel

class TestThreatDetectionFlow(unittest.TestCase):
    """
    Integration tests for the threat detection and response flow.
    Tests the interaction between the threat analyzer and response engine.
    """
    
    def setUp(self):
        """Set up the test environment"""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create policy file
        self.policy_path = os.path.join(self.temp_dir, "policies.yaml")
        with open(self.policy_path, 'w') as f:
            f.write("""
            - id: policy-001
              name: Test Policy
              description: Policy for testing
              triggers:
                categories: [unauthorized_access, credential_compromise]
                severity_min: medium
              actions:
                - type: block_ip
                  description: Block source IP
                  parameters:
                    duration: 3600
                - type: disable_credentials
                  description: Disable compromised credentials
                  parameters:
                    revoke_sessions: true
            """)
        
        # Create model for testing
        self.model = AnomalyDetectionModel(input_dim=32, hidden_dim=64)
        
        # Create mock model provider
        self.model_provider = MagicMock()
        self.model_provider.get_current_model.return_value = self.model
        
        # Create the threat analyzer
        self.analyzer = ThreatAnalyzer(detection_threshold=0.7, model_provider=self.model_provider)
        
        # Create the action executor
        self.action_executor = ActionExecutor(policy_path=self.policy_path)
        
        # Sample security events for testing
        self.auth_events = []
        for i in range(6):  # Create 6 failed login events
            self.auth_events.append({
                "event_type": "authentication",
                "timestamp": datetime.now().isoformat(),
                "user_id": "test_user",
                "authentication": {
                    "success": False,
                    "attempts": 1,
                    "type": "password",
                    "source_is_new": True
                },
                "source": {
                    "ip": "192.168.1.100",
                    "is_known_location": False
                }
            })
        
        # Get event loop
        self.loop = asyncio.get_event_loop()
    
    def tearDown(self):
        """Clean up after tests"""
        shutil.rmtree(self.temp_dir)
    
    def test_detect_and_respond_flow(self):
        """Test full flow from detection to response"""
        async def _test():
            # Step 1: Detect threats from events
            provider_id = "test-aws"
            threats = await self.analyzer.analyze_events(provider_id, self.auth_events)
            
            # Verify we have at least one threat
            self.assertGreater(len(threats), 0)
            threat = threats[0]
            
            # Verify threat properties
            self.assertEqual(threat.provider_id, provider_id)
            self.assertEqual(threat.category, ThreatCategory.UNAUTHORIZED_ACCESS)
            self.assertGreaterEqual(threat.confidence, 0.7)  # Should meet threshold
            
            # Step 2: Generate response plan
            response_plan = await self.action_executor.create_response_plan(
                threat=threat,
                provider_id=provider_id,
                provider_type="aws"
            )
            
            # Verify response plan
            self.assertIsNotNone(response_plan)
            self.assertEqual(response_plan.threat_id, threat.id)
            
            # Should have actions based on the policy
            self.assertGreaterEqual(len(response_plan.actions), 1)
            
            # Verify first action is to block IP (based on policy)
            action = response_plan.actions[0]
            self.assertEqual(action.type, "block_ip")
            self.assertEqual(action.parameters["duration"], 3600)
            
            # Step 3: Execute action (mock)
            mock_connector = MagicMock()
            mock_connector.provider_type = "aws"
            
            # Mock successful execution
            mock_result = MagicMock()
            mock_result.status = "successful"
            mock_connector.execute_security_action.return_value = asyncio.Future()
            mock_connector.execute_security_action.return_value.set_result(mock_result)
            
            # Execute action
            await mock_connector.execute_security_action(action)
            
            # Step 4: Update threat status
            updated = await self.analyzer.update_threat_state(
                threat_id=threat.id,
                action_taken=action.type,
                action_result="successful"
            )
            
            # Verify update was successful
            self.assertTrue(updated)
            
            # Step 5: Verify threat status changed
            updated_threat = await self.analyzer.get_threat(threat.id)
            self.assertEqual(updated_threat.status, "mitigated")
            self.assertEqual(len(updated_threat.response_actions), 1)
            
        self.loop.run_until_complete(_test())
    
    @patch('threat_detection.analyzer.ThreatAnalyzer._apply_ml_detection')
    def test_ml_detection_integration(self, mock_ml_detection):
        """Test ML model integration in threat detection"""
        async def _test():
            # Create a mock ML detection
            ml_threat = MagicMock()
            ml_threat.id = "ml-threat-1"
            ml_threat.provider_id = "test-aws"
            ml_threat.category = ThreatCategory.UNAUTHORIZED_ACCESS
            ml_threat.severity = ThreatSeverity.HIGH
            ml_threat.confidence = 0.9
            ml_threat.description = "ML detected anomaly"
            ml_threat.affected_resources = ["test_user"]
            ml_threat.to_dict = MagicMock(return_value={
                "id": ml_threat.id,
                "provider_id": ml_threat.provider_id,
                "category": ml_threat.category.value,
                "severity": ml_threat.severity.value,
                "confidence": ml_threat.confidence
            })
            
            # Mock the ML detection method
            mock_ml_detection.return_value = [ml_threat]
            
            # Detect threats with ML
            provider_id = "test-aws"
            threats = await self.analyzer.analyze_events(provider_id, self.auth_events, use_encryption=True)
            
            # Verify ML was called with encryption flag
            mock_ml_detection.assert_called_once_with(provider_id, self.auth_events, True)
            
            # Verify threat was detected
            self.assertEqual(len(threats), 1)
            self.assertEqual(threats[0], ml_threat)
            
            # Get active threats
            active_threats = await self.analyzer.get_active_threats()
            self.assertEqual(len(active_threats), 1)
            
            # Get statistics
            stats = await self.analyzer.get_detection_statistics()
            self.assertEqual(stats[ThreatCategory.UNAUTHORIZED_ACCESS.value], 1)
            
        self.loop.run_until_complete(_test())

if __name__ == '__main__':
    unittest.main()