import unittest
import numpy as np
import os
import sys
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from threat_detection.analyzer import ThreatAnalyzer, ThreatDetection, ThreatCategory, ThreatSeverity

class TestThreatAnalyzer(unittest.TestCase):
    """
    Unit tests for the ThreatAnalyzer class.
    """
    
    def setUp(self):
        """Set up the test environment"""
        # Create a mock model provider
        self.model_provider = MagicMock()
        
        # Create analyzer with mock model provider
        self.analyzer = ThreatAnalyzer(detection_threshold=0.7, model_provider=self.model_provider)
        
        # Sample events for testing
        self.auth_events = [
            {
                "event_type": "authentication",
                "timestamp": datetime.now().isoformat(),
                "user_id": "user123",
                "authentication": {
                    "success": False,
                    "attempts": 1,
                    "type": "password"
                },
                "source": {
                    "ip": "192.168.1.1"
                }
            }
        ]
        
        # Multiple failed auth events from same user
        self.multiple_failed_auth = []
        for i in range(6):
            self.multiple_failed_auth.append({
                "event_type": "authentication",
                "timestamp": datetime.now().isoformat(),
                "user_id": "user456",
                "authentication": {
                    "success": False,
                    "attempts": 1,
                    "type": "password"
                },
                "source": {
                    "ip": "192.168.1.2"
                }
            })
    
    def test_initialization(self):
        """Test that the analyzer initializes correctly"""
        self.assertEqual(self.analyzer.detection_threshold, 0.7)
        self.assertIs(self.analyzer.model_provider, self.model_provider)
        self.assertGreater(len(self.analyzer.rules), 0)  # Should have some default rules
    
    def test_feature_extraction(self):
        """Test feature extraction from events"""
        auth_features = self.analyzer._extract_auth_features(self.auth_events)
        self.assertEqual(auth_features.shape, (1, 32))  # Should match the expected shape
        
        # First feature should be 0 (auth success = False)
        self.assertEqual(auth_features[0, 0], 0)
    
    def test_determine_event_type(self):
        """Test event type determination"""
        auth_type = self.analyzer._determine_event_type(self.auth_events[0])
        self.assertEqual(auth_type, "auth_events")
        
        # Test with network event
        network_event = {"network": {"bytes_in": 1000}}
        network_type = self.analyzer._determine_event_type(network_event)
        self.assertEqual(network_type, "network_flow")
    
    @patch('threat_detection.analyzer.ThreatAnalyzer._apply_detection_rules')
    @patch('threat_detection.analyzer.ThreatAnalyzer._apply_ml_detection')
    async def test_analyze_events(self, mock_ml, mock_rules):
        """Test events analysis with both rules and ML"""
        # Mock return values
        rule_threat = ThreatDetection(
            id="rule-threat-1",
            provider_id="test-provider",
            timestamp=datetime.now(),
            category=ThreatCategory.UNAUTHORIZED_ACCESS,
            severity=ThreatSeverity.MEDIUM,
            confidence=0.8,
            description="Rule-based threat",
            affected_resources=["user123"]
        )
        
        ml_threat = ThreatDetection(
            id="ml-threat-1",
            provider_id="test-provider",
            timestamp=datetime.now(),
            category=ThreatCategory.DATA_EXFILTRATION,
            severity=ThreatSeverity.HIGH,
            confidence=0.9,
            description="ML-based threat",
            affected_resources=["resource123"]
        )
        
        mock_rules.return_value = [rule_threat]
        mock_ml.return_value = [ml_threat]
        
        # Analyze events
        threats = await self.analyzer.analyze_events("test-provider", self.auth_events)
        
        # Verify both detection methods were called
        mock_rules.assert_called_once()
        mock_ml.assert_called_once()
        
        # Verify combined results
        self.assertEqual(len(threats), 2)
        self.assertIn(rule_threat, threats)
        self.assertIn(ml_threat, threats)
        
        # Verify stats updated
        self.assertEqual(
            self.analyzer._detection_stats[ThreatCategory.UNAUTHORIZED_ACCESS.value], 1
        )
        self.assertEqual(
            self.analyzer._detection_stats[ThreatCategory.DATA_EXFILTRATION.value], 1
        )
    
    async def test_rule_based_detection(self):
        """Test rule-based threat detection"""
        # Use the multiple failed auth events which should trigger the rule
        threats = await self.analyzer._apply_detection_rules("test-provider", self.multiple_failed_auth)
        
        # Should find at least one threat
        self.assertGreater(len(threats), 0)
        
        # Verify threat properties
        threat = threats[0]
        self.assertEqual(threat.provider_id, "test-provider")
        self.assertEqual(threat.category, ThreatCategory.UNAUTHORIZED_ACCESS)
        self.assertEqual(threat.severity, ThreatSeverity.MEDIUM)
        self.assertIn("user456", threat.affected_resources)
    
    async def test_threat_state_management(self):
        """Test updating and retrieving threat state"""
        # Create a sample threat
        threat = ThreatDetection(
            id="test-threat-1",
            provider_id="test-provider",
            timestamp=datetime.now(),
            category=ThreatCategory.UNAUTHORIZED_ACCESS,
            severity=ThreatSeverity.MEDIUM,
            confidence=0.8,
            description="Test threat",
            affected_resources=["user123"]
        )
        
        # Add threat to active threats
        async with self.analyzer._threat_lock:
            self.analyzer._active_threats[threat.id] = threat
        
        # Update threat state
        success = await self.analyzer.update_threat_state(
            threat_id=threat.id,
            action_taken="block_ip",
            action_result="successful"
        )
        
        # Verify update was successful
        self.assertTrue(success)
        self.assertEqual(len(threat.response_actions), 1)
        self.assertEqual(threat.status, "mitigated")
        
        # Get threat by ID
        retrieved_threat = await self.analyzer.get_threat(threat.id)
        self.assertEqual(retrieved_threat, threat)
        
        # Get all active threats
        active_threats = await self.analyzer.get_active_threats()
        self.assertEqual(len(active_threats), 1)
        self.assertEqual(active_threats[0], threat)
        
        # Clear resolved threats
        cleared = await self.analyzer.clear_resolved_threats()
        self.assertEqual(cleared, 1)  # Should have cleared 1 threat
        
        # Verify it's gone
        active_threats = await self.analyzer.get_active_threats()
        self.assertEqual(len(active_threats), 0)
    
    def test_categorize_ml_detection(self):
        """Test categorization of ML detections"""
        # Test various event types and scores
        cat1, sev1 = self.analyzer._categorize_ml_detection("network_flow", 0.95)
        self.assertEqual(cat1, ThreatCategory.UNAUTHORIZED_ACCESS)
        self.assertEqual(sev1, ThreatSeverity.CRITICAL)
        
        cat2, sev2 = self.analyzer._categorize_ml_detection("auth_events", 0.85)
        self.assertEqual(cat2, ThreatCategory.CREDENTIAL_COMPROMISE)
        self.assertEqual(sev2, ThreatSeverity.HIGH)
        
        cat3, sev3 = self.analyzer._categorize_ml_detection("api_calls", 0.75)
        self.assertEqual(cat3, ThreatCategory.API_ABUSE)
        self.assertEqual(sev3, ThreatSeverity.MEDIUM)
        
        cat4, sev4 = self.analyzer._categorize_ml_detection("resource_usage", 0.65)
        self.assertEqual(cat4, ThreatCategory.RESOURCE_HIJACKING)
        self.assertEqual(sev4, ThreatSeverity.LOW)

if __name__ == '__main__':
    unittest.main()