import asyncio
import logging
from typing import Dict, List, Any
import uuid
import time

from core.config import OrchestrationConfig
from federated_learning.coordinator import FederatedCoordinator
from crypto.homomorphic_engine import HomomorphicEngine
from threat_detection.analyzer import ThreatAnalyzer
from response_engine.action_executor import ActionExecutor
from cloud_providers.connector_base import CloudProviderConnector
from system_optimizations.parallel_executor import ParallelExecutor

class SecurityOrchestrator:
    """
    Core orchestrator component that coordinates federated learning, threat detection,
    and automated responses across cloud environments with homomorphic encryption.
    """
    
    def __init__(self, config: OrchestrationConfig):
        """
        Initialize the security orchestrator with given configuration.
        
        Args:
            config: Configuration object for the orchestrator
        """
        self.config = config
        self.logger = logging.getLogger("orchestrator")
        
        # Initialize core components
        self.crypto_engine = HomomorphicEngine(
            key_size=config.crypto_settings.key_size,
            security_level=config.crypto_settings.security_level
        )
        
        self.federated_coordinator = FederatedCoordinator(
            model_config=config.federated_learning,
            crypto_engine=self.crypto_engine,
            min_clients=config.federated_learning.min_clients
        )
        
        self.threat_analyzer = ThreatAnalyzer(
            detection_threshold=config.detection_threshold,
            model_provider=self.federated_coordinator
        )
        
        self.action_executor = ActionExecutor(
            policy_path=config.response_policies_path
        )
        
        self.cloud_connectors: Dict[str, CloudProviderConnector] = {}
        self.parallel_executor = ParallelExecutor(
            max_workers=config.system.max_workers,
            task_queue_size=config.system.task_queue_size
        )
        
        self.is_running = False
        self._orchestration_id = str(uuid.uuid4())
        self.logger.info(f"Initialized SecurityOrchestrator with ID {self._orchestration_id}")

    async def initialize_cloud_connectors(self):
        """Initialize connections to configured cloud providers"""
        for provider_config in self.config.cloud_providers:
            connector = provider_config.create_connector()
            await connector.initialize()
            self.cloud_connectors[provider_config.provider_id] = connector
            self.logger.info(f"Initialized cloud connector for {provider_config.provider_id}")

    async def start(self):
        """Start the orchestration process"""
        if self.is_running:
            self.logger.warning("Orchestrator is already running")
            return
        
        self.logger.info("Starting security orchestrator...")
        self.is_running = True
        
        # Initialize cloud connectors
        await self.initialize_cloud_connectors()
        
        # Start federated learning coordinator
        await self.federated_coordinator.start()
        
        # Start event collection and processing
        self._task_event_processing = asyncio.create_task(self._process_events())
        
        # Start model update cycle
        self._task_model_updates = asyncio.create_task(self._periodic_model_updates())
        
        self.logger.info("Security orchestrator started successfully")

    async def stop(self):
        """Stop the orchestration process"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping security orchestrator...")
        self.is_running = False
        
        # Cancel running tasks
        if hasattr(self, '_task_event_processing'):
            self._task_event_processing.cancel()
        
        if hasattr(self, '_task_model_updates'):
            self._task_model_updates.cancel()
        
        # Stop federated learning coordinator
        await self.federated_coordinator.stop()
        
        # Disconnect cloud providers
        for provider_id, connector in self.cloud_connectors.items():
            await connector.disconnect()
            self.logger.info(f"Disconnected from cloud provider {provider_id}")
        
        # Shutdown parallel executor
        await self.parallel_executor.shutdown()
        
        self.logger.info("Security orchestrator stopped successfully")

    async def _process_events(self):
        """Continuously process security events from all cloud providers"""
        self.logger.info("Starting security event processing")
        
        while self.is_running:
            try:
                collection_tasks = []
                
                # Collect events from all cloud providers in parallel
                for provider_id, connector in self.cloud_connectors.items():
                    task = self.parallel_executor.submit(connector.collect_security_events)
                    collection_tasks.append((provider_id, task))
                
                # Process collected events
                for provider_id, task in collection_tasks:
                    events = await task
                    if events:
                        self.logger.debug(f"Collected {len(events)} events from {provider_id}")
                        await self._analyze_events(provider_id, events)
                
                # Small delay to prevent CPU overutilization
                await asyncio.sleep(self.config.event_polling_interval)
                
            except asyncio.CancelledError:
                self.logger.info("Event processing task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in event processing: {e}", exc_info=True)
                await asyncio.sleep(5)  # Backoff on error
    
    async def _analyze_events(self, provider_id: str, events: List[Dict[Any, Any]]):
        """
        Analyze collected security events using the threat analyzer
        
        Args:
            provider_id: ID of the cloud provider
            events: List of security events to analyze
        """
        analysis_tasks = []
        
        # Submit events for parallel analysis
        for event_batch in self._batch_events(events, self.config.analysis_batch_size):
            task = self.parallel_executor.submit(
                self.threat_analyzer.analyze_events,
                provider_id=provider_id,
                events=event_batch,
                use_encryption=self.config.use_encryption_for_analysis
            )
            analysis_tasks.append(task)
        
        # Process analysis results and trigger responses
        for task in analysis_tasks:
            threats = await task
            for threat in threats:
                if threat.confidence >= self.config.response_threshold:
                    self.logger.warning(
                        f"Critical threat detected in {provider_id}: {threat.description} "
                        f"(confidence: {threat.confidence:.2f})"
                    )
                    await self._respond_to_threat(provider_id, threat)
                else:
                    self.logger.info(
                        f"Potential threat in {provider_id}: {threat.description} "
                        f"(confidence: {threat.confidence:.2f})"
                    )
    
    async def _respond_to_threat(self, provider_id: str, threat):
        """Execute appropriate response actions for detected threats"""
        connector = self.cloud_connectors.get(provider_id)
        if not connector:
            self.logger.error(f"Cannot respond to threat: connector for {provider_id} not found")
            return
        
        response_plan = await self.action_executor.create_response_plan(
            threat=threat,
            provider_id=provider_id,
            provider_type=connector.provider_type
        )
        
        if not response_plan.actions:
            self.logger.warning(f"No response actions generated for threat: {threat.id}")
            return
        
        self.logger.info(f"Executing {len(response_plan.actions)} response actions for threat {threat.id}")
        
        for action in response_plan.actions:
            try:
                result = await connector.execute_security_action(action)
                self.logger.info(f"Action {action.type} executed with status: {result.status}")
                
                # Update threat state with response information
                await self.threat_analyzer.update_threat_state(
                    threat_id=threat.id,
                    action_taken=action.type,
                    action_result=result.status
                )
                
            except Exception as e:
                self.logger.error(f"Failed to execute action {action.type}: {e}", exc_info=True)
    
    async def _periodic_model_updates(self):
        """Periodically trigger federated model updates"""
        self.logger.info("Starting periodic model update cycle")
        
        while self.is_running:
            try:
                next_update = time.time() + self.config.model_update_interval
                
                # Trigger federated learning round
                self.logger.info("Initiating federated learning update round")
                await self.federated_coordinator.trigger_update_round()
                
                # Wait until next scheduled update
                now = time.time()
                if next_update > now:
                    await asyncio.sleep(next_update - now)
                
            except asyncio.CancelledError:
                self.logger.info("Model update task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in federated model update: {e}", exc_info=True)
                await asyncio.sleep(60)  # Backoff on error
    
    def _batch_events(self, events, batch_size):
        """Split events into batches of specified size"""
        for i in range(0, len(events), batch_size):
            yield events[i:i + batch_size]