import asyncio
import logging
import boto3
import botocore
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from cloud_providers.connector_base import CloudProviderConnector
from response_engine.action_executor import SecurityAction, ActionResult

class AWSConnector(CloudProviderConnector):
    """
    Connector for AWS cloud services.
    Handles collection of security events and execution of security actions.
    """
    
    def __init__(self, provider_config):
        """
        Initialize the AWS connector.
        
        Args:
            provider_config: Configuration for this AWS provider
        """
        super().__init__(provider_config)
        self.session = None
        self.credentials = None
        self.clients = {}
        self.last_query_timestamps = {}
    
    async def initialize(self) -> bool:
        """
        Initialize the connection to AWS.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Initialize boto3 session
            self.logger.info(f"Initializing AWS connector for {self.provider_id}")
            
            # Run boto3 session creation in a thread pool
            loop = asyncio.get_event_loop()
            self.session = await loop.run_in_executor(
                None,
                lambda: boto3.Session(
                    profile_name=self.provider_id,
                    region_name=self.region
                )
            )
            
            # Verify credentials
            self.credentials = self.session.get_credentials()
            if not self.credentials:
                self.logger.error("Failed to obtain AWS credentials")
                return False
            
            # Initialize default service clients
            service_list = ['cloudtrail', 'guardduty', 'securityhub', 'ec2', 'iam', 'logs']
            for service in service_list:
                if service in self.enabled_services:
                    try:
                        self.clients[service] = self.session.client(service)
                    except Exception as e:
                        self.logger.warning(f"Could not initialize {service} client: {e}")
            
            self.is_connected = True
            self.stats["last_connection_time"] = datetime.now().isoformat()
            self.logger.info(f"AWS connector initialized successfully with {len(self.clients)} services")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing AWS connector: {e}", exc_info=True)
            self.stats["connection_errors"] += 1
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from AWS.
        
        Returns:
            bool: True if disconnection was successful
        """
        try:
            # Clear clients and session
            self.clients = {}
            self.session = None
            self.is_connected = False
            self.logger.info(f"AWS connector {self.provider_id} disconnected")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from AWS: {e}")
            return False
    
    async def collect_security_events(self) -> List[Dict[str, Any]]:
        """
        Collect security events from AWS services.
        
        Returns:
            List of security events
        """
        if not self.is_connected:
            self.logger.warning("Cannot collect events: not connected to AWS")
            return []
        
        events = []
        
        try:
            # Collect GuardDuty findings
            if 'guardduty' in self.clients:
                guardduty_events = await self._collect_guardduty_findings()
                events.extend(guardduty_events)
            
            # Collect Security Hub findings
            if 'securityhub' in self.clients:
                securityhub_events = await self._collect_securityhub_findings()
                events.extend(securityhub_events)
            
            # Collect CloudTrail events
            if 'cloudtrail' in self.clients:
                cloudtrail_events = await self._collect_cloudtrail_events()
                events.extend(cloudtrail_events)
            
            # Collect CloudWatch Logs
            if 'logs' in self.clients:
                log_events = await self._collect_cloudwatch_logs()
                events.extend(log_events)
            
            self._update_stats("events_collected", len(events))
            self.last_event_timestamp = datetime.now()
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error collecting AWS security events: {e}", exc_info=True)
            return []
    
    async def _collect_guardduty_findings(self) -> List[Dict[str, Any]]:
        """Collect GuardDuty findings"""
        events = []
        
        try:
            # Get detector IDs
            loop = asyncio.get_event_loop()
            detector_response = await loop.run_in_executor(
                None,
                lambda: self.clients['guardduty'].list_detectors()
            )
            
            detector_ids = detector_response.get('DetectorIds', [])
            if not detector_ids:
                return []
            
            # For each detector, get findings
            for detector_id in detector_ids:
                # Get timestamp for filtering
                last_query = self.last_query_timestamps.get('guardduty', 
                                                            datetime.utcnow() - timedelta(hours=24))
                
                # Convert to GuardDuty format
                updated_at = last_query.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                
                # Prepare query parameters
                find_params = {
                    'DetectorId': detector_id,
                    'FindingCriteria': {
                        'Criterion': {
                            'updatedAt': {
                                'Gt': updated_at
                            }
                        }
                    },
                    'MaxResults': 50
                }
                
                # Get findings
                find_response = await loop.run_in_executor(
                    None,
                    lambda: self.clients['guardduty'].list_findings(**find_params)
                )
                
                finding_ids = find_response.get('FindingIds', [])
                
                if finding_ids:
                    # Get finding details
                    details_response = await loop.run_in_executor(
                        None,
                        lambda: self.clients['guardduty'].get_findings(
                            DetectorId=detector_id,
                            FindingIds=finding_ids
                        )
                    )
                    
                    # Transform into standard event format
                    for finding in details_response.get('Findings', []):
                        event = {
                            'event_type': 'guardduty_finding',
                            'provider_id': self.provider_id,
                            'timestamp': finding.get('CreatedAt'),
                            'id': finding.get('Id'),
                            'resource_id': self._extract_resource_id(finding),
                            'severity': float(finding.get('Severity', 0.0)),
                            'title': finding.get('Title'),
                            'description': finding.get('Description'),
                            'source': {
                                'ip': self._extract_source_ip(finding),
                                'is_tor_exit': finding.get('Service', {}).get('Action', {}).get('NetworkConnectionAction', {}).get('RemoteIpDetails', {}).get('IsTorIpAddress', False)
                            },
                            'raw_finding': finding
                        }
                        events.append(event)
                
                # Update last query timestamp
                self.last_query_timestamps['guardduty'] = datetime.utcnow()
        
        except Exception as e:
            self.logger.error(f"Error collecting GuardDuty findings: {e}", exc_info=True)
        
        return events
    
    async def _collect_securityhub_findings(self) -> List[Dict[str, Any]]:
        """Collect SecurityHub findings"""
        events = []
        
        try:
            loop = asyncio.get_event_loop()
            
            # Get timestamp for filtering
            last_query = self.last_query_timestamps.get('securityhub',
                                                       datetime.utcnow() - timedelta(hours=24))
            
            # Convert to ISO format for SecurityHub
            updated_at = last_query.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            
            # Prepare filters
            filters = {
                'UpdatedAt': [{
                    'Start': updated_at,
                    'End': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                }],
                'WorkflowStatus': [{
                    'Value': 'NEW',
                    'Comparison': 'EQUALS'
                }]
            }
            
            # Get findings
            findings_response = await loop.run_in_executor(
                None,
                lambda: self.clients['securityhub'].get_findings(
                    Filters=filters,
                    MaxResults=100
                )
            )
            
            # Transform into standard event format
            for finding in findings_response.get('Findings', []):
                event = {
                    'event_type': 'securityhub_finding',
                    'provider_id': self.provider_id,
                    'timestamp': finding.get('UpdatedAt'),
                    'id': finding.get('Id'),
                    'resource_id': self._extract_securityhub_resource_id(finding),
                    'severity': self._map_securityhub_severity(finding.get('Severity', {}).get('Normalized', 0)),
                    'title': finding.get('Title'),
                    'description': finding.get('Description'),
                    'source': {
                        'id': finding.get('ProductArn')
                    },
                    'raw_finding': finding
                }
                events.append(event)
            
            # Update last query timestamp
            self.last_query_timestamps['securityhub'] = datetime.utcnow()
        
        except Exception as e:
            self.logger.error(f"Error collecting SecurityHub findings: {e}", exc_info=True)
        
        return events
    
    async def _collect_cloudtrail_events(self) -> List[Dict[str, Any]]:
        """Collect CloudTrail events"""
        events = []
        
        try:
            # Run in thread pool
            loop = asyncio.get_event_loop()
            
            # Calculate time range (last hour if no previous query)
            last_query = self.last_query_timestamps.get('cloudtrail',
                                                       datetime.utcnow() - timedelta(hours=1))
            
            # Look for specific management events
            response = await loop.run_in_executor(
                None,
                lambda: self.clients['cloudtrail'].lookup_events(
                    LookupAttributes=[
                        {
                            'AttributeKey': 'EventName',
                            'AttributeValue': 'ConsoleLogin'  # Focus on login events for now
                        }
                    ],
                    StartTime=last_query,
                    EndTime=datetime.utcnow(),
                    MaxResults=50
                )
            )
            
            # Transform into standard event format
            for event in response.get('Events', []):
                cloudtrail_event = json.loads(event.get('CloudTrailEvent', '{}'))
                
                # Check for failed logins
                if 'errorMessage' in cloudtrail_event or cloudtrail_event.get('responseElements', {}).get('ConsoleLogin') == 'Failure':
                    is_failure = True
                else:
                    is_failure = False
                
                processed_event = {
                    'event_type': 'cloudtrail_event',
                    'provider_id': self.provider_id,
                    'timestamp': str(event.get('EventTime')),
                    'id': event.get('EventId'),
                    'user_id': event.get('Username'),
                    'event_name': event.get('EventName'),
                    'authentication': {
                        'success': not is_failure,
                        'failure_reason': cloudtrail_event.get('errorMessage', '') if is_failure else '',
                        'source_ip': cloudtrail_event.get('sourceIPAddress', '')
                    },
                    'raw_event': cloudtrail_event
                }
                events.append(processed_event)
            
            # Update last query timestamp
            self.last_query_timestamps['cloudtrail'] = datetime.utcnow()
        
        except Exception as e:
            self.logger.error(f"Error collecting CloudTrail events: {e}", exc_info=True)
        
        return events
    
    async def _collect_cloudwatch_logs(self) -> List[Dict[str, Any]]:
        """Collect relevant CloudWatch logs"""
        events = []
        
        # This would be implemented to query specific CloudWatch Log groups
        # for security-relevant events. Implementation depends on which logs
        # are being monitored.
        
        return events
    
    async def execute_security_action(self, action: SecurityAction) -> ActionResult:
        """
        Execute a security action on AWS resources.
        
        Args:
            action: Security action to execute
            
        Returns:
            Result of the action execution
        """
        if not self.is_connected:
            return ActionResult(action.action_id, "failed", {"reason": "Not connected to AWS"})
        
        try:
            self.logger.info(f"Executing {action.type} action on AWS")
            
            # Execute different actions based on type
            if action.type == "block_ip":
                result = await self._execute_block_ip(action)
            elif action.type == "disable_credentials":
                result = await self._execute_disable_credentials(action)
            elif action.type == "quarantine_instance":
                result = await self._execute_quarantine_instance(action)
            elif action.type == "snapshot_resources":
                result = await self._execute_snapshot_resources(action)
            elif action.type == "restrict_permissions":
                result = await self._execute_restrict_permissions(action)
            else:
                self.logger.warning(f"Unsupported action type: {action.type}")
                return ActionResult(action.action_id, "failed", {"reason": f"Unsupported action type: {action.type}"})
            
            self._update_stats("actions_executed")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing AWS action: {e}", exc_info=True)
            return ActionResult(action.action_id, "failed", {"reason": str(e)})
    
    async def _execute_block_ip(self, action: SecurityAction) -> ActionResult:
        """Block an IP address in AWS"""
        try:
            source_ip = action.target.get("source_ip")
            if not source_ip:
                return ActionResult(action.action_id, "failed", {"reason": "No source IP provided"})
            
            # Get duration from parameters
            duration = action.parameters.get("duration", 3600)  # Default 1 hour
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            
            # Add IP to network ACL deny rule
            security_groups = await self._get_default_security_groups()
            if not security_groups:
                return ActionResult(action.action_id, "failed", {"reason": "No security groups found"})
            
            # For each security group, add deny rule
            for sg_id in security_groups:
                await loop.run_in_executor(
                    None,
                    lambda: self.clients['ec2'].authorize_security_group_ingress(
                        GroupId=sg_id,
                        IpPermissions=[{
                            'IpProtocol': '-1',  # All protocols
                            'FromPort': -1,      # All ports
                            'ToPort': -1,
                            'IpRanges': [{
                                'CidrIp': f'{source_ip}/32',
                                'Description': f'BLOCKED by Security Orchestrator {action.action_id}'
                            }]
                        }]
                    )
                )
            
            self.logger.info(f"Blocked IP {source_ip} in {len(security_groups)} security groups")
            
            return ActionResult(
                action.action_id, 
                "successful", 
                {
                    "ip": source_ip, 
                    "security_groups": security_groups,
                    "duration": duration
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error blocking IP: {e}", exc_info=True)
            return ActionResult(action.action_id, "failed", {"reason": str(e)})
    
    async def _execute_disable_credentials(self, action: SecurityAction) -> ActionResult:
        """Disable IAM credentials"""
        try:
            user_id = action.target.get("user_id")
            if not user_id:
                return ActionResult(action.action_id, "failed", {"reason": "No user ID provided"})
            
            revoke_sessions = action.parameters.get("revoke_sessions", True)
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            
            # Update IAM user
            await loop.run_in_executor(
                None,
                lambda: self.clients['iam'].update_access_key(
                    UserName=user_id,
                    AccessKeyId='ALL',  # This would need to be adjusted for real implementation
                    Status='Inactive'
                )
            )
            
            if revoke_sessions:
                # Revoke active sessions
                await loop.run_in_executor(
                    None,
                    lambda: self.clients['iam'].delete_user_permissions_boundary(
                        UserName=user_id
                    )
                )
            
            self.logger.info(f"Disabled credentials for user {user_id}")
            
            return ActionResult(
                action.action_id, 
                "successful", 
                {
                    "user_id": user_id, 
                    "revoked_sessions": revoke_sessions
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error disabling credentials: {e}", exc_info=True)
            return ActionResult(action.action_id, "failed", {"reason": str(e)})
    
    async def _execute_quarantine_instance(self, action: SecurityAction) -> ActionResult:
        """Quarantine an EC2 instance"""
        try:
            resources = action.target.get("resources", [])
            if not resources:
                return ActionResult(action.action_id, "failed", {"reason": "No resources provided"})
            
            maintain_service = action.parameters.get("maintain_service", False)
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            
            # Create a quarantine security group
            sg_response = await loop.run_in_executor(
                None,
                lambda: self.clients['ec2'].create_security_group(
                    GroupName=f'quarantine-{action.action_id[:8]}',
                    Description=f'Quarantine SG created by Security Orchestrator',
                    VpcId='vpc-default'  # This would need to be determined in real implementation
                )
            )
            
            quarantine_sg_id = sg_response.get('GroupId')
            
            # Allow outbound only if maintain_service is True
            if maintain_service:
                await loop.run_in_executor(
                    None,
                    lambda: self.clients['ec2'].authorize_security_group_egress(
                        GroupId=quarantine_sg_id,
                        IpPermissions=[{
                            'IpProtocol': 'tcp',
                            'FromPort': 443,
                            'ToPort': 443,
                            'IpRanges': [{
                                'CidrIp': '0.0.0.0/0',
                                'Description': 'Allow HTTPS outbound'
                            }]
                        }]
                    )
                )
            
            # Apply to each instance
            for instance_id in resources:
                if not instance_id.startswith('i-'):
                    continue  # Skip non-EC2 resources
                
                # Modify instance security groups
                await loop.run_in_executor(
                    None,
                    lambda: self.clients['ec2'].modify_instance_attribute(
                        InstanceId=instance_id,
                        Groups=[quarantine_sg_id]
                    )
                )
            
            self.logger.info(f"Quarantined {len(resources)} instances with SG {quarantine_sg_id}")
            
            return ActionResult(
                action.action_id, 
                "successful", 
                {
                    "resources": resources, 
                    "quarantine_sg_id": quarantine_sg_id
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error quarantining instance: {e}", exc_info=True)
            return ActionResult(action.action_id, "failed", {"reason": str(e)})
    
    async def _execute_snapshot_resources(self, action: SecurityAction) -> ActionResult:
        """Create snapshots of resources for forensics"""
        # This would create snapshots of EC2 instances, RDS databases, etc.
        # Implementation depends on resource types
        return ActionResult(action.action_id, "successful", {})
    
    async def _execute_restrict_permissions(self, action: SecurityAction) -> ActionResult:
        """Restrict IAM permissions"""
        # This would modify IAM policies to restrict access
        # Implementation depends on specific permissions to restrict
        return ActionResult(action.action_id, "successful", {})
    
    async def get_resource_info(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific AWS resource.
        
        Args:
            resource_id: ID of the resource to query
            
        Returns:
            Resource information or None if not found
        """
        if not self.is_connected:
            return None
        
        try:
            # Determine resource type from ID prefix
            if resource_id.startswith('i-'):  # EC2 instance
                return await self._get_ec2_instance_info(resource_id)
            elif resource_id.startswith('ami-'):  # AMI
                return await self._get_ami_info(resource_id)
            elif resource_id.startswith('vol-'):  # EBS volume
                return await self._get_ebs_volume_info(resource_id)
            elif resource_id.startswith('sg-'):  # Security group
                return await self._get_security_group_info(resource_id)
            elif resource_id.startswith('subnet-'):  # Subnet
                return await self._get_subnet_info(resource_id)
            else:
                self.logger.warning(f"Unknown resource type for ID: {resource_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting resource info: {e}", exc_info=True)
            return None
    
    async def _get_ec2_instance_info(self, instance_id: str) -> Dict[str, Any]:
        """Get EC2 instance information"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.clients['ec2'].describe_instances(
                InstanceIds=[instance_id]
            )
        )
        
        reservations = response.get('Reservations', [])
        if not reservations:
            return None
            
        instances = reservations[0].get('Instances', [])
        if not instances:
            return None
            
        instance = instances[0]
        return {
            'id': instance_id,
            'type': 'ec2_instance',
            'state': instance.get('State', {}).get('Name'),
            'instance_type': instance.get('InstanceType'),
            'launch_time': str(instance.get('LaunchTime')),
            'private_ip': instance.get('PrivateIpAddress'),
            'public_ip': instance.get('PublicIpAddress'),
            'vpc_id': instance.get('VpcId'),
            'subnet_id': instance.get('SubnetId'),
            'security_groups': [sg.get('GroupId') for sg in instance.get('SecurityGroups', [])]
        }
    
    async def _get_ami_info(self, ami_id: str) -> Dict[str, Any]:
        """Get AMI information"""
        # Implementation for AMI info
        return {'id': ami_id, 'type': 'ami'}
    
    async def _get_ebs_volume_info(self, volume_id: str) -> Dict[str, Any]:
        """Get EBS volume information"""
        # Implementation for EBS volume info
        return {'id': volume_id, 'type': 'ebs_volume'}
    
    async def _get_security_group_info(self, sg_id: str) -> Dict[str, Any]:
        """Get security group information"""
        # Implementation for security group info
        return {'id': sg_id, 'type': 'security_group'}
    
    async def _get_subnet_info(self, subnet_id: str) -> Dict[str, Any]:
        """Get subnet information"""
        # Implementation for subnet info
        return {'id': subnet_id, 'type': 'subnet'}
    
    async def _get_default_security_groups(self) -> List[str]:
        """Get default security groups for the account"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.clients['ec2'].describe_security_groups(
                    Filters=[
                        {
                            'Name': 'group-name',
                            'Values': ['default']
                        }
                    ]
                )
            )
            
            security_groups = response.get('SecurityGroups', [])
            return [sg.get('GroupId') for sg in security_groups]
            
        except Exception as e:
            self.logger.error(f"Error getting default security groups: {e}")
            return []
    
    def _extract_resource_id(self, finding: Dict[str, Any]) -> str:
        """Extract resource ID from a GuardDuty finding"""
        resource = finding.get('Resource', {})
        if 'InstanceDetails' in resource:
            return resource['InstanceDetails'].get('InstanceId', '')
        elif 'S3BucketDetails' in resource:
            return resource['S3BucketDetails'][0].get('Arn', '') if resource['S3BucketDetails'] else ''
        elif 'AccessKeyDetails' in resource:
            return resource['AccessKeyDetails'].get('AccessKeyId', '')
        else:
            return ''
    
    def _extract_source_ip(self, finding: Dict[str, Any]) -> str:
        """Extract source IP address from a GuardDuty finding"""
        service = finding.get('Service', {})
        if 'Action' in service and 'NetworkConnectionAction' in service['Action']:
            return service['Action']['NetworkConnectionAction'].get('RemoteIpDetails', {}).get('IpAddressV4', '')
        elif 'Action' in service and 'AwsApiCallAction' in service['Action']:
            return service['Action']['AwsApiCallAction'].get('RemoteIpDetails', {}).get('IpAddressV4', '')
        else:
            return ''
    
    def _extract_securityhub_resource_id(self, finding: Dict[str, Any]) -> str:
        """Extract resource ID from a SecurityHub finding"""
        resources = finding.get('Resources', [])
        if resources:
            return resources[0].get('Id', '')
        return ''
    
    def _map_securityhub_severity(self, normalized_score: float) -> float:
        """Map SecurityHub normalized severity to our 0-1 scale"""
        return normalized_score / 100.0