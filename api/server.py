import asyncio
import logging
import uvicorn
import json
from fastapi import FastAPI, Depends, HTTPException, Request, Response, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Any, Optional
import time
from datetime import datetime
import uuid
import os
import jwt

from core.orchestrator import SecurityOrchestrator
from core.config import OrchestrationConfig

class APIServer:
    """
    API server for the security orchestrator providing RESTful endpoints
    to monitor and control the orchestration process.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the API server.
        
        Args:
            config_path: Path to the orchestrator configuration
        """
        self.logger = logging.getLogger("api.server")
        
        # Load configuration
        self.config = OrchestrationConfig.from_file(config_path)
        
        # Initialize the orchestrator
        self.orchestrator = SecurityOrchestrator(self.config)
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Cloud Security Orchestrator API",
            description="API for autonomous cloud security orchestration",
            version="1.0.0"
        )
        
        # Security
        self.security = HTTPBearer()
        self.secret_key = os.getenv("API_SECRET_KEY", "your-secret-key-here")
        
        # Set up middleware
        self._setup_middleware()
        
        # Define routes
        self._setup_routes()
        
        # Initialize server state
        self.initialized = False
        self.server_start_time = None
        
        self.logger.info("API server initialized")
    
    def _setup_middleware(self):
        """Set up API middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, restrict to specific origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            
            try:
                response = await call_next(request)
                process_time = time.time() - start_time
                
                # Log request details
                self.logger.info(
                    f"{request.method} {request.url.path} "
                    f"completed in {process_time:.4f}s with status {response.status_code}"
                )
                
                return response
            except Exception as e:
                self.logger.error(f"Request error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"detail": "Internal server error"}
                )
    
    def _setup_routes(self):
        """Set up API routes"""
        
        # Authentication
        async def authenticate(credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            try:
                token = credentials.credentials
                payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
                return payload
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            except jwt.InvalidTokenError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        
        # Health check (no auth required)
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "initialized": self.initialized,
                "uptime": time.time() - self.server_start_time if self.server_start_time else 0
            }
        
        # Authentication endpoint
        @self.app.post("/auth/token")
        async def get_token(request: Request):
            try:
                data = await request.json()
                username = data.get("username")
                password = data.get("password")
                
                # In production, authenticate against secure database
                if username == "admin" and password == "secure_password":
                    # Create JWT token
                    expiration = int(time.time()) + 3600  # 1 hour expiration
                    payload = {
                        "sub": username,
                        "exp": expiration,
                        "iat": int(time.time()),
                        "role": "admin"
                    }
                    token = jwt.encode(payload, self.secret_key, algorithm="HS256")
                    
                    return {"access_token": token, "token_type": "bearer", "expires_at": expiration}
                else:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid credentials",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
            except Exception as e:
                self.logger.error(f"Authentication error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Authentication error"
                )
        
        # Orchestrator status
        @self.app.get("/api/status")
        async def get_status(_: Dict = Depends(authenticate)):
            if not self.initialized:
                return {
                    "status": "not_initialized",
                    "message": "Orchestrator not yet initialized"
                }
            
            cloud_statuses = {}
            for provider_id, connector in self.orchestrator.cloud_connectors.items():
                cloud_statuses[provider_id] = await connector.get_connection_status()
            
            return {
                "status": "running" if self.orchestrator.is_running else "stopped",
                "orchestration_id": self.orchestrator._orchestration_id,
                "cloud_providers": cloud_statuses,
                "federated_learning": {
                    "active_clients": len(self.orchestrator.federated_coordinator.active_clients),
                    "model_version": self.orchestrator.federated_coordinator.current_model_version
                }
            }
        
        # Initialize orchestrator
        @self.app.post("/api/init")
        async def initialize_orchestrator(background_tasks: BackgroundTasks, _: Dict = Depends(authenticate)):
            if self.initialized:
                return {"status": "already_initialized"}
            
            # Initialize in background to not block the API response
            background_tasks.add_task(self._initialize_orchestrator)
            
            return {"status": "initializing"}
        
        # Start orchestration
        @self.app.post("/api/start")
        async def start_orchestration(_: Dict = Depends(authenticate)):
            if not self.initialized:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Orchestrator not initialized"
                )
            
            if self.orchestrator.is_running:
                return {"status": "already_running"}
            
            await self.orchestrator.start()
            return {"status": "started"}
        
        # Stop orchestration
        @self.app.post("/api/stop")
        async def stop_orchestration(_: Dict = Depends(authenticate)):
            if not self.initialized or not self.orchestrator.is_running:
                return {"status": "not_running"}
            
            await self.orchestrator.stop()
            return {"status": "stopped"}
        
        # Get active threats
        @self.app.get("/api/threats")
        async def get_threats(_: Dict = Depends(authenticate)):
            if not self.initialized:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Orchestrator not initialized"
                )
            
            threats = await self.orchestrator.threat_analyzer.get_active_threats()
            return {
                "count": len(threats),
                "threats": [threat.to_dict() for threat in threats]
            }
        
        # Get threat details
        @self.app.get("/api/threats/{threat_id}")
        async def get_threat_details(threat_id: str, _: Dict = Depends(authenticate)):
            if not self.initialized:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Orchestrator not initialized"
                )
            
            threat = await self.orchestrator.threat_analyzer.get_threat(threat_id)
            if not threat:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Threat with ID {threat_id} not found"
                )
            
            return threat.to_dict()
        
        # Manual trigger federated learning round
        @self.app.post("/api/federated/update")
        async def trigger_federated_update(_: Dict = Depends(authenticate)):
            if not self.initialized:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Orchestrator not initialized"
                )
            
            success = await self.orchestrator.federated_coordinator.trigger_update_round()
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Could not trigger federated update"
                )
            
            return {"status": "update_triggered"}
        
        # Get federated learning status
        @self.app.get("/api/federated/status")
        async def get_federated_status(_: Dict = Depends(authenticate)):
            if not self.initialized:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Orchestrator not initialized"
                )
            
            coordinator = self.orchestrator.federated_coordinator
            return {
                "status": "running" if coordinator.is_running else "stopped",
                "active_clients": len(coordinator.active_clients),
                "total_clients": len(coordinator.clients),
                "model_version": coordinator.current_model_version,
                "round_in_progress": coordinator.round_in_progress,
                "current_round_id": coordinator.current_round_id
            }
    
    async def _initialize_orchestrator(self):
        """Initialize the orchestrator in background"""
        try:
            self.logger.info("Beginning orchestrator initialization")
            
            # Initialize cloud connectors
            await self.orchestrator.initialize_cloud_connectors()
            
            # Set initialized flag
            self.initialized = True
            self.server_start_time = time.time()
            
            self.logger.info("Orchestrator initialization complete")
        except Exception as e:
            self.logger.error(f"Error initializing orchestrator: {e}", exc_info=True)
            self.initialized = False
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8080):
        """
        Start the API server.
        
        Args:
            host: Host IP to bind to
            port: Port to listen on
        """
        config = uvicorn.Config(self.app, host=host, port=port)
        server = uvicorn.Server(config)
        
        # Start the server in the current event loop
        self.logger.info(f"Starting API server on {host}:{port}")
        await server.serve()
    
    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """
        Run the API server (blocking call).
        
        Args:
            host: Host IP to bind to
            port: Port to listen on
        """
        uvicorn.run(self.app, host=host, port=port)