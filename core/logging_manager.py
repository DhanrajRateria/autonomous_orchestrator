import logging
import logging.handlers
import os
import time
import json
import threading
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

class LoggingManager:
    """
    Advanced logging manager for the security orchestrator.
    Handles log rotation, structured logging, and filtering of sensitive data.
    """
    
    def __init__(self, 
                log_dir: str = "logs", 
                log_level: str = "INFO",
                max_size_mb: int = 50, 
                backup_count: int = 10,
                enable_console: bool = True,
                structured_format: bool = True):
        """
        Initialize the logging manager.
        
        Args:
            log_dir: Directory to store log files
            log_level: Default logging level
            max_size_mb: Maximum log file size in MB
            backup_count: Number of backup files to keep
            enable_console: Whether to enable console logging
            structured_format: Use structured JSON logging format
        """
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.structured_format = structured_format
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Track initialized loggers
        self._initialized_loggers = set()
        self._logger_lock = threading.Lock()
        
        # Define sensitive patterns to be redacted
        self.sensitive_patterns = [
            ('password', '<REDACTED>'),
            ('secret', '<REDACTED>'),
            ('key', '<REDACTED>'),
            ('token', '<REDACTED>'),
            ('credential', '<REDACTED>')
        ]
        
        # Configure the root logger
        self.configure_root_logger()
    
    def configure_root_logger(self):
        """Configure the root logger with default handlers"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Add file handler
        main_log_path = self.log_dir / "orchestrator.log"
        file_handler = self._create_file_handler(main_log_path)
        root_logger.addHandler(file_handler)
        
        # Add console handler if enabled
        if self.enable_console:
            console_handler = self._create_console_handler()
            root_logger.addHandler(console_handler)
        
        # Add error log handler
        error_log_path = self.log_dir / "error.log"
        error_handler = self._create_file_handler(error_log_path, level=logging.ERROR)
        root_logger.addHandler(error_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger configured with appropriate handlers.
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        
        # Check if we've already set up this logger
        with self._logger_lock:
            if name in self._initialized_loggers:
                return logger
            
            # Create component-specific log file for certain components
            if name.split('.')[0] in ["threat", "federated", "crypto", "api"]:
                component = name.split('.')[0]
                component_log_path = self.log_dir / f"{component}.log"
                handler = self._create_file_handler(component_log_path)
                logger.addHandler(handler)
            
            # Mark as initialized
            self._initialized_loggers.add(name)
        
        return logger
    
    def _create_file_handler(self, log_path: Path, level: int = None) -> logging.Handler:
        """Create a rotating file handler"""
        handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=self.max_size_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        
        if level is not None:
            handler.setLevel(level)
        
        if self.structured_format:
            formatter = self._create_json_formatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        handler.setFormatter(formatter)
        return handler
    
    def _create_console_handler(self) -> logging.Handler:
        """Create a console handler"""
        handler = logging.StreamHandler()
        
        # Use simpler format for console output
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        handler.setFormatter(formatter)
        return handler
    
    def _create_json_formatter(self):
        """Create a formatter for structured JSON logging"""
        
        class JsonFormatter(logging.Formatter):
            def __init__(self, sensitive_patterns):
                super().__init__()
                self.sensitive_patterns = sensitive_patterns
            
            def format(self, record):
                log_data = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                }
                
                # Add exception info if available
                if record.exc_info:
                    log_data['exception'] = self.formatException(record.exc_info)
                
                # Add extra fields
                if hasattr(record, 'extra'):
                    for key, value in record.extra.items():
                        log_data[key] = value
                
                # Redact sensitive information
                log_json = json.dumps(log_data)
                for pattern, replacement in self.sensitive_patterns:
                    log_json = log_json.replace(f'"{pattern}": "', f'"{pattern}": "{replacement}')
                
                return log_json
        
        return JsonFormatter(self.sensitive_patterns)
    
    def set_level(self, level: str, logger_name: Optional[str] = None):
        """
        Set the logging level for a specific logger or all loggers.
        
        Args:
            level: Logging level name (DEBUG, INFO, etc.)
            logger_name: Name of the logger to affect, or None for all
        """
        log_level = getattr(logging, level.upper(), None)
        if log_level is None:
            return
        
        if logger_name:
            logging.getLogger(logger_name).setLevel(log_level)
        else:
            logging.getLogger().setLevel(log_level)
    
    def add_sensitive_pattern(self, pattern: str, replacement: str = '<REDACTED>'):
        """
        Add a pattern to be treated as sensitive information.
        
        Args:
            pattern: Pattern to match
            replacement: Text to replace it with
        """
        self.sensitive_patterns.append((pattern, replacement))


# Singleton instance
_logging_manager = None

def setup_logging(log_dir: str = "logs", log_level: str = "INFO", **kwargs) -> LoggingManager:
    """
    Set up and configure global logging.
    
    Args:
        log_dir: Directory to store log files
        log_level: Default logging level
        **kwargs: Additional configuration options
        
    Returns:
        LoggingManager instance
    """
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager(log_dir=log_dir, log_level=log_level, **kwargs)
    
    return _logging_manager

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = setup_logging()
    
    return _logging_manager.get_logger(name)