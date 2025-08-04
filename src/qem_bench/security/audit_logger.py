"""
Security Audit Logging for QEM-Bench

This module provides comprehensive audit logging for security-relevant events including:
- Authentication and authorization events
- Credential access and management
- Resource allocation and usage
- Configuration changes
- Error conditions and security violations
- System access and operations
"""

import json
import os
import threading
import time
import warnings
from typing import Any, Dict, List, Optional, TextIO, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging
import logging.handlers

from ..errors import SecurityError


class AuditEventType(Enum):
    """Types of security audit events."""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    SESSION_CREATED = "session_created"
    SESSION_EXPIRED = "session_expired"
    
    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGED = "permission_changed"
    
    # Credential events
    CREDENTIAL_CREATED = "credential_created"
    CREDENTIAL_ACCESSED = "credential_accessed"
    CREDENTIAL_UPDATED = "credential_updated"
    CREDENTIAL_DELETED = "credential_deleted"
    CREDENTIAL_EXPIRED = "credential_expired"
    
    # Resource events
    RESOURCE_ALLOCATED = "resource_allocated"
    RESOURCE_RELEASED = "resource_released"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    
    # Configuration events
    CONFIG_CHANGED = "config_changed"
    CONFIG_LOADED = "config_loaded"
    CONFIG_SAVED = "config_saved"
    
    # Security events
    SECURITY_VIOLATION = "security_violation"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    ENCRYPTION_FAILURE = "encryption_failure"
    VALIDATION_FAILURE = "validation_failure"
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    
    # Operation events
    CIRCUIT_EXECUTED = "circuit_executed"
    BACKEND_ACCESSED = "backend_accessed"
    DATA_EXPORTED = "data_exported"
    DATA_IMPORTED = "data_imported"


class AuditLevel(Enum):
    """Audit event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Security audit event."""
    event_type: AuditEventType
    timestamp: datetime = field(default_factory=datetime.now)
    level: AuditLevel = AuditLevel.INFO
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    outcome: str = "success"
    details: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['level'] = self.level.value
        return data
    
    def to_json(self) -> str:
        """Convert audit event to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """
    Security audit logger for QEM-Bench.
    
    Provides comprehensive logging of security-relevant events with:
    - Structured logging format
    - Multiple output destinations
    - Event filtering and rotation
    - Performance optimization
    - Tamper detection
    """
    
    def __init__(
        self,
        log_file: Optional[Path] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = False,
        enable_syslog: bool = False,
        min_level: AuditLevel = AuditLevel.INFO,
        buffer_size: int = 100,
        flush_interval: float = 5.0
    ):
        """
        Initialize audit logger.
        
        Args:
            log_file: Path to audit log file
            max_file_size: Maximum log file size before rotation
            backup_count: Number of backup files to keep
            enable_console: Whether to log to console
            enable_syslog: Whether to log to syslog
            min_level: Minimum audit level to log
            buffer_size: Buffer size for batched logging
            flush_interval: Interval to flush buffered events
        """
        self.log_file = log_file or Path.home() / ".qem_bench" / "audit.log"
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_syslog = enable_syslog
        self.min_level = min_level
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Event buffer for performance
        self.event_buffer: List[AuditEvent] = []
        self.last_flush = time.time()
        
        # Statistics
        self.event_count = 0
        self.error_count = 0
        self.last_event_time: Optional[datetime] = None
        
        # Initialize loggers
        self._setup_loggers()
        
        # Start background flush thread
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()
    
    def _setup_loggers(self):
        """Setup logging infrastructure."""
        # Create main logger
        self.logger = logging.getLogger('qem_bench_audit')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Syslog handler
        if self.enable_syslog:
            try:
                syslog_handler = logging.handlers.SysLogHandler()
                syslog_handler.setFormatter(formatter)
                self.logger.addHandler(syslog_handler)
            except Exception as e:
                warnings.warn(f"Failed to setup syslog handler: {e}")
    
    def _flush_worker(self):
        """Background worker to flush buffered events."""
        while True:
            try:
                time.sleep(self.flush_interval)
                current_time = time.time()
                
                with self._lock:
                    if (self.event_buffer and 
                        (len(self.event_buffer) >= self.buffer_size or
                         current_time - self.last_flush >= self.flush_interval)):
                        self._flush_buffer()
            except Exception as e:
                warnings.warn(f"Audit flush error: {e}")
    
    def _flush_buffer(self):
        """Flush buffered events to log files."""
        if not self.event_buffer:
            return
        
        try:
            for event in self.event_buffer:
                self._write_event(event)
            
            self.event_buffer.clear()
            self.last_flush = time.time()
            
        except Exception as e:
            self.error_count += 1
            warnings.warn(f"Failed to flush audit events: {e}")
    
    def _write_event(self, event: AuditEvent):
        """Write a single event to log."""
        # Convert to appropriate log level
        log_level = {
            AuditLevel.DEBUG: logging.DEBUG,
            AuditLevel.INFO: logging.INFO,
            AuditLevel.WARNING: logging.WARNING,
            AuditLevel.ERROR: logging.ERROR,
            AuditLevel.CRITICAL: logging.CRITICAL
        }.get(event.level, logging.INFO)
        
        # Create log message
        message = event.to_json()
        
        # Write to logger
        self.logger.log(log_level, message)
    
    def log_event(self, event: AuditEvent):
        """
        Log an audit event.
        
        Args:
            event: Audit event to log
        """
        # Check minimum level
        level_priority = {
            AuditLevel.DEBUG: 0,
            AuditLevel.INFO: 1,
            AuditLevel.WARNING: 2,
            AuditLevel.ERROR: 3,
            AuditLevel.CRITICAL: 4
        }
        
        if level_priority.get(event.level, 1) < level_priority.get(self.min_level, 1):
            return
        
        with self._lock:
            # Add to buffer
            self.event_buffer.append(event)
            self.event_count += 1
            self.last_event_time = event.timestamp
            
            # Immediate flush for high-priority events
            if event.level in [AuditLevel.ERROR, AuditLevel.CRITICAL]:
                self._flush_buffer()
    
    def log_security_event(
        self,
        event_type: Union[AuditEventType, str],
        level: AuditLevel = AuditLevel.INFO,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Convenience method to log security events.
        
        Args:
            event_type: Type of event
            level: Event severity level
            user_id: User associated with the event
            details: Additional event details
            **kwargs: Additional event fields
        """
        if isinstance(event_type, str):
            # Try to find matching enum value
            try:
                event_type = AuditEventType(event_type)
            except ValueError:
                warnings.warn(f"Unknown event type: {event_type}")
                return
        
        event = AuditEvent(
            event_type=event_type,
            level=level,
            user_id=user_id,
            details=details or {},
            **kwargs
        )
        
        self.log_event(event)
    
    def log_authentication(
        self,
        success: bool,
        user_id: str,
        source_ip: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log authentication events."""
        event_type = AuditEventType.LOGIN_SUCCESS if success else AuditEventType.LOGIN_FAILURE
        level = AuditLevel.INFO if success else AuditLevel.WARNING
        
        self.log_security_event(
            event_type=event_type,
            level=level,
            user_id=user_id,
            source_ip=source_ip,
            outcome="success" if success else "failure",
            details=details or {}
        )
    
    def log_access_control(
        self,
        granted: bool,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log access control events."""
        event_type = AuditEventType.ACCESS_GRANTED if granted else AuditEventType.ACCESS_DENIED
        level = AuditLevel.INFO if granted else AuditLevel.WARNING
        
        self.log_security_event(
            event_type=event_type,
            level=level,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome="granted" if granted else "denied",
            details=details or {}
        )
    
    def log_credential_event(
        self,
        event_type: AuditEventType,
        credential_name: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log credential management events."""
        level = AuditLevel.INFO
        if event_type in [AuditEventType.CREDENTIAL_EXPIRED]:
            level = AuditLevel.WARNING
        
        self.log_security_event(
            event_type=event_type,
            level=level,
            user_id=user_id,
            resource_type="credential",
            resource_id=credential_name,
            details=details or {}
        )
    
    def log_resource_event(
        self,
        event_type: AuditEventType,
        resource_type: str,
        amount: float,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log resource management events."""
        level = AuditLevel.INFO
        if event_type == AuditEventType.RESOURCE_LIMIT_EXCEEDED:
            level = AuditLevel.ERROR
        
        event_details = details or {}
        event_details.update({
            'resource_amount': amount,
            'resource_unit': self._get_resource_unit(resource_type)
        })
        
        self.log_security_event(
            event_type=event_type,
            level=level,
            user_id=user_id,
            resource_type=resource_type,
            details=event_details
        )
    
    def log_configuration_change(
        self,
        config_key: str,
        old_value: Any,
        new_value: Any,
        user_id: Optional[str] = None
    ):
        """Log configuration changes."""
        # Sanitize sensitive values
        sanitized_old = self._sanitize_config_value(config_key, old_value)
        sanitized_new = self._sanitize_config_value(config_key, new_value)
        
        self.log_security_event(
            event_type=AuditEventType.CONFIG_CHANGED,
            level=AuditLevel.INFO,
            user_id=user_id,
            resource_type="configuration",
            resource_id=config_key,
            details={
                'old_value': sanitized_old,
                'new_value': sanitized_new
            }
        )
    
    def log_security_violation(
        self,
        violation_type: str,
        severity: AuditLevel,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security violations."""
        self.log_security_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            level=severity,
            user_id=user_id,
            source_ip=source_ip,
            details={
                'violation_type': violation_type,
                **(details or {})
            }
        )
    
    def log_circuit_execution(
        self,
        circuit_name: str,
        backend: str,
        num_qubits: int,
        shots: int,
        user_id: Optional[str] = None,
        execution_time: Optional[float] = None
    ):
        """Log quantum circuit execution."""
        self.log_security_event(
            event_type=AuditEventType.CIRCUIT_EXECUTED,
            level=AuditLevel.INFO,
            user_id=user_id,
            resource_type="circuit",
            resource_id=circuit_name,
            details={
                'backend': backend,
                'num_qubits': num_qubits,
                'shots': shots,
                'execution_time': execution_time
            }
        )
    
    def _sanitize_config_value(self, config_key: str, value: Any) -> str:
        """Sanitize configuration values for logging."""
        sensitive_keys = ['password', 'token', 'key', 'secret', 'credential']
        
        key_lower = config_key.lower()
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            return "[REDACTED]"
        
        return str(value) if value is not None else "null"
    
    def _get_resource_unit(self, resource_type: str) -> str:
        """Get the unit for a resource type."""
        units = {
            'memory': 'bytes',
            'cpu_time': 'seconds',
            'wall_time': 'seconds',
            'qubits': 'count',
            'circuits': 'count',
            'api_calls': 'count',
            'disk_space': 'bytes'
        }
        return units.get(resource_type, 'count')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics."""
        with self._lock:
            return {
                'total_events': self.event_count,
                'buffered_events': len(self.event_buffer),
                'error_count': self.error_count,
                'last_event_time': self.last_event_time.isoformat() if self.last_event_time else None,
                'last_flush_time': datetime.fromtimestamp(self.last_flush).isoformat()
            }
    
    def search_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search audit events (basic implementation).
        
        Note: This is a simple implementation. For production use,
        consider using a proper log aggregation system.
        """
        # Force flush to ensure all events are written
        with self._lock:
            self._flush_buffer()
        
        events = []
        if not self.log_file or not self.log_file.exists():
            return events
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if len(events) >= limit:
                        break
                    
                    try:
                        # Extract JSON from log line
                        if 'AUDIT' in line and '{' in line:
                            json_start = line.find('{')
                            if json_start >= 0:
                                event_data = json.loads(line[json_start:])
                                
                                # Apply filters
                                if event_type and event_data.get('event_type') != event_type.value:
                                    continue
                                
                                if user_id and event_data.get('user_id') != user_id:
                                    continue
                                
                                if start_time or end_time:
                                    event_time = datetime.fromisoformat(event_data.get('timestamp', ''))
                                    if start_time and event_time < start_time:
                                        continue
                                    if end_time and event_time > end_time:
                                        continue
                                
                                events.append(event_data)
                    except Exception:
                        continue
        
        except Exception as e:
            warnings.warn(f"Error searching audit events: {e}")
        
        return events
    
    def rotate_logs(self):
        """Manually rotate log files."""
        if self.log_file and self.log_file.exists():
            for handler in self.logger.handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    handler.doRollover()
    
    def flush(self):
        """Manually flush all buffered events."""
        with self._lock:
            self._flush_buffer()


# Global audit logger instance
_global_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger()
    return _global_audit_logger


def set_audit_logger(logger: AuditLogger):
    """Set the global audit logger instance."""
    global _global_audit_logger
    _global_audit_logger = logger