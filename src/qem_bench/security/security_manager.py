"""
Comprehensive Security Manager for QEM-Bench

This module provides a centralized security management system that coordinates
all security components and provides high-level security orchestration.
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import hashlib
import secrets

from .config import SecureConfig, get_secure_config
from .credentials import CredentialManager, get_credential_manager
from .input_sanitizer import InputSanitizer, InputType
from .resource_limiter import ResourceLimiter, get_global_resource_limiter
from .access_control import AccessControl, Permission
from .audit_logger import AuditLogger, AuditEvent, AuditEventType, AuditLevel
from .crypto_utils import CryptoUtils


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    enable_input_validation: bool = True
    enable_resource_limiting: bool = True
    enable_access_control: bool = True
    enable_audit_logging: bool = True
    enable_crypto_validation: bool = True
    
    # Thresholds and limits
    max_input_size: int = 10_000_000  # 10MB
    max_execution_time: int = 300  # 5 minutes
    max_memory_usage: int = 1_000_000_000  # 1GB
    
    # Security levels
    min_password_length: int = 12
    require_2fa: bool = False
    session_timeout: int = 3600  # 1 hour
    
    # Validation settings
    allowed_file_types: List[str] = field(default_factory=lambda: ['.py', '.json', '.yml', '.yaml'])
    blocked_operations: List[str] = field(default_factory=lambda: ['exec', 'eval'])


class SecurityManager:
    """
    Centralized security management system.
    
    Coordinates all security components and provides high-level security orchestration
    for the QEM-Bench quantum error mitigation framework.
    """
    
    def __init__(
        self,
        policy: Optional[SecurityPolicy] = None,
        config: Optional[SecureConfig] = None
    ):
        """Initialize the security manager."""
        self._policy = policy or SecurityPolicy()
        self._config = config or get_secure_config()
        
        # Initialize security components
        self._credential_manager = get_credential_manager()
        self._input_sanitizer = InputSanitizer()
        self._resource_limiter = get_global_resource_limiter()
        self._access_control = AccessControl()
        self._audit_logger = AuditLogger()
        self._crypto_utils = CryptoUtils()
        
        # Session and state management
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._security_alerts: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
        # Initialize security monitoring
        self._setup_monitoring()
        
        # Log security manager initialization
        self._audit_logger.log_event(
            AuditEvent(
                event_type=AuditEventType.SYSTEM_START,
                details={"component": "security_manager", "policy_hash": self._compute_policy_hash()}
            )
        )
    
    def _setup_monitoring(self) -> None:
        """Set up security monitoring and alerting."""
        # Configure security-specific logging
        logger = logging.getLogger("qem_bench.security")
        logger.setLevel(logging.INFO)
        
        # Set up periodic security checks
        self._last_security_check = datetime.now()
        self._security_check_interval = timedelta(minutes=5)
    
    def _compute_policy_hash(self) -> str:
        """Compute hash of current security policy."""
        policy_str = str(self._policy)
        return hashlib.sha256(policy_str.encode()).hexdigest()[:16]
    
    def validate_input(
        self,
        input_data: Any,
        input_type: InputType,
        context: Optional[str] = None
    ) -> bool:
        """
        Validate input data using security policies.
        
        Args:
            input_data: Data to validate
            input_type: Type of input (string, circuit, etc.)
            context: Optional context for validation
            
        Returns:
            True if input is valid and safe
            
        Raises:
            SecurityError: If input is invalid or potentially dangerous
        """
        if not self._policy.enable_input_validation:
            return True
        
        try:
            # Use input sanitizer for validation
            is_valid = self._input_sanitizer.validate_input(
                input_data, input_type, context
            )
            
            # Additional security checks
            if isinstance(input_data, str):
                if len(input_data) > self._policy.max_input_size:
                    self._log_security_event(
                        "input_size_exceeded",
                        AuditLevel.WARNING,
                        f"Input size {len(input_data)} exceeds limit {self._policy.max_input_size}"
                    )
                    return False
                
                # Check for blocked operations
                for blocked_op in self._policy.blocked_operations:
                    if blocked_op in input_data:
                        self._log_security_event(
                            "blocked_operation_detected",
                            AuditLevel.CRITICAL,
                            f"Blocked operation '{blocked_op}' detected in input"
                        )
                        return False
            
            if is_valid:
                self._log_security_event(
                    "input_validated",
                    AuditLevel.DEBUG,
                    f"Input validation successful for {input_type}",
                    {"context": context}
                )
            
            return is_valid
            
        except Exception as e:
            self._log_security_event(
                "input_validation_error",
                AuditLevel.ERROR,
                f"Error during input validation: {e}"
            )
            return False
    
    def check_resource_limits(
        self,
        operation: str,
        estimated_resources: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Check if operation is within resource limits.
        
        Args:
            operation: Name of operation to check
            estimated_resources: Estimated resource usage
            
        Returns:
            True if operation is within limits
        """
        if not self._policy.enable_resource_limiting:
            return True
        
        try:
            # Check with resource limiter
            within_limits = self._resource_limiter.check_limits(
                operation, estimated_resources
            )
            
            if not within_limits:
                self._log_security_event(
                    "resource_limit_exceeded",
                    AuditLevel.WARNING,
                    f"Resource limits exceeded for operation: {operation}",
                    {"estimated_resources": estimated_resources}
                )
            
            return within_limits
            
        except Exception as e:
            self._log_security_event(
                "resource_check_error",
                AuditLevel.ERROR,
                f"Error checking resource limits: {e}"
            )
            return False
    
    def check_permission(
        self,
        user_id: str,
        permission: Permission,
        resource: Optional[str] = None
    ) -> bool:
        """
        Check if user has required permission.
        
        Args:
            user_id: User identifier
            permission: Required permission
            resource: Optional resource identifier
            
        Returns:
            True if user has permission
        """
        if not self._policy.enable_access_control:
            return True
        
        try:
            has_permission = self._access_control.check_permission(
                user_id, permission, resource
            )
            
            if not has_permission:
                self._log_security_event(
                    "access_denied",
                    AuditLevel.WARNING,
                    f"Access denied for user {user_id}, permission: {permission}",
                    {"resource": resource}
                )
            
            return has_permission
            
        except Exception as e:
            self._log_security_event(
                "permission_check_error",
                AuditLevel.ERROR,
                f"Error checking permissions: {e}"
            )
            return False
    
    def create_secure_session(
        self,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a secure user session.
        
        Args:
            user_id: User identifier
            metadata: Optional session metadata
            
        Returns:
            Session token
        """
        session_id = secrets.token_urlsafe(32)
        
        session_data = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(seconds=self._policy.session_timeout),
            "metadata": metadata or {},
            "active": True
        }
        
        with self._lock:
            self._active_sessions[session_id] = session_data
        
        self._log_security_event(
            "session_created",
            AuditLevel.INFO,
            f"Secure session created for user: {user_id}",
            {"session_id": session_id[:8] + "..."}
        )
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Validate and retrieve session information.
        
        Args:
            session_id: Session token
            
        Returns:
            Session data if valid, None otherwise
        """
        with self._lock:
            session = self._active_sessions.get(session_id)
            
            if not session:
                return None
            
            # Check expiration
            if datetime.now() > session["expires_at"]:
                del self._active_sessions[session_id]
                self._log_security_event(
                    "session_expired",
                    AuditLevel.INFO,
                    f"Session expired: {session_id[:8]}..."
                )
                return None
            
            # Check if active
            if not session["active"]:
                return None
            
            return session.copy()
    
    def revoke_session(self, session_id: str) -> bool:
        """
        Revoke a user session.
        
        Args:
            session_id: Session token to revoke
            
        Returns:
            True if session was revoked
        """
        with self._lock:
            if session_id in self._active_sessions:
                self._active_sessions[session_id]["active"] = False
                self._log_security_event(
                    "session_revoked",
                    AuditLevel.INFO,
                    f"Session revoked: {session_id[:8]}..."
                )
                return True
        
        return False
    
    def perform_security_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive security check.
        
        Returns:
            Security check results
        """
        check_results = {
            "timestamp": datetime.now(),
            "overall_status": "healthy",
            "checks": {},
            "alerts": []
        }
        
        try:
            # Check credential security
            cred_status = self._credential_manager.get_security_status()
            check_results["checks"]["credentials"] = cred_status
            
            # Check resource usage
            resource_status = self._resource_limiter.get_status()
            check_results["checks"]["resources"] = resource_status
            
            # Check active sessions
            with self._lock:
                expired_sessions = 0
                for session_id, session in self._active_sessions.items():
                    if datetime.now() > session["expires_at"]:
                        expired_sessions += 1
                
                check_results["checks"]["sessions"] = {
                    "active_count": len(self._active_sessions),
                    "expired_count": expired_sessions
                }
            
            # Check security alerts
            alert_count = len(self._security_alerts)
            check_results["checks"]["alerts"] = {"count": alert_count}
            
            if alert_count > 10:
                check_results["overall_status"] = "warning"
                check_results["alerts"].append("High number of security alerts")
            
            # Log security check
            self._log_security_event(
                "security_check_completed",
                AuditLevel.INFO,
                f"Security check completed with status: {check_results['overall_status']}"
            )
            
            self._last_security_check = datetime.now()
            
        except Exception as e:
            check_results["overall_status"] = "error"
            check_results["error"] = str(e)
            
            self._log_security_event(
                "security_check_failed",
                AuditLevel.ERROR,
                f"Security check failed: {e}"
            )
        
        return check_results
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security-related metrics."""
        with self._lock:
            return {
                "active_sessions": len(self._active_sessions),
                "security_alerts": len(self._security_alerts),
                "last_security_check": self._last_security_check,
                "policy_hash": self._compute_policy_hash(),
                "components_status": {
                    "credential_manager": "active",
                    "input_sanitizer": "active", 
                    "resource_limiter": "active",
                    "access_control": "active",
                    "audit_logger": "active"
                }
            }
    
    def _log_security_event(
        self,
        event_type: str,
        severity: AuditLevel,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a security event."""
        if not self._policy.enable_audit_logging:
            return
        
        event = AuditEvent(
            event_type=AuditEventType.SYSTEM_START,  # Using a generic type for now
            level=severity,
            details={"message": message, "event_type": event_type, **(metadata or {})}
        )
        
        self._audit_logger.log_event(event)
        
        # Add to local alerts if warning or critical
        if severity in [AuditLevel.WARNING, AuditLevel.CRITICAL]:
            with self._lock:
                self._security_alerts.append({
                    "timestamp": datetime.now(),
                    "event_type": event_type,
                    "severity": severity.value,
                    "message": message
                })
                
                # Keep only last 100 alerts
                if len(self._security_alerts) > 100:
                    self._security_alerts = self._security_alerts[-100:]
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count removed."""
        removed_count = 0
        current_time = datetime.now()
        
        with self._lock:
            expired_sessions = [
                session_id for session_id, session in self._active_sessions.items()
                if current_time > session["expires_at"]
            ]
            
            for session_id in expired_sessions:
                del self._active_sessions[session_id]
                removed_count += 1
        
        if removed_count > 0:
            self._log_security_event(
                "expired_sessions_cleaned",
                AuditLevel.INFO,
                f"Cleaned up {removed_count} expired sessions"
            )
        
        return removed_count
    
    def update_policy(self, new_policy: SecurityPolicy) -> None:
        """Update security policy."""
        old_hash = self._compute_policy_hash()
        self._policy = new_policy
        new_hash = self._compute_policy_hash()
        
        self._log_security_event(
            "policy_updated",
            AuditLevel.INFO,
            "Security policy updated",
            {"old_hash": old_hash, "new_hash": new_hash}
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        # Clean up expired sessions
        self.cleanup_expired_sessions()
        
        # Log final metrics
        metrics = self.get_security_metrics()
        self._log_security_event(
            "security_manager_shutdown",
            AuditLevel.INFO,
            "Security Manager shutting down",
            {"final_metrics": metrics}
        )


# Global security manager instance
_global_security_manager: Optional[SecurityManager] = None
_security_manager_lock = threading.Lock()


def get_security_manager() -> SecurityManager:
    """Get or create global security manager instance."""
    global _global_security_manager
    
    if _global_security_manager is None:
        with _security_manager_lock:
            if _global_security_manager is None:
                _global_security_manager = SecurityManager()
    
    return _global_security_manager


def set_security_manager(manager: SecurityManager) -> None:
    """Set global security manager instance."""
    global _global_security_manager
    
    with _security_manager_lock:
        _global_security_manager = manager