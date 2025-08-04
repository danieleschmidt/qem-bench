"""
Access Control and Rate Limiting for QEM-Bench

This module provides:
- Role-based access control (RBAC)
- Permission management
- Rate limiting for API calls and operations
- Session management
- Access logging and monitoring
"""

import time
import threading
from typing import Dict, List, Optional, Set, Any, Callable, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import warnings

from ..errors import SecurityError
from .audit_logger import AuditLogger, AuditEventType, AuditLevel


class Permission(Enum):
    """System permissions."""
    # Circuit operations
    CREATE_CIRCUIT = "create_circuit"
    EXECUTE_CIRCUIT = "execute_circuit"
    MODIFY_CIRCUIT = "modify_circuit"
    DELETE_CIRCUIT = "delete_circuit"
    
    # Backend access
    ACCESS_SIMULATOR = "access_simulator"
    ACCESS_HARDWARE = "access_hardware"
    CONFIGURE_BACKEND = "configure_backend"
    
    # Data operations
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    EXPORT_DATA = "export_data"
    DELETE_DATA = "delete_data"
    
    # System administration
    ADMIN_USERS = "admin_users"
    ADMIN_CONFIG = "admin_config"
    ADMIN_LOGS = "admin_logs"
    ADMIN_RESOURCES = "admin_resources"
    
    # Monitoring
    VIEW_METRICS = "view_metrics"
    VIEW_LOGS = "view_logs"
    SYSTEM_MONITOR = "system_monitor"


class Role(Enum):
    """System roles with default permissions."""
    GUEST = "guest"
    USER = "user"
    RESEARCHER = "researcher"
    ADMIN = "admin"
    SYSTEM = "system"


@dataclass
class User:
    """User account information."""
    user_id: str
    username: str
    roles: Set[Role] = field(default_factory=set)
    permissions: Set[Permission] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    active: bool = True
    session_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitRule:
    """Rate limiting rule."""
    limit: int  # Maximum number of requests
    window_seconds: int  # Time window in seconds
    burst_limit: Optional[int] = None  # Burst limit (short-term)
    burst_window_seconds: int = 1  # Burst window in seconds


class RateLimitResult(NamedTuple):
    """Result of rate limit check."""
    allowed: bool
    current_count: int
    limit: int
    reset_time: datetime
    retry_after: Optional[int] = None


class RateLimiter:
    """
    Token bucket rate limiter with burst support.
    
    Implements sliding window rate limiting with optional burst protection.
    """
    
    def __init__(self):
        """Initialize rate limiter."""
        self._lock = threading.RLock()
        self._requests: Dict[str, deque] = defaultdict(deque)
        self._burst_requests: Dict[str, deque] = defaultdict(deque)
    
    def check_rate_limit(
        self,
        key: str,
        rule: RateLimitRule,
        current_time: Optional[float] = None
    ) -> RateLimitResult:
        """
        Check if a request is within rate limits.
        
        Args:
            key: Rate limiting key (e.g., user_id, ip_address)
            rule: Rate limiting rule to apply
            current_time: Current timestamp (for testing)
            
        Returns:
            RateLimitResult with decision and metadata
        """
        if current_time is None:
            current_time = time.time()
        
        with self._lock:
            # Clean old requests
            self._cleanup_old_requests(key, rule, current_time)
            
            # Check burst limit first
            if rule.burst_limit:
                burst_count = len(self._burst_requests[key])
                if burst_count >= rule.burst_limit:
                    return RateLimitResult(
                        allowed=False,
                        current_count=burst_count,
                        limit=rule.burst_limit,
                        reset_time=datetime.fromtimestamp(current_time + rule.burst_window_seconds),
                        retry_after=rule.burst_window_seconds
                    )
            
            # Check main rate limit
            request_count = len(self._requests[key])
            if request_count >= rule.limit:
                oldest_request = self._requests[key][0] if self._requests[key] else current_time
                reset_time = oldest_request + rule.window_seconds
                retry_after = int(reset_time - current_time) if reset_time > current_time else 0
                
                return RateLimitResult(
                    allowed=False,
                    current_count=request_count,
                    limit=rule.limit,
                    reset_time=datetime.fromtimestamp(reset_time),
                    retry_after=retry_after if retry_after > 0 else None
                )
            
            # Request is allowed
            return RateLimitResult(
                allowed=True,
                current_count=request_count,
                limit=rule.limit,
                reset_time=datetime.fromtimestamp(current_time + rule.window_seconds)
            )
    
    def record_request(
        self,
        key: str,
        rule: RateLimitRule,
        current_time: Optional[float] = None
    ):
        """Record a request for rate limiting."""
        if current_time is None:
            current_time = time.time()
        
        with self._lock:
            self._requests[key].append(current_time)
            if rule.burst_limit:
                self._burst_requests[key].append(current_time)
            
            # Clean up periodically
            self._cleanup_old_requests(key, rule, current_time)
    
    def _cleanup_old_requests(self, key: str, rule: RateLimitRule, current_time: float):
        """Clean up old requests outside the time window."""
        # Clean main window
        window_start = current_time - rule.window_seconds
        while self._requests[key] and self._requests[key][0] < window_start:
            self._requests[key].popleft()
        
        # Clean burst window
        if rule.burst_limit:
            burst_start = current_time - rule.burst_window_seconds
            while self._burst_requests[key] and self._burst_requests[key][0] < burst_start:
                self._burst_requests[key].popleft()
    
    def get_stats(self, key: str) -> Dict[str, Any]:
        """Get rate limiting statistics for a key."""
        with self._lock:
            return {
                'current_requests': len(self._requests[key]),
                'burst_requests': len(self._burst_requests[key]),
                'oldest_request': self._requests[key][0] if self._requests[key] else None,
                'newest_request': self._requests[key][-1] if self._requests[key] else None
            }


class AccessControl:
    """
    Access control system for QEM-Bench.
    
    Provides role-based access control, permission management,
    and rate limiting functionality.
    """
    
    def __init__(self, enable_audit_logging: bool = True):
        """
        Initialize access control system.
        
        Args:
            enable_audit_logging: Whether to enable audit logging
        """
        self.enable_audit_logging = enable_audit_logging
        
        # Thread safety
        self._lock = threading.RLock()
        
        # User and role management
        self.users: Dict[str, User] = {}
        self.role_permissions: Dict[Role, Set[Permission]] = {}
        
        # Rate limiting
        self.rate_limiter = RateLimiter()
        self.rate_limit_rules: Dict[str, RateLimitRule] = {}
        
        # Audit logging
        if enable_audit_logging:
            self.audit_logger = AuditLogger()
        else:
            self.audit_logger = None
        
        # Setup default roles and permissions
        self._setup_default_roles()
        self._setup_default_rate_limits()
    
    def _setup_default_roles(self):
        """Setup default role permissions."""
        # Guest permissions
        self.role_permissions[Role.GUEST] = {
            Permission.READ_DATA,
            Permission.VIEW_METRICS
        }
        
        # User permissions
        self.role_permissions[Role.USER] = {
            Permission.CREATE_CIRCUIT,
            Permission.EXECUTE_CIRCUIT,
            Permission.MODIFY_CIRCUIT,
            Permission.ACCESS_SIMULATOR,
            Permission.READ_DATA,
            Permission.WRITE_DATA,
            Permission.VIEW_METRICS,
            Permission.VIEW_LOGS
        }
        
        # Researcher permissions
        self.role_permissions[Role.RESEARCHER] = {
            Permission.CREATE_CIRCUIT,
            Permission.EXECUTE_CIRCUIT,
            Permission.MODIFY_CIRCUIT,
            Permission.DELETE_CIRCUIT,
            Permission.ACCESS_SIMULATOR,
            Permission.ACCESS_HARDWARE,
            Permission.READ_DATA,
            Permission.WRITE_DATA,
            Permission.EXPORT_DATA,
            Permission.VIEW_METRICS,
            Permission.VIEW_LOGS
        }
        
        # Admin permissions
        self.role_permissions[Role.ADMIN] = set(Permission)
        
        # System permissions (internal use)
        self.role_permissions[Role.SYSTEM] = set(Permission)
    
    def _setup_default_rate_limits(self):
        """Setup default rate limiting rules."""
        self.rate_limit_rules.update({
            # API rate limits
            'api_calls': RateLimitRule(
                limit=100,
                window_seconds=60,
                burst_limit=10,
                burst_window_seconds=1
            ),
            
            # Circuit execution limits
            'circuit_execution': RateLimitRule(
                limit=50,
                window_seconds=300,  # 5 minutes
                burst_limit=5,
                burst_window_seconds=10
            ),
            
            # Backend access limits
            'backend_access': RateLimitRule(
                limit=20,
                window_seconds=60,
                burst_limit=3,
                burst_window_seconds=1
            ),
            
            # Data export limits
            'data_export': RateLimitRule(
                limit=10,
                window_seconds=3600,  # 1 hour
                burst_limit=2,
                burst_window_seconds=60
            ),
            
            # Authentication attempts
            'auth_attempts': RateLimitRule(
                limit=5,
                window_seconds=300,  # 5 minutes
                burst_limit=3,
                burst_window_seconds=60
            )
        })
    
    def create_user(
        self,
        user_id: str,
        username: str,
        roles: Optional[Set[Role]] = None,
        permissions: Optional[Set[Permission]] = None
    ) -> User:
        """
        Create a new user.
        
        Args:
            user_id: Unique user identifier
            username: Username
            roles: User roles
            permissions: Additional permissions
            
        Returns:
            Created user object
        """
        with self._lock:
            if user_id in self.users:
                raise SecurityError(f"User {user_id} already exists")
            
            user = User(
                user_id=user_id,
                username=username,
                roles=roles or {Role.USER},
                permissions=permissions or set()
            )
            
            self.users[user_id] = user
            
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    event_type=AuditEventType.LOGIN_SUCCESS,
                    user_id=user_id,
                    details={'username': username, 'roles': [r.value for r in user.roles]}
                )
            
            return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def update_user_roles(self, user_id: str, roles: Set[Role]):
        """Update user roles."""
        with self._lock:
            user = self.users.get(user_id)
            if not user:
                raise SecurityError(f"User {user_id} not found")
            
            old_roles = user.roles.copy()
            user.roles = roles
            
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    event_type=AuditEventType.PERMISSION_CHANGED,
                    user_id=user_id,
                    details={
                        'old_roles': [r.value for r in old_roles],
                        'new_roles': [r.value for r in roles]
                    }
                )
    
    def add_user_permission(self, user_id: str, permission: Permission):
        """Add permission to user."""
        with self._lock:
            user = self.users.get(user_id)
            if not user:
                raise SecurityError(f"User {user_id} not found")
            
            user.permissions.add(permission)
            
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    event_type=AuditEventType.PERMISSION_CHANGED,
                    user_id=user_id,
                    details={'added_permission': permission.value}
                )
    
    def remove_user_permission(self, user_id: str, permission: Permission):
        """Remove permission from user."""
        with self._lock:
            user = self.users.get(user_id)
            if not user:
                raise SecurityError(f"User {user_id} not found")
            
            user.permissions.discard(permission)
            
            if self.audit_logger:
                self.audit_logger.log_security_event(
                    event_type=AuditEventType.PERMISSION_CHANGED,
                    user_id=user_id,
                    details={'removed_permission': permission.value}
                )
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """
        Check if user has a specific permission.
        
        Args:
            user_id: User identifier
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        user = self.users.get(user_id)
        if not user or not user.active:
            return False
        
        # Check direct permissions
        if permission in user.permissions:
            return True
        
        # Check role-based permissions
        for role in user.roles:
            role_perms = self.role_permissions.get(role, set())
            if permission in role_perms:
                return True
        
        return False
    
    def require_permission(
        self,
        user_id: str,
        permission: Permission,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None
    ):
        """
        Require a specific permission, raising error if not granted.
        
        Args:
            user_id: User identifier
            permission: Required permission
            resource_type: Optional resource type
            resource_id: Optional resource identifier
            
        Raises:
            SecurityError: If permission is not granted
        """
        has_permission = self.check_permission(user_id, permission)
        
        if self.audit_logger:
            self.audit_logger.log_access_control(
                granted=has_permission,
                user_id=user_id,
                resource_type=resource_type or "system",
                resource_id=resource_id or "unknown",
                action=permission.value
            )
        
        if not has_permission:
            raise SecurityError(f"Permission denied: {permission.value}")
    
    def check_rate_limit(
        self,
        user_id: str,
        operation: str,
        record_request: bool = True
    ) -> RateLimitResult:
        """
        Check rate limits for a user operation.
        
        Args:
            user_id: User identifier
            operation: Operation type
            record_request: Whether to record the request
            
        Returns:
            RateLimitResult
        """
        rule = self.rate_limit_rules.get(operation)
        if not rule:
            # No rate limit rule - allow by default
            return RateLimitResult(
                allowed=True,
                current_count=0,
                limit=float('inf'),
                reset_time=datetime.now()
            )
        
        key = f"{user_id}:{operation}"
        result = self.rate_limiter.check_rate_limit(key, rule)
        
        if record_request and result.allowed:
            self.rate_limiter.record_request(key, rule)
        
        # Log rate limit violations
        if not result.allowed and self.audit_logger:
            self.audit_logger.log_security_event(
                event_type=AuditEventType.RESOURCE_LIMIT_EXCEEDED,
                level=AuditLevel.WARNING,
                user_id=user_id,
                details={
                    'operation': operation,
                    'current_count': result.current_count,
                    'limit': result.limit,
                    'retry_after': result.retry_after
                }
            )
        
        return result
    
    def add_rate_limit_rule(self, operation: str, rule: RateLimitRule):
        """Add or update a rate limit rule."""
        self.rate_limit_rules[operation] = rule
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user."""
        user = self.users.get(user_id)
        if not user or not user.active:
            return set()
        
        permissions = user.permissions.copy()
        
        # Add role-based permissions
        for role in user.roles:
            role_perms = self.role_permissions.get(role, set())
            permissions.update(role_perms)
        
        return permissions
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user statistics."""
        user = self.users.get(user_id)
        if not user:
            return {}
        
        # Get rate limiting stats
        rate_limit_stats = {}
        for operation in self.rate_limit_rules:
            key = f"{user_id}:{operation}"
            rate_limit_stats[operation] = self.rate_limiter.get_stats(key)
        
        return {
            'user_id': user.user_id,
            'username': user.username,
            'roles': [r.value for r in user.roles],
            'permissions': [p.value for p in self.get_user_permissions(user_id)],
            'created_at': user.created_at.isoformat(),
            'last_login': user.last_login.isoformat() if user.last_login else None,
            'active': user.active,
            'rate_limits': rate_limit_stats
        }


# Decorators for access control
def require_permission(permission: Permission):
    """Decorator to require a specific permission."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Try to extract user_id from arguments or context
            user_id = kwargs.get('user_id') or getattr(args[0], 'user_id', None) if args else None
            
            if user_id:
                access_control = get_global_access_control()
                access_control.require_permission(user_id, permission)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limit(operation: str):
    """Decorator to apply rate limiting to a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Try to extract user_id from arguments or context
            user_id = kwargs.get('user_id') or getattr(args[0], 'user_id', None) if args else None
            
            if user_id:
                access_control = get_global_access_control()
                result = access_control.check_rate_limit(user_id, operation)
                
                if not result.allowed:
                    raise SecurityError(
                        f"Rate limit exceeded for {operation}. "
                        f"Try again in {result.retry_after} seconds."
                    )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global access control instance
_global_access_control: Optional[AccessControl] = None


def get_global_access_control() -> AccessControl:
    """Get the global access control instance."""
    global _global_access_control
    if _global_access_control is None:
        _global_access_control = AccessControl()
    return _global_access_control


def set_global_access_control(access_control: AccessControl):
    """Set the global access control instance."""
    global _global_access_control
    _global_access_control = access_control