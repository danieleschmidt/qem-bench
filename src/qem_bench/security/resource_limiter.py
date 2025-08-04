"""
Resource Limiting and Management for QEM-Bench

This module provides protection against resource exhaustion attacks by:
- Limiting memory usage
- Controlling CPU time
- Managing concurrent operations
- Tracking resource quotas per user/session
- Monitoring system resources
- Implementing circuit complexity limits
- Rate limiting quantum backend access
"""

import os
import time
import psutil
import threading
from typing import Dict, Optional, Any, List, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import functools
import warnings

from ..errors import ResourceError, TimeoutError, MemoryError, QubitLimitError


class ResourceType(Enum):
    """Types of resources that can be limited."""
    MEMORY = "memory"
    CPU_TIME = "cpu_time"
    WALL_TIME = "wall_time"
    QUBITS = "qubits"
    CIRCUITS = "circuits"
    API_CALLS = "api_calls"
    CONCURRENT_OPERATIONS = "concurrent_operations"
    DISK_SPACE = "disk_space"
    NETWORK_BANDWIDTH = "network_bandwidth"


@dataclass
class ResourceQuota:
    """Resource quota definition."""
    resource_type: ResourceType
    limit: float
    window_seconds: Optional[int] = None  # For rate-based limits
    per_user: bool = False
    warning_threshold: float = 0.8
    description: str = ""


class ResourceUsage(NamedTuple):
    """Resource usage measurement."""
    resource_type: ResourceType
    current: float
    limit: float
    percentage: float
    timestamp: datetime


@dataclass
class UserSession:
    """User session for tracking per-user resources."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    resource_usage: Dict[ResourceType, float] = field(default_factory=dict)
    active_operations: int = 0
    total_operations: int = 0


class ResourceLimiter:
    """
    Resource limiter and monitor for QEM-Bench.
    
    Provides protection against resource exhaustion attacks by monitoring
    and limiting various types of resource usage including memory, CPU time,
    quantum circuit complexity, and API call rates.
    """
    
    def __init__(
        self,
        enable_monitoring: bool = True,
        monitoring_interval: float = 1.0,
        enable_enforcement: bool = True,
        enable_per_user_limits: bool = True
    ):
        """
        Initialize resource limiter.
        
        Args:
            enable_monitoring: Whether to enable continuous monitoring
            monitoring_interval: Monitoring interval in seconds
            enable_enforcement: Whether to enforce limits
            enable_per_user_limits: Whether to track per-user limits
        """
        self.enable_monitoring = enable_monitoring
        self.monitoring_interval = monitoring_interval
        self.enable_enforcement = enable_enforcement
        self.enable_per_user_limits = enable_per_user_limits
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Resource quotas
        self.quotas: Dict[ResourceType, ResourceQuota] = {}
        
        # Current resource usage
        self.current_usage: Dict[ResourceType, float] = defaultdict(float)
        
        # Per-user sessions and usage
        self.user_sessions: Dict[str, UserSession] = {}
        
        # Rate limiting
        self.rate_limits: Dict[ResourceType, deque] = defaultdict(deque)
        
        # Active operations tracking
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring data
        self.monitoring_enabled = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.usage_history: Dict[ResourceType, List[ResourceUsage]] = defaultdict(list)
        
        # Setup default quotas
        self._setup_default_quotas()
        
        # Start monitoring if enabled
        if enable_monitoring:
            self.start_monitoring()
    
    def _setup_default_quotas(self):
        """Setup default resource quotas."""
        default_quotas = [
            ResourceQuota(
                resource_type=ResourceType.MEMORY,
                limit=2 * 1024 * 1024 * 1024,  # 2GB
                description="Maximum memory usage in bytes"
            ),
            ResourceQuota(
                resource_type=ResourceType.CPU_TIME,
                limit=300.0,  # 5 minutes
                description="Maximum CPU time per operation in seconds"
            ),
            ResourceQuota(
                resource_type=ResourceType.WALL_TIME,
                limit=600.0,  # 10 minutes
                description="Maximum wall time per operation in seconds"
            ),
            ResourceQuota(
                resource_type=ResourceType.QUBITS,
                limit=50,
                description="Maximum number of qubits per circuit"
            ),
            ResourceQuota(
                resource_type=ResourceType.CIRCUITS,
                limit=1000,
                window_seconds=3600,  # Per hour
                description="Maximum number of circuits per hour"
            ),
            ResourceQuota(
                resource_type=ResourceType.API_CALLS,
                limit=100,
                window_seconds=60,  # Per minute
                per_user=True,
                description="Maximum API calls per minute per user"
            ),
            ResourceQuota(
                resource_type=ResourceType.CONCURRENT_OPERATIONS,
                limit=10,
                description="Maximum concurrent operations"
            ),
            ResourceQuota(
                resource_type=ResourceType.DISK_SPACE,
                limit=1 * 1024 * 1024 * 1024,  # 1GB
                description="Maximum disk space usage in bytes"
            ),
        ]
        
        for quota in default_quotas:
            self.add_quota(quota)
    
    def add_quota(self, quota: ResourceQuota):
        """Add or update a resource quota."""
        with self._lock:
            self.quotas[quota.resource_type] = quota
    
    def remove_quota(self, resource_type: ResourceType):
        """Remove a resource quota."""
        with self._lock:
            if resource_type in self.quotas:
                del self.quotas[resource_type]
    
    def get_quota(self, resource_type: ResourceType) -> Optional[ResourceQuota]:
        """Get a resource quota."""
        return self.quotas.get(resource_type)
    
    def check_resource_limit(
        self,
        resource_type: ResourceType,
        requested_amount: float,
        user_id: Optional[str] = None,
        operation_id: Optional[str] = None
    ) -> bool:
        """
        Check if a resource allocation request would exceed limits.
        
        Args:
            resource_type: Type of resource being requested
            requested_amount: Amount of resource requested
            user_id: Optional user identifier
            operation_id: Optional operation identifier
            
        Returns:
            True if request is within limits
            
        Raises:
            ResourceError: If limits would be exceeded and enforcement is enabled
        """
        with self._lock:
            quota = self.quotas.get(resource_type)
            if not quota:
                return True  # No limit set
            
            # Calculate current usage
            if quota.per_user and user_id:
                current = self._get_user_resource_usage(user_id, resource_type)
            else:
                current = self.current_usage[resource_type]
            
            # Add rate-based checking
            if quota.window_seconds:
                current = self._get_windowed_usage(resource_type, quota.window_seconds)
            
            # Check if request would exceed limit
            if current + requested_amount > quota.limit:
                if self.enable_enforcement:
                    raise ResourceError(
                        f"{resource_type.value} limit exceeded: "
                        f"{current + requested_amount} > {quota.limit}"
                    )
                return False
            
            # Warning threshold check
            usage_percentage = (current + requested_amount) / quota.limit
            if usage_percentage > quota.warning_threshold:
                warnings.warn(
                    f"{resource_type.value} usage high: "
                    f"{usage_percentage:.1%} of limit"
                )
            
            return True
    
    def allocate_resource(
        self,
        resource_type: ResourceType,
        amount: float,
        user_id: Optional[str] = None,
        operation_id: Optional[str] = None
    ) -> bool:
        """
        Allocate a resource after checking limits.
        
        Args:
            resource_type: Type of resource to allocate
            amount: Amount to allocate
            user_id: Optional user identifier
            operation_id: Optional operation identifier
            
        Returns:
            True if allocated successfully
        """
        if not self.check_resource_limit(resource_type, amount, user_id, operation_id):
            return False
        
        with self._lock:
            quota = self.quotas.get(resource_type)
            
            # Update usage tracking
            if quota and quota.per_user and user_id:
                session = self._get_or_create_session(user_id)
                session.resource_usage[resource_type] = session.resource_usage.get(resource_type, 0) + amount
                session.last_activity = datetime.now()
            else:
                self.current_usage[resource_type] += amount
            
            # Add to rate limiting tracking
            if quota and quota.window_seconds:
                self.rate_limits[resource_type].append((datetime.now(), amount))
                self._cleanup_rate_limit_history(resource_type, quota.window_seconds)
            
            # Track operation
            if operation_id:
                self.active_operations[operation_id] = {
                    'resource_type': resource_type,
                    'amount': amount,
                    'user_id': user_id,
                    'start_time': datetime.now()
                }
        
        return True
    
    def release_resource(
        self,
        resource_type: ResourceType,
        amount: float,
        user_id: Optional[str] = None,
        operation_id: Optional[str] = None
    ):
        """
        Release a previously allocated resource.
        
        Args:
            resource_type: Type of resource to release
            amount: Amount to release
            user_id: Optional user identifier
            operation_id: Optional operation identifier
        """
        with self._lock:
            quota = self.quotas.get(resource_type)
            
            # Update usage tracking
            if quota and quota.per_user and user_id:
                session = self.user_sessions.get(user_id)
                if session:
                    current = session.resource_usage.get(resource_type, 0)
                    session.resource_usage[resource_type] = max(0, current - amount)
                    session.last_activity = datetime.now()
            else:
                self.current_usage[resource_type] = max(0, self.current_usage[resource_type] - amount)
            
            # Remove operation tracking
            if operation_id and operation_id in self.active_operations:
                del self.active_operations[operation_id]
    
    def _get_user_resource_usage(self, user_id: str, resource_type: ResourceType) -> float:
        """Get current resource usage for a user."""
        session = self.user_sessions.get(user_id)
        if not session:
            return 0.0
        return session.resource_usage.get(resource_type, 0.0)
    
    def _get_windowed_usage(self, resource_type: ResourceType, window_seconds: int) -> float:
        """Get resource usage within a time window."""
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        
        total_usage = 0.0
        for timestamp, amount in self.rate_limits[resource_type]:
            if timestamp >= cutoff_time:
                total_usage += amount
        
        return total_usage
    
    def _cleanup_rate_limit_history(self, resource_type: ResourceType, window_seconds: int):
        """Clean up old rate limit history."""
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        rate_queue = self.rate_limits[resource_type]
        
        while rate_queue and rate_queue[0][0] < cutoff_time:
            rate_queue.popleft()
    
    def _get_or_create_session(self, user_id: str) -> UserSession:
        """Get or create a user session."""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = UserSession(session_id=user_id)
        return self.user_sessions[user_id]
    
    def start_monitoring(self):
        """Start resource monitoring thread."""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
    
    def _monitoring_worker(self):
        """Background monitoring worker."""
        while self.monitoring_enabled:
            try:
                self._collect_system_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                warnings.warn(f"Resource monitoring error: {e}")
    
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # Memory usage
            memory_usage = psutil.virtual_memory().used
            self._record_usage(ResourceType.MEMORY, memory_usage)
            
            # CPU usage (process-specific)
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            self._record_usage(ResourceType.CPU_TIME, cpu_percent)
            
            # Disk usage (for temporary files)
            disk_usage = psutil.disk_usage('/tmp').used
            self._record_usage(ResourceType.DISK_SPACE, disk_usage)
            
        except Exception as e:
            warnings.warn(f"Failed to collect system metrics: {e}")
    
    def _record_usage(self, resource_type: ResourceType, current_usage: float):
        """Record resource usage measurement."""
        with self._lock:
            quota = self.quotas.get(resource_type)
            if not quota:
                return
            
            percentage = (current_usage / quota.limit) * 100 if quota.limit > 0 else 0
            
            usage = ResourceUsage(
                resource_type=resource_type,
                current=current_usage,
                limit=quota.limit,
                percentage=percentage,
                timestamp=datetime.now()
            )
            
            # Keep history limited
            history = self.usage_history[resource_type]
            history.append(usage)
            if len(history) > 1000:  # Keep last 1000 measurements
                history.pop(0)
            
            # Check for threshold violations
            if percentage > quota.warning_threshold * 100:
                warnings.warn(
                    f"{resource_type.value} usage high: {percentage:.1f}%"
                )
    
    def get_current_usage(self, resource_type: ResourceType) -> Optional[ResourceUsage]:
        """Get current usage for a resource type."""
        history = self.usage_history.get(resource_type, [])
        if not history:
            return None
        return history[-1]
    
    def get_usage_history(
        self,
        resource_type: ResourceType,
        hours: int = 1
    ) -> List[ResourceUsage]:
        """Get usage history for a resource type."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        history = self.usage_history.get(resource_type, [])
        
        return [usage for usage in history if usage.timestamp >= cutoff_time]
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system resource usage summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'resources': {},
            'active_operations': len(self.active_operations),
            'active_sessions': len(self.user_sessions)
        }
        
        for resource_type in self.quotas:
            current_usage = self.get_current_usage(resource_type)
            if current_usage:
                summary['resources'][resource_type.value] = {
                    'current': current_usage.current,
                    'limit': current_usage.limit,
                    'percentage': current_usage.percentage
                }
        
        return summary
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Clean up expired user sessions."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._lock:
            expired_sessions = [
                session_id for session_id, session in self.user_sessions.items()
                if session.last_activity < cutoff_time
            ]
            
            for session_id in expired_sessions:
                del self.user_sessions[session_id]
        
        return len(expired_sessions)
    
    def get_user_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get resource usage summary for a specific user."""
        session = self.user_sessions.get(user_id)
        if not session:
            return None
        
        return {
            'session_id': session.session_id,
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'active_operations': session.active_operations,
            'total_operations': session.total_operations,
            'resource_usage': {
                resource_type.value: usage
                for resource_type, usage in session.resource_usage.items()
            }
        }


# Context manager for resource allocation
class ResourceContext:
    """Context manager for automatic resource management."""
    
    def __init__(
        self,
        limiter: ResourceLimiter,
        resource_type: ResourceType,
        amount: float,
        user_id: Optional[str] = None,
        operation_id: Optional[str] = None
    ):
        self.limiter = limiter
        self.resource_type = resource_type
        self.amount = amount
        self.user_id = user_id
        self.operation_id = operation_id
        self.allocated = False
    
    def __enter__(self):
        self.allocated = self.limiter.allocate_resource(
            self.resource_type,
            self.amount,
            self.user_id,
            self.operation_id
        )
        if not self.allocated:
            raise ResourceError(f"Failed to allocate {self.resource_type.value}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.allocated:
            self.limiter.release_resource(
                self.resource_type,
                self.amount,
                self.user_id,
                self.operation_id
            )


# Decorators for resource management
def limit_memory(max_memory_mb: int):
    """Decorator to limit memory usage of a function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter = get_global_resource_limiter()
            with ResourceContext(
                limiter,
                ResourceType.MEMORY,
                max_memory_mb * 1024 * 1024,
                operation_id=f"{func.__name__}_{id(args)}"
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def limit_qubits(max_qubits: int):
    """Decorator to limit number of qubits in quantum operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter = get_global_resource_limiter()
            with ResourceContext(
                limiter,
                ResourceType.QUBITS,
                max_qubits,
                operation_id=f"{func.__name__}_{id(args)}"
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def limit_execution_time(max_seconds: float):
    """Decorator to limit execution time of a function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter = get_global_resource_limiter()
            with ResourceContext(
                limiter,
                ResourceType.WALL_TIME,
                max_seconds,
                operation_id=f"{func.__name__}_{id(args)}"
            ):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    if elapsed > max_seconds:
                        raise TimeoutError(f"Function exceeded time limit: {elapsed:.2f}s > {max_seconds}s")
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    if elapsed > max_seconds:
                        raise TimeoutError(f"Function exceeded time limit: {elapsed:.2f}s > {max_seconds}s")
                    raise
        return wrapper
    return decorator


# Global resource limiter instance
_global_resource_limiter: Optional[ResourceLimiter] = None


def get_global_resource_limiter() -> ResourceLimiter:
    """Get the global resource limiter instance."""
    global _global_resource_limiter
    if _global_resource_limiter is None:
        _global_resource_limiter = ResourceLimiter()
    return _global_resource_limiter


def set_global_resource_limiter(limiter: ResourceLimiter):
    """Set the global resource limiter instance."""
    global _global_resource_limiter
    _global_resource_limiter = limiter