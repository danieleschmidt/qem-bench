"""
Resilience Manager for QEM-Bench

Provides centralized resilience and fault tolerance orchestration for
quantum error mitigation operations.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, get_circuit_breaker
from ..errors import (
    QEMBenchError, ZNEError, PECError, VDError, CDRError,
    BackendConnectionError, ResourceExhaustionError, SecurityError
)


class ResilienceStrategy(Enum):
    """Available resilience strategies."""
    FAIL_FAST = "fail_fast"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ADAPTIVE_TIMEOUT = "adaptive_timeout"


@dataclass
class ResiliencePolicy:
    """Resilience policy configuration."""
    # Circuit breaker settings
    enable_circuit_breaker: bool = True
    circuit_failure_threshold: int = 5
    circuit_timeout_seconds: int = 60
    
    # Retry settings
    enable_retry: bool = True
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    
    # Timeout settings
    default_timeout_seconds: int = 300
    enable_adaptive_timeout: bool = True
    timeout_percentile: float = 0.95
    
    # Graceful degradation
    enable_graceful_degradation: bool = True
    degraded_mode_threshold: float = 0.8  # Error rate threshold
    
    # Health monitoring
    health_check_interval_seconds: int = 30
    unhealthy_threshold: int = 3
    
    # Performance tracking
    track_performance_metrics: bool = True
    metrics_window_minutes: int = 15


class ResilienceManager:
    """
    Centralized resilience management for QEM-Bench operations.
    
    Coordinates circuit breakers, retry policies, timeouts, and graceful
    degradation to ensure robust quantum error mitigation execution.
    """
    
    def __init__(
        self,
        policy: Optional[ResiliencePolicy] = None,
        name: str = "default"
    ):
        """Initialize resilience manager."""
        self.name = name
        self.policy = policy or ResiliencePolicy()
        
        # Component tracking
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._operation_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._health_status: Dict[str, bool] = {}
        
        # Performance tracking
        self._performance_history: List[Dict[str, Any]] = []
        self._adaptive_timeouts: Dict[str, float] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Logging
        self.logger = logging.getLogger(f"qem_bench.resilience.manager.{name}")
        
        # Background monitoring
        self._monitoring_thread = None
        self._shutdown_event = threading.Event()
        
        if self.policy.track_performance_metrics:
            self._start_monitoring()
        
        self.logger.info(f"Resilience manager '{name}' initialized with policy: {policy}")
    
    def execute_with_resilience(
        self,
        operation: Callable[[], Any],
        operation_name: str,
        strategies: Optional[List[ResilienceStrategy]] = None,
        timeout: Optional[float] = None,
        fallback: Optional[Callable[[], Any]] = None
    ) -> Any:
        """
        Execute operation with resilience strategies.
        
        Args:
            operation: Function to execute
            operation_name: Name for monitoring/logging
            strategies: Resilience strategies to apply
            timeout: Operation timeout (uses adaptive if None)
            fallback: Fallback function if operation fails
            
        Returns:
            Result of operation or fallback
            
        Raises:
            Exception: If operation fails and no fallback available
        """
        if strategies is None:
            strategies = [
                ResilienceStrategy.CIRCUIT_BREAKER,
                ResilienceStrategy.RETRY_WITH_BACKOFF,
                ResilienceStrategy.ADAPTIVE_TIMEOUT
            ]
        
        # Determine timeout
        effective_timeout = self._get_effective_timeout(operation_name, timeout)
        
        # Execute with strategies
        start_time = time.time()
        last_exception = None
        
        try:
            # Apply circuit breaker if enabled
            if (ResilienceStrategy.CIRCUIT_BREAKER in strategies and 
                self.policy.enable_circuit_breaker):
                
                circuit_breaker = self._get_or_create_circuit_breaker(operation_name)
                
                # Check if operation is allowed
                if circuit_breaker.is_open:
                    if fallback:
                        self.logger.warning(
                            f"Circuit breaker open for '{operation_name}', using fallback"
                        )
                        return self._execute_with_timeout(fallback, effective_timeout)
                    else:
                        raise QEMBenchError(
                            f"Circuit breaker open for operation '{operation_name}'"
                        )
            
            # Apply retry with backoff if enabled
            if (ResilienceStrategy.RETRY_WITH_BACKOFF in strategies and 
                self.policy.enable_retry):
                
                return self._execute_with_retry(
                    operation, operation_name, effective_timeout
                )
            
            # Apply adaptive timeout
            if ResilienceStrategy.ADAPTIVE_TIMEOUT in strategies:
                return self._execute_with_timeout(operation, effective_timeout)
            
            # Execute directly
            return operation()
        
        except Exception as e:
            last_exception = e
            execution_time = time.time() - start_time
            
            # Record failure metrics
            self._record_operation_result(
                operation_name, False, execution_time, str(e)
            )
            
            # Try graceful degradation
            if (ResilienceStrategy.GRACEFUL_DEGRADATION in strategies and 
                self.policy.enable_graceful_degradation and fallback):
                
                self.logger.warning(
                    f"Operation '{operation_name}' failed, attempting graceful degradation"
                )
                
                try:
                    return self._execute_with_timeout(fallback, effective_timeout)
                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback for '{operation_name}' also failed: {fallback_error}"
                    )
                    raise e
            
            raise e
        
        finally:
            execution_time = time.time() - start_time
            if last_exception is None:
                # Record success metrics
                self._record_operation_result(
                    operation_name, True, execution_time
                )
    
    def _execute_with_retry(
        self,
        operation: Callable[[], Any],
        operation_name: str,
        timeout: float
    ) -> Any:
        """Execute operation with retry policy."""
        last_exception = None
        delay = self.policy.base_delay_seconds
        
        for attempt in range(self.policy.max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.debug(
                        f"Retry attempt {attempt} for '{operation_name}' "
                        f"after {delay:.2f}s delay"
                    )
                    time.sleep(delay)
                
                return self._execute_with_timeout(operation, timeout)
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry based on exception type
                if not self._should_retry(e) or attempt >= self.policy.max_retries:
                    break
                
                # Calculate next delay with exponential backoff
                delay = min(
                    delay * self.policy.backoff_multiplier,
                    self.policy.max_delay_seconds
                )
        
        # All retries exhausted
        self.logger.error(
            f"Operation '{operation_name}' failed after {self.policy.max_retries} retries"
        )
        raise last_exception
    
    def _execute_with_timeout(
        self,
        operation: Callable[[], Any],
        timeout: float
    ) -> Any:
        """Execute operation with timeout."""
        import signal
        import threading
        
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = operation()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Operation timed out
            raise TimeoutError(f"Operation timed out after {timeout:.2f} seconds")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if exception should trigger retry."""
        # Don't retry on certain types of errors
        no_retry_types = [
            ValueError,  # Input validation errors
            TypeError,   # Type errors
            KeyboardInterrupt,  # User interruption
        ]
        
        if any(isinstance(exception, t) for t in no_retry_types):
            return False
        
        # Retry on backend and resource errors
        retry_types = [
            BackendConnectionError,
            ResourceExhaustionError,
            ConnectionError,
            TimeoutError,
        ]
        
        return any(isinstance(exception, t) for t in retry_types)
    
    def _get_effective_timeout(
        self,
        operation_name: str,
        timeout: Optional[float]
    ) -> float:
        """Get effective timeout for operation."""
        if timeout is not None:
            return timeout
        
        # Use adaptive timeout if enabled
        if (self.policy.enable_adaptive_timeout and 
            operation_name in self._adaptive_timeouts):
            return self._adaptive_timeouts[operation_name]
        
        return self.policy.default_timeout_seconds
    
    def _get_or_create_circuit_breaker(self, operation_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        with self._lock:
            if operation_name not in self._circuit_breakers:
                config = CircuitBreakerConfig(
                    failure_threshold=self.policy.circuit_failure_threshold,
                    timeout=self.policy.circuit_timeout_seconds
                )
                
                self._circuit_breakers[operation_name] = get_circuit_breaker(
                    f"{self.name}_{operation_name}", config
                )
            
            return self._circuit_breakers[operation_name]
    
    def _record_operation_result(
        self,
        operation_name: str,
        success: bool,
        execution_time: float,
        error_message: Optional[str] = None
    ) -> None:
        """Record operation result for metrics tracking."""
        if not self.policy.track_performance_metrics:
            return
        
        with self._lock:
            if operation_name not in self._operation_metrics:
                self._operation_metrics[operation_name] = []
            
            record = {
                "timestamp": datetime.now(),
                "success": success,
                "execution_time": execution_time,
                "error_message": error_message
            }
            
            self._operation_metrics[operation_name].append(record)
            
            # Keep only recent metrics
            cutoff = datetime.now() - timedelta(minutes=self.policy.metrics_window_minutes)
            self._operation_metrics[operation_name] = [
                r for r in self._operation_metrics[operation_name]
                if r["timestamp"] >= cutoff
            ]
            
            # Update adaptive timeout
            if self.policy.enable_adaptive_timeout and success:
                self._update_adaptive_timeout(operation_name, execution_time)
    
    def _update_adaptive_timeout(
        self,
        operation_name: str,
        execution_time: float
    ) -> None:
        """Update adaptive timeout based on execution time."""
        with self._lock:
            metrics = self._operation_metrics.get(operation_name, [])
            if len(metrics) < 5:  # Need minimum samples
                return
            
            # Calculate percentile-based timeout
            execution_times = [
                r["execution_time"] for r in metrics
                if r["success"]
            ]
            
            if execution_times:
                execution_times.sort()
                percentile_index = int(len(execution_times) * self.policy.timeout_percentile)
                percentile_time = execution_times[min(percentile_index, len(execution_times) - 1)]
                
                # Add buffer (50% more than percentile)
                adaptive_timeout = percentile_time * 1.5
                
                # Clamp to reasonable bounds
                adaptive_timeout = max(10.0, min(adaptive_timeout, 1800.0))  # 10s to 30min
                
                self._adaptive_timeouts[operation_name] = adaptive_timeout
                
                self.logger.debug(
                    f"Updated adaptive timeout for '{operation_name}': {adaptive_timeout:.2f}s "
                    f"(based on {len(execution_times)} samples)"
                )
    
    def _start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self._monitoring_thread.start()
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        self.logger.info(f"Started resilience monitoring for '{self.name}'")
        
        while not self._shutdown_event.wait(self.policy.health_check_interval_seconds):
            try:
                self._perform_health_check()
                self._update_health_status()
                self._cleanup_old_metrics()
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
        
        self.logger.info(f"Stopped resilience monitoring for '{self.name}'")
    
    def _perform_health_check(self) -> None:
        """Perform health check on all operations."""
        with self._lock:
            for operation_name, metrics in self._operation_metrics.items():
                if not metrics:
                    continue
                
                # Calculate error rate in recent window
                recent_cutoff = datetime.now() - timedelta(minutes=5)
                recent_metrics = [
                    m for m in metrics if m["timestamp"] >= recent_cutoff
                ]
                
                if len(recent_metrics) >= 10:  # Minimum samples for health check
                    error_rate = sum(1 for m in recent_metrics if not m["success"]) / len(recent_metrics)
                    is_healthy = error_rate < self.policy.degraded_mode_threshold
                    
                    previous_health = self._health_status.get(operation_name, True)
                    self._health_status[operation_name] = is_healthy
                    
                    if previous_health != is_healthy:
                        status = "healthy" if is_healthy else "unhealthy"
                        self.logger.warning(
                            f"Operation '{operation_name}' is now {status} "
                            f"(error rate: {error_rate:.2%})"
                        )
    
    def _update_health_status(self) -> None:
        """Update overall health status."""
        # Could implement overall system health logic here
        pass
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics data."""
        cutoff = datetime.now() - timedelta(minutes=self.policy.metrics_window_minutes)
        
        with self._lock:
            for operation_name in list(self._operation_metrics.keys()):
                old_count = len(self._operation_metrics[operation_name])
                self._operation_metrics[operation_name] = [
                    m for m in self._operation_metrics[operation_name]
                    if m["timestamp"] >= cutoff
                ]
                new_count = len(self._operation_metrics[operation_name])
                
                if old_count != new_count:
                    self.logger.debug(
                        f"Cleaned up {old_count - new_count} old metrics for '{operation_name}'"
                    )
    
    def get_resilience_metrics(self) -> Dict[str, Any]:
        """Get resilience metrics."""
        with self._lock:
            operation_stats = {}
            
            for operation_name, metrics in self._operation_metrics.items():
                if not metrics:
                    continue
                
                success_count = sum(1 for m in metrics if m["success"])
                total_count = len(metrics)
                error_rate = (total_count - success_count) / total_count if total_count > 0 else 0
                
                execution_times = [m["execution_time"] for m in metrics if m["success"]]
                avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
                
                operation_stats[operation_name] = {
                    "total_operations": total_count,
                    "success_count": success_count,
                    "error_rate": error_rate,
                    "avg_execution_time": avg_execution_time,
                    "is_healthy": self._health_status.get(operation_name, True),
                    "adaptive_timeout": self._adaptive_timeouts.get(operation_name),
                    "circuit_breaker_state": (
                        self._circuit_breakers[operation_name].state.value
                        if operation_name in self._circuit_breakers else None
                    )
                }
            
            return {
                "manager_name": self.name,
                "policy": {
                    "enable_circuit_breaker": self.policy.enable_circuit_breaker,
                    "enable_retry": self.policy.enable_retry,
                    "enable_graceful_degradation": self.policy.enable_graceful_degradation,
                    "max_retries": self.policy.max_retries,
                    "default_timeout": self.policy.default_timeout_seconds
                },
                "operations": operation_stats,
                "active_circuit_breakers": len(self._circuit_breakers),
                "monitoring_active": self._monitoring_thread and self._monitoring_thread.is_alive()
            }
    
    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._circuit_breakers.values():
                breaker.reset()
        
        self.logger.info("All circuit breakers reset")
    
    def update_policy(self, new_policy: ResiliencePolicy) -> None:
        """Update resilience policy."""
        old_policy = self.policy
        self.policy = new_policy
        
        self.logger.info(f"Resilience policy updated for '{self.name}'")
        
        # Restart monitoring if needed
        if (new_policy.track_performance_metrics and 
            not old_policy.track_performance_metrics):
            self._start_monitoring()
    
    def shutdown(self) -> None:
        """Shutdown resilience manager."""
        self._shutdown_event.set()
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        
        self.logger.info(f"Resilience manager '{self.name}' shut down")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Global resilience managers
_global_managers: Dict[str, ResilienceManager] = {}
_managers_lock = threading.Lock()


def get_resilience_manager(
    name: str = "default",
    policy: Optional[ResiliencePolicy] = None
) -> ResilienceManager:
    """Get or create resilience manager."""
    with _managers_lock:
        if name not in _global_managers:
            _global_managers[name] = ResilienceManager(policy, name)
        
        return _global_managers[name]