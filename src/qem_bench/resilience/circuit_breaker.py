"""
Circuit Breaker Implementation for QEM-Bench

Provides fault tolerance by monitoring failures and preventing cascading failures
in quantum error mitigation operations.
"""

import time
import threading
import logging
from enum import Enum
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Failures before opening
    timeout: int = 60  # Seconds to wait before half-open
    recovery_threshold: int = 3  # Successes needed to close
    monitoring_window: int = 300  # Seconds to track failures
    max_concurrent_calls: int = 100  # Max concurrent operations


@dataclass
class FailureRecord:
    """Record of a failure."""
    timestamp: datetime
    error_type: str
    error_message: str
    operation: str


class CircuitBreaker:
    """
    Circuit breaker for quantum error mitigation operations.
    
    Monitors operation failures and automatically opens the circuit
    when failure rate exceeds thresholds, preventing cascading failures.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """Initialize circuit breaker."""
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_state_change = datetime.now()
        
        # Failure tracking
        self._failures: List[FailureRecord] = []
        self._concurrent_calls = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Logging
        self.logger = logging.getLogger(f"qem_bench.resilience.circuit_breaker.{name}")
        
        self.logger.info(f"Circuit breaker '{name}' initialized with config: {config}")
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing)."""
        return self.state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN
    
    def call(self, operation: Callable[[], Any], operation_name: str = "unknown") -> Any:
        """
        Execute operation through circuit breaker.
        
        Args:
            operation: Function to execute
            operation_name: Name for logging/monitoring
            
        Returns:
            Result of operation
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception from the operation
        """
        # Check if circuit allows operation
        if not self._can_execute():
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open. "
                f"Failures: {self._failure_count}, "
                f"Last failure: {self._last_failure_time}"
            )
        
        # Check concurrent call limit
        with self._lock:
            if self._concurrent_calls >= self.config.max_concurrent_calls:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' concurrent call limit exceeded: "
                    f"{self._concurrent_calls}/{self.config.max_concurrent_calls}"
                )
            
            self._concurrent_calls += 1
        
        start_time = time.time()
        try:
            # Execute operation
            result = operation()
            
            # Record success
            execution_time = time.time() - start_time
            self._on_success(operation_name, execution_time)
            
            return result
            
        except Exception as e:
            # Record failure
            execution_time = time.time() - start_time
            self._on_failure(operation_name, e, execution_time)
            raise
            
        finally:
            with self._lock:
                self._concurrent_calls -= 1
    
    def _can_execute(self) -> bool:
        """Check if operation can be executed."""
        with self._lock:
            now = datetime.now()
            
            if self._state == CircuitState.CLOSED:
                return True
            
            elif self._state == CircuitState.OPEN:
                # Check if timeout period has passed
                if (self._last_failure_time and 
                    now - self._last_failure_time >= timedelta(seconds=self.config.timeout)):
                    
                    # Move to half-open state
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    self._last_state_change = now
                    
                    self.logger.info(
                        f"Circuit breaker '{self.name}' moving to HALF_OPEN state"
                    )
                    return True
                
                return False
            
            elif self._state == CircuitState.HALF_OPEN:
                return True
            
            return False
    
    def _on_success(self, operation_name: str, execution_time: float) -> None:
        """Handle successful operation."""
        with self._lock:
            self._success_count += 1
            
            # Clean old failures
            self._clean_old_failures()
            
            if self._state == CircuitState.HALF_OPEN:
                if self._success_count >= self.config.recovery_threshold:
                    # Close circuit
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._failures.clear()
                    self._last_state_change = datetime.now()
                    
                    self.logger.info(
                        f"Circuit breaker '{self.name}' recovered and CLOSED. "
                        f"Successful operations: {self._success_count}"
                    )
            
            self.logger.debug(
                f"Operation '{operation_name}' succeeded in {execution_time:.3f}s. "
                f"Success count: {self._success_count}"
            )
    
    def _on_failure(self, operation_name: str, error: Exception, execution_time: float) -> None:
        """Handle failed operation."""
        with self._lock:
            now = datetime.now()
            
            # Record failure
            failure_record = FailureRecord(
                timestamp=now,
                error_type=type(error).__name__,
                error_message=str(error),
                operation=operation_name
            )
            self._failures.append(failure_record)
            
            # Clean old failures and recount
            self._clean_old_failures()
            self._failure_count = len(self._failures)
            self._last_failure_time = now
            
            self.logger.warning(
                f"Operation '{operation_name}' failed in {execution_time:.3f}s: "
                f"{type(error).__name__}: {error}. "
                f"Failure count: {self._failure_count}"
            )
            
            # Check if circuit should open
            if (self._state == CircuitState.CLOSED and 
                self._failure_count >= self.config.failure_threshold):
                
                self._state = CircuitState.OPEN
                self._last_state_change = now
                
                self.logger.error(
                    f"Circuit breaker '{self.name}' OPENED due to {self._failure_count} failures. "
                    f"Recent failures: {[f.error_type for f in self._failures[-5:]]}"
                )
            
            elif self._state == CircuitState.HALF_OPEN:
                # Return to open state
                self._state = CircuitState.OPEN
                self._success_count = 0
                self._last_state_change = now
                
                self.logger.warning(
                    f"Circuit breaker '{self.name}' returned to OPEN state after failure"
                )
    
    def _clean_old_failures(self) -> None:
        """Remove failures outside monitoring window."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.config.monitoring_window)
        
        self._failures = [
            f for f in self._failures 
            if f.timestamp >= cutoff
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            self._clean_old_failures()
            
            recent_failures = self._failures[-10:] if self._failures else []
            
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": len(self._failures),
                "success_count": self._success_count,
                "concurrent_calls": self._concurrent_calls,
                "last_failure_time": self._last_failure_time,
                "last_state_change": self._last_state_change,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "timeout": self.config.timeout,
                    "recovery_threshold": self.config.recovery_threshold,
                    "monitoring_window": self.config.monitoring_window
                },
                "recent_failures": [
                    {
                        "timestamp": f.timestamp,
                        "error_type": f.error_type,
                        "error_message": f.error_message[:100] + "..." if len(f.error_message) > 100 else f.error_message,
                        "operation": f.operation
                    }
                    for f in recent_failures
                ]
            }
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._failures.clear()
            self._last_failure_time = None
            self._last_state_change = datetime.now()
            
            self.logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED state")
    
    def force_open(self) -> None:
        """Force circuit breaker to open state."""
        with self._lock:
            self._state = CircuitState.OPEN
            self._last_state_change = datetime.now()
            
            self.logger.warning(f"Circuit breaker '{self.name}' manually forced to OPEN state")
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"CircuitBreaker(name='{self.name}', state={self._state.value}, "
            f"failures={len(self._failures)}, concurrent={self._concurrent_calls})"
        )


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger("qem_bench.resilience.circuit_breaker_registry")
    
    def get_or_create(
        self, 
        name: str, 
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
                self.logger.info(f"Created new circuit breaker: {name}")
            
            return self._breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        with self._lock:
            return {
                name: breaker.get_metrics()
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
        
        self.logger.info("All circuit breakers reset")
    
    def cleanup_unused(self, max_age_minutes: int = 60) -> int:
        """Remove unused circuit breakers."""
        cutoff = datetime.now() - timedelta(minutes=max_age_minutes)
        removed_count = 0
        
        with self._lock:
            unused_breakers = [
                name for name, breaker in self._breakers.items()
                if breaker._last_state_change < cutoff and breaker._concurrent_calls == 0
            ]
            
            for name in unused_breakers:
                del self._breakers[name]
                removed_count += 1
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} unused circuit breakers")
        
        return removed_count


# Global registry instance
_global_registry: Optional[CircuitBreakerRegistry] = None
_registry_lock = threading.Lock()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get global circuit breaker registry."""
    global _global_registry
    
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = CircuitBreakerRegistry()
    
    return _global_registry


def get_circuit_breaker(
    name: str, 
    config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get or create circuit breaker from global registry."""
    registry = get_circuit_breaker_registry()
    return registry.get_or_create(name, config)