"""
Error recovery mechanisms for QEM-Bench.

Provides automatic error recovery, fallback strategies, and graceful degradation
to ensure robust operation even when components fail.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Callable, TypeVar, Union, Tuple
from dataclasses import dataclass
from functools import wraps
from abc import ABC, abstractmethod
import traceback


T = TypeVar('T')


class RecoveryError(Exception):
    """Exception raised when all recovery strategies fail."""
    
    def __init__(self, original_errors: List[Exception], component: str = "unknown"):
        self.original_errors = original_errors
        self.component = component
        
        error_summary = f"All recovery strategies failed for {component}:\n"
        for i, error in enumerate(original_errors, 1):
            error_summary += f"  {i}. {type(error).__name__}: {str(error)}\n"
        
        super().__init__(error_summary)


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    
    strategy_name: str
    success: bool
    error: Optional[Exception]
    execution_time: float
    metadata: Dict[str, Any]


@dataclass
class RecoveryResult:
    """Result of recovery operation."""
    
    success: bool
    result: Any
    attempts: List[RecoveryAttempt]
    total_time: float
    final_strategy: Optional[str] = None
    
    @property
    def num_attempts(self) -> int:
        return len(self.attempts)
    
    @property
    def failed_attempts(self) -> List[RecoveryAttempt]:
        return [attempt for attempt in self.attempts if not attempt.success]
    
    @property
    def successful_attempt(self) -> Optional[RecoveryAttempt]:
        return next((attempt for attempt in self.attempts if attempt.success), None)


class RecoveryStrategy(ABC):
    """Abstract base class for recovery strategies."""
    
    def __init__(self, name: str, max_retries: int = 3, backoff_factor: float = 1.5):
        self.name = name
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(f"recovery.{name}")
    
    @abstractmethod
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if strategy can handle the given error."""
        pass
    
    @abstractmethod
    def recover(self, func: Callable[..., T], args: Tuple, kwargs: Dict[str, Any], 
                error: Exception, context: Dict[str, Any]) -> T:
        """Attempt to recover from the error."""
        pass
    
    def execute_with_retries(self, func: Callable[..., T], args: Tuple, kwargs: Dict[str, Any], 
                           context: Dict[str, Any]) -> Tuple[T, RecoveryAttempt]:
        """Execute recovery strategy with retries."""
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff
                    delay = self.backoff_factor ** (attempt - 1)
                    time.sleep(delay)
                    self.logger.info(f"Retry {attempt}/{self.max_retries} for {self.name}")
                
                result = self.recover(func, args, kwargs, last_error, context)
                execution_time = time.time() - start_time
                
                return result, RecoveryAttempt(
                    strategy_name=self.name,
                    success=True,
                    error=None,
                    execution_time=execution_time,
                    metadata={"attempts": attempt + 1}
                )
            
            except Exception as e:
                last_error = e
                self.logger.warning(f"Recovery attempt {attempt + 1} failed: {str(e)}")
        
        execution_time = time.time() - start_time
        return None, RecoveryAttempt(
            strategy_name=self.name,
            success=False,
            error=last_error,
            execution_time=execution_time,
            metadata={"attempts": self.max_retries + 1}
        )


class DefaultValueStrategy(RecoveryStrategy):
    """Recovery strategy that returns a default value."""
    
    def __init__(self, default_value: Any, error_types: List[type] = None):
        super().__init__("default_value", max_retries=0)
        self.default_value = default_value
        self.error_types = error_types or [Exception]
    
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        return any(isinstance(error, error_type) for error_type in self.error_types)
    
    def recover(self, func: Callable[..., T], args: Tuple, kwargs: Dict[str, Any], 
                error: Exception, context: Dict[str, Any]) -> T:
        self.logger.info(f"Returning default value due to {type(error).__name__}: {str(error)}")
        return self.default_value


class FallbackFunctionStrategy(RecoveryStrategy):
    """Recovery strategy that calls a fallback function."""
    
    def __init__(self, fallback_func: Callable, error_types: List[type] = None):
        super().__init__("fallback_function")
        self.fallback_func = fallback_func
        self.error_types = error_types or [Exception]
    
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        return any(isinstance(error, error_type) for error_type in self.error_types)
    
    def recover(self, func: Callable[..., T], args: Tuple, kwargs: Dict[str, Any], 
                error: Exception, context: Dict[str, Any]) -> T:
        self.logger.info(f"Using fallback function due to {type(error).__name__}")
        try:
            return self.fallback_func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Fallback function also failed: {str(e)}")
            raise


class RetryStrategy(RecoveryStrategy):
    """Recovery strategy that simply retries the original function."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5, 
                 error_types: List[type] = None):
        super().__init__("retry", max_retries, backoff_factor)
        self.error_types = error_types or [Exception]
    
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        # Don't retry certain error types
        non_retryable = (TypeError, ValueError, AttributeError, ImportError)
        if isinstance(error, non_retryable):
            return False
        
        return any(isinstance(error, error_type) for error_type in self.error_types)
    
    def recover(self, func: Callable[..., T], args: Tuple, kwargs: Dict[str, Any], 
                error: Exception, context: Dict[str, Any]) -> T:
        self.logger.info(f"Retrying original function after {type(error).__name__}")
        return func(*args, **kwargs)


class ParameterAdjustmentStrategy(RecoveryStrategy):
    """Recovery strategy that adjusts function parameters."""
    
    def __init__(self, adjustment_rules: Dict[str, Callable]):
        super().__init__("parameter_adjustment")
        self.adjustment_rules = adjustment_rules
    
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        # Check if we have adjustment rules for the error type or context
        error_type_name = type(error).__name__
        return error_type_name in self.adjustment_rules or 'default' in self.adjustment_rules
    
    def recover(self, func: Callable[..., T], args: Tuple, kwargs: Dict[str, Any], 
                error: Exception, context: Dict[str, Any]) -> T:
        error_type_name = type(error).__name__
        
        # Get adjustment rule
        adjustment_func = self.adjustment_rules.get(error_type_name)
        if adjustment_func is None:
            adjustment_func = self.adjustment_rules.get('default')
        
        if adjustment_func is None:
            raise error  # Cannot handle this error
        
        # Apply adjustments
        new_args, new_kwargs = adjustment_func(args, kwargs, error, context)
        self.logger.info(f"Adjusted parameters for {error_type_name}")
        
        return func(*new_args, **new_kwargs)


class CircuitSimplificationStrategy(RecoveryStrategy):
    """Recovery strategy specific to quantum circuits - simplifies circuit when simulation fails."""
    
    def __init__(self):
        super().__init__("circuit_simplification")
    
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        # Handle memory errors, runtime errors during simulation
        return isinstance(error, (MemoryError, RuntimeError)) and 'circuit' in context
    
    def recover(self, func: Callable[..., T], args: Tuple, kwargs: Dict[str, Any], 
                error: Exception, context: Dict[str, Any]) -> T:
        circuit = context.get('circuit')
        if circuit is None:
            raise error
        
        self.logger.info(f"Attempting circuit simplification due to {type(error).__name__}")
        
        # Try different simplification strategies
        simplified_circuit = self._simplify_circuit(circuit)
        
        # Replace circuit in args/kwargs
        new_args, new_kwargs = self._replace_circuit(args, kwargs, simplified_circuit)
        
        return func(*new_args, **new_kwargs)
    
    def _simplify_circuit(self, circuit: Any) -> Any:
        """Simplify quantum circuit."""
        # Try removing measurements first
        if hasattr(circuit, 'measurements') and circuit.measurements:
            circuit.measurements.clear()
            return circuit
        
        # Try reducing circuit depth
        if hasattr(circuit, 'gates') and len(circuit.gates) > 10:
            # Keep only first half of gates
            circuit.gates = circuit.gates[:len(circuit.gates)//2]
            return circuit
        
        return circuit
    
    def _replace_circuit(self, args: Tuple, kwargs: Dict[str, Any], new_circuit: Any) -> Tuple[Tuple, Dict[str, Any]]:
        """Replace circuit in function arguments."""
        # Simple replacement - assume circuit is first argument
        if args:
            new_args = (new_circuit,) + args[1:]
            return new_args, kwargs
        
        # Or in kwargs
        if 'circuit' in kwargs:
            new_kwargs = kwargs.copy()
            new_kwargs['circuit'] = new_circuit
            return args, new_kwargs
        
        return args, kwargs


class ZNERecoveryStrategy(RecoveryStrategy):
    """Recovery strategy specific to ZNE - adjusts noise factors when extrapolation fails."""
    
    def __init__(self):
        super().__init__("zne_recovery")
    
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        return ('zne' in context or 'noise_factors' in context) and \
               isinstance(error, (ValueError, RuntimeError, np.linalg.LinAlgError))
    
    def recover(self, func: Callable[..., T], args: Tuple, kwargs: Dict[str, Any], 
                error: Exception, context: Dict[str, Any]) -> T:
        self.logger.info(f"Attempting ZNE parameter adjustment due to {type(error).__name__}")
        
        # Adjust ZNE parameters
        new_args, new_kwargs = self._adjust_zne_parameters(args, kwargs, error, context)
        
        return func(*new_args, **new_kwargs)
    
    def _adjust_zne_parameters(self, args: Tuple, kwargs: Dict[str, Any], 
                              error: Exception, context: Dict[str, Any]) -> Tuple[Tuple, Dict[str, Any]]:
        """Adjust ZNE parameters based on error type."""
        new_kwargs = kwargs.copy()
        
        # Reduce noise factor range
        if 'noise_factors' in new_kwargs:
            noise_factors = new_kwargs['noise_factors']
            # Use smaller range
            new_kwargs['noise_factors'] = [f for f in noise_factors if f <= 2.5]
        
        # Change extrapolation method
        if 'extrapolator' in new_kwargs:
            if new_kwargs['extrapolator'] == 'richardson':
                new_kwargs['extrapolator'] = 'polynomial'
                self.logger.info("Switched extrapolation method from richardson to polynomial")
        
        return args, new_kwargs


class RecoveryManager:
    """Manages recovery strategies and executes them in order."""
    
    def __init__(self):
        self.strategies: List[RecoveryStrategy] = []
        self.logger = logging.getLogger("recovery.manager")
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Setup default recovery strategies."""
        # Order matters - more specific strategies first
        self.add_strategy(CircuitSimplificationStrategy())
        self.add_strategy(ZNERecoveryStrategy())
        self.add_strategy(RetryStrategy(max_retries=2))
    
    def add_strategy(self, strategy: RecoveryStrategy):
        """Add a recovery strategy."""
        self.strategies.append(strategy)
        self.logger.info(f"Added recovery strategy: {strategy.name}")
    
    def remove_strategy(self, name: str) -> bool:
        """Remove a recovery strategy by name."""
        for i, strategy in enumerate(self.strategies):
            if strategy.name == name:
                del self.strategies[i]
                self.logger.info(f"Removed recovery strategy: {name}")
                return True
        return False
    
    def execute_with_recovery(self, func: Callable[..., T], *args, 
                             context: Optional[Dict[str, Any]] = None, **kwargs) -> RecoveryResult:
        """Execute function with automatic recovery."""
        start_time = time.time()
        context = context or {}
        attempts = []
        original_error = None
        
        # Try original function first
        try:
            result = func(*args, **kwargs)
            total_time = time.time() - start_time
            
            return RecoveryResult(
                success=True,
                result=result,
                attempts=[],
                total_time=total_time,
                final_strategy="original"
            )
        
        except Exception as e:
            original_error = e
            self.logger.warning(f"Original function failed: {type(e).__name__}: {str(e)}")
        
        # Try recovery strategies
        for strategy in self.strategies:
            if not strategy.can_handle(original_error, context):
                continue
            
            self.logger.info(f"Trying recovery strategy: {strategy.name}")
            
            try:
                result, attempt = strategy.execute_with_retries(func, args, kwargs, context)
                attempts.append(attempt)
                
                if attempt.success:
                    total_time = time.time() - start_time
                    self.logger.info(f"Recovery successful using {strategy.name}")
                    
                    return RecoveryResult(
                        success=True,
                        result=result,
                        attempts=attempts,
                        total_time=total_time,
                        final_strategy=strategy.name
                    )
            
            except Exception as e:
                self.logger.error(f"Recovery strategy {strategy.name} failed: {str(e)}")
                continue
        
        # All strategies failed
        total_time = time.time() - start_time
        all_errors = [original_error] + [attempt.error for attempt in attempts if attempt.error]
        
        return RecoveryResult(
            success=False,
            result=RecoveryError(all_errors, context.get('component', 'unknown')),
            attempts=attempts,
            total_time=total_time
        )
    
    def list_strategies(self) -> List[str]:
        """List all registered strategies."""
        return [strategy.name for strategy in self.strategies]


# Global recovery manager
_recovery_manager = RecoveryManager()


def with_recovery(context: Optional[Dict[str, Any]] = None):
    """Decorator to add automatic recovery to functions."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            recovery_result = _recovery_manager.execute_with_recovery(
                func, *args, context=context, **kwargs
            )
            
            if recovery_result.success:
                return recovery_result.result
            else:
                raise recovery_result.result  # This will be a RecoveryError
        
        return wrapper
    return decorator


def add_recovery_strategy(strategy: RecoveryStrategy):
    """Add a recovery strategy to global manager."""
    _recovery_manager.add_strategy(strategy)


def execute_with_recovery(func: Callable[..., T], *args, 
                         context: Optional[Dict[str, Any]] = None, **kwargs) -> RecoveryResult:
    """Execute function with recovery using global manager."""
    return _recovery_manager.execute_with_recovery(func, *args, context=context, **kwargs)


class RobustWrapper:
    """Wrapper that adds recovery capabilities to any object."""
    
    def __init__(self, wrapped_object: Any, context: Optional[Dict[str, Any]] = None):
        self._wrapped = wrapped_object
        self._context = context or {}
    
    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._wrapped, name)
        
        if callable(attr):
            # Wrap callable attributes with recovery
            @wraps(attr)
            def robust_method(*args, **kwargs):
                result = execute_with_recovery(
                    attr, *args, context=self._context, **kwargs
                )
                if result.success:
                    return result.result
                else:
                    raise result.result
            
            return robust_method
        else:
            return attr
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            setattr(self._wrapped, name, value)


def make_robust(obj: Any, context: Optional[Dict[str, Any]] = None) -> RobustWrapper:
    """Make any object robust by wrapping it with recovery capabilities."""
    return RobustWrapper(obj, context)


# Example usage and convenience functions
def create_robust_simulator(simulator_class: type, *args, **kwargs) -> RobustWrapper:
    """Create a robust quantum simulator with automatic error recovery."""
    simulator = simulator_class(*args, **kwargs)
    context = {'component': 'simulator', 'type': 'quantum_simulation'}
    return make_robust(simulator, context)


def create_robust_zne(zne_class: type, *args, **kwargs) -> RobustWrapper:
    """Create a robust ZNE instance with automatic error recovery."""
    zne = zne_class(*args, **kwargs)
    context = {'component': 'zne', 'type': 'error_mitigation'}
    return make_robust(zne, context)