"""
Comprehensive error handling framework for QEM-Bench.

This module provides custom exception classes, error recovery strategies,
and graceful degradation mechanisms for the quantum error mitigation library.
"""

import warnings
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import traceback
import sys


class ErrorSeverity(Enum):
    """Error severity levels for QEM-Bench operations."""
    LOW = "low"          # Warning-level issues that don't stop execution
    MEDIUM = "medium"    # Errors that affect specific operations
    HIGH = "high"        # Critical errors that stop execution
    CRITICAL = "critical"  # System-level failures


@dataclass
class ErrorContext:
    """Context information for errors and exceptions."""
    operation: str
    module: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    recovery_actions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Capture stack trace if not provided."""
        if self.stack_trace is None:
            self.stack_trace = traceback.format_stack()


# Base Exception Classes
class QEMBenchError(Exception):
    """
    Base exception class for QEM-Bench errors.
    
    All QEM-Bench specific exceptions should inherit from this class.
    """
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "context": {
                "operation": self.context.operation if self.context else "unknown",
                "module": self.context.module if self.context else "unknown",
                "parameters": self.context.parameters if self.context else {},
                "suggestions": self.context.suggestions if self.context else [],
                "recovery_actions": self.context.recovery_actions if self.context else []
            },
            "cause": str(self.cause) if self.cause else None
        }
    
    def with_suggestion(self, suggestion: str) -> "QEMBenchError":
        """Add a suggestion for resolving the error."""
        if self.context is None:
            self.context = ErrorContext(operation="unknown", module="unknown")
        self.context.suggestions.append(suggestion)
        return self
    
    def with_recovery_action(self, action: str) -> "QEMBenchError":
        """Add a recovery action for the error."""
        if self.context is None:
            self.context = ErrorContext(operation="unknown", module="unknown")
        self.context.recovery_actions.append(action)
        return self


# Validation Errors
class ValidationError(QEMBenchError):
    """Base class for validation errors."""
    
    def __init__(self, message: str, parameter: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.parameter = parameter


class ParameterValidationError(ValidationError):
    """Error raised when parameter validation fails."""
    
    def __init__(
        self,
        parameter: str,
        value: Any,
        expected_type: type = None,
        valid_range: tuple = None,
        valid_values: List[Any] = None,
        **kwargs
    ):
        self.parameter = parameter
        self.value = value
        self.expected_type = expected_type
        self.valid_range = valid_range
        self.valid_values = valid_values
        
        # Build detailed error message
        msg_parts = [f"Invalid parameter '{parameter}': {value}"]
        
        if expected_type:
            msg_parts.append(f"Expected type: {expected_type.__name__}")
        if valid_range:
            msg_parts.append(f"Valid range: {valid_range}")
        if valid_values:
            msg_parts.append(f"Valid values: {valid_values}")
        
        message = ". ".join(msg_parts)
        super().__init__(message, parameter=parameter, **kwargs)


class TypeValidationError(ValidationError):
    """Error raised when type validation fails."""
    
    def __init__(self, parameter: str, value: Any, expected_type: type, **kwargs):
        message = f"Parameter '{parameter}' must be of type {expected_type.__name__}, got {type(value).__name__}"
        super().__init__(message, parameter=parameter, **kwargs)


class RangeValidationError(ValidationError):
    """Error raised when range validation fails."""
    
    def __init__(self, parameter: str, value: Any, valid_range: tuple, **kwargs):
        message = f"Parameter '{parameter}' = {value} is outside valid range {valid_range}"
        super().__init__(message, parameter=parameter, **kwargs)


class CircuitValidationError(ValidationError):
    """Error raised when circuit validation fails."""
    pass


# Noise Model Errors
class NoiseModelError(QEMBenchError):
    """Base class for noise model errors."""
    pass


class InvalidNoiseModelError(NoiseModelError):
    """Error raised when a noise model is invalid or malformed."""
    pass


class NoiseChannelError(NoiseModelError):
    """Error raised for noise channel issues."""
    pass


class KrausOperatorError(NoiseModelError):
    """Error raised when Kraus operators are invalid."""
    
    def __init__(self, message: str, operators: List[Any] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.operators = operators


# Mitigation Errors
class MitigationError(QEMBenchError):
    """Base class for quantum error mitigation errors."""
    pass


class ZNEError(MitigationError):
    """Error raised during Zero-Noise Extrapolation."""
    pass


class PECError(MitigationError):
    """Error raised during Probabilistic Error Cancellation."""
    pass


class VDError(MitigationError):
    """Error raised during Virtual Distillation."""
    pass


class CDRError(MitigationError):
    """Error raised during Clifford Data Regression."""
    pass


class ExtrapolationError(MitigationError):
    """Error raised during extrapolation operations."""
    pass


class NoiseScalingError(MitigationError):
    """Error raised during noise scaling operations."""
    pass


# Backend Errors
class BackendError(QEMBenchError):
    """Base class for backend-related errors."""
    pass


class BackendConnectionError(BackendError):
    """Error raised when backend connection fails."""
    pass


class BackendConfigurationError(BackendError):
    """Error raised when backend configuration is invalid."""
    pass


class BackendCompatibilityError(BackendError):
    """Error raised when backend is incompatible with operation."""
    pass


class ExecutionError(BackendError):
    """Error raised during circuit execution."""
    pass


# Resource Errors
class ResourceError(QEMBenchError):
    """Base class for resource-related errors."""
    pass


class MemoryError(ResourceError):
    """Error raised when memory limits are exceeded."""
    pass


class TimeoutError(ResourceError):
    """Error raised when operations timeout."""
    pass


class QubitLimitError(ResourceError):
    """Error raised when qubit limits are exceeded."""
    pass


# Configuration Errors
class ConfigurationError(QEMBenchError):
    """Base class for configuration errors."""
    pass


class DependencyError(ConfigurationError):
    """Error raised when required dependencies are missing."""
    
    def __init__(self, dependency: str, operation: str = None, **kwargs):
        self.dependency = dependency
        self.operation = operation
        
        message = f"Missing dependency: {dependency}"
        if operation:
            message += f" (required for {operation})"
        
        super().__init__(message, **kwargs)
        
        # Add installation suggestion
        self.with_suggestion(f"Install {dependency} using: pip install {dependency}")


class VersionError(ConfigurationError):
    """Error raised when dependency versions are incompatible."""
    
    def __init__(
        self,
        dependency: str,
        current_version: str,
        required_version: str,
        **kwargs
    ):
        message = f"Incompatible version of {dependency}: {current_version} (required: {required_version})"
        super().__init__(message, **kwargs)


# Security Errors
class SecurityError(QEMBenchError):
    """Base class for security-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)


class InputSanitizationError(SecurityError):
    """Error raised when input sanitization fails."""
    pass


class SerializationError(SecurityError):
    """Error raised during unsafe serialization/deserialization."""
    pass


# Error Recovery and Handling
class ErrorHandler:
    """Central error handling and recovery system."""
    
    def __init__(self):
        self.error_handlers: Dict[type, Callable] = {}
        self.fallback_handlers: List[Callable] = []
        self.error_log: List[Dict[str, Any]] = []
    
    def register_handler(self, error_type: type, handler: Callable):
        """Register a specific error handler."""
        self.error_handlers[error_type] = handler
    
    def register_fallback_handler(self, handler: Callable):
        """Register a fallback error handler."""
        self.fallback_handlers.append(handler)
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> Any:
        """
        Handle an error using registered handlers.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            
        Returns:
            Result from error handler or None
        """
        # Log the error
        error_dict = {
            "type": type(error).__name__,
            "message": str(error),
            "context": context.to_dict() if context else None,
            "timestamp": None  # Will be set by logging system
        }
        self.error_log.append(error_dict)
        
        # Try specific handler first
        error_type = type(error)
        if error_type in self.error_handlers:
            try:
                return self.error_handlers[error_type](error, context)
            except Exception as handler_error:
                warnings.warn(f"Error handler failed: {handler_error}")
        
        # Try fallback handlers
        for handler in self.fallback_handlers:
            try:
                result = handler(error, context)
                if result is not None:
                    return result
            except Exception as handler_error:
                warnings.warn(f"Fallback handler failed: {handler_error}")
        
        # No handler worked, re-raise
        raise error
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about handled errors."""
        if not self.error_log:
            return {"total_errors": 0}
        
        error_types = {}
        for error in self.error_log:
            error_type = error["type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_log),
            "error_types": error_types,
            "most_common": max(error_types.items(), key=lambda x: x[1]) if error_types else None
        }


# Global error handler instance
_global_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _global_error_handler


def handle_error(error: Exception, context: Optional[ErrorContext] = None) -> Any:
    """Handle an error using the global error handler."""
    return _global_error_handler.handle_error(error, context)


# Common error handling decorators
def handle_exceptions(
    operation: str,
    module: str,
    reraise: bool = True,
    default_return: Any = None
):
    """
    Decorator for handling exceptions in QEM-Bench operations.
    
    Args:
        operation: Name of the operation being performed
        module: Module name where the operation occurs
        reraise: Whether to re-raise the exception after handling
        default_return: Default value to return if exception is caught
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    operation=operation,
                    module=module,
                    parameters={"args": args, "kwargs": kwargs}
                )
                
                try:
                    result = handle_error(e, context)
                    if result is not None:
                        return result
                except Exception:
                    pass
                
                if reraise:
                    raise
                else:
                    return default_return
        
        return wrapper
    return decorator


def safe_operation(
    operation: str,
    module: str,
    fallback_value: Any = None,
    log_errors: bool = True
):
    """
    Decorator for safe operations that shouldn't crash the system.
    
    Args:
        operation: Name of the operation
        module: Module name
        fallback_value: Value to return on error
        log_errors: Whether to log errors
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    context = ErrorContext(
                        operation=operation,
                        module=module,
                        parameters={"args": args, "kwargs": kwargs}
                    )
                    
                    # Convert to QEMBenchError if not already
                    if not isinstance(e, QEMBenchError):
                        e = QEMBenchError(
                            f"Error in {operation}: {str(e)}",
                            severity=ErrorSeverity.LOW,
                            context=context,
                            cause=e
                        )
                    
                    # Log but don't re-raise
                    try:
                        handle_error(e, context)
                    except Exception:
                        pass
                
                return fallback_value
        
        return wrapper
    return decorator


# Error recovery strategies
class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def can_recover(self, error: Exception, context: Optional[ErrorContext] = None) -> bool:
        """Check if this strategy can recover from the given error."""
        return False
    
    def recover(self, error: Exception, context: Optional[ErrorContext] = None) -> Any:
        """Attempt to recover from the error."""
        raise NotImplementedError


class RetryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy that retries the operation."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    def can_recover(self, error: Exception, context: Optional[ErrorContext] = None) -> bool:
        # Can retry network errors, temporary failures, etc.
        return isinstance(error, (BackendConnectionError, TimeoutError, ExecutionError))
    
    def recover(self, error: Exception, context: Optional[ErrorContext] = None) -> Any:
        import time
        
        for attempt in range(self.max_retries):
            try:
                # This would need to re-execute the original operation
                # Implementation depends on how we store the original operation
                time.sleep(self.backoff_factor * (2 ** attempt))
                # Re-execute operation here
                return None
            except Exception as retry_error:
                if attempt == self.max_retries - 1:
                    raise retry_error
                continue


class FallbackStrategy(ErrorRecoveryStrategy):
    """Recovery strategy that uses fallback values or methods."""
    
    def __init__(self, fallback_map: Dict[type, Any]):
        self.fallback_map = fallback_map
    
    def can_recover(self, error: Exception, context: Optional[ErrorContext] = None) -> bool:
        return type(error) in self.fallback_map
    
    def recover(self, error: Exception, context: Optional[ErrorContext] = None) -> Any:
        return self.fallback_map.get(type(error))


# Setup default error handlers
def setup_default_error_handlers():
    """Setup default error handlers for common scenarios."""
    handler = get_error_handler()
    
    # Dependency errors
    def handle_dependency_error(error: DependencyError, context: Optional[ErrorContext] = None):
        warnings.warn(f"Missing dependency: {error.dependency}. Some features may not work.")
        return None
    
    handler.register_handler(DependencyError, handle_dependency_error)
    
    # Parameter validation errors
    def handle_parameter_error(error: ParameterValidationError, context: Optional[ErrorContext] = None):
        warnings.warn(f"Parameter validation failed: {error.message}")
        # Could return a default value based on parameter type
        return None
    
    handler.register_handler(ParameterValidationError, handle_parameter_error)
    
    # Add recovery strategies
    retry_strategy = RetryStrategy(max_retries=3)
    fallback_strategy = FallbackStrategy({
        ValidationError: None,
        ConfigurationError: None
    })


# Initialize default handlers
setup_default_error_handlers()


# Utility functions
def create_error_context(
    operation: str,
    module: str,
    **parameters
) -> ErrorContext:
    """Create an error context with the given parameters."""
    return ErrorContext(
        operation=operation,
        module=module,
        parameters=parameters
    )


def raise_validation_error(
    parameter: str,
    value: Any,
    message: str = None,
    **kwargs
) -> None:
    """Raise a parameter validation error with context."""
    if message is None:
        message = f"Invalid value for parameter '{parameter}': {value}"
    
    raise ParameterValidationError(
        parameter=parameter,
        value=value,
        **kwargs
    ).with_suggestion(f"Check the documentation for valid values of '{parameter}'")


def raise_type_error(parameter: str, value: Any, expected_type: type) -> None:
    """Raise a type validation error."""
    raise TypeValidationError(
        parameter=parameter,
        value=value,
        expected_type=expected_type
    ).with_suggestion(f"Convert {parameter} to {expected_type.__name__} before passing")


def raise_range_error(parameter: str, value: Any, valid_range: tuple) -> None:
    """Raise a range validation error."""
    raise RangeValidationError(
        parameter=parameter,
        value=value,
        valid_range=valid_range
    ).with_suggestion(f"Use a value between {valid_range[0]} and {valid_range[1]}")


def check_dependency(
    dependency_name: str,
    operation: str = None,
    min_version: str = None
) -> bool:
    """
    Check if a dependency is available and optionally check version.
    
    Args:
        dependency_name: Name of the dependency
        operation: Operation that requires this dependency
        min_version: Minimum required version
        
    Returns:
        True if dependency is available and meets requirements
        
    Raises:
        DependencyError: If dependency is missing or version is incompatible
    """
    try:
        import importlib
        module = importlib.import_module(dependency_name)
        
        if min_version:
            if hasattr(module, '__version__'):
                current_version = module.__version__
                # Simple version comparison - in practice would use packaging.version
                if current_version < min_version:
                    raise VersionError(
                        dependency=dependency_name,
                        current_version=current_version,
                        required_version=min_version
                    )
            else:
                warnings.warn(f"Cannot check version of {dependency_name}")
        
        return True
        
    except ImportError:
        raise DependencyError(
            dependency=dependency_name,
            operation=operation
        )


# Export all error classes and utilities
__all__ = [
    # Enums
    "ErrorSeverity",
    
    # Context and base classes
    "ErrorContext",
    "QEMBenchError",
    
    # Validation errors
    "ValidationError",
    "ParameterValidationError", 
    "TypeValidationError",
    "RangeValidationError",
    "CircuitValidationError",
    
    # Noise model errors
    "NoiseModelError",
    "InvalidNoiseModelError",
    "NoiseChannelError",
    "KrausOperatorError",
    
    # Mitigation errors
    "MitigationError",
    "ZNEError",
    "PECError", 
    "VDError",
    "CDRError",
    "ExtrapolationError",
    "NoiseScalingError",
    
    # Backend errors
    "BackendError",
    "BackendConnectionError",
    "BackendConfigurationError",
    "BackendCompatibilityError",
    "ExecutionError",
    
    # Resource errors
    "ResourceError",
    "MemoryError",
    "TimeoutError",
    "QubitLimitError",
    
    # Configuration errors
    "ConfigurationError",
    "DependencyError",
    "VersionError",
    
    # Security errors
    "SecurityError",
    "InputSanitizationError",
    "SerializationError",
    
    # Error handling
    "ErrorHandler",
    "get_error_handler",
    "handle_error",
    
    # Decorators
    "handle_exceptions",
    "safe_operation",
    
    # Recovery strategies
    "ErrorRecoveryStrategy",
    "RetryStrategy",
    "FallbackStrategy",
    
    # Utilities
    "create_error_context",
    "raise_validation_error",
    "raise_type_error", 
    "raise_range_error",
    "check_dependency",
    "setup_default_error_handlers"
]