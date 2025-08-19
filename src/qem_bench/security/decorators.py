"""
Security Decorators for QEM-Bench

This module provides decorators for adding security features to functions:
- Authentication and authorization
- Input validation and sanitization
- Rate limiting
- Audit logging
- Resource management
- Error handling with security context
"""

import functools
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, Union
from datetime import datetime

from ..errors import SecurityError, ValidationError
from .access_control import Permission, get_global_access_control
from .input_sanitizer import InputType, get_global_input_sanitizer
from .resource_limiter import ResourceType, get_global_resource_limiter
from .audit_logger import AuditEventType, AuditLevel, get_audit_logger
from .crypto_utils import get_crypto_utils


def require_authentication(
    permissions: Optional[List[Permission]] = None,
    roles: Optional[List[str]] = None
):
    """
    Decorator to require authentication and optionally specific permissions.
    
    Args:
        permissions: List of required permissions
        roles: List of required roles
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract user_id from kwargs or first argument if it's a method
            user_id = kwargs.get('user_id')
            if not user_id and args and hasattr(args[0], 'user_id'):
                user_id = args[0].user_id
            
            if not user_id:
                raise SecurityError("Authentication required: no user_id provided")
            
            access_control = get_global_access_control()
            
            # Check if user exists and is active
            user = access_control.get_user(user_id)
            if not user or not user.active:
                raise SecurityError(f"Invalid or inactive user: {user_id}")
            
            # Check permissions
            if permissions:
                for permission in permissions:
                    access_control.require_permission(user_id, permission)
            
            # Check roles
            if roles:
                user_roles = {role.value for role in user.roles}
                required_roles = set(roles)
                if not required_roles.intersection(user_roles):
                    raise SecurityError(f"Required roles not found: {required_roles}")
            
            # Update last activity
            user.last_login = datetime.now()
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_input(
    **field_types: Union[InputType, Dict[str, Any]]
):
    """
    Decorator to validate function inputs.
    
    Args:
        **field_types: Mapping of parameter names to InputType or validation config
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sanitizer = get_global_input_sanitizer()
            
            # Get function signature for parameter mapping
            import inspect
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            # Validate positional arguments
            validated_args = []
            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                    if param_name in field_types:
                        field_config = field_types[param_name]
                        if isinstance(field_config, InputType):
                            validated_arg = sanitizer.sanitize(arg, field_config)
                        else:
                            # Advanced configuration
                            input_type = field_config.get('type', InputType.STRING)
                            validated_arg = sanitizer.sanitize(arg, input_type, **field_config)
                        validated_args.append(validated_arg)
                    else:
                        validated_args.append(arg)
                else:
                    validated_args.append(arg)
            
            # Validate keyword arguments
            validated_kwargs = {}
            for key, value in kwargs.items():
                if key in field_types:
                    field_config = field_types[key]
                    if isinstance(field_config, InputType):
                        validated_value = sanitizer.sanitize(value, field_config)
                    else:
                        # Advanced configuration
                        input_type = field_config.get('type', InputType.STRING)
                        validated_value = sanitizer.sanitize(value, input_type, **field_config)
                    validated_kwargs[key] = validated_value
                else:
                    validated_kwargs[key] = value
            
            return func(*validated_args, **validated_kwargs)
        
        return wrapper
    return decorator


def rate_limit(
    operation: str,
    requests_per_minute: Optional[int] = None,
    burst_limit: Optional[int] = None
):
    """
    Decorator to apply rate limiting to a function.
    
    Args:
        operation: Operation name for rate limiting
        requests_per_minute: Requests per minute limit
        burst_limit: Burst limit for short-term spikes
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract user_id
            user_id = kwargs.get('user_id')
            if not user_id and args and hasattr(args[0], 'user_id'):
                user_id = args[0].user_id
            
            if user_id:
                access_control = get_global_access_control()
                
                # Add custom rate limit rule if specified
                if requests_per_minute or burst_limit:
                    from .access_control import RateLimitRule
                    rule = RateLimitRule(
                        limit=requests_per_minute or 60,
                        window_seconds=60,
                        burst_limit=burst_limit,
                        burst_window_seconds=1
                    )
                    access_control.add_rate_limit_rule(operation, rule)
                
                # Check rate limit
                result = access_control.check_rate_limit(user_id, operation)
                
                if not result.allowed:
                    raise SecurityError(
                        f"Rate limit exceeded for {operation}. "
                        f"Try again in {result.retry_after} seconds."
                    )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def audit_log(
    event_type: Union[AuditEventType, str],
    level: AuditLevel = AuditLevel.INFO,
    include_args: bool = False,
    include_result: bool = False,
    sensitive_params: Optional[List[str]] = None
):
    """
    Decorator to add audit logging to a function.
    
    Args:
        event_type: Type of audit event
        level: Audit level
        include_args: Whether to include function arguments in log
        include_result: Whether to include function result in log
        sensitive_params: List of parameter names to redact
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            
            # Extract user_id
            user_id = kwargs.get('user_id')
            if not user_id and args and hasattr(args[0], 'user_id'):
                user_id = args[0].user_id
            
            # Prepare details
            details = {
                'function': func.__name__,
                'module': func.__module__,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add arguments if requested
            if include_args:
                import inspect
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                
                # Add positional arguments
                for i, arg in enumerate(args):
                    if i < len(param_names):
                        param_name = param_names[i]
                        if sensitive_params and param_name in sensitive_params:
                            details[f'arg_{param_name}'] = '[REDACTED]'
                        else:
                            details[f'arg_{param_name}'] = str(arg)[:100]  # Limit length
                
                # Add keyword arguments
                for key, value in kwargs.items():
                    if sensitive_params and key in sensitive_params:
                        details[f'kwarg_{key}'] = '[REDACTED]'
                    else:
                        details[f'kwarg_{key}'] = str(value)[:100]  # Limit length
            
            start_time = time.time()
            exception_occurred = False
            
            try:
                result = func(*args, **kwargs)
                
                # Add result if requested
                if include_result:
                    details['result'] = str(result)[:200]  # Limit length
                
                return result
                
            except Exception as e:
                exception_occurred = True
                details['error'] = str(e)
                details['exception_type'] = type(e).__name__
                raise
            
            finally:
                # Calculate execution time
                execution_time = time.time() - start_time
                details['execution_time'] = execution_time
                details['success'] = not exception_occurred
                
                # Log the event
                if isinstance(event_type, str):
                    try:
                        audit_event_type = AuditEventType(event_type)
                    except ValueError:
                        audit_event_type = AuditEventType.SYSTEM_STARTUP  # Fallback
                else:
                    audit_event_type = event_type
                
                audit_logger.log_security_event(
                    event_type=audit_event_type,
                    level=level,
                    user_id=user_id,
                    details=details
                )
        
        return wrapper
    return decorator


def secure_operation(
    permissions: Optional[List[Permission]] = None,
    validate_inputs: Optional[Dict[str, InputType]] = None,
    rate_limit_operation: Optional[str] = None,
    audit_event: Optional[AuditEventType] = None,
    resource_limits: Optional[Dict[ResourceType, float]] = None
):
    """
    Comprehensive security decorator combining multiple security features.
    
    Args:
        permissions: Required permissions
        validate_inputs: Input validation configuration
        rate_limit_operation: Rate limiting operation name
        audit_event: Audit event type
        resource_limits: Resource limits to enforce
    """
    def decorator(func: Callable) -> Callable:
        # Apply individual decorators in the right order
        decorated_func = func
        
        # Resource limits (innermost)
        if resource_limits:
            def resource_wrapper(*args, **kwargs):
                resource_limiter = get_global_resource_limiter()
                user_id = kwargs.get('user_id') or (args[0].user_id if args and hasattr(args[0], 'user_id') else None)
                
                # Check all resource limits
                for resource_type, limit in resource_limits.items():
                    if not resource_limiter.check_resource_limit(resource_type, limit, user_id):
                        raise SecurityError(f"Resource limit exceeded: {resource_type.value}")
                
                return decorated_func(*args, **kwargs)
            
            decorated_func = resource_wrapper
        
        # Input validation
        if validate_inputs:
            decorated_func = validate_input(**validate_inputs)(decorated_func)
        
        # Rate limiting
        if rate_limit_operation:
            decorated_func = rate_limit(rate_limit_operation)(decorated_func)
        
        # Authentication and authorization
        if permissions:
            decorated_func = require_authentication(permissions)(decorated_func)
        
        # Audit logging (outermost)
        if audit_event:
            decorated_func = audit_log(audit_event, include_args=True)(decorated_func)
        
        return decorated_func
    
    return decorator


def encrypt_sensitive_data(
    sensitive_fields: List[str],
    return_encrypted: bool = False
):
    """
    Decorator to automatically encrypt sensitive data in function results.
    
    Args:
        sensitive_fields: List of field names to encrypt
        return_encrypted: Whether to return encrypted data or decrypt for caller
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Only process dict results
            if not isinstance(result, dict):
                return result
            
            crypto = get_crypto_utils()
            processed_result = result.copy()
            
            for field in sensitive_fields:
                if field in processed_result:
                    value = processed_result[field]
                    if isinstance(value, str):
                        if return_encrypted:
                            # Encrypt for storage/transmission
                            encrypted_value = crypto.encrypt(value.encode('utf-8'))
                            processed_result[field] = encrypted_value.hex()
                        else:
                            # Data is already encrypted, decrypt for use
                            try:
                                encrypted_bytes = bytes.fromhex(value)
                                decrypted_value = crypto.decrypt(encrypted_bytes).decode('utf-8')
                                processed_result[field] = decrypted_value
                            except Exception:
                                # Value might not be encrypted
                                pass
            
            return processed_result
        
        return wrapper
    return decorator


def timeout(seconds: float):
    """
    Decorator to add timeout to function execution.
    
    Args:
        seconds: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise SecurityError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore old handler and cancel alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


def circuit_security(
    max_qubits: int = 50,
    max_depth: int = 1000,
    max_shots: int = 1000000,
    allowed_backends: Optional[List[str]] = None
):
    """
    Decorator for quantum circuit security validation.
    
    Args:
        max_qubits: Maximum number of qubits allowed
        max_depth: Maximum circuit depth allowed
        max_shots: Maximum number of shots allowed
        allowed_backends: List of allowed backend names
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract circuit parameters
            circuit_params = {}
            
            # Check kwargs for circuit parameters
            if 'num_qubits' in kwargs:
                circuit_params['num_qubits'] = kwargs['num_qubits']
            if 'depth' in kwargs:
                circuit_params['depth'] = kwargs['depth']
            if 'shots' in kwargs:
                circuit_params['shots'] = kwargs['shots']
            if 'backend' in kwargs:
                circuit_params['backend'] = kwargs['backend']
            
            # Check if first argument is a circuit object
            if args and hasattr(args[0], 'num_qubits'):
                circuit = args[0]
                circuit_params['num_qubits'] = getattr(circuit, 'num_qubits', 0)
                circuit_params['depth'] = getattr(circuit, 'depth', 0)
            
            # Validate parameters
            if 'num_qubits' in circuit_params:
                if circuit_params['num_qubits'] > max_qubits:
                    raise SecurityError(f"Too many qubits: {circuit_params['num_qubits']} > {max_qubits}")
            
            if 'depth' in circuit_params:
                if circuit_params['depth'] > max_depth:
                    raise SecurityError(f"Circuit depth too large: {circuit_params['depth']} > {max_depth}")
            
            if 'shots' in circuit_params:
                if circuit_params['shots'] > max_shots:
                    raise SecurityError(f"Too many shots: {circuit_params['shots']} > {max_shots}")
            
            if 'backend' in circuit_params and allowed_backends:
                if circuit_params['backend'] not in allowed_backends:
                    raise SecurityError(f"Backend not allowed: {circuit_params['backend']}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def error_handler(
    log_errors: bool = True,
    reraise: bool = True,
    fallback_value: Any = None
):
    """
    Decorator for secure error handling.
    
    Args:
        log_errors: Whether to log errors
        reraise: Whether to re-raise errors
        fallback_value: Value to return on error (if not reraising)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    audit_logger = get_audit_logger()
                    user_id = kwargs.get('user_id') or (args[0].user_id if args and hasattr(args[0], 'user_id') else None)
                    
                    audit_logger.log_security_event(
                        event_type=AuditEventType.SECURITY_VIOLATION,
                        level=AuditLevel.ERROR,
                        user_id=user_id,
                        details={
                            'function': func.__name__,
                            'error': str(e),
                            'error_type': type(e).__name__
                        }
                    )
                
                if reraise:
                    raise
                else:
                    return fallback_value
        
        return wrapper
    return decorator