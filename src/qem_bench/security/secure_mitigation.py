"""
Security-Enhanced Mitigation Methods

This module demonstrates how to integrate security features into quantum error
mitigation methods, providing a template for securing the entire library.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from functools import wraps

from ..mitigation.zne.core import ZeroNoiseExtrapolation, ZNEConfig
from ..mitigation.zne.result import ZNEResult
from .decorators import (
    secure_operation, validate_input, audit_log, circuit_security,
    require_authentication, rate_limit
)
from .access_control import Permission
from .input_sanitizer import InputType
from .audit_logger import AuditEventType
from .resource_limiter import ResourceType


class SecureZeroNoiseExtrapolation(ZeroNoiseExtrapolation):
    """
    Security-enhanced Zero-Noise Extrapolation.
    
    This class demonstrates how to add comprehensive security features
    to quantum error mitigation methods while maintaining the original API.
    
    Security features added:
    - Input validation and sanitization
    - Authentication and authorization
    - Resource limiting
    - Rate limiting
    - Audit logging
    - Circuit parameter validation
    """
    
    def __init__(
        self,
        user_id: Optional[str] = None,
        enable_security: bool = True,
        **kwargs
    ):
        """
        Initialize secure ZNE with security features.
        
        Args:
            user_id: User identifier for authentication and audit logging
            enable_security: Whether to enable security features
            **kwargs: Arguments passed to parent ZeroNoiseExtrapolation
        """
        self.user_id = user_id
        self.enable_security = enable_security
        
        # Validate noise factors before passing to parent
        if 'noise_factors' in kwargs and kwargs['noise_factors']:
            kwargs['noise_factors'] = self._validate_noise_factors(kwargs['noise_factors'])
        
        super().__init__(**kwargs)
    
    def _validate_noise_factors(self, noise_factors: List[float]) -> List[float]:
        """Validate noise factors with security constraints."""
        from .input_sanitizer import get_global_input_sanitizer
        
        sanitizer = get_global_input_sanitizer()
        
        # Validate each noise factor
        validated_factors = []
        for factor in noise_factors:
            # Sanitize as float
            validated_factor = sanitizer.sanitize(factor, InputType.FLOAT)
            
            # Security constraints
            if validated_factor < 1.0:
                raise ValueError(f"Noise factor must be >= 1.0, got {validated_factor}")
            if validated_factor > 10.0:  # Security limit
                raise ValueError(f"Noise factor too large: {validated_factor} > 10.0")
            
            validated_factors.append(validated_factor)
        
        # Limit number of factors for resource management
        if len(validated_factors) > 20:
            raise ValueError(f"Too many noise factors: {len(validated_factors)} > 20")
        
        return validated_factors
    
    @secure_operation(
        permissions=[Permission.EXECUTE_CIRCUIT],
        validate_inputs={
            'num_shots': {'type': InputType.INTEGER, 'min_value': 1, 'max_value': 1000000},
            'backend_name': InputType.BACKEND_NAME,
        },
        rate_limit_operation='zne_execution',
        audit_event=AuditEventType.CIRCUIT_EXECUTED,
        resource_limits={
            ResourceType.QUBITS: 50,
            ResourceType.MEMORY: 1024 * 1024 * 1024,  # 1GB
            ResourceType.CPU_TIME: 300  # 5 minutes
        }
    )
    @circuit_security(
        max_qubits=50,
        max_depth=1000,
        max_shots=1000000,
        allowed_backends=['jax_simulator', 'qiskit_simulator']
    )
    def mitigate(
        self,
        circuit: Any,
        backend: Any,
        observable: Any,
        num_shots: int = 1000,
        backend_name: Optional[str] = None,
        **kwargs
    ) -> ZNEResult:
        """
        Perform Zero-Noise Extrapolation with security features.
        
        Args:
            circuit: Quantum circuit to mitigate
            backend: Quantum backend for execution
            observable: Observable to measure
            num_shots: Number of measurement shots
            backend_name: Name of the backend (for validation)
            **kwargs: Additional arguments
            
        Returns:
            ZNE result with mitigation data
        """
        if not self.enable_security:
            # Fallback to parent implementation without security
            return super().mitigate(circuit, backend, observable, **kwargs)
        
        # Additional circuit validation
        if hasattr(circuit, 'num_qubits'):
            if circuit.num_qubits > 50:
                raise ValueError(f"Circuit has too many qubits: {circuit.num_qubits}")
        
        if hasattr(circuit, 'depth'):
            if circuit.depth > 1000:
                raise ValueError(f"Circuit depth too large: {circuit.depth}")
        
        # Validate backend
        if backend_name and backend_name not in ['jax_simulator', 'qiskit_simulator']:
            raise ValueError(f"Backend not allowed: {backend_name}")
        
        # Perform mitigation with parent class
        result = super().mitigate(circuit, backend, observable, **kwargs)
        
        # Add security metadata to result
        if hasattr(result, 'metadata'):
            result.metadata['security_enabled'] = True
            result.metadata['user_id'] = self.user_id
            result.metadata['validated_parameters'] = {
                'num_shots': num_shots,
                'noise_factors': self.noise_factors,
                'backend': backend_name
            }
        
        return result
    
    @validate_input(
        noise_factors=InputType.JSON,
        extrapolator=InputType.STRING
    )
    @audit_log(AuditEventType.CONFIG_CHANGED, include_args=True)
    def update_config(
        self,
        noise_factors: Optional[List[float]] = None,
        extrapolator: Optional[str] = None,
        **kwargs
    ):
        """Update ZNE configuration with security validation."""
        if noise_factors is not None:
            validated_factors = self._validate_noise_factors(noise_factors)
            self.noise_factors = validated_factors
            self.config.noise_factors = validated_factors
        
        if extrapolator is not None:
            # Validate extrapolator name
            allowed_extrapolators = ['richardson', 'exponential', 'polynomial']
            if extrapolator not in allowed_extrapolators:
                raise ValueError(f"Invalid extrapolator: {extrapolator}")
            
            self.extrapolator = self._create_extrapolator(extrapolator)
        
        # Update other configuration parameters
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    @require_authentication([Permission.VIEW_METRICS])
    @audit_log(AuditEventType.DATA_EXPORTED)
    def get_statistics(self) -> Dict[str, Any]:
        """Get ZNE statistics with security logging."""
        stats = {
            'noise_factors': self.noise_factors,
            'extrapolator_type': type(self.extrapolator).__name__,
            'config': {
                'fit_bootstrap': self.config.fit_bootstrap,
                'confidence_level': self.config.confidence_level,
                'max_iterations': self.config.max_iterations
            },
            'security_enabled': self.enable_security,
            'user_id': self.user_id
        }
        
        return stats


def secure_mitigation_factory(
    mitigation_class: type,
    user_id: Optional[str] = None,
    security_config: Optional[Dict[str, Any]] = None
) -> type:
    """
    Factory function to create security-enhanced versions of mitigation classes.
    
    Args:
        mitigation_class: Base mitigation class to enhance
        user_id: User identifier for authentication
        security_config: Security configuration options
        
    Returns:
        Security-enhanced mitigation class
    """
    security_config = security_config or {}
    
    class SecureMitigationWrapper(mitigation_class):
        def __init__(self, *args, **kwargs):
            self.user_id = user_id
            self.security_config = security_config
            super().__init__(*args, **kwargs)
        
        def mitigate(self, *args, **kwargs):
            # Add security wrapper to mitigate method
            @secure_operation(
                permissions=[Permission.EXECUTE_CIRCUIT],
                rate_limit_operation=f'{mitigation_class.__name__.lower()}_execution',
                audit_event=AuditEventType.CIRCUIT_EXECUTED
            )
            def secure_mitigate(*args, **kwargs):
                return super(SecureMitigationWrapper, self).mitigate(*args, **kwargs)
            
            return secure_mitigate(*args, **kwargs)
    
    SecureMitigationWrapper.__name__ = f"Secure{mitigation_class.__name__}"
    SecureMitigationWrapper.__doc__ = f"Security-enhanced {mitigation_class.__name__}"
    
    return SecureMitigationWrapper


# Example: Create secure versions of all mitigation methods
def create_secure_mitigation_suite(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a suite of security-enhanced mitigation methods.
    
    Args:
        user_id: User identifier for authentication
        
    Returns:
        Dictionary of secure mitigation classes
    """
    from ..mitigation.pec.core import ProbabilisticErrorCancellation
    from ..mitigation.vd.core import VirtualDistillation
    from ..mitigation.cdr.core import CliffordDataRegression
    
    suite = {
        'ZNE': SecureZeroNoiseExtrapolation,
        'PEC': secure_mitigation_factory(ProbabilisticErrorCancellation, user_id),
        'VD': secure_mitigation_factory(VirtualDistillation, user_id),
        'CDR': secure_mitigation_factory(CliffordDataRegression, user_id),
    }
    
    return suite


# Decorator for securing arbitrary mitigation functions
def secure_mitigation_function(
    permissions: Optional[List[Permission]] = None,
    max_qubits: int = 50,
    max_shots: int = 1000000,
    rate_limit_operation: Optional[str] = None
):
    """
    Decorator to add security features to mitigation functions.
    
    Args:
        permissions: Required permissions
        max_qubits: Maximum number of qubits allowed
        max_shots: Maximum number of shots allowed
        rate_limit_operation: Rate limiting operation name
    """
    def decorator(func):
        @secure_operation(
            permissions=permissions or [Permission.EXECUTE_CIRCUIT],
            rate_limit_operation=rate_limit_operation or f'{func.__name__}_execution',
            audit_event=AuditEventType.CIRCUIT_EXECUTED,
            resource_limits={
                ResourceType.QUBITS: max_qubits,
                ResourceType.CPU_TIME: 300
            }
        )
        @circuit_security(
            max_qubits=max_qubits,
            max_shots=max_shots
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    return decorator