"""
Core validation framework for QEM-Bench.

Provides comprehensive input validation, error checking, and data sanitization
for all QEM-Bench components to ensure robust operation.
"""

import numpy as np
import jax.numpy as jnp
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from functools import wraps


class ValidationError(Exception):
    """Base exception for validation errors."""
    
    def __init__(self, message: str, component: str = "unknown", 
                 suggestions: Optional[List[str]] = None):
        self.component = component
        self.suggestions = suggestions or []
        
        # Format error message with context
        full_message = f"[{component}] {message}"
        if self.suggestions:
            full_message += "\n\nSuggestions:\n"
            for i, suggestion in enumerate(self.suggestions, 1):
                full_message += f"  {i}. {suggestion}\n"
        
        super().__init__(full_message)


class ConfigurationError(ValidationError):
    """Error in system or component configuration."""
    pass


class DataValidationError(ValidationError):
    """Error in data validation."""
    pass


class QuantumValidationError(ValidationError):
    """Error in quantum-specific validation."""
    pass


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    component: str
    data: Optional[Dict[str, Any]] = None
    
    def add_error(self, error: str) -> None:
        """Add an error to the result."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the result."""
        self.warnings.append(warning)
    
    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another validation result."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            component=f"{self.component}+{other.component}",
            data={**(self.data or {}), **(other.data or {})}
        )
    
    def raise_if_invalid(self) -> None:
        """Raise ValidationError if validation failed."""
        if not self.is_valid:
            error_msg = "; ".join(self.errors)
            raise ValidationError(error_msg, self.component)


class Validator(ABC):
    """Abstract base validator class."""
    
    def __init__(self, component: str = "validator"):
        self.component = component
    
    @abstractmethod
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data and return result."""
        pass
    
    def __call__(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Make validator callable."""
        return self.validate(data, context)


class QuantumCircuitValidator(Validator):
    """Validator for quantum circuits."""
    
    def __init__(self):
        super().__init__("quantum_circuit")
    
    def validate(self, circuit: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate quantum circuit."""
        result = ValidationResult(True, [], [], self.component)
        
        try:
            # Check if circuit has required attributes
            if not hasattr(circuit, 'num_qubits'):
                result.add_error("Circuit must have 'num_qubits' attribute")
            elif circuit.num_qubits <= 0:
                result.add_error(f"Number of qubits must be positive, got {circuit.num_qubits}")
            elif circuit.num_qubits > 30:
                result.add_warning(f"Large number of qubits ({circuit.num_qubits}) may cause memory issues")
            
            if not hasattr(circuit, 'gates'):
                result.add_error("Circuit must have 'gates' attribute")
            else:
                # Validate individual gates
                for i, gate in enumerate(circuit.gates):
                    gate_result = self._validate_gate(gate, circuit.num_qubits, i)
                    result = result.merge(gate_result)
            
            # Check circuit depth
            if hasattr(circuit, 'depth'):
                if circuit.depth > 1000:
                    result.add_warning(f"Very deep circuit ({circuit.depth} layers) may be slow to simulate")
            
            # Validate measurements if present
            if hasattr(circuit, 'measurements') and circuit.measurements:
                for meas in circuit.measurements:
                    if 'qubit' not in meas:
                        result.add_error("Measurement must specify 'qubit'")
                    elif not (0 <= meas['qubit'] < circuit.num_qubits):
                        result.add_error(f"Measurement qubit {meas['qubit']} out of range [0, {circuit.num_qubits-1}]")
        
        except Exception as e:
            result.add_error(f"Unexpected error during circuit validation: {str(e)}")
        
        return result
    
    def _validate_gate(self, gate: Dict[str, Any], num_qubits: int, gate_index: int) -> ValidationResult:
        """Validate individual gate."""
        result = ValidationResult(True, [], [], f"{self.component}.gate_{gate_index}")
        
        # Check required fields
        if 'qubits' not in gate:
            result.add_error("Gate must specify 'qubits'")
            return result
        
        if 'type' not in gate:
            result.add_error("Gate must specify 'type'")
        
        if 'matrix' not in gate:
            result.add_error("Gate must have 'matrix'")
        
        # Validate qubit indices
        qubits = gate['qubits']
        if not isinstance(qubits, list):
            result.add_error("Gate qubits must be a list")
            return result
        
        for qubit in qubits:
            if not isinstance(qubit, int):
                result.add_error(f"Qubit index must be integer, got {type(qubit)}")
            elif not (0 <= qubit < num_qubits):
                result.add_error(f"Qubit index {qubit} out of range [0, {num_qubits-1}]")
        
        # Check for duplicate qubits in multi-qubit gates
        if len(qubits) > 1 and len(set(qubits)) != len(qubits):
            result.add_error(f"Duplicate qubits in gate: {qubits}")
        
        # Validate matrix dimensions
        if 'matrix' in gate:
            matrix = gate['matrix']
            expected_size = 2 ** len(qubits)
            
            if hasattr(matrix, 'shape'):
                if matrix.shape != (expected_size, expected_size):
                    result.add_error(f"Gate matrix shape {matrix.shape} doesn't match {len(qubits)} qubits")
            
            # Check if matrix is unitary (approximately)
            if hasattr(matrix, 'shape') and matrix.shape[0] <= 8:  # Only check small matrices
                try:
                    identity = jnp.eye(matrix.shape[0])
                    product = matrix @ jnp.conj(matrix.T)
                    if not jnp.allclose(product, identity, atol=1e-10):
                        result.add_warning(f"Gate matrix may not be unitary")
                except:
                    pass  # Skip unitarity check if computation fails
        
        return result


class ObservableValidator(Validator):
    """Validator for quantum observables."""
    
    def __init__(self):
        super().__init__("observable")
    
    def validate(self, observable: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate quantum observable."""
        result = ValidationResult(True, [], [], self.component)
        
        try:
            # Check if observable has required methods
            if not hasattr(observable, 'matrix'):
                result.add_error("Observable must have 'matrix' property")
            
            if not hasattr(observable, 'expectation_value'):
                result.add_error("Observable must have 'expectation_value' method")
            
            # Validate matrix if accessible
            if hasattr(observable, 'matrix'):
                matrix = observable.matrix
                
                if hasattr(matrix, 'shape'):
                    # Check if matrix is square
                    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
                        result.add_error(f"Observable matrix must be square, got shape {matrix.shape}")
                    
                    # Check if matrix is Hermitian (for small matrices)
                    if matrix.shape[0] <= 16:  # Only check small matrices
                        try:
                            if not jnp.allclose(matrix, jnp.conj(matrix.T), atol=1e-10):
                                result.add_error("Observable matrix must be Hermitian")
                        except:
                            result.add_warning("Could not verify observable Hermiticity")
            
            # Check Pauli string if present
            if hasattr(observable, 'pauli_string'):
                pauli_result = self._validate_pauli_string(observable.pauli_string, observable.qubits)
                result = result.merge(pauli_result)
        
        except Exception as e:
            result.add_error(f"Unexpected error during observable validation: {str(e)}")
        
        return result
    
    def _validate_pauli_string(self, pauli_string: str, qubits: List[int]) -> ValidationResult:
        """Validate Pauli string."""
        result = ValidationResult(True, [], [], f"{self.component}.pauli")
        
        valid_paulis = set("IXYZ")
        
        if not all(p in valid_paulis for p in pauli_string.upper()):
            result.add_error(f"Invalid Pauli operators in string: {pauli_string}")
        
        if len(pauli_string) != len(qubits):
            result.add_error(f"Pauli string length ({len(pauli_string)}) doesn't match qubits ({len(qubits)})")
        
        return result


class ZNEConfigValidator(Validator):
    """Validator for ZNE configuration."""
    
    def __init__(self):
        super().__init__("zne_config")
    
    def validate(self, config: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate ZNE configuration."""
        result = ValidationResult(True, [], [], self.component)
        
        try:
            # Validate noise factors
            if hasattr(config, 'noise_factors'):
                noise_factors = config.noise_factors
                
                if not isinstance(noise_factors, (list, tuple, np.ndarray)):
                    result.add_error("Noise factors must be a list or array")
                elif len(noise_factors) < 2:
                    result.add_error("Need at least 2 noise factors for extrapolation")
                elif not all(f >= 1.0 for f in noise_factors):
                    result.add_error("All noise factors must be ≥ 1.0")
                elif len(set(noise_factors)) != len(noise_factors):
                    result.add_warning("Duplicate noise factors detected")
                elif max(noise_factors) > 10:
                    result.add_warning(f"Very large noise factor ({max(noise_factors)}) may introduce artifacts")
                
                # Check spacing
                if len(noise_factors) >= 3:
                    sorted_factors = sorted(noise_factors)
                    spacings = [sorted_factors[i+1] - sorted_factors[i] for i in range(len(sorted_factors)-1)]
                    if max(spacings) / min(spacings) > 10:
                        result.add_warning("Irregular spacing in noise factors may affect extrapolation quality")
            
            # Validate extrapolation method
            if hasattr(config, 'extrapolator'):
                valid_methods = {"richardson", "exponential", "polynomial", "adaptive"}
                if config.extrapolator not in valid_methods:
                    result.add_error(f"Unknown extrapolator '{config.extrapolator}'. Valid: {valid_methods}")
            
            # Validate confidence level
            if hasattr(config, 'confidence_level'):
                cl = config.confidence_level
                if not (0 < cl < 1):
                    result.add_error(f"Confidence level must be between 0 and 1, got {cl}")
                elif cl < 0.5:
                    result.add_warning(f"Very low confidence level ({cl}) may not be meaningful")
            
            # Validate bootstrap settings
            if hasattr(config, 'fit_bootstrap') and config.fit_bootstrap:
                if hasattr(config, 'num_bootstrap_samples'):
                    n_boot = config.num_bootstrap_samples
                    if n_boot < 100:
                        result.add_warning(f"Low bootstrap samples ({n_boot}) may give unreliable confidence intervals")
                    elif n_boot > 10000:
                        result.add_warning(f"Many bootstrap samples ({n_boot}) may be slow")
        
        except Exception as e:
            result.add_error(f"Unexpected error during ZNE config validation: {str(e)}")
        
        return result


class NumericValidator(Validator):
    """Validator for numeric data."""
    
    def __init__(self, component: str = "numeric"):
        super().__init__(component)
    
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate numeric data."""
        result = ValidationResult(True, [], [], self.component)
        context = context or {}
        
        try:
            # Check for NaN and infinity
            if hasattr(data, '__iter__') and not isinstance(data, str):
                # Array-like data
                if hasattr(data, 'shape'):  # JAX/numpy array
                    if jnp.any(jnp.isnan(data)):
                        result.add_error("Data contains NaN values")
                    if jnp.any(jnp.isinf(data)):
                        result.add_error("Data contains infinite values")
                else:
                    # List/tuple
                    for i, val in enumerate(data):
                        if np.isnan(val):
                            result.add_error(f"Data contains NaN at index {i}")
                        if np.isinf(val):
                            result.add_error(f"Data contains infinite value at index {i}")
            else:
                # Scalar data
                if np.isnan(data):
                    result.add_error("Data is NaN")
                if np.isinf(data):
                    result.add_error("Data is infinite")
            
            # Check ranges if specified
            if 'min_value' in context:
                min_val = context['min_value']
                if hasattr(data, '__iter__') and not isinstance(data, str):
                    if jnp.any(data < min_val):
                        result.add_error(f"Data contains values below minimum {min_val}")
                else:
                    if data < min_val:
                        result.add_error(f"Data value {data} below minimum {min_val}")
            
            if 'max_value' in context:
                max_val = context['max_value']
                if hasattr(data, '__iter__') and not isinstance(data, str):
                    if jnp.any(data > max_val):
                        result.add_error(f"Data contains values above maximum {max_val}")
                else:
                    if data > max_val:
                        result.add_error(f"Data value {data} above maximum {max_val}")
        
        except Exception as e:
            result.add_error(f"Unexpected error during numeric validation: {str(e)}")
        
        return result


class ValidationRegistry:
    """Registry for validators."""
    
    def __init__(self):
        self._validators: Dict[str, Validator] = {}
        self._default_validators()
    
    def _default_validators(self):
        """Register default validators."""
        self.register("quantum_circuit", QuantumCircuitValidator())
        self.register("observable", ObservableValidator())
        self.register("zne_config", ZNEConfigValidator())
        self.register("numeric", NumericValidator())
    
    def register(self, name: str, validator: Validator):
        """Register a validator."""
        self._validators[name] = validator
    
    def get(self, name: str) -> Optional[Validator]:
        """Get validator by name."""
        return self._validators.get(name)
    
    def validate(self, name: str, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data using named validator."""
        validator = self.get(name)
        if validator is None:
            raise ValueError(f"Unknown validator: {name}")
        return validator.validate(data, context)
    
    def list_validators(self) -> List[str]:
        """List available validators."""
        return list(self._validators.keys())


# Global validator registry
_registry = ValidationRegistry()


def validate(name: str, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """Validate data using global registry."""
    return _registry.validate(name, data, context)


def register_validator(name: str, validator: Validator):
    """Register validator in global registry."""
    _registry.register(name, validator)


def validate_and_raise(name: str, data: Any, context: Optional[Dict[str, Any]] = None):
    """Validate data and raise exception if invalid."""
    result = validate(name, data, context)
    result.raise_if_invalid()


def validated(validator_name: str, context: Optional[Dict[str, Any]] = None):
    """Decorator to validate function arguments."""
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate first argument by default
            if args:
                validate_and_raise(validator_name, args[0], context)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Convenience validators for common use cases
def validate_positive_integer(value: int, name: str = "value") -> None:
    """Validate positive integer."""
    if not isinstance(value, int):
        raise ValidationError(f"{name} must be integer, got {type(value)}", "validation")
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}", "validation")


def validate_probability(value: float, name: str = "probability") -> None:
    """Validate probability value."""
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(value)}", "validation")
    if not (0 <= value <= 1):
        raise ValidationError(f"{name} must be between 0 and 1, got {value}", "validation")


def validate_array_shape(array: Any, expected_shape: Tuple[int, ...], name: str = "array") -> None:
    """Validate array shape."""
    if not hasattr(array, 'shape'):
        raise ValidationError(f"{name} must have shape attribute", "validation")
    if array.shape != expected_shape:
        raise ValidationError(f"{name} shape {array.shape} doesn't match expected {expected_shape}", "validation")


def validate_density_matrix(rho: Any, tolerance: float = 1e-10) -> None:
    """Validate density matrix properties."""
    # Check Hermiticity
    if not jnp.allclose(rho, jnp.conj(rho.T), atol=tolerance):
        raise QuantumValidationError("Density matrix must be Hermitian", "quantum")
    
    # Check positive semidefinite
    eigenvals = jnp.real(jnp.linalg.eigvals(rho))
    if jnp.any(eigenvals < -tolerance):
        raise QuantumValidationError("Density matrix must be positive semidefinite", "quantum")
    
    # Check trace = 1
    trace = jnp.real(jnp.trace(rho))
    if not jnp.allclose(trace, 1.0, atol=tolerance):
        raise QuantumValidationError(f"Density matrix trace must be 1, got {trace}", "quantum")


def validate_unitary_matrix(U: Any, tolerance: float = 1e-10) -> None:
    """Validate unitary matrix."""
    if not hasattr(U, 'shape') or len(U.shape) != 2 or U.shape[0] != U.shape[1]:
        raise ValidationError("Unitary matrix must be square", "quantum")
    
    # Check U† U = I
    identity = jnp.eye(U.shape[0])
    product = jnp.conj(U.T) @ U
    
    if not jnp.allclose(product, identity, atol=tolerance):
        raise QuantumValidationError("Matrix is not unitary", "quantum",
            ["Check matrix calculation", "Ensure proper normalization", "Verify gate decomposition"])