"""Core Zero-Noise Extrapolation implementation."""

import numpy as np
from typing import List, Union, Optional, Dict, Any, Callable
from dataclasses import dataclass
import warnings

from .scaling import NoiseScaler, UnitaryFoldingScaler
from .extrapolation import Extrapolator, RichardsonExtrapolator
from .result import ZNEResult


@dataclass
class ZNEConfig:
    """Configuration for Zero-Noise Extrapolation."""
    noise_factors: List[float]
    extrapolator: str = "richardson"
    fit_bootstrap: bool = False
    confidence_level: float = 0.95
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6


class ZeroNoiseExtrapolation:
    """
    Zero-Noise Extrapolation for quantum error mitigation.
    
    ZNE works by artificially scaling the noise in a quantum circuit,
    measuring the expectation value at different noise levels, and
    extrapolating to the zero-noise limit.
    
    Args:
        noise_scaler: Method for scaling noise in quantum circuits
        noise_factors: Factors by which to scale the noise (≥1)
        extrapolator: Method for extrapolating to zero noise
        config: Additional configuration parameters
    
    Example:
        >>> zne = ZeroNoiseExtrapolation(
        ...     noise_factors=[1, 1.5, 2, 2.5, 3],
        ...     extrapolator="richardson"
        ... )
        >>> result = zne.mitigate(circuit, backend, observable)
        >>> print(f"Mitigated value: {result.mitigated_value:.4f}")
    """
    
    def __init__(
        self,
        noise_scaler: Optional[NoiseScaler] = None,
        noise_factors: Optional[List[float]] = None,
        extrapolator: Union[str, Extrapolator] = "richardson",
        config: Optional[ZNEConfig] = None,
        **kwargs
    ):
        # Set default noise scaler
        self.noise_scaler = noise_scaler or UnitaryFoldingScaler()
        
        # Set default noise factors
        if noise_factors is None:
            noise_factors = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        # Validate noise factors
        if not all(factor >= 1.0 for factor in noise_factors):
            raise ValueError("All noise factors must be ≥ 1.0")
        if len(noise_factors) < 2:
            raise ValueError("At least 2 noise factors required for extrapolation")
        
        self.noise_factors = sorted(noise_factors)
        
        # Set up extrapolator
        if isinstance(extrapolator, str):
            self.extrapolator = self._create_extrapolator(extrapolator)
        else:
            self.extrapolator = extrapolator
            
        # Configuration
        self.config = config or ZNEConfig(noise_factors=self.noise_factors)
        
        # Additional kwargs for backward compatibility
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _create_extrapolator(self, method: str) -> Extrapolator:
        """Create extrapolator from string specification."""
        from .extrapolation import (
            RichardsonExtrapolator,
            ExponentialExtrapolator, 
            PolynomialExtrapolator
        )
        
        extrapolators = {
            "richardson": RichardsonExtrapolator,
            "exponential": ExponentialExtrapolator,
            "polynomial": PolynomialExtrapolator
        }
        
        if method not in extrapolators:
            available = ", ".join(extrapolators.keys())
            raise ValueError(f"Unknown extrapolator '{method}'. Available: {available}")
        
        return extrapolators[method]()
    
    def mitigate(
        self,
        circuit: Any,
        backend: Any,
        observable: Optional[Any] = None,
        shots: int = 1024,
        **execution_kwargs
    ) -> ZNEResult:
        """
        Apply zero-noise extrapolation to mitigate errors.
        
        Args:
            circuit: Quantum circuit to execute
            backend: Quantum backend for execution
            observable: Observable to measure (if None, use all-Z)
            shots: Number of measurement shots per noise factor
            **execution_kwargs: Additional arguments for circuit execution
            
        Returns:
            ZNEResult containing raw and mitigated expectation values
        """
        # Execute circuits at different noise levels
        noise_values, expectation_values = self._execute_noise_scaled_circuits(
            circuit, backend, observable, shots, **execution_kwargs
        )
        
        # Perform extrapolation to zero noise
        mitigated_value, extrapolation_data = self._extrapolate_to_zero_noise(
            noise_values, expectation_values
        )
        
        # Calculate error metrics
        raw_value = expectation_values[0]  # Value at noise factor 1.0
        error_reduction = self._calculate_error_reduction(
            raw_value, mitigated_value, extrapolation_data.get("ideal_value")
        )
        
        return ZNEResult(
            raw_value=raw_value,
            mitigated_value=mitigated_value,
            noise_factors=noise_values,
            expectation_values=expectation_values,
            extrapolation_data=extrapolation_data,
            error_reduction=error_reduction,
            config=self.config
        )
    
    def _execute_noise_scaled_circuits(
        self,
        circuit: Any,
        backend: Any, 
        observable: Optional[Any],
        shots: int,
        **execution_kwargs
    ) -> tuple[List[float], List[float]]:
        """Execute circuits with scaled noise levels."""
        expectation_values = []
        
        for noise_factor in self.noise_factors:
            # Scale noise in the circuit
            scaled_circuit = self.noise_scaler.scale_noise(circuit, noise_factor)
            
            # Execute the scaled circuit
            if hasattr(backend, 'run_with_observable'):
                # Backend supports observable measurement
                result = backend.run_with_observable(
                    scaled_circuit, observable, shots=shots, **execution_kwargs
                )
                expectation_values.append(result.expectation_value)
            else:
                # Fallback to standard execution
                result = backend.run(scaled_circuit, shots=shots, **execution_kwargs)
                expectation_value = self._extract_expectation_value(
                    result, observable
                )
                expectation_values.append(expectation_value)
        
        return self.noise_factors, expectation_values
    
    def _extract_expectation_value(
        self, 
        result: Any, 
        observable: Optional[Any]
    ) -> float:
        """Extract expectation value from measurement result."""
        if observable is None:
            # Default to all-Z measurement (computational basis)
            if hasattr(result, 'get_counts'):
                counts = result.get_counts()
                total_shots = sum(counts.values())
                
                # Calculate <Z⊗Z⊗...⊗Z> expectation value
                expectation = 0.0
                for bitstring, count in counts.items():
                    # Count number of |1⟩ states
                    num_ones = bitstring.count('1')
                    parity = (-1) ** num_ones
                    expectation += parity * count / total_shots
                
                return expectation
            else:
                raise ValueError("Cannot extract expectation value from result")
        else:
            # Use provided observable
            return observable.expectation_value(result)
    
    def _extrapolate_to_zero_noise(
        self,
        noise_values: List[float],
        expectation_values: List[float]
    ) -> tuple[float, Dict[str, Any]]:
        """Extrapolate expectation values to zero noise."""
        # Convert to numpy arrays
        x = np.array(noise_values)
        y = np.array(expectation_values)
        
        # Perform extrapolation
        mitigated_value, fit_data = self.extrapolator.extrapolate(x, y)
        
        # Add confidence intervals if bootstrap enabled
        if self.config.fit_bootstrap:
            confidence_interval = self._compute_bootstrap_confidence(
                x, y, self.config.confidence_level
            )
            fit_data["confidence_interval"] = confidence_interval
        
        return mitigated_value, fit_data
    
    def _compute_bootstrap_confidence(
        self,
        x: np.ndarray,
        y: np.ndarray, 
        confidence_level: float,
        n_bootstrap: int = 1000
    ) -> Dict[str, float]:
        """Compute bootstrap confidence intervals."""
        bootstrap_estimates = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(x), size=len(x), replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            
            # Fit and extrapolate
            try:
                estimate, _ = self.extrapolator.extrapolate(x_boot, y_boot)
                bootstrap_estimates.append(estimate)
            except:
                # Skip failed fits
                continue
        
        if not bootstrap_estimates:
            warnings.warn("Bootstrap confidence interval computation failed")
            return {"lower": np.nan, "upper": np.nan}
        
        # Compute confidence interval
        alpha = 1 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        return {
            "lower": np.percentile(bootstrap_estimates, lower_percentile),
            "upper": np.percentile(bootstrap_estimates, upper_percentile)
        }
    
    def _calculate_error_reduction(
        self,
        raw_value: float,
        mitigated_value: float,
        ideal_value: Optional[float] = None
    ) -> Optional[float]:
        """Calculate error reduction percentage."""
        if ideal_value is None:
            return None
        
        raw_error = abs(raw_value - ideal_value)
        mitigated_error = abs(mitigated_value - ideal_value)
        
        if raw_error == 0:
            return 1.0 if mitigated_error == 0 else 0.0
        
        return (raw_error - mitigated_error) / raw_error
    
    def execute(
        self,
        circuit: Any,
        backend: Any,
        shots_per_factor: int = 1024,
        **kwargs
    ) -> ZNEResult:
        """
        Legacy method name for backward compatibility.
        
        Use `mitigate()` instead.
        """
        warnings.warn(
            "execute() is deprecated, use mitigate() instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.mitigate(circuit, backend, shots=shots_per_factor, **kwargs)


# Convenience function for quick ZNE
def zero_noise_extrapolation(
    circuit: Any,
    backend: Any,
    noise_factors: Optional[List[float]] = None,
    extrapolator: str = "richardson",
    shots: int = 1024,
    **kwargs
) -> ZNEResult:
    """
    Convenience function for quick zero-noise extrapolation.
    
    Args:
        circuit: Quantum circuit to execute
        backend: Quantum backend
        noise_factors: Noise scaling factors
        extrapolator: Extrapolation method
        shots: Number of shots per noise factor
        **kwargs: Additional arguments
        
    Returns:
        ZNEResult with mitigation results
    """
    zne = ZeroNoiseExtrapolation(
        noise_factors=noise_factors,
        extrapolator=extrapolator,
        **kwargs
    )
    return zne.mitigate(circuit, backend, shots=shots)