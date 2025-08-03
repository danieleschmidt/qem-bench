"""Noise scaling methods for Zero-Noise Extrapolation."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import copy


class NoiseScaler(ABC):
    """Abstract base class for noise scaling methods."""
    
    @abstractmethod
    def scale_noise(self, circuit: Any, noise_factor: float) -> Any:
        """
        Scale the noise in a quantum circuit by a given factor.
        
        Args:
            circuit: Quantum circuit to modify
            noise_factor: Factor by which to scale noise (≥1)
            
        Returns:
            Modified circuit with scaled noise
        """
        pass


class UnitaryFoldingScaler(NoiseScaler):
    """
    Unitary folding noise scaling for ZNE.
    
    This method scales noise by inserting additional gate pairs
    (U, U†) that ideally cancel but amplify noise in practice.
    
    Example:
        For noise factor λ = 2.5:
        - Apply unitary folding 2 times (adding 2 to noise factor)
        - Add partial folding with probability 0.5 for remaining 0.5
    """
    
    def __init__(self, folding_method: str = "global"):
        """
        Initialize unitary folding scaler.
        
        Args:
            folding_method: Method for folding ("global", "local", "random")
        """
        self.folding_method = folding_method
        
        if folding_method not in ["global", "local", "random"]:
            raise ValueError(
                f"Unknown folding method: {folding_method}. "
                "Choose from: global, local, random"
            )
    
    def scale_noise(self, circuit: Any, noise_factor: float) -> Any:
        """Scale noise using unitary folding."""
        if noise_factor < 1.0:
            raise ValueError("Noise factor must be ≥ 1.0")
        
        if noise_factor == 1.0:
            return copy.deepcopy(circuit)
        
        # Calculate number of complete folds and remainder
        num_folds = int(noise_factor)
        remainder = noise_factor - num_folds
        
        # Create folded circuit
        folded_circuit = self._apply_folding(circuit, num_folds, remainder)
        
        return folded_circuit
    
    def _apply_folding(self, circuit: Any, num_folds: int, remainder: float) -> Any:
        """Apply unitary folding to the circuit."""
        # This is a simplified implementation
        # In practice, would need specific circuit library support
        
        if hasattr(circuit, 'copy'):
            folded_circuit = circuit.copy()
        else:
            folded_circuit = copy.deepcopy(circuit)
        
        # Apply complete folds
        for _ in range(num_folds - 1):  # -1 because original circuit counts as 1
            if self.folding_method == "global":
                folded_circuit = self._global_fold(folded_circuit)
            elif self.folding_method == "local":
                folded_circuit = self._local_fold(folded_circuit)
            elif self.folding_method == "random":
                folded_circuit = self._random_fold(folded_circuit)
        
        # Apply partial fold with probability = remainder
        if remainder > 0 and np.random.random() < remainder:
            if self.folding_method == "global":
                folded_circuit = self._global_fold(folded_circuit)
            elif self.folding_method == "local":
                folded_circuit = self._local_fold(folded_circuit)
            elif self.folding_method == "random":
                folded_circuit = self._random_fold(folded_circuit)
        
        return folded_circuit
    
    def _global_fold(self, circuit: Any) -> Any:
        """Apply global unitary folding (U → U · U† · U)."""
        # Simplified implementation - would need circuit library specific code
        # For now, we'll add a marker that this circuit has been folded
        if hasattr(circuit, '_zne_noise_factor'):
            circuit._zne_noise_factor *= 2
        else:
            circuit._zne_noise_factor = 2.0
        return circuit
    
    def _local_fold(self, circuit: Any) -> Any:
        """Apply local unitary folding to random gates."""
        # Simplified implementation
        if hasattr(circuit, '_zne_noise_factor'):
            circuit._zne_noise_factor *= 1.5
        else:
            circuit._zne_noise_factor = 1.5
        return circuit
    
    def _random_fold(self, circuit: Any) -> Any:
        """Apply random unitary folding."""
        # Simplified implementation
        if hasattr(circuit, '_zne_noise_factor'):
            circuit._zne_noise_factor *= 1.8
        else:
            circuit._zne_noise_factor = 1.8
        return circuit


class PulseStretchScaler(NoiseScaler):
    """
    Pulse stretching noise scaling for ZNE.
    
    This method scales noise by stretching gate durations,
    which increases decoherence and other time-dependent errors.
    
    Note: Requires backend support for pulse-level control.
    """
    
    def __init__(self, backend: Any, pulse_stretch_factor: float = 1.0):
        """
        Initialize pulse stretching scaler.
        
        Args:
            backend: Quantum backend with pulse support
            pulse_stretch_factor: Base stretching factor
        """
        self.backend = backend
        self.pulse_stretch_factor = pulse_stretch_factor
        
        # Check if backend supports pulse control
        if not hasattr(backend, 'pulse_support') or not backend.pulse_support:
            raise ValueError("Backend does not support pulse-level control")
    
    def scale_noise(self, circuit: Any, noise_factor: float) -> Any:
        """Scale noise by stretching pulse durations."""
        if noise_factor < 1.0:
            raise ValueError("Noise factor must be ≥ 1.0")
        
        if noise_factor == 1.0:
            return copy.deepcopy(circuit)
        
        # Convert circuit to pulse schedule
        pulse_schedule = self.backend.circuit_to_pulse(circuit)
        
        # Stretch all pulse durations
        stretched_schedule = self._stretch_pulses(pulse_schedule, noise_factor)
        
        # Convert back to circuit representation
        stretched_circuit = self.backend.pulse_to_circuit(stretched_schedule)
        
        return stretched_circuit
    
    def _stretch_pulses(self, pulse_schedule: Any, factor: float) -> Any:
        """Stretch pulse durations by given factor."""
        # Simplified implementation - would need backend-specific code
        if hasattr(pulse_schedule, 'duration'):
            pulse_schedule.duration *= factor
        
        if hasattr(pulse_schedule, '_zne_pulse_stretch'):
            pulse_schedule._zne_pulse_stretch *= factor
        else:
            pulse_schedule._zne_pulse_stretch = factor
        
        return pulse_schedule


class GlobalDepolarizingScaler(NoiseScaler):
    """
    Global depolarizing noise scaling for ZNE.
    
    This method scales noise by adding depolarizing channels
    after each gate operation.
    """
    
    def __init__(self, base_depolarizing_rate: float = 0.001):
        """
        Initialize global depolarizing scaler.
        
        Args:
            base_depolarizing_rate: Base depolarizing error rate
        """
        self.base_depolarizing_rate = base_depolarizing_rate
    
    def scale_noise(self, circuit: Any, noise_factor: float) -> Any:
        """Scale noise by adding depolarizing channels."""
        if noise_factor < 1.0:
            raise ValueError("Noise factor must be ≥ 1.0")
        
        if noise_factor == 1.0:
            return copy.deepcopy(circuit)
        
        # Calculate scaled depolarizing rate
        scaled_rate = self.base_depolarizing_rate * (noise_factor - 1.0)
        
        # Create noisy circuit
        noisy_circuit = self._add_depolarizing_noise(circuit, scaled_rate)
        
        return noisy_circuit
    
    def _add_depolarizing_noise(self, circuit: Any, error_rate: float) -> Any:
        """Add depolarizing noise to circuit."""
        # Simplified implementation - would need circuit library specific code
        if hasattr(circuit, 'copy'):
            noisy_circuit = circuit.copy()
        else:
            noisy_circuit = copy.deepcopy(circuit)
        
        # Add noise marker
        if hasattr(noisy_circuit, '_zne_depolarizing_rate'):
            noisy_circuit._zne_depolarizing_rate += error_rate
        else:
            noisy_circuit._zne_depolarizing_rate = error_rate
        
        return noisy_circuit


class ParametricNoiseScaler(NoiseScaler):
    """
    Parametric noise scaling using gate parameter modification.
    
    This method scales noise by modifying gate parameters
    to introduce controlled errors.
    """
    
    def __init__(self, parameter_noise_std: float = 0.01):
        """
        Initialize parametric noise scaler.
        
        Args:
            parameter_noise_std: Standard deviation of parameter noise
        """
        self.parameter_noise_std = parameter_noise_std
    
    def scale_noise(self, circuit: Any, noise_factor: float) -> Any:
        """Scale noise by modifying gate parameters."""
        if noise_factor < 1.0:
            raise ValueError("Noise factor must be ≥ 1.0")
        
        if noise_factor == 1.0:
            return copy.deepcopy(circuit)
        
        # Calculate scaled parameter noise
        scaled_noise_std = self.parameter_noise_std * (noise_factor - 1.0)
        
        # Create noisy circuit
        noisy_circuit = self._add_parameter_noise(circuit, scaled_noise_std)
        
        return noisy_circuit
    
    def _add_parameter_noise(self, circuit: Any, noise_std: float) -> Any:
        """Add parameter noise to circuit gates."""
        # Simplified implementation
        if hasattr(circuit, 'copy'):
            noisy_circuit = circuit.copy()
        else:
            noisy_circuit = copy.deepcopy(circuit)
        
        # Add noise marker
        if hasattr(noisy_circuit, '_zne_parameter_noise'):
            noisy_circuit._zne_parameter_noise += noise_std
        else:
            noisy_circuit._zne_parameter_noise = noise_std
        
        return noisy_circuit


# Utility function to create scaler from string
def create_noise_scaler(scaler_type: str, **kwargs) -> NoiseScaler:
    """
    Create a noise scaler from string specification.
    
    Args:
        scaler_type: Type of scaler ("folding", "pulse", "depolarizing", "parametric")
        **kwargs: Additional arguments for scaler initialization
        
    Returns:
        Configured noise scaler
    """
    scalers = {
        "folding": UnitaryFoldingScaler,
        "pulse": PulseStretchScaler,
        "depolarizing": GlobalDepolarizingScaler,
        "parametric": ParametricNoiseScaler
    }
    
    if scaler_type not in scalers:
        available = ", ".join(scalers.keys())
        raise ValueError(f"Unknown scaler type '{scaler_type}'. Available: {available}")
    
    return scalers[scaler_type](**kwargs)