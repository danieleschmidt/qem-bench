"""Base classes for noise models."""

import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NoiseChannel:
    """
    Represents a quantum noise channel.
    
    Attributes:
        name: Name of the noise channel
        kraus_operators: List of Kraus operators defining the channel
        probability: Probability of applying this channel
        qubits: Qubits affected by this channel
    """
    name: str
    kraus_operators: List[jnp.ndarray]
    probability: float = 1.0
    qubits: Optional[List[int]] = None
    
    def __post_init__(self):
        """Validate noise channel."""
        if not (0 <= self.probability <= 1):
            raise ValueError("Probability must be between 0 and 1")
        
        # Validate Kraus operators form valid CPTP map
        total_kraus = sum(jnp.conj(K).T @ K for K in self.kraus_operators)
        identity = jnp.eye(total_kraus.shape[0])
        
        if not jnp.allclose(total_kraus, identity, atol=1e-6):
            raise ValueError("Kraus operators do not form valid CPTP map")


class NoiseModel(ABC):
    """Abstract base class for quantum noise models."""
    
    def __init__(self, name: str):
        self.name = name
        self.channels: Dict[str, NoiseChannel] = {}
    
    @abstractmethod
    def get_noise_channels(self, circuit: Any) -> List[NoiseChannel]:
        """
        Get noise channels for a given circuit.
        
        Args:
            circuit: Quantum circuit to add noise to
            
        Returns:
            List of noise channels to apply
        """
        pass
    
    @abstractmethod
    def apply_noise(self, circuit: Any) -> Any:
        """
        Apply noise model to a quantum circuit.
        
        Args:
            circuit: Clean quantum circuit
            
        Returns:
            Noisy quantum circuit
        """
        pass
    
    def add_channel(self, channel: NoiseChannel) -> None:
        """Add a noise channel to the model."""
        self.channels[channel.name] = channel
    
    def remove_channel(self, channel_name: str) -> None:
        """Remove a noise channel from the model."""
        if channel_name in self.channels:
            del self.channels[channel_name]
    
    def get_channel(self, channel_name: str) -> Optional[NoiseChannel]:
        """Get a specific noise channel."""
        return self.channels.get(channel_name)
    
    def list_channels(self) -> List[str]:
        """List all channel names in the model."""
        return list(self.channels.keys())
    
    def scale_noise(self, factor: float) -> "NoiseModel":
        """
        Scale the strength of all noise channels by a factor.
        
        Args:
            factor: Scaling factor for noise strength
            
        Returns:
            New noise model with scaled noise
        """
        scaled_model = self.__class__(f"{self.name}_scaled_{factor}")
        
        for channel_name, channel in self.channels.items():
            # Scale noise by modifying Kraus operators
            scaled_kraus = []
            for K in channel.kraus_operators:
                # Simple scaling: interpolate between identity and noise
                identity = jnp.eye(K.shape[0])
                scaled_K = jnp.sqrt(1 - factor + factor) * identity + jnp.sqrt(factor) * (K - identity)
                scaled_kraus.append(scaled_K)
            
            scaled_channel = NoiseChannel(
                name=f"{channel.name}_scaled",
                kraus_operators=scaled_kraus,
                probability=min(channel.probability * factor, 1.0),
                qubits=channel.qubits
            )
            scaled_model.add_channel(scaled_channel)
        
        return scaled_model
    
    def compose(self, other: "NoiseModel") -> "NoiseModel":
        """
        Compose this noise model with another.
        
        Args:
            other: Other noise model to compose with
            
        Returns:
            Composite noise model
        """
        from .composite import CompositeNoiseModel
        return CompositeNoiseModel([self, other])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert noise model to dictionary representation."""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "channels": {
                name: {
                    "name": channel.name,
                    "probability": channel.probability,
                    "qubits": channel.qubits,
                    "kraus_operators": [K.tolist() for K in channel.kraus_operators]
                }
                for name, channel in self.channels.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NoiseModel":
        """Create noise model from dictionary representation."""
        # This would need specific implementation for each subclass
        raise NotImplementedError("Subclasses must implement from_dict")
    
    def __str__(self) -> str:
        """String representation of noise model."""
        lines = [f"NoiseModel: {self.name}"]
        lines.append(f"Type: {self.__class__.__name__}")
        lines.append(f"Channels: {len(self.channels)}")
        
        for name, channel in self.channels.items():
            lines.append(f"  - {name}: p={channel.probability:.4f}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"{self.__class__.__name__}(name='{self.name}', channels={len(self.channels)})"


# Utility functions for common Kraus operators
def pauli_kraus_operators() -> Dict[str, jnp.ndarray]:
    """Get Kraus operators for Pauli matrices."""
    return {
        "I": jnp.array([[1, 0], [0, 1]], dtype=jnp.complex64),
        "X": jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64),
        "Y": jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64),
        "Z": jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
    }


def depolarizing_kraus_operators(p: float) -> List[jnp.ndarray]:
    """
    Generate Kraus operators for depolarizing channel.
    
    Args:
        p: Depolarizing probability
        
    Returns:
        List of Kraus operators
    """
    paulis = pauli_kraus_operators()
    
    # Depolarizing channel: ρ → (1-p)ρ + p/3(XρX + YρY + ZρZ)
    kraus_ops = []
    
    # Identity component
    kraus_ops.append(jnp.sqrt(1 - p) * paulis["I"])
    
    # Pauli components
    for pauli in ["X", "Y", "Z"]:
        kraus_ops.append(jnp.sqrt(p/3) * paulis[pauli])
    
    return kraus_ops


def amplitude_damping_kraus_operators(gamma: float) -> List[jnp.ndarray]:
    """
    Generate Kraus operators for amplitude damping channel.
    
    Args:
        gamma: Decay probability
        
    Returns:
        List of Kraus operators
    """
    # Amplitude damping: |1⟩ → |0⟩ with probability γ
    K0 = jnp.array([[1, 0], [0, jnp.sqrt(1 - gamma)]], dtype=jnp.complex64)
    K1 = jnp.array([[0, jnp.sqrt(gamma)], [0, 0]], dtype=jnp.complex64)
    
    return [K0, K1]


def phase_damping_kraus_operators(gamma: float) -> List[jnp.ndarray]:
    """
    Generate Kraus operators for phase damping channel.
    
    Args:
        gamma: Dephasing probability
        
    Returns:
        List of Kraus operators
    """
    # Phase damping: loss of coherence without energy loss
    K0 = jnp.array([[1, 0], [0, jnp.sqrt(1 - gamma)]], dtype=jnp.complex64)
    K1 = jnp.array([[0, 0], [0, jnp.sqrt(gamma)]], dtype=jnp.complex64)
    
    return [K0, K1]


def two_qubit_depolarizing_kraus_operators(p: float) -> List[jnp.ndarray]:
    """
    Generate Kraus operators for two-qubit depolarizing channel.
    
    Args:
        p: Depolarizing probability
        
    Returns:
        List of Kraus operators for two-qubit system
    """
    paulis = pauli_kraus_operators()
    pauli_names = ["I", "X", "Y", "Z"]
    
    kraus_ops = []
    
    # Two-qubit Pauli operators
    for i, pauli1 in enumerate(pauli_names):
        for j, pauli2 in enumerate(pauli_names):
            if i == 0 and j == 0:
                # Identity component
                coeff = jnp.sqrt(1 - p)
            else:
                # Pauli components
                coeff = jnp.sqrt(p / 15)  # 15 = 4² - 1
            
            two_qubit_pauli = jnp.kron(paulis[pauli1], paulis[pauli2])
            kraus_ops.append(coeff * two_qubit_pauli)
    
    return kraus_ops