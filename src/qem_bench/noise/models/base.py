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


def generalized_amplitude_damping_kraus_operators(
    gamma: float, 
    temperature: float = 0.0
) -> List[jnp.ndarray]:
    """
    Generate Kraus operators for generalized amplitude damping channel.
    
    Includes thermal effects at finite temperature.
    
    Args:
        gamma: Decay probability
        temperature: Thermal excitation probability
        
    Returns:
        List of Kraus operators
    """
    p = temperature / (1 + temperature) if temperature > 0 else 0.0
    
    # Four Kraus operators for generalized amplitude damping
    K0 = jnp.array([
        [1, 0],
        [0, jnp.sqrt(1 - gamma)]
    ], dtype=jnp.complex64)
    
    K1 = jnp.array([
        [0, jnp.sqrt(gamma * (1 - p))],
        [0, 0]
    ], dtype=jnp.complex64)
    
    K2 = jnp.array([
        [jnp.sqrt(gamma * p), 0],
        [0, 0]
    ], dtype=jnp.complex64)
    
    K3 = jnp.array([
        [0, 0],
        [jnp.sqrt(gamma * p), jnp.sqrt(1 - gamma)]
    ], dtype=jnp.complex64)
    
    return [K0, K1, K2, K3]


def bit_flip_kraus_operators(p: float) -> List[jnp.ndarray]:
    """
    Generate Kraus operators for bit flip channel.
    
    Args:
        p: Bit flip probability
        
    Returns:
        List of Kraus operators
    """
    paulis = pauli_kraus_operators()
    
    return [
        jnp.sqrt(1 - p) * paulis["I"],
        jnp.sqrt(p) * paulis["X"]
    ]


def phase_flip_kraus_operators(p: float) -> List[jnp.ndarray]:
    """
    Generate Kraus operators for phase flip channel.
    
    Args:
        p: Phase flip probability
        
    Returns:
        List of Kraus operators
    """
    paulis = pauli_kraus_operators()
    
    return [
        jnp.sqrt(1 - p) * paulis["I"],
        jnp.sqrt(p) * paulis["Z"]
    ]


def bit_phase_flip_kraus_operators(p: float) -> List[jnp.ndarray]:
    """
    Generate Kraus operators for bit-phase flip channel.
    
    Args:
        p: Bit-phase flip probability
        
    Returns:
        List of Kraus operators
    """
    paulis = pauli_kraus_operators()
    
    return [
        jnp.sqrt(1 - p) * paulis["I"],
        jnp.sqrt(p) * paulis["Y"]
    ]


def pauli_channel_kraus_operators(px: float, py: float, pz: float) -> List[jnp.ndarray]:
    """
    Generate Kraus operators for general Pauli channel.
    
    Args:
        px: X error probability
        py: Y error probability  
        pz: Z error probability
        
    Returns:
        List of Kraus operators
    """
    pi = 1 - px - py - pz
    if pi < 0:
        raise ValueError("Probabilities must sum to <= 1")
    
    paulis = pauli_kraus_operators()
    
    return [
        jnp.sqrt(pi) * paulis["I"],
        jnp.sqrt(px) * paulis["X"],
        jnp.sqrt(py) * paulis["Y"],
        jnp.sqrt(pz) * paulis["Z"]
    ]


def thermal_relaxation_kraus_operators(
    t1: float,
    t2: float, 
    gate_time: float,
    excited_state_population: float = 0.0
) -> List[jnp.ndarray]:
    """
    Generate Kraus operators for thermal relaxation channel.
    
    Models both T1 and T2 processes with optional thermal population.
    
    Args:
        t1: T1 relaxation time
        t2: T2 dephasing time
        gate_time: Duration of gate operation
        excited_state_population: Thermal population of excited state
        
    Returns:
        List of Kraus operators
    """
    # Calculate decay probabilities
    gamma1 = 1 - jnp.exp(-gate_time / t1) if t1 > 0 else 0.0
    gamma_phi = 1 - jnp.exp(-gate_time / t2) if t2 > 0 else 0.0
    
    # Effective dephasing rate
    gamma2 = gamma_phi - gamma1 / 2
    gamma2 = jnp.maximum(0.0, gamma2)
    
    # Thermal population
    p_th = excited_state_population
    
    # Kraus operators for thermal relaxation
    K0 = jnp.array([
        [1, 0],
        [0, jnp.sqrt(1 - gamma1)]
    ], dtype=jnp.complex64)
    
    K1 = jnp.array([
        [0, jnp.sqrt(gamma1 * (1 - p_th))],
        [0, 0]
    ], dtype=jnp.complex64)
    
    K2 = jnp.array([
        [jnp.sqrt(gamma1 * p_th), 0],
        [0, 0]
    ], dtype=jnp.complex64)
    
    K3 = jnp.array([
        [0, 0],
        [0, jnp.sqrt(gamma2)]
    ], dtype=jnp.complex64)
    
    return [K0, K1, K2, K3]


def reset_kraus_operators(p: float) -> List[jnp.ndarray]:
    """
    Generate Kraus operators for reset channel.
    
    Resets qubit to |0⟩ state with probability p.
    
    Args:
        p: Reset probability
        
    Returns:
        List of Kraus operators
    """
    # No reset
    K0 = jnp.sqrt(1 - p) * jnp.eye(2, dtype=jnp.complex64)
    
    # Reset to |0⟩
    K1 = jnp.sqrt(p) * jnp.array([
        [1, 0],
        [1, 0]
    ], dtype=jnp.complex64)
    
    return [K0, K1]


# Advanced noise channel utilities
class NoiseChannelFactory:
    """Factory class for creating common noise channels."""
    
    @staticmethod
    def depolarizing(p: float, name: str = "depolarizing") -> NoiseChannel:
        """Create depolarizing noise channel."""
        kraus_ops = depolarizing_kraus_operators(p)
        return NoiseChannel(name=name, kraus_operators=kraus_ops)
    
    @staticmethod
    def amplitude_damping(gamma: float, name: str = "amplitude_damping") -> NoiseChannel:
        """Create amplitude damping noise channel."""
        kraus_ops = amplitude_damping_kraus_operators(gamma)
        return NoiseChannel(name=name, kraus_operators=kraus_ops)
    
    @staticmethod
    def phase_damping(gamma: float, name: str = "phase_damping") -> NoiseChannel:
        """Create phase damping noise channel."""
        kraus_ops = phase_damping_kraus_operators(gamma)
        return NoiseChannel(name=name, kraus_operators=kraus_ops)
    
    @staticmethod
    def thermal_relaxation(
        t1: float, 
        t2: float, 
        gate_time: float, 
        temperature: float = 0.0,
        name: str = "thermal_relaxation"
    ) -> NoiseChannel:
        """Create thermal relaxation noise channel."""
        kraus_ops = thermal_relaxation_kraus_operators(t1, t2, gate_time, temperature)
        return NoiseChannel(name=name, kraus_operators=kraus_ops)
    
    @staticmethod
    def pauli_channel(
        px: float, 
        py: float, 
        pz: float, 
        name: str = "pauli_channel"
    ) -> NoiseChannel:
        """Create general Pauli channel."""
        kraus_ops = pauli_channel_kraus_operators(px, py, pz)
        return NoiseChannel(name=name, kraus_operators=kraus_ops)
    
    @staticmethod
    def two_qubit_depolarizing(p: float, name: str = "two_qubit_depolarizing") -> NoiseChannel:
        """Create two-qubit depolarizing channel."""
        kraus_ops = two_qubit_depolarizing_kraus_operators(p)
        return NoiseChannel(name=name, kraus_operators=kraus_ops)