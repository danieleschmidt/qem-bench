"""Coherent noise model implementation."""

import jax.numpy as jnp
import numpy as np
from typing import List, Any, Dict, Optional
from .base import NoiseModel, NoiseChannel


class CoherentNoise(NoiseModel):
    """
    Coherent noise model for systematic errors.
    
    Models coherent errors such as:
    - Over/under rotation errors in gates
    - Control field calibration errors
    - Crosstalk between qubits
    """
    
    def __init__(
        self,
        rotation_error_std: float = 0.01,
        crosstalk_strength: float = 0.005,
        control_amplitude_error: float = 0.02,
        phase_drift_rate: float = 0.001
    ):
        """
        Initialize coherent noise model.
        
        Args:
            rotation_error_std: Standard deviation of rotation angle errors
            crosstalk_strength: Strength of crosstalk between qubits
            control_amplitude_error: Relative error in control pulse amplitudes
            phase_drift_rate: Rate of phase drift over time
        """
        super().__init__("coherent")
        
        self.rotation_error_std = rotation_error_std
        self.crosstalk_strength = crosstalk_strength
        self.control_amplitude_error = control_amplitude_error
        self.phase_drift_rate = phase_drift_rate
        
        # Create coherent error channels
        self._create_channels()
    
    def _create_channels(self) -> None:
        """Create coherent noise channels."""
        # Rotation error channel
        if self.rotation_error_std > 0:
            rotation_channel = NoiseChannel(
                name="rotation_error",
                kraus_operators=self._rotation_error_kraus(),
                probability=1.0
            )
            self.add_channel(rotation_channel)
        
        # Amplitude error channel
        if self.control_amplitude_error > 0:
            amplitude_channel = NoiseChannel(
                name="amplitude_error",
                kraus_operators=self._amplitude_error_kraus(),
                probability=1.0
            )
            self.add_channel(amplitude_channel)
        
        # Phase drift channel
        if self.phase_drift_rate > 0:
            phase_channel = NoiseChannel(
                name="phase_drift",
                kraus_operators=self._phase_drift_kraus(),
                probability=1.0
            )
            self.add_channel(phase_channel)
    
    def _rotation_error_kraus(self) -> List[jnp.ndarray]:
        """Generate Kraus operators for rotation errors."""
        # Sample rotation error
        error_angle = np.random.normal(0, self.rotation_error_std)
        
        # Small rotation around random axis
        axis = np.random.choice(['x', 'y', 'z'])
        
        if axis == 'x':
            error_matrix = jnp.array([
                [jnp.cos(error_angle/2), -1j*jnp.sin(error_angle/2)],
                [-1j*jnp.sin(error_angle/2), jnp.cos(error_angle/2)]
            ], dtype=jnp.complex64)
        elif axis == 'y':
            error_matrix = jnp.array([
                [jnp.cos(error_angle/2), -jnp.sin(error_angle/2)],
                [jnp.sin(error_angle/2), jnp.cos(error_angle/2)]
            ], dtype=jnp.complex64)
        else:  # z
            error_matrix = jnp.array([
                [jnp.exp(-1j*error_angle/2), 0],
                [0, jnp.exp(1j*error_angle/2)]
            ], dtype=jnp.complex64)
        
        return [error_matrix]
    
    def _amplitude_error_kraus(self) -> List[jnp.ndarray]:
        """Generate Kraus operators for amplitude errors."""
        # Sample amplitude error
        amplitude_factor = 1 + np.random.normal(0, self.control_amplitude_error)
        amplitude_factor = max(0.1, min(2.0, amplitude_factor))  # Clamp to reasonable range
        
        # Scale identity matrix (simplified model)
        error_matrix = jnp.sqrt(amplitude_factor) * jnp.eye(2, dtype=jnp.complex64)
        
        return [error_matrix]
    
    def _phase_drift_kraus(self) -> List[jnp.ndarray]:
        """Generate Kraus operators for phase drift."""
        # Accumulating phase drift
        drift_phase = self.phase_drift_rate * np.random.exponential(1.0)
        
        # Global phase shift
        error_matrix = jnp.exp(1j * drift_phase) * jnp.eye(2, dtype=jnp.complex64)
        
        return [error_matrix]
    
    def get_noise_channels(self, circuit: Any) -> List[NoiseChannel]:
        """Get coherent noise channels for circuit."""
        channels = []
        
        if hasattr(circuit, 'gates'):
            for i, gate in enumerate(circuit.gates):
                gate_type = gate.get("type", "unknown")
                qubits = gate.get("qubits", [])
                
                # Add rotation errors to rotation gates
                if gate.get("name", "").startswith(("RX", "RY", "RZ")) and "rotation_error" in self.channels:
                    channel = self.channels["rotation_error"]
                    error_channel = NoiseChannel(
                        name=f"rotation_error_gate_{i}",
                        kraus_operators=self._rotation_error_kraus(),
                        probability=1.0,
                        qubits=qubits
                    )
                    channels.append(error_channel)
                
                # Add amplitude errors to all gates
                if "amplitude_error" in self.channels:
                    channel = self.channels["amplitude_error"]
                    amp_channel = NoiseChannel(
                        name=f"amplitude_error_gate_{i}",
                        kraus_operators=self._amplitude_error_kraus(),
                        probability=1.0,
                        qubits=qubits
                    )
                    channels.append(amp_channel)
                
                # Add phase drift based on gate position (time)
                if "phase_drift" in self.channels:
                    channel = self.channels["phase_drift"]
                    phase_channel = NoiseChannel(
                        name=f"phase_drift_gate_{i}",
                        kraus_operators=self._phase_drift_kraus(),
                        probability=1.0,
                        qubits=qubits
                    )
                    channels.append(phase_channel)
                
                # Add crosstalk for multi-qubit gates
                if len(qubits) > 1 and self.crosstalk_strength > 0:
                    crosstalk_channel = self._create_crosstalk_channel(qubits, i)
                    channels.append(crosstalk_channel)
        
        return channels
    
    def _create_crosstalk_channel(self, qubits: List[int], gate_index: int) -> NoiseChannel:
        """Create crosstalk channel for multi-qubit gates."""
        # Simple crosstalk model: unwanted coupling between qubits
        crosstalk_angle = np.random.normal(0, self.crosstalk_strength)
        
        # Create small two-qubit rotation (simplified)
        cos_half = jnp.cos(crosstalk_angle / 2)
        sin_half = jnp.sin(crosstalk_angle / 2)
        
        # Simplified two-qubit crosstalk matrix
        crosstalk_matrix = jnp.array([
            [cos_half, 0, 0, -1j*sin_half],
            [0, cos_half, -1j*sin_half, 0],
            [0, -1j*sin_half, cos_half, 0],
            [-1j*sin_half, 0, 0, cos_half]
        ], dtype=jnp.complex64)
        
        return NoiseChannel(
            name=f"crosstalk_gate_{gate_index}",
            kraus_operators=[crosstalk_matrix],
            probability=1.0,
            qubits=qubits[:2]  # Apply to first two qubits
        )
    
    def apply_noise(self, circuit: Any) -> Any:
        """Apply coherent noise to circuit."""
        if hasattr(circuit, 'copy'):
            noisy_circuit = circuit.copy()
        else:
            import copy
            noisy_circuit = copy.deepcopy(circuit)
        
        # Apply coherent errors by modifying gate parameters
        if hasattr(noisy_circuit, 'gates'):
            for gate in noisy_circuit.gates:
                self._apply_coherent_errors_to_gate(gate)
        
        # Add noise markers
        noisy_circuit._noise_model = self
        noisy_circuit._noise_channels = self.get_noise_channels(circuit)
        
        return noisy_circuit
    
    def _apply_coherent_errors_to_gate(self, gate: Dict[str, Any]) -> None:
        """Apply coherent errors directly to gate parameters."""
        gate_name = gate.get("name", "")
        
        # Add rotation errors to parameterized gates
        if gate_name.startswith(("RX", "RY", "RZ")):
            if "angle" in gate:
                error = np.random.normal(0, self.rotation_error_std)
                gate["angle"] += error
        
        # Add amplitude errors by scaling gate matrix
        if "matrix" in gate:
            amplitude_error = 1 + np.random.normal(0, self.control_amplitude_error)
            amplitude_error = max(0.1, min(2.0, amplitude_error))
            
            # This is a simplified approach - in practice would need more sophisticated handling
            gate["matrix"] = gate["matrix"] * jnp.sqrt(amplitude_error)
    
    def scale_noise(self, factor: float) -> "CoherentNoise":
        """Scale coherent noise parameters.""" 
        return CoherentNoise(
            rotation_error_std=self.rotation_error_std * factor,
            crosstalk_strength=self.crosstalk_strength * factor,
            control_amplitude_error=self.control_amplitude_error * factor,
            phase_drift_rate=self.phase_drift_rate * factor
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            "rotation_error_std": float(self.rotation_error_std),
            "crosstalk_strength": float(self.crosstalk_strength),
            "control_amplitude_error": float(self.control_amplitude_error),
            "phase_drift_rate": float(self.phase_drift_rate)
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoherentNoise":
        """Create from dictionary representation."""
        return cls(
            rotation_error_std=data["rotation_error_std"],
            crosstalk_strength=data["crosstalk_strength"],
            control_amplitude_error=data["control_amplitude_error"],
            phase_drift_rate=data["phase_drift_rate"]
        )
    
    def __str__(self) -> str:
        """String representation."""
        lines = [f"CoherentNoise Model"]
        lines.append(f"Rotation error std: {self.rotation_error_std:.6f}")
        lines.append(f"Crosstalk strength: {self.crosstalk_strength:.6f}")
        lines.append(f"Amplitude error: {self.control_amplitude_error:.4f}")
        lines.append(f"Phase drift rate: {self.phase_drift_rate:.6f}")
        return "\n".join(lines)