"""Depolarizing noise model implementation."""

import jax.numpy as jnp
from typing import List, Any, Dict, Optional
from .base import NoiseModel, NoiseChannel, depolarizing_kraus_operators, two_qubit_depolarizing_kraus_operators


class DepolarizingNoise(NoiseModel):
    """
    Depolarizing noise model.
    
    Applies depolarizing channels after each gate operation.
    The depolarizing channel maps a state ρ to:
    ρ → (1-p)ρ + p/d * I
    
    where p is the depolarizing probability and d is the dimension.
    """
    
    def __init__(
        self,
        single_qubit_error_rate: float = 0.001,
        two_qubit_error_rate: float = 0.01,
        readout_error_rate: float = 0.02
    ):
        """
        Initialize depolarizing noise model.
        
        Args:
            single_qubit_error_rate: Error rate for single-qubit gates
            two_qubit_error_rate: Error rate for two-qubit gates
            readout_error_rate: Error rate for readout operations
        """
        super().__init__("depolarizing")
        
        self.single_qubit_error_rate = single_qubit_error_rate
        self.two_qubit_error_rate = two_qubit_error_rate
        self.readout_error_rate = readout_error_rate
        
        # Create noise channels
        self._create_channels()
    
    def _create_channels(self) -> None:
        """Create depolarizing noise channels."""
        # Single-qubit depolarizing channel
        if self.single_qubit_error_rate > 0:
            single_qubit_kraus = depolarizing_kraus_operators(self.single_qubit_error_rate)
            single_channel = NoiseChannel(
                name="single_qubit_depolarizing",
                kraus_operators=single_qubit_kraus,
                probability=1.0
            )
            self.add_channel(single_channel)
        
        # Two-qubit depolarizing channel
        if self.two_qubit_error_rate > 0:
            two_qubit_kraus = two_qubit_depolarizing_kraus_operators(self.two_qubit_error_rate)
            two_channel = NoiseChannel(
                name="two_qubit_depolarizing",
                kraus_operators=two_qubit_kraus,
                probability=1.0
            )
            self.add_channel(two_channel)
        
        # Readout error channel (simplified as single-qubit depolarizing)
        if self.readout_error_rate > 0:
            readout_kraus = depolarizing_kraus_operators(self.readout_error_rate)
            readout_channel = NoiseChannel(
                name="readout_error",
                kraus_operators=readout_kraus,
                probability=1.0
            )
            self.add_channel(readout_channel)
    
    def get_noise_channels(self, circuit: Any) -> List[NoiseChannel]:
        """
        Get noise channels for circuit gates.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            List of noise channels to apply
        """
        channels = []
        
        # Add noise after each gate
        if hasattr(circuit, 'gates'):
            for gate in circuit.gates:
                gate_type = gate.get("type", "unknown")
                
                if gate_type == "single":
                    if "single_qubit_depolarizing" in self.channels:
                        channel = self.channels["single_qubit_depolarizing"]
                        # Create channel instance for specific qubits
                        gate_channel = NoiseChannel(
                            name=f"{channel.name}_{gate['qubits']}",
                            kraus_operators=channel.kraus_operators,
                            probability=channel.probability,
                            qubits=gate["qubits"]
                        )
                        channels.append(gate_channel)
                
                elif gate_type in ["two", "multi"]:
                    if "two_qubit_depolarizing" in self.channels:
                        channel = self.channels["two_qubit_depolarizing"]
                        gate_channel = NoiseChannel(
                            name=f"{channel.name}_{gate['qubits']}",
                            kraus_operators=channel.kraus_operators,
                            probability=channel.probability,
                            qubits=gate["qubits"]
                        )
                        channels.append(gate_channel)
        
        # Add readout noise if measurements present
        if hasattr(circuit, 'measurements') and circuit.measurements:
            if "readout_error" in self.channels:
                for measurement in circuit.measurements:
                    channel = self.channels["readout_error"]
                    readout_channel = NoiseChannel(
                        name=f"readout_error_{measurement['qubit']}",
                        kraus_operators=channel.kraus_operators,
                        probability=channel.probability,
                        qubits=[measurement['qubit']]
                    )
                    channels.append(readout_channel)
        
        return channels
    
    def apply_noise(self, circuit: Any) -> Any:
        """
        Apply depolarizing noise to circuit.
        
        Args:
            circuit: Clean quantum circuit
            
        Returns:
            Circuit with depolarizing noise applied
        """
        # This is a simplified implementation
        # In practice, would need to insert noise operations into circuit
        
        if hasattr(circuit, 'copy'):
            noisy_circuit = circuit.copy()
        else:
            import copy
            noisy_circuit = copy.deepcopy(circuit)
        
        # Add noise markers for simulation
        noisy_circuit._noise_model = self
        noisy_circuit._noise_channels = self.get_noise_channels(circuit)
        
        return noisy_circuit
    
    def scale_noise(self, factor: float) -> "DepolarizingNoise":
        """
        Scale depolarizing noise by a factor.
        
        Args:
            factor: Scaling factor
            
        Returns:
            New noise model with scaled parameters
        """
        return DepolarizingNoise(
            single_qubit_error_rate=min(self.single_qubit_error_rate * factor, 1.0),
            two_qubit_error_rate=min(self.two_qubit_error_rate * factor, 1.0),
            readout_error_rate=min(self.readout_error_rate * factor, 1.0)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            "single_qubit_error_rate": float(self.single_qubit_error_rate),
            "two_qubit_error_rate": float(self.two_qubit_error_rate),
            "readout_error_rate": float(self.readout_error_rate)
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DepolarizingNoise":
        """Create from dictionary representation."""
        return cls(
            single_qubit_error_rate=data["single_qubit_error_rate"],
            two_qubit_error_rate=data["two_qubit_error_rate"],
            readout_error_rate=data["readout_error_rate"]
        )
    
    def get_effective_error_rate(self, gate_type: str) -> float:
        """
        Get effective error rate for a gate type.
        
        Args:
            gate_type: Type of gate ("single", "two", "readout")
            
        Returns:
            Effective error rate
        """
        if gate_type == "single":
            return self.single_qubit_error_rate
        elif gate_type in ["two", "multi"]:
            return self.two_qubit_error_rate
        elif gate_type == "readout":
            return self.readout_error_rate
        else:
            return 0.0
    
    def __str__(self) -> str:
        """String representation."""
        lines = [f"DepolarizingNoise Model"]
        lines.append(f"Single-qubit error rate: {self.single_qubit_error_rate:.6f}")
        lines.append(f"Two-qubit error rate: {self.two_qubit_error_rate:.6f}")
        lines.append(f"Readout error rate: {self.readout_error_rate:.6f}")
        lines.append(f"Channels: {len(self.channels)}")
        return "\n".join(lines)


class ThermalNoise(DepolarizingNoise):
    """
    Thermal noise model based on finite temperature.
    
    Extends depolarizing noise with temperature-dependent error rates.
    """
    
    def __init__(
        self,
        temperature: float = 0.01,  # In units where ℏω/k_B = 1
        t1_time: float = 100.0,     # T1 relaxation time (μs)
        t2_time: float = 50.0,      # T2 dephasing time (μs)
        gate_time: float = 0.1      # Gate operation time (μs)
    ):
        """
        Initialize thermal noise model.
        
        Args:
            temperature: Effective temperature
            t1_time: T1 relaxation time
            t2_time: T2 dephasing time  
            gate_time: Gate operation time
        """
        self.temperature = temperature
        self.t1_time = t1_time
        self.t2_time = t2_time
        self.gate_time = gate_time
        
        # Calculate error rates from physical parameters
        single_error_rate = self._calculate_thermal_error_rate()
        two_error_rate = single_error_rate * 2  # Approximate scaling
        readout_error_rate = single_error_rate * 0.5
        
        super().__init__(
            single_qubit_error_rate=single_error_rate,
            two_qubit_error_rate=two_error_rate,
            readout_error_rate=readout_error_rate
        )
        
        self.name = "thermal"
    
    def _calculate_thermal_error_rate(self) -> float:
        """Calculate error rate from thermal parameters."""
        import jax.numpy as jnp
        
        # Simple model: error rate increases with temperature and gate time
        # and decreases with coherence times
        
        thermal_factor = 1 / (1 + jnp.exp(-1/self.temperature))  # Fermi-Dirac like
        decoherence_factor = self.gate_time / min(self.t1_time, self.t2_time)
        
        error_rate = thermal_factor * decoherence_factor * 0.01  # Base scale
        
        return min(float(error_rate), 0.5)  # Cap at 50%
    
    def set_temperature(self, temperature: float) -> None:
        """Update temperature and recalculate error rates."""
        self.temperature = temperature
        
        # Recalculate error rates
        single_error_rate = self._calculate_thermal_error_rate()
        self.single_qubit_error_rate = single_error_rate
        self.two_qubit_error_rate = single_error_rate * 2
        self.readout_error_rate = single_error_rate * 0.5
        
        # Recreate channels
        self.channels.clear()
        self._create_channels()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            "temperature": float(self.temperature),
            "t1_time": float(self.t1_time),
            "t2_time": float(self.t2_time),
            "gate_time": float(self.gate_time)
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThermalNoise":
        """Create from dictionary representation."""
        return cls(
            temperature=data["temperature"],
            t1_time=data["t1_time"],
            t2_time=data["t2_time"],
            gate_time=data["gate_time"]
        )