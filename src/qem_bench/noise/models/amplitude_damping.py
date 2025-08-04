"""Amplitude damping noise model implementation."""

import jax.numpy as jnp
import numpy as np
from typing import List, Any, Dict, Optional
from .base import NoiseModel, NoiseChannel, amplitude_damping_kraus_operators, generalized_amplitude_damping_kraus_operators


class AmplitudeDampingNoiseModel(NoiseModel):
    """
    Amplitude damping noise model for T1 relaxation.
    
    Models energy relaxation from |1⟩ to |0⟩ state with probability γ.
    Can include thermal effects at finite temperature.
    """
    
    def __init__(
        self,
        t1_times: Dict[int, float],
        gate_times: Dict[str, float] = None,
        temperature: float = 0.0,
        thermal_population: Optional[Dict[int, float]] = None
    ):
        """
        Initialize amplitude damping noise model.
        
        Args:
            t1_times: T1 relaxation times for each qubit (in μs)
            gate_times: Gate operation times for different gate types (in μs)
            temperature: System temperature (affects thermal population)
            thermal_population: Explicit thermal population for each qubit
        """
        super().__init__("amplitude_damping")
        
        self.t1_times = t1_times
        self.gate_times = gate_times or {
            "single": 0.1,  # Single-qubit gate time
            "two": 0.5,     # Two-qubit gate time
            "readout": 1.0  # Readout time
        }
        self.temperature = temperature
        self.thermal_population = thermal_population or {}
        
        # Create amplitude damping channels for each qubit
        self._create_channels()
    
    def _create_channels(self) -> None:
        """Create amplitude damping noise channels."""
        for qubit, t1 in self.t1_times.items():
            # Single-qubit gate channel
            gamma_single = self._calculate_decay_probability(t1, self.gate_times["single"])
            temp = self.thermal_population.get(qubit, self.temperature)
            
            if gamma_single > 0:
                if temp > 0:
                    # Use generalized amplitude damping for thermal effects
                    kraus_ops = generalized_amplitude_damping_kraus_operators(gamma_single, temp)
                    channel_name = f"amplitude_damping_thermal_q{qubit}"
                else:
                    # Standard amplitude damping
                    kraus_ops = amplitude_damping_kraus_operators(gamma_single)
                    channel_name = f"amplitude_damping_q{qubit}"
                
                channel = NoiseChannel(
                    name=channel_name,
                    kraus_operators=kraus_ops,
                    probability=1.0,
                    qubits=[qubit]
                )
                self.add_channel(channel)
            
            # Two-qubit gate channel (higher decay due to longer duration)
            gamma_two = self._calculate_decay_probability(t1, self.gate_times["two"])
            if gamma_two > 0:
                if temp > 0:
                    kraus_ops_two = generalized_amplitude_damping_kraus_operators(gamma_two, temp)
                    channel_name_two = f"amplitude_damping_two_thermal_q{qubit}"
                else:
                    kraus_ops_two = amplitude_damping_kraus_operators(gamma_two)
                    channel_name_two = f"amplitude_damping_two_q{qubit}"
                
                channel_two = NoiseChannel(
                    name=channel_name_two,
                    kraus_operators=kraus_ops_two,
                    probability=1.0,
                    qubits=[qubit]
                )
                self.add_channel(channel_two)
            
            # Readout channel
            gamma_readout = self._calculate_decay_probability(t1, self.gate_times["readout"])
            if gamma_readout > 0:
                if temp > 0:
                    kraus_ops_readout = generalized_amplitude_damping_kraus_operators(gamma_readout, temp)
                    channel_name_readout = f"amplitude_damping_readout_thermal_q{qubit}"
                else:
                    kraus_ops_readout = amplitude_damping_kraus_operators(gamma_readout)
                    channel_name_readout = f"amplitude_damping_readout_q{qubit}"
                
                channel_readout = NoiseChannel(
                    name=channel_name_readout,
                    kraus_operators=kraus_ops_readout,
                    probability=1.0,
                    qubits=[qubit]
                )
                self.add_channel(channel_readout)
    
    def _calculate_decay_probability(self, t1: float, gate_time: float) -> float:
        """Calculate amplitude damping probability for given T1 and gate time."""
        if t1 <= 0:
            return 0.0
        return float(1 - jnp.exp(-gate_time / t1))
    
    def get_noise_channels(self, circuit: Any) -> List[NoiseChannel]:
        """
        Get amplitude damping channels for circuit gates.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            List of noise channels to apply
        """
        channels = []
        
        if hasattr(circuit, 'gates'):
            for gate in circuit.gates:
                gate_type = gate.get("type", "unknown")
                qubits = gate.get("qubits", [])
                
                # Apply amplitude damping to each qubit in the gate
                for qubit in qubits:
                    if gate_type == "single":
                        channel_name = f"amplitude_damping_q{qubit}"
                        if "thermal" in [c for c in self.channels.keys() if f"q{qubit}" in c]:
                            channel_name = f"amplitude_damping_thermal_q{qubit}"
                    elif gate_type in ["two", "multi"]:
                        channel_name = f"amplitude_damping_two_q{qubit}"
                        if "thermal" in [c for c in self.channels.keys() if f"q{qubit}" in c]:
                            channel_name = f"amplitude_damping_two_thermal_q{qubit}"
                    else:
                        continue
                    
                    if channel_name in self.channels:
                        channel = self.channels[channel_name]
                        gate_channel = NoiseChannel(
                            name=f"{channel.name}_gate_{id(gate)}",
                            kraus_operators=channel.kraus_operators,
                            probability=channel.probability,
                            qubits=[qubit]
                        )
                        channels.append(gate_channel)
        
        # Apply readout noise to measurements
        if hasattr(circuit, 'measurements') and circuit.measurements:
            for measurement in circuit.measurements:
                qubit = measurement.get('qubit')
                if qubit is not None:
                    channel_name = f"amplitude_damping_readout_q{qubit}"
                    if "thermal" in [c for c in self.channels.keys() if f"q{qubit}" in c]:
                        channel_name = f"amplitude_damping_readout_thermal_q{qubit}"
                    
                    if channel_name in self.channels:
                        channel = self.channels[channel_name]
                        readout_channel = NoiseChannel(
                            name=f"{channel.name}_measurement_{id(measurement)}",
                            kraus_operators=channel.kraus_operators,
                            probability=channel.probability,
                            qubits=[qubit]
                        )
                        channels.append(readout_channel)
        
        return channels
    
    def apply_noise(self, circuit: Any) -> Any:
        """
        Apply amplitude damping noise to circuit.
        
        Args:
            circuit: Clean quantum circuit
            
        Returns:
            Circuit with amplitude damping noise applied
        """
        if hasattr(circuit, 'copy'):
            noisy_circuit = circuit.copy()
        else:
            import copy
            noisy_circuit = copy.deepcopy(circuit)
        
        # Add noise markers for simulation
        noisy_circuit._noise_model = self
        noisy_circuit._noise_channels = self.get_noise_channels(circuit)
        
        return noisy_circuit
    
    def set_temperature(self, temperature: float) -> None:
        """Update system temperature and recreate channels."""
        self.temperature = temperature
        self.channels.clear()
        self._create_channels()
    
    def set_t1_time(self, qubit: int, t1_time: float) -> None:
        """Update T1 time for specific qubit."""
        self.t1_times[qubit] = t1_time
        
        # Remove old channels for this qubit
        to_remove = [name for name in self.channels.keys() if f"q{qubit}" in name]
        for name in to_remove:
            del self.channels[name]
        
        # Recreate channels
        self._create_channels()
    
    def get_effective_decay_rate(self, qubit: int, gate_type: str) -> float:
        """
        Get effective decay rate for a qubit and gate type.
        
        Args:
            qubit: Qubit index
            gate_type: Type of gate ("single", "two", "readout")
            
        Returns:
            Effective decay probability
        """
        if qubit not in self.t1_times:
            return 0.0
        
        t1 = self.t1_times[qubit]
        gate_time = self.gate_times.get(gate_type, 0.0)
        
        return self._calculate_decay_probability(t1, gate_time)
    
    def scale_noise(self, factor: float) -> "AmplitudeDampingNoiseModel":
        """
        Scale amplitude damping noise by factor.
        
        Args:
            factor: Scaling factor
            
        Returns:
            New noise model with scaled T1 times
        """
        # Scale T1 times (shorter T1 = more noise)
        scaled_t1_times = {qubit: t1 / factor for qubit, t1 in self.t1_times.items()}
        
        return AmplitudeDampingNoiseModel(
            t1_times=scaled_t1_times,
            gate_times=self.gate_times.copy(),
            temperature=self.temperature,
            thermal_population=self.thermal_population.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            "t1_times": {str(k): float(v) for k, v in self.t1_times.items()},
            "gate_times": {k: float(v) for k, v in self.gate_times.items()},
            "temperature": float(self.temperature),
            "thermal_population": {str(k): float(v) for k, v in self.thermal_population.items()}
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AmplitudeDampingNoiseModel":
        """Create from dictionary representation."""
        return cls(
            t1_times={int(k): v for k, v in data["t1_times"].items()},
            gate_times=data["gate_times"],
            temperature=data["temperature"],
            thermal_population={int(k): v for k, v in data["thermal_population"].items()}
        )
    
    def __str__(self) -> str:
        """String representation."""
        lines = [f"AmplitudeDampingNoiseModel"]
        lines.append(f"T1 times: {self.t1_times}")
        lines.append(f"Gate times: {self.gate_times}")
        lines.append(f"Temperature: {self.temperature}")
        if self.thermal_population:
            lines.append(f"Thermal populations: {self.thermal_population}")
        lines.append(f"Channels: {len(self.channels)}")
        return "\n".join(lines)


class T1DecayModel(AmplitudeDampingNoiseModel):
    """
    Simplified T1 decay model with uniform parameters.
    
    Convenience class for common use cases.
    """
    
    def __init__(
        self,
        num_qubits: int,
        t1_time: float = 100.0,
        single_gate_time: float = 0.1,
        two_gate_time: float = 0.5,
        readout_time: float = 1.0,
        temperature: float = 0.0
    ):
        """
        Initialize uniform T1 decay model.
        
        Args:
            num_qubits: Number of qubits
            t1_time: T1 relaxation time for all qubits (μs)
            single_gate_time: Single-qubit gate time (μs)
            two_gate_time: Two-qubit gate time (μs)
            readout_time: Readout time (μs)
            temperature: System temperature
        """
        t1_times = {i: t1_time for i in range(num_qubits)}
        gate_times = {
            "single": single_gate_time,
            "two": two_gate_time,
            "readout": readout_time
        }
        
        super().__init__(
            t1_times=t1_times,
            gate_times=gate_times,
            temperature=temperature
        )
        
        self.name = "t1_decay"
    
    @classmethod
    def from_device_parameters(
        cls,
        device_params: Dict[str, Any]
    ) -> "T1DecayModel":
        """
        Create T1 decay model from device parameter dictionary.
        
        Args:
            device_params: Dictionary with device parameters
            
        Returns:
            T1DecayModel instance
        """
        return cls(
            num_qubits=device_params.get("num_qubits", 5),
            t1_time=device_params.get("t1_time", 100.0),
            single_gate_time=device_params.get("single_gate_time", 0.1),
            two_gate_time=device_params.get("two_gate_time", 0.5),
            readout_time=device_params.get("readout_time", 1.0),
            temperature=device_params.get("temperature", 0.0)
        )