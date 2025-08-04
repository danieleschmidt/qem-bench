"""Phase damping noise model implementation."""

import jax.numpy as jnp
import numpy as np
from typing import List, Any, Dict, Optional
from .base import NoiseModel, NoiseChannel, phase_damping_kraus_operators, thermal_relaxation_kraus_operators


class PhaseDampingNoiseModel(NoiseModel):
    """
    Phase damping noise model for T2 dephasing.
    
    Models loss of quantum coherence without energy loss.
    Can model pure dephasing (T2*) or include T1 effects for full T2.
    """
    
    def __init__(
        self,
        t2_times: Dict[int, float],
        t1_times: Optional[Dict[int, float]] = None,
        gate_times: Dict[str, float] = None,
        pure_dephasing: bool = True
    ):
        """
        Initialize phase damping noise model.
        
        Args:
            t2_times: T2 dephasing times for each qubit (in μs)
            t1_times: T1 relaxation times for each qubit (optional, for full T2 model)
            gate_times: Gate operation times for different gate types (in μs)
            pure_dephasing: If True, model pure dephasing (T2*), else full T2
        """
        super().__init__("phase_damping")
        
        self.t2_times = t2_times
        self.t1_times = t1_times or {}
        self.gate_times = gate_times or {
            "single": 0.1,  # Single-qubit gate time
            "two": 0.5,     # Two-qubit gate time
            "readout": 1.0  # Readout time
        }
        self.pure_dephasing = pure_dephasing
        
        # Create phase damping channels for each qubit
        self._create_channels()
    
    def _create_channels(self) -> None:
        """Create phase damping noise channels."""
        for qubit, t2 in self.t2_times.items():
            t1 = self.t1_times.get(qubit, float('inf'))
            
            # Single-qubit gate channel
            gamma_single = self._calculate_dephasing_probability(t2, t1, self.gate_times["single"])
            if gamma_single > 0:
                if self.pure_dephasing or t1 == float('inf'):
                    # Pure dephasing
                    kraus_ops = phase_damping_kraus_operators(gamma_single)
                    channel_name = f"phase_damping_q{qubit}"
                else:
                    # Full T2 with T1 effects
                    kraus_ops = thermal_relaxation_kraus_operators(
                        t1, t2, self.gate_times["single"], 0.0
                    )
                    channel_name = f"thermal_relaxation_q{qubit}"
                
                channel = NoiseChannel(
                    name=channel_name,
                    kraus_operators=kraus_ops,
                    probability=1.0,
                    qubits=[qubit]
                )
                self.add_channel(channel)
            
            # Two-qubit gate channel
            gamma_two = self._calculate_dephasing_probability(t2, t1, self.gate_times["two"])
            if gamma_two > 0:
                if self.pure_dephasing or t1 == float('inf'):
                    kraus_ops_two = phase_damping_kraus_operators(gamma_two)
                    channel_name_two = f"phase_damping_two_q{qubit}"
                else:
                    kraus_ops_two = thermal_relaxation_kraus_operators(
                        t1, t2, self.gate_times["two"], 0.0
                    )
                    channel_name_two = f"thermal_relaxation_two_q{qubit}"
                
                channel_two = NoiseChannel(
                    name=channel_name_two,
                    kraus_operators=kraus_ops_two,
                    probability=1.0,
                    qubits=[qubit]
                )
                self.add_channel(channel_two)
            
            # Readout channel
            gamma_readout = self._calculate_dephasing_probability(t2, t1, self.gate_times["readout"])
            if gamma_readout > 0:
                if self.pure_dephasing or t1 == float('inf'):
                    kraus_ops_readout = phase_damping_kraus_operators(gamma_readout)
                    channel_name_readout = f"phase_damping_readout_q{qubit}"
                else:
                    kraus_ops_readout = thermal_relaxation_kraus_operators(
                        t1, t2, self.gate_times["readout"], 0.0
                    )
                    channel_name_readout = f"thermal_relaxation_readout_q{qubit}"
                
                channel_readout = NoiseChannel(
                    name=channel_name_readout,
                    kraus_operators=kraus_ops_readout,
                    probability=1.0,
                    qubits=[qubit]
                )
                self.add_channel(channel_readout)
    
    def _calculate_dephasing_probability(self, t2: float, t1: float, gate_time: float) -> float:
        """
        Calculate dephasing probability for given T2, T1, and gate time.
        
        For pure dephasing: γ_φ = 1 - exp(-t/T2*)
        For full T2: T2* calculated from T1 and T2
        """
        if t2 <= 0:
            return 0.0
        
        if self.pure_dephasing or t1 == float('inf'):
            # Pure dephasing
            return float(1 - jnp.exp(-gate_time / t2))
        else:
            # Full T2 model: 1/T2 = 1/2T1 + 1/T2*
            # So T2* = 1/(1/T2 - 1/2T1)
            t2_star_inv = 1/t2 - 1/(2*t1)
            if t2_star_inv <= 0:
                return 0.0
            t2_star = 1/t2_star_inv
            return float(1 - jnp.exp(-gate_time / t2_star))
    
    def get_noise_channels(self, circuit: Any) -> List[NoiseChannel]:
        """
        Get phase damping channels for circuit gates.
        
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
                
                # Apply phase damping to each qubit in the gate
                for qubit in qubits:
                    if gate_type == "single":
                        if self.pure_dephasing:
                            channel_name = f"phase_damping_q{qubit}"
                        else:
                            channel_name = f"thermal_relaxation_q{qubit}"
                    elif gate_type in ["two", "multi"]:
                        if self.pure_dephasing:
                            channel_name = f"phase_damping_two_q{qubit}"
                        else:
                            channel_name = f"thermal_relaxation_two_q{qubit}"
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
        
        # Apply dephasing noise to measurements
        if hasattr(circuit, 'measurements') and circuit.measurements:
            for measurement in circuit.measurements:
                qubit = measurement.get('qubit')
                if qubit is not None:
                    if self.pure_dephasing:
                        channel_name = f"phase_damping_readout_q{qubit}"
                    else:
                        channel_name = f"thermal_relaxation_readout_q{qubit}"
                    
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
        Apply phase damping noise to circuit.
        
        Args:
            circuit: Clean quantum circuit
            
        Returns:
            Circuit with phase damping noise applied
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
    
    def set_t2_time(self, qubit: int, t2_time: float) -> None:
        """Update T2 time for specific qubit.""" 
        self.t2_times[qubit] = t2_time
        
        # Remove old channels for this qubit
        to_remove = [name for name in self.channels.keys() if f"q{qubit}" in name]
        for name in to_remove:
            del self.channels[name]
        
        # Recreate channels
        self._create_channels()
    
    def set_t1_time(self, qubit: int, t1_time: float) -> None:
        """Update T1 time for specific qubit (affects full T2 model)."""
        self.t1_times[qubit] = t1_time
        
        if not self.pure_dephasing:
            # Remove old channels for this qubit and recreate
            to_remove = [name for name in self.channels.keys() if f"q{qubit}" in name]
            for name in to_remove:
                del self.channels[name]
            
            # Recreate channels
            self._create_channels()
    
    def get_effective_dephasing_rate(self, qubit: int, gate_type: str) -> float:
        """
        Get effective dephasing rate for a qubit and gate type.
        
        Args:
            qubit: Qubit index
            gate_type: Type of gate ("single", "two", "readout")
            
        Returns:
            Effective dephasing probability
        """
        if qubit not in self.t2_times:
            return 0.0
        
        t2 = self.t2_times[qubit]
        t1 = self.t1_times.get(qubit, float('inf'))
        gate_time = self.gate_times.get(gate_type, 0.0)
        
        return self._calculate_dephasing_probability(t2, t1, gate_time)
    
    def scale_noise(self, factor: float) -> "PhaseDampingNoiseModel":
        """
        Scale phase damping noise by factor.
        
        Args:
            factor: Scaling factor
            
        Returns:
            New noise model with scaled T2 times
        """
        # Scale T2 times (shorter T2 = more noise)
        scaled_t2_times = {qubit: t2 / factor for qubit, t2 in self.t2_times.items()}
        scaled_t1_times = {qubit: t1 / factor for qubit, t1 in self.t1_times.items()}
        
        return PhaseDampingNoiseModel(
            t2_times=scaled_t2_times,
            t1_times=scaled_t1_times,
            gate_times=self.gate_times.copy(),
            pure_dephasing=self.pure_dephasing
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            "t2_times": {str(k): float(v) for k, v in self.t2_times.items()},
            "t1_times": {str(k): float(v) for k, v in self.t1_times.items()},
            "gate_times": {k: float(v) for k, v in self.gate_times.items()},
            "pure_dephasing": bool(self.pure_dephasing)
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseDampingNoiseModel":
        """Create from dictionary representation."""
        return cls(
            t2_times={int(k): v for k, v in data["t2_times"].items()},
            t1_times={int(k): v for k, v in data["t1_times"].items()},
            gate_times=data["gate_times"],
            pure_dephasing=data["pure_dephasing"]
        )
    
    def __str__(self) -> str:
        """String representation."""
        lines = [f"PhaseDampingNoiseModel"]
        lines.append(f"T2 times: {self.t2_times}")
        if self.t1_times:
            lines.append(f"T1 times: {self.t1_times}")
        lines.append(f"Gate times: {self.gate_times}")
        lines.append(f"Pure dephasing: {self.pure_dephasing}")
        lines.append(f"Channels: {len(self.channels)}")
        return "\n".join(lines)


class T2DephaseModel(PhaseDampingNoiseModel):
    """
    Simplified T2 dephasing model with uniform parameters.
    
    Convenience class for common use cases.
    """
    
    def __init__(
        self,
        num_qubits: int,
        t2_time: float = 50.0,
        t1_time: Optional[float] = None,
        single_gate_time: float = 0.1,
        two_gate_time: float = 0.5,
        readout_time: float = 1.0,
        pure_dephasing: bool = True
    ):
        """
        Initialize uniform T2 dephasing model.
        
        Args:
            num_qubits: Number of qubits
            t2_time: T2 dephasing time for all qubits (μs)
            t1_time: T1 relaxation time for all qubits (μs, optional)
            single_gate_time: Single-qubit gate time (μs)
            two_gate_time: Two-qubit gate time (μs)
            readout_time: Readout time (μs)
            pure_dephasing: If True, model pure dephasing
        """
        t2_times = {i: t2_time for i in range(num_qubits)}
        t1_times = {i: t1_time for i in range(num_qubits)} if t1_time else {}
        gate_times = {
            "single": single_gate_time,
            "two": two_gate_time,
            "readout": readout_time
        }
        
        super().__init__(
            t2_times=t2_times,
            t1_times=t1_times,
            gate_times=gate_times,
            pure_dephasing=pure_dephasing
        )
        
        self.name = "t2_dephase"
    
    @classmethod
    def from_device_parameters(
        cls,
        device_params: Dict[str, Any]
    ) -> "T2DephaseModel":
        """
        Create T2 dephasing model from device parameter dictionary.
        
        Args:
            device_params: Dictionary with device parameters
            
        Returns:
            T2DephaseModel instance
        """
        return cls(
            num_qubits=device_params.get("num_qubits", 5),
            t2_time=device_params.get("t2_time", 50.0),
            t1_time=device_params.get("t1_time"),
            single_gate_time=device_params.get("single_gate_time", 0.1),
            two_gate_time=device_params.get("two_gate_time", 0.5),
            readout_time=device_params.get("readout_time", 1.0),
            pure_dephasing=device_params.get("pure_dephasing", True)
        )


class RamseyDephaseModel(PhaseDampingNoiseModel):
    """
    Ramsey-type dephasing model with Gaussian frequency fluctuations.
    
    Models inhomogeneous broadening and frequency noise.
    """
    
    def __init__(
        self,
        num_qubits: int,
        t2_star_time: float = 20.0,
        frequency_std: float = 1.0,  # MHz
        single_gate_time: float = 0.1,
        two_gate_time: float = 0.5
    ):
        """
        Initialize Ramsey dephasing model.
        
        Args:
            num_qubits: Number of qubits
            t2_star_time: T2* time for inhomogeneous dephasing (μs)
            frequency_std: Standard deviation of frequency fluctuations (MHz)
            single_gate_time: Single-qubit gate time (μs)
            two_gate_time: Two-qubit gate time (μs)
        """
        # Generate random frequency offsets for each qubit
        np.random.seed(42)  # For reproducibility
        frequency_offsets = np.random.normal(0, frequency_std, num_qubits)
        
        # Calculate effective T2* times including frequency noise
        t2_times = {}
        for i in range(num_qubits):
            # T2* includes both homogeneous and inhomogeneous contributions
            df = frequency_offsets[i]  # MHz
            t2_inhom = 1 / (2 * np.pi * abs(df)) if df != 0 else float('inf')  # μs
            
            # Combined T2*: 1/T2* = 1/T2*(homog) + 1/T2*(inhomog)
            t2_eff = 1 / (1/t2_star_time + 1/t2_inhom)
            t2_times[i] = t2_eff
        
        gate_times = {
            "single": single_gate_time,
            "two": two_gate_time,
            "readout": 1.0
        }
        
        super().__init__(
            t2_times=t2_times,
            gate_times=gate_times,
            pure_dephasing=True  # Ramsey is pure dephasing
        )
        
        self.name = "ramsey_dephase"
        self.frequency_offsets = frequency_offsets
        self.frequency_std = frequency_std
    
    def get_frequency_offset(self, qubit: int) -> float:
        """Get frequency offset for specific qubit."""
        return self.frequency_offsets[qubit]
    
    def __str__(self) -> str:
        """String representation."""
        lines = [f"RamseyDephaseModel"]
        lines.append(f"Frequency std: {self.frequency_std} MHz")
        lines.append(f"Frequency offsets: {self.frequency_offsets}")
        lines.append(f"Effective T2* times: {self.t2_times}")
        lines.append(f"Channels: {len(self.channels)}")
        return "\n".join(lines)