"""Time-dependent drift noise model implementation."""

import jax.numpy as jnp
import numpy as np
from typing import List, Any, Dict, Optional, Callable, Tuple
from .base import NoiseModel, NoiseChannel


class DriftNoiseModel(NoiseModel):
    """
    Time-dependent drift noise model.
    
    Models temporal variations in quantum device parameters, including:
    - Frequency drift due to environmental fluctuations
    - Amplitude drift in control pulses
    - Phase drift in oscillators
    - Temperature-dependent parameter changes
    - Long-term aging effects
    """
    
    def __init__(
        self,
        num_qubits: int,
        frequency_drift: Optional[Dict[int, Callable[[float], float]]] = None,
        amplitude_drift: Optional[Dict[int, Callable[[float], float]]] = None,
        phase_drift: Optional[Dict[int, Callable[[float], float]]] = None,
        t1_drift: Optional[Dict[int, Callable[[float], float]]] = None,
        t2_drift: Optional[Dict[int, Callable[[float], float]]] = None,
        time_unit: str = "us"
    ):
        """
        Initialize drift noise model.
        
        Args:
            num_qubits: Number of qubits
            frequency_drift: Functions describing frequency drift vs time
                           {qubit: function(time) -> frequency_shift_Hz}
            amplitude_drift: Functions describing amplitude drift vs time
                           {qubit: function(time) -> amplitude_factor}
            phase_drift: Functions describing phase drift vs time
                       {qubit: function(time) -> phase_shift_rad}
            t1_drift: Functions describing T1 drift vs time
                    {qubit: function(time) -> t1_time_us}
            t2_drift: Functions describing T2 drift vs time
                    {qubit: function(time) -> t2_time_us}
            time_unit: Time unit for drift functions ("us", "ms", "s")
        """
        super().__init__("drift")
        
        self.num_qubits = num_qubits
        self.frequency_drift = frequency_drift or {}
        self.amplitude_drift = amplitude_drift or {}
        self.phase_drift = phase_drift or {}
        self.t1_drift = t1_drift or {}
        self.t2_drift = t2_drift or {}
        self.time_unit = time_unit
        
        # Current time in simulation
        self.current_time = 0.0
        
        # Cache for time-dependent parameters
        self._parameter_cache = {}
        self._cache_time = -1.0
        
        # Create initial drift channels
        self._create_channels()
    
    def _create_channels(self) -> None:
        """Create drift noise channels (time-dependent)."""
        # Channels are created dynamically based on current time
        # This method sets up the framework
        
        for qubit in range(self.num_qubits):
            # Frequency drift channel
            if qubit in self.frequency_drift:
                channel = NoiseChannel(
                    name=f"frequency_drift_q{qubit}",
                    kraus_operators=[jnp.eye(2, dtype=jnp.complex64)],  # Placeholder
                    probability=1.0,
                    qubits=[qubit]
                )
                self.add_channel(channel)
            
            # Amplitude drift channel
            if qubit in self.amplitude_drift:
                channel = NoiseChannel(
                    name=f"amplitude_drift_q{qubit}",
                    kraus_operators=[jnp.eye(2, dtype=jnp.complex64)],  # Placeholder
                    probability=1.0,
                    qubits=[qubit]
                )
                self.add_channel(channel)
            
            # Phase drift channel
            if qubit in self.phase_drift:
                channel = NoiseChannel(
                    name=f"phase_drift_q{qubit}",
                    kraus_operators=[jnp.eye(2, dtype=jnp.complex64)],  # Placeholder
                    probability=1.0,
                    qubits=[qubit]
                )
                self.add_channel(channel)
            
            # T1 drift affects amplitude damping
            if qubit in self.t1_drift:
                channel = NoiseChannel(
                    name=f"t1_drift_q{qubit}",
                    kraus_operators=[jnp.eye(2, dtype=jnp.complex64)],  # Placeholder
                    probability=1.0,
                    qubits=[qubit]
                )
                self.add_channel(channel)
            
            # T2 drift affects dephasing
            if qubit in self.t2_drift:
                channel = NoiseChannel(
                    name=f"t2_drift_q{qubit}",
                    kraus_operators=[jnp.eye(2, dtype=jnp.complex64)],  # Placeholder
                    probability=1.0,
                    qubits=[qubit]
                )
                self.add_channel(channel)
    
    def _update_channels_for_time(self, time: float) -> None:
        """Update channel Kraus operators for current time."""
        if abs(time - self._cache_time) < 1e-9:
            return  # Already cached
        
        self._cache_time = time
        self._parameter_cache.clear()
        
        for qubit in range(self.num_qubits):
            # Update frequency drift channel
            if qubit in self.frequency_drift:
                freq_shift = self.frequency_drift[qubit](time)
                kraus_ops = self._create_frequency_drift_kraus(freq_shift)
                
                channel_name = f"frequency_drift_q{qubit}"
                if channel_name in self.channels:
                    self.channels[channel_name].kraus_operators = kraus_ops
            
            # Update amplitude drift channel
            if qubit in self.amplitude_drift:
                amp_factor = self.amplitude_drift[qubit](time)
                kraus_ops = self._create_amplitude_drift_kraus(amp_factor)
                
                channel_name = f"amplitude_drift_q{qubit}"
                if channel_name in self.channels:
                    self.channels[channel_name].kraus_operators = kraus_ops
            
            # Update phase drift channel
            if qubit in self.phase_drift:
                phase_shift = self.phase_drift[qubit](time)
                kraus_ops = self._create_phase_drift_kraus(phase_shift)
                
                channel_name = f"phase_drift_q{qubit}"
                if channel_name in self.channels:
                    self.channels[channel_name].kraus_operators = kraus_ops
            
            # Update T1 drift channel
            if qubit in self.t1_drift:
                t1_time = self.t1_drift[qubit](time)
                kraus_ops = self._create_t1_drift_kraus(t1_time)
                
                channel_name = f"t1_drift_q{qubit}"
                if channel_name in self.channels:
                    self.channels[channel_name].kraus_operators = kraus_ops
            
            # Update T2 drift channel
            if qubit in self.t2_drift:
                t2_time = self.t2_drift[qubit](time)
                kraus_ops = self._create_t2_drift_kraus(t2_time)
                
                channel_name = f"t2_drift_q{qubit}"
                if channel_name in self.channels:
                    self.channels[channel_name].kraus_operators = kraus_ops
    
    def _create_frequency_drift_kraus(self, freq_shift: float) -> List[jnp.ndarray]:
        """
        Create Kraus operators for frequency drift.
        
        Args:
            freq_shift: Frequency shift in Hz
            
        Returns:
            List of Kraus operators
        """
        # Typical gate time
        gate_time = 0.1e-6  # 100 ns
        
        # Z rotation due to frequency shift
        theta = 2 * np.pi * freq_shift * gate_time
        
        drift_matrix = jnp.array([
            [jnp.exp(-1j * theta / 2), 0],
            [0, jnp.exp(1j * theta / 2)]
        ], dtype=jnp.complex64)
        
        return [drift_matrix]
    
    def _create_amplitude_drift_kraus(self, amp_factor: float) -> List[jnp.ndarray]:
        """
        Create Kraus operators for amplitude drift.
        
        Args:
            amp_factor: Amplitude scaling factor
            
        Returns:
            List of Kraus operators
        """
        # Amplitude scaling affects gate strength
        # Model as scaling of rotation angle
        
        drift_matrix = jnp.sqrt(amp_factor) * jnp.eye(2, dtype=jnp.complex64)
        
        return [drift_matrix]
    
    def _create_phase_drift_kraus(self, phase_shift: float) -> List[jnp.ndarray]:
        """
        Create Kraus operators for phase drift.
        
        Args:
            phase_shift: Phase shift in radians
            
        Returns:
            List of Kraus operators
        """
        # Global phase shift
        drift_matrix = jnp.exp(1j * phase_shift) * jnp.eye(2, dtype=jnp.complex64)
        
        return [drift_matrix]
    
    def _create_t1_drift_kraus(self, t1_time: float) -> List[jnp.ndarray]:
        """
        Create Kraus operators for T1 drift.
        
        Args:
            t1_time: Current T1 time
            
        Returns:
            List of Kraus operators
        """
        # Model as time-varying amplitude damping
        gate_time = 0.1e-6  # 100 ns
        gamma = 1 - jnp.exp(-gate_time / (t1_time * 1e-6)) if t1_time > 0 else 0.0
        
        K0 = jnp.array([
            [1, 0],
            [0, jnp.sqrt(1 - gamma)]
        ], dtype=jnp.complex64)
        
        K1 = jnp.array([
            [0, jnp.sqrt(gamma)],
            [0, 0]
        ], dtype=jnp.complex64)
        
        return [K0, K1]
    
    def _create_t2_drift_kraus(self, t2_time: float) -> List[jnp.ndarray]:
        """
        Create Kraus operators for T2 drift.
        
        Args:
            t2_time: Current T2 time
            
        Returns:
            List of Kraus operators
        """
        # Model as time-varying phase damping
        gate_time = 0.1e-6  # 100 ns
        gamma = 1 - jnp.exp(-gate_time / (t2_time * 1e-6)) if t2_time > 0 else 0.0
        
        K0 = jnp.array([
            [1, 0],
            [0, jnp.sqrt(1 - gamma)]
        ], dtype=jnp.complex64)
        
        K1 = jnp.array([
            [0, 0],
            [0, jnp.sqrt(gamma)]
        ], dtype=jnp.complex64)
        
        return [K0, K1]
    
    def set_time(self, time: float) -> None:
        """
        Set current simulation time and update drift parameters.
        
        Args:
            time: Current time in specified time unit
        """
        self.current_time = time
        self._update_channels_for_time(time)
    
    def advance_time(self, dt: float) -> None:
        """
        Advance simulation time by dt.
        
        Args:
            dt: Time increment in specified time unit
        """
        self.set_time(self.current_time + dt)
    
    def get_noise_channels(self, circuit: Any) -> List[NoiseChannel]:
        """
        Get drift noise channels for circuit gates.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            List of noise channels to apply
        """
        channels = []
        
        # Update channels for current time
        self._update_channels_for_time(self.current_time)
        
        if hasattr(circuit, 'gates'):
            for gate_idx, gate in enumerate(circuit.gates):
                qubits = gate.get("qubits", [])
                
                # Estimate gate execution time and advance time
                gate_time = self._estimate_gate_time(gate)
                gate_execution_time = self.current_time + gate_time
                
                # Apply drift at gate execution time
                for qubit in qubits:
                    # Frequency drift
                    if f"frequency_drift_q{qubit}" in self.channels:
                        self._update_channels_for_time(gate_execution_time)
                        channel = self.channels[f"frequency_drift_q{qubit}"]
                        drift_channel = NoiseChannel(
                            name=f"{channel.name}_gate_{gate_idx}",
                            kraus_operators=channel.kraus_operators,
                            probability=channel.probability,
                            qubits=[qubit]
                        )
                        channels.append(drift_channel)
                    
                    # Amplitude drift
                    if f"amplitude_drift_q{qubit}" in self.channels:
                        channel = self.channels[f"amplitude_drift_q{qubit}"]
                        drift_channel = NoiseChannel(
                            name=f"{channel.name}_gate_{gate_idx}",
                            kraus_operators=channel.kraus_operators,
                            probability=channel.probability,
                            qubits=[qubit]
                        )
                        channels.append(drift_channel)
                    
                    # Phase drift
                    if f"phase_drift_q{qubit}" in self.channels:
                        channel = self.channels[f"phase_drift_q{qubit}"]
                        drift_channel = NoiseChannel(
                            name=f"{channel.name}_gate_{gate_idx}",
                            kraus_operators=channel.kraus_operators,
                            probability=channel.probability,
                            qubits=[qubit]
                        )
                        channels.append(drift_channel)
                    
                    # T1 drift
                    if f"t1_drift_q{qubit}" in self.channels:
                        channel = self.channels[f"t1_drift_q{qubit}"]
                        drift_channel = NoiseChannel(
                            name=f"{channel.name}_gate_{gate_idx}",
                            kraus_operators=channel.kraus_operators,
                            probability=channel.probability,
                            qubits=[qubit]
                        )
                        channels.append(drift_channel)
                    
                    # T2 drift
                    if f"t2_drift_q{qubit}" in self.channels:
                        channel = self.channels[f"t2_drift_q{qubit}"]
                        drift_channel = NoiseChannel(
                            name=f"{channel.name}_gate_{gate_idx}",
                            kraus_operators=channel.kraus_operators,
                            probability=channel.probability,
                            qubits=[qubit]
                        )
                        channels.append(drift_channel)
                
                # Advance time after gate
                self.advance_time(gate_time)
        
        return channels
    
    def _estimate_gate_time(self, gate: Dict[str, Any]) -> float:
        """
        Estimate gate execution time.
        
        Args:
            gate: Gate dictionary
            
        Returns:
            Gate time in current time unit
        """
        gate_type = gate.get("type", "single")
        gate_name = gate.get("name", "unknown")
        
        # Default gate times (in microseconds)
        if gate_type == "single":
            return 0.1  # 100 ns
        elif gate_type == "two":
            return 0.5  # 500 ns
        elif gate_name == "readout":
            return 1.0  # 1 μs
        else:
            return 0.1
    
    def apply_noise(self, circuit: Any) -> Any:
        """
        Apply drift noise to circuit.
        
        Args:
            circuit: Clean quantum circuit
            
        Returns:
            Circuit with drift noise applied
        """
        if hasattr(circuit, 'copy'):
            noisy_circuit = circuit.copy()
        else:
            import copy
            noisy_circuit = copy.deepcopy(circuit)
        
        # Reset time for circuit execution
        self.set_time(0.0)
        
        # Add noise markers for simulation
        noisy_circuit._noise_model = self
        noisy_circuit._noise_channels = self.get_noise_channels(circuit)
        
        return noisy_circuit
    
    def get_parameter_at_time(self, qubit: int, parameter: str, time: float) -> float:
        """
        Get drifted parameter value at specific time.
        
        Args:
            qubit: Qubit index
            parameter: Parameter name ("frequency", "amplitude", "phase", "t1", "t2")
            time: Time value
            
        Returns:
            Parameter value at specified time
        """
        if parameter == "frequency" and qubit in self.frequency_drift:
            return self.frequency_drift[qubit](time)
        elif parameter == "amplitude" and qubit in self.amplitude_drift:
            return self.amplitude_drift[qubit](time)
        elif parameter == "phase" and qubit in self.phase_drift:
            return self.phase_drift[qubit](time)
        elif parameter == "t1" and qubit in self.t1_drift:
            return self.t1_drift[qubit](time)
        elif parameter == "t2" and qubit in self.t2_drift:
            return self.t2_drift[qubit](time)
        else:
            return 0.0  # No drift
    
    def scale_noise(self, factor: float) -> "DriftNoiseModel":
        """
        Scale drift noise by factor.
        
        Args:
            factor: Scaling factor
            
        Returns:
            New noise model with scaled drift
        """
        # Scale drift functions
        scaled_freq_drift = {
            qubit: lambda t, orig_func=func: orig_func(t) * factor
            for qubit, func in self.frequency_drift.items()
        }
        scaled_amp_drift = {
            qubit: lambda t, orig_func=func: 1 + (orig_func(t) - 1) * factor
            for qubit, func in self.amplitude_drift.items()
        }
        scaled_phase_drift = {
            qubit: lambda t, orig_func=func: orig_func(t) * factor
            for qubit, func in self.phase_drift.items()
        }
        # T1/T2 drift scaling: shorter times = more noise
        scaled_t1_drift = {
            qubit: lambda t, orig_func=func: orig_func(t) / factor
            for qubit, func in self.t1_drift.items()
        }
        scaled_t2_drift = {
            qubit: lambda t, orig_func=func: orig_func(t) / factor
            for qubit, func in self.t2_drift.items()
        }
        
        return DriftNoiseModel(
            num_qubits=self.num_qubits,
            frequency_drift=scaled_freq_drift,
            amplitude_drift=scaled_amp_drift,
            phase_drift=scaled_phase_drift,
            t1_drift=scaled_t1_drift,
            t2_drift=scaled_t2_drift,
            time_unit=self.time_unit
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        # Note: Cannot serialize functions, so this is limited
        base_dict = super().to_dict()
        base_dict.update({
            "num_qubits": self.num_qubits,
            "time_unit": self.time_unit,
            "current_time": float(self.current_time),
            "has_frequency_drift": list(self.frequency_drift.keys()),
            "has_amplitude_drift": list(self.amplitude_drift.keys()),
            "has_phase_drift": list(self.phase_drift.keys()),
            "has_t1_drift": list(self.t1_drift.keys()),
            "has_t2_drift": list(self.t2_drift.keys())
        })
        return base_dict
    
    def __str__(self) -> str:
        """String representation."""
        lines = [f"DriftNoiseModel"]
        lines.append(f"Qubits: {self.num_qubits}")
        lines.append(f"Current time: {self.current_time} {self.time_unit}")
        lines.append(f"Frequency drift: {len(self.frequency_drift)} qubits")
        lines.append(f"Amplitude drift: {len(self.amplitude_drift)} qubits")
        lines.append(f"Phase drift: {len(self.phase_drift)} qubits")
        lines.append(f"T1 drift: {len(self.t1_drift)} qubits")
        lines.append(f"T2 drift: {len(self.t2_drift)} qubits")
        lines.append(f"Channels: {len(self.channels)}")
        return "\n".join(lines)


# Predefined drift functions
class DriftFunctions:
    """Collection of common drift functions."""
    
    @staticmethod
    def linear_drift(rate: float, offset: float = 0.0) -> Callable[[float], float]:
        """Linear drift: f(t) = offset + rate * t"""
        return lambda t: offset + rate * t
    
    @staticmethod
    def exponential_drift(decay_constant: float, amplitude: float = 1.0) -> Callable[[float], float]:
        """Exponential drift: f(t) = amplitude * exp(-t / decay_constant)"""
        return lambda t: amplitude * np.exp(-t / decay_constant)
    
    @staticmethod
    def sinusoidal_drift(frequency: float, amplitude: float = 1.0, phase: float = 0.0) -> Callable[[float], float]:
        """Sinusoidal drift: f(t) = amplitude * sin(2π * frequency * t + phase)"""
        return lambda t: amplitude * np.sin(2 * np.pi * frequency * t + phase)
    
    @staticmethod
    def random_walk_drift(step_size: float, seed: int = 42) -> Callable[[float], float]:
        """Random walk drift with given step size."""
        np.random.seed(seed)
        steps = np.random.normal(0, step_size, 10000)  # Pre-generate steps
        
        def drift_func(t):
            index = int(t * 10)  # 10 steps per time unit
            if index >= len(steps):
                index = len(steps) - 1
            return np.cumsum(steps[:index+1])[-1] if index >= 0 else 0.0
        
        return drift_func
    
    @staticmethod
    def polynomial_drift(coefficients: List[float]) -> Callable[[float], float]:
        """Polynomial drift: f(t) = sum(c_i * t^i)"""
        return lambda t: sum(c * (t ** i) for i, c in enumerate(coefficients))
    
    @staticmethod
    def temperature_drift(temp_coeff: float, base_temp: float = 300.0) -> Callable[[float], float]:
        """Temperature-dependent drift with assumed temperature variation."""
        # Assume temperature varies sinusoidally over long time scales
        temp_variation = lambda t: base_temp + 5 * np.sin(2 * np.pi * t / 3600)  # 1-hour cycle
        return lambda t: temp_coeff * (temp_variation(t) - base_temp)


class AgingDriftModel(DriftNoiseModel):
    """
    Drift model for long-term device aging effects.
    
    Models systematic parameter changes over device lifetime.
    """
    
    def __init__(
        self,
        num_qubits: int,
        aging_rate: float = 1e-6,  # Per hour
        initial_params: Optional[Dict[str, Dict[int, float]]] = None
    ):
        """
        Initialize aging drift model.
        
        Args:
            num_qubits: Number of qubits
            aging_rate: Rate of parameter degradation per hour
            initial_params: Initial parameter values
        """
        # Create aging drift functions
        frequency_drift = {
            i: DriftFunctions.linear_drift(-aging_rate * 1e6, 0.0)  # Hz
            for i in range(num_qubits)
        }
        
        t1_drift = {
            i: DriftFunctions.exponential_drift(1000.0, 100.0)  # μs, 1000 hour decay
            for i in range(num_qubits)
        }
        
        t2_drift = {
            i: DriftFunctions.exponential_drift(1000.0, 50.0)  # μs, 1000 hour decay
            for i in range(num_qubits)
        }
        
        super().__init__(
            num_qubits=num_qubits,
            frequency_drift=frequency_drift,
            t1_drift=t1_drift,
            t2_drift=t2_drift,
            time_unit="hours"
        )
        
        self.name = "aging_drift"
        self.aging_rate = aging_rate