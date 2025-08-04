"""Noise replayer for reproducible noise simulation."""

import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import time

from ..models.base import NoiseModel, NoiseChannel
from .recorder import NoiseTrace, NoiseRecord


@dataclass
class ReplayConfig:
    """Configuration for noise replay."""
    
    # Replay mode
    exact_replay: bool = True  # If False, use statistical replay
    interpolate_missing: bool = True  # Interpolate noise between recorded events
    scale_time: float = 1.0  # Time scaling factor
    
    # Filtering options
    filter_qubits: Optional[List[int]] = None  # Only replay noise on these qubits
    filter_gates: Optional[List[str]] = None  # Only replay noise for these gates
    filter_error_types: Optional[List[str]] = None  # Only replay these error types
    
    # Statistical options (for non-exact replay)
    preserve_correlations: bool = True  # Preserve temporal/spatial correlations
    randomize_phases: bool = False  # Randomize error phases while preserving magnitudes
    
    # Environmental replay
    include_environmental: bool = True  # Include environmental effects
    environmental_scaling: float = 1.0  # Scale environmental effects


class NoiseReplayer(NoiseModel):
    """
    Replays recorded noise patterns for reproducible simulation.
    
    Can replay noise either exactly as recorded or statistically
    while preserving key characteristics and correlations.
    """
    
    def __init__(
        self,
        noise_trace: NoiseTrace,
        config: Optional[ReplayConfig] = None
    ):
        """
        Initialize noise replayer.
        
        Args:
            noise_trace: Recorded noise trace to replay
            config: Replay configuration
        """
        super().__init__(f"replayer_{noise_trace.device_name}")
        
        self.noise_trace = noise_trace
        self.config = config or ReplayConfig()
        
        # Current replay state
        self.replay_time = 0.0
        self.circuit_index = 0
        self.gate_index = 0
        
        # Indexed records for fast lookup
        self._build_lookup_tables()
        
        # Statistical models for non-exact replay
        self._error_models = {}
        if not self.config.exact_replay:
            self._build_statistical_models()
        
        # Create replay channels
        self._create_replay_channels()
    
    def _build_lookup_tables(self) -> None:
        """Build lookup tables for efficient noise record access."""
        self.records_by_circuit = {}
        self.records_by_qubit = {}
        self.records_by_gate = {}
        self.records_by_time = {}
        
        for record in self.noise_trace.records:
            # By circuit
            circuit_id = record.circuit_id
            if circuit_id not in self.records_by_circuit:
                self.records_by_circuit[circuit_id] = []
            self.records_by_circuit[circuit_id].append(record)
            
            # By qubit
            for qubit in record.qubit_indices:
                if qubit not in self.records_by_qubit:
                    self.records_by_qubit[qubit] = []
                self.records_by_qubit[qubit].append(record)
            
            # By gate
            gate_key = f"{record.gate_name}_{record.gate_type}"
            if gate_key not in self.records_by_gate:
                self.records_by_gate[gate_key] = []
            self.records_by_gate[gate_key].append(record)
            
            # By time (binned)
            time_bin = int(record.timestamp)
            if time_bin not in self.records_by_time:
                self.records_by_time[time_bin] = []
            self.records_by_time[time_bin].append(record)
    
    def _build_statistical_models(self) -> None:
        """Build statistical models for non-exact replay."""
        # Error rate models per qubit/gate combination
        for qubit in self.records_by_qubit:
            qubit_records = self.records_by_qubit[qubit]
            
            # Group by gate type
            gate_stats = {}
            for record in qubit_records:
                gate_key = f"{record.gate_name}_{record.gate_type}"
                if gate_key not in gate_stats:
                    gate_stats[gate_key] = {
                        "total": 0,
                        "errors": 0,
                        "error_types": {},
                        "magnitudes": [],
                        "phases": []
                    }
                
                stats = gate_stats[gate_key]
                stats["total"] += 1
                
                if record.error_occurred:
                    stats["errors"] += 1
                    error_type = record.error_type
                    stats["error_types"][error_type] = stats["error_types"].get(error_type, 0) + 1
                    stats["magnitudes"].append(record.error_magnitude)
                    stats["phases"].append(record.error_phase)
            
            # Convert to statistical models
            qubit_models = {}
            for gate_key, stats in gate_stats.items():
                if stats["total"] > 0:
                    error_rate = stats["errors"] / stats["total"]
                    
                    # Error type distribution
                    error_type_probs = {}
                    if stats["error_types"]:
                        total_errors = sum(stats["error_types"].values())
                        for error_type, count in stats["error_types"].items():
                            error_type_probs[error_type] = count / total_errors
                    
                    # Magnitude and phase distributions
                    magnitude_mean = np.mean(stats["magnitudes"]) if stats["magnitudes"] else 0.0
                    magnitude_std = np.std(stats["magnitudes"]) if len(stats["magnitudes"]) > 1 else 0.01
                    
                    phase_mean = np.mean(stats["phases"]) if stats["phases"] else 0.0
                    phase_std = np.std(stats["phases"]) if len(stats["phases"]) > 1 else 0.1
                    
                    qubit_models[gate_key] = {
                        "error_rate": error_rate,
                        "error_type_probs": error_type_probs,
                        "magnitude_mean": magnitude_mean,
                        "magnitude_std": magnitude_std,
                        "phase_mean": phase_mean,
                        "phase_std": phase_std
                    }
            
            self._error_models[qubit] = qubit_models
    
    def _create_replay_channels(self) -> None:
        """Create noise channels for replay."""
        # Create channels for each unique error type and qubit combination
        error_types = set()
        qubits = set()
        
        for record in self.noise_trace.records:
            if record.error_occurred:
                error_types.add(record.error_type)
                qubits.update(record.qubit_indices)
        
        # Create a channel for each error type on each qubit
        for qubit in qubits:
            for error_type in error_types:
                channel_name = f"replay_{error_type}_q{qubit}"
                
                # Create placeholder Kraus operators (will be updated during replay)
                kraus_ops = [jnp.eye(2, dtype=jnp.complex64)]
                
                channel = NoiseChannel(
                    name=channel_name,
                    kraus_operators=kraus_ops,
                    probability=1.0,
                    qubits=[qubit]
                )
                self.add_channel(channel)
    
    def reset_replay(self) -> None:
        """Reset replay state to beginning."""
        self.replay_time = 0.0
        self.circuit_index = 0
        self.gate_index = 0
    
    def get_noise_channels(self, circuit: Any) -> List[NoiseChannel]:
        """
        Get noise channels for circuit based on recorded trace.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            List of noise channels to apply
        """
        channels = []
        circuit_id = f"circuit_{self.circuit_index}"
        
        if hasattr(circuit, 'gates'):
            for gate_idx, gate in enumerate(circuit.gates):
                gate_name = gate.get("name", "unknown")
                gate_type = gate.get("type", "single")
                qubits = gate.get("qubits", [])
                
                # Get noise events for this gate
                if self.config.exact_replay:
                    gate_channels = self._get_exact_replay_channels(
                        circuit_id, gate_idx, gate_name, gate_type, qubits
                    )
                else:
                    gate_channels = self._get_statistical_replay_channels(
                        gate_name, gate_type, qubits
                    )
                
                channels.extend(gate_channels)
        
        self.circuit_index += 1
        return channels
    
    def _get_exact_replay_channels(
        self,
        circuit_id: str,
        gate_idx: int,
        gate_name: str,
        gate_type: str,
        qubits: List[int]
    ) -> List[NoiseChannel]:
        """Get noise channels for exact replay."""
        channels = []
        
        # Look for matching records
        matching_records = []
        
        # First, try to find records for this specific circuit and gate
        if circuit_id in self.records_by_circuit:
            circuit_records = self.records_by_circuit[circuit_id]
            for record in circuit_records:
                if (record.gate_index == gate_idx and
                    record.gate_name == gate_name and
                    set(record.qubit_indices) == set(qubits)):
                    matching_records.append(record)
        
        # If no exact match, look for similar gates
        if not matching_records and self.config.interpolate_missing:
            gate_key = f"{gate_name}_{gate_type}"
            if gate_key in self.records_by_gate:
                # Find records with overlapping qubits
                for record in self.records_by_gate[gate_key]:
                    if any(q in qubits for q in record.qubit_indices):
                        matching_records.append(record)
        
        # Create channels from matching records
        for record in matching_records:
            if record.error_occurred:
                # Apply filtering
                if self.config.filter_qubits and not any(q in self.config.filter_qubits for q in record.qubit_indices):
                    continue
                if self.config.filter_gates and record.gate_name not in self.config.filter_gates:
                    continue
                if self.config.filter_error_types and record.error_type not in self.config.filter_error_types:
                    continue
                
                # Create Kraus operators for this specific error
                kraus_ops = self._create_error_kraus_operators(record)
                
                channel = NoiseChannel(
                    name=f"replay_{record.error_type}_gate_{gate_idx}",
                    kraus_operators=kraus_ops,
                    probability=1.0,
                    qubits=record.qubit_indices
                )
                channels.append(channel)
        
        return channels
    
    def _get_statistical_replay_channels(
        self,
        gate_name: str,
        gate_type: str,
        qubits: List[int]
    ) -> List[NoiseChannel]:
        """Get noise channels for statistical replay."""
        channels = []
        gate_key = f"{gate_name}_{gate_type}"
        
        for qubit in qubits:
            # Apply filtering
            if self.config.filter_qubits and qubit not in self.config.filter_qubits:
                continue
            
            if qubit in self._error_models and gate_key in self._error_models[qubit]:
                model = self._error_models[qubit][gate_key]
                
                # Decide if error occurs
                if np.random.random() < model["error_rate"]:
                    # Choose error type
                    error_types = list(model["error_type_probs"].keys())
                    if error_types:
                        probs = list(model["error_type_probs"].values())
                        error_type = np.random.choice(error_types, p=probs)
                        
                        # Apply filtering
                        if self.config.filter_error_types and error_type not in self.config.filter_error_types:
                            continue
                        
                        # Sample error characteristics
                        magnitude = np.random.normal(model["magnitude_mean"], model["magnitude_std"])
                        magnitude = max(0, min(1, magnitude))  # Clamp to [0, 1]
                        
                        if self.config.randomize_phases:
                            phase = np.random.uniform(0, 2 * np.pi)
                        else:
                            phase = np.random.normal(model["phase_mean"], model["phase_std"])
                        
                        # Create synthetic record
                        synthetic_record = NoiseRecord(
                            timestamp=self.replay_time,
                            circuit_id=f"synthetic_{self.circuit_index}",
                            gate_index=self.gate_index,
                            qubit_indices=[qubit],
                            gate_name=gate_name,
                            gate_type=gate_type,
                            error_occurred=True,
                            error_type=error_type,
                            error_magnitude=magnitude,
                            error_phase=phase
                        )
                        
                        # Create Kraus operators
                        kraus_ops = self._create_error_kraus_operators(synthetic_record)
                        
                        channel = NoiseChannel(
                            name=f"statistical_replay_{error_type}_q{qubit}",
                            kraus_operators=kraus_ops,
                            probability=1.0,
                            qubits=[qubit]
                        )
                        channels.append(channel)
        
        return channels
    
    def _create_error_kraus_operators(self, record: NoiseRecord) -> List[jnp.ndarray]:
        """
        Create Kraus operators for a specific error record.
        
        Args:
            record: Noise record containing error information
            
        Returns:
            List of Kraus operators
        """
        error_type = record.error_type
        magnitude = record.error_magnitude
        phase = record.error_phase
        
        if error_type == "bit_flip":
            # Bit flip error: X rotation
            theta = magnitude * np.pi
            cos_half = jnp.cos(theta / 2)
            sin_half = jnp.sin(theta / 2)
            
            K0 = cos_half * jnp.array([[1, 0], [0, 1]], dtype=jnp.complex64)
            K1 = sin_half * jnp.exp(1j * phase) * jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
            
            return [K0, K1]
        
        elif error_type == "phase_flip":
            # Phase flip error: Z rotation
            theta = magnitude * np.pi
            cos_half = jnp.cos(theta / 2)
            sin_half = jnp.sin(theta / 2)
            
            K0 = cos_half * jnp.array([[1, 0], [0, 1]], dtype=jnp.complex64)
            K1 = sin_half * jnp.exp(1j * phase) * jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
            
            return [K0, K1]
        
        elif error_type == "amplitude_error":
            # Amplitude error: scaling
            amplitude_factor = 1.0 + magnitude * (np.cos(phase))
            amplitude_factor = max(0.1, min(2.0, amplitude_factor))
            
            K = jnp.sqrt(amplitude_factor) * jnp.array([[1, 0], [0, 1]], dtype=jnp.complex64)
            return [K]
        
        elif error_type == "decoherence":
            # Decoherence: amplitude damping
            gamma = magnitude
            
            K0 = jnp.array([[1, 0], [0, jnp.sqrt(1 - gamma)]], dtype=jnp.complex64)
            K1 = jnp.array([[0, jnp.sqrt(gamma)], [0, 0]], dtype=jnp.complex64)
            
            return [K0, K1]
        
        elif error_type == "crosstalk":
            # Crosstalk: two-qubit error (simplified to single-qubit for now)
            theta = magnitude * 0.1  # Small crosstalk angle
            
            cos_half = jnp.cos(theta / 2)
            sin_half = jnp.sin(theta / 2)
            
            K0 = cos_half * jnp.array([[1, 0], [0, 1]], dtype=jnp.complex64)
            K1 = sin_half * jnp.exp(1j * phase) * jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
            
            return [K0, K1]
        
        else:
            # Unknown error type: default to depolarizing
            p = magnitude * 0.1  # Small depolarizing probability
            
            from ..models.base import depolarizing_kraus_operators
            return depolarizing_kraus_operators(p)
    
    def apply_noise(self, circuit: Any) -> Any:
        """
        Apply replayed noise to circuit.
        
        Args:
            circuit: Clean quantum circuit
            
        Returns:
            Circuit with replayed noise applied
        """
        if hasattr(circuit, 'copy'):
            noisy_circuit = circuit.copy()
        else:
            import copy
            noisy_circuit = copy.deepcopy(circuit)
        
        # Add noise markers for simulation
        noisy_circuit._noise_model = self
        noisy_circuit._noise_channels = self.get_noise_channels(circuit)
        
        # Update replay time
        self.replay_time += self._estimate_circuit_time(circuit)
        
        return noisy_circuit
    
    def _estimate_circuit_time(self, circuit: Any) -> float:
        """Estimate circuit execution time."""
        if hasattr(circuit, 'gates'):
            num_gates = len(circuit.gates)
            # Rough estimate: 100ns per single gate, 500ns per two-qubit gate
            total_time = 0.0
            for gate in circuit.gates:
                gate_type = gate.get("type", "single")
                if gate_type == "single":
                    total_time += 100e-9  # 100 ns
                elif gate_type == "two":
                    total_time += 500e-9  # 500 ns
                else:
                    total_time += 200e-9  # Default
            return total_time * self.config.scale_time
        else:
            return 1e-6  # Default 1 μs
    
    def get_replay_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the replay process.
        
        Returns:
            Dictionary with replay statistics
        """
        total_records = len(self.noise_trace.records)
        error_records = [r for r in self.noise_trace.records if r.error_occurred]
        
        stats = {
            "trace_info": {
                "device_name": self.noise_trace.device_name,
                "session_id": self.noise_trace.session_id,
                "duration": self.noise_trace.end_time - self.noise_trace.start_time,
                "num_circuits": self.noise_trace.num_circuits
            },
            "record_counts": {
                "total_records": total_records,
                "error_records": len(error_records),
                "error_rate": len(error_records) / total_records if total_records > 0 else 0
            },
            "replay_config": {
                "exact_replay": self.config.exact_replay,
                "interpolate_missing": self.config.interpolate_missing,
                "scale_time": self.config.scale_time,
                "filter_qubits": self.config.filter_qubits,
                "filter_gates": self.config.filter_gates,
                "filter_error_types": self.config.filter_error_types
            },
            "replay_state": {
                "current_time": self.replay_time,
                "circuit_index": self.circuit_index,
                "gate_index": self.gate_index
            }
        }
        
        # Error type distribution
        if error_records:
            error_types = {}
            for record in error_records:
                error_type = record.error_type
                error_types[error_type] = error_types.get(error_type, 0) + 1
            stats["error_distribution"] = error_types
        
        # Qubit usage
        qubit_counts = {}
        for record in self.noise_trace.records:
            for qubit in record.qubit_indices:
                qubit_counts[qubit] = qubit_counts.get(qubit, 0) + 1
        stats["qubit_usage"] = qubit_counts
        
        return stats
    
    def create_filtered_replayer(
        self,
        filter_config: Dict[str, Any]
    ) -> "NoiseReplayer":
        """
        Create a new replayer with additional filtering.
        
        Args:
            filter_config: Filtering configuration
            
        Returns:
            New filtered replayer
        """
        new_config = ReplayConfig(
            exact_replay=self.config.exact_replay,
            interpolate_missing=self.config.interpolate_missing,
            scale_time=self.config.scale_time,
            filter_qubits=filter_config.get("qubits", self.config.filter_qubits),
            filter_gates=filter_config.get("gates", self.config.filter_gates),
            filter_error_types=filter_config.get("error_types", self.config.filter_error_types),
            preserve_correlations=self.config.preserve_correlations,
            randomize_phases=filter_config.get("randomize_phases", self.config.randomize_phases),
            include_environmental=self.config.include_environmental,
            environmental_scaling=filter_config.get("environmental_scaling", self.config.environmental_scaling)
        )
        
        return NoiseReplayer(self.noise_trace, new_config)
    
    def scale_noise(self, factor: float) -> "NoiseReplayer":
        """
        Scale replayed noise by factor.
        
        Args:
            factor: Scaling factor
            
        Returns:
            New replayer with scaled noise
        """
        # Create scaled trace by modifying error magnitudes
        scaled_records = []
        for record in self.noise_trace.records:
            if record.error_occurred:
                scaled_record = NoiseRecord(
                    timestamp=record.timestamp,
                    circuit_id=record.circuit_id,
                    gate_index=record.gate_index,
                    qubit_indices=record.qubit_indices,
                    gate_name=record.gate_name,
                    gate_type=record.gate_type,
                    error_occurred=record.error_occurred,
                    error_type=record.error_type,
                    error_magnitude=min(1.0, record.error_magnitude * factor),
                    error_phase=record.error_phase,
                    temperature=record.temperature,
                    magnetic_field=record.magnetic_field,
                    control_amplitudes=record.control_amplitudes,
                    frequencies=record.frequencies,
                    expected_outcome=record.expected_outcome,
                    actual_outcome=record.actual_outcome,
                    measurement_fidelity=record.measurement_fidelity
                )
                scaled_records.append(scaled_record)
            else:
                scaled_records.append(record)
        
        # Create scaled trace
        from .recorder import NoiseTrace
        scaled_trace = NoiseTrace(
            device_name=self.noise_trace.device_name,
            session_id=f"{self.noise_trace.session_id}_scaled_{factor}",
            start_time=self.noise_trace.start_time,
            end_time=self.noise_trace.end_time,
            num_circuits=self.noise_trace.num_circuits,
            records=scaled_records,
            device_parameters=self.noise_trace.device_parameters,
            environmental_conditions=self.noise_trace.environmental_conditions,
            calibration_data=self.noise_trace.calibration_data
        )
        
        return NoiseReplayer(scaled_trace, self.config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            "trace_session_id": self.noise_trace.session_id,
            "trace_device": self.noise_trace.device_name,
            "replay_config": {
                "exact_replay": self.config.exact_replay,
                "interpolate_missing": self.config.interpolate_missing,
                "scale_time": self.config.scale_time,
                "filter_qubits": self.config.filter_qubits,
                "filter_gates": self.config.filter_gates,
                "filter_error_types": self.config.filter_error_types,
                "preserve_correlations": self.config.preserve_correlations,
                "randomize_phases": self.config.randomize_phases,
                "include_environmental": self.config.include_environmental,
                "environmental_scaling": self.config.environmental_scaling
            }
        })
        return base_dict
    
    def __str__(self) -> str:
        """String representation."""
        mode = "Exact" if self.config.exact_replay else "Statistical"
        return f"NoiseReplayer({mode}, {self.noise_trace.device_name}, {len(self.noise_trace.records)} events)"