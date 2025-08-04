"""Noise recorder for capturing real device noise patterns."""

import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import time
import pickle
import json
from pathlib import Path

from ..models.base import NoiseModel, NoiseChannel


@dataclass
class NoiseRecord:
    """Single noise event record."""
    
    timestamp: float
    circuit_id: str
    gate_index: int
    qubit_indices: List[int]
    gate_name: str
    gate_type: str
    
    # Noise characteristics
    error_occurred: bool
    error_type: str
    error_magnitude: float
    error_phase: float
    
    # Environmental conditions
    temperature: Optional[float] = None
    magnetic_field: Optional[float] = None
    control_amplitudes: Optional[Dict[str, float]] = None
    frequencies: Optional[Dict[int, float]] = None
    
    # Measurement outcomes (if available)
    expected_outcome: Optional[str] = None
    actual_outcome: Optional[str] = None
    measurement_fidelity: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NoiseRecord":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class NoiseTrace:
    """Collection of noise records forming a trace."""
    
    device_name: str
    session_id: str
    start_time: float
    end_time: float
    num_circuits: int
    
    records: List[NoiseRecord]
    
    # Session metadata
    device_parameters: Dict[str, Any]
    environmental_conditions: Dict[str, Any]
    calibration_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device_name": self.device_name,
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "num_circuits": self.num_circuits,
            "records": [r.to_dict() for r in self.records],
            "device_parameters": self.device_parameters,
            "environmental_conditions": self.environmental_conditions,
            "calibration_data": self.calibration_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NoiseTrace":
        """Create from dictionary."""
        records = [NoiseRecord.from_dict(r) for r in data["records"]]
        return cls(
            device_name=data["device_name"],
            session_id=data["session_id"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            num_circuits=data["num_circuits"],
            records=records,
            device_parameters=data["device_parameters"],
            environmental_conditions=data["environmental_conditions"],
            calibration_data=data.get("calibration_data")
        )


class NoiseRecorder:
    """
    Records noise events from real quantum device executions.
    
    Captures noise patterns, error correlations, and environmental
    conditions for later replay and analysis.
    """
    
    def __init__(
        self,
        device_name: str,
        storage_path: Optional[str] = None,
        record_environment: bool = True,
        record_measurements: bool = True,
        compression: bool = True
    ):
        """
        Initialize noise recorder.
        
        Args:
            device_name: Name of the quantum device
            storage_path: Path for storing noise traces
            record_environment: Whether to record environmental data
            record_measurements: Whether to record measurement outcomes
            compression: Whether to compress stored data
        """
        self.device_name = device_name
        self.storage_path = Path(storage_path) if storage_path else Path("./noise_traces")
        self.record_environment = record_environment
        self.record_measurements = record_measurements
        self.compression = compression
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Current recording session
        self.current_session: Optional[str] = None
        self.session_start_time: Optional[float] = None
        self.current_records: List[NoiseRecord] = []
        self.circuit_counter = 0
        
        # Device state tracking
        self.last_calibration: Optional[Dict[str, Any]] = None
        self.current_parameters: Dict[str, Any] = {}
        self.environmental_sensors: Dict[str, Any] = {}
        
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new recording session.
        
        Args:
            session_id: Optional session identifier
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        self.current_session = session_id
        self.session_start_time = time.time()
        self.current_records = []
        self.circuit_counter = 0
        
        print(f"Started noise recording session: {session_id}")
        return session_id
    
    def end_session(self) -> Optional[NoiseTrace]:
        """
        End current recording session and save trace.
        
        Returns:
            Complete noise trace
        """
        if self.current_session is None:
            print("No active recording session")
            return None
        
        end_time = time.time()
        
        # Create noise trace
        trace = NoiseTrace(
            device_name=self.device_name,
            session_id=self.current_session,
            start_time=self.session_start_time,
            end_time=end_time,
            num_circuits=self.circuit_counter,
            records=self.current_records.copy(),
            device_parameters=self.current_parameters.copy(),
            environmental_conditions=self.environmental_sensors.copy(),
            calibration_data=self.last_calibration
        )
        
        # Save trace
        self._save_trace(trace)
        
        print(f"Ended recording session: {self.current_session}")
        print(f"Recorded {len(self.current_records)} noise events")
        
        # Reset session
        self.current_session = None
        self.session_start_time = None
        self.current_records = []
        self.circuit_counter = 0
        
        return trace
    
    def record_circuit_execution(
        self,
        circuit: Any,
        backend_result: Any,
        circuit_id: Optional[str] = None
    ) -> None:
        """
        Record noise events from circuit execution.
        
        Args:
            circuit: Quantum circuit that was executed
            backend_result: Result from backend execution
            circuit_id: Optional circuit identifier
        """
        if self.current_session is None:
            print("Warning: No active recording session")
            return
        
        if circuit_id is None:
            circuit_id = f"circuit_{self.circuit_counter}"
        
        self.circuit_counter += 1
        
        # Extract noise events from circuit and results
        self._extract_noise_events(circuit, backend_result, circuit_id)
        
        # Update environmental conditions
        if self.record_environment:
            self._update_environmental_data()
    
    def _extract_noise_events(
        self,
        circuit: Any,
        backend_result: Any,
        circuit_id: str
    ) -> None:
        """Extract noise events from circuit execution."""
        current_time = time.time()
        
        # Process each gate in the circuit
        if hasattr(circuit, 'gates'):
            for gate_idx, gate in enumerate(circuit.gates):
                gate_name = gate.get("name", "unknown")
                gate_type = gate.get("type", "single")
                qubits = gate.get("qubits", [])
                
                # Detect errors based on various indicators
                error_detected = self._detect_gate_error(gate, backend_result, gate_idx)
                
                if error_detected or np.random.random() < 0.1:  # Record some non-error events too
                    error_type, error_magnitude, error_phase = self._characterize_error(
                        gate, backend_result, gate_idx
                    )
                    
                    record = NoiseRecord(
                        timestamp=current_time + gate_idx * 1e-6,  # Approximate gate timing
                        circuit_id=circuit_id,
                        gate_index=gate_idx,
                        qubit_indices=qubits,
                        gate_name=gate_name,
                        gate_type=gate_type,
                        error_occurred=error_detected,
                        error_type=error_type,
                        error_magnitude=error_magnitude,
                        error_phase=error_phase,
                        temperature=self.environmental_sensors.get("temperature"),
                        magnetic_field=self.environmental_sensors.get("magnetic_field"),
                        control_amplitudes=self._get_control_amplitudes(qubits),
                        frequencies=self._get_qubit_frequencies(qubits)
                    )
                    
                    # Add measurement outcomes if available
                    if self.record_measurements and hasattr(backend_result, 'measurements'):
                        record.expected_outcome = self._get_expected_outcome(circuit, qubits)
                        record.actual_outcome = self._get_actual_outcome(backend_result, qubits)
                        record.measurement_fidelity = self._calculate_measurement_fidelity(
                            record.expected_outcome, record.actual_outcome
                        )
                    
                    self.current_records.append(record)
    
    def _detect_gate_error(self, gate: Dict[str, Any], result: Any, gate_idx: int) -> bool:
        """
        Detect if an error occurred during gate execution.
        
        This is a simplified implementation. In practice, would use:
        - Process tomography results
        - Randomized benchmarking data
        - Error syndrome detection
        - Statistical analysis of measurement outcomes
        """
        # Simulate error detection based on gate type and environmental conditions
        gate_type = gate.get("type", "single")
        
        # Base error rates
        if gate_type == "single":
            base_error_rate = 0.001
        elif gate_type == "two":
            base_error_rate = 0.01
        else:
            base_error_rate = 0.005
        
        # Environmental factors
        temp_factor = 1.0
        if "temperature" in self.environmental_sensors:
            temp = self.environmental_sensors["temperature"]
            temp_factor = 1.0 + 0.01 * (temp - 300)  # Increase errors with temperature
        
        # Drift factors
        time_factor = 1.0 + 0.0001 * (time.time() - self.session_start_time)
        
        effective_error_rate = base_error_rate * temp_factor * time_factor
        
        return np.random.random() < effective_error_rate
    
    def _characterize_error(
        self,
        gate: Dict[str, Any],
        result: Any,
        gate_idx: int
    ) -> Tuple[str, float, float]:
        """
        Characterize the type and magnitude of detected error.
        
        Returns:
            (error_type, magnitude, phase)
        """
        gate_type = gate.get("type", "single")
        gate_name = gate.get("name", "unknown")
        
        # Determine error type based on gate and conditions
        error_types = ["bit_flip", "phase_flip", "amplitude_error", "decoherence"]
        
        if gate_type == "single":
            if "X" in gate_name:
                error_type = np.random.choice(["bit_flip", "amplitude_error"], p=[0.7, 0.3])
            elif "Z" in gate_name:
                error_type = np.random.choice(["phase_flip", "amplitude_error"], p=[0.7, 0.3])
            else:
                error_type = np.random.choice(error_types, p=[0.25, 0.25, 0.25, 0.25])
        else:
            error_type = np.random.choice(["crosstalk", "decoherence", "amplitude_error"], 
                                        p=[0.4, 0.3, 0.3])
        
        # Error magnitude (0 to 1)
        if error_type == "decoherence":
            magnitude = np.random.exponential(0.1)
        else:
            magnitude = np.random.rayleigh(0.05)
        
        magnitude = min(magnitude, 1.0)
        
        # Error phase (0 to 2π)
        phase = np.random.uniform(0, 2 * np.pi)
        
        return error_type, float(magnitude), float(phase)
    
    def _update_environmental_data(self) -> None:
        """Update environmental sensor readings."""
        if not self.record_environment:
            return
        
        # Simulate environmental sensor readings
        # In practice, would interface with actual sensors
        
        base_temp = 300.0  # Kelvin
        temp_variation = 2.0 * np.sin(time.time() / 3600)  # Hourly variation
        self.environmental_sensors["temperature"] = base_temp + temp_variation
        
        base_field = 1e-6  # Tesla
        field_noise = np.random.normal(0, 1e-8)
        self.environmental_sensors["magnetic_field"] = base_field + field_noise
        
        # Control system parameters
        self.environmental_sensors["control_stability"] = 0.99 + 0.01 * np.random.random()
        self.environmental_sensors["power_supply_voltage"] = 12.0 + 0.1 * np.random.normal()
    
    def _get_control_amplitudes(self, qubits: List[int]) -> Dict[str, float]:
        """Get control pulse amplitudes for qubits."""
        amplitudes = {}
        for qubit in qubits:
            # Simulate control amplitude with drift
            base_amplitude = 1.0
            drift = 0.01 * np.sin(time.time() / 1000)  # Slow drift
            noise = np.random.normal(0, 0.005)
            amplitudes[f"qubit_{qubit}"] = base_amplitude + drift + noise
        return amplitudes
    
    def _get_qubit_frequencies(self, qubits: List[int]) -> Dict[int, float]:
        """Get current qubit frequencies."""
        frequencies = {}
        for qubit in qubits:
            # Simulate frequency with drift
            base_freq = 5.0 + 0.1 * qubit  # GHz
            drift = 0.001 * np.sin(time.time() / 3600)  # Hourly drift
            noise = np.random.normal(0, 0.0001)
            frequencies[qubit] = (base_freq + drift + noise) * 1e9  # Convert to Hz
        return frequencies
    
    def _get_expected_outcome(self, circuit: Any, qubits: List[int]) -> Optional[str]:
        """Get expected measurement outcome for qubits."""
        # This would require circuit simulation
        # For now, return random expected outcome
        if len(qubits) == 1:
            return np.random.choice(["0", "1"])
        else:
            num_qubits = len(qubits)
            outcome = np.random.randint(0, 2**num_qubits)
            return format(outcome, f'0{num_qubits}b')
    
    def _get_actual_outcome(self, result: Any, qubits: List[int]) -> Optional[str]:
        """Get actual measurement outcome from backend result."""
        # Extract from backend result
        # For now, simulate with some error probability
        expected = self._get_expected_outcome(None, qubits)
        if expected and np.random.random() < 0.95:  # 95% fidelity
            return expected
        else:
            # Return flipped outcome
            if len(qubits) == 1:
                return "1" if expected == "0" else "0"
            else:
                # Flip random bit
                outcome_int = int(expected, 2)
                flip_bit = np.random.randint(0, len(qubits))
                flipped = outcome_int ^ (1 << flip_bit)
                return format(flipped, f'0{len(qubits)}b')
    
    def _calculate_measurement_fidelity(
        self,
        expected: Optional[str],
        actual: Optional[str]
    ) -> Optional[float]:
        """Calculate measurement fidelity."""
        if expected is None or actual is None:
            return None
        
        if expected == actual:
            return 1.0
        else:
            # Calculate Hamming distance
            if len(expected) == len(actual):
                hamming_dist = sum(c1 != c2 for c1, c2 in zip(expected, actual))
                return 1.0 - hamming_dist / len(expected)
            else:
                return 0.0
    
    def _save_trace(self, trace: NoiseTrace) -> None:
        """Save noise trace to storage."""
        filename = f"{trace.device_name}_{trace.session_id}.json"
        filepath = self.storage_path / filename
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(trace.to_dict(), f, indent=2)
        
        # Also save compressed pickle for large traces
        if self.compression and len(trace.records) > 1000:
            pickle_filename = f"{trace.device_name}_{trace.session_id}.pkl"
            pickle_filepath = self.storage_path / pickle_filename
            
            with open(pickle_filepath, 'wb') as f:
                pickle.dump(trace, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Saved noise trace to {filepath}")
    
    def load_trace(self, filename: str) -> NoiseTrace:
        """
        Load noise trace from file.
        
        Args:
            filename: Trace filename
            
        Returns:
            Loaded noise trace
        """
        filepath = self.storage_path / filename
        
        if filename.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return NoiseTrace.from_dict(data)
        elif filename.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    
    def list_traces(self) -> List[str]:
        """List available noise traces."""
        traces = []
        for file in self.storage_path.glob(f"{self.device_name}_*.json"):
            traces.append(file.name)
        for file in self.storage_path.glob(f"{self.device_name}_*.pkl"):
            traces.append(file.name)
        return sorted(traces)
    
    def get_trace_info(self, filename: str) -> Dict[str, Any]:
        """
        Get information about a noise trace without loading full data.
        
        Args:
            filename: Trace filename
            
        Returns:
            Trace metadata
        """
        trace = self.load_trace(filename)
        return {
            "device_name": trace.device_name,
            "session_id": trace.session_id,
            "duration": trace.end_time - trace.start_time,
            "num_circuits": trace.num_circuits,
            "num_records": len(trace.records),  
            "start_time": trace.start_time,
            "end_time": trace.end_time
        }
    
    def analyze_trace(self, trace: NoiseTrace) -> Dict[str, Any]:
        """
        Analyze noise trace for patterns and statistics.
        
        Args:
            trace: Noise trace to analyze
            
        Returns:
            Analysis results
        """
        records = trace.records
        if not records:
            return {"error": "No records in trace"}
        
        analysis = {
            "total_events": len(records),
            "error_rate": sum(1 for r in records if r.error_occurred) / len(records),
            "error_types": {},
            "qubit_error_rates": {},
            "gate_error_rates": {},
            "temporal_patterns": {},
            "environmental_correlations": {}
        }
        
        # Error type distribution
        error_records = [r for r in records if r.error_occurred]
        if error_records:
            for record in error_records:
                error_type = record.error_type
                analysis["error_types"][error_type] = analysis["error_types"].get(error_type, 0) + 1
        
        # Per-qubit error rates
        qubit_events = {}
        qubit_errors = {}
        
        for record in records:
            for qubit in record.qubit_indices:
                qubit_events[qubit] = qubit_events.get(qubit, 0) + 1
                if record.error_occurred:
                    qubit_errors[qubit] = qubit_errors.get(qubit, 0) + 1
        
        for qubit in qubit_events:
            analysis["qubit_error_rates"][qubit] = qubit_errors.get(qubit, 0) / qubit_events[qubit]
        
        # Per-gate error rates
        gate_events = {}
        gate_errors = {}
        
        for record in records:
            gate = record.gate_name
            gate_events[gate] = gate_events.get(gate, 0) + 1
            if record.error_occurred:
                gate_errors[gate] = gate_errors.get(gate, 0) + 1
        
        for gate in gate_events:
            analysis["gate_error_rates"][gate] = gate_errors.get(gate, 0) / gate_events[gate]
        
        # Temporal patterns
        timestamps = [r.timestamp for r in records]
        if len(timestamps) > 1:
            analysis["temporal_patterns"] = {
                "duration": max(timestamps) - min(timestamps),
                "event_rate": len(records) / (max(timestamps) - min(timestamps)),
                "time_between_errors": self._analyze_error_timing(error_records)
            }
        
        # Environmental correlations
        if any(r.temperature for r in records):
            temps = [r.temperature for r in records if r.temperature is not None]
            error_temps = [r.temperature for r in error_records if r.temperature is not None]
            
            if temps and error_temps:
                analysis["environmental_correlations"]["temperature"] = {
                    "avg_temp_all": np.mean(temps),
                    "avg_temp_errors": np.mean(error_temps),
                    "correlation": np.corrcoef([r.temperature or 0 for r in records],
                                             [1 if r.error_occurred else 0 for r in records])[0, 1]
                }
        
        return analysis
    
    def _analyze_error_timing(self, error_records: List[NoiseRecord]) -> Dict[str, float]:
        """Analyze timing patterns in error occurrences."""
        if len(error_records) < 2:
            return {}
        
        timestamps = [r.timestamp for r in error_records]
        intervals = np.diff(timestamps)
        
        return {
            "mean_interval": float(np.mean(intervals)),
            "std_interval": float(np.std(intervals)),
            "min_interval": float(np.min(intervals)),
            "max_interval": float(np.max(intervals))
        }
    
    def update_calibration(self, calibration_data: Dict[str, Any]) -> None:
        """Update device calibration data."""
        self.last_calibration = calibration_data.copy()
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update device parameters."""
        self.current_parameters.update(parameters)
    
    def __str__(self) -> str:
        """String representation."""
        active_session = f" (Active: {self.current_session})" if self.current_session else ""
        return f"NoiseRecorder for {self.device_name}{active_session}"