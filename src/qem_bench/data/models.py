"""Data models for QEM-Bench experiment results and metadata."""

import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json


class ExperimentStatus(Enum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MitigationMethod(Enum):
    """Quantum error mitigation methods."""
    ZNE = "zero_noise_extrapolation"
    PEC = "probabilistic_error_cancellation"
    VD = "virtual_distillation"
    CDR = "clifford_data_regression"
    SYMMETRY = "symmetry_verification"
    CUSTOM = "custom"


@dataclass
class ExperimentResult:
    """
    Results from a quantum error mitigation experiment.
    
    Contains all data from running QEM techniques on quantum circuits,
    including raw measurements, mitigated results, and performance metrics.
    """
    
    # Unique identifiers
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Experiment metadata
    experiment_name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    status: ExperimentStatus = ExperimentStatus.PENDING
    
    # Circuit information
    circuit_name: str = ""
    circuit_qubits: int = 0
    circuit_depth: int = 0
    circuit_gates: int = 0
    circuit_hash: str = ""
    
    # Mitigation configuration
    mitigation_method: MitigationMethod = MitigationMethod.ZNE
    mitigation_config: Dict[str, Any] = field(default_factory=dict)
    
    # Backend information
    backend_name: str = ""
    backend_type: str = ""  # "simulator", "hardware", "cloud"
    
    # Execution parameters
    shots: int = 1024
    noise_factors: List[float] = field(default_factory=list)
    
    # Raw results
    raw_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)  # noise_factor -> counts
    raw_expectation_values: Dict[str, float] = field(default_factory=dict)  # noise_factor -> value
    
    # Mitigated results
    mitigated_expectation_value: Optional[float] = None
    extrapolation_data: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    fidelity: Optional[float] = None
    error_reduction: Optional[float] = None
    improvement_factor: Optional[float] = None
    tvd: Optional[float] = None  # Total variation distance
    
    # Resource consumption
    execution_time: Optional[float] = None  # seconds
    memory_usage: Optional[int] = None  # bytes
    shots_overhead: Optional[float] = None  # multiplicative factor
    
    # Error information
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling enums and datetime objects."""
        data = asdict(self)
        
        # Handle enum serialization
        data["status"] = self.status.value
        data["mitigation_method"] = self.mitigation_method.value
        
        # Handle datetime serialization
        for field_name in ["created_at", "started_at", "completed_at"]:
            if data[field_name] is not None:
                data[field_name] = data[field_name].isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary, handling deserialization."""
        # Handle enum deserialization
        if "status" in data:
            data["status"] = ExperimentStatus(data["status"])
        if "mitigation_method" in data:
            data["mitigation_method"] = MitigationMethod(data["mitigation_method"])
        
        # Handle datetime deserialization
        for field_name in ["created_at", "started_at", "completed_at"]:
            if data.get(field_name) is not None:
                data[field_name] = datetime.fromisoformat(data[field_name])
        
        return cls(**data)
    
    def mark_started(self) -> None:
        """Mark experiment as started."""
        self.status = ExperimentStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def mark_completed(self) -> None:
        """Mark experiment as completed."""
        self.status = ExperimentStatus.COMPLETED
        self.completed_at = datetime.utcnow()
    
    def mark_failed(self, error_message: str, traceback: Optional[str] = None) -> None:
        """Mark experiment as failed."""
        self.status = ExperimentStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        self.error_traceback = traceback
    
    @property
    def duration(self) -> Optional[float]:
        """Get experiment duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_successful(self) -> bool:
        """Check if experiment completed successfully."""
        return self.status == ExperimentStatus.COMPLETED and self.error_message is None


@dataclass
class BenchmarkRun:
    """
    Results from running a benchmark suite.
    
    Contains aggregated results from multiple experiments
    run as part of a benchmarking campaign.
    """
    
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Benchmark configuration
    benchmark_name: str = ""
    benchmark_version: str = "1.0.0"
    description: str = ""
    
    # Execution parameters
    circuits: List[str] = field(default_factory=list)  # Circuit names
    mitigation_methods: List[str] = field(default_factory=list)
    backends: List[str] = field(default_factory=list)
    shots_per_circuit: int = 1024
    repetitions: int = 1
    
    # Results
    experiment_results: List[str] = field(default_factory=list)  # Experiment IDs
    
    # Aggregated metrics
    total_experiments: int = 0
    successful_experiments: int = 0
    failed_experiments: int = 0
    mean_fidelity: Optional[float] = None
    std_fidelity: Optional[float] = None
    mean_error_reduction: Optional[float] = None
    std_error_reduction: Optional[float] = None
    
    # Resource consumption
    total_execution_time: Optional[float] = None
    total_shots: int = 0
    
    # Status
    status: ExperimentStatus = ExperimentStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkRun":
        """Create from dictionary."""
        if "status" in data:
            data["status"] = ExperimentStatus(data["status"])
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class CircuitMetadata:
    """
    Metadata about quantum circuits used in experiments.
    
    Stores information about circuit structure, properties,
    and theoretical expectations for benchmarking.
    """
    
    circuit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Circuit identification
    name: str = ""
    category: str = ""  # "quantum_volume", "random", "algorithmic", etc.
    hash: str = ""  # Content-based hash for deduplication
    
    # Circuit structure
    num_qubits: int = 0
    depth: int = 0
    num_gates: int = 0
    gate_counts: Dict[str, int] = field(default_factory=dict)
    
    # Circuit properties
    connectivity_graph: List[List[int]] = field(default_factory=list)  # Adjacency list
    entangling_gates: int = 0
    max_qubit_degree: int = 0
    
    # Theoretical properties
    ideal_statevector: Optional[List[complex]] = None
    ideal_expectation_values: Dict[str, float] = field(default_factory=dict)
    
    # Generation parameters
    generation_seed: Optional[int] = None
    generation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Source information
    source_file: Optional[str] = None
    source_format: str = ""  # "qasm", "qiskit", "cirq", "jax", etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        
        # Handle complex numbers in statevector
        if self.ideal_statevector:
            data["ideal_statevector"] = [
                {"real": c.real, "imag": c.imag} for c in self.ideal_statevector
            ]
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitMetadata":
        """Create from dictionary."""
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        
        # Handle complex numbers in statevector
        if "ideal_statevector" in data and data["ideal_statevector"]:
            data["ideal_statevector"] = [
                complex(c["real"], c["imag"]) for c in data["ideal_statevector"]
            ]
        
        return cls(**data)


@dataclass
class NoiseCharacterization:
    """
    Characterization data for quantum device noise.
    
    Stores measured noise parameters and models
    for specific quantum backends.
    """
    
    characterization_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Device information
    backend_name: str = ""
    backend_version: str = ""
    num_qubits: int = 0
    
    # Single-qubit noise parameters
    t1_times: Dict[int, float] = field(default_factory=dict)  # qubit -> T1 (μs)
    t2_times: Dict[int, float] = field(default_factory=dict)  # qubit -> T2 (μs)
    single_qubit_gate_errors: Dict[int, float] = field(default_factory=dict)
    
    # Two-qubit noise parameters
    two_qubit_gate_errors: Dict[str, float] = field(default_factory=dict)  # "q1-q2" -> error
    crosstalk_matrix: List[List[float]] = field(default_factory=list)
    
    # Readout errors
    readout_errors: Dict[int, Dict[str, float]] = field(default_factory=dict)  # qubit -> {"0->1": p, "1->0": p}
    
    # Device topology
    coupling_map: List[List[int]] = field(default_factory=list)
    
    # Noise model parameters
    depolarizing_rates: Dict[str, float] = field(default_factory=dict)
    thermal_population: Dict[int, float] = field(default_factory=dict)
    
    # Measurement metadata
    measurement_date: datetime = field(default_factory=datetime.utcnow)
    measurement_protocol: str = ""
    calibration_shots: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["measurement_date"] = self.measurement_date.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NoiseCharacterization":
        """Create from dictionary."""
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "measurement_date" in data:
            data["measurement_date"] = datetime.fromisoformat(data["measurement_date"])
        return cls(**data)


@dataclass
class DeviceCalibration:
    """
    Device calibration data and parameters.
    
    Stores calibration information for quantum devices
    including gate fidelities and optimal parameters.
    """
    
    calibration_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Device information
    backend_name: str = ""
    calibration_date: datetime = field(default_factory=datetime.utcnow)
    
    # Gate calibrations
    single_qubit_fidelities: Dict[int, Dict[str, float]] = field(default_factory=dict)
    two_qubit_fidelities: Dict[str, float] = field(default_factory=dict)
    
    # Optimal gate parameters
    optimal_pulse_parameters: Dict[str, Dict[str, float]] = field(default_factory=dict)
    gate_times: Dict[str, float] = field(default_factory=dict)
    
    # Frequency calibrations
    qubit_frequencies: Dict[int, float] = field(default_factory=dict)  # Hz
    anharmonicities: Dict[int, float] = field(default_factory=dict)  # Hz
    
    # Calibration quality metrics
    calibration_score: Optional[float] = None
    drift_rate: Optional[float] = None  # per hour
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["calibration_date"] = self.calibration_date.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeviceCalibration":
        """Create from dictionary."""
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "calibration_date" in data:
            data["calibration_date"] = datetime.fromisoformat(data["calibration_date"])
        return cls(**data)