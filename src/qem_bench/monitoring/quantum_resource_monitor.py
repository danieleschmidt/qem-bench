"""Quantum resource monitoring for QEM-Bench experiments."""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import json


logger = logging.getLogger(__name__)


@dataclass
class QuantumResourceUsage:
    """Record of quantum resource usage for an operation."""
    operation_name: str
    timestamp: float
    shots_requested: int = 0
    shots_executed: int = 0
    circuits_count: int = 0
    total_gates: int = 0
    gate_breakdown: Dict[str, int] = field(default_factory=dict)
    circuit_depth: int = 0
    qubits_used: int = 0
    backend_name: Optional[str] = None
    queue_time: Optional[float] = None  # Time spent in backend queue
    execution_time: Optional[float] = None  # Actual execution time
    cost_estimate: Optional[float] = None  # Estimated cost (if applicable)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceStats:
    """Aggregated statistics for quantum resource usage."""
    operation_name: str
    total_shots: int
    total_circuits: int
    total_gates: int
    avg_shots_per_circuit: float
    avg_gates_per_circuit: float
    avg_circuit_depth: float
    avg_qubits_used: float
    total_queue_time: float
    total_execution_time: float
    total_cost_estimate: float
    gate_distribution: Dict[str, int]
    backend_usage: Dict[str, int]
    success_rate: float  # Percentage of successful executions


@dataclass
class QuantumResourceMonitorConfig:
    """Configuration for quantum resource monitoring."""
    enabled: bool = True
    max_records_per_operation: int = 1000
    track_gate_breakdown: bool = True
    track_costs: bool = True
    cost_per_shot: Dict[str, float] = field(default_factory=dict)  # Cost per shot by backend


class QuantumResourceMonitor:
    """
    Monitor for tracking quantum resource usage in error mitigation experiments.
    
    This monitor tracks quantum-specific resources like shots, circuits, gates,
    qubits, and backend usage patterns. It helps optimize resource allocation
    and understand the cost implications of different mitigation strategies.
    
    Example:
        >>> monitor = QuantumResourceMonitor()
        >>> 
        >>> # Manual resource tracking
        >>> with monitor.track_execution("zne_experiment") as tracker:
        ...     tracker.record_circuit(circuit, shots=1024, backend="ibm_perth")
        ...     result = backend.run(circuit, shots=1024)
        ...     tracker.record_completion(queue_time=5.2, execution_time=2.1)
        >>> 
        >>> stats = monitor.get_stats("zne_experiment")
        >>> print(f"Total shots: {stats.total_shots}")
    """
    
    def __init__(self, config: Optional[QuantumResourceMonitorConfig] = None):
        self.config = config or QuantumResourceMonitorConfig()
        self._records: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.max_records_per_operation)
        )
        self._lock = threading.Lock()
        self._active_trackers: Dict[str, 'ResourceTracker'] = {}
    
    @contextmanager
    def track_execution(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for tracking quantum resource usage.
        
        Args:
            operation_name: Name of the operation being tracked
            metadata: Additional metadata to store with the resource record
        
        Returns:
            ResourceTracker object for recording resource usage
        """
        if not self.config.enabled:
            yield ResourceTracker(None, operation_name, metadata or {})
            return
        
        tracker_id = f"{operation_name}_{time.time()}_{threading.get_ident()}"
        tracker = ResourceTracker(self, operation_name, metadata or {})
        
        with self._lock:
            self._active_trackers[tracker_id] = tracker
        
        try:
            yield tracker
        finally:
            # Finalize the resource record
            if tracker.has_data():
                record = tracker.to_resource_usage()
                with self._lock:
                    self._records[operation_name].append(record)
            
            with self._lock:
                self._active_trackers.pop(tracker_id, None)
    
    def record_usage(self, operation_name: str, **kwargs):
        """
        Directly record resource usage.
        
        Args:
            operation_name: Name of the operation
            **kwargs: Resource usage parameters (shots_requested, circuits_count, etc.)
        """
        if not self.config.enabled:
            return
        
        usage = QuantumResourceUsage(
            operation_name=operation_name,
            timestamp=time.time(),
            **kwargs
        )
        
        with self._lock:
            self._records[operation_name].append(usage)
    
    def get_records(self, operation_name: str, 
                   duration_seconds: Optional[float] = None) -> List[QuantumResourceUsage]:
        """
        Get resource usage records for an operation.
        
        Args:
            operation_name: Name of the operation
            duration_seconds: If specified, only return records from this many
                            seconds ago
        
        Returns:
            List of QuantumResourceUsage objects
        """
        with self._lock:
            records = list(self._records[operation_name])
        
        if duration_seconds is None:
            return records
        
        cutoff_time = time.time() - duration_seconds
        return [r for r in records if r.timestamp >= cutoff_time]
    
    def get_stats(self, operation_name: str, 
                 duration_seconds: Optional[float] = None) -> Optional[ResourceStats]:
        """
        Get aggregated statistics for quantum resource usage.
        
        Args:
            operation_name: Name of the operation
            duration_seconds: If specified, only analyze recent records
        
        Returns:
            ResourceStats object or None if no records exist
        """
        records = self.get_records(operation_name, duration_seconds)
        if not records:
            return None
        
        # Aggregate statistics
        total_shots = sum(r.shots_executed for r in records)
        total_circuits = sum(r.circuits_count for r in records)
        total_gates = sum(r.total_gates for r in records)
        
        # Gate distribution
        gate_distribution = defaultdict(int)
        for record in records:
            for gate, count in record.gate_breakdown.items():
                gate_distribution[gate] += count
        
        # Backend usage
        backend_usage = defaultdict(int)
        for record in records:
            if record.backend_name:
                backend_usage[record.backend_name] += 1
        
        # Timing and cost aggregation
        total_queue_time = sum(r.queue_time or 0 for r in records)
        total_execution_time = sum(r.execution_time or 0 for r in records)
        total_cost_estimate = sum(r.cost_estimate or 0 for r in records)
        
        # Success rate (assuming records with execution_time are successful)
        successful_records = len([r for r in records if r.execution_time is not None])
        success_rate = (successful_records / len(records)) * 100 if records else 0
        
        # Averages
        circuit_records = [r for r in records if r.circuits_count > 0]
        avg_shots_per_circuit = total_shots / total_circuits if total_circuits > 0 else 0
        avg_gates_per_circuit = total_gates / total_circuits if total_circuits > 0 else 0
        avg_circuit_depth = sum(r.circuit_depth for r in circuit_records) / len(circuit_records) if circuit_records else 0
        avg_qubits_used = sum(r.qubits_used for r in circuit_records) / len(circuit_records) if circuit_records else 0
        
        return ResourceStats(
            operation_name=operation_name,
            total_shots=total_shots,
            total_circuits=total_circuits,
            total_gates=total_gates,
            avg_shots_per_circuit=avg_shots_per_circuit,
            avg_gates_per_circuit=avg_gates_per_circuit,
            avg_circuit_depth=avg_circuit_depth,
            avg_qubits_used=avg_qubits_used,
            total_queue_time=total_queue_time,
            total_execution_time=total_execution_time,
            total_cost_estimate=total_cost_estimate,
            gate_distribution=dict(gate_distribution),
            backend_usage=dict(backend_usage),
            success_rate=success_rate
        )
    
    def get_all_operation_names(self) -> List[str]:
        """Get names of all monitored operations."""
        with self._lock:
            return list(self._records.keys())
    
    def get_active_trackers(self) -> Dict[str, 'ResourceTracker']:
        """Get currently active resource trackers."""
        with self._lock:
            return dict(self._active_trackers)
    
    def get_global_stats(self, duration_seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        Get global statistics across all operations.
        
        Args:
            duration_seconds: If specified, only analyze recent records
        
        Returns:
            Dictionary with global resource statistics
        """
        all_records = []
        for op_name in self.get_all_operation_names():
            all_records.extend(self.get_records(op_name, duration_seconds))
        
        if not all_records:
            return {}
        
        # Global aggregations
        global_stats = {
            'total_operations': len(self.get_all_operation_names()),
            'total_executions': len(all_records),
            'total_shots': sum(r.shots_executed for r in all_records),
            'total_circuits': sum(r.circuits_count for r in all_records),
            'total_gates': sum(r.total_gates for r in all_records),
            'total_queue_time': sum(r.queue_time or 0 for r in all_records),
            'total_execution_time': sum(r.execution_time or 0 for r in all_records),
            'total_cost_estimate': sum(r.cost_estimate or 0 for r in all_records),
        }
        
        # Backend distribution
        backend_counts = defaultdict(int)
        for record in all_records:
            if record.backend_name:
                backend_counts[record.backend_name] += 1
        global_stats['backend_distribution'] = dict(backend_counts)
        
        # Most used gates
        gate_counts = defaultdict(int)
        for record in all_records:
            for gate, count in record.gate_breakdown.items():
                gate_counts[gate] += count
        global_stats['top_gates'] = dict(sorted(gate_counts.items(), 
                                               key=lambda x: x[1], reverse=True)[:10])
        
        # Efficiency metrics
        if global_stats['total_circuits'] > 0:
            global_stats['avg_shots_per_circuit'] = global_stats['total_shots'] / global_stats['total_circuits']
            global_stats['avg_gates_per_circuit'] = global_stats['total_gates'] / global_stats['total_circuits']
        
        return global_stats
    
    def clear_records(self, operation_name: Optional[str] = None):
        """
        Clear resource usage records.
        
        Args:
            operation_name: If specified, clear only records for this operation.
                          If None, clear all records.
        """
        with self._lock:
            if operation_name:
                if operation_name in self._records:
                    self._records[operation_name].clear()
            else:
                self._records.clear()
    
    def export_stats(self, operation_name: str, filepath: str, 
                    duration_seconds: Optional[float] = None):
        """
        Export resource statistics for an operation to a file.
        
        Args:
            operation_name: Name of the operation
            filepath: Path to export file (JSON format)
            duration_seconds: If specified, only export recent records
        """
        stats = self.get_stats(operation_name, duration_seconds)
        records = self.get_records(operation_name, duration_seconds)
        
        export_data = {
            'operation_name': operation_name,
            'export_timestamp': time.time(),
            'duration_filter_seconds': duration_seconds,
            'statistics': {
                'total_shots': stats.total_shots if stats else 0,
                'total_circuits': stats.total_circuits if stats else 0,
                'total_gates': stats.total_gates if stats else 0,
                'avg_shots_per_circuit': stats.avg_shots_per_circuit if stats else 0,
                'avg_gates_per_circuit': stats.avg_gates_per_circuit if stats else 0,
                'avg_circuit_depth': stats.avg_circuit_depth if stats else 0,
                'avg_qubits_used': stats.avg_qubits_used if stats else 0,
                'total_queue_time': stats.total_queue_time if stats else 0,
                'total_execution_time': stats.total_execution_time if stats else 0,
                'total_cost_estimate': stats.total_cost_estimate if stats else 0,
                'gate_distribution': stats.gate_distribution if stats else {},
                'backend_usage': stats.backend_usage if stats else {},
                'success_rate': stats.success_rate if stats else 0
            },
            'records': []
        }
        
        for record in records:
            record_data = {
                'timestamp': record.timestamp,
                'shots_requested': record.shots_requested,
                'shots_executed': record.shots_executed,
                'circuits_count': record.circuits_count,
                'total_gates': record.total_gates,
                'gate_breakdown': record.gate_breakdown,
                'circuit_depth': record.circuit_depth,
                'qubits_used': record.qubits_used,
                'backend_name': record.backend_name,
                'queue_time': record.queue_time,
                'execution_time': record.execution_time,
                'cost_estimate': record.cost_estimate,
                'metadata': record.metadata
            }
            export_data['records'].append(record_data)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported quantum resource data for '{operation_name}' to {filepath}")


class ResourceTracker:
    """Helper class for tracking quantum resource usage within a context."""
    
    def __init__(self, monitor: Optional[QuantumResourceMonitor], 
                 operation_name: str, metadata: Dict[str, Any]):
        self.monitor = monitor
        self.operation_name = operation_name
        self.metadata = metadata
        self.timestamp = time.time()
        
        # Resource tracking fields
        self.shots_requested = 0
        self.shots_executed = 0
        self.circuits_count = 0
        self.total_gates = 0
        self.gate_breakdown: Dict[str, int] = defaultdict(int)
        self.circuit_depth = 0
        self.qubits_used = 0
        self.backend_name: Optional[str] = None
        self.queue_time: Optional[float] = None
        self.execution_time: Optional[float] = None
        self.cost_estimate: Optional[float] = None
    
    def record_circuit(self, circuit: Any, shots: int = 0, backend: Optional[str] = None):
        """
        Record a circuit execution.
        
        Args:
            circuit: The quantum circuit
            shots: Number of shots requested
            backend: Backend name
        """
        self.circuits_count += 1
        self.shots_requested += shots
        
        if backend:
            self.backend_name = backend
        
        # Analyze circuit if possible
        if hasattr(circuit, 'num_qubits'):
            self.qubits_used = max(self.qubits_used, circuit.num_qubits)
        
        if hasattr(circuit, 'depth'):
            self.circuit_depth = max(self.circuit_depth, circuit.depth())
        
        # Count gates if possible
        if hasattr(circuit, 'count_ops'):
            gate_counts = circuit.count_ops()
            for gate, count in gate_counts.items():
                self.gate_breakdown[gate] += count
                self.total_gates += count
        elif hasattr(circuit, 'data'):
            # Try to analyze circuit data
            for instruction in circuit.data:
                gate_name = instruction[0].name if hasattr(instruction[0], 'name') else str(instruction[0])
                self.gate_breakdown[gate_name] += 1
                self.total_gates += 1
    
    def record_completion(self, shots_executed: Optional[int] = None,
                         queue_time: Optional[float] = None,
                         execution_time: Optional[float] = None):
        """
        Record completion of circuit execution.
        
        Args:
            shots_executed: Actual number of shots executed
            queue_time: Time spent in backend queue (seconds)
            execution_time: Actual execution time (seconds)
        """
        if shots_executed is not None:
            self.shots_executed = shots_executed
        else:
            self.shots_executed = self.shots_requested
        
        if queue_time is not None:
            self.queue_time = queue_time
        
        if execution_time is not None:
            self.execution_time = execution_time
        
        # Calculate cost estimate if configured
        if (self.monitor and self.backend_name and 
            self.backend_name in self.monitor.config.cost_per_shot):
            cost_per_shot = self.monitor.config.cost_per_shot[self.backend_name]
            self.cost_estimate = self.shots_executed * cost_per_shot
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the resource record."""
        self.metadata[key] = value
    
    def has_data(self) -> bool:
        """Check if any resource data has been recorded."""
        return (self.circuits_count > 0 or self.shots_requested > 0 or 
                self.total_gates > 0 or bool(self.metadata))
    
    def to_resource_usage(self) -> QuantumResourceUsage:
        """Convert to QuantumResourceUsage record."""
        return QuantumResourceUsage(
            operation_name=self.operation_name,
            timestamp=self.timestamp,
            shots_requested=self.shots_requested,
            shots_executed=self.shots_executed,
            circuits_count=self.circuits_count,
            total_gates=self.total_gates,
            gate_breakdown=dict(self.gate_breakdown),
            circuit_depth=self.circuit_depth,
            qubits_used=self.qubits_used,
            backend_name=self.backend_name,
            queue_time=self.queue_time,
            execution_time=self.execution_time,
            cost_estimate=self.cost_estimate,
            metadata=self.metadata
        )