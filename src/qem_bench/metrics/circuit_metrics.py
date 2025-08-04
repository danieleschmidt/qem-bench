"""Circuit analysis and metrics for QEM-Bench."""

import time
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class CircuitAnalysis:
    """Comprehensive analysis of a quantum circuit."""
    circuit_id: str
    timestamp: float
    
    # Basic properties
    num_qubits: int
    num_gates: int
    circuit_depth: int
    
    # Gate statistics
    gate_counts: Dict[str, int] = field(default_factory=dict)
    gate_distribution: Dict[str, float] = field(default_factory=dict)  # Percentages
    
    # Complexity metrics
    two_qubit_gate_count: int = 0
    single_qubit_gate_count: int = 0
    measurement_count: int = 0
    
    # Topology analysis
    connectivity_graph: Optional[Dict[int, List[int]]] = None
    qubit_usage: Dict[int, int] = field(default_factory=dict)  # Gates per qubit
    
    # Advanced metrics
    circuit_volume: Optional[float] = None  # Volume metric
    entanglement_entropy: Optional[float] = None
    expressibility: Optional[float] = None
    
    # Performance indicators
    estimated_runtime: Optional[float] = None  # Estimated execution time
    noise_susceptibility: Optional[float] = None  # Estimated noise impact
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def gate_density(self) -> float:
        """Gates per qubit ratio."""
        return self.num_gates / self.num_qubits if self.num_qubits > 0 else 0
    
    @property
    def depth_efficiency(self) -> float:
        """Ratio of gates to depth (measure of parallelization)."""
        return self.num_gates / self.circuit_depth if self.circuit_depth > 0 else 0
    
    @property
    def two_qubit_ratio(self) -> float:
        """Ratio of two-qubit to total gates."""
        return self.two_qubit_gate_count / self.num_gates if self.num_gates > 0 else 0


class CircuitMetrics:
    """
    Circuit analyzer for extracting metrics and properties from quantum circuits.
    
    Analyzes quantum circuits to extract various metrics including gate counts,
    depth, connectivity patterns, complexity measures, and performance estimates.
    
    Example:
        >>> analyzer = CircuitMetrics()
        >>> 
        >>> # Analyze a single circuit
        >>> analysis = analyzer.analyze_circuit(circuit, circuit_id="ghz_4")
        >>> print(f"Circuit depth: {analysis.circuit_depth}")
        >>> print(f"Two-qubit gate ratio: {analysis.two_qubit_ratio:.2%}")
        >>> 
        >>> # Batch analysis
        >>> circuits = [circuit1, circuit2, circuit3]
        >>> analyses = analyzer.analyze_batch(circuits)
        >>> summary = analyzer.get_batch_summary(analyses)
    """
    
    def __init__(self):
        # Define gate categories
        self.single_qubit_gates = {
            'x', 'y', 'z', 'h', 'i', 's', 't', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3',
            'sx', 'sy', 'p', 'phase', 'tdg', 'sdg', 'reset'
        }
        
        self.two_qubit_gates = {
            'cx', 'cy', 'cz', 'ch', 'crx', 'cry', 'crz', 'cu', 'cu1', 'cu2', 'cu3',
            'cnot', 'swap', 'iswap', 'cphase', 'cr', 'dcx', 'rzz', 'rxx', 'ryy',
            'ccx', 'cswap'  # These are actually 3-qubit gates but often treated as 2-qubit
        }
        
        self.measurement_gates = {'measure', 'barrier', 'snapshot'}
        
        # Circuit history for comparative analysis
        self._analysis_history: List[CircuitAnalysis] = []
    
    def analyze_circuit(self, circuit: Any, circuit_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> CircuitAnalysis:
        """
        Analyze a quantum circuit and extract comprehensive metrics.
        
        Args:
            circuit: Quantum circuit to analyze
            circuit_id: Unique identifier for the circuit
            metadata: Additional metadata to include
        
        Returns:
            CircuitAnalysis object with detailed metrics
        """
        if circuit_id is None:
            circuit_id = f"circuit_{int(time.time())}_{id(circuit)}"
        
        analysis = CircuitAnalysis(
            circuit_id=circuit_id,
            timestamp=time.time(),
            num_qubits=self._get_num_qubits(circuit),
            num_gates=self._get_num_gates(circuit),
            circuit_depth=self._get_circuit_depth(circuit),
            metadata=metadata or {}
        )
        
        # Gate analysis
        analysis.gate_counts = self._analyze_gates(circuit)
        analysis.gate_distribution = self._calculate_gate_distribution(analysis.gate_counts)
        
        # Gate categorization
        analysis.single_qubit_gate_count = sum(
            count for gate, count in analysis.gate_counts.items()
            if gate.lower() in self.single_qubit_gates
        )
        analysis.two_qubit_gate_count = sum(
            count for gate, count in analysis.gate_counts.items()
            if gate.lower() in self.two_qubit_gates
        )
        analysis.measurement_count = sum(
            count for gate, count in analysis.gate_counts.items()
            if gate.lower() in self.measurement_gates
        )
        
        # Topology analysis
        analysis.connectivity_graph = self._analyze_connectivity(circuit)
        analysis.qubit_usage = self._analyze_qubit_usage(circuit)
        
        # Advanced metrics
        analysis.circuit_volume = self._calculate_circuit_volume(circuit, analysis)
        analysis.estimated_runtime = self._estimate_runtime(circuit, analysis)
        analysis.noise_susceptibility = self._estimate_noise_susceptibility(circuit, analysis)
        
        # Store in history
        self._analysis_history.append(analysis)
        
        return analysis
    
    def _get_num_qubits(self, circuit: Any) -> int:
        """Get number of qubits in the circuit."""
        if hasattr(circuit, 'num_qubits'):
            return circuit.num_qubits
        elif hasattr(circuit, 'n_qubits'):
            return circuit.n_qubits
        elif hasattr(circuit, 'qubits'):
            return len(circuit.qubits)
        else:
            # Try to infer from circuit data
            try:
                if hasattr(circuit, 'data'):
                    max_qubit = 0
                    for instruction in circuit.data:
                        if hasattr(instruction, 'qubits'):
                            for qubit in instruction.qubits:
                                qubit_idx = qubit.index if hasattr(qubit, 'index') else int(qubit)
                                max_qubit = max(max_qubit, qubit_idx)
                    return max_qubit + 1
            except:
                pass
            
            logger.warning("Could not determine number of qubits, assuming 1")
            return 1
    
    def _get_num_gates(self, circuit: Any) -> int:
        """Get total number of gates in the circuit."""
        if hasattr(circuit, 'size'):
            return circuit.size()
        elif hasattr(circuit, 'count_ops'):
            return sum(circuit.count_ops().values())
        elif hasattr(circuit, 'data'):
            return len([inst for inst in circuit.data if not self._is_barrier_or_measure(inst)])
        else:
            logger.warning("Could not determine number of gates, assuming 0")
            return 0
    
    def _get_circuit_depth(self, circuit: Any) -> int:
        """Get circuit depth."""
        if hasattr(circuit, 'depth'):
            return circuit.depth()
        else:
            # Estimate depth (simplified)
            return max(1, self._get_num_gates(circuit) // max(1, self._get_num_qubits(circuit)))
    
    def _analyze_gates(self, circuit: Any) -> Dict[str, int]:
        """Analyze gate types and counts."""
        gate_counts = defaultdict(int)
        
        try:
            if hasattr(circuit, 'count_ops'):
                # Qiskit-style circuit
                ops = circuit.count_ops()
                for gate_name, count in ops.items():
                    gate_counts[gate_name] = count
            elif hasattr(circuit, 'data'):
                # Generic circuit with data attribute
                for instruction in circuit.data:
                    gate_name = self._get_gate_name(instruction)
                    gate_counts[gate_name] += 1
            else:
                logger.warning("Could not analyze gates in circuit")
        except Exception as e:
            logger.error(f"Error analyzing gates: {e}")
        
        return dict(gate_counts)
    
    def _get_gate_name(self, instruction: Any) -> str:
        """Extract gate name from instruction."""
        if hasattr(instruction, 'operation'):
            if hasattr(instruction.operation, 'name'):
                return instruction.operation.name
        if hasattr(instruction, 'name'):
            return instruction.name
        if len(instruction) > 0 and hasattr(instruction[0], 'name'):
            return instruction[0].name
        return 'unknown'
    
    def _is_barrier_or_measure(self, instruction: Any) -> bool:
        """Check if instruction is a barrier or measurement."""
        gate_name = self._get_gate_name(instruction).lower()
        return gate_name in self.measurement_gates
    
    def _calculate_gate_distribution(self, gate_counts: Dict[str, int]) -> Dict[str, float]:
        """Calculate percentage distribution of gates."""
        total_gates = sum(gate_counts.values())
        if total_gates == 0:
            return {}
        
        return {
            gate: (count / total_gates) * 100
            for gate, count in gate_counts.items()
        }
    
    def _analyze_connectivity(self, circuit: Any) -> Optional[Dict[int, List[int]]]:
        """Analyze qubit connectivity in the circuit."""
        try:
            connectivity = defaultdict(set)
            
            if hasattr(circuit, 'data'):
                for instruction in circuit.data:
                    if hasattr(instruction, 'qubits'):
                        qubits = instruction.qubits
                        if len(qubits) >= 2:
                            # Two-qubit or multi-qubit gate
                            qubit_indices = [
                                q.index if hasattr(q, 'index') else int(q)
                                for q in qubits
                            ]
                            
                            # Add connections between all pairs
                            for i, q1 in enumerate(qubit_indices):
                                for q2 in qubit_indices[i+1:]:
                                    connectivity[q1].add(q2)
                                    connectivity[q2].add(q1)
            
            # Convert to regular dict with lists
            return {qubit: list(connections) for qubit, connections in connectivity.items()}
            
        except Exception as e:
            logger.debug(f"Error analyzing connectivity: {e}")
            return None
    
    def _analyze_qubit_usage(self, circuit: Any) -> Dict[int, int]:
        """Analyze how many gates operate on each qubit."""
        try:
            qubit_usage = defaultdict(int)
            
            if hasattr(circuit, 'data'):
                for instruction in circuit.data:
                    if hasattr(instruction, 'qubits'):
                        for qubit in instruction.qubits:
                            qubit_idx = qubit.index if hasattr(qubit, 'index') else int(qubit)
                            qubit_usage[qubit_idx] += 1
            
            return dict(qubit_usage)
            
        except Exception as e:
            logger.debug(f"Error analyzing qubit usage: {e}")
            return {}
    
    def _calculate_circuit_volume(self, circuit: Any, analysis: CircuitAnalysis) -> Optional[float]:
        """Calculate circuit volume metric."""
        try:
            # Volume = num_qubits * circuit_depth (simplified)
            # This is a basic metric; more sophisticated versions could consider
            # gate weights, connectivity constraints, etc.
            return analysis.num_qubits * analysis.circuit_depth
        except Exception:
            return None
    
    def _estimate_runtime(self, circuit: Any, analysis: CircuitAnalysis) -> Optional[float]:
        """Estimate circuit execution time."""
        try:
            # Very rough estimate based on gate counts and types
            # In practice, this would depend on backend specifications
            
            base_time_per_gate = 0.001  # 1ms per gate (rough estimate)
            single_qubit_overhead = 0.0005  # 0.5ms per single-qubit gate
            two_qubit_overhead = 0.002  # 2ms per two-qubit gate
            
            estimated_time = (
                analysis.num_gates * base_time_per_gate +
                analysis.single_qubit_gate_count * single_qubit_overhead +
                analysis.two_qubit_gate_count * two_qubit_overhead
            )
            
            return estimated_time
            
        except Exception:
            return None
    
    def _estimate_noise_susceptibility(self, circuit: Any, analysis: CircuitAnalysis) -> Optional[float]:
        """Estimate circuit's susceptibility to noise."""
        try:
            # Simple model: more two-qubit gates and greater depth increase noise susceptibility
            base_susceptibility = 0.01  # 1% base error
            two_qubit_penalty = 0.005  # 0.5% per two-qubit gate  
            depth_penalty = 0.001  # 0.1% per unit depth
            
            susceptibility = (
                base_susceptibility +
                analysis.two_qubit_gate_count * two_qubit_penalty +
                analysis.circuit_depth * depth_penalty
            )
            
            return min(1.0, susceptibility)  # Cap at 100%
            
        except Exception:
            return None
    
    def analyze_batch(self, circuits: List[Any], 
                     circuit_ids: Optional[List[str]] = None) -> List[CircuitAnalysis]:
        """
        Analyze a batch of circuits.
        
        Args:
            circuits: List of quantum circuits
            circuit_ids: Optional list of circuit identifiers
        
        Returns:
            List of CircuitAnalysis objects
        """
        if circuit_ids is None:
            circuit_ids = [f"batch_circuit_{i}" for i in range(len(circuits))]
        
        analyses = []
        for i, circuit in enumerate(circuits):
            try:
                analysis = self.analyze_circuit(
                    circuit, 
                    circuit_id=circuit_ids[i] if i < len(circuit_ids) else f"batch_circuit_{i}"
                )
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing circuit {i}: {e}")
        
        return analyses
    
    def get_batch_summary(self, analyses: List[CircuitAnalysis]) -> Dict[str, Any]:
        """
        Get summary statistics for a batch of circuit analyses.
        
        Args:
            analyses: List of CircuitAnalysis objects
        
        Returns:
            Dictionary with summary statistics
        """
        if not analyses:
            return {'error': 'No analyses provided'}
        
        # Basic statistics
        num_qubits = [a.num_qubits for a in analyses]
        num_gates = [a.num_gates for a in analyses]
        depths = [a.circuit_depth for a in analyses]
        two_qubit_ratios = [a.two_qubit_ratio for a in analyses]
        
        # Gate distribution across all circuits
        all_gates = defaultdict(int)
        for analysis in analyses:
            for gate, count in analysis.gate_counts.items():
                all_gates[gate] += count
        
        # Most common gates
        total_gates = sum(all_gates.values())
        gate_distribution = {
            gate: (count / total_gates) * 100
            for gate, count in all_gates.items()
        }
        top_gates = dict(sorted(gate_distribution.items(), key=lambda x: x[1], reverse=True)[:10])
        
        summary = {
            'total_circuits': len(analyses),
            'qubit_statistics': {
                'min': min(num_qubits),
                'max': max(num_qubits),
                'avg': np.mean(num_qubits),
                'std': np.std(num_qubits)
            },
            'gate_statistics': {
                'min': min(num_gates),
                'max': max(num_gates), 
                'avg': np.mean(num_gates),
                'std': np.std(num_gates),
                'total': sum(num_gates)
            },
            'depth_statistics': {
                'min': min(depths),
                'max': max(depths),
                'avg': np.mean(depths),
                'std': np.std(depths)
            },
            'two_qubit_ratio_statistics': {
                'min': min(two_qubit_ratios),
                'max': max(two_qubit_ratios),
                'avg': np.mean(two_qubit_ratios),
                'std': np.std(two_qubit_ratios)
            },
            'top_gates': top_gates,
            'complexity_indicators': {
                'avg_circuit_volume': np.mean([a.circuit_volume for a in analyses if a.circuit_volume is not None]),
                'avg_estimated_runtime': np.mean([a.estimated_runtime for a in analyses if a.estimated_runtime is not None]),
                'avg_noise_susceptibility': np.mean([a.noise_susceptibility for a in analyses if a.noise_susceptibility is not None])
            }
        }
        
        return summary
    
    def compare_circuits(self, analysis1: CircuitAnalysis, analysis2: CircuitAnalysis) -> Dict[str, Any]:
        """
        Compare two circuit analyses.
        
        Args:
            analysis1: First circuit analysis
            analysis2: Second circuit analysis
        
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            'circuit_ids': [analysis1.circuit_id, analysis2.circuit_id],
            'size_comparison': {
                'qubit_difference': analysis2.num_qubits - analysis1.num_qubits,
                'gate_difference': analysis2.num_gates - analysis1.num_gates,
                'depth_difference': analysis2.circuit_depth - analysis1.circuit_depth
            },
            'complexity_comparison': {
                'two_qubit_ratio_diff': analysis2.two_qubit_ratio - analysis1.two_qubit_ratio,
                'gate_density_diff': analysis2.gate_density - analysis1.gate_density,
                'depth_efficiency_diff': analysis2.depth_efficiency - analysis1.depth_efficiency
            },
            'performance_comparison': {}
        }
        
        # Performance comparison (if available)
        if analysis1.estimated_runtime and analysis2.estimated_runtime:
            comparison['performance_comparison']['runtime_ratio'] = analysis2.estimated_runtime / analysis1.estimated_runtime
        
        if analysis1.noise_susceptibility and analysis2.noise_susceptibility:
            comparison['performance_comparison']['noise_susceptibility_diff'] = (
                analysis2.noise_susceptibility - analysis1.noise_susceptibility
            )
        
        # Gate distribution comparison
        all_gates = set(analysis1.gate_counts.keys()) | set(analysis2.gate_counts.keys())
        gate_differences = {}
        for gate in all_gates:
            count1 = analysis1.gate_counts.get(gate, 0)
            count2 = analysis2.gate_counts.get(gate, 0)
            gate_differences[gate] = count2 - count1
        
        comparison['gate_differences'] = gate_differences
        
        return comparison
    
    def get_analysis_history(self, limit: Optional[int] = None) -> List[CircuitAnalysis]:
        """Get historical circuit analyses."""
        if limit:
            return self._analysis_history[-limit:]
        return list(self._analysis_history)
    
    def clear_history(self):
        """Clear analysis history."""
        self._analysis_history.clear()
    
    def export_analysis(self, analysis: CircuitAnalysis, filepath: str):
        """Export circuit analysis to a JSON file."""
        import json
        
        # Convert analysis to serializable format
        export_data = {
            'circuit_id': analysis.circuit_id,
            'timestamp': analysis.timestamp,
            'basic_properties': {
                'num_qubits': analysis.num_qubits,
                'num_gates': analysis.num_gates,
                'circuit_depth': analysis.circuit_depth
            },
            'gate_analysis': {
                'gate_counts': analysis.gate_counts,
                'gate_distribution': analysis.gate_distribution,
                'single_qubit_gates': analysis.single_qubit_gate_count,
                'two_qubit_gates': analysis.two_qubit_gate_count,
                'measurements': analysis.measurement_count
            },
            'derived_metrics': {
                'gate_density': analysis.gate_density,
                'depth_efficiency': analysis.depth_efficiency,
                'two_qubit_ratio': analysis.two_qubit_ratio
            },
            'topology_analysis': {
                'connectivity_graph': analysis.connectivity_graph,
                'qubit_usage': analysis.qubit_usage
            },
            'advanced_metrics': {
                'circuit_volume': analysis.circuit_volume,
                'estimated_runtime': analysis.estimated_runtime,
                'noise_susceptibility': analysis.noise_susceptibility
            },
            'metadata': analysis.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported circuit analysis to {filepath}")