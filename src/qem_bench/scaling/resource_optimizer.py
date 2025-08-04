"""
Resource optimization system for quantum error mitigation workloads.

This module provides intelligent resource optimization including smart batching
of circuits, compilation optimization across devices, shot allocation across
backends, and dynamic parameter adjustment based on queue times and performance.
"""

import time
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

from ..security import SecureConfig


logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Resource optimization strategies."""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_TIME = "minimize_time"
    MAXIMIZE_FIDELITY = "maximize_fidelity"
    BALANCED = "balanced"


class BatchingStrategy(Enum):
    """Circuit batching strategies."""
    SIMILARITY_BASED = "similarity_based"
    SIZE_BASED = "size_based"
    BACKEND_OPTIMAL = "backend_optimal"
    DEADLINE_AWARE = "deadline_aware"


class CompilationLevel(Enum):
    """Circuit compilation optimization levels."""
    BASIC = "basic"           # Basic transpilation
    OPTIMIZED = "optimized"   # Standard optimization
    AGGRESSIVE = "aggressive" # Maximum optimization
    CUSTOM = "custom"         # Custom optimization strategy


@dataclass
class CircuitBatch:
    """Represents a batch of circuits for optimized execution."""
    id: str
    circuits: List[Dict[str, Any]]
    
    # Batch characteristics
    total_shots: int = 0
    estimated_execution_time: float = 0.0
    estimated_cost: float = 0.0
    
    # Backend requirements
    min_qubits: int = 1
    required_gates: Set[str] = field(default_factory=set)
    preferred_backends: List[str] = field(default_factory=list)
    
    # Optimization metadata
    batching_strategy: BatchingStrategy = BatchingStrategy.SIMILARITY_BASED
    optimization_score: float = 0.0
    similarity_score: float = 0.0
    
    # Execution tracking
    assigned_backend: Optional[str] = None
    submission_time: Optional[float] = None
    completion_time: Optional[float] = None
    
    def size(self) -> int:
        """Get batch size (number of circuits)."""
        return len(self.circuits)
    
    def is_compatible_with_backend(self, backend_info: Dict[str, Any]) -> bool:
        """Check if batch is compatible with backend."""
        # Check qubit requirements
        if backend_info.get("num_qubits", 0) < self.min_qubits:
            return False
        
        # Check gate set compatibility
        backend_gates = set(backend_info.get("gate_set", []))
        if not self.required_gates.issubset(backend_gates):
            return False
        
        return True
    
    def execution_efficiency(self) -> float:
        """Calculate execution efficiency score."""
        if self.size() == 0:
            return 0.0
        
        # Efficiency based on batch size and similarity
        size_efficiency = min(self.size() / 10.0, 1.0)  # Optimal around 10 circuits
        similarity_efficiency = self.similarity_score
        
        return (size_efficiency + similarity_efficiency) / 2.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert batch to dictionary."""
        return {
            "id": self.id,
            "size": self.size(),
            "total_shots": self.total_shots,
            "estimated_execution_time": self.estimated_execution_time,
            "estimated_cost": self.estimated_cost,
            "min_qubits": self.min_qubits,
            "required_gates": list(self.required_gates),
            "preferred_backends": self.preferred_backends,
            "batching_strategy": self.batching_strategy.value,
            "optimization_score": self.optimization_score,
            "similarity_score": self.similarity_score,
            "assigned_backend": self.assigned_backend,
            "execution_efficiency": self.execution_efficiency()
        }


@dataclass
class CompilationResult:
    """Result of circuit compilation optimization."""
    original_circuit: Dict[str, Any]
    optimized_circuit: Dict[str, Any]
    
    # Optimization metrics
    gate_count_reduction: int = 0
    depth_reduction: int = 0
    fidelity_improvement: float = 0.0
    compilation_time: float = 0.0
    
    # Backend-specific optimizations
    backend_id: str = ""
    coupling_map_optimized: bool = False
    gate_set_optimized: bool = False
    
    def optimization_ratio(self) -> float:
        """Calculate overall optimization ratio."""
        original_gates = self.original_circuit.get("gate_count", 0)
        if original_gates == 0:
            return 1.0
        
        return self.gate_count_reduction / original_gates
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gate_count_reduction": self.gate_count_reduction,
            "depth_reduction": self.depth_reduction,
            "fidelity_improvement": self.fidelity_improvement,
            "compilation_time": self.compilation_time,
            "backend_id": self.backend_id,
            "coupling_map_optimized": self.coupling_map_optimized,
            "gate_set_optimized": self.gate_set_optimized,
            "optimization_ratio": self.optimization_ratio()
        }


@dataclass
class ShotAllocation:
    """Shot allocation across multiple backends."""
    total_shots: int
    allocations: Dict[str, int]  # backend_id -> shots
    
    # Allocation strategy
    strategy: str = "quality_weighted"
    confidence_target: float = 0.95
    
    # Optimization metrics
    expected_fidelity: float = 0.0
    expected_cost: float = 0.0
    expected_completion_time: float = 0.0
    
    def allocation_efficiency(self) -> float:
        """Calculate allocation efficiency."""
        if self.total_shots == 0:
            return 0.0
        
        allocated_shots = sum(self.allocations.values())
        return allocated_shots / self.total_shots
    
    def backend_count(self) -> int:
        """Get number of backends used."""
        return len([shots for shots in self.allocations.values() if shots > 0])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_shots": self.total_shots,
            "allocations": self.allocations,
            "strategy": self.strategy,
            "confidence_target": self.confidence_target,
            "expected_fidelity": self.expected_fidelity,
            "expected_cost": self.expected_cost,
            "expected_completion_time": self.expected_completion_time,
            "allocation_efficiency": self.allocation_efficiency(),
            "backend_count": self.backend_count()
        }


class CircuitBatcher:
    """
    Intelligent circuit batching system for optimized execution.
    
    Features:
    - Multiple batching strategies (similarity, size, backend-optimal)
    - Circuit similarity analysis
    - Backend-aware batching
    - Dynamic batch sizing based on queue conditions
    """
    
    def __init__(self, config: Optional[SecureConfig] = None):
        self.config = config or SecureConfig()
        
        # Batching configuration
        self.max_batch_size = 20
        self.min_batch_size = 2
        self.similarity_threshold = 0.7
        self.max_batching_time = 300.0  # 5 minutes
        
        # Batching statistics
        self.batching_stats = {
            "batches_created": 0,
            "circuits_batched": 0,
            "average_batch_size": 0.0,
            "average_similarity": 0.0
        }
        
        logger.info("CircuitBatcher initialized")
    
    def create_batches(
        self,
        circuits: List[Dict[str, Any]],
        strategy: BatchingStrategy = BatchingStrategy.SIMILARITY_BASED,
        backend_info: Optional[Dict[str, Any]] = None
    ) -> List[CircuitBatch]:
        """Create optimized batches from circuits."""
        if not circuits:
            return []
        
        if strategy == BatchingStrategy.SIMILARITY_BASED:
            return self._create_similarity_batches(circuits)
        elif strategy == BatchingStrategy.SIZE_BASED:
            return self._create_size_batches(circuits)
        elif strategy == BatchingStrategy.BACKEND_OPTIMAL:
            return self._create_backend_optimal_batches(circuits, backend_info)
        elif strategy == BatchingStrategy.DEADLINE_AWARE:
            return self._create_deadline_aware_batches(circuits)
        else:
            return self._create_similarity_batches(circuits)  # Default
    
    def _create_similarity_batches(self, circuits: List[Dict[str, Any]]) -> List[CircuitBatch]:
        """Create batches based on circuit similarity."""
        batches = []
        remaining_circuits = circuits.copy()
        batch_id_counter = 0
        
        while remaining_circuits:
            # Start new batch with first remaining circuit
            seed_circuit = remaining_circuits.pop(0)
            batch_circuits = [seed_circuit]
            
            # Find similar circuits
            similar_circuits = []
            for i, circuit in enumerate(remaining_circuits):
                similarity = self._calculate_circuit_similarity(seed_circuit, circuit)
                if similarity >= self.similarity_threshold:
                    similar_circuits.append((i, circuit, similarity))
            
            # Sort by similarity and add to batch
            similar_circuits.sort(key=lambda x: x[2], reverse=True)
            
            for i, circuit, similarity in similar_circuits:
                if len(batch_circuits) >= self.max_batch_size:
                    break
                batch_circuits.append(circuit)
            
            # Remove added circuits from remaining
            added_indices = [i for i, _, _ in similar_circuits[:self.max_batch_size - 1]]
            for i in sorted(added_indices, reverse=True):
                remaining_circuits.pop(i)
            
            # Create batch
            if len(batch_circuits) >= self.min_batch_size:
                batch = self._create_batch_from_circuits(
                    f"similarity_batch_{batch_id_counter}",
                    batch_circuits,
                    BatchingStrategy.SIMILARITY_BASED
                )
                batches.append(batch)
                batch_id_counter += 1
            else:
                # Single circuit batch if no similar circuits found
                batch = self._create_batch_from_circuits(
                    f"single_batch_{batch_id_counter}",
                    batch_circuits,
                    BatchingStrategy.SIMILARITY_BASED
                )
                batches.append(batch)
                batch_id_counter += 1
        
        self._update_batching_stats(batches)
        return batches
    
    def _create_size_batches(self, circuits: List[Dict[str, Any]]) -> List[CircuitBatch]:
        """Create batches based on optimal size."""
        batches = []
        
        # Sort circuits by execution time estimate
        sorted_circuits = sorted(
            circuits,
            key=lambda c: c.get("estimated_execution_time", 60.0)
        )
        
        batch_id_counter = 0
        for i in range(0, len(sorted_circuits), self.max_batch_size):
            batch_circuits = sorted_circuits[i:i + self.max_batch_size]
            
            batch = self._create_batch_from_circuits(
                f"size_batch_{batch_id_counter}",
                batch_circuits,
                BatchingStrategy.SIZE_BASED
            )
            batches.append(batch)
            batch_id_counter += 1
        
        self._update_batching_stats(batches)
        return batches
    
    def _create_backend_optimal_batches(
        self,
        circuits: List[Dict[str, Any]],
        backend_info: Optional[Dict[str, Any]]
    ) -> List[CircuitBatch]:
        """Create batches optimized for specific backend."""
        if not backend_info:
            return self._create_similarity_batches(circuits)
        
        # Group circuits by backend compatibility
        compatible_circuits = []
        incompatible_circuits = []
        
        backend_qubits = backend_info.get("num_qubits", 100)
        backend_gates = set(backend_info.get("gate_set", []))
        
        for circuit in circuits:
            circuit_qubits = circuit.get("num_qubits", 1)
            circuit_gates = set(circuit.get("gates_used", []))
            
            if (circuit_qubits <= backend_qubits and
                circuit_gates.issubset(backend_gates)):
                compatible_circuits.append(circuit)
            else:
                incompatible_circuits.append(circuit)
        
        # Create batches from compatible circuits
        batches = []
        if compatible_circuits:
            compatible_batches = self._create_similarity_batches(compatible_circuits)
            for batch in compatible_batches:
                batch.batching_strategy = BatchingStrategy.BACKEND_OPTIMAL
                batch.preferred_backends = [backend_info.get("id", "")]
            batches.extend(compatible_batches)
        
        # Handle incompatible circuits separately
        if incompatible_circuits:
            incompatible_batches = self._create_similarity_batches(incompatible_circuits)
            batches.extend(incompatible_batches)
        
        return batches
    
    def _create_deadline_aware_batches(self, circuits: List[Dict[str, Any]]) -> List[CircuitBatch]:
        """Create batches considering circuit deadlines."""
        # Sort by deadline urgency
        urgent_circuits = []
        normal_circuits = []
        
        current_time = time.time()
        
        for circuit in circuits:
            deadline = circuit.get("deadline")
            if deadline and deadline - current_time < 3600:  # Less than 1 hour
                urgent_circuits.append(circuit)
            else:
                normal_circuits.append(circuit)
        
        batches = []
        
        # Process urgent circuits first with smaller batches
        if urgent_circuits:
            urgent_batch_size = min(self.max_batch_size // 2, len(urgent_circuits))
            for i in range(0, len(urgent_circuits), urgent_batch_size):
                batch_circuits = urgent_circuits[i:i + urgent_batch_size]
                batch = self._create_batch_from_circuits(
                    f"urgent_batch_{i // urgent_batch_size}",
                    batch_circuits,
                    BatchingStrategy.DEADLINE_AWARE
                )
                batches.append(batch)
        
        # Process normal circuits with regular batching
        if normal_circuits:
            normal_batches = self._create_similarity_batches(normal_circuits)
            batches.extend(normal_batches)
        
        return batches
    
    def _create_batch_from_circuits(
        self,
        batch_id: str,
        circuits: List[Dict[str, Any]],
        strategy: BatchingStrategy
    ) -> CircuitBatch:
        """Create a CircuitBatch from list of circuits."""
        # Calculate batch characteristics
        total_shots = sum(c.get("shots", 1024) for c in circuits)
        estimated_time = sum(c.get("estimated_execution_time", 60.0) for c in circuits)
        min_qubits = max(c.get("num_qubits", 1) for c in circuits)
        
        # Collect required gates
        required_gates = set()
        for circuit in circuits:
            gates = circuit.get("gates_used", [])
            required_gates.update(gates)
        
        # Calculate similarity score
        similarity_score = self._calculate_batch_similarity(circuits)
        
        batch = CircuitBatch(
            id=batch_id,
            circuits=circuits,
            total_shots=total_shots,
            estimated_execution_time=estimated_time,
            min_qubits=min_qubits,
            required_gates=required_gates,
            batching_strategy=strategy,
            similarity_score=similarity_score
        )
        
        # Calculate optimization score
        batch.optimization_score = self._calculate_batch_optimization_score(batch)
        
        return batch
    
    def _calculate_circuit_similarity(
        self,
        circuit1: Dict[str, Any],
        circuit2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two circuits."""
        similarity_score = 0.0
        
        # Qubit count similarity
        qubits1 = circuit1.get("num_qubits", 1)
        qubits2 = circuit2.get("num_qubits", 1)
        qubit_similarity = 1.0 - abs(qubits1 - qubits2) / max(qubits1, qubits2)
        similarity_score += qubit_similarity * 0.3
        
        # Gate set similarity
        gates1 = set(circuit1.get("gates_used", []))
        gates2 = set(circuit2.get("gates_used", []))
        
        if gates1 or gates2:
            gate_intersection = len(gates1 & gates2)
            gate_union = len(gates1 | gates2)
            gate_similarity = gate_intersection / gate_union if gate_union > 0 else 0
            similarity_score += gate_similarity * 0.4
        
        # Circuit depth similarity
        depth1 = circuit1.get("depth", 10)
        depth2 = circuit2.get("depth", 10)
        depth_similarity = 1.0 - abs(depth1 - depth2) / max(depth1, depth2)
        similarity_score += depth_similarity * 0.3
        
        return similarity_score
    
    def _calculate_batch_similarity(self, circuits: List[Dict[str, Any]]) -> float:
        """Calculate overall similarity within a batch."""
        if len(circuits) <= 1:
            return 1.0
        
        total_similarity = 0.0
        comparisons = 0
        
        for i in range(len(circuits)):
            for j in range(i + 1, len(circuits)):
                similarity = self._calculate_circuit_similarity(circuits[i], circuits[j])
                total_similarity += similarity
                comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0.0
    
    def _calculate_batch_optimization_score(self, batch: CircuitBatch) -> float:
        """Calculate optimization score for a batch."""
        # Size efficiency (optimal around 8-12 circuits)
        size_score = 1.0 - abs(batch.size() - 10) / 10.0
        size_score = max(0, size_score)
        
        # Similarity bonus
        similarity_score = batch.similarity_score
        
        # Execution efficiency
        execution_score = batch.execution_efficiency()
        
        return (size_score + similarity_score + execution_score) / 3.0
    
    def _update_batching_stats(self, batches: List[CircuitBatch]) -> None:
        """Update batching statistics."""
        if not batches:
            return
        
        self.batching_stats["batches_created"] += len(batches)
        total_circuits = sum(batch.size() for batch in batches)
        self.batching_stats["circuits_batched"] += total_circuits
        
        # Update averages
        self.batching_stats["average_batch_size"] = total_circuits / len(batches)
        self.batching_stats["average_similarity"] = np.mean([
            batch.similarity_score for batch in batches
        ])
    
    def get_batching_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        return self.batching_stats.copy()


class CompilationOptimizer:
    """
    Circuit compilation optimizer for multiple quantum backends.
    
    Features:
    - Backend-specific circuit optimization
    - Gate count and depth reduction
    - Coupling map optimization
    - Multi-level optimization strategies
    """
    
    def __init__(self, config: Optional[SecureConfig] = None):
        self.config = config or SecureConfig()
        
        # Compilation configuration
        self.default_optimization_level = CompilationLevel.OPTIMIZED
        self.max_compilation_time = 60.0  # seconds
        
        # Optimization techniques
        self.optimization_passes = [
            "gate_cancellation",
            "gate_fusion", 
            "circuit_depth_reduction",
            "qubit_routing",
            "gate_decomposition"
        ]
        
        # Compilation statistics
        self.compilation_stats = {
            "circuits_compiled": 0,
            "total_gate_reduction": 0,
            "total_depth_reduction": 0,
            "average_compilation_time": 0.0
        }
        
        logger.info("CompilationOptimizer initialized")
    
    async def optimize_circuit(
        self,
        circuit: Dict[str, Any],
        backend_info: Dict[str, Any],
        optimization_level: CompilationLevel = CompilationLevel.OPTIMIZED
    ) -> CompilationResult:
        """Optimize circuit for specific backend."""
        start_time = time.time()
        
        # Create copy of original circuit
        original_circuit = circuit.copy()
        optimized_circuit = circuit.copy()
        
        # Apply optimization passes based on level
        if optimization_level == CompilationLevel.BASIC:
            optimized_circuit = await self._apply_basic_optimization(
                optimized_circuit, backend_info
            )
        elif optimization_level == CompilationLevel.OPTIMIZED:
            optimized_circuit = await self._apply_standard_optimization(
                optimized_circuit, backend_info
            )
        elif optimization_level == CompilationLevel.AGGRESSIVE:
            optimized_circuit = await self._apply_aggressive_optimization(
                optimized_circuit, backend_info
            )
        
        compilation_time = time.time() - start_time
        
        # Calculate optimization metrics
        result = self._calculate_optimization_metrics(
            original_circuit, optimized_circuit, backend_info, compilation_time
        )
        
        self._update_compilation_stats(result)
        
        return result
    
    async def _apply_basic_optimization(
        self,
        circuit: Dict[str, Any],
        backend_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply basic optimization passes."""
        # Simulate basic optimization
        await asyncio.sleep(0.01)  # Simulate computation time
        
        # Basic gate cancellation
        original_gates = circuit.get("gate_count", 100)
        circuit["gate_count"] = int(original_gates * 0.95)  # 5% reduction
        
        # Basic depth reduction
        original_depth = circuit.get("depth", 20)
        circuit["depth"] = int(original_depth * 0.98)  # 2% reduction
        
        return circuit
    
    async def _apply_standard_optimization(
        self,
        circuit: Dict[str, Any],
        backend_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply standard optimization passes."""
        await asyncio.sleep(0.05)  # Simulate computation time
        
        # Gate cancellation and fusion
        original_gates = circuit.get("gate_count", 100)
        circuit["gate_count"] = int(original_gates * 0.85)  # 15% reduction
        
        # Depth reduction
        original_depth = circuit.get("depth", 20)
        circuit["depth"] = int(original_depth * 0.90)  # 10% reduction
        
        # Qubit routing optimization
        coupling_map = backend_info.get("coupling_map", [])
        if coupling_map:
            circuit["coupling_optimized"] = True
            # Additional gate reduction from routing optimization
            circuit["gate_count"] = int(circuit["gate_count"] * 0.95)
        
        return circuit
    
    async def _apply_aggressive_optimization(
        self,
        circuit: Dict[str, Any],
        backend_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply aggressive optimization passes."""
        await asyncio.sleep(0.1)  # Simulate longer computation time
        
        # All optimizations from standard level
        circuit = await self._apply_standard_optimization(circuit, backend_info)
        
        # Additional aggressive optimizations
        circuit["gate_count"] = int(circuit["gate_count"] * 0.90)  # Additional 10% reduction
        circuit["depth"] = int(circuit["depth"] * 0.85)  # Additional depth reduction
        
        # Gate set optimization
        backend_gates = set(backend_info.get("gate_set", []))
        circuit_gates = set(circuit.get("gates_used", []))
        
        if circuit_gates.issubset(backend_gates):
            circuit["gate_set_optimized"] = True
            circuit["gate_count"] = int(circuit["gate_count"] * 0.93)  # Further reduction
        
        return circuit
    
    def _calculate_optimization_metrics(
        self,
        original_circuit: Dict[str, Any],
        optimized_circuit: Dict[str, Any],
        backend_info: Dict[str, Any],
        compilation_time: float
    ) -> CompilationResult:
        """Calculate optimization metrics."""
        original_gates = original_circuit.get("gate_count", 0)
        optimized_gates = optimized_circuit.get("gate_count", 0)
        gate_reduction = original_gates - optimized_gates
        
        original_depth = original_circuit.get("depth", 0)
        optimized_depth = optimized_circuit.get("depth", 0)
        depth_reduction = original_depth - optimized_depth
        
        # Estimate fidelity improvement
        fidelity_improvement = self._estimate_fidelity_improvement(
            gate_reduction, depth_reduction, backend_info
        )
        
        return CompilationResult(
            original_circuit=original_circuit,
            optimized_circuit=optimized_circuit,
            gate_count_reduction=gate_reduction,
            depth_reduction=depth_reduction,
            fidelity_improvement=fidelity_improvement,
            compilation_time=compilation_time,
            backend_id=backend_info.get("id", ""),
            coupling_map_optimized=optimized_circuit.get("coupling_optimized", False),
            gate_set_optimized=optimized_circuit.get("gate_set_optimized", False)
        )
    
    def _estimate_fidelity_improvement(
        self,
        gate_reduction: int,
        depth_reduction: int,
        backend_info: Dict[str, Any]
    ) -> float:
        """Estimate fidelity improvement from optimization."""
        # Simple model: each gate has small error probability
        base_gate_error = 0.001  # 0.1% error per gate
        base_decoherence_error = 0.0001  # Error per depth unit
        
        gate_fidelity_improvement = gate_reduction * base_gate_error
        depth_fidelity_improvement = depth_reduction * base_decoherence_error
        
        return gate_fidelity_improvement + depth_fidelity_improvement
    
    def _update_compilation_stats(self, result: CompilationResult) -> None:
        """Update compilation statistics."""
        self.compilation_stats["circuits_compiled"] += 1
        self.compilation_stats["total_gate_reduction"] += result.gate_count_reduction
        self.compilation_stats["total_depth_reduction"] += result.depth_reduction
        
        # Update average compilation time
        current_avg = self.compilation_stats["average_compilation_time"]
        count = self.compilation_stats["circuits_compiled"]
        new_avg = ((current_avg * (count - 1)) + result.compilation_time) / count
        self.compilation_stats["average_compilation_time"] = new_avg
    
    async def optimize_batch(
        self,
        circuits: List[Dict[str, Any]],
        backend_info: Dict[str, Any],
        optimization_level: CompilationLevel = CompilationLevel.OPTIMIZED
    ) -> List[CompilationResult]:
        """Optimize a batch of circuits for backend."""
        results = []
        
        # Optimize circuits in parallel (simulate with small delays)
        tasks = []
        for circuit in circuits:
            task = self.optimize_circuit(circuit, backend_info, optimization_level)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        return results
    
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get compilation statistics."""
        return self.compilation_stats.copy()


class ShotAllocator:
    """
    Intelligent shot allocation system across multiple backends.
    
    Features:
    - Quality-weighted shot allocation
    - Cost-aware distribution
    - Confidence-based optimization
    - Dynamic reallocation based on performance
    """
    
    def __init__(self, config: Optional[SecureConfig] = None):
        self.config = config or SecureConfig()
        
        # Allocation configuration
        self.min_shots_per_backend = 100
        self.max_backends_per_job = 5
        self.confidence_target = 0.95
        
        # Allocation strategies
        self.allocation_strategies = {
            "equal": self._allocate_equal,
            "quality_weighted": self._allocate_quality_weighted,
            "cost_optimized": self._allocate_cost_optimized,
            "time_optimized": self._allocate_time_optimized,
            "confidence_maximized": self._allocate_confidence_maximized
        }
        
        # Allocation statistics
        self.allocation_stats = {
            "jobs_allocated": 0,
            "total_shots_allocated": 0,
            "backends_used": set(),
            "average_backends_per_job": 0.0
        }
        
        logger.info("ShotAllocator initialized")
    
    def allocate_shots(
        self,
        total_shots: int,
        available_backends: List[Dict[str, Any]],
        strategy: str = "quality_weighted",
        constraints: Optional[Dict[str, Any]] = None
    ) -> ShotAllocation:
        """Allocate shots across available backends."""
        if not available_backends:
            return ShotAllocation(total_shots, {})
        
        constraints = constraints or {}
        
        # Filter backends based on constraints
        suitable_backends = self._filter_suitable_backends(
            available_backends, constraints
        )
        
        if not suitable_backends:
            return ShotAllocation(total_shots, {})
        
        # Apply allocation strategy
        if strategy in self.allocation_strategies:
            allocation_func = self.allocation_strategies[strategy]
            allocations = allocation_func(total_shots, suitable_backends, constraints)
        else:
            # Default to quality weighted
            allocations = self._allocate_quality_weighted(
                total_shots, suitable_backends, constraints
            )
        
        # Create allocation result
        result = ShotAllocation(
            total_shots=total_shots,
            allocations=allocations,
            strategy=strategy,
            confidence_target=constraints.get("confidence_target", self.confidence_target)
        )
        
        # Calculate expected metrics
        result.expected_fidelity = self._calculate_expected_fidelity(
            allocations, suitable_backends
        )
        result.expected_cost = self._calculate_expected_cost(
            allocations, suitable_backends
        )
        result.expected_completion_time = self._calculate_expected_time(
            allocations, suitable_backends
        )
        
        self._update_allocation_stats(result, suitable_backends)
        
        return result
    
    def _filter_suitable_backends(
        self,
        backends: List[Dict[str, Any]],
        constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Filter backends based on constraints."""
        suitable_backends = []
        
        for backend in backends:
            # Check availability
            if not backend.get("is_available", True):
                continue
            
            # Check minimum fidelity
            min_fidelity = constraints.get("min_fidelity", 0.0)
            if backend.get("fidelity", 1.0) < min_fidelity:
                continue
            
            # Check maximum cost
            max_cost = constraints.get("max_cost_per_shot")
            if max_cost and backend.get("cost_per_shot", 0.0) > max_cost:
                continue
            
            # Check maximum queue time
            max_queue_time = constraints.get("max_queue_time")
            if max_queue_time and backend.get("queue_time", 0.0) > max_queue_time:
                continue
            
            suitable_backends.append(backend)
        
        return suitable_backends
    
    def _allocate_equal(
        self,
        total_shots: int,
        backends: List[Dict[str, Any]],
        constraints: Dict[str, Any]
    ) -> Dict[str, int]:
        """Allocate shots equally across backends."""
        num_backends = min(len(backends), self.max_backends_per_job)
        shots_per_backend = total_shots // num_backends
        remaining_shots = total_shots % num_backends
        
        allocations = {}
        for i, backend in enumerate(backends[:num_backends]):
            backend_shots = shots_per_backend
            if i < remaining_shots:
                backend_shots += 1
            
            if backend_shots >= self.min_shots_per_backend:
                allocations[backend["id"]] = backend_shots
        
        return allocations
    
    def _allocate_quality_weighted(
        self,
        total_shots: int,
        backends: List[Dict[str, Any]],
        constraints: Dict[str, Any]
    ) -> Dict[str, int]:
        """Allocate shots weighted by backend quality."""
        # Calculate quality weights
        quality_weights = {}
        total_weight = 0.0
        
        for backend in backends:
            fidelity = backend.get("fidelity", 0.9)
            reliability = backend.get("reliability", 0.95)
            
            # Combine fidelity and reliability
            quality = fidelity * reliability
            quality_weights[backend["id"]] = quality
            total_weight += quality
        
        # Allocate shots based on weights
        allocations = {}
        allocated_shots = 0
        
        for backend in backends:
            if len(allocations) >= self.max_backends_per_job:
                break
            
            backend_id = backend["id"]
            weight = quality_weights[backend_id]
            backend_shots = int((weight / total_weight) * total_shots)
            
            if backend_shots >= self.min_shots_per_backend:
                allocations[backend_id] = backend_shots
                allocated_shots += backend_shots
        
        # Distribute remaining shots to highest quality backends
        remaining_shots = total_shots - allocated_shots
        if remaining_shots > 0:
            sorted_backends = sorted(
                backends,
                key=lambda b: quality_weights[b["id"]],
                reverse=True
            )
            
            for backend in sorted_backends:
                if remaining_shots <= 0:
                    break
                backend_id = backend["id"]
                if backend_id in allocations:
                    additional_shots = min(remaining_shots, 100)
                    allocations[backend_id] += additional_shots
                    remaining_shots -= additional_shots
        
        return allocations
    
    def _allocate_cost_optimized(
        self,
        total_shots: int,
        backends: List[Dict[str, Any]],
        constraints: Dict[str, Any]
    ) -> Dict[str, int]:
        """Allocate shots to minimize total cost."""
        # Sort backends by cost per shot
        sorted_backends = sorted(
            backends,
            key=lambda b: b.get("cost_per_shot", 0.0)
        )
        
        allocations = {}
        remaining_shots = total_shots
        
        for backend in sorted_backends:
            if len(allocations) >= self.max_backends_per_job:
                break
            
            # Allocate as many shots as possible to lowest cost backend
            backend_capacity = backend.get("max_shots_per_job", remaining_shots)
            backend_shots = min(remaining_shots, backend_capacity)
            
            if backend_shots >= self.min_shots_per_backend:
                allocations[backend["id"]] = backend_shots
                remaining_shots -= backend_shots
            
            if remaining_shots <= 0:
                break
        
        return allocations
    
    def _allocate_time_optimized(
        self,
        total_shots: int,
        backends: List[Dict[str, Any]],
        constraints: Dict[str, Any]
    ) -> Dict[str, int]:
        """Allocate shots to minimize total execution time."""
        # Sort backends by estimated execution time
        sorted_backends = sorted(
            backends,
            key=lambda b: b.get("queue_time", 0.0) + b.get("execution_time", 60.0)
        )
        
        # Use parallel execution approach - distribute across fastest backends
        num_backends = min(len(sorted_backends), self.max_backends_per_job)
        fastest_backends = sorted_backends[:num_backends]
        
        return self._allocate_equal(total_shots, fastest_backends, constraints)
    
    def _allocate_confidence_maximized(
        self,
        total_shots: int,
        backends: List[Dict[str, Any]],
        constraints: Dict[str, Any]
    ) -> Dict[str, int]:
        """Allocate shots to maximize result confidence."""
        # This is more complex - would involve statistical analysis
        # For now, use quality-weighted as approximation
        return self._allocate_quality_weighted(total_shots, backends, constraints)
    
    def _calculate_expected_fidelity(
        self,
        allocations: Dict[str, int],
        backends: List[Dict[str, Any]]
    ) -> float:
        """Calculate expected overall fidelity."""
        if not allocations:
            return 0.0
        
        backend_map = {b["id"]: b for b in backends}
        total_shots = sum(allocations.values())
        weighted_fidelity = 0.0
        
        for backend_id, shots in allocations.items():
            if backend_id in backend_map:
                backend_fidelity = backend_map[backend_id].get("fidelity", 0.9)
                weight = shots / total_shots
                weighted_fidelity += backend_fidelity * weight
        
        return weighted_fidelity
    
    def _calculate_expected_cost(
        self,
        allocations: Dict[str, int],
        backends: List[Dict[str, Any]]
    ) -> float:
        """Calculate expected total cost."""
        backend_map = {b["id"]: b for b in backends}
        total_cost = 0.0
        
        for backend_id, shots in allocations.items():
            if backend_id in backend_map:
                cost_per_shot = backend_map[backend_id].get("cost_per_shot", 0.0)
                total_cost += shots * cost_per_shot
        
        return total_cost
    
    def _calculate_expected_time(
        self,
        allocations: Dict[str, int],
        backends: List[Dict[str, Any]]
    ) -> float:
        """Calculate expected completion time (parallel execution)."""
        backend_map = {b["id"]: b for b in backends}
        max_time = 0.0
        
        for backend_id, shots in allocations.items():
            if backend_id in backend_map:
                backend = backend_map[backend_id]
                queue_time = backend.get("queue_time", 0.0)
                
                # Estimate execution time based on shots
                base_execution_time = backend.get("execution_time", 60.0)
                shot_scaling_factor = shots / 1024  # Normalize to 1024 shots
                execution_time = base_execution_time * shot_scaling_factor
                
                total_time = queue_time + execution_time
                max_time = max(max_time, total_time)
        
        return max_time
    
    def _update_allocation_stats(
        self,
        allocation: ShotAllocation,
        backends: List[Dict[str, Any]]
    ) -> None:
        """Update allocation statistics."""
        self.allocation_stats["jobs_allocated"] += 1
        self.allocation_stats["total_shots_allocated"] += allocation.total_shots
        
        # Track backend usage
        for backend_id in allocation.allocations:
            self.allocation_stats["backends_used"].add(backend_id)
        
        # Update average backends per job
        current_avg = self.allocation_stats["average_backends_per_job"]
        count = self.allocation_stats["jobs_allocated"]
        new_backends_count = allocation.backend_count()
        new_avg = ((current_avg * (count - 1)) + new_backends_count) / count
        self.allocation_stats["average_backends_per_job"] = new_avg
    
    def rebalance_allocation(
        self,
        original_allocation: ShotAllocation,
        backend_performance: Dict[str, Dict[str, float]]
    ) -> ShotAllocation:
        """Rebalance shot allocation based on observed performance."""
        # Calculate performance scores
        performance_scores = {}
        for backend_id, perf_data in backend_performance.items():
            if backend_id in original_allocation.allocations:
                # Simple performance score based on success rate and execution time
                success_rate = perf_data.get("success_rate", 1.0)
                relative_speed = 1.0 / max(perf_data.get("execution_time", 60.0), 1.0)
                performance_scores[backend_id] = success_rate * relative_speed
        
        if not performance_scores:
            return original_allocation
        
        # Rebalance based on performance
        total_score = sum(performance_scores.values())
        new_allocations = {}
        
        for backend_id, score in performance_scores.items():
            weight = score / total_score
            new_shots = int(weight * original_allocation.total_shots)
            
            if new_shots >= self.min_shots_per_backend:
                new_allocations[backend_id] = new_shots
        
        # Create new allocation
        rebalanced = ShotAllocation(
            total_shots=original_allocation.total_shots,
            allocations=new_allocations,
            strategy=f"rebalanced_{original_allocation.strategy}"
        )
        
        return rebalanced
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get allocation statistics."""
        stats = self.allocation_stats.copy()
        stats["backends_used"] = len(stats["backends_used"])  # Convert set to count
        return stats


class ResourceOptimizer:
    """
    Main resource optimizer coordinating all optimization components.
    
    This class orchestrates circuit batching, compilation optimization, and
    shot allocation to provide comprehensive resource optimization for
    quantum error mitigation workloads.
    
    Example:
        >>> optimizer = ResourceOptimizer()
        >>> 
        >>> # Optimize a set of circuits for execution
        >>> circuits = [circuit1, circuit2, circuit3]
        >>> backends = [backend_info1, backend_info2]
        >>> optimization = await optimizer.optimize_workload(circuits, backends)
    """
    
    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        config: Optional[SecureConfig] = None
    ):
        self.strategy = strategy
        self.config = config or SecureConfig()
        
        # Core components
        self.circuit_batcher = CircuitBatcher(config)
        self.compilation_optimizer = CompilationOptimizer(config)
        self.shot_allocator = ShotAllocator(config)
        
        # Optimization history
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Overall statistics
        self.optimizer_stats = {
            "workloads_optimized": 0,
            "circuits_processed": 0,
            "total_optimization_time": 0.0,
            "average_efficiency_gain": 0.0
        }
        
        logger.info(f"ResourceOptimizer initialized with strategy: {strategy.value}")
    
    async def optimize_workload(
        self,
        circuits: List[Dict[str, Any]],
        available_backends: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize complete workload including batching, compilation, and allocation."""
        start_time = time.time()
        
        constraints = constraints or {}
        
        logger.info(f"Optimizing workload: {len(circuits)} circuits, {len(available_backends)} backends")
        
        # Step 1: Create optimized circuit batches
        batching_strategy = self._select_batching_strategy(constraints)
        batches = self.circuit_batcher.create_batches(
            circuits, batching_strategy, available_backends[0] if available_backends else None
        )
        
        logger.info(f"Created {len(batches)} optimized batches")
        
        # Step 2: Optimize compilation for each backend
        compilation_results = {}
        for backend in available_backends:
            backend_results = []
            for batch in batches:
                if batch.is_compatible_with_backend(backend):
                    # Optimize representative circuit from batch
                    if batch.circuits:
                        result = await self.compilation_optimizer.optimize_circuit(
                            batch.circuits[0], backend, self._select_compilation_level()
                        )
                        backend_results.append(result)
            
            compilation_results[backend["id"]] = backend_results
        
        # Step 3: Optimize shot allocation across backends
        allocation_results = {}
        for batch in batches:
            suitable_backends = [
                backend for backend in available_backends
                if batch.is_compatible_with_backend(backend)
            ]
            
            if suitable_backends:
                allocation = self.shot_allocator.allocate_shots(
                    batch.total_shots,
                    suitable_backends,
                    self._select_allocation_strategy(),
                    constraints
                )
                allocation_results[batch.id] = allocation
        
        # Step 4: Calculate optimization metrics
        optimization_time = time.time() - start_time
        efficiency_metrics = self._calculate_efficiency_metrics(
            batches, compilation_results, allocation_results
        )
        
        # Create optimization result
        result = {
            "optimization_strategy": self.strategy.value,
            "optimization_time": optimization_time,
            "original_circuits": len(circuits),
            "optimized_batches": len(batches),
            "batches": [batch.to_dict() for batch in batches],
            "compilation_results": {
                backend_id: [r.to_dict() for r in results]
                for backend_id, results in compilation_results.items()
            },
            "shot_allocations": {
                batch_id: allocation.to_dict()
                for batch_id, allocation in allocation_results.items()
            },
            "efficiency_metrics": efficiency_metrics,
            "recommendations": self._generate_optimization_recommendations(
                batches, compilation_results, allocation_results
            )
        }
        
        # Update statistics
        self._update_optimizer_stats(result)
        
        # Store in history
        self.optimization_history.append({
            "timestamp": time.time(),
            "result": result
        })
        
        return result
    
    def _select_batching_strategy(self, constraints: Dict[str, Any]) -> BatchingStrategy:
        """Select batching strategy based on optimization strategy and constraints."""
        if self.strategy == OptimizationStrategy.MINIMIZE_TIME:
            return BatchingStrategy.DEADLINE_AWARE
        elif self.strategy == OptimizationStrategy.MINIMIZE_COST:
            return BatchingStrategy.SIZE_BASED
        elif self.strategy == OptimizationStrategy.MAXIMIZE_FIDELITY:
            return BatchingStrategy.BACKEND_OPTIMAL
        else:  # BALANCED
            return BatchingStrategy.SIMILARITY_BASED
    
    def _select_compilation_level(self) -> CompilationLevel:
        """Select compilation level based on optimization strategy."""
        if self.strategy == OptimizationStrategy.MINIMIZE_TIME:
            return CompilationLevel.BASIC
        elif self.strategy == OptimizationStrategy.MAXIMIZE_FIDELITY:
            return CompilationLevel.AGGRESSIVE
        else:
            return CompilationLevel.OPTIMIZED
    
    def _select_allocation_strategy(self) -> str:
        """Select shot allocation strategy."""
        if self.strategy == OptimizationStrategy.MINIMIZE_COST:
            return "cost_optimized"
        elif self.strategy == OptimizationStrategy.MINIMIZE_TIME:
            return "time_optimized"
        elif self.strategy == OptimizationStrategy.MAXIMIZE_FIDELITY:
            return "quality_weighted"
        else:  # BALANCED
            return "quality_weighted"
    
    def _calculate_efficiency_metrics(
        self,
        batches: List[CircuitBatch],
        compilation_results: Dict[str, List[CompilationResult]],
        allocation_results: Dict[str, ShotAllocation]
    ) -> Dict[str, Any]:
        """Calculate overall efficiency metrics."""
        metrics = {}
        
        # Batching efficiency
        if batches:
            avg_batch_efficiency = np.mean([batch.execution_efficiency() for batch in batches])
            metrics["batching_efficiency"] = avg_batch_efficiency
            metrics["batch_size_distribution"] = {
                "mean": np.mean([batch.size() for batch in batches]),
                "std": np.std([batch.size() for batch in batches]),
                "min": min([batch.size() for batch in batches]),
                "max": max([batch.size() for batch in batches])
            }
        
        # Compilation efficiency
        all_compilation_results = []
        for results in compilation_results.values():
            all_compilation_results.extend(results)
        
        if all_compilation_results:
            avg_gate_reduction = np.mean([r.gate_count_reduction for r in all_compilation_results])
            avg_depth_reduction = np.mean([r.depth_reduction for r in all_compilation_results])
            avg_compilation_time = np.mean([r.compilation_time for r in all_compilation_results])
            
            metrics["compilation_efficiency"] = {
                "average_gate_reduction": avg_gate_reduction,
                "average_depth_reduction": avg_depth_reduction,
                "average_compilation_time": avg_compilation_time,
                "total_circuits_compiled": len(all_compilation_results)
            }
        
        # Allocation efficiency
        if allocation_results:
            allocations = list(allocation_results.values())
            avg_allocation_efficiency = np.mean([a.allocation_efficiency() for a in allocations])
            avg_backends_used = np.mean([a.backend_count() for a in allocations])
            
            metrics["allocation_efficiency"] = {
                "average_allocation_efficiency": avg_allocation_efficiency,
                "average_backends_per_job": avg_backends_used,
                "total_shots_allocated": sum(a.total_shots for a in allocations)
            }
        
        return metrics
    
    def _generate_optimization_recommendations(
        self,
        batches: List[CircuitBatch],
        compilation_results: Dict[str, List[CompilationResult]],
        allocation_results: Dict[str, ShotAllocation]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Batching recommendations
        if batches:
            avg_batch_size = np.mean([batch.size() for batch in batches])
            if avg_batch_size < 3:
                recommendations.append("Consider increasing batch sizes for better efficiency")
            elif avg_batch_size > 15:
                recommendations.append("Consider smaller batch sizes to reduce queue times")
            
            low_similarity_batches = [b for b in batches if b.similarity_score < 0.5]
            if len(low_similarity_batches) > len(batches) * 0.3:
                recommendations.append("Many batches have low similarity - consider different batching strategy")
        
        # Compilation recommendations
        all_results = []
        for results in compilation_results.values():
            all_results.extend(results)
        
        if all_results:
            avg_optimization_ratio = np.mean([r.optimization_ratio() for r in all_results])
            if avg_optimization_ratio < 0.1:
                recommendations.append("Low compilation optimization gains - circuits may already be optimal")
            elif avg_optimization_ratio > 0.3:
                recommendations.append("High optimization potential - consider aggressive compilation")
        
        # Allocation recommendations
        if allocation_results:
            allocations = list(allocation_results.values())
            single_backend_jobs = [a for a in allocations if a.backend_count() == 1]
            
            if len(single_backend_jobs) > len(allocations) * 0.8:
                recommendations.append("Most jobs use single backend - consider multi-backend allocation")
        
        return recommendations
    
    def _update_optimizer_stats(self, result: Dict[str, Any]) -> None:
        """Update optimizer statistics."""
        self.optimizer_stats["workloads_optimized"] += 1
        self.optimizer_stats["circuits_processed"] += result["original_circuits"]
        self.optimizer_stats["total_optimization_time"] += result["optimization_time"]
        
        # Update average efficiency gain
        efficiency_metrics = result.get("efficiency_metrics", {})
        batching_efficiency = efficiency_metrics.get("batching_efficiency", 0.5)
        
        current_avg = self.optimizer_stats["average_efficiency_gain"]
        count = self.optimizer_stats["workloads_optimized"]
        new_avg = ((current_avg * (count - 1)) + batching_efficiency) / count
        self.optimizer_stats["average_efficiency_gain"] = new_avg
    
    async def optimize_single_job(
        self,
        circuit: Dict[str, Any],
        backend: Dict[str, Any],
        shots: int = 1024,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize a single job (circuit + backend + shots)."""
        # Compile circuit for backend
        compilation_result = await self.compilation_optimizer.optimize_circuit(
            circuit, backend, self._select_compilation_level()
        )
        
        # Optimize shot allocation (single backend case)
        allocation = self.shot_allocator.allocate_shots(
            shots, [backend], self._select_allocation_strategy(), constraints
        )
        
        return {
            "compilation_result": compilation_result.to_dict(),
            "shot_allocation": allocation.to_dict(),
            "expected_fidelity": allocation.expected_fidelity,
            "expected_cost": allocation.expected_cost,
            "expected_time": allocation.expected_completion_time
        }
    
    def get_optimization_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get optimization analytics for specified period."""
        cutoff_time = time.time() - days * 24 * 3600
        recent_optimizations = [
            opt for opt in self.optimization_history
            if opt["timestamp"] > cutoff_time
        ]
        
        if not recent_optimizations:
            return {"message": "No recent optimizations"}
        
        # Analyze trends
        optimization_times = [opt["result"]["optimization_time"] for opt in recent_optimizations]
        efficiency_gains = []
        
        for opt in recent_optimizations:
            metrics = opt["result"].get("efficiency_metrics", {})
            batching_eff = metrics.get("batching_efficiency", 0.5)
            efficiency_gains.append(batching_eff)
        
        return {
            "analysis_period_days": days,
            "total_optimizations": len(recent_optimizations),
            "optimization_time_stats": {
                "mean": np.mean(optimization_times),
                "std": np.std(optimization_times),
                "min": np.min(optimization_times),
                "max": np.max(optimization_times)
            },
            "efficiency_trend": {
                "mean_efficiency": np.mean(efficiency_gains),
                "trend": "improving" if np.mean(efficiency_gains[-5:]) > np.mean(efficiency_gains[:5]) else "stable"
            },
            "optimizer_stats": self.optimizer_stats.copy(),
            "component_stats": {
                "batching": self.circuit_batcher.get_batching_stats(),
                "compilation": self.compilation_optimizer.get_compilation_stats(),
                "allocation": self.shot_allocator.get_allocation_stats()
            }
        }
    
    def set_optimization_strategy(self, strategy: OptimizationStrategy) -> None:
        """Change optimization strategy."""
        self.strategy = strategy
        logger.info(f"Optimization strategy changed to: {strategy.value}")
    
    def get_component_stats(self) -> Dict[str, Any]:
        """Get statistics from all optimizer components."""
        return {
            "circuit_batcher": self.circuit_batcher.get_batching_stats(),
            "compilation_optimizer": self.compilation_optimizer.get_compilation_stats(),
            "shot_allocator": self.shot_allocator.get_allocation_stats(),
            "resource_optimizer": self.optimizer_stats.copy()
        }