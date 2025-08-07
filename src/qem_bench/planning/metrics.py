"""
Planning Metrics and Analytics

Comprehensive metrics collection and analysis for quantum-inspired
task planning performance and effectiveness.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import statistics

from .core import Task, TaskState


class ComplexityMeasure(Enum):
    """Task complexity measurement methods"""
    QUANTUM_VOLUME = "quantum_volume"
    ENTANGLEMENT_ENTROPY = "entanglement_entropy"
    CIRCUIT_DEPTH = "circuit_depth"
    RESOURCE_COMPLEXITY = "resource_complexity"
    DEPENDENCY_COMPLEXITY = "dependency_complexity"


@dataclass
class TaskComplexity:
    """Comprehensive task complexity metrics"""
    task_id: str
    quantum_volume: float = 0.0
    entanglement_entropy: float = 0.0
    circuit_depth: float = 0.0
    resource_complexity: float = 0.0
    dependency_complexity: float = 0.0
    overall_complexity: float = 0.0
    complexity_variance: float = 0.0


@dataclass
class PlanningMetrics:
    """Planning performance and quality metrics"""
    
    # Planning efficiency
    planning_time: float = 0.0
    optimization_iterations: int = 0
    convergence_achieved: bool = False
    quantum_fidelity: float = 0.0
    
    # Schedule quality
    total_completion_time: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    dependency_satisfaction: float = 1.0
    task_priority_score: float = 0.0
    
    # Quantum metrics
    superposition_width: float = 0.0
    entanglement_strength: float = 0.0
    interference_effectiveness: float = 0.0
    measurement_accuracy: float = 0.0
    
    # Performance comparison
    classical_baseline_time: Optional[float] = None
    quantum_speedup: Optional[float] = None
    energy_efficiency: float = 0.0
    
    # Reliability metrics
    task_failure_rate: float = 0.0
    rescheduling_frequency: float = 0.0
    adaptation_responsiveness: float = 0.0


class PlanningAnalyzer:
    """
    Advanced analytics for quantum-inspired planning
    
    Provides comprehensive analysis of planning performance,
    quantum metrics, and optimization effectiveness.
    """
    
    def __init__(self):
        self.task_history: List[Tuple[Task, datetime, datetime]] = []
        self.planning_history: List[PlanningMetrics] = []
        self.complexity_cache: Dict[str, TaskComplexity] = {}
        self._classical_baselines: Dict[str, float] = {}
    
    def analyze_task_complexity(self, task: Task, 
                              measure: ComplexityMeasure = ComplexityMeasure.QUANTUM_VOLUME) -> TaskComplexity:
        """
        Analyze task complexity using quantum-inspired metrics
        
        Args:
            task: Task to analyze
            measure: Primary complexity measure to use
            
        Returns:
            TaskComplexity with detailed metrics
        """
        if task.id in self.complexity_cache:
            return self.complexity_cache[task.id]
        
        complexity = TaskComplexity(task_id=task.id)
        
        # Quantum volume: circuit width × depth
        complexity.quantum_volume = self._compute_quantum_volume(task)
        
        # Entanglement entropy: dependency interconnection measure
        complexity.entanglement_entropy = self._compute_entanglement_entropy(task)
        
        # Circuit depth: execution complexity
        complexity.circuit_depth = self._compute_circuit_depth(task)
        
        # Resource complexity: resource requirement spread
        complexity.resource_complexity = self._compute_resource_complexity(task)
        
        # Dependency complexity: dependency graph complexity
        complexity.dependency_complexity = self._compute_dependency_complexity(task)
        
        # Overall complexity: weighted combination
        complexity.overall_complexity = self._compute_overall_complexity(complexity)
        
        # Complexity variance: uncertainty in complexity estimate
        complexity.complexity_variance = self._compute_complexity_variance(complexity)
        
        self.complexity_cache[task.id] = complexity
        return complexity
    
    @jax.jit
    def _compute_quantum_volume(self, task: Task) -> float:
        """Compute quantum volume-inspired complexity"""
        # Analogous to quantum volume: width × depth
        width = len(task.resources) if task.resources else 1
        depth = task.complexity
        
        # Add quantum fluctuations
        base_volume = width * depth
        quantum_factor = jnp.log2(base_volume + 1)  # Log scaling like QV
        
        return float(base_volume * quantum_factor)
    
    def _compute_entanglement_entropy(self, task: Task) -> float:
        """Compute entanglement entropy from dependencies"""
        if not task.dependencies:
            return 0.0
        
        # Shannon entropy of dependency distribution
        n_deps = len(task.dependencies)
        if n_deps <= 1:
            return 0.0
        
        # Uniform distribution assumption
        prob = 1.0 / n_deps
        entropy = -n_deps * prob * np.log2(prob)
        
        return entropy
    
    def _compute_circuit_depth(self, task: Task) -> float:
        """Compute circuit depth analogy"""
        # Combination of complexity and dependency depth
        base_depth = task.complexity
        
        # Add dependency depth contribution
        dependency_depth = len(task.dependencies) * 0.5
        
        return base_depth + dependency_depth
    
    def _compute_resource_complexity(self, task: Task) -> float:
        """Compute resource requirement complexity"""
        if not task.resources:
            return 0.0
        
        resource_values = list(task.resources.values())
        
        # Statistical measures of resource spread
        mean_resource = np.mean(resource_values)
        std_resource = np.std(resource_values) if len(resource_values) > 1 else 0.0
        
        # Complexity increases with both magnitude and spread
        return mean_resource * (1 + std_resource / (mean_resource + 1e-6))
    
    def _compute_dependency_complexity(self, task: Task) -> float:
        """Compute dependency graph complexity"""
        # Simple measure based on dependency count
        n_deps = len(task.dependencies)
        
        if n_deps == 0:
            return 0.0
        
        # Complexity grows super-linearly with dependencies
        return n_deps * np.log(n_deps + 1)
    
    def _compute_overall_complexity(self, complexity: TaskComplexity) -> float:
        """Compute weighted overall complexity score"""
        weights = {
            'quantum_volume': 0.3,
            'entanglement_entropy': 0.2,
            'circuit_depth': 0.2,
            'resource_complexity': 0.15,
            'dependency_complexity': 0.15
        }
        
        overall = (
            weights['quantum_volume'] * complexity.quantum_volume +
            weights['entanglement_entropy'] * complexity.entanglement_entropy +
            weights['circuit_depth'] * complexity.circuit_depth +
            weights['resource_complexity'] * complexity.resource_complexity +
            weights['dependency_complexity'] * complexity.dependency_complexity
        )
        
        return overall
    
    def _compute_complexity_variance(self, complexity: TaskComplexity) -> float:
        """Compute variance in complexity estimates"""
        complexity_values = [
            complexity.quantum_volume,
            complexity.entanglement_entropy * 5,  # Scale to similar range
            complexity.circuit_depth,
            complexity.resource_complexity * 2,
            complexity.dependency_complexity
        ]
        
        return float(np.var(complexity_values))
    
    def analyze_planning_performance(self, tasks: Dict[str, Task], 
                                   planning_result: Dict[str, Any]) -> PlanningMetrics:
        """
        Analyze planning performance and generate comprehensive metrics
        
        Args:
            tasks: Tasks that were planned
            planning_result: Result from planning algorithm
            
        Returns:
            PlanningMetrics with detailed analysis
        """
        metrics = PlanningMetrics()
        
        # Basic planning metrics
        metrics.planning_time = planning_result.get('computation_time', 0.0)
        metrics.optimization_iterations = planning_result.get('iterations', 0)
        metrics.convergence_achieved = planning_result.get('convergence_achieved', False)
        metrics.quantum_fidelity = planning_result.get('quantum_fidelity', 0.0)
        
        # Schedule quality analysis
        schedule = planning_result.get('schedule', [])
        if schedule:
            metrics.total_completion_time = schedule[-1]['end_time'] if schedule else 0.0
            metrics.resource_utilization = self._analyze_resource_utilization(schedule)
            metrics.dependency_satisfaction = self._analyze_dependency_satisfaction(tasks, schedule)
            metrics.task_priority_score = self._analyze_priority_satisfaction(tasks, schedule)
        
        # Quantum metrics analysis
        metrics.superposition_width = self._analyze_superposition_effectiveness(planning_result)
        metrics.entanglement_strength = self._analyze_entanglement_utilization(tasks)
        metrics.interference_effectiveness = self._analyze_interference_quality(planning_result)
        metrics.measurement_accuracy = self._analyze_measurement_precision(planning_result)
        
        # Performance comparison
        classical_time = self._estimate_classical_baseline(tasks)
        if classical_time > 0:
            metrics.classical_baseline_time = classical_time
            metrics.quantum_speedup = classical_time / metrics.total_completion_time if metrics.total_completion_time > 0 else 1.0
        
        # Energy efficiency (quantum computation analogy)
        metrics.energy_efficiency = self._compute_energy_efficiency(tasks, planning_result)
        
        # Store in history
        self.planning_history.append(metrics)
        
        return metrics
    
    def _analyze_resource_utilization(self, schedule: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze resource utilization patterns"""
        resource_usage = {}
        resource_time = {}
        
        for event in schedule:
            duration = event['end_time'] - event['start_time']
            
            for resource_id, amount in event.get('resources', {}).items():
                if resource_id not in resource_usage:
                    resource_usage[resource_id] = 0.0
                    resource_time[resource_id] = 0.0
                
                resource_usage[resource_id] += amount * duration
                resource_time[resource_id] += duration
        
        # Calculate utilization rates
        utilization = {}
        for resource_id in resource_usage:
            if resource_time[resource_id] > 0:
                utilization[resource_id] = resource_usage[resource_id] / resource_time[resource_id]
            else:
                utilization[resource_id] = 0.0
        
        return utilization
    
    def _analyze_dependency_satisfaction(self, tasks: Dict[str, Task], 
                                       schedule: List[Dict[str, Any]]) -> float:
        """Analyze how well dependencies are satisfied"""
        if not schedule:
            return 1.0
        
        violations = 0
        total_dependencies = 0
        
        # Build execution order
        execution_order = {event['task_id']: i for i, event in enumerate(schedule)}
        
        for task in tasks.values():
            for dep_id in task.dependencies:
                total_dependencies += 1
                
                task_pos = execution_order.get(task.id, -1)
                dep_pos = execution_order.get(dep_id, -1)
                
                # Dependency violation if dependency executes after dependent
                if dep_pos >= task_pos or dep_pos == -1:
                    violations += 1
        
        if total_dependencies == 0:
            return 1.0
        
        return 1.0 - (violations / total_dependencies)
    
    def _analyze_priority_satisfaction(self, tasks: Dict[str, Task],
                                     schedule: List[Dict[str, Any]]) -> float:
        """Analyze how well task priorities are respected"""
        if not schedule:
            return 1.0
        
        # Calculate weighted priority score
        total_weight = 0.0
        weighted_satisfaction = 0.0
        
        for i, event in enumerate(schedule):
            task = tasks.get(event['task_id'])
            if task:
                # Higher priority tasks should execute earlier
                position_score = 1.0 - (i / len(schedule))  # Earlier = higher score
                priority_weight = task.priority
                
                weighted_satisfaction += position_score * priority_weight
                total_weight += priority_weight
        
        return weighted_satisfaction / total_weight if total_weight > 0 else 1.0
    
    def _analyze_superposition_effectiveness(self, planning_result: Dict[str, Any]) -> float:
        """Analyze effectiveness of quantum superposition in exploration"""
        # Proxy: how much the quantum solution differs from naive ordering
        quantum_fidelity = planning_result.get('quantum_fidelity', 1.0)
        
        # Lower fidelity indicates more exploration (superposition)
        superposition_width = 1.0 - quantum_fidelity
        
        return max(0.0, min(1.0, superposition_width))
    
    def _analyze_entanglement_utilization(self, tasks: Dict[str, Task]) -> float:
        """Analyze how well task entanglements (dependencies) are utilized"""
        if not tasks:
            return 0.0
        
        total_tasks = len(tasks)
        entangled_tasks = sum(1 for task in tasks.values() if task.dependencies)
        
        # Entanglement strength proportional to connectivity
        connectivity = entangled_tasks / total_tasks
        
        # Also consider average entanglement degree
        avg_dependencies = np.mean([len(task.dependencies) for task in tasks.values()])
        max_possible_deps = total_tasks - 1
        
        if max_possible_deps > 0:
            entanglement_density = avg_dependencies / max_possible_deps
            return connectivity * entanglement_density
        
        return connectivity
    
    def _analyze_interference_quality(self, planning_result: Dict[str, Any]) -> float:
        """Analyze quantum interference effectiveness"""
        # Interference quality based on convergence speed and optimality
        iterations = planning_result.get('iterations', 1)
        max_iterations = planning_result.get('max_iterations', iterations)
        convergence_achieved = planning_result.get('convergence_achieved', False)
        
        # Fast convergence indicates good interference
        convergence_speed = 1.0 - (iterations / max_iterations)
        
        # Weight by whether convergence was achieved
        interference_quality = convergence_speed * (1.0 if convergence_achieved else 0.5)
        
        return max(0.0, min(1.0, interference_quality))
    
    def _analyze_measurement_precision(self, planning_result: Dict[str, Any]) -> float:
        """Analyze precision of quantum measurement (solution extraction)"""
        # Measurement precision related to solution confidence
        quantum_fidelity = planning_result.get('quantum_fidelity', 0.0)
        convergence_achieved = planning_result.get('convergence_achieved', False)
        
        # High fidelity + convergence = precise measurement
        precision = quantum_fidelity * (1.0 if convergence_achieved else 0.8)
        
        return max(0.0, min(1.0, precision))
    
    def _estimate_classical_baseline(self, tasks: Dict[str, Task]) -> float:
        """Estimate classical algorithm completion time"""
        # Simple baseline: sum of task durations with dependency delays
        task_list = list(tasks.values())
        
        # Sort by priority (classical greedy approach)
        sorted_tasks = sorted(task_list, key=lambda t: -t.priority)
        
        total_time = 0.0
        completed = set()
        
        for task in sorted_tasks:
            # Add dependency delay
            dependency_delay = len(task.dependencies) * 0.1
            
            # Add task duration
            task_duration = task.duration_estimate or 1.0
            
            total_time += dependency_delay + task_duration
        
        return total_time
    
    def _compute_energy_efficiency(self, tasks: Dict[str, Task], 
                                 planning_result: Dict[str, Any]) -> float:
        """Compute energy efficiency of quantum planning"""
        # Quantum energy analogy: iterations × fidelity / completion_time
        iterations = planning_result.get('iterations', 1)
        fidelity = planning_result.get('quantum_fidelity', 1.0)
        completion_time = planning_result.get('total_time', 1.0)
        
        if completion_time <= 0:
            return 0.0
        
        # Higher fidelity with fewer iterations and faster completion = efficient
        efficiency = fidelity / (iterations * completion_time + 1e-6)
        
        return min(1.0, efficiency * 100)  # Scale for readability
    
    def generate_complexity_report(self, tasks: Dict[str, Task]) -> Dict[str, Any]:
        """Generate comprehensive complexity analysis report"""
        complexities = [self.analyze_task_complexity(task) for task in tasks.values()]
        
        if not complexities:
            return {"error": "No tasks to analyze"}
        
        # Overall statistics
        overall_complexities = [c.overall_complexity for c in complexities]
        
        report = {
            "total_tasks": len(tasks),
            "complexity_statistics": {
                "mean": statistics.mean(overall_complexities),
                "median": statistics.median(overall_complexities),
                "stdev": statistics.stdev(overall_complexities) if len(overall_complexities) > 1 else 0.0,
                "min": min(overall_complexities),
                "max": max(overall_complexities)
            },
            "quantum_volume_distribution": [c.quantum_volume for c in complexities],
            "entanglement_entropy_distribution": [c.entanglement_entropy for c in complexities],
            "most_complex_tasks": sorted(complexities, key=lambda c: c.overall_complexity, reverse=True)[:5],
            "simplest_tasks": sorted(complexities, key=lambda c: c.overall_complexity)[:5]
        }
        
        return report
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report"""
        if not self.planning_history:
            return {"error": "No planning history available"}
        
        recent_metrics = self.planning_history[-10:]  # Last 10 planning sessions
        
        report = {
            "planning_sessions": len(self.planning_history),
            "recent_performance": {
                "avg_planning_time": statistics.mean([m.planning_time for m in recent_metrics]),
                "avg_quantum_fidelity": statistics.mean([m.quantum_fidelity for m in recent_metrics]),
                "convergence_rate": sum(m.convergence_achieved for m in recent_metrics) / len(recent_metrics),
                "avg_quantum_speedup": statistics.mean([m.quantum_speedup for m in recent_metrics if m.quantum_speedup]),
            },
            "quantum_metrics": {
                "avg_superposition_width": statistics.mean([m.superposition_width for m in recent_metrics]),
                "avg_entanglement_strength": statistics.mean([m.entanglement_strength for m in recent_metrics]),
                "avg_interference_effectiveness": statistics.mean([m.interference_effectiveness for m in recent_metrics]),
                "avg_measurement_accuracy": statistics.mean([m.measurement_accuracy for m in recent_metrics]),
            },
            "quality_metrics": {
                "avg_dependency_satisfaction": statistics.mean([m.dependency_satisfaction for m in recent_metrics]),
                "avg_priority_score": statistics.mean([m.task_priority_score for m in recent_metrics]),
                "avg_energy_efficiency": statistics.mean([m.energy_efficiency for m in recent_metrics]),
            }
        }
        
        return report