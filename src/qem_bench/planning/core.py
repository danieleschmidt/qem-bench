"""
Core Quantum-Inspired Task Planner

Implements quantum optimization principles for classical task planning,
using superposition, entanglement, and interference concepts.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

# from ..jax.states import QuantumState  # Not available
from ..optimization import PerformanceOptimizer
from ..metrics import MetricsCollector


class TaskState(Enum):
    """Task execution states using quantum state analogy"""
    SUPERPOSITION = "superposition"  # Multiple potential solutions
    ENTANGLED = "entangled"         # Dependent on other tasks
    COLLAPSED = "collapsed"         # Solution determined
    MEASURED = "measured"           # Task completed


@dataclass
class Task:
    """Quantum-inspired task representation"""
    id: str
    name: str
    complexity: float  # Quantum "energy" level
    dependencies: List[str] = field(default_factory=list)
    resources: Dict[str, float] = field(default_factory=dict)
    duration_estimate: float = 0.0
    priority: float = 1.0
    state: TaskState = TaskState.SUPERPOSITION
    quantum_weight: Optional[jnp.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class PlanningConfig:
    """Configuration for quantum-inspired planning"""
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    quantum_annealing_schedule: str = "linear"
    superposition_width: float = 0.1
    entanglement_strength: float = 0.5
    interference_factor: float = 0.3
    measurement_threshold: float = 0.8
    use_gpu: bool = True
    enable_monitoring: bool = True


class QuantumInspiredPlanner:
    """
    Quantum-inspired task planner using JAX for acceleration
    
    Implements quantum optimization concepts:
    - Superposition: Explore multiple solution paths simultaneously  
    - Entanglement: Model task dependencies and correlations
    - Interference: Constructive/destructive optimization
    - Measurement: Collapse to optimal solution
    """
    
    def __init__(self, config: PlanningConfig = None):
        self.config = config or PlanningConfig()
        self.tasks: Dict[str, Task] = {}
        self.quantum_state: Optional[QuantumState] = None
        self.optimizer = PerformanceOptimizer() if config and config.use_gpu else None
        self.metrics = MetricsCollector() if config and config.enable_monitoring else None
        self._solution_cache: Dict[str, Any] = {}
        
        # Initialize quantum-inspired components
        self._initialize_quantum_components()
    
    def _initialize_quantum_components(self):
        """Initialize quantum computation components"""
        self._hamiltonian = None
        self._coupling_matrix = None
        self._annealing_schedule = self._create_annealing_schedule()
    
    @jax.jit
    def _create_annealing_schedule(self) -> jnp.ndarray:
        """Create quantum annealing schedule"""
        if self.config.quantum_annealing_schedule == "linear":
            return jnp.linspace(0, 1, self.config.max_iterations)
        elif self.config.quantum_annealing_schedule == "exponential":
            return 1 - jnp.exp(-jnp.linspace(0, 5, self.config.max_iterations))
        else:
            return jnp.ones(self.config.max_iterations)
    
    def add_task(self, task: Task) -> None:
        """Add task to quantum superposition space"""
        if self.metrics:
            self.metrics.record_event("task_added", {"task_id": task.id})
            
        # Initialize quantum weight vector
        task.quantum_weight = jnp.array([
            task.complexity,
            task.priority, 
            len(task.dependencies),
            sum(task.resources.values())
        ])
        
        self.tasks[task.id] = task
        self._invalidate_cache()
    
    def add_dependency(self, task_id: str, dependency_id: str) -> None:
        """Create entanglement between tasks"""
        if task_id in self.tasks and dependency_id in self.tasks:
            self.tasks[task_id].dependencies.append(dependency_id)
            self.tasks[task_id].state = TaskState.ENTANGLED
            self._invalidate_cache()
    
    @jax.jit
    def _compute_hamiltonian(self, task_weights: jnp.ndarray) -> jnp.ndarray:
        """Compute system Hamiltonian for optimization"""
        n_tasks = task_weights.shape[0]
        
        # Diagonal terms: individual task costs
        diagonal = jnp.diag(task_weights[:, 0])  # complexity
        
        # Off-diagonal terms: interaction costs
        interactions = jnp.zeros((n_tasks, n_tasks))
        
        return diagonal + self.config.entanglement_strength * interactions
    
    @jax.jit  
    def _apply_quantum_superposition(self, state_vector: jnp.ndarray, iteration: int) -> jnp.ndarray:
        """Apply superposition to explore solution space"""
        annealing_factor = self._annealing_schedule[iteration]
        
        # Add quantum fluctuations
        noise = jax.random.normal(
            jax.random.PRNGKey(iteration), 
            state_vector.shape
        ) * self.config.superposition_width * (1 - annealing_factor)
        
        superposed_state = state_vector + noise
        return superposed_state / jnp.linalg.norm(superposed_state)
    
    @jax.jit
    def _apply_quantum_interference(self, state_vector: jnp.ndarray) -> jnp.ndarray:
        """Apply quantum interference for optimization"""
        # Constructive interference amplifies good solutions
        # Destructive interference suppresses poor solutions
        
        amplitude_squared = jnp.abs(state_vector) ** 2
        mean_amplitude = jnp.mean(amplitude_squared)
        
        interference_mask = jnp.where(
            amplitude_squared > mean_amplitude,
            1 + self.config.interference_factor,  # Constructive
            1 - self.config.interference_factor   # Destructive  
        )
        
        return state_vector * interference_mask
    
    def _measure_quantum_state(self, state_vector: jnp.ndarray) -> List[int]:
        """Collapse quantum state to classical solution"""
        probabilities = jnp.abs(state_vector) ** 2
        
        # Sort tasks by probability (quantum measurement)
        task_ids = list(self.tasks.keys())
        sorted_indices = jnp.argsort(-probabilities)
        
        return [task_ids[i] for i in sorted_indices]
    
    def plan(self, objective: str = "minimize_completion_time") -> Dict[str, Any]:
        """
        Generate optimal task execution plan using quantum-inspired optimization
        
        Args:
            objective: Optimization objective ("minimize_completion_time", "minimize_cost", etc.)
            
        Returns:
            Planning result with optimal schedule and metrics
        """
        if not self.tasks:
            return {"schedule": [], "total_time": 0, "error": "No tasks to plan"}
        
        if self.metrics:
            start_time = datetime.now()
            self.metrics.record_event("planning_started", {"objective": objective})
        
        try:
            # Initialize quantum state
            n_tasks = len(self.tasks)
            task_weights = jnp.array([task.quantum_weight for task in self.tasks.values()])
            
            # Create initial superposition state
            initial_state = jnp.ones(n_tasks) / jnp.sqrt(n_tasks)
            state_vector = initial_state
            
            # Quantum-inspired optimization loop
            for iteration in range(self.config.max_iterations):
                # Apply superposition (exploration)
                state_vector = self._apply_quantum_superposition(state_vector, iteration)
                
                # Apply Hamiltonian evolution
                hamiltonian = self._compute_hamiltonian(task_weights)
                state_vector = jnp.dot(hamiltonian, state_vector)
                
                # Apply interference (amplification/suppression)
                state_vector = self._apply_quantum_interference(state_vector)
                
                # Normalize
                state_vector = state_vector / jnp.linalg.norm(state_vector)
                
                # Check convergence
                if iteration > 0:
                    change = jnp.linalg.norm(state_vector - prev_state)
                    if change < self.config.convergence_threshold:
                        break
                
                prev_state = state_vector.copy()
            
            # Measure final state to get classical solution
            optimal_schedule = self._measure_quantum_state(state_vector)
            
            # Build detailed schedule
            schedule = self._build_schedule(optimal_schedule, objective)
            
            if self.metrics:
                end_time = datetime.now()
                planning_time = (end_time - start_time).total_seconds()
                self.metrics.record_metric("planning_time", planning_time)
                self.metrics.record_event("planning_completed", {
                    "iterations": iteration + 1,
                    "convergence": change if iteration > 0 else None
                })
            
            return {
                "schedule": schedule,
                "total_time": schedule[-1]["end_time"] if schedule else 0,
                "iterations": iteration + 1,
                "convergence_achieved": iteration < self.config.max_iterations - 1,
                "quantum_fidelity": float(jnp.abs(jnp.dot(initial_state, state_vector)) ** 2),
                "objective": objective
            }
            
        except Exception as e:
            if self.metrics:
                self.metrics.record_event("planning_failed", {"error": str(e)})
            return {"schedule": [], "total_time": 0, "error": str(e)}
    
    def _build_schedule(self, task_order: List[str], objective: str) -> List[Dict[str, Any]]:
        """Build detailed execution schedule from task ordering"""
        schedule = []
        current_time = 0.0
        completed_tasks = set()
        
        for task_id in task_order:
            task = self.tasks[task_id]
            
            # Check if dependencies are satisfied
            deps_satisfied = all(dep in completed_tasks for dep in task.dependencies)
            if not deps_satisfied:
                continue
                
            # Calculate start time considering resource constraints
            start_time = current_time
            duration = task.duration_estimate or 1.0
            end_time = start_time + duration
            
            schedule.append({
                "task_id": task_id,
                "task_name": task.name,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "resources": task.resources,
                "priority": task.priority,
                "dependencies_satisfied": deps_satisfied
            })
            
            completed_tasks.add(task_id)
            current_time = end_time
        
        return schedule
    
    def replan(self, completed_tasks: List[str], failed_tasks: List[str] = None) -> Dict[str, Any]:
        """Replan after task completion or failure"""
        # Remove completed tasks
        for task_id in completed_tasks:
            if task_id in self.tasks:
                del self.tasks[task_id]
        
        # Handle failed tasks (re-add with higher complexity)
        if failed_tasks:
            for task_id in failed_tasks:
                if task_id in self.tasks:
                    self.tasks[task_id].complexity *= 1.5  # Increase complexity
                    self.tasks[task_id].state = TaskState.SUPERPOSITION  # Reset state
        
        self._invalidate_cache()
        return self.plan()
    
    def get_task_criticality(self, task_id: str) -> float:
        """Calculate task criticality using quantum entanglement measure"""
        if task_id not in self.tasks:
            return 0.0
            
        task = self.tasks[task_id]
        
        # Base criticality from complexity and priority
        base_criticality = task.complexity * task.priority
        
        # Entanglement contribution (how many tasks depend on this one)
        dependents = sum(1 for t in self.tasks.values() if task_id in t.dependencies)
        entanglement_factor = 1 + (dependents * self.config.entanglement_strength)
        
        return base_criticality * entanglement_factor
    
    def _invalidate_cache(self):
        """Invalidate solution cache when tasks change"""
        self._solution_cache.clear()
    
    def get_planning_state(self) -> Dict[str, Any]:
        """Get current planning system state"""
        return {
            "num_tasks": len(self.tasks),
            "num_entangled": sum(1 for t in self.tasks.values() if t.state == TaskState.ENTANGLED),
            "total_complexity": sum(t.complexity for t in self.tasks.values()),
            "config": self.config,
            "cache_size": len(self._solution_cache)
        }