"""
Quantum-Inspired Scheduler

Real-time task scheduling with quantum optimization principles
and adaptive resource management.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set
from enum import Enum
import threading
import time
from datetime import datetime, timedelta

from .core import Task, TaskState, PlanningConfig
from .optimizer import QuantumTaskOptimizer, OptimizationStrategy
from ..metrics import MetricsCollector


class SchedulingPolicy(Enum):
    """Scheduling policies for quantum-inspired scheduling"""
    QUANTUM_PRIORITY = "quantum_priority"
    ENTANGLEMENT_AWARE = "entanglement_aware"
    SUPERPOSITION_FAIR = "superposition_fair"
    ADIABATIC_ADAPTIVE = "adiabatic_adaptive"
    HYBRID_REACTIVE = "hybrid_reactive"


@dataclass
class ResourceState:
    """Quantum-inspired resource state representation"""
    resource_id: str
    capacity: float
    available: float
    quantum_efficiency: float = 1.0
    entangled_tasks: Set[str] = field(default_factory=set)
    coherence_time: float = float('inf')  # Resource stability


@dataclass
class ScheduleEvent:
    """Scheduled task execution event"""
    task_id: str
    start_time: datetime
    expected_end_time: datetime
    assigned_resources: Dict[str, float]
    quantum_priority: float
    dependencies_ready: bool = True


class QuantumScheduler:
    """
    Quantum-inspired real-time task scheduler
    
    Features:
    - Real-time quantum optimization
    - Adaptive resource allocation
    - Entanglement-aware dependency management
    - Superposition-based load balancing
    - Coherence-aware resource stability
    """
    
    def __init__(self, config: PlanningConfig = None, policy: SchedulingPolicy = None):
        self.config = config or PlanningConfig()
        self.policy = policy or SchedulingPolicy.QUANTUM_PRIORITY
        
        # Core components
        self.optimizer = QuantumTaskOptimizer(self.config)
        self.metrics = MetricsCollector() if self.config.enable_monitoring else None
        
        # State management
        self.tasks: Dict[str, Task] = {}
        self.resources: Dict[str, ResourceState] = {}
        self.active_schedule: List[ScheduleEvent] = []
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        
        # Real-time scheduling
        self._scheduler_thread = None
        self._stop_scheduler = threading.Event()
        self._schedule_lock = threading.RLock()
        
        # Quantum state tracking
        self._quantum_state_cache = {}
        self._last_optimization = datetime.now()
        self._optimization_interval = timedelta(seconds=10)  # Re-optimize every 10s
        
    def add_resource(self, resource_id: str, capacity: float, 
                    quantum_efficiency: float = 1.0) -> None:
        """Add a resource with quantum properties"""
        resource = ResourceState(
            resource_id=resource_id,
            capacity=capacity,
            available=capacity,
            quantum_efficiency=quantum_efficiency
        )
        
        with self._schedule_lock:
            self.resources[resource_id] = resource
            
        if self.metrics:
            self.metrics.record_event("resource_added", {
                "resource_id": resource_id,
                "capacity": capacity,
                "efficiency": quantum_efficiency
            })
    
    def submit_task(self, task: Task) -> str:
        """Submit task for quantum scheduling"""
        if self.metrics:
            self.metrics.record_event("task_submitted", {"task_id": task.id})
        
        with self._schedule_lock:
            # Initialize quantum properties
            task.state = TaskState.SUPERPOSITION
            task.quantum_weight = self._compute_quantum_weight(task)
            
            self.tasks[task.id] = task
            
            # Trigger re-optimization if needed
            self._maybe_reoptimize()
            
        return task.id
    
    @jax.jit
    def _compute_quantum_weight(self, task: Task) -> jnp.ndarray:
        """Compute quantum weight vector for task"""
        # Multi-dimensional quantum representation
        base_weight = jnp.array([
            task.complexity,
            task.priority,
            len(task.dependencies),
            sum(task.resources.values()) if task.resources else 0.0,
        ])
        
        # Add quantum fluctuations for exploration
        key = jax.random.PRNGKey(hash(task.id) % 2**32)
        quantum_noise = jax.random.normal(key, base_weight.shape) * 0.1
        
        return base_weight + quantum_noise
    
    def start_scheduler(self) -> None:
        """Start real-time quantum scheduler"""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            return
            
        self._stop_scheduler.clear()
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
        if self.metrics:
            self.metrics.record_event("scheduler_started", {"policy": self.policy.value})
    
    def stop_scheduler(self) -> None:
        """Stop real-time scheduler"""
        self._stop_scheduler.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
            
        if self.metrics:
            self.metrics.record_event("scheduler_stopped", {})
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        while not self._stop_scheduler.is_set():
            try:
                with self._schedule_lock:
                    # Check for completed tasks
                    self._update_task_states()
                    
                    # Check if re-optimization needed
                    if self._should_reoptimize():
                        self._quantum_reoptimize()
                    
                    # Execute next scheduled tasks
                    self._execute_ready_tasks()
                    
                    # Resource decoherence simulation
                    self._update_resource_coherence()
                
                # Sleep briefly to allow other threads
                time.sleep(0.1)
                
            except Exception as e:
                if self.metrics:
                    self.metrics.record_event("scheduler_error", {"error": str(e)})
                time.sleep(1.0)  # Longer sleep on error
    
    def _should_reoptimize(self) -> bool:
        """Determine if quantum re-optimization is needed"""
        time_since_optimization = datetime.now() - self._last_optimization
        
        # Re-optimize if:
        # 1. Time interval exceeded
        # 2. New tasks added
        # 3. Resource states changed significantly
        # 4. Task failures detected
        
        return (
            time_since_optimization > self._optimization_interval or
            len(self.tasks) > len(self._quantum_state_cache) or
            len(self.failed_tasks) > 0
        )
    
    def _quantum_reoptimize(self) -> None:
        """Perform quantum re-optimization of schedule"""
        if not self.tasks:
            return
            
        if self.metrics:
            start_time = time.time()
            self.metrics.record_event("quantum_reoptimization_started", {})
        
        try:
            # Select optimization strategy based on policy
            if self.policy == SchedulingPolicy.QUANTUM_PRIORITY:
                strategy = OptimizationStrategy.QUANTUM_ANNEALING
            elif self.policy == SchedulingPolicy.ENTANGLEMENT_AWARE:
                strategy = OptimizationStrategy.VARIATIONAL_QUANTUM
            elif self.policy == SchedulingPolicy.SUPERPOSITION_FAIR:
                strategy = OptimizationStrategy.ADIABATIC_QUANTUM
            elif self.policy == SchedulingPolicy.ADIABATIC_ADAPTIVE:
                strategy = OptimizationStrategy.QUANTUM_APPROXIMATE
            else:  # HYBRID_REACTIVE
                strategy = OptimizationStrategy.HYBRID_CLASSICAL
            
            # Run quantum optimization
            result = self.optimizer.optimize(
                {k: v for k, v in self.tasks.items() if k not in self.completed_tasks},
                strategy=strategy
            )
            
            # Update schedule with optimization result
            self._update_schedule_from_optimization(result)
            
            # Cache quantum state
            self._quantum_state_cache = {task_id: task for task_id, task in self.tasks.items()}
            self._last_optimization = datetime.now()
            
            if self.metrics:
                optimization_time = time.time() - start_time
                self.metrics.record_metric("quantum_optimization_time", optimization_time)
                self.metrics.record_metric("quantum_fidelity", result.quantum_fidelity)
                
        except Exception as e:
            if self.metrics:
                self.metrics.record_event("quantum_optimization_failed", {"error": str(e)})
    
    def _update_schedule_from_optimization(self, optimization_result) -> None:
        """Update schedule based on quantum optimization result"""
        # Clear old schedule for pending tasks
        self.active_schedule = [event for event in self.active_schedule 
                              if event.task_id in self.completed_tasks]
        
        current_time = datetime.now()
        
        # Build new schedule from optimization
        for i, task_id in enumerate(optimization_result.optimal_solution):
            if task_id in self.completed_tasks:
                continue
                
            task = self.tasks[task_id]
            
            # Calculate quantum priority based on optimization
            quantum_priority = optimization_result.quantum_fidelity * (1.0 / (i + 1))
            
            # Estimate execution time
            duration = timedelta(seconds=task.duration_estimate or 1.0)
            start_time = current_time + timedelta(seconds=i * 0.5)  # Small stagger
            end_time = start_time + duration
            
            # Allocate resources quantum-efficiently
            allocated_resources = self._allocate_quantum_resources(task, start_time)
            
            # Create schedule event
            event = ScheduleEvent(
                task_id=task_id,
                start_time=start_time,
                expected_end_time=end_time,
                assigned_resources=allocated_resources,
                quantum_priority=quantum_priority,
                dependencies_ready=self._check_dependencies_ready(task_id)
            )
            
            self.active_schedule.append(event)
    
    def _allocate_quantum_resources(self, task: Task, start_time: datetime) -> Dict[str, float]:
        """Allocate resources using quantum efficiency principles"""
        allocated = {}
        
        for resource_type, required_amount in task.resources.items():
            best_resource = None
            best_efficiency = 0.0
            
            # Find most quantum-efficient available resource
            for resource_id, resource in self.resources.items():
                if resource.resource_id.startswith(resource_type) and resource.available >= required_amount:
                    # Quantum efficiency includes coherence factor
                    coherence_factor = self._calculate_coherence_factor(resource, start_time)
                    total_efficiency = resource.quantum_efficiency * coherence_factor
                    
                    if total_efficiency > best_efficiency:
                        best_efficiency = total_efficiency
                        best_resource = resource
            
            if best_resource:
                allocated[best_resource.resource_id] = required_amount
                best_resource.available -= required_amount
                best_resource.entangled_tasks.add(task.id)
        
        return allocated
    
    def _calculate_coherence_factor(self, resource: ResourceState, start_time: datetime) -> float:
        """Calculate quantum coherence factor for resource"""
        # Simulate coherence decay over time
        current_time = datetime.now()
        time_delta = (start_time - current_time).total_seconds()
        
        if resource.coherence_time == float('inf'):
            return 1.0
        
        # Exponential coherence decay
        coherence_factor = jnp.exp(-time_delta / resource.coherence_time)
        return float(jnp.maximum(coherence_factor, 0.1))  # Minimum coherence
    
    def _check_dependencies_ready(self, task_id: str) -> bool:
        """Check if task dependencies are satisfied"""
        task = self.tasks[task_id]
        return all(dep_id in self.completed_tasks for dep_id in task.dependencies)
    
    def _execute_ready_tasks(self) -> None:
        """Execute tasks that are ready to run"""
        current_time = datetime.now()
        
        for event in self.active_schedule:
            if (event.task_id not in self.completed_tasks and
                event.start_time <= current_time and
                event.dependencies_ready):
                
                self._execute_task(event)
    
    def _execute_task(self, event: ScheduleEvent) -> None:
        """Execute a single task (simulation)"""
        task = self.tasks[event.task_id]
        
        # Simulate task execution
        task.state = TaskState.COLLAPSED
        
        if self.metrics:
            self.metrics.record_event("task_started", {
                "task_id": event.task_id,
                "quantum_priority": event.quantum_priority,
                "resources": event.assigned_resources
            })
        
        # For simulation, mark as completed immediately
        # In real implementation, this would trigger actual execution
        self._complete_task(event.task_id)
    
    def _complete_task(self, task_id: str) -> None:
        """Mark task as completed and free resources"""
        if task_id not in self.tasks:
            return
            
        task = self.tasks[task_id]
        task.state = TaskState.MEASURED
        
        # Free allocated resources
        for event in self.active_schedule:
            if event.task_id == task_id:
                for resource_id, amount in event.assigned_resources.items():
                    if resource_id in self.resources:
                        self.resources[resource_id].available += amount
                        self.resources[resource_id].entangled_tasks.discard(task_id)
                break
        
        # Mark as completed
        self.completed_tasks.add(task_id)
        
        # Update dependent task states
        for other_task in self.tasks.values():
            if task_id in other_task.dependencies and other_task.state == TaskState.ENTANGLED:
                deps_satisfied = all(dep in self.completed_tasks for dep in other_task.dependencies)
                if deps_satisfied:
                    other_task.state = TaskState.SUPERPOSITION
        
        if self.metrics:
            self.metrics.record_event("task_completed", {"task_id": task_id})
    
    def _update_task_states(self) -> None:
        """Update task states based on current schedule"""
        current_time = datetime.now()
        
        # Check for overdue tasks
        for event in self.active_schedule:
            if (event.task_id not in self.completed_tasks and
                current_time > event.expected_end_time + timedelta(seconds=30)):
                
                # Mark as potentially failed
                if event.task_id not in self.failed_tasks:
                    self.failed_tasks.add(event.task_id)
                    if self.metrics:
                        self.metrics.record_event("task_timeout", {"task_id": event.task_id})
    
    def _update_resource_coherence(self) -> None:
        """Update resource coherence states (quantum decoherence simulation)"""
        for resource in self.resources.values():
            if resource.coherence_time != float('inf'):
                # Simulate gradual coherence decay
                decay_rate = 0.01  # 1% per scheduler cycle
                resource.quantum_efficiency *= (1 - decay_rate)
                resource.quantum_efficiency = max(resource.quantum_efficiency, 0.5)
    
    def _maybe_reoptimize(self) -> None:
        """Check if immediate reoptimization is needed"""
        # Immediate reoptimization triggers
        if (len(self.tasks) - len(self.completed_tasks)) > 10:  # Many pending tasks
            self._quantum_reoptimize()
    
    def get_schedule_status(self) -> Dict[str, Any]:
        """Get current schedule status"""
        with self._schedule_lock:
            pending_tasks = len(self.tasks) - len(self.completed_tasks) - len(self.failed_tasks)
            
            # Calculate quantum metrics
            total_quantum_priority = sum(event.quantum_priority for event in self.active_schedule)
            avg_resource_efficiency = np.mean([r.quantum_efficiency for r in self.resources.values()]) if self.resources else 0.0
            
            return {
                "total_tasks": len(self.tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "pending_tasks": pending_tasks,
                "active_schedule_events": len(self.active_schedule),
                "total_resources": len(self.resources),
                "avg_resource_efficiency": avg_resource_efficiency,
                "total_quantum_priority": total_quantum_priority,
                "last_optimization": self._last_optimization,
                "scheduler_running": self._scheduler_thread and self._scheduler_thread.is_alive(),
                "policy": self.policy.value
            }
    
    def get_resource_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed resource status"""
        with self._schedule_lock:
            return {
                resource_id: {
                    "capacity": resource.capacity,
                    "available": resource.available,
                    "utilization": (resource.capacity - resource.available) / resource.capacity,
                    "quantum_efficiency": resource.quantum_efficiency,
                    "entangled_tasks": len(resource.entangled_tasks),
                    "coherence_time": resource.coherence_time
                }
                for resource_id, resource in self.resources.items()
            }
    
    def force_reoptimization(self) -> Dict[str, Any]:
        """Force immediate quantum reoptimization"""
        with self._schedule_lock:
            self._quantum_reoptimize()
            
        return {
            "reoptimization_completed": True,
            "optimization_time": self._last_optimization,
            "active_schedule_size": len(self.active_schedule)
        }