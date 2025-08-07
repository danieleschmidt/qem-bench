"""
High-Performance Quantum Planning Optimization

Advanced performance optimization, caching, parallelization, and scalability
features for quantum-inspired task planning at enterprise scale.
"""

import jax
import jax.numpy as jnp
from jax import vmap, pmap, jit
import jax.experimental.sparse as jsparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from enum import Enum
import numpy as np
import threading
import multiprocessing
import concurrent.futures
from datetime import datetime, timedelta
import hashlib
import pickle
import psutil
import os

from .core import Task, PlanningConfig
from .optimizer import OptimizationStrategy, OptimizationResult
from ..optimization import PerformanceOptimizer, CacheManager
from ..monitoring import MetricsCollector


class ComputeBackend(Enum):
    """Available compute backends"""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"


class ParallelizationStrategy(Enum):
    """Parallelization strategies"""
    TASK_LEVEL = "task_level"
    SOLUTION_SPACE = "solution_space"
    MULTI_OBJECTIVE = "multi_objective"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    backend: ComputeBackend = ComputeBackend.CPU
    max_workers: int = None
    memory_limit_gb: float = 8.0
    gpu_memory_fraction: float = 0.8
    enable_jit: bool = True
    enable_vectorization: bool = True
    enable_parallelization: bool = True
    cache_size_mb: int = 1024
    batch_size: int = 32
    chunk_size: int = 1000
    prefetch_factor: int = 2
    enable_profiling: bool = False
    optimization_level: int = 2  # 0=none, 1=basic, 2=aggressive, 3=experimental


@dataclass
class ComputeResource:
    """Compute resource specification"""
    resource_type: ComputeBackend
    device_id: int
    memory_gb: float
    compute_units: int
    utilization: float = 0.0
    availability: float = 1.0


class HighPerformancePlanner:
    """
    High-performance quantum-inspired planner with advanced optimization
    
    Features:
    - Multi-backend compute (CPU/GPU/TPU)
    - Distributed parallel processing
    - Advanced caching and memoization
    - JIT compilation and vectorization
    - Memory-aware optimization
    - Adaptive resource management
    - Performance profiling and monitoring
    """
    
    def __init__(self, config: PerformanceConfig = None, planning_config: PlanningConfig = None):
        self.perf_config = config or PerformanceConfig()
        self.planning_config = planning_config or PlanningConfig()
        
        # Initialize compute backend
        self._setup_compute_backend()
        
        # Performance components
        self.cache_manager = CacheManager(cache_size_mb=self.perf_config.cache_size_mb)
        self.metrics = MetricsCollector() if self.perf_config.enable_profiling else None
        
        # Resource management
        self.compute_resources = self._discover_compute_resources()
        self.resource_manager = ResourceManager(self.compute_resources)
        
        # Compilation cache
        self._compiled_functions: Dict[str, Callable] = {}
        self._vectorized_functions: Dict[str, Callable] = {}
        
        # Parallel execution
        self.executor = self._create_executor()
        
        # Memory management
        self.memory_manager = MemoryManager(self.perf_config.memory_limit_gb)
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor() if self.perf_config.enable_profiling else None
    
    def _setup_compute_backend(self) -> None:
        """Setup JAX compute backend"""
        if self.perf_config.backend == ComputeBackend.GPU:
            jax.config.update('jax_platform_name', 'gpu')
            # Configure GPU memory
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(self.perf_config.gpu_memory_fraction)
        elif self.perf_config.backend == ComputeBackend.TPU:
            jax.config.update('jax_platform_name', 'tpu')
        else:  # CPU or others
            jax.config.update('jax_platform_name', 'cpu')
    
    def _discover_compute_resources(self) -> List[ComputeResource]:
        """Discover available compute resources"""
        resources = []
        
        # CPU resources
        cpu_count = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        resources.append(ComputeResource(
            resource_type=ComputeBackend.CPU,
            device_id=0,
            memory_gb=memory_gb,
            compute_units=cpu_count
        ))
        
        # GPU resources (if available)
        try:
            import jax.lib.xla_bridge as xla_bridge
            backend = xla_bridge.get_backend()
            
            if 'gpu' in str(backend.platform).lower():
                for i in range(backend.device_count()):
                    resources.append(ComputeResource(
                        resource_type=ComputeBackend.GPU,
                        device_id=i,
                        memory_gb=8.0,  # Default GPU memory estimate
                        compute_units=1000  # Simplified compute unit estimate
                    ))
        except Exception:
            pass  # No GPU available
        
        return resources
    
    def _create_executor(self) -> concurrent.futures.Executor:
        """Create appropriate parallel executor"""
        if self.perf_config.enable_parallelization:
            max_workers = self.perf_config.max_workers or min(32, (os.cpu_count() or 1) + 4)
            
            if self.perf_config.backend in [ComputeBackend.GPU, ComputeBackend.TPU]:
                # Use thread pool for GPU/TPU to avoid GIL issues
                return concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            else:
                # Use process pool for CPU-intensive tasks
                return concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        else:
            # Dummy executor for sequential execution
            return DummyExecutor()
    
    @jit
    def _vectorized_hamiltonian_computation(self, task_weights: jnp.ndarray, 
                                          coupling_matrix: jnp.ndarray) -> jnp.ndarray:
        """JIT-compiled vectorized Hamiltonian computation"""
        n_tasks = task_weights.shape[0]
        
        # Vectorized diagonal computation
        diagonal = jnp.diag(task_weights[:, 0])
        
        # Vectorized coupling computation
        coupling_strength = 0.5
        coupling_terms = coupling_strength * coupling_matrix
        
        return diagonal + coupling_terms
    
    @jit
    def _batch_quantum_evolution(self, initial_states: jnp.ndarray, 
                                hamiltonians: jnp.ndarray, 
                                time_steps: jnp.ndarray) -> jnp.ndarray:
        """Batch quantum state evolution for multiple scenarios"""
        def single_evolution(state, hamiltonian, dt):
            # Simplified quantum evolution
            evolution_operator = jax.scipy.linalg.expm(-1j * dt * hamiltonian)
            return jnp.dot(evolution_operator, state)
        
        # Vectorize over batch dimension
        batch_evolution = vmap(single_evolution, in_axes=(0, 0, 0))
        return batch_evolution(initial_states, hamiltonians, time_steps)
    
    def _parallel_optimization_search(self, tasks: Dict[str, Task], 
                                    strategies: List[OptimizationStrategy],
                                    num_parallel: int = 4) -> List[OptimizationResult]:
        """Parallel optimization with multiple strategies"""
        
        def run_single_optimization(strategy_params):
            strategy, task_subset, seed = strategy_params
            
            # Set random seed for reproducibility
            np.random.seed(seed)
            
            # Create optimizer instance
            from .optimizer import QuantumTaskOptimizer
            optimizer = QuantumTaskOptimizer(self.planning_config)
            
            # Run optimization
            return optimizer.optimize(task_subset, strategy=strategy)
        
        # Prepare parallel tasks
        parallel_tasks = []
        task_items = list(tasks.items())
        
        for i, strategy in enumerate(strategies):
            # Create task subset for this strategy
            subset_size = len(task_items) // len(strategies)
            start_idx = i * subset_size
            end_idx = start_idx + subset_size if i < len(strategies) - 1 else len(task_items)
            
            task_subset = dict(task_items[start_idx:end_idx])
            
            parallel_tasks.append((strategy, task_subset, 42 + i))  # Different seeds
        
        # Execute in parallel
        if self.perf_config.enable_parallelization:
            with self.executor as executor:
                futures = [executor.submit(run_single_optimization, params) 
                          for params in parallel_tasks]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
        else:
            results = [run_single_optimization(params) for params in parallel_tasks]
        
        return results
    
    def _cached_quantum_computation(self, cache_key: str, 
                                  computation_fn: Callable,
                                  *args, **kwargs) -> Any:
        """Cached quantum computation with intelligent cache management"""
        
        # Check cache first
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            if self.metrics:
                self.metrics.record_event("cache_hit", {"key": cache_key[:32]})
            return cached_result
        
        # Compute if not cached
        if self.metrics:
            start_time = datetime.now()
            self.metrics.record_event("cache_miss", {"key": cache_key[:32]})
        
        result = computation_fn(*args, **kwargs)
        
        # Cache the result
        computation_time = (datetime.now() - start_time).total_seconds() if self.metrics else 0
        self.cache_manager.set(cache_key, result, metadata={
            'computation_time': computation_time,
            'timestamp': datetime.now()
        })
        
        return result
    
    def _create_cache_key(self, tasks: Dict[str, Task], config: Any, 
                         strategy: str, additional_params: Dict[str, Any] = None) -> str:
        """Create deterministic cache key for computation"""
        # Create hash from task configuration
        task_hash = hashlib.md5()
        
        # Sort tasks by ID for deterministic ordering
        sorted_tasks = sorted(tasks.items())
        for task_id, task in sorted_tasks:
            task_str = f"{task_id}:{task.complexity}:{task.priority}:{hash(tuple(task.dependencies))}"
            task_hash.update(task_str.encode())
        
        # Add config hash
        config_str = f"{config}:{strategy}"
        if additional_params:
            config_str += f":{hash(tuple(sorted(additional_params.items())))}"
        
        combined_hash = hashlib.md5((task_hash.hexdigest() + config_str).encode()).hexdigest()
        return f"quantum_plan_{combined_hash}"
    
    def _memory_aware_batch_processing(self, large_dataset: List[Any], 
                                     processing_fn: Callable,
                                     batch_size: int = None) -> List[Any]:
        """Memory-aware batch processing with dynamic sizing"""
        if not large_dataset:
            return []
        
        batch_size = batch_size or self.perf_config.batch_size
        
        # Adaptive batch sizing based on available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < 2.0:  # Low memory
            batch_size = max(1, batch_size // 4)
        elif available_memory_gb < 4.0:  # Medium memory
            batch_size = max(1, batch_size // 2)
        
        results = []
        for i in range(0, len(large_dataset), batch_size):
            batch = large_dataset[i:i + batch_size]
            
            # Monitor memory usage
            memory_before = psutil.Process().memory_info().rss / (1024**3)
            
            batch_result = processing_fn(batch)
            results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
            
            # Check memory usage and adjust batch size if needed
            memory_after = psutil.Process().memory_info().rss / (1024**3)
            memory_increase = memory_after - memory_before
            
            if memory_increase > 0.5:  # More than 500MB increase
                batch_size = max(1, int(batch_size * 0.8))  # Reduce batch size
        
        return results
    
    def optimize_planning_performance(self, tasks: Dict[str, Task],
                                    strategies: List[OptimizationStrategy] = None,
                                    performance_target: str = "speed") -> Dict[str, Any]:
        """
        High-performance planning optimization with multiple strategies
        
        Args:
            tasks: Tasks to optimize
            strategies: Optimization strategies to try
            performance_target: "speed", "quality", or "balanced"
            
        Returns:
            Optimized planning result with performance metrics
        """
        if self.performance_monitor:
            self.performance_monitor.start_profiling("optimize_planning_performance")
        
        start_time = datetime.now()
        
        try:
            # Default strategies based on performance target
            if strategies is None:
                if performance_target == "speed":
                    strategies = [OptimizationStrategy.QUANTUM_ANNEALING, OptimizationStrategy.HYBRID_CLASSICAL]
                elif performance_target == "quality":
                    strategies = [OptimizationStrategy.VARIATIONAL_QUANTUM, OptimizationStrategy.ADIABATIC_QUANTUM]
                else:  # balanced
                    strategies = [OptimizationStrategy.QUANTUM_APPROXIMATE, OptimizationStrategy.QUANTUM_ANNEALING]
            
            # Create cache key
            cache_key = self._create_cache_key(tasks, self.planning_config, 
                                             f"multi_strategy_{performance_target}",
                                             {"strategies": [s.value for s in strategies]})
            
            # Try cache first for complete result
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                cached_result['cache_hit'] = True
                return cached_result
            
            # Parallel optimization with multiple strategies
            optimization_results = self._parallel_optimization_search(tasks, strategies)
            
            # Select best result based on performance target
            best_result = self._select_best_result(optimization_results, performance_target)
            
            # Enhanced result with performance data
            total_time = (datetime.now() - start_time).total_seconds()
            
            enhanced_result = {
                **(best_result.__dict__ if hasattr(best_result, '__dict__') else best_result),
                'performance_optimization': {
                    'strategies_tested': len(strategies),
                    'total_optimization_time': total_time,
                    'performance_target': performance_target,
                    'backend_used': self.perf_config.backend.value,
                    'parallel_workers': self.perf_config.max_workers,
                    'memory_usage_gb': psutil.Process().memory_info().rss / (1024**3),
                    'cache_utilization': self.cache_manager.get_cache_stats()
                }
            }
            
            # Cache the result
            self.cache_manager.set(cache_key, enhanced_result)
            
            return enhanced_result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'performance_optimization': {
                    'total_time': (datetime.now() - start_time).total_seconds(),
                    'performance_target': performance_target
                }
            }
        finally:
            if self.performance_monitor:
                self.performance_monitor.stop_profiling("optimize_planning_performance")
    
    def _select_best_result(self, results: List[OptimizationResult], 
                          performance_target: str) -> OptimizationResult:
        """Select best optimization result based on performance target"""
        if not results:
            raise ValueError("No optimization results to select from")
        
        if performance_target == "speed":
            # Select result with best convergence/iteration ratio
            return min(results, key=lambda r: r.convergence_iterations)
        elif performance_target == "quality":
            # Select result with best objective value
            return min(results, key=lambda r: r.objective_value)
        else:  # balanced
            # Weighted combination of quality and speed
            def balanced_score(r):
                normalized_iterations = r.convergence_iterations / 1000.0  # Normalize
                return r.objective_value + 0.3 * normalized_iterations
            
            return min(results, key=balanced_score)
    
    def distributed_planning(self, tasks: Dict[str, Task], 
                           num_partitions: int = None,
                           coordination_strategy: str = "hierarchical") -> Dict[str, Any]:
        """
        Distributed planning across multiple compute resources
        
        Args:
            tasks: Tasks to plan across distributed resources
            num_partitions: Number of partitions (default: number of compute resources)
            coordination_strategy: "hierarchical", "consensus", or "competitive"
            
        Returns:
            Distributed planning result
        """
        if self.performance_monitor:
            self.performance_monitor.start_profiling("distributed_planning")
        
        start_time = datetime.now()
        
        try:
            # Default partitioning
            num_partitions = num_partitions or min(len(self.compute_resources), 4)
            
            # Partition tasks intelligently
            task_partitions = self._partition_tasks(tasks, num_partitions)
            
            # Distributed optimization
            if coordination_strategy == "hierarchical":
                result = self._hierarchical_distributed_planning(task_partitions)
            elif coordination_strategy == "consensus":
                result = self._consensus_distributed_planning(task_partitions)
            else:  # competitive
                result = self._competitive_distributed_planning(task_partitions)
            
            # Add distributed planning metadata
            result['distributed_planning'] = {
                'num_partitions': num_partitions,
                'coordination_strategy': coordination_strategy,
                'total_time': (datetime.now() - start_time).total_seconds(),
                'compute_resources_used': len(self.compute_resources),
                'resource_utilization': self.resource_manager.get_utilization_stats()
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'distributed_planning': {
                    'coordination_strategy': coordination_strategy,
                    'total_time': (datetime.now() - start_time).total_seconds()
                }
            }
        finally:
            if self.performance_monitor:
                self.performance_monitor.stop_profiling("distributed_planning")
    
    def _partition_tasks(self, tasks: Dict[str, Task], num_partitions: int) -> List[Dict[str, Task]]:
        """Intelligently partition tasks for distributed processing"""
        if num_partitions <= 1:
            return [tasks]
        
        task_list = list(tasks.items())
        
        # Sort by complexity for balanced partitioning
        task_list.sort(key=lambda x: x[1].complexity, reverse=True)
        
        # Create balanced partitions
        partitions = [dict() for _ in range(num_partitions)]
        partition_complexities = [0.0] * num_partitions
        
        # Greedy assignment to least loaded partition
        for task_id, task in task_list:
            min_partition = min(range(num_partitions), key=lambda i: partition_complexities[i])
            partitions[min_partition][task_id] = task
            partition_complexities[min_partition] += task.complexity
        
        # Handle dependencies - move dependent tasks to same partition if possible
        self._resolve_cross_partition_dependencies(partitions)
        
        return [p for p in partitions if p]  # Remove empty partitions
    
    def _resolve_cross_partition_dependencies(self, partitions: List[Dict[str, Task]]) -> None:
        """Resolve dependencies across partitions by task migration"""
        for partition_idx, partition in enumerate(partitions):
            tasks_to_move = []
            
            for task_id, task in partition.items():
                for dep_id in task.dependencies:
                    # Find which partition contains the dependency
                    dep_partition_idx = None
                    for i, other_partition in enumerate(partitions):
                        if i != partition_idx and dep_id in other_partition:
                            dep_partition_idx = i
                            break
                    
                    if dep_partition_idx is not None:
                        # Move this task to the dependency's partition
                        tasks_to_move.append((task_id, task, dep_partition_idx))
            
            # Perform moves
            for task_id, task, target_partition_idx in tasks_to_move:
                if task_id in partition:
                    del partition[task_id]
                    partitions[target_partition_idx][task_id] = task
    
    def _hierarchical_distributed_planning(self, partitions: List[Dict[str, Task]]) -> Dict[str, Any]:
        """Hierarchical distributed planning with master-worker coordination"""
        # Phase 1: Local optimization on each partition
        local_results = []
        
        def optimize_partition(partition_data):
            partition_idx, partition_tasks = partition_data
            from .optimizer import QuantumTaskOptimizer
            
            optimizer = QuantumTaskOptimizer(self.planning_config)
            result = optimizer.optimize(partition_tasks, OptimizationStrategy.QUANTUM_ANNEALING)
            
            return {
                'partition_idx': partition_idx,
                'local_result': result,
                'task_count': len(partition_tasks)
            }
        
        # Parallel local optimization
        partition_data = [(i, partition) for i, partition in enumerate(partitions)]
        
        if self.perf_config.enable_parallelization:
            with self.executor as executor:
                futures = [executor.submit(optimize_partition, data) for data in partition_data]
                local_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        else:
            local_results = [optimize_partition(data) for data in partition_data]
        
        # Phase 2: Global coordination
        global_schedule = []
        total_time = 0.0
        
        # Merge local schedules with dependency resolution
        for local_result in sorted(local_results, key=lambda x: x['partition_idx']):
            local_schedule = local_result['local_result'].optimal_solution
            
            # Adjust timing based on cross-partition dependencies
            for task_id in local_schedule:
                # Simple concatenation for now - could be more sophisticated
                global_schedule.append(task_id)
        
        return {
            'success': True,
            'optimal_solution': global_schedule,
            'objective_value': sum(r['local_result'].objective_value for r in local_results),
            'convergence_iterations': max(r['local_result'].convergence_iterations for r in local_results),
            'quantum_fidelity': np.mean([r['local_result'].quantum_fidelity for r in local_results]),
            'local_results': local_results
        }
    
    def _consensus_distributed_planning(self, partitions: List[Dict[str, Task]]) -> Dict[str, Any]:
        """Consensus-based distributed planning"""
        # Each partition proposes a solution, then consensus is reached
        proposals = []
        
        def generate_proposal(partition_data):
            partition_idx, partition_tasks = partition_data
            from .optimizer import QuantumTaskOptimizer
            
            # Try multiple strategies and return best
            optimizer = QuantumTaskOptimizer(self.planning_config)
            strategies = [OptimizationStrategy.QUANTUM_ANNEALING, OptimizationStrategy.VARIATIONAL_QUANTUM]
            
            best_result = None
            best_score = float('inf')
            
            for strategy in strategies:
                result = optimizer.optimize(partition_tasks, strategy)
                if result.objective_value < best_score:
                    best_score = result.objective_value
                    best_result = result
            
            return {
                'partition_idx': partition_idx,
                'proposal': best_result,
                'confidence': 1.0 / (1.0 + best_score)  # Convert to confidence score
            }
        
        # Generate proposals in parallel
        partition_data = [(i, partition) for i, partition in enumerate(partitions)]
        
        if self.perf_config.enable_parallelization:
            with self.executor as executor:
                futures = [executor.submit(generate_proposal, data) for data in partition_data]
                proposals = [future.result() for future in concurrent.futures.as_completed(futures)]
        else:
            proposals = [generate_proposal(data) for data in partition_data]
        
        # Weighted consensus based on confidence scores
        total_weight = sum(p['confidence'] for p in proposals)
        weighted_solution = []
        
        # Merge solutions based on confidence weights
        for proposal in sorted(proposals, key=lambda x: x['confidence'], reverse=True):
            solution = proposal['proposal'].optimal_solution
            weight = proposal['confidence'] / total_weight
            
            # Add tasks from this proposal (simplified merging)
            for task_id in solution:
                if task_id not in weighted_solution:
                    weighted_solution.append(task_id)
        
        # Calculate consensus metrics
        avg_objective = np.mean([p['proposal'].objective_value for p in proposals])
        avg_fidelity = np.mean([p['proposal'].quantum_fidelity for p in proposals])
        
        return {
            'success': True,
            'optimal_solution': weighted_solution,
            'objective_value': avg_objective,
            'quantum_fidelity': avg_fidelity,
            'consensus_confidence': total_weight / len(proposals),
            'proposals': proposals
        }
    
    def _competitive_distributed_planning(self, partitions: List[Dict[str, Task]]) -> Dict[str, Any]:
        """Competitive distributed planning - best result wins"""
        competitors = []
        
        def compete(partition_data):
            partition_idx, partition_tasks = partition_data
            from .optimizer import QuantumTaskOptimizer
            
            # Use most aggressive optimization for competition
            optimizer = QuantumTaskOptimizer(self.planning_config)
            result = optimizer.optimize(partition_tasks, OptimizationStrategy.VARIATIONAL_QUANTUM)
            
            return {
                'partition_idx': partition_idx,
                'result': result,
                'score': result.objective_value + 0.1 * result.convergence_iterations  # Speed bonus
            }
        
        # Run competition in parallel
        partition_data = [(i, partition) for i, partition in enumerate(partitions)]
        
        if self.perf_config.enable_parallelization:
            with self.executor as executor:
                futures = [executor.submit(compete, data) for data in partition_data]
                competitors = [future.result() for future in concurrent.futures.as_completed(futures)]
        else:
            competitors = [compete(data) for data in partition_data]
        
        # Select winner
        winner = min(competitors, key=lambda x: x['score'])
        
        return {
            'success': True,
            'optimal_solution': winner['result'].optimal_solution,
            'objective_value': winner['result'].objective_value,
            'convergence_iterations': winner['result'].convergence_iterations,
            'quantum_fidelity': winner['result'].quantum_fidelity,
            'winning_partition': winner['partition_idx'],
            'competition_results': competitors
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            'compute_backend': self.perf_config.backend.value,
            'available_resources': len(self.compute_resources),
            'cache_stats': self.cache_manager.get_cache_stats(),
            'memory_usage_gb': psutil.Process().memory_info().rss / (1024**3),
            'system_memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'utilization': psutil.virtual_memory().percent / 100.0
            },
            'cpu_utilization': psutil.cpu_percent() / 100.0,
            'resource_utilization': self.resource_manager.get_utilization_stats()
        }
        
        if self.performance_monitor:
            metrics['profiling_data'] = self.performance_monitor.get_profiling_summary()
        
        return metrics
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if hasattr(self.executor, 'shutdown'):
            self.executor.shutdown(wait=True)
        
        if self.performance_monitor:
            self.performance_monitor.cleanup()


class ResourceManager:
    """Manages compute resources and load balancing"""
    
    def __init__(self, resources: List[ComputeResource]):
        self.resources = {f"{r.resource_type.value}_{r.device_id}": r for r in resources}
        self.utilization_history = []
        self._lock = threading.Lock()
    
    def allocate_resource(self, resource_requirements: Dict[str, float]) -> Optional[str]:
        """Allocate best available resource"""
        with self._lock:
            best_resource = None
            best_score = float('inf')
            
            for resource_id, resource in self.resources.items():
                if resource.availability < 0.1:  # Nearly fully utilized
                    continue
                
                # Calculate allocation score (lower is better)
                score = resource.utilization + 0.1 * (1.0 - resource.availability)
                
                if score < best_score:
                    best_score = score
                    best_resource = resource_id
            
            if best_resource:
                # Update utilization
                self.resources[best_resource].utilization += 0.1
                self.resources[best_resource].availability -= 0.1
                
            return best_resource
    
    def release_resource(self, resource_id: str, utilization_delta: float = 0.1) -> None:
        """Release resource allocation"""
        with self._lock:
            if resource_id in self.resources:
                self.resources[resource_id].utilization = max(0.0, 
                    self.resources[resource_id].utilization - utilization_delta)
                self.resources[resource_id].availability = min(1.0,
                    self.resources[resource_id].availability + utilization_delta)
    
    def get_utilization_stats(self) -> Dict[str, Any]:
        """Get resource utilization statistics"""
        with self._lock:
            total_resources = len(self.resources)
            avg_utilization = np.mean([r.utilization for r in self.resources.values()])
            avg_availability = np.mean([r.availability for r in self.resources.values()])
            
            return {
                'total_resources': total_resources,
                'average_utilization': avg_utilization,
                'average_availability': avg_availability,
                'resource_breakdown': {
                    rid: {
                        'utilization': r.utilization,
                        'availability': r.availability,
                        'type': r.resource_type.value
                    } for rid, r in self.resources.items()
                }
            }


class MemoryManager:
    """Memory-aware computation management"""
    
    def __init__(self, memory_limit_gb: float):
        self.memory_limit_gb = memory_limit_gb
        self.current_usage_gb = 0.0
        self._allocations = {}
        self._lock = threading.Lock()
    
    def request_memory(self, size_gb: float, allocation_id: str) -> bool:
        """Request memory allocation"""
        with self._lock:
            available = self.memory_limit_gb - self.current_usage_gb
            
            if size_gb <= available:
                self._allocations[allocation_id] = size_gb
                self.current_usage_gb += size_gb
                return True
            
            return False
    
    def release_memory(self, allocation_id: str) -> None:
        """Release memory allocation"""
        with self._lock:
            if allocation_id in self._allocations:
                self.current_usage_gb -= self._allocations[allocation_id]
                del self._allocations[allocation_id]
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        with self._lock:
            return {
                'limit_gb': self.memory_limit_gb,
                'used_gb': self.current_usage_gb,
                'available_gb': self.memory_limit_gb - self.current_usage_gb,
                'utilization': self.current_usage_gb / self.memory_limit_gb,
                'active_allocations': len(self._allocations)
            }


class PerformanceMonitor:
    """Performance monitoring and profiling"""
    
    def __init__(self):
        self.active_profiles = {}
        self.completed_profiles = []
        self._lock = threading.Lock()
    
    def start_profiling(self, operation_name: str) -> None:
        """Start profiling an operation"""
        with self._lock:
            self.active_profiles[operation_name] = {
                'start_time': datetime.now(),
                'start_memory': psutil.Process().memory_info().rss / (1024**3),
                'start_cpu': psutil.cpu_percent()
            }
    
    def stop_profiling(self, operation_name: str) -> None:
        """Stop profiling an operation"""
        with self._lock:
            if operation_name in self.active_profiles:
                profile = self.active_profiles[operation_name]
                end_time = datetime.now()
                
                profile_data = {
                    'operation': operation_name,
                    'duration': (end_time - profile['start_time']).total_seconds(),
                    'memory_delta': psutil.Process().memory_info().rss / (1024**3) - profile['start_memory'],
                    'start_time': profile['start_time'],
                    'end_time': end_time
                }
                
                self.completed_profiles.append(profile_data)
                del self.active_profiles[operation_name]
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get profiling summary"""
        with self._lock:
            if not self.completed_profiles:
                return {}
            
            # Aggregate statistics
            total_operations = len(self.completed_profiles)
            avg_duration = np.mean([p['duration'] for p in self.completed_profiles])
            total_duration = sum(p['duration'] for p in self.completed_profiles)
            
            operations_by_type = {}
            for profile in self.completed_profiles:
                op = profile['operation']
                if op not in operations_by_type:
                    operations_by_type[op] = []
                operations_by_type[op].append(profile)
            
            operation_stats = {}
            for op_type, profiles in operations_by_type.items():
                operation_stats[op_type] = {
                    'count': len(profiles),
                    'avg_duration': np.mean([p['duration'] for p in profiles]),
                    'total_duration': sum(p['duration'] for p in profiles),
                    'avg_memory_delta': np.mean([p['memory_delta'] for p in profiles])
                }
            
            return {
                'total_operations': total_operations,
                'average_duration': avg_duration,
                'total_duration': total_duration,
                'operations_by_type': operation_stats,
                'active_profiles': list(self.active_profiles.keys())
            }
    
    def cleanup(self) -> None:
        """Cleanup profiling data"""
        with self._lock:
            self.active_profiles.clear()
            self.completed_profiles.clear()


class DummyExecutor:
    """Dummy executor for sequential execution"""
    
    def submit(self, fn, *args, **kwargs):
        """Submit function for immediate execution"""
        class DummyFuture:
            def __init__(self, result):
                self._result = result
            
            def result(self):
                return self._result
        
        return DummyFuture(fn(*args, **kwargs))
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass