"""
Parallel execution framework for quantum error mitigation computations.

This module provides intelligent parallel execution strategies including
multi-threading, multi-processing, and distributed computing support.
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum
import queue
import functools
import weakref

import numpy as np
import jax
import jax.numpy as jnp

from ..logging import get_logger


class ExecutionStrategy(Enum):
    """Parallel execution strategies."""
    THREAD = "thread"          # Multi-threading (I/O bound)
    PROCESS = "process"        # Multi-processing (CPU bound)
    HYBRID = "hybrid"          # Hybrid thread/process approach
    DISTRIBUTED = "distributed"  # Distributed computing
    ADAPTIVE = "adaptive"      # Adaptive based on workload


class WorkloadType(Enum):
    """Types of computational workloads."""
    IO_BOUND = "io_bound"      # I/O intensive operations
    CPU_BOUND = "cpu_bound"    # CPU intensive computations
    MEMORY_BOUND = "memory_bound"  # Memory intensive operations
    MIXED = "mixed"            # Mixed workload characteristics


@dataclass
class ExecutionConfig:
    """Configuration for parallel execution."""
    
    # Execution strategy
    strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    workload_type: WorkloadType = WorkloadType.MIXED
    
    # Resource limits
    max_workers: int = mp.cpu_count()
    max_memory_gb: float = 8.0
    timeout_seconds: Optional[float] = None
    
    # Chunking and batching
    chunk_size: Optional[int] = None
    auto_chunk_size: bool = True
    min_chunk_size: int = 1
    max_chunk_size: int = 1000
    
    # Performance tuning
    enable_load_balancing: bool = True
    enable_work_stealing: bool = True
    prefetch_factor: int = 2
    
    # Backend-specific settings
    jax_parallel_backend: str = "multiprocessing"  # or "threading"
    use_jax_pmap: bool = False
    distributed_nodes: List[str] = field(default_factory=list)
    
    # Monitoring and debugging
    enable_profiling: bool = True
    log_execution_times: bool = False
    debug_mode: bool = False


@dataclass
class ExecutionResult:
    """Result from parallel execution."""
    
    results: List[Any]
    execution_time: float
    worker_count: int
    strategy_used: ExecutionStrategy
    chunk_sizes: List[int]
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[Exception] = field(default_factory=list)


class TaskQueue:
    """Thread-safe task queue with work stealing."""
    
    def __init__(self, maxsize: int = 0):
        self.queue = queue.Queue(maxsize=maxsize)
        self.completed_tasks = 0
        self.total_tasks = 0
        self._lock = threading.Lock()
    
    def put_tasks(self, tasks: List[Any]) -> None:
        """Add multiple tasks to the queue."""
        with self._lock:
            for task in tasks:
                self.queue.put(task)
            self.total_tasks += len(tasks)
    
    def get_task(self) -> Optional[Any]:
        """Get a task from the queue."""
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None
    
    def task_done(self) -> None:
        """Mark a task as completed."""
        with self._lock:
            self.completed_tasks += 1
        self.queue.task_done()
    
    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        with self._lock:
            return self.completed_tasks >= self.total_tasks
    
    def get_progress(self) -> float:
        """Get completion progress (0.0 to 1.0)."""
        with self._lock:
            if self.total_tasks == 0:
                return 1.0
            return self.completed_tasks / self.total_tasks


class ParallelExecutor:
    """
    Intelligent parallel executor for quantum computations.
    
    This class provides optimized parallel execution strategies for quantum
    error mitigation computations, automatically selecting the best approach
    based on workload characteristics and available resources.
    
    Features:
    - Multi-threading for I/O-bound operations
    - Multi-processing for CPU-bound computations
    - Hybrid execution strategies
    - Automatic chunking and load balancing
    - Work stealing for better resource utilization
    - JAX-specific optimizations (pmap, vmap)
    - Performance monitoring and profiling
    - Fault tolerance and error handling
    
    Example:
        >>> executor = ParallelExecutor(strategy=ExecutionStrategy.ADAPTIVE)
        >>> 
        >>> def compute_expectation(circuit, backend):
        ...     return backend.run(circuit).expectation_value
        >>> 
        >>> circuits = [create_circuit(i) for i in range(100)]
        >>> results = executor.execute_batch(
        ...     compute_expectation, 
        ...     [(circuit, backend) for circuit in circuits]
        ... )
    """
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        """
        Initialize parallel executor.
        
        Args:
            config: Execution configuration (uses defaults if None)
        """
        self.config = config or ExecutionConfig()
        self.logger = get_logger()
        
        # Execution statistics
        self.stats = {
            'total_executions': 0,
            'thread_executions': 0,
            'process_executions': 0,
            'hybrid_executions': 0,
            'total_execution_time': 0.0,
            'total_tasks_processed': 0,
            'errors': 0,
        }
        
        # Resource monitoring
        self._current_workers = 0
        self._peak_workers = 0
        self._lock = threading.RLock()
        
        # Adaptive parameters
        self._workload_history: List[Dict[str, Any]] = []
        self._optimal_chunk_sizes: Dict[str, int] = {}
        
        self.logger.info(f"ParallelExecutor initialized with strategy: {self.config.strategy.value}")
    
    def execute_batch(
        self,
        func: Callable,
        args_list: List[Tuple],
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = None,
        **func_kwargs
    ) -> ExecutionResult:
        """
        Execute function in parallel across multiple inputs.
        
        Args:
            func: Function to execute in parallel
            args_list: List of argument tuples for each function call
            chunk_size: Override chunk size for this execution
            timeout: Override timeout for this execution
            **func_kwargs: Additional keyword arguments for function
            
        Returns:
            ExecutionResult containing results and performance metrics
        """
        start_time = time.time()
        
        # Determine execution strategy
        strategy = self._determine_strategy(func, args_list)
        
        # Determine optimal chunk size
        effective_chunk_size = self._determine_chunk_size(
            func, args_list, chunk_size
        )
        
        # Execute based on strategy
        if strategy == ExecutionStrategy.THREAD:
            result = self._execute_threaded(
                func, args_list, effective_chunk_size, timeout, **func_kwargs
            )
        elif strategy == ExecutionStrategy.PROCESS:
            result = self._execute_multiprocess(
                func, args_list, effective_chunk_size, timeout, **func_kwargs
            )
        elif strategy == ExecutionStrategy.HYBRID:
            result = self._execute_hybrid(
                func, args_list, effective_chunk_size, timeout, **func_kwargs
            )
        else:  # ADAPTIVE or fallback
            result = self._execute_adaptive(
                func, args_list, effective_chunk_size, timeout, **func_kwargs
            )
        
        execution_time = time.time() - start_time
        
        # Update statistics
        self._update_stats(strategy, execution_time, len(args_list))
        
        # Record workload characteristics for adaptive optimization
        self._record_workload(func, args_list, strategy, execution_time, effective_chunk_size)
        
        return ExecutionResult(
            results=result,
            execution_time=execution_time,
            worker_count=self._current_workers,
            strategy_used=strategy,
            chunk_sizes=[effective_chunk_size],
            performance_metrics=self._get_execution_metrics(strategy, execution_time, len(args_list))
        )
    
    def execute_map(
        self,
        func: Callable,
        iterables: List[List[Any]],
        chunk_size: Optional[int] = None,
        **func_kwargs
    ) -> List[Any]:
        """
        Execute function with map-like interface.
        
        Args:
            func: Function to map over iterables
            iterables: List of iterables to map over
            chunk_size: Chunk size for processing
            **func_kwargs: Additional function arguments
            
        Returns:
            List of results
        """
        # Convert to args_list format
        args_list = list(zip(*iterables))
        
        result = self.execute_batch(func, args_list, chunk_size, **func_kwargs)
        return result.results
    
    def execute_jax_batch(
        self,
        func: Callable,
        arrays: List[jnp.ndarray],
        use_pmap: Optional[bool] = None,
        **func_kwargs
    ) -> List[Any]:
        """
        Execute JAX functions in parallel using JAX-specific optimizations.
        
        Args:
            func: JAX function to execute
            arrays: List of JAX arrays to process
            use_pmap: Whether to use pmap (None for auto-detection)
            **func_kwargs: Additional function arguments
            
        Returns:
            List of results
        """
        if use_pmap is None:
            use_pmap = self.config.use_jax_pmap and len(jax.devices()) > 1
        
        if use_pmap:
            return self._execute_jax_pmap(func, arrays, **func_kwargs)
        else:
            return self._execute_jax_vmap(func, arrays, **func_kwargs)
    
    def submit_async(
        self,
        func: Callable,
        args_list: List[Tuple],
        callback: Optional[Callable] = None,
        **func_kwargs
    ) -> Future:
        """
        Submit batch execution asynchronously.
        
        Args:
            func: Function to execute
            args_list: List of argument tuples
            callback: Optional callback for results
            **func_kwargs: Additional function arguments
            
        Returns:
            Future object for async execution
        """
        # Use ThreadPoolExecutor for async submission
        executor = ThreadPoolExecutor(max_workers=1)
        
        def wrapped_execution():
            result = self.execute_batch(func, args_list, **func_kwargs)
            if callback:
                callback(result)
            return result
        
        return executor.submit(wrapped_execution)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        with self._lock:
            avg_execution_time = (
                self.stats['total_execution_time'] / self.stats['total_executions']
                if self.stats['total_executions'] > 0 else 0
            )
            
            return {
                'execution_summary': {
                    'total_executions': self.stats['total_executions'],
                    'total_tasks_processed': self.stats['total_tasks_processed'],
                    'total_execution_time': self.stats['total_execution_time'],
                    'average_execution_time': avg_execution_time,
                    'errors': self.stats['errors'],
                },
                'strategy_usage': {
                    'thread_executions': self.stats['thread_executions'],
                    'process_executions': self.stats['process_executions'],
                    'hybrid_executions': self.stats['hybrid_executions'],
                },
                'resource_usage': {
                    'current_workers': self._current_workers,
                    'peak_workers': self._peak_workers,
                    'max_workers_configured': self.config.max_workers,
                },
                'adaptive_learning': {
                    'workload_samples': len(self._workload_history),
                    'learned_chunk_sizes': dict(self._optimal_chunk_sizes),
                },
                'config': {
                    'strategy': self.config.strategy.value,
                    'workload_type': self.config.workload_type.value,
                    'max_workers': self.config.max_workers,
                    'enable_load_balancing': self.config.enable_load_balancing,
                }
            }
    
    def _determine_strategy(self, func: Callable, args_list: List[Tuple]) -> ExecutionStrategy:
        """Determine the best execution strategy for the workload."""
        if self.config.strategy != ExecutionStrategy.ADAPTIVE:
            return self.config.strategy
        
        # Adaptive strategy selection based on workload characteristics
        task_count = len(args_list)
        
        # Simple heuristics for strategy selection
        if task_count <= 2:
            return ExecutionStrategy.THREAD  # Low overhead for small tasks
        
        # Analyze function characteristics
        func_name = getattr(func, '__name__', 'unknown')
        
        # Check workload history for similar functions
        for record in self._workload_history[-10:]:  # Check last 10 records
            if record['function_name'] == func_name:
                return record['strategy']
        
        # Default heuristics
        if self.config.workload_type == WorkloadType.IO_BOUND:
            return ExecutionStrategy.THREAD
        elif self.config.workload_type == WorkloadType.CPU_BOUND:
            return ExecutionStrategy.PROCESS
        else:
            # For mixed workload, use hybrid or thread based on task count
            return ExecutionStrategy.HYBRID if task_count > 50 else ExecutionStrategy.THREAD
    
    def _determine_chunk_size(
        self,
        func: Callable,
        args_list: List[Tuple],
        override_chunk_size: Optional[int]
    ) -> int:
        """Determine optimal chunk size for the workload."""
        if override_chunk_size is not None:
            return override_chunk_size
        
        if not self.config.auto_chunk_size:
            return self.config.chunk_size or 1
        
        task_count = len(args_list)
        worker_count = min(self.config.max_workers, task_count)
        
        # Check learned optimal chunk sizes
        func_name = getattr(func, '__name__', 'unknown')
        if func_name in self._optimal_chunk_sizes:
            learned_size = self._optimal_chunk_sizes[func_name]
            return max(self.config.min_chunk_size, 
                      min(self.config.max_chunk_size, learned_size))
        
        # Default chunk size calculation
        base_chunk_size = max(1, task_count // (worker_count * 4))
        
        # Adjust based on workload type
        if self.config.workload_type == WorkloadType.IO_BOUND:
            chunk_size = min(base_chunk_size, 10)  # Smaller chunks for I/O
        elif self.config.workload_type == WorkloadType.CPU_BOUND:
            chunk_size = max(base_chunk_size, 5)   # Larger chunks for CPU
        else:
            chunk_size = base_chunk_size
        
        return max(self.config.min_chunk_size, 
                  min(self.config.max_chunk_size, chunk_size))
    
    def _execute_threaded(
        self,
        func: Callable,
        args_list: List[Tuple],
        chunk_size: int,
        timeout: Optional[float],
        **func_kwargs
    ) -> List[Any]:
        """Execute using thread pool."""
        max_workers = min(self.config.max_workers, len(args_list))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            futures = []
            for args in args_list:
                future = executor.submit(func, *args, **func_kwargs)
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures, timeout=timeout):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Task execution failed: {e}")
                    results.append(None)
                    self.stats['errors'] += 1
        
        self._current_workers = max_workers
        self._peak_workers = max(self._peak_workers, max_workers)
        
        return results
    
    def _execute_multiprocess(
        self,
        func: Callable,
        args_list: List[Tuple],
        chunk_size: int,
        timeout: Optional[float],
        **func_kwargs
    ) -> List[Any]:
        """Execute using process pool."""
        max_workers = min(self.config.max_workers, len(args_list))
        
        # Create wrapper function for multiprocessing
        def wrapper(args_and_kwargs):
            args, kwargs = args_and_kwargs
            return func(*args, **kwargs)
        
        # Prepare arguments
        args_with_kwargs = [(args, func_kwargs) for args in args_list]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            futures = []
            for args_kwargs in args_with_kwargs:
                future = executor.submit(wrapper, args_kwargs)
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures, timeout=timeout):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Process execution failed: {e}")
                    results.append(None)
                    self.stats['errors'] += 1
        
        self._current_workers = max_workers
        self._peak_workers = max(self._peak_workers, max_workers)
        
        return results
    
    def _execute_hybrid(
        self,
        func: Callable,
        args_list: List[Tuple],
        chunk_size: int,
        timeout: Optional[float],
        **func_kwargs
    ) -> List[Any]:
        """Execute using hybrid thread/process approach."""
        # Split workload between threads and processes
        total_tasks = len(args_list)
        process_ratio = 0.7 if self.config.workload_type == WorkloadType.CPU_BOUND else 0.3
        
        process_tasks = int(total_tasks * process_ratio)
        thread_tasks = total_tasks - process_tasks
        
        process_args = args_list[:process_tasks]
        thread_args = args_list[process_tasks:]
        
        # Execute both in parallel
        futures = []
        
        # Submit process execution
        if process_args:
            executor = ThreadPoolExecutor(max_workers=1)
            process_future = executor.submit(
                self._execute_multiprocess, func, process_args, chunk_size, timeout, **func_kwargs
            )
            futures.append(('process', process_future))
        
        # Submit thread execution
        if thread_args:
            executor = ThreadPoolExecutor(max_workers=1)
            thread_future = executor.submit(
                self._execute_threaded, func, thread_args, chunk_size, timeout, **func_kwargs
            )
            futures.append(('thread', thread_future))
        
        # Collect results
        all_results = []
        for execution_type, future in futures:
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"Hybrid {execution_type} execution failed: {e}")
                self.stats['errors'] += 1
        
        return all_results
    
    def _execute_adaptive(
        self,
        func: Callable,
        args_list: List[Tuple],
        chunk_size: int,
        timeout: Optional[float],
        **func_kwargs
    ) -> List[Any]:
        """Execute using adaptive strategy selection."""
        # For adaptive, fall back to thread execution as it's generally safer
        return self._execute_threaded(func, args_list, chunk_size, timeout, **func_kwargs)
    
    def _execute_jax_pmap(
        self,
        func: Callable,
        arrays: List[jnp.ndarray],
        **func_kwargs
    ) -> List[Any]:
        """Execute using JAX pmap for multi-device parallelization."""
        try:
            # Stack arrays for pmap
            stacked_arrays = jnp.stack(arrays)
            
            # Apply pmap
            pmapped_func = jax.pmap(func, **func_kwargs)
            
            # Execute on multiple devices
            results = pmapped_func(stacked_arrays)
            
            # Convert back to list
            return [results[i] for i in range(len(arrays))]
            
        except Exception as e:
            self.logger.warning(f"JAX pmap execution failed: {e}, falling back to vmap")
            return self._execute_jax_vmap(func, arrays, **func_kwargs)
    
    def _execute_jax_vmap(
        self,
        func: Callable,
        arrays: List[jnp.ndarray],
        **func_kwargs
    ) -> List[Any]:
        """Execute using JAX vmap for vectorized execution."""
        try:
            # Stack arrays for vmap
            stacked_arrays = jnp.stack(arrays)
            
            # Apply vmap
            vmapped_func = jax.vmap(func, **func_kwargs)
            
            # Execute vectorized
            results = vmapped_func(stacked_arrays)
            
            # Convert back to list
            return [results[i] for i in range(len(arrays))]
            
        except Exception as e:
            self.logger.error(f"JAX vmap execution failed: {e}")
            return [None] * len(arrays)
    
    def _update_stats(self, strategy: ExecutionStrategy, execution_time: float, task_count: int) -> None:
        """Update execution statistics."""
        with self._lock:
            self.stats['total_executions'] += 1
            self.stats['total_execution_time'] += execution_time
            self.stats['total_tasks_processed'] += task_count
            
            if strategy == ExecutionStrategy.THREAD:
                self.stats['thread_executions'] += 1
            elif strategy == ExecutionStrategy.PROCESS:
                self.stats['process_executions'] += 1
            elif strategy == ExecutionStrategy.HYBRID:
                self.stats['hybrid_executions'] += 1
    
    def _record_workload(
        self,
        func: Callable,
        args_list: List[Tuple],
        strategy: ExecutionStrategy,
        execution_time: float,
        chunk_size: int
    ) -> None:
        """Record workload characteristics for adaptive learning."""
        func_name = getattr(func, '__name__', 'unknown')
        
        workload_record = {
            'function_name': func_name,
            'task_count': len(args_list),
            'strategy': strategy,
            'execution_time': execution_time,
            'chunk_size': chunk_size,
            'throughput': len(args_list) / execution_time if execution_time > 0 else 0,
            'timestamp': time.time(),
        }
        
        self._workload_history.append(workload_record)
        
        # Maintain history size limit
        if len(self._workload_history) > 100:
            self._workload_history.pop(0)
        
        # Update optimal chunk size
        if func_name in self._optimal_chunk_sizes:
            # Simple moving average
            current_optimal = self._optimal_chunk_sizes[func_name]
            self._optimal_chunk_sizes[func_name] = int((current_optimal + chunk_size) / 2)
        else:
            self._optimal_chunk_sizes[func_name] = chunk_size
    
    def _get_execution_metrics(
        self, 
        strategy: ExecutionStrategy, 
        execution_time: float, 
        task_count: int
    ) -> Dict[str, Any]:
        """Get detailed execution metrics."""
        return {
            'strategy': strategy.value,
            'execution_time': execution_time,
            'task_count': task_count,
            'throughput': task_count / execution_time if execution_time > 0 else 0,
            'workers_used': self._current_workers,
            'tasks_per_worker': task_count / self._current_workers if self._current_workers > 0 else 0,
        }


def create_parallel_executor(
    strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
    max_workers: Optional[int] = None,
    workload_type: WorkloadType = WorkloadType.MIXED,
    **config_kwargs
) -> ParallelExecutor:
    """
    Create a parallel executor with specified configuration.
    
    Args:
        strategy: Execution strategy to use
        max_workers: Maximum number of workers (None for CPU count)
        workload_type: Type of workload being processed
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured ParallelExecutor instance
    """
    config = ExecutionConfig(
        strategy=strategy,
        max_workers=max_workers or mp.cpu_count(),
        workload_type=workload_type,
        **config_kwargs
    )
    return ParallelExecutor(config)


def execute_parallel(
    func: Callable,
    args_list: List[Tuple],
    strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
    max_workers: Optional[int] = None,
    **func_kwargs
) -> List[Any]:
    """
    Convenience function for quick parallel execution.
    
    Args:
        func: Function to execute in parallel
        args_list: List of argument tuples
        strategy: Execution strategy
        max_workers: Maximum number of workers
        **func_kwargs: Additional function arguments
        
    Returns:
        List of results
    """
    executor = create_parallel_executor(strategy=strategy, max_workers=max_workers)
    result = executor.execute_batch(func, args_list, **func_kwargs)
    return result.results