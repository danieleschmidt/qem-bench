"""
Performance optimization system for QEM-Bench.

Provides advanced performance optimization including concurrent processing,
vectorization, resource pooling, and adaptive scaling.
"""

import time
import threading
import multiprocessing
import concurrent.futures
import queue
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import wraps, partial
import asyncio
import numpy as np
from collections import deque, defaultdict
import psutil
import gc

from ..monitoring.logger import get_logger


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_peak_mb: float = 0.0
    concurrent_tasks: int = 1
    cache_hits: int = 0
    cache_misses: int = 0
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000
    
    @property
    def throughput_ops_per_sec(self) -> float:
        """Get throughput in operations per second."""
        if self.duration_ms == 0:
            return 0.0
        return 1000.0 / self.duration_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation': self.operation_name,
            'duration_ms': self.duration_ms,
            'cpu_percent': self.cpu_usage_percent,
            'memory_mb': self.memory_usage_mb,
            'memory_peak_mb': self.memory_peak_mb,
            'concurrent_tasks': self.concurrent_tasks,
            'throughput_ops_per_sec': self.throughput_ops_per_sec,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }


class PerformanceProfiler:
    """Context manager for performance profiling."""
    
    def __init__(self, operation_name: str, logger: Optional[Any] = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger("performance")
        self.metrics = PerformanceMetrics(operation_name, time.time())
        self.process = psutil.Process()
        
    def __enter__(self):
        # Record initial state
        self.metrics.memory_usage_mb = self.process.memory_info().rss / (1024 * 1024)
        self.metrics.memory_peak_mb = self.metrics.memory_usage_mb
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Record final state
        self.metrics.end_time = time.time()
        final_memory = self.process.memory_info().rss / (1024 * 1024)
        self.metrics.memory_peak_mb = max(self.metrics.memory_peak_mb, final_memory)
        
        # Log performance metrics
        self.logger.log_performance(
            self.operation_name,
            self.metrics.duration_ms,
            self.metrics.to_dict()
        )
    
    def update_metrics(self, **kwargs):
        """Update performance metrics."""
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)


def profile_performance(operation_name: str = None):
    """Decorator to profile function performance."""
    
    def decorator(func: Callable):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceProfiler(operation_name) as profiler:
                # Estimate input size
                try:
                    input_size = sum(len(str(arg)) for arg in args) + sum(len(str(v)) for v in kwargs.values())
                    profiler.update_metrics(input_size=input_size)
                except:
                    pass
                
                result = func(*args, **kwargs)
                
                # Estimate output size
                try:
                    output_size = len(str(result))
                    profiler.update_metrics(output_size=output_size)
                except:
                    pass
                
                return result
        
        return wrapper
    return decorator


class ResourcePool:
    """Generic resource pool for expensive objects."""
    
    def __init__(self, resource_factory: Callable, max_size: int = 10, 
                 idle_timeout: float = 300):  # 5 minutes
        self.resource_factory = resource_factory
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        
        self._pool = queue.Queue(maxsize=max_size)
        self._in_use = set()
        self._last_used = {}
        self._lock = threading.RLock()
        self._cleanup_thread = None
        self._shutdown = False
        
        self.logger = get_logger("resource_pool")
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_loop():
            while not self._shutdown:
                try:
                    self._cleanup_idle_resources()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    self.logger.error(f"Resource pool cleanup error: {e}", e)
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_idle_resources(self):
        """Remove idle resources from pool."""
        current_time = time.time()
        
        with self._lock:
            expired_resources = []
            temp_queue = queue.Queue()
            
            # Check all pooled resources
            while not self._pool.empty():
                try:
                    resource = self._pool.get_nowait()
                    resource_id = id(resource)
                    
                    if (resource_id in self._last_used and 
                        current_time - self._last_used[resource_id] > self.idle_timeout):
                        # Resource has been idle too long
                        expired_resources.append(resource)
                        del self._last_used[resource_id]
                    else:
                        # Keep resource
                        temp_queue.put(resource)
                except queue.Empty:
                    break
            
            # Put back non-expired resources
            while not temp_queue.empty():
                self._pool.put(temp_queue.get())
            
            if expired_resources:
                self.logger.debug(f"Cleaned up {len(expired_resources)} idle resources")
    
    def acquire(self) -> Any:
        """Acquire resource from pool."""
        with self._lock:
            try:
                # Try to get existing resource
                resource = self._pool.get_nowait()
                self._in_use.add(id(resource))
                return resource
            except queue.Empty:
                # Create new resource if under limit
                if len(self._in_use) < self.max_size:
                    resource = self.resource_factory()
                    self._in_use.add(id(resource))
                    self.logger.debug("Created new pooled resource")
                    return resource
                else:
                    # Pool is at capacity, wait for resource
                    self.logger.warning("Resource pool at capacity, waiting...")
                    resource = self._pool.get(timeout=30)  # 30 second timeout
                    self._in_use.add(id(resource))
                    return resource
    
    def release(self, resource: Any):
        """Release resource back to pool."""
        with self._lock:
            resource_id = id(resource)
            
            if resource_id in self._in_use:
                self._in_use.remove(resource_id)
                self._last_used[resource_id] = time.time()
                
                try:
                    self._pool.put_nowait(resource)
                except queue.Full:
                    # Pool is full, discard resource
                    self.logger.debug("Pool full, discarding resource")
    
    def shutdown(self):
        """Shutdown resource pool."""
        self._shutdown = True
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        # Clear pool
        with self._lock:
            while not self._pool.empty():
                try:
                    self._pool.get_nowait()
                except queue.Empty:
                    break
            
            self._in_use.clear()
            self._last_used.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                'pool_size': self._pool.qsize(),
                'in_use': len(self._in_use),
                'max_size': self.max_size,
                'utilization': len(self._in_use) / self.max_size
            }


class ConcurrentExecutor:
    """High-performance concurrent executor for quantum operations."""
    
    def __init__(self, max_workers: Optional[int] = None, 
                 executor_type: str = "thread"):
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.executor_type = executor_type
        
        self.logger = get_logger("concurrent_executor")
        
        # Create appropriate executor
        if executor_type == "thread":
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        elif executor_type == "process":
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            raise ValueError(f"Unknown executor type: {executor_type}")
        
        self._stats = {
            'submitted': 0, 'completed': 0, 'failed': 0, 'total_time': 0.0
        }
        self._lock = threading.Lock()
    
    def submit_batch(self, func: Callable, args_list: List[Tuple], 
                    timeout: Optional[float] = None) -> List[Any]:
        """Submit batch of tasks and return results."""
        if not args_list:
            return []
        
        start_time = time.time()
        
        # Submit all tasks
        futures = []
        for args in args_list:
            if isinstance(args, tuple):
                future = self.executor.submit(func, *args)
            else:
                future = self.executor.submit(func, args)
            futures.append(future)
        
        with self._lock:
            self._stats['submitted'] += len(futures)
        
        # Collect results
        results = []
        completed_count = 0
        failed_count = 0
        
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
                completed_count += 1
            except Exception as e:
                self.logger.error(f"Concurrent task failed: {e}", e)
                results.append(None)
                failed_count += 1
        
        execution_time = time.time() - start_time
        
        with self._lock:
            self._stats['completed'] += completed_count
            self._stats['failed'] += failed_count
            self._stats['total_time'] += execution_time
        
        self.logger.info(
            f"Batch execution: {completed_count} completed, {failed_count} failed, "
            f"{execution_time:.2f}s total",
            "batch_execution",
            {
                'batch_size': len(args_list),
                'completed': completed_count,
                'failed': failed_count,
                'execution_time': execution_time,
                'throughput': len(args_list) / execution_time if execution_time > 0 else 0
            }
        )
        
        return results
    
    def map_concurrent(self, func: Callable, iterable: List[Any], 
                      chunk_size: Optional[int] = None,
                      timeout: Optional[float] = None) -> List[Any]:
        """Concurrent map operation."""
        if chunk_size is None:
            chunk_size = max(1, len(iterable) // (self.max_workers * 4))
        
        # Split into chunks
        chunks = [iterable[i:i + chunk_size] 
                 for i in range(0, len(iterable), chunk_size)]
        
        # Process chunks concurrently
        def process_chunk(chunk):
            return [func(item) for item in chunk]
        
        chunk_results = self.submit_batch(process_chunk, chunks, timeout)
        
        # Flatten results
        results = []
        for chunk_result in chunk_results:
            if chunk_result is not None:
                results.extend(chunk_result)
        
        return results
    
    def shutdown(self, wait: bool = True):
        """Shutdown executor."""
        self.executor.shutdown(wait=wait)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        with self._lock:
            total_tasks = self._stats['submitted']
            avg_time = (self._stats['total_time'] / total_tasks 
                       if total_tasks > 0 else 0)
            
            return {
                'executor_type': self.executor_type,
                'max_workers': self.max_workers,
                'submitted': self._stats['submitted'],
                'completed': self._stats['completed'],
                'failed': self._stats['failed'],
                'success_rate': (self._stats['completed'] / total_tasks 
                               if total_tasks > 0 else 0),
                'avg_task_time': avg_time,
                'total_time': self._stats['total_time']
            }


class AdaptiveScheduler:
    """Adaptive scheduler that optimizes execution based on workload."""
    
    def __init__(self):
        self.logger = get_logger("adaptive_scheduler")
        
        # Performance history
        self._performance_history = defaultdict(deque)
        self._optimal_configs = {}
        self._lock = threading.RLock()
        
        # System monitoring
        self._system_load_history = deque(maxlen=100)
        self._monitor_thread = None
        self._monitoring = False
        
        # Scheduling strategies
        self.strategies = {
            'cpu_intensive': {'executor_type': 'process', 'chunk_size': 1},
            'io_intensive': {'executor_type': 'thread', 'chunk_size': 10},
            'memory_intensive': {'executor_type': 'thread', 'chunk_size': 2},
            'balanced': {'executor_type': 'thread', 'chunk_size': 5}
        }
        
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start system monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                try:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_percent = psutil.virtual_memory().percent
                    load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else cpu_percent / 100
                    
                    load_info = {
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory_percent,
                        'load_avg': load_avg
                    }
                    
                    with self._lock:
                        self._system_load_history.append(load_info)
                    
                    time.sleep(5)  # Monitor every 5 seconds
                
                except Exception as e:
                    self.logger.error(f"System monitoring error: {e}", e)
                    time.sleep(5)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def get_current_system_load(self) -> Dict[str, float]:
        """Get current system load metrics."""
        with self._lock:
            if not self._system_load_history:
                return {'cpu_percent': 0, 'memory_percent': 0, 'load_avg': 0}
            
            recent = list(self._system_load_history)[-5:]  # Last 5 measurements
            
            return {
                'cpu_percent': sum(x['cpu_percent'] for x in recent) / len(recent),
                'memory_percent': sum(x['memory_percent'] for x in recent) / len(recent),
                'load_avg': sum(x['load_avg'] for x in recent) / len(recent)
            }
    
    def classify_workload(self, operation_name: str, workload_size: int) -> str:
        """Classify workload type based on operation and system state."""
        system_load = self.get_current_system_load()
        
        # Classification heuristics
        if 'simulation' in operation_name.lower() or 'quantum' in operation_name.lower():
            if workload_size > 100:
                return 'cpu_intensive'
            else:
                return 'balanced'
        elif 'io' in operation_name.lower() or 'file' in operation_name.lower():
            return 'io_intensive'
        elif 'memory' in operation_name.lower() or system_load['memory_percent'] > 80:
            return 'memory_intensive'
        else:
            return 'balanced'
    
    def get_optimal_config(self, operation_name: str, workload_size: int) -> Dict[str, Any]:
        """Get optimal execution configuration for operation."""
        workload_type = self.classify_workload(operation_name, workload_size)
        
        # Check if we have learned optimal config
        config_key = f"{operation_name}:{workload_type}:{workload_size//10}"
        
        with self._lock:
            if config_key in self._optimal_configs:
                return self._optimal_configs[config_key]
        
        # Use default strategy
        base_config = self.strategies[workload_type].copy()
        
        # Adjust based on system load
        system_load = self.get_current_system_load()
        
        if system_load['cpu_percent'] > 80:
            # High CPU load - reduce parallelism
            base_config['chunk_size'] = max(1, base_config['chunk_size'] // 2)
        elif system_load['cpu_percent'] < 30:
            # Low CPU load - increase parallelism
            base_config['chunk_size'] = min(20, base_config['chunk_size'] * 2)
        
        return base_config
    
    def record_performance(self, operation_name: str, workload_size: int, 
                          config: Dict[str, Any], duration: float):
        """Record performance for future optimization."""
        workload_type = self.classify_workload(operation_name, workload_size)
        config_key = f"{operation_name}:{workload_type}:{workload_size//10}"
        
        perf_data = {
            'config': config,
            'duration': duration,
            'timestamp': time.time(),
            'workload_size': workload_size
        }
        
        with self._lock:
            self._performance_history[config_key].append(perf_data)
            
            # Keep only recent history (last 50 measurements)
            if len(self._performance_history[config_key]) > 50:
                self._performance_history[config_key].popleft()
            
            # Update optimal config if we have enough data
            if len(self._performance_history[config_key]) >= 5:
                self._update_optimal_config(config_key)
    
    def _update_optimal_config(self, config_key: str):
        """Update optimal configuration based on performance history."""
        history = list(self._performance_history[config_key])
        
        # Group by config and calculate average performance
        config_performance = defaultdict(list)
        
        for entry in history[-20:]:  # Use recent history
            config_str = str(sorted(entry['config'].items()))
            config_performance[config_str].append(entry['duration'])
        
        # Find best performing config
        best_config = None
        best_avg_duration = float('inf')
        
        for config_str, durations in config_performance.items():
            if len(durations) >= 3:  # Need at least 3 samples
                avg_duration = sum(durations) / len(durations)
                if avg_duration < best_avg_duration:
                    best_avg_duration = avg_duration
                    # Reconstruct config from string
                    for entry in history:
                        if str(sorted(entry['config'].items())) == config_str:
                            best_config = entry['config']
                            break
        
        if best_config:
            self._optimal_configs[config_key] = best_config
            self.logger.debug(f"Updated optimal config for {config_key}: {best_config}")


class OptimizedQuantumExecutor:
    """High-performance executor optimized for quantum operations."""
    
    def __init__(self):
        self.logger = get_logger("quantum_executor")
        self.scheduler = AdaptiveScheduler()
        self._executors = {}
        self._resource_pools = {}
        
        # Initialize common resource pools
        self._setup_resource_pools()
        
    def _setup_resource_pools(self):
        """Setup resource pools for expensive objects."""
        # Simulator pool
        def create_simulator():
            try:
                from ..jax.simulator import JAXSimulator
                return JAXSimulator(num_qubits=5)  # Default size
            except ImportError:
                return None
        
        self._resource_pools['simulator'] = ResourcePool(
            create_simulator, max_size=5, idle_timeout=300
        )
    
    def _get_executor(self, executor_type: str, max_workers: int) -> ConcurrentExecutor:
        """Get or create executor of specified type."""
        key = f"{executor_type}:{max_workers}"
        
        if key not in self._executors:
            self._executors[key] = ConcurrentExecutor(
                max_workers=max_workers, executor_type=executor_type
            )
        
        return self._executors[key]
    
    def execute_quantum_batch(self, operation_name: str, func: Callable, 
                             args_list: List[Tuple], timeout: Optional[float] = None) -> List[Any]:
        """Execute batch of quantum operations with optimization."""
        if not args_list:
            return []
        
        start_time = time.time()
        workload_size = len(args_list)
        
        # Get optimal configuration
        config = self.scheduler.get_optimal_config(operation_name, workload_size)
        
        # Create optimized executor
        max_workers = min(config.get('max_workers', psutil.cpu_count()), workload_size)
        executor = self._get_executor(config['executor_type'], max_workers)
        
        with PerformanceProfiler(f"batch_{operation_name}") as profiler:
            profiler.update_metrics(concurrent_tasks=max_workers)
            
            # Execute batch
            if config.get('chunk_size', 1) > 1:
                # Use chunked execution for better throughput
                results = executor.map_concurrent(
                    func, args_list, 
                    chunk_size=config['chunk_size'],
                    timeout=timeout
                )
            else:
                # Direct batch execution
                results = executor.submit_batch(func, args_list, timeout)
            
            # Record performance for learning
            duration = time.time() - start_time
            self.scheduler.record_performance(operation_name, workload_size, config, duration)
            
            profiler.update_metrics(
                cache_hits=getattr(func, 'cache_stats', lambda: {'hits': 0})()['hits'],
                cache_misses=getattr(func, 'cache_stats', lambda: {'misses': 0})()['misses']
            )
            
            self.logger.info(
                f"Quantum batch {operation_name}: {len(results)} operations in {duration:.2f}s",
                f"batch_{operation_name}",
                {
                    'batch_size': workload_size,
                    'duration': duration,
                    'throughput': workload_size / duration if duration > 0 else 0,
                    'config': config
                }
            )
            
            return results
    
    def with_resource_pool(self, resource_type: str, func: Callable):
        """Execute function with pooled resource."""
        if resource_type not in self._resource_pools:
            return func()
        
        pool = self._resource_pools[resource_type]
        resource = pool.acquire()
        
        try:
            return func(resource)
        finally:
            pool.release(resource)
    
    def shutdown(self):
        """Shutdown all executors and resource pools."""
        for executor in self._executors.values():
            executor.shutdown()
        
        for pool in self._resource_pools.values():
            pool.shutdown()
        
        self.scheduler.stop_monitoring()
        
        self.logger.info("Quantum executor shutdown complete")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'executors': {k: e.get_stats() for k, e in self._executors.items()},
            'resource_pools': {k: p.get_stats() for k, p in self._resource_pools.items()},
            'system_load': self.scheduler.get_current_system_load(),
            'optimal_configs': len(self.scheduler._optimal_configs)
        }
        
        return stats


# Global optimized executor
_quantum_executor = OptimizedQuantumExecutor()


def optimize_quantum_operation(operation_name: str, timeout: Optional[float] = None):
    """Decorator to optimize quantum operations."""
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(batch_args, *args, **kwargs):
            if isinstance(batch_args, list):
                # Batch operation
                return _quantum_executor.execute_quantum_batch(
                    operation_name, func, batch_args, timeout
                )
            else:
                # Single operation
                with PerformanceProfiler(operation_name):
                    return func(batch_args, *args, **kwargs)
        
        return wrapper
    return decorator


def get_quantum_executor() -> OptimizedQuantumExecutor:
    """Get global quantum executor."""
    return _quantum_executor


def get_performance_summary() -> Dict[str, Any]:
    """Get comprehensive performance summary."""
    return _quantum_executor.get_performance_stats()


# Memory optimization utilities
class MemoryOptimizer:
    """Memory optimization utilities."""
    
    def __init__(self):
        self.logger = get_logger("memory_optimizer")
    
    def optimize_memory_usage(self):
        """Optimize current memory usage."""
        # Force garbage collection
        collected = gc.collect()
        
        # Get memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        
        self.logger.info(
            f"Memory optimization: collected {collected} objects, "
            f"RSS: {memory_info.rss / (1024*1024):.1f} MB",
            "memory_optimization",
            {
                'collected_objects': collected,
                'rss_mb': memory_info.rss / (1024*1024),
                'vms_mb': memory_info.vms / (1024*1024)
            }
        )
        
        return collected
    
    def estimate_operation_memory(self, operation_func: Callable, *args, **kwargs) -> Dict[str, float]:
        """Estimate memory usage of an operation."""
        process = psutil.Process()
        
        # Measure before
        initial_memory = process.memory_info().rss
        
        # Execute operation
        try:
            start_time = time.time()
            result = operation_func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Measure after
            peak_memory = process.memory_info().rss
            
            memory_delta = peak_memory - initial_memory
            
            return {
                'initial_mb': initial_memory / (1024*1024),
                'peak_mb': peak_memory / (1024*1024),
                'delta_mb': memory_delta / (1024*1024),
                'duration_ms': duration * 1000,
                'memory_rate_mb_per_sec': (memory_delta / (1024*1024)) / duration if duration > 0 else 0
            }
        
        except Exception as e:
            self.logger.error(f"Memory estimation failed: {e}", e)
            return {'error': str(e)}


# Global memory optimizer
memory_optimizer = MemoryOptimizer()


# Auto-scaling based on system load
class AutoScaler:
    """Automatic scaling of computational resources."""
    
    def __init__(self):
        self.logger = get_logger("autoscaler")
        self._scaling_history = deque(maxlen=100)
        
        # Scaling thresholds
        self.cpu_scale_up_threshold = 80.0
        self.cpu_scale_down_threshold = 30.0
        self.memory_scale_up_threshold = 85.0
        
        # Scaling factors
        self.scale_up_factor = 1.5
        self.scale_down_factor = 0.7
        
        # Minimum/maximum workers
        self.min_workers = 1
        self.max_workers = psutil.cpu_count() * 2
    
    def should_scale(self, current_workers: int, system_load: Dict[str, float]) -> Tuple[bool, int, str]:
        """Determine if scaling is needed."""
        cpu_percent = system_load['cpu_percent']
        memory_percent = system_load['memory_percent']
        
        # Scale up conditions
        if (cpu_percent > self.cpu_scale_up_threshold or 
            memory_percent > self.memory_scale_up_threshold):
            
            new_workers = min(self.max_workers, 
                            int(current_workers * self.scale_up_factor))
            
            if new_workers > current_workers:
                reason = f"High resource usage (CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%)"
                return True, new_workers, reason
        
        # Scale down conditions
        elif (cpu_percent < self.cpu_scale_down_threshold and 
              memory_percent < 50 and  # Memory not too high
              current_workers > self.min_workers):
            
            new_workers = max(self.min_workers, 
                            int(current_workers * self.scale_down_factor))
            
            if new_workers < current_workers:
                reason = f"Low resource usage (CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%)"
                return True, new_workers, reason
        
        return False, current_workers, "No scaling needed"
    
    def record_scaling_event(self, old_workers: int, new_workers: int, 
                           reason: str, system_load: Dict[str, float]):
        """Record scaling event for analysis."""
        event = {
            'timestamp': time.time(),
            'old_workers': old_workers,
            'new_workers': new_workers,
            'reason': reason,
            'system_load': system_load.copy()
        }
        
        self._scaling_history.append(event)
        
        self.logger.info(
            f"Auto-scaling: {old_workers} → {new_workers} workers ({reason})",
            "autoscaling",
            event
        )
    
    def get_scaling_history(self) -> List[Dict[str, Any]]:
        """Get recent scaling history."""
        return list(self._scaling_history)


# Global auto-scaler
auto_scaler = AutoScaler()