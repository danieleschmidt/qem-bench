"""
Performance optimizer for automatic performance tuning of quantum error mitigation.

This module provides intelligent optimization of mitigation methods, circuit execution,
and resource allocation to maximize performance while maintaining correctness.
"""

import time
import logging
import warnings
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import jax
import jax.numpy as jnp
import numpy as np

from ..logging import get_logger


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    CONSERVATIVE = "conservative"  # Safe optimizations only
    AGGRESSIVE = "aggressive"     # All available optimizations
    ADAPTIVE = "adaptive"         # Adapt based on performance metrics
    CUSTOM = "custom"            # User-defined optimization rules


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU = "gpu" 
    TPU = "tpu"
    MEMORY = "memory"
    NETWORK = "network"


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    
    # General optimization settings
    strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE
    enable_jit: bool = True
    enable_caching: bool = True
    enable_parallel: bool = True
    enable_memory_pool: bool = True
    
    # Performance targets
    target_speedup: float = 2.0
    max_memory_usage_gb: float = 8.0
    timeout_seconds: float = 300.0
    
    # JIT compilation settings
    jit_warmup_iterations: int = 3
    jit_cache_size: int = 128
    
    # Caching settings
    cache_max_size_gb: float = 2.0
    cache_ttl_seconds: float = 3600.0
    enable_disk_cache: bool = False
    
    # Parallel execution settings
    max_workers: int = mp.cpu_count()
    chunk_size: Optional[int] = None
    use_processes: bool = False  # Use threads by default
    
    # Memory management
    memory_pool_size_gb: float = 4.0
    enable_memory_mapping: bool = True
    garbage_collection_threshold: float = 0.8
    
    # Profiling and monitoring
    enable_profiling: bool = True
    profiling_interval: float = 1.0
    performance_history_size: int = 100
    
    # Auto-tuning parameters
    auto_tune_enabled: bool = True
    tune_iterations: int = 10
    tune_tolerance: float = 0.05


@dataclass
class OptimizationResult:
    """Result of performance optimization."""
    
    original_time: float
    optimized_time: float
    speedup: float
    memory_usage_mb: float
    optimization_overhead: float
    applied_optimizations: List[str]
    performance_metrics: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)


class PerformanceOptimizer:
    """
    Automatic performance optimizer for quantum error mitigation.
    
    This class provides intelligent optimization of quantum circuits, mitigation
    methods, and resource allocation to maximize performance while maintaining
    correctness.
    
    Features:
    - Automatic JIT compilation with warm-up
    - Intelligent caching with multi-level storage
    - Parallel execution optimization
    - Memory management and pooling
    - Performance profiling and analysis
    - Auto-tuning of optimization parameters
    - Resource usage monitoring and scaling
    
    Example:
        >>> optimizer = PerformanceOptimizer()
        >>> optimized_zne = optimizer.optimize_mitigation_method(zne_instance)
        >>> result = optimized_zne.mitigate(circuit, backend, observable)
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize performance optimizer.
        
        Args:
            config: Optimization configuration (uses defaults if None)
        """
        self.config = config or OptimizationConfig()
        self.logger = get_logger()
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self._optimization_cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
        # Component instances (lazy initialization)
        self._cache_manager = None
        self._jit_compiler = None
        self._parallel_executor = None
        self._memory_manager = None
        self._profiler = None
        
        # Auto-tuning state
        self._auto_tune_state = {
            'iteration': 0,
            'best_params': None,
            'best_performance': float('inf')
        }
        
        self.logger.info(f"PerformanceOptimizer initialized with strategy: {self.config.strategy.value}")
    
    @property
    def cache_manager(self):
        """Lazy initialization of cache manager."""
        if self._cache_manager is None:
            from .cache_manager import CacheManager, CacheConfig
            cache_config = CacheConfig(
                max_size_gb=self.config.cache_max_size_gb,
                ttl_seconds=self.config.cache_ttl_seconds,
                enable_disk_cache=self.config.enable_disk_cache
            )
            self._cache_manager = CacheManager(cache_config)
        return self._cache_manager
    
    @property
    def jit_compiler(self):
        """Lazy initialization of JIT compiler."""
        if self._jit_compiler is None:
            from .jit_compiler import JITCompiler, CompilationConfig
            compile_config = CompilationConfig(
                warmup_iterations=self.config.jit_warmup_iterations,
                cache_size=self.config.jit_cache_size
            )
            self._jit_compiler = JITCompiler(compile_config)
        return self._jit_compiler
    
    @property
    def parallel_executor(self):
        """Lazy initialization of parallel executor."""
        if self._parallel_executor is None:
            from .parallel_executor import ParallelExecutor, ExecutionStrategy
            strategy = ExecutionStrategy.PROCESS if self.config.use_processes else ExecutionStrategy.THREAD
            self._parallel_executor = ParallelExecutor(
                strategy=strategy,
                max_workers=self.config.max_workers,
                chunk_size=self.config.chunk_size
            )
        return self._parallel_executor
    
    @property
    def memory_manager(self):
        """Lazy initialization of memory manager."""
        if self._memory_manager is None:
            from .memory_manager import MemoryManager
            self._memory_manager = MemoryManager(
                pool_size_gb=self.config.memory_pool_size_gb,
                enable_memory_mapping=self.config.enable_memory_mapping,
                gc_threshold=self.config.garbage_collection_threshold
            )
        return self._memory_manager
    
    @property
    def profiler(self):
        """Lazy initialization of performance profiler."""
        if self._profiler is None:
            from .profiler import PerformanceProfiler
            self._profiler = PerformanceProfiler(
                enable=self.config.enable_profiling,
                interval=self.config.profiling_interval
            )
        return self._profiler
    
    def optimize_function(
        self,
        func: Callable,
        *args,
        enable_jit: Optional[bool] = None,
        enable_cache: Optional[bool] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Tuple[Any, OptimizationResult]:
        """
        Optimize a general function with available optimizations.
        
        Args:
            func: Function to optimize
            *args: Positional arguments for function
            enable_jit: Override JIT compilation setting
            enable_cache: Override caching setting
            cache_key: Custom cache key (auto-generated if None)
            **kwargs: Keyword arguments for function
            
        Returns:
            Tuple of (function_result, optimization_result)
        """
        start_time = time.time()
        applied_optimizations = []
        
        # Generate cache key if not provided
        if cache_key is None:
            cache_key = self._generate_cache_key(func, args, kwargs)
        
        # Check cache first
        if (enable_cache or (enable_cache is None and self.config.enable_caching)):
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                applied_optimizations.append("cache_hit")
                optimization_time = time.time() - start_time
                
                return cached_result, OptimizationResult(
                    original_time=0.0,  # Not measured for cache hits
                    optimized_time=optimization_time,
                    speedup=float('inf'),  # Infinite speedup for cache hits
                    memory_usage_mb=0.0,
                    optimization_overhead=optimization_time,
                    applied_optimizations=applied_optimizations,
                    performance_metrics={"cache_hit": True}
                )
        
        # Measure baseline performance
        baseline_start = time.time()
        baseline_result = func(*args, **kwargs)
        baseline_time = time.time() - baseline_start
        
        # Apply JIT compilation if enabled
        optimized_func = func
        if (enable_jit or (enable_jit is None and self.config.enable_jit)):
            try:
                optimized_func = self.jit_compiler.compile(func)
                applied_optimizations.append("jit_compilation")
            except Exception as e:
                self.logger.warning(f"JIT compilation failed: {e}")
        
        # Execute optimized function
        optimized_start = time.time()
        try:
            optimized_result = optimized_func(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Optimized execution failed: {e}, falling back to original")
            optimized_result = baseline_result
        
        optimized_time = time.time() - optimized_start
        total_time = time.time() - start_time
        
        # Cache the result if caching is enabled
        if (enable_cache or (enable_cache is None and self.config.enable_caching)):
            self.cache_manager.put(cache_key, optimized_result)
            applied_optimizations.append("result_cached")
        
        # Calculate performance metrics
        speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
        memory_usage = self._estimate_memory_usage(optimized_result)
        
        optimization_result = OptimizationResult(
            original_time=baseline_time,
            optimized_time=optimized_time,
            speedup=speedup,
            memory_usage_mb=memory_usage,
            optimization_overhead=total_time - optimized_time,
            applied_optimizations=applied_optimizations,
            performance_metrics={
                "baseline_time": baseline_time,
                "optimized_time": optimized_time,
                "total_time": total_time,
                "memory_mb": memory_usage
            }
        )
        
        # Record performance for auto-tuning
        self._record_performance(optimization_result)
        
        return optimized_result, optimization_result
    
    def optimize_mitigation_method(self, mitigation_instance: Any) -> Any:
        """
        Create an optimized version of a mitigation method.
        
        Args:
            mitigation_instance: Instance of mitigation method (ZNE, PEC, etc.)
            
        Returns:
            Optimized mitigation method instance
        """
        from .optimized_mitigation import create_optimized_mitigation
        return create_optimized_mitigation(mitigation_instance, self)
    
    def optimize_circuit_execution(
        self,
        circuits: Union[Any, List[Any]],
        backend: Any,
        execution_func: Optional[Callable] = None,
        **execution_kwargs
    ) -> Tuple[Any, OptimizationResult]:
        """
        Optimize execution of quantum circuits.
        
        Args:
            circuits: Single circuit or list of circuits
            backend: Quantum backend
            execution_func: Custom execution function
            **execution_kwargs: Arguments for circuit execution
            
        Returns:
            Tuple of (execution_results, optimization_result)
        """
        if not isinstance(circuits, list):
            circuits = [circuits]
        
        if execution_func is None:
            execution_func = lambda c, b, **kw: b.run(c, **kw)
        
        # Determine optimal execution strategy
        if len(circuits) > 1 and self.config.enable_parallel:
            # Parallel execution for multiple circuits
            def parallel_execution():
                return self.parallel_executor.execute_batch(
                    execution_func, 
                    [(circuit, backend) for circuit in circuits],
                    **execution_kwargs
                )
            
            return self.optimize_function(
                parallel_execution,
                cache_key=f"circuit_batch_{len(circuits)}_{hash(str(execution_kwargs))}"
            )
        else:
            # Single circuit execution
            circuit = circuits[0]
            return self.optimize_function(
                execution_func,
                circuit,
                backend,
                cache_key=f"circuit_single_{hash(str(circuit))}_{hash(str(execution_kwargs))}",
                **execution_kwargs
            )
    
    def auto_tune(self, workload_func: Callable, *args, **kwargs) -> OptimizationConfig:
        """
        Automatically tune optimization parameters for a specific workload.
        
        Args:
            workload_func: Function representing the workload to optimize
            *args: Arguments for workload function
            **kwargs: Keyword arguments for workload function
            
        Returns:
            Optimized configuration
        """
        best_config = self.config
        best_time = float('inf')
        
        # Parameter search space
        search_space = {
            'jit_warmup_iterations': [1, 2, 3, 5],
            'max_workers': [1, 2, 4, mp.cpu_count()],
            'chunk_size': [None, 1, 10, 100],
            'cache_max_size_gb': [0.5, 1.0, 2.0, 4.0],
        }
        
        self.logger.info(f"Starting auto-tuning with {self.config.tune_iterations} iterations")
        
        for iteration in range(self.config.tune_iterations):
            # Sample parameters
            test_config = self._sample_config(search_space)
            
            # Create temporary optimizer with test config
            temp_optimizer = PerformanceOptimizer(test_config)
            
            try:
                # Test the configuration
                start_time = time.time()
                result, opt_result = temp_optimizer.optimize_function(
                    workload_func, *args, **kwargs
                )
                execution_time = time.time() - start_time
                
                # Update best configuration if better
                if execution_time < best_time:
                    best_time = execution_time
                    best_config = test_config
                    self.logger.info(f"New best configuration found: {execution_time:.4f}s")
                
            except Exception as e:
                self.logger.warning(f"Auto-tune iteration {iteration} failed: {e}")
                continue
        
        self.logger.info(f"Auto-tuning completed. Best time: {best_time:.4f}s")
        return best_config
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary containing performance statistics and recommendations
        """
        if not self.performance_history:
            return {"status": "No performance data available"}
        
        # Calculate statistics
        speedups = [entry['speedup'] for entry in self.performance_history]
        memory_usage = [entry['memory_usage_mb'] for entry in self.performance_history]
        
        avg_speedup = np.mean(speedups)
        max_speedup = np.max(speedups)
        avg_memory = np.mean(memory_usage)
        max_memory = np.max(memory_usage)
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        return {
            "performance_summary": {
                "total_optimizations": len(self.performance_history),
                "average_speedup": avg_speedup,
                "maximum_speedup": max_speedup,
                "average_memory_mb": avg_memory,
                "maximum_memory_mb": max_memory,
            },
            "optimization_statistics": {
                "jit_compilation_success_rate": self._calc_success_rate("jit_compilation"),
                "cache_hit_rate": self._calc_success_rate("cache_hit"),
                "parallel_execution_usage": self._calc_success_rate("parallel_execution"),
            },
            "resource_usage": {
                "memory_pool_usage": self.memory_manager.get_usage_stats() if self._memory_manager else {},
                "cache_usage": self.cache_manager.get_usage_stats() if self._cache_manager else {},
            },
            "recommendations": recommendations,
            "config": {
                "current_strategy": self.config.strategy.value,
                "optimizations_enabled": {
                    "jit": self.config.enable_jit,
                    "caching": self.config.enable_caching,
                    "parallel": self.config.enable_parallel,
                    "memory_pool": self.config.enable_memory_pool,
                }
            }
        }
    
    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function and arguments."""
        func_name = getattr(func, '__name__', str(func))
        args_hash = hash(str(args))
        kwargs_hash = hash(str(sorted(kwargs.items())))
        return f"{func_name}_{args_hash}_{kwargs_hash}"
    
    def _estimate_memory_usage(self, result: Any) -> float:
        """Estimate memory usage of result in MB."""
        try:
            if hasattr(result, 'nbytes'):
                return result.nbytes / (1024 * 1024)
            elif isinstance(result, (list, tuple)):
                total_bytes = sum(
                    getattr(item, 'nbytes', 64) for item in result  # 64 bytes default
                )
                return total_bytes / (1024 * 1024)
            else:
                return 0.1  # Default estimate
        except:
            return 0.1
    
    def _record_performance(self, result: OptimizationResult) -> None:
        """Record performance metrics for analysis."""
        with self._lock:
            self.performance_history.append({
                'timestamp': time.time(),
                'speedup': result.speedup,
                'memory_usage_mb': result.memory_usage_mb,
                'applied_optimizations': result.applied_optimizations,
                'original_time': result.original_time,
                'optimized_time': result.optimized_time,
            })
            
            # Maintain history size limit
            if len(self.performance_history) > self.config.performance_history_size:
                self.performance_history.pop(0)
    
    def _sample_config(self, search_space: Dict[str, List[Any]]) -> OptimizationConfig:
        """Sample a configuration from the search space."""
        import random
        
        new_config = OptimizationConfig()
        
        for param, values in search_space.items():
            if hasattr(new_config, param):
                setattr(new_config, param, random.choice(values))
        
        return new_config
    
    def _calc_success_rate(self, optimization_type: str) -> float:
        """Calculate success rate for specific optimization type."""
        if not self.performance_history:
            return 0.0
        
        total = len(self.performance_history)
        successful = sum(
            1 for entry in self.performance_history
            if optimization_type in entry['applied_optimizations']
        )
        
        return successful / total
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if not self.performance_history:
            return ["Collect more performance data for recommendations"]
        
        # Analyze performance patterns
        avg_speedup = np.mean([entry['speedup'] for entry in self.performance_history])
        jit_success_rate = self._calc_success_rate("jit_compilation")
        cache_hit_rate = self._calc_success_rate("cache_hit")
        
        # Generate specific recommendations
        if avg_speedup < 1.5:
            recommendations.append("Consider enabling more aggressive optimizations")
        
        if jit_success_rate < 0.5:
            recommendations.append("JIT compilation success rate is low - check for incompatible operations")
        
        if cache_hit_rate < 0.2:
            recommendations.append("Cache hit rate is low - consider increasing cache size or adjusting TTL")
        
        avg_memory = np.mean([entry['memory_usage_mb'] for entry in self.performance_history])
        if avg_memory > self.config.max_memory_usage_gb * 1024 * 0.8:
            recommendations.append("Memory usage is high - consider enabling memory pooling or reducing batch sizes")
        
        if not recommendations:
            recommendations.append("Performance is optimal with current configuration")
        
        return recommendations


def create_performance_optimizer(
    strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE,
    **config_kwargs
) -> PerformanceOptimizer:
    """
    Create a performance optimizer with specified strategy.
    
    Args:
        strategy: Optimization strategy to use
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured PerformanceOptimizer instance
    """
    config = OptimizationConfig(strategy=strategy, **config_kwargs)
    return PerformanceOptimizer(config)