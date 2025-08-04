"""
Performance optimization framework for QEM-Bench.

This module provides comprehensive performance optimization features including:
- Automatic performance tuning
- Intelligent caching systems
- JIT compilation optimizations
- Parallel execution strategies
- Memory and resource optimization
- Performance profiling and analysis
"""

from .performance_optimizer import PerformanceOptimizer, OptimizationConfig
from .cache_manager import CacheManager, CacheConfig, CacheStrategy
from .jit_compiler import JITCompiler, CompilationConfig
from .parallel_executor import ParallelExecutor, ExecutionStrategy
from .memory_manager import MemoryManager, MemoryPool
from .profiler import PerformanceProfiler, ProfilingResult
from .auto_scaler import AutoScaler, ScalingConfig
from .load_balancer import LoadBalancer, BalancingStrategy

__all__ = [
    # Core optimization components
    "PerformanceOptimizer",
    "OptimizationConfig",
    # Caching system
    "CacheManager", 
    "CacheConfig",
    "CacheStrategy",
    # JIT compilation
    "JITCompiler",
    "CompilationConfig",
    # Parallel execution
    "ParallelExecutor",
    "ExecutionStrategy",
    # Memory management
    "MemoryManager",
    "MemoryPool",
    # Performance profiling
    "PerformanceProfiler",
    "ProfilingResult",
    # Auto-scaling
    "AutoScaler",
    "ScalingConfig",
    # Load balancing
    "LoadBalancer",
    "BalancingStrategy",
]