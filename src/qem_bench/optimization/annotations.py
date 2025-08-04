"""
Performance annotations for quantum circuit operations.

This module provides decorators and annotations to mark quantum operations
with performance characteristics, optimization hints, and resource requirements.
"""

import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import inspect

from ..logging import get_logger


class PerformanceHint(Enum):
    """Performance optimization hints."""
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_BOUND = "io_bound"
    CACHE_FRIENDLY = "cache_friendly"
    PARALLELIZABLE = "parallelizable"
    JIT_COMPATIBLE = "jit_compatible"
    PURE_FUNCTION = "pure_function"
    EXPENSIVE = "expensive"
    CHEAP = "cheap"


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    QUANTUM_BACKEND = "quantum_backend"
    NETWORK = "network"
    DISK = "disk"


@dataclass
class PerformanceProfile:
    """Performance profile for a quantum operation."""
    
    # Operation characteristics
    operation_name: str
    complexity_class: str = "O(n)"  # Big-O complexity
    expected_runtime_ms: float = 0.0
    memory_requirement_mb: float = 0.0
    
    # Optimization hints
    hints: List[PerformanceHint] = field(default_factory=list)
    
    # Resource requirements
    required_resources: Dict[ResourceType, float] = field(default_factory=dict)
    
    # Scaling properties
    scales_with_qubits: bool = True
    scales_with_shots: bool = True
    scales_with_circuit_depth: bool = True
    
    # Caching properties
    cacheable: bool = True
    cache_key_params: List[str] = field(default_factory=list)
    
    # Parallelization properties
    parallelizable: bool = False
    parallel_strategy: Optional[str] = None
    min_parallel_size: int = 1
    
    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionMetrics:
    """Metrics collected during operation execution."""
    
    operation_name: str
    execution_time: float
    memory_used_mb: float
    cpu_utilization: float
    
    # Input parameters
    input_params: Dict[str, Any] = field(default_factory=dict)
    
    # Performance counters
    cache_hit: bool = False
    parallel_execution: bool = False
    jit_compiled: bool = False
    
    # Resource usage
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Optimization applied
    optimizations: List[str] = field(default_factory=list)


class PerformanceRegistry:
    """Registry for performance profiles and execution metrics."""
    
    def __init__(self):
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.metrics_history: List[ExecutionMetrics] = []
        self.logger = get_logger(__name__)
    
    def register_profile(self, profile: PerformanceProfile) -> None:
        """Register a performance profile."""
        self.profiles[profile.operation_name] = profile
        self.logger.debug(f"Registered performance profile for {profile.operation_name}")
    
    def get_profile(self, operation_name: str) -> Optional[PerformanceProfile]:
        """Get performance profile for an operation."""
        return self.profiles.get(operation_name)
    
    def record_metrics(self, metrics: ExecutionMetrics) -> None:
        """Record execution metrics."""
        self.metrics_history.append(metrics)
        
        # Update performance profile if available
        if metrics.operation_name in self.profiles:
            profile = self.profiles[metrics.operation_name]
            # Simple exponential moving average for runtime estimate
            alpha = 0.1
            profile.expected_runtime_ms = (
                alpha * metrics.execution_time * 1000 + 
                (1 - alpha) * profile.expected_runtime_ms
            )
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        operation_metrics = [
            m for m in self.metrics_history 
            if m.operation_name == operation_name
        ]
        
        if not operation_metrics:
            return {}
        
        execution_times = [m.execution_time for m in operation_metrics]
        memory_usage = [m.memory_used_mb for m in operation_metrics]
        
        return {
            'call_count': len(operation_metrics),
            'total_time': sum(execution_times),
            'average_time': sum(execution_times) / len(execution_times),
            'min_time': min(execution_times),
            'max_time': max(execution_times),
            'average_memory': sum(memory_usage) / len(memory_usage),
            'cache_hit_rate': sum(1 for m in operation_metrics if m.cache_hit) / len(operation_metrics),
            'parallel_usage_rate': sum(1 for m in operation_metrics if m.parallel_execution) / len(operation_metrics),
        }
    
    def clear_metrics(self) -> None:
        """Clear all recorded metrics."""
        self.metrics_history.clear()


# Global registry instance
_performance_registry = PerformanceRegistry()


def get_performance_registry() -> PerformanceRegistry:
    """Get the global performance registry."""
    return _performance_registry


def performance_profile(
    complexity: str = "O(n)",
    expected_runtime_ms: float = 0.0,
    memory_requirement_mb: float = 0.0,
    hints: Optional[List[PerformanceHint]] = None,
    resources: Optional[Dict[ResourceType, float]] = None,
    cacheable: bool = True,
    parallelizable: bool = False,
    **kwargs
) -> Callable:
    """
    Decorator to annotate functions with performance profiles.
    
    Args:
        complexity: Big-O complexity notation
        expected_runtime_ms: Expected runtime in milliseconds
        memory_requirement_mb: Memory requirement in MB
        hints: List of performance hints
        resources: Required resources
        cacheable: Whether the operation is cacheable
        parallelizable: Whether the operation can be parallelized
        **kwargs: Additional profile parameters
        
    Returns:
        Decorated function with performance profile
    """
    def decorator(func: Callable) -> Callable:
        # Create performance profile
        profile = PerformanceProfile(
            operation_name=func.__name__,
            complexity_class=complexity,
            expected_runtime_ms=expected_runtime_ms,
            memory_requirement_mb=memory_requirement_mb,
            hints=hints or [],
            required_resources=resources or {},
            cacheable=cacheable,
            parallelizable=parallelizable,
            **kwargs
        )
        
        # Register profile
        _performance_registry.register_profile(profile)
        
        # Add profile as function attribute
        func._performance_profile = profile
        
        return func
    
    return decorator


def performance_monitor(
    track_memory: bool = True,
    track_cpu: bool = True,
    auto_optimize: bool = False
) -> Callable:
    """
    Decorator to monitor performance of quantum operations.
    
    Args:
        track_memory: Track memory usage
        track_cpu: Track CPU utilization
        auto_optimize: Automatically apply optimizations based on profile
        
    Returns:
        Decorated function with performance monitoring
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = func.__name__
            
            # Get performance profile if available
            profile = _performance_registry.get_profile(operation_name)
            
            # Start monitoring
            start_time = time.perf_counter()
            
            # Track memory if requested
            import psutil
            process = psutil.Process() if track_memory or track_cpu else None
            
            memory_before = process.memory_info().rss / (1024**2) if process else 0
            cpu_before = process.cpu_percent() if process else 0
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate metrics
                execution_time = time.perf_counter() - start_time
                memory_after = process.memory_info().rss / (1024**2) if process else 0
                cpu_after = process.cpu_percent() if process else 0
                
                memory_used = memory_after - memory_before
                cpu_utilization = (cpu_before + cpu_after) / 2
                
                # Extract input parameters for caching analysis
                input_params = {}
                if args:
                    input_params['args_hash'] = hash(str(args))
                if kwargs:
                    input_params['kwargs_hash'] = hash(str(sorted(kwargs.items())))
                
                # Create execution metrics
                metrics = ExecutionMetrics(
                    operation_name=operation_name,
                    execution_time=execution_time,
                    memory_used_mb=max(0, memory_used),
                    cpu_utilization=cpu_utilization,
                    input_params=input_params,
                )
                
                # Record metrics
                _performance_registry.record_metrics(metrics)
                
                return result
                
            except Exception as e:
                # Record failure metrics
                execution_time = time.perf_counter() - start_time
                metrics = ExecutionMetrics(
                    operation_name=operation_name,
                    execution_time=execution_time,
                    memory_used_mb=0,
                    cpu_utilization=0,
                )
                _performance_registry.record_metrics(metrics)
                raise
        
        return wrapper
    return decorator


def quantum_operation(
    gate_count: Optional[int] = None,
    qubit_count: Optional[int] = None,
    depth: Optional[int] = None,
    **profile_kwargs
) -> Callable:
    """
    Decorator specifically for quantum gate operations.
    
    Args:
        gate_count: Number of gates in the operation
        qubit_count: Number of qubits involved
        depth: Circuit depth
        **profile_kwargs: Additional profile parameters
        
    Returns:
        Decorated quantum operation
    """
    def decorator(func: Callable) -> Callable:
        # Determine complexity based on quantum operation characteristics
        if qubit_count:
            if gate_count and gate_count > qubit_count:
                complexity = f"O(2^{qubit_count} * {gate_count})"
            else:
                complexity = f"O(2^{qubit_count})"
        else:
            complexity = "O(n)"
        
        # Add quantum-specific hints
        hints = profile_kwargs.get('hints', [])
        if qubit_count and qubit_count > 10:
            hints.append(PerformanceHint.MEMORY_INTENSIVE)
        if gate_count and gate_count > 100:
            hints.append(PerformanceHint.CPU_INTENSIVE)
        
        # Apply performance profile
        profiled_func = performance_profile(
            complexity=complexity,
            hints=hints,
            **profile_kwargs
        )(func)
        
        # Apply performance monitoring
        monitored_func = performance_monitor()(profiled_func)
        
        # Add quantum-specific metadata
        monitored_func._gate_count = gate_count
        monitored_func._qubit_count = qubit_count
        monitored_func._depth = depth
        
        return monitored_func
    
    return decorator


def expensive_operation(
    cache_result: bool = True,
    prefer_parallel: bool = True,
    **profile_kwargs
) -> Callable:
    """
    Decorator for expensive quantum operations that benefit from optimization.
    
    Args:
        cache_result: Whether to cache the result
        prefer_parallel: Whether to prefer parallel execution
        **profile_kwargs: Additional profile parameters
        
    Returns:
        Decorated expensive operation
    """
    def decorator(func: Callable) -> Callable:
        hints = profile_kwargs.get('hints', [])
        hints.append(PerformanceHint.EXPENSIVE)
        
        if cache_result:
            hints.append(PerformanceHint.CACHE_FRIENDLY)
        
        if prefer_parallel:
            hints.append(PerformanceHint.PARALLELIZABLE)
        
        return performance_profile(
            hints=hints,
            cacheable=cache_result,
            parallelizable=prefer_parallel,
            **profile_kwargs
        )(func)
    
    return decorator


def circuit_compilation(
    target_backend: Optional[str] = None,
    optimization_level: int = 1
) -> Callable:
    """
    Decorator for circuit compilation operations.
    
    Args:
        target_backend: Target backend for compilation
        optimization_level: Optimization level (0-3)
        
    Returns:
        Decorated compilation function
    """
    def decorator(func: Callable) -> Callable:
        hints = [
            PerformanceHint.CPU_INTENSIVE,
            PerformanceHint.CACHE_FRIENDLY,
            PerformanceHint.EXPENSIVE
        ]
        
        resources = {ResourceType.CPU: optimization_level * 0.5}
        if target_backend:
            resources[ResourceType.QUANTUM_BACKEND] = 0.1
        
        return performance_profile(
            complexity="O(n * d)",  # gates * depth
            hints=hints,
            resources=resources,
            cacheable=True,
            cache_key_params=['circuit_hash', 'backend_config'],
            metadata={
                'target_backend': target_backend,
                'optimization_level': optimization_level
            }
        )(func)
    
    return decorator


def state_vector_operation(
    num_qubits: int,
    operation_type: str = "gate"
) -> Callable:
    """
    Decorator for state vector operations.
    
    Args:
        num_qubits: Number of qubits in the state vector
        operation_type: Type of operation (gate, measurement, etc.)
        
    Returns:
        Decorated state vector operation
    """
    def decorator(func: Callable) -> Callable:
        # Memory requirement grows exponentially with qubits
        memory_mb = (2 ** num_qubits) * 16 / (1024**2)  # Complex128 = 16 bytes
        
        hints = [PerformanceHint.MEMORY_INTENSIVE]
        if num_qubits > 20:
            hints.append(PerformanceHint.EXPENSIVE)
        
        if operation_type == "gate":
            hints.append(PerformanceHint.JIT_COMPATIBLE)
        
        return performance_profile(
            complexity=f"O(2^{num_qubits})",
            memory_requirement_mb=memory_mb,
            hints=hints,
            scales_with_qubits=True,
            parallelizable=num_qubits > 15,
            metadata={
                'num_qubits': num_qubits,
                'operation_type': operation_type
            }
        )(func)
    
    return decorator


def mitigation_method(
    method_name: str,
    overhead_factor: float = 1.0,
    **profile_kwargs
) -> Callable:
    """
    Decorator for error mitigation methods.
    
    Args:
        method_name: Name of the mitigation method
        overhead_factor: Computational overhead factor
        **profile_kwargs: Additional profile parameters
        
    Returns:
        Decorated mitigation method
    """
    def decorator(func: Callable) -> Callable:
        hints = [PerformanceHint.EXPENSIVE]
        
        if overhead_factor > 5.0:
            hints.append(PerformanceHint.CPU_INTENSIVE)
        
        if method_name.lower() in ['zne', 'pec']:
            hints.append(PerformanceHint.PARALLELIZABLE)
        
        return performance_profile(
            expected_runtime_ms=overhead_factor * 1000,  # Base estimate
            hints=hints,
            parallelizable=overhead_factor > 2.0,
            metadata={
                'method_name': method_name,
                'overhead_factor': overhead_factor
            },
            **profile_kwargs
        )(func)
    
    return decorator


def get_operation_recommendations(operation_name: str) -> List[str]:
    """
    Get optimization recommendations for an operation.
    
    Args:
        operation_name: Name of the operation
        
    Returns:
        List of optimization recommendations
    """
    registry = get_performance_registry()
    profile = registry.get_profile(operation_name)
    stats = registry.get_operation_stats(operation_name)
    
    recommendations = []
    
    if not profile:
        return ["No performance profile available - consider adding @performance_profile decorator"]
    
    # Analyze hints
    if PerformanceHint.EXPENSIVE in profile.hints:
        if profile.cacheable and stats.get('cache_hit_rate', 0) < 0.5:
            recommendations.append("Operation is expensive but cache hit rate is low - improve caching strategy")
    
    if PerformanceHint.PARALLELIZABLE in profile.hints:
        if stats.get('parallel_usage_rate', 0) < 0.5:
            recommendations.append("Operation is parallelizable but parallel execution rate is low")
    
    if PerformanceHint.JIT_COMPATIBLE in profile.hints:
        recommendations.append("Operation is JIT compatible - consider using @jax.jit decorator")
    
    if PerformanceHint.MEMORY_INTENSIVE in profile.hints:
        recommendations.append("Memory intensive operation - consider memory pooling or streaming")
    
    # Analyze performance trends
    if stats and stats.get('call_count', 0) > 10:
        avg_time = stats.get('average_time', 0)
        if avg_time > profile.expected_runtime_ms / 1000 * 2:
            recommendations.append("Operation taking longer than expected - investigate performance bottlenecks")
    
    return recommendations or ["Operation performance appears optimal"]


# Example usage annotations for common quantum operations
@quantum_operation(gate_count=1, qubit_count=1, depth=1)
def single_qubit_gate(state, gate_matrix):
    """Apply single qubit gate."""
    pass


@quantum_operation(gate_count=1, qubit_count=2, depth=1)
def two_qubit_gate(state, gate_matrix):
    """Apply two qubit gate."""
    pass


@expensive_operation(cache_result=True, prefer_parallel=True)
def quantum_simulation(circuit, shots=1024):
    """Run quantum circuit simulation."""
    pass


@circuit_compilation(optimization_level=2)
def compile_circuit(circuit, backend):
    """Compile quantum circuit for backend."""
    pass


@mitigation_method("ZNE", overhead_factor=3.0)
def zero_noise_extrapolation(circuit, backend, noise_factors):
    """Apply zero-noise extrapolation."""
    pass