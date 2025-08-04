"""
JIT compiler for optimizing JAX operations in quantum error mitigation.

This module provides intelligent JIT compilation with warm-up, caching,
and adaptive optimization strategies for maximum performance.
"""

import time
import functools
import threading
import weakref
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import inspect
import pickle
import hashlib

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
import numpy as np

from ..logging import get_logger


class CompilationStrategy(Enum):
    """JIT compilation strategies."""
    EAGER = "eager"          # Compile immediately
    LAZY = "lazy"            # Compile on first use
    ADAPTIVE = "adaptive"    # Adapt based on usage patterns
    BATCH = "batch"          # Batch compilation for related functions


class OptimizationLevel(Enum):
    """Optimization levels for compilation."""
    NONE = 0      # No optimization
    BASIC = 1     # Basic optimizations
    AGGRESSIVE = 2  # Aggressive optimizations
    MAXIMUM = 3   # Maximum optimizations (may be unstable)


@dataclass
class CompilationConfig:
    """Configuration for JIT compilation."""
    
    # Compilation strategy
    strategy: CompilationStrategy = CompilationStrategy.ADAPTIVE
    optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE
    
    # Warm-up settings
    warmup_iterations: int = 3
    warmup_sizes: List[int] = field(default_factory=lambda: [1, 4, 16])
    auto_warmup: bool = True
    
    # Caching settings
    cache_size: int = 128
    cache_compiled_functions: bool = True
    persistent_cache: bool = False
    
    # Performance settings
    static_argnums: Optional[Tuple[int, ...]] = None
    static_argnames: Optional[Tuple[str, ...]] = None
    donate_argnums: Optional[Tuple[int, ...]] = None
    
    # Backend settings
    backend: Optional[str] = None  # 'cpu', 'gpu', 'tpu', or None for auto
    device_count: int = 1
    
    # Advanced settings
    inline: bool = True
    keep_unused: bool = False
    abstracted_axes: Optional[Dict[str, str]] = None


@dataclass
class CompilationResult:
    """Result of JIT compilation."""
    
    original_function: Callable
    compiled_function: Callable
    compilation_time: float
    warmup_time: float
    memory_usage_mb: float
    optimization_level: OptimizationLevel
    backend_info: Dict[str, Any]
    static_args: Dict[str, Any] = field(default_factory=dict)


class FunctionCache:
    """Cache for compiled functions with intelligent eviction."""
    
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.cache: Dict[str, Tuple[Callable, CompilationResult]] = {}
        self.access_order: List[str] = []
        self.access_count: Dict[str, int] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Tuple[Callable, CompilationResult]]:
        """Get compiled function from cache."""
        with self._lock:
            if key in self.cache:
                # Update access statistics
                self.access_count[key] = self.access_count.get(key, 0) + 1
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                
                return self.cache[key]
            return None
    
    def put(self, key: str, func: Callable, result: CompilationResult) -> None:
        """Put compiled function into cache."""
        with self._lock:
            # Evict if necessary
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = (func, result)
            self.access_order.append(key)
            self.access_count[key] = 1
    
    def _evict_lru(self) -> None:
        """Evict least recently used function."""
        if self.access_order:
            key_to_evict = self.access_order.pop(0)
            if key_to_evict in self.cache:
                del self.cache[key_to_evict]
            if key_to_evict in self.access_count:
                del self.access_count[key_to_evict]


class JITCompiler:
    """
    Intelligent JIT compiler for quantum operations using JAX.
    
    This class provides automatic JIT compilation with warm-up, caching,
    and adaptive optimization strategies to maximize performance of
    quantum error mitigation computations.
    
    Features:
    - Intelligent function analysis and compilation
    - Automatic warm-up with representative inputs
    - Function caching with eviction policies
    - Multi-backend support (CPU, GPU, TPU)
    - Adaptive optimization based on usage patterns
    - Performance profiling and monitoring
    - Batch compilation of related functions
    
    Example:
        >>> compiler = JITCompiler()
        >>> 
        >>> @compiler.jit
        >>> def quantum_operation(state, gate_matrix):
        ...     return jnp.dot(gate_matrix, state)
        >>> 
        >>> # Function is automatically compiled and warmed up
        >>> result = quantum_operation(initial_state, hadamard_gate)
    """
    
    def __init__(self, config: Optional[CompilationConfig] = None):
        """
        Initialize JIT compiler.
        
        Args:
            config: Compilation configuration (uses defaults if None)
        """
        self.config = config or CompilationConfig()
        self.logger = get_logger(__name__)
        
        # Function cache
        self.function_cache = FunctionCache(self.config.cache_size)
        
        # Compilation statistics
        self.stats = {
            'compilations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'warmup_time_total': 0.0,
            'compilation_time_total': 0.0,
        }
        
        # Adaptive compilation state
        self.usage_patterns: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # Set up JAX backend if specified
        if self.config.backend:
            jax.config.update('jax_platform_name', self.config.backend)
        
        self.logger.info(f"JITCompiler initialized with backend: {jax.default_backend()}")
    
    def compile(
        self,
        func: Callable,
        static_argnums: Optional[Tuple[int, ...]] = None,
        static_argnames: Optional[Tuple[str, ...]] = None,
        donate_argnums: Optional[Tuple[int, ...]] = None,
        warmup_inputs: Optional[List[Tuple]] = None,
        force_recompile: bool = False
    ) -> Callable:
        """
        Compile function with JIT optimization.
        
        Args:
            func: Function to compile
            static_argnums: Indices of static arguments
            static_argnames: Names of static arguments  
            donate_argnums: Indices of arguments to donate
            warmup_inputs: Inputs for warm-up compilation
            force_recompile: Force recompilation even if cached
            
        Returns:
            JIT-compiled function
        """
        # Generate cache key
        func_key = self._generate_function_key(
            func, static_argnums, static_argnames, donate_argnums
        )
        
        # Check cache first
        if not force_recompile:
            cached = self.function_cache.get(func_key)
            if cached is not None:
                self.stats['cache_hits'] += 1
                self.logger.debug(f"Using cached compilation for {func.__name__}")
                return cached[0]
        
        self.stats['cache_misses'] += 1
        
        # Perform compilation
        start_time = time.time()
        
        try:
            # Create JIT configuration
            jit_kwargs = self._build_jit_kwargs(static_argnums, static_argnames, donate_argnums)
            
            # Compile the function
            compiled_func = jit(func, **jit_kwargs)
            
            compilation_time = time.time() - start_time
            
            # Perform warm-up
            warmup_start = time.time()
            if self.config.auto_warmup or warmup_inputs:
                self._warmup_function(compiled_func, func, warmup_inputs)
            warmup_time = time.time() - warmup_start
            
            # Create compilation result
            result = CompilationResult(
                original_function=func,
                compiled_function=compiled_func,
                compilation_time=compilation_time,
                warmup_time=warmup_time,
                memory_usage_mb=self._estimate_memory_usage(compiled_func),
                optimization_level=self.config.optimization_level,
                backend_info=self._get_backend_info(),
                static_args={
                    'static_argnums': static_argnums,
                    'static_argnames': static_argnames,
                    'donate_argnums': donate_argnums
                }
            )
            
            # Cache the result
            if self.config.cache_compiled_functions:
                self.function_cache.put(func_key, compiled_func, result)
            
            # Update statistics
            self.stats['compilations'] += 1
            self.stats['compilation_time_total'] += compilation_time
            self.stats['warmup_time_total'] += warmup_time
            
            self.logger.info(
                f"Compiled {func.__name__} in {compilation_time:.3f}s "
                f"(warmup: {warmup_time:.3f}s)"
            )
            
            return compiled_func
            
        except Exception as e:
            self.logger.error(f"Failed to compile {func.__name__}: {e}")
            # Return original function as fallback
            return func
    
    def jit(
        self,
        static_argnums: Optional[Union[int, Tuple[int, ...]]] = None,
        static_argnames: Optional[Union[str, Tuple[str, ...]]] = None,
        donate_argnums: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Callable:
        """
        Decorator for JIT compilation.
        
        Args:
            static_argnums: Static argument indices
            static_argnames: Static argument names
            donate_argnums: Donated argument indices
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            # Normalize arguments to tuples
            static_argnums_tuple = self._normalize_to_tuple(static_argnums)
            static_argnames_tuple = self._normalize_to_tuple(static_argnames)
            donate_argnums_tuple = self._normalize_to_tuple(donate_argnums)
            
            # Use lazy compilation strategy
            if self.config.strategy == CompilationStrategy.LAZY:
                return self._create_lazy_compiled_function(
                    func, static_argnums_tuple, static_argnames_tuple, donate_argnums_tuple
                )
            else:
                # Compile immediately
                return self.compile(
                    func, static_argnums_tuple, static_argnames_tuple, donate_argnums_tuple
                )
        
        return decorator
    
    def vmap_compile(
        self,
        func: Callable,
        in_axes: Union[int, Tuple] = 0,
        out_axes: Union[int, Tuple] = 0,
        axis_name: Optional[str] = None
    ) -> Callable:
        """
        Compile function with vectorization (vmap).
        
        Args:
            func: Function to vectorize and compile
            in_axes: Input axes for vectorization
            out_axes: Output axes for vectorization
            axis_name: Name for the mapped axis
            
        Returns:
            Vectorized and compiled function
        """
        # First apply vmap
        vmapped_func = vmap(func, in_axes=in_axes, out_axes=out_axes, axis_name=axis_name)
        
        # Then compile
        return self.compile(vmapped_func)
    
    def pmap_compile(
        self,
        func: Callable,
        axis_name: Optional[str] = None,
        in_axes: Union[int, Tuple] = 0,
        out_axes: Union[int, Tuple] = 0
    ) -> Callable:
        """
        Compile function with parallelization (pmap).
        
        Args:
            func: Function to parallelize and compile
            axis_name: Name for the mapped axis
            in_axes: Input axes for parallelization
            out_axes: Output axes for parallelization
            
        Returns:
            Parallelized and compiled function
        """
        # Apply pmap
        pmapped_func = pmap(
            func, 
            axis_name=axis_name,
            in_axes=in_axes,
            out_axes=out_axes
        )
        
        return pmapped_func  # pmap already compiles
    
    def compile_batch(self, functions: List[Callable], **kwargs) -> List[Callable]:
        """
        Compile multiple functions together for better optimization.
        
        Args:
            functions: List of functions to compile
            **kwargs: Additional compilation arguments
            
        Returns:
            List of compiled functions
        """
        compiled_functions = []
        
        # Compile functions sequentially (could be optimized for batch compilation)
        for func in functions:
            compiled_func = self.compile(func, **kwargs)
            compiled_functions.append(compiled_func)
        
        return compiled_functions
    
    def analyze_function(self, func: Callable) -> Dict[str, Any]:
        """
        Analyze function for compilation optimization opportunities.
        
        Args:
            func: Function to analyze
            
        Returns:
            Analysis results with optimization recommendations
        """
        analysis = {
            'function_name': func.__name__,
            'signature': str(inspect.signature(func)),
            'source_available': True,
            'recommendations': []
        }
        
        try:
            # Try to get source code
            source = inspect.getsource(func)
            analysis['source_length'] = len(source)
            
            # Simple static analysis
            if 'jnp.' in source or 'jax.' in source:
                analysis['recommendations'].append("Function uses JAX operations - good for JIT")
            
            if 'for ' in source or 'while ' in source:
                analysis['recommendations'].append("Function contains loops - may benefit from JIT")
            
            if 'numpy' in source and 'jax' not in source:
                analysis['recommendations'].append("Function uses NumPy - consider converting to JAX")
                
        except Exception:
            analysis['source_available'] = False
            analysis['recommendations'].append("Source not available - runtime analysis needed")
        
        return analysis
    
    def get_compilation_stats(self) -> Dict[str, Any]:
        """
        Get compilation statistics.
        
        Returns:
            Dictionary with compilation statistics
        """
        with self._lock:
            cache_size = len(self.function_cache.cache)
            total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
            hit_rate = (self.stats['cache_hits'] / total_requests) if total_requests > 0 else 0
            
            return {
                'total_compilations': self.stats['compilations'],
                'cache_hits': self.stats['cache_hits'],
                'cache_misses': self.stats['cache_misses'],
                'cache_hit_rate': hit_rate,
                'cached_functions': cache_size,
                'max_cache_size': self.config.cache_size,
                'total_compilation_time': self.stats['compilation_time_total'],
                'total_warmup_time': self.stats['warmup_time_total'],
                'average_compilation_time': (
                    self.stats['compilation_time_total'] / self.stats['compilations']
                    if self.stats['compilations'] > 0 else 0
                ),
                'backend': jax.default_backend(),
                'devices': [str(device) for device in jax.devices()],
            }
    
    def clear_cache(self) -> None:
        """Clear the function compilation cache."""
        with self._lock:
            self.function_cache.cache.clear()
            self.function_cache.access_order.clear()
            self.function_cache.access_count.clear()
            self.logger.info("Compilation cache cleared")
    
    def _generate_function_key(
        self,
        func: Callable,
        static_argnums: Optional[Tuple[int, ...]] = None,
        static_argnames: Optional[Tuple[str, ...]] = None,
        donate_argnums: Optional[Tuple[int, ...]] = None
    ) -> str:
        """Generate unique key for function compilation."""
        func_name = func.__name__
        func_module = getattr(func, '__module__', 'unknown')
        
        # Try to get function source code for more precise caching
        try:
            source = inspect.getsource(func)
            source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        except:
            # Fallback to function id
            source_hash = str(id(func))[-8:]
        
        # Include compilation parameters in key
        static_args_str = str(static_argnums) + str(static_argnames) + str(donate_argnums)
        args_hash = hashlib.md5(static_args_str.encode()).hexdigest()[:8]
        
        return f"{func_module}.{func_name}_{source_hash}_{args_hash}"
    
    def _build_jit_kwargs(
        self,
        static_argnums: Optional[Tuple[int, ...]] = None,
        static_argnames: Optional[Tuple[str, ...]] = None,
        donate_argnums: Optional[Tuple[int, ...]] = None
    ) -> Dict[str, Any]:
        """Build kwargs for JAX jit function."""
        kwargs = {}
        
        # Use provided arguments or config defaults
        if static_argnums is not None:
            kwargs['static_argnums'] = static_argnums
        elif self.config.static_argnums is not None:
            kwargs['static_argnums'] = self.config.static_argnums
        
        if static_argnames is not None:
            kwargs['static_argnames'] = static_argnames
        elif self.config.static_argnames is not None:
            kwargs['static_argnames'] = self.config.static_argnames
        
        if donate_argnums is not None:
            kwargs['donate_argnums'] = donate_argnums
        elif self.config.donate_argnums is not None:
            kwargs['donate_argnums'] = self.config.donate_argnums
        
        # Add other JAX-specific options
        if hasattr(self.config, 'inline'):
            kwargs['inline'] = self.config.inline
        
        return kwargs
    
    def _warmup_function(
        self,
        compiled_func: Callable,
        original_func: Callable,
        warmup_inputs: Optional[List[Tuple]] = None
    ) -> None:
        """Warm up compiled function with representative inputs."""
        if warmup_inputs:
            # Use provided warmup inputs
            for inputs in warmup_inputs:
                try:
                    compiled_func(*inputs)
                except Exception as e:
                    self.logger.warning(f"Warmup failed with provided inputs: {e}")
        else:
            # Generate synthetic warmup inputs
            self._generate_synthetic_warmup(compiled_func, original_func)
    
    def _generate_synthetic_warmup(self, compiled_func: Callable, original_func: Callable) -> None:
        """Generate synthetic inputs for function warmup."""
        try:
            # Try to infer input shapes from function signature
            sig = inspect.signature(original_func)
            
            # Generate simple test inputs for each parameter
            test_args = []
            for param in sig.parameters.values():
                if param.annotation == jnp.ndarray or 'array' in str(param.annotation).lower():
                    # Generate small test arrays of different sizes
                    for size in self.config.warmup_sizes:
                        test_array = jnp.ones((size,), dtype=jnp.float32)
                        test_args.append((test_array,))
                
            # Run warmup iterations
            for _ in range(self.config.warmup_iterations):
                for args in test_args[:3]:  # Limit to first 3 to avoid excessive warmup
                    try:
                        compiled_func(*args)
                        break  # Successful warmup
                    except Exception:
                        continue  # Try next set of args
                        
        except Exception as e:
            self.logger.debug(f"Synthetic warmup generation failed: {e}")
    
    def _create_lazy_compiled_function(
        self,
        func: Callable,
        static_argnums: Optional[Tuple[int, ...]] = None,
        static_argnames: Optional[Tuple[str, ...]] = None,
        donate_argnums: Optional[Tuple[int, ...]] = None
    ) -> Callable:
        """Create a lazily compiled function wrapper."""
        compiled_func = None
        
        @functools.wraps(func)
        def lazy_wrapper(*args, **kwargs):
            nonlocal compiled_func
            if compiled_func is None:
                # Compile on first use
                compiled_func = self.compile(
                    func, static_argnums, static_argnames, donate_argnums,
                    warmup_inputs=[(args[:len(args)//2] if args else (),)]  # Use actual args for warmup
                )
            return compiled_func(*args, **kwargs)
        
        return lazy_wrapper
    
    def _normalize_to_tuple(self, value: Optional[Union[int, str, Tuple]]) -> Optional[Tuple]:
        """Normalize value to tuple format."""
        if value is None:
            return None
        elif isinstance(value, (int, str)):
            return (value,)
        elif isinstance(value, (list, tuple)):
            return tuple(value)
        else:
            return (value,)
    
    def _estimate_memory_usage(self, compiled_func: Callable) -> float:
        """Estimate memory usage of compiled function."""
        # This is a rough estimate - in practice would need more sophisticated analysis
        return 1.0  # MB
    
    def _get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current JAX backend."""
        return {
            'backend': jax.default_backend(),
            'devices': [str(device) for device in jax.devices()],
            'device_count': len(jax.devices()),
            'platform': jax.lib.xla_bridge.get_backend().platform,
        }


def create_jit_compiler(
    strategy: CompilationStrategy = CompilationStrategy.ADAPTIVE,
    optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE,
    **config_kwargs
) -> JITCompiler:
    """
    Create a JIT compiler with specified configuration.
    
    Args:
        strategy: Compilation strategy
        optimization_level: Level of optimization to apply
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured JITCompiler instance
    """
    config = CompilationConfig(
        strategy=strategy,
        optimization_level=optimization_level,
        **config_kwargs
    )
    return JITCompiler(config)


# Global default compiler instance
_default_compiler = None

def get_default_compiler() -> JITCompiler:
    """Get the default global JIT compiler instance."""
    global _default_compiler
    if _default_compiler is None:
        _default_compiler = JITCompiler()
    return _default_compiler


def jit_optimize(func: Callable) -> Callable:
    """
    Convenience decorator using the default compiler.
    
    Args:
        func: Function to JIT compile
        
    Returns:
        JIT-compiled function
    """
    compiler = get_default_compiler()
    return compiler.compile(func)