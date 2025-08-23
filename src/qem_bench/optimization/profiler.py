"""
Performance profiling and analysis tools for quantum error mitigation.

This module provides comprehensive profiling capabilities including timing,
memory usage, bottleneck identification, and performance regression testing.
"""

import time
import threading
import functools
import cProfile
import pstats
import io
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, ContextManager
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import weakref
import psutil
import gc

import numpy as np
import jax
import jax.numpy as jnp

from ..logging import get_logger


class ProfilingMode(Enum):
    """Profiling modes with different levels of detail."""
    BASIC = "basic"        # Basic timing and memory
    DETAILED = "detailed"  # Detailed function profiling
    ADVANCED = "advanced"  # Advanced with call graphs
    CONTINUOUS = "continuous"  # Continuous background profiling


class MetricType(Enum):
    """Types of performance metrics."""
    TIME = "time"
    MEMORY = "memory"
    CALLS = "calls"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    CUSTOM = "custom"


@dataclass
class ProfilingResult:
    """Result from performance profiling."""
    
    function_name: str
    total_time: float
    call_count: int
    average_time: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_percent: float
    
    # Detailed metrics
    call_hierarchy: Dict[str, Any] = field(default_factory=dict)
    memory_timeline: List[Tuple[float, float]] = field(default_factory=list)
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def throughput(self) -> float:
        """Calculate throughput (calls per second)."""
        return self.call_count / self.total_time if self.total_time > 0 else 0
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (0-1, higher is better)."""
        # Simple heuristic combining time and memory efficiency
        time_efficiency = min(1.0, 1.0 / max(0.001, self.average_time))
        memory_efficiency = min(1.0, 100.0 / max(1.0, self.memory_peak_mb))
        return (time_efficiency + memory_efficiency) / 2


@dataclass
class BenchmarkResult:
    """Result from performance benchmarking."""
    
    benchmark_name: str
    baseline_time: float
    current_time: float
    speedup: float
    regression: bool
    
    baseline_memory: float
    current_memory: float
    memory_improvement: float
    
    confidence_interval: Tuple[float, float]
    statistical_significance: float
    
    details: Dict[str, Any] = field(default_factory=dict)


class PerformanceTimer:
    """High-precision timer for performance measurements."""
    
    def __init__(self, name: str = "timer"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.measurements: List[float] = []
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def start(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self):
        """Stop timing and record measurement."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        self.measurements.append(self.elapsed_time)
        return self.elapsed_time
    
    def reset(self):
        """Reset timer."""
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.measurements.clear()
    
    def get_stats(self) -> Dict[str, float]:
        """Get timing statistics."""
        if not self.measurements:
            return {}
        
        measurements = np.array(self.measurements)
        return {
            'count': len(measurements),
            'total': np.sum(measurements),
            'mean': np.mean(measurements),
            'median': np.median(measurements),
            'std': np.std(measurements),
            'min': np.min(measurements),
            'max': np.max(measurements),
            'p95': np.percentile(measurements, 95),
            'p99': np.percentile(measurements, 99),
        }


class MemoryProfiler:
    """Memory usage profiler with timeline tracking."""
    
    def __init__(self, name: str = "memory"):
        self.name = name
        self.process = psutil.Process()
        self.start_memory = None
        self.peak_memory = 0
        self.timeline: List[Tuple[float, float]] = []
        self._monitoring = False
        self._monitor_thread = None
        
        # Start tracemalloc if not already started
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def start(self):
        """Start memory profiling."""
        self.start_memory = self.process.memory_info().rss / (1024**2)  # MB
        self.peak_memory = self.start_memory
        self.timeline.clear()
        
        # Start background monitoring
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self._monitor_thread.start()
        
        return self
    
    def stop(self):
        """Stop memory profiling."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        return self.get_stats()
    
    def _monitor_memory(self):
        """Background memory monitoring."""
        start_time = time.time()
        
        while self._monitoring:
            try:
                current_memory = self.process.memory_info().rss / (1024**2)
                elapsed = time.time() - start_time
                
                self.timeline.append((elapsed, current_memory))
                self.peak_memory = max(self.peak_memory, current_memory)
                
                time.sleep(0.1)  # Sample every 100ms
                
            except Exception:
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory profiling statistics."""
        if self.start_memory is None:
            return {}
        
        current_memory = self.process.memory_info().rss / (1024**2)
        delta_memory = current_memory - self.start_memory
        
        # Get tracemalloc snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        return {
            'start_memory_mb': self.start_memory,
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.peak_memory,
            'delta_memory_mb': delta_memory,
            'timeline': self.timeline.copy(),
            'top_allocations': [
                {
                    'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                    'size_mb': stat.size / (1024**2),
                    'count': stat.count
                }
                for stat in top_stats[:10]
            ]
        }


class FunctionProfiler:
    """Comprehensive function profiler using cProfile."""
    
    def __init__(self, name: str = "function"):
        self.name = name
        self.profiler = cProfile.Profile()
        self.stats = None
        self.is_profiling = False
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def start(self):
        """Start function profiling."""
        self.profiler.enable()
        self.is_profiling = True
        return self
    
    def stop(self):
        """Stop function profiling."""
        if self.is_profiling:
            self.profiler.disable()
            self.is_profiling = False
        
        # Generate stats
        stats_stream = io.StringIO()
        self.stats = pstats.Stats(self.profiler, stream=stats_stream)
        self.stats.sort_stats('cumulative')
        
        return self.get_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get function profiling statistics."""
        if self.stats is None:
            return {}
        
        # Extract key statistics
        stats_dict = {}
        
        # Get total stats
        total_calls = self.stats.total_calls
        total_time = sum(func_stats[2] for func_stats in self.stats.stats.values())
        
        # Get top functions by cumulative time
        top_functions = []
        for func, (cc, nc, tt, ct, callers) in list(self.stats.stats.items())[:20]:
            top_functions.append({
                'function': f"{func[0]}:{func[1]}({func[2]})",
                'calls': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'time_per_call': tt / nc if nc > 0 else 0,
            })
        
        return {
            'total_calls': total_calls,
            'total_time': total_time,
            'top_functions': top_functions,
            'call_hierarchy': self._build_call_hierarchy(),
        }
    
    def _build_call_hierarchy(self) -> Dict[str, Any]:
        """Build call hierarchy from profiling stats."""
        if self.stats is None:
            return {}
        
        hierarchy = {}
        for func, (cc, nc, tt, ct, callers) in self.stats.stats.items():
            func_name = f"{func[0]}:{func[1]}({func[2]})"
            hierarchy[func_name] = {
                'calls': nc,
                'total_time': tt,
                'callers': [
                    f"{caller[0]}:{caller[1]}({caller[2]})"
                    for caller in callers.keys()
                ]
            }
        
        return hierarchy


class PerformanceProfiler:
    """
    Comprehensive performance profiler for quantum error mitigation.
    
    This class provides detailed profiling of quantum computations including
    timing analysis, memory usage tracking, bottleneck identification,
    and performance regression testing.
    
    Features:
    - Multi-level profiling (basic, detailed, advanced)
    - Function-level timing and memory profiling
    - Call hierarchy analysis
    - Bottleneck identification and recommendations
    - Performance regression testing
    - Continuous background profiling
    - Custom metrics and benchmarks
    - Statistical analysis and confidence intervals
    
    Example:
        >>> profiler = PerformanceProfiler()
        >>> 
        >>> @profiler.profile
        >>> def quantum_computation(circuit, backend):
        ...     return backend.run(circuit).result()
        >>> 
        >>> result = quantum_computation(my_circuit, my_backend)
        >>> report = profiler.get_report()
        >>> print(f"Execution time: {report.total_time:.3f}s")
    """
    
    def __init__(
        self,
        mode: ProfilingMode = ProfilingMode.BASIC,
        enable: bool = True,
        interval: float = 1.0
    ):
        """
        Initialize performance profiler.
        
        Args:
            mode: Profiling mode (basic, detailed, advanced, continuous)
            enable: Enable profiling (can be disabled for production)
            interval: Sampling interval for continuous profiling
        """
        self.mode = mode
        self.enable = enable
        self.interval = interval
        self.logger = get_logger()
        
        # Profiling state
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        self.completed_profiles: List[ProfilingResult] = []
        self.benchmarks: Dict[str, List[BenchmarkResult]] = defaultdict(list)
        
        # Background profiling
        self._continuous_profiling = False
        self._continuous_thread = None
        self._background_stats = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance baselines
        self.baselines: Dict[str, Dict[str, float]] = {}
        
        if self.mode == ProfilingMode.CONTINUOUS and self.enable:
            self._start_continuous_profiling()
        
        self.logger.info(f"PerformanceProfiler initialized in {mode.value} mode")
    
    def profile(
        self,
        name: Optional[str] = None,
        track_memory: bool = True,
        track_calls: bool = None
    ) -> Callable:
        """
        Decorator for profiling functions.
        
        Args:
            name: Optional name for the profile (uses function name if None)
            track_memory: Track memory usage
            track_calls: Track function calls (None for auto-detect based on mode)
            
        Returns:
            Decorated function with profiling
        """
        if track_calls is None:
            track_calls = self.mode in [ProfilingMode.DETAILED, ProfilingMode.ADVANCED]
        
        def decorator(func: Callable) -> Callable:
            profile_name = name or func.__name__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable:
                    return func(*args, **kwargs)
                
                return self._profile_function(
                    func, profile_name, track_memory, track_calls, args, kwargs
                )
            
            return wrapper
        return decorator
    
    def profile_context(
        self,
        name: str,
        track_memory: bool = True,
        track_calls: bool = None
    ) -> ContextManager:
        """
        Context manager for profiling code blocks.
        
        Args:
            name: Name for the profile
            track_memory: Track memory usage
            track_calls: Track function calls
            
        Returns:
            Context manager for profiling
        """
        return ProfileContext(self, name, track_memory, track_calls)
    
    def benchmark(
        self,
        name: str,
        func: Callable,
        args: Tuple = (),
        kwargs: Optional[Dict] = None,
        iterations: int = 10,
        warmup_iterations: int = 3
    ) -> BenchmarkResult:
        """
        Benchmark a function with statistical analysis.
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            args: Function arguments
            kwargs: Function keyword arguments
            iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Benchmark result with statistical analysis
        """
        kwargs = kwargs or {}
        
        # Warmup
        for _ in range(warmup_iterations):
            try:
                func(*args, **kwargs)
            except:
                pass  # Ignore warmup errors
        
        # Collect measurements
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            with PerformanceTimer() as timer, MemoryProfiler() as mem_profiler:
                try:
                    result = func(*args, **kwargs)
                    times.append(timer.elapsed_time)
                    mem_stats = mem_profiler.get_stats()
                    memory_usage.append(mem_stats.get('peak_memory_mb', 0))
                except Exception as e:
                    self.logger.warning(f"Benchmark iteration failed: {e}")
                    continue
        
        if not times:
            raise RuntimeError("All benchmark iterations failed")
        
        # Statistical analysis
        times = np.array(times)
        memory_usage = np.array(memory_usage)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        # Confidence interval (95%)
        confidence_interval = (
            mean_time - 1.96 * std_time / np.sqrt(len(times)),
            mean_time + 1.96 * std_time / np.sqrt(len(times))
        )
        
        # Compare with baseline if available
        baseline_time = self.baselines.get(name, {}).get('time', mean_time)
        baseline_memory = self.baselines.get(name, {}).get('memory', np.mean(memory_usage))
        
        speedup = baseline_time / mean_time if mean_time > 0 else 1.0
        memory_improvement = (baseline_memory - np.mean(memory_usage)) / baseline_memory if baseline_memory > 0 else 0.0
        
        # Check for regression (>5% slower)
        regression = speedup < 0.95
        
        # Statistical significance (t-test assumption)
        statistical_significance = abs(mean_time - baseline_time) / (std_time / np.sqrt(len(times))) if std_time > 0 else 0
        
        benchmark_result = BenchmarkResult(
            benchmark_name=name,
            baseline_time=baseline_time,
            current_time=mean_time,
            speedup=speedup,
            regression=regression,
            baseline_memory=baseline_memory,
            current_memory=np.mean(memory_usage),
            memory_improvement=memory_improvement,
            confidence_interval=confidence_interval,
            statistical_significance=statistical_significance,
            details={
                'iterations': iterations,
                'times': times.tolist(),
                'memory_usage': memory_usage.tolist(),
                'std_time': std_time,
                'cv': std_time / mean_time if mean_time > 0 else 0,  # Coefficient of variation
            }
        )
        
        # Store benchmark result
        self.benchmarks[name].append(benchmark_result)
        
        # Update baseline if this is the first run
        if name not in self.baselines:
            self.baselines[name] = {
                'time': mean_time,
                'memory': np.mean(memory_usage)
            }
        
        return benchmark_result
    
    def set_baseline(self, name: str, time: float, memory: float) -> None:
        """
        Set performance baseline for comparison.
        
        Args:
            name: Benchmark name
            time: Baseline execution time
            memory: Baseline memory usage
        """
        self.baselines[name] = {'time': time, 'memory': memory}
    
    def get_profile_results(self, name: Optional[str] = None) -> List[ProfilingResult]:
        """
        Get profiling results.
        
        Args:
            name: Optional name filter
            
        Returns:
            List of profiling results
        """
        if name is None:
            return self.completed_profiles.copy()
        else:
            return [r for r in self.completed_profiles if r.function_name == name]
    
    def get_benchmark_results(self, name: Optional[str] = None) -> Dict[str, List[BenchmarkResult]]:
        """
        Get benchmark results.
        
        Args:
            name: Optional name filter
            
        Returns:
            Dictionary of benchmark results
        """
        if name is None:
            return dict(self.benchmarks)
        else:
            return {name: self.benchmarks.get(name, [])}
    
    def analyze_bottlenecks(self, threshold_percent: float = 5.0) -> List[Dict[str, Any]]:
        """
        Analyze performance bottlenecks across all profiles.
        
        Args:
            threshold_percent: Minimum percentage of total time to be considered a bottleneck
            
        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []
        
        # Aggregate function timing data
        function_times = defaultdict(list)
        total_time = 0
        
        for profile in self.completed_profiles:
            function_times[profile.function_name].append(profile.total_time)
            total_time += profile.total_time
        
        # Identify bottlenecks
        for func_name, times in function_times.items():
            total_func_time = sum(times)
            percentage = (total_func_time / total_time) * 100 if total_time > 0 else 0
            
            if percentage >= threshold_percent:
                bottlenecks.append({
                    'function': func_name,
                    'total_time': total_func_time,
                    'percentage': percentage,
                    'call_count': len(times),
                    'average_time': np.mean(times),
                    'recommendations': self._generate_bottleneck_recommendations(func_name, times)
                })
        
        # Sort by percentage descending
        bottlenecks.sort(key=lambda x: x['percentage'], reverse=True)
        
        return bottlenecks
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary with performance analysis
        """
        # Profile statistics
        profile_stats = {
            'total_profiles': len(self.completed_profiles),
            'total_time': sum(p.total_time for p in self.completed_profiles),
            'average_time': np.mean([p.total_time for p in self.completed_profiles]) if self.completed_profiles else 0,
            'total_calls': sum(p.call_count for p in self.completed_profiles),
        }
        
        # Memory statistics
        if self.completed_profiles:
            memory_peaks = [p.memory_peak_mb for p in self.completed_profiles]
            memory_stats = {
                'peak_memory_mb': max(memory_peaks),
                'average_memory_mb': np.mean(memory_peaks),
                'total_memory_allocated_mb': sum(p.memory_delta_mb for p in self.completed_profiles),
            }
        else:
            memory_stats = {}
        
        # Benchmark statistics
        benchmark_stats = {}
        for name, results in self.benchmarks.items():
            if results:
                latest = results[-1]
                benchmark_stats[name] = {
                    'latest_speedup': latest.speedup,
                    'regression': latest.regression,
                    'runs': len(results),
                    'trend': self._calculate_performance_trend(results),
                }
        
        return {
            'profile_statistics': profile_stats,
            'memory_statistics': memory_stats,
            'benchmark_statistics': benchmark_stats,
            'bottlenecks': self.analyze_bottlenecks(),
            'recommendations': self._generate_performance_recommendations(),
            'configuration': {
                'mode': self.mode.value,
                'enabled': self.enable,
                'continuous_profiling': self._continuous_profiling,
            }
        }
    
    def clear_results(self) -> None:
        """Clear all profiling and benchmark results."""
        with self._lock:
            self.completed_profiles.clear()
            self.benchmarks.clear()
            self.baselines.clear()
            self._background_stats.clear()
    
    def _profile_function(
        self,
        func: Callable,
        name: str,
        track_memory: bool,
        track_calls: bool,
        args: Tuple,
        kwargs: Dict
    ) -> Any:
        """Internal function profiling implementation."""
        # Initialize profilers
        timer = PerformanceTimer(name)
        memory_profiler = MemoryProfiler(name) if track_memory else None
        function_profiler = FunctionProfiler(name) if track_calls else None
        
        # Start profiling
        timer.start()
        if memory_profiler:
            memory_profiler.start()
        if function_profiler:
            function_profiler.start()
        
        # Monitor CPU usage
        process = psutil.Process()
        cpu_start = process.cpu_percent()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Stop profiling
            elapsed_time = timer.stop()
            memory_stats = memory_profiler.stop() if memory_profiler else {}
            function_stats = function_profiler.stop() if function_profiler else {}
            
            cpu_end = process.cpu_percent()
            cpu_percent = (cpu_end + cpu_start) / 2  # Average CPU usage
            
            # Create profiling result
            profile_result = ProfilingResult(
                function_name=name,
                total_time=elapsed_time,
                call_count=1,
                average_time=elapsed_time,
                memory_peak_mb=memory_stats.get('peak_memory_mb', 0),
                memory_delta_mb=memory_stats.get('delta_memory_mb', 0),
                cpu_percent=cpu_percent,
                call_hierarchy=function_stats.get('call_hierarchy', {}),
                memory_timeline=memory_stats.get('timeline', []),
                custom_metrics={}
            )
            
            # Store result
            with self._lock:
                self.completed_profiles.append(profile_result)
                
                # Maintain result history size
                if len(self.completed_profiles) > 1000:
                    self.completed_profiles.pop(0)
            
            return result
            
        except Exception as e:
            # Stop profilers even on exception
            timer.stop()
            if memory_profiler:
                memory_profiler.stop()
            if function_profiler:
                function_profiler.stop()
            
            self.logger.error(f"Profiled function {name} failed: {e}")
            raise
    
    def _start_continuous_profiling(self) -> None:
        """Start continuous background profiling."""
        def continuous_loop():
            process = psutil.Process()
            
            while self._continuous_profiling:
                try:
                    # Collect system metrics
                    cpu_percent = process.cpu_percent()
                    memory_info = process.memory_info()
                    
                    stats = {
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_rss_mb': memory_info.rss / (1024**2),
                        'memory_vms_mb': memory_info.vms / (1024**2),
                    }
                    
                    self._background_stats.append(stats)
                    
                    time.sleep(self.interval)
                    
                except Exception as e:
                    self.logger.error(f"Continuous profiling error: {e}")
                    time.sleep(self.interval)
        
        self._continuous_profiling = True
        self._continuous_thread = threading.Thread(target=continuous_loop, daemon=True)
        self._continuous_thread.start()
    
    def _generate_bottleneck_recommendations(self, func_name: str, times: List[float]) -> List[str]:
        """Generate recommendations for addressing bottlenecks."""
        recommendations = []
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        cv = std_time / avg_time if avg_time > 0 else 0
        
        if avg_time > 1.0:
            recommendations.append("Function takes >1s - consider optimization or caching")
        
        if cv > 0.5:
            recommendations.append("High time variance - investigate inconsistent performance")
        
        if len(times) > 100:
            recommendations.append("Frequently called function - prime candidate for JIT compilation")
        
        # JAX-specific recommendations
        if 'jax' in func_name.lower() or 'jnp' in func_name.lower():
            recommendations.append("JAX function detected - ensure JIT compilation is enabled")
        
        return recommendations or ["Consider general optimization techniques"]
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate general performance recommendations."""
        recommendations = []
        
        if not self.completed_profiles:
            return ["No profiling data available"]
        
        # Analyze overall patterns
        total_time = sum(p.total_time for p in self.completed_profiles)
        avg_memory = np.mean([p.memory_peak_mb for p in self.completed_profiles])
        
        if total_time > 10.0:
            recommendations.append("Total execution time is high - consider parallel processing")
        
        if avg_memory > 1000:  # 1GB
            recommendations.append("High memory usage detected - consider memory optimization")
        
        # Check for frequent small functions
        small_functions = [p for p in self.completed_profiles if p.total_time < 0.01]
        if len(small_functions) > len(self.completed_profiles) * 0.5:
            recommendations.append("Many small functions detected - consider function inlining or batching")
        
        # JAX-specific recommendations
        jax_functions = [p for p in self.completed_profiles if 'jax' in p.function_name.lower()]
        if jax_functions and not any('jit' in p.function_name.lower() for p in jax_functions):
            recommendations.append("JAX functions without JIT detected - enable JIT compilation")
        
        return recommendations or ["Performance appears optimal"]
    
    def _calculate_performance_trend(self, results: List[BenchmarkResult]) -> str:
        """Calculate performance trend from benchmark results."""
        if len(results) < 3:
            return "insufficient_data"
        
        recent_speedups = [r.speedup for r in results[-5:]]  # Last 5 results
        
        if len(recent_speedups) >= 3:
            # Simple linear trend
            x = np.arange(len(recent_speedups))
            slope = np.polyfit(x, recent_speedups, 1)[0]
            
            if slope > 0.01:
                return "improving"
            elif slope < -0.01:
                return "degrading"
            else:
                return "stable"
        
        return "stable"


class ProfileContext:
    """Context manager for profiling code blocks."""
    
    def __init__(
        self,
        profiler: PerformanceProfiler,
        name: str,
        track_memory: bool = True,
        track_calls: bool = None
    ):
        self.profiler = profiler
        self.name = name
        self.track_memory = track_memory
        self.track_calls = track_calls or profiler.mode in [ProfilingMode.DETAILED, ProfilingMode.ADVANCED]
        
        self.timer = None
        self.memory_profiler = None
        self.function_profiler = None
    
    def __enter__(self):
        if not self.profiler.enable:
            return self
        
        # Initialize profilers
        self.timer = PerformanceTimer(self.name)
        
        if self.track_memory:
            self.memory_profiler = MemoryProfiler(self.name)
        
        if self.track_calls:
            self.function_profiler = FunctionProfiler(self.name)
        
        # Start profiling
        self.timer.start()
        if self.memory_profiler:
            self.memory_profiler.start()
        if self.function_profiler:
            self.function_profiler.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.profiler.enable:
            return
        
        # Stop profiling
        elapsed_time = self.timer.stop()
        memory_stats = self.memory_profiler.stop() if self.memory_profiler else {}
        function_stats = self.function_profiler.stop() if self.function_profiler else {}
        
        # Create and store result
        result = ProfilingResult(
            function_name=self.name,
            total_time=elapsed_time,
            call_count=1,
            average_time=elapsed_time,
            memory_peak_mb=memory_stats.get('peak_memory_mb', 0),
            memory_delta_mb=memory_stats.get('delta_memory_mb', 0),
            cpu_percent=0,  # Not tracked in context manager
            call_hierarchy=function_stats.get('call_hierarchy', {}),
            memory_timeline=memory_stats.get('timeline', []),
        )
        
        with self.profiler._lock:
            self.profiler.completed_profiles.append(result)


def create_profiler(
    mode: ProfilingMode = ProfilingMode.BASIC,
    enable: bool = True,
    **kwargs
) -> PerformanceProfiler:
    """
    Create a performance profiler with specified configuration.
    
    Args:
        mode: Profiling mode
        enable: Enable profiling
        **kwargs: Additional configuration
        
    Returns:
        Configured PerformanceProfiler instance
    """
    return PerformanceProfiler(mode=mode, enable=enable, **kwargs)


# Global profiler instance for convenience
_default_profiler = None

def get_default_profiler() -> PerformanceProfiler:
    """Get the default global profiler instance."""
    global _default_profiler
    if _default_profiler is None:
        _default_profiler = PerformanceProfiler()
    return _default_profiler


def profile_function(func: Callable, name: Optional[str] = None) -> Callable:
    """
    Convenience decorator using the default profiler.
    
    Args:
        func: Function to profile
        name: Optional profile name
        
    Returns:
        Decorated function with profiling
    """
    profiler = get_default_profiler()
    return profiler.profile(name=name)(func)