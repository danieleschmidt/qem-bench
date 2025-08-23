"""
Comprehensive performance testing and benchmarking suite.

This module provides automated performance testing, regression detection,
and benchmarking capabilities for quantum error mitigation methods.
"""

import time
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
import gc

import numpy as np
import jax
import jax.numpy as jnp

from .performance_optimizer import PerformanceOptimizer
from .profiler import PerformanceProfiler, ProfilingMode
from ..logging import get_logger


class BenchmarkType(Enum):
    """Types of benchmarks."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    MEMORY = "memory"
    SCALABILITY = "scalability"
    REGRESSION = "regression"
    STRESS = "stress"


class TestStatus(Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    
    # Test parameters
    iterations: int = 10
    warmup_iterations: int = 3
    timeout_seconds: float = 300.0
    
    # Statistical analysis
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    regression_threshold: float = 0.1  # 10% regression
    
    # Resource limits
    max_memory_gb: float = 16.0
    max_cpu_percent: float = 90.0
    
    # Scalability testing
    scale_factors: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    qubit_sizes: List[int] = field(default_factory=lambda: [4, 6, 8, 10, 12])
    shot_counts: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000])
    
    # Environment
    enable_jit: bool = True
    enable_parallel: bool = True
    enable_caching: bool = True


@dataclass
class BenchmarkResult:
    """Result from a single benchmark test."""
    
    test_name: str
    benchmark_type: BenchmarkType
    status: TestStatus
    
    # Performance metrics
    execution_time: float
    throughput: float  # ops/second
    memory_peak_mb: float
    memory_delta_mb: float
    
    # Statistical analysis
    mean_time: float
    std_time: float
    median_time: float
    p95_time: float
    confidence_interval: Tuple[float, float]
    
    # Comparison with baseline
    baseline_time: Optional[float] = None
    speedup: Optional[float] = None
    regression: bool = False
    
    # Test parameters
    test_params: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    error_message: Optional[str] = None
    
    # Raw measurements
    measurements: List[float] = field(default_factory=list)


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    
    total_execution_time: float
    results: List[BenchmarkResult] = field(default_factory=list)
    
    # Summary statistics
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    regression_detected: bool = False
    
    @property
    def success_rate(self) -> float:
        """Calculate test success rate."""
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0


class QuantumBenchmarks:
    """
    Comprehensive benchmarking suite for quantum error mitigation.
    
    This class provides automated performance testing, regression detection,
    and scalability analysis for quantum error mitigation methods.
    
    Features:
    - Automated performance regression testing
    - Scalability analysis across different problem sizes
    - Memory usage profiling and leak detection
    - Throughput and latency benchmarking
    - Statistical analysis with confidence intervals
    - Baseline comparison and trend analysis
    - Stress testing under resource constraints
    
    Example:
        >>> benchmarks = QuantumBenchmarks()
        >>> 
        >>> # Add mitigation method benchmarks
        >>> benchmarks.add_mitigation_benchmark("ZNE", zne_instance)
        >>> benchmarks.add_mitigation_benchmark("PEC", pec_instance)
        >>> 
        >>> # Run comprehensive benchmark suite
        >>> results = benchmarks.run_full_suite()
        >>> print(f"Overall success rate: {results.success_rate:.2%}")
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize benchmark suite.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self.logger = get_logger()
        
        # Test registry
        self.tests: Dict[str, Callable] = {}
        self.test_configs: Dict[str, Dict[str, Any]] = {}
        self.baselines: Dict[str, float] = {}
        
        # Results storage
        self.current_results: List[BenchmarkResult] = []
        self.historical_results: List[BenchmarkSuite] = []
        
        # Profiler integration
        self.profiler = PerformanceProfiler(mode=ProfilingMode.DETAILED)
        
        self.logger.info("QuantumBenchmarks suite initialized")
    
    def add_test(
        self,
        name: str,
        test_func: Callable,
        benchmark_type: BenchmarkType = BenchmarkType.LATENCY,
        **test_params
    ) -> None:
        """
        Add a benchmark test.
        
        Args:
            name: Test name
            test_func: Function to benchmark
            benchmark_type: Type of benchmark
            **test_params: Additional test parameters
        """
        self.tests[name] = test_func
        self.test_configs[name] = {
            'type': benchmark_type,
            'params': test_params
        }
        
        self.logger.debug(f"Added benchmark test: {name}")
    
    def add_mitigation_benchmark(
        self,
        method_name: str,
        mitigation_instance: Any,
        test_circuits: Optional[List[Any]] = None,
        backend: Optional[Any] = None
    ) -> None:
        """
        Add benchmark for a mitigation method.
        
        Args:
            method_name: Name of mitigation method
            mitigation_instance: Mitigation method instance
            test_circuits: Test circuits to use
            backend: Quantum backend for testing
        """
        if test_circuits is None:
            test_circuits = self._create_test_circuits()
        
        if backend is None:
            from ..jax.simulator import JAXSimulator
            backend = JAXSimulator(num_qubits=8)
        
        # Add latency benchmark
        def latency_test():
            circuit = test_circuits[0]
            return mitigation_instance.mitigate(circuit, backend, shots=1000)
        
        self.add_test(
            f"{method_name}_latency",
            latency_test,
            BenchmarkType.LATENCY,
            method=method_name
        )
        
        # Add throughput benchmark
        def throughput_test():
            results = []
            for circuit in test_circuits[:5]:  # Test on multiple circuits
                result = mitigation_instance.mitigate(circuit, backend, shots=100)
                results.append(result)
            return results
        
        self.add_test(
            f"{method_name}_throughput",
            throughput_test,
            BenchmarkType.THROUGHPUT,
            method=method_name
        )
        
        # Add scalability benchmark
        def scalability_test(num_qubits: int):
            from ..benchmarks.circuits import create_benchmark_circuit
            circuit = create_benchmark_circuit("random", num_qubits=num_qubits, depth=10)
            
            # Adjust backend size if needed
            if hasattr(backend, 'num_qubits') and backend.num_qubits < num_qubits:
                test_backend = JAXSimulator(num_qubits=num_qubits)
            else:
                test_backend = backend
            
            return mitigation_instance.mitigate(circuit, test_backend, shots=500)
        
        self.add_test(
            f"{method_name}_scalability",
            scalability_test,
            BenchmarkType.SCALABILITY,
            method=method_name
        )
    
    def run_single_test(
        self,
        test_name: str,
        iterations: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Run a single benchmark test.
        
        Args:
            test_name: Name of test to run
            iterations: Number of iterations (uses config default if None)
            
        Returns:
            Benchmark result
        """
        if test_name not in self.tests:
            raise ValueError(f"Test '{test_name}' not found")
        
        test_func = self.tests[test_name]
        test_config = self.test_configs[test_name]
        benchmark_type = test_config['type']
        test_params = test_config['params']
        
        iterations = iterations or self.config.iterations
        
        self.logger.info(f"Running benchmark: {test_name}")
        
        # Warmup
        try:
            for _ in range(self.config.warmup_iterations):
                if benchmark_type == BenchmarkType.SCALABILITY:
                    # Use smallest scale for warmup
                    test_func(self.config.qubit_sizes[0])
                else:
                    test_func()
        except Exception as e:
            self.logger.warning(f"Warmup failed for {test_name}: {e}")
        
        # Collect measurements
        measurements = []
        memory_measurements = []
        start_suite_time = time.time()
        
        import psutil
        process = psutil.Process()
        
        for iteration in range(iterations):
            # Memory before
            memory_before = process.memory_info().rss / (1024**2)
            
            # Execute test
            try:
                start_time = time.perf_counter()
                
                if benchmark_type == BenchmarkType.SCALABILITY:
                    # Test different scales
                    scale_times = []
                    for scale in self.config.qubit_sizes:
                        scale_start = time.perf_counter()
                        test_func(scale)
                        scale_time = time.perf_counter() - scale_start
                        scale_times.append(scale_time)
                    
                    execution_time = sum(scale_times)
                else:
                    test_func()
                    execution_time = time.perf_counter() - start_time
                
                measurements.append(execution_time)
                
                # Memory after
                memory_after = process.memory_info().rss / (1024**2)
                memory_measurements.append(memory_after - memory_before)
                
            except Exception as e:
                self.logger.error(f"Test iteration {iteration} failed: {e}")
                return BenchmarkResult(
                    test_name=test_name,
                    benchmark_type=benchmark_type,
                    status=TestStatus.ERROR,
                    execution_time=0.0,
                    throughput=0.0,
                    memory_peak_mb=0.0,
                    memory_delta_mb=0.0,
                    mean_time=0.0,
                    std_time=0.0,
                    median_time=0.0,
                    p95_time=0.0,
                    confidence_interval=(0.0, 0.0),
                    error_message=str(e),
                    test_params=test_params
                )
        
        total_suite_time = time.time() - start_suite_time
        
        # Statistical analysis
        if not measurements:
            return BenchmarkResult(
                test_name=test_name,
                benchmark_type=benchmark_type,
                status=TestStatus.FAILED,
                execution_time=0.0,
                throughput=0.0,
                memory_peak_mb=0.0,
                memory_delta_mb=0.0,
                mean_time=0.0,
                std_time=0.0,
                median_time=0.0,
                p95_time=0.0,
                confidence_interval=(0.0, 0.0),
                error_message="No successful measurements",
                test_params=test_params
            )
        
        mean_time = statistics.mean(measurements)
        std_time = statistics.stdev(measurements) if len(measurements) > 1 else 0.0
        median_time = statistics.median(measurements)
        p95_time = np.percentile(measurements, 95)
        
        # Confidence interval
        if len(measurements) > 1:
            margin_error = 1.96 * std_time / np.sqrt(len(measurements))
            confidence_interval = (mean_time - margin_error, mean_time + margin_error)
        else:
            confidence_interval = (mean_time, mean_time)
        
        # Calculate throughput
        if benchmark_type == BenchmarkType.THROUGHPUT:
            throughput = len(measurements) / total_suite_time
        else:
            throughput = 1.0 / mean_time if mean_time > 0 else 0.0
        
        # Memory analysis
        memory_peak = max(memory_measurements) if memory_measurements else 0.0
        memory_delta = statistics.mean(memory_measurements) if memory_measurements else 0.0
        
        # Regression analysis
        baseline_time = self.baselines.get(test_name)
        speedup = None
        regression = False
        
        if baseline_time:
            speedup = baseline_time / mean_time if mean_time > 0 else 0.0
            regression = speedup < (1.0 - self.config.regression_threshold)
        
        # Determine status
        if regression:
            status = TestStatus.FAILED
        elif mean_time > 0:
            status = TestStatus.PASSED
        else:
            status = TestStatus.FAILED
        
        result = BenchmarkResult(
            test_name=test_name,
            benchmark_type=benchmark_type,
            status=status,
            execution_time=mean_time,
            throughput=throughput,
            memory_peak_mb=memory_peak,
            memory_delta_mb=memory_delta,
            mean_time=mean_time,
            std_time=std_time,
            median_time=median_time,
            p95_time=p95_time,
            confidence_interval=confidence_interval,
            baseline_time=baseline_time,
            speedup=speedup,
            regression=regression,
            test_params=test_params,
            measurements=measurements
        )
        
        self.logger.info(f"Completed benchmark {test_name}: {status.value}")
        return result
    
    def run_full_suite(self) -> BenchmarkSuite:
        """
        Run the complete benchmark suite.
        
        Returns:
            Complete benchmark suite results
        """
        self.logger.info("Starting full benchmark suite")
        start_time = time.time()
        
        results = []
        passed = 0
        failed = 0
        skipped = 0
        
        for test_name in self.tests:
            try:
                result = self.run_single_test(test_name)
                results.append(result)
                
                if result.status == TestStatus.PASSED:
                    passed += 1
                elif result.status == TestStatus.FAILED:
                    failed += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to run test {test_name}: {e}")
                failed += 1
        
        total_time = time.time() - start_time
        
        # Generate performance summary
        performance_summary = self._generate_performance_summary(results)
        
        # Check for regressions
        regression_detected = any(r.regression for r in results)
        
        suite_result = BenchmarkSuite(
            suite_name="QEM Performance Suite",
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=failed,
            skipped_tests=skipped,
            total_execution_time=total_time,
            results=results,
            performance_summary=performance_summary,
            regression_detected=regression_detected
        )
        
        # Store results
        self.historical_results.append(suite_result)
        self.current_results = results
        
        self.logger.info(
            f"Benchmark suite completed: {passed}/{len(results)} tests passed "
            f"in {total_time:.2f}s"
        )
        
        return suite_result
    
    def run_regression_tests(self) -> BenchmarkSuite:
        """
        Run regression tests against baselines.
        
        Returns:
            Regression test results
        """
        self.logger.info("Running regression tests")
        
        # Only run tests that have baselines
        regression_tests = {
            name: func for name, func in self.tests.items()
            if name in self.baselines
        }
        
        if not regression_tests:
            self.logger.warning("No baseline data available for regression testing")
            return BenchmarkSuite(
                suite_name="Regression Tests",
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                total_execution_time=0.0
            )
        
        # Run subset of tests
        original_tests = self.tests.copy()
        self.tests = regression_tests
        
        try:
            results = self.run_full_suite()
            results.suite_name = "Regression Tests"
            return results
        finally:
            self.tests = original_tests
    
    def set_baseline(self, test_name: str, baseline_time: float) -> None:
        """
        Set performance baseline for a test.
        
        Args:
            test_name: Name of test
            baseline_time: Baseline execution time
        """
        self.baselines[test_name] = baseline_time
        self.logger.info(f"Set baseline for {test_name}: {baseline_time:.4f}s")
    
    def update_baselines_from_results(self, results: Optional[List[BenchmarkResult]] = None) -> None:
        """
        Update baselines from current results.
        
        Args:
            results: Results to use for baselines (uses current if None)
        """
        results = results or self.current_results
        
        for result in results:
            if result.status == TestStatus.PASSED:
                self.set_baseline(result.test_name, result.mean_time)
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive benchmark report.
        
        Returns:
            Detailed benchmark report
        """
        if not self.current_results:
            return {"status": "No benchmark results available"}
        
        # Overall statistics
        total_tests = len(self.current_results)
        passed_tests = sum(1 for r in self.current_results if r.status == TestStatus.PASSED)
        
        # Performance statistics
        execution_times = [r.mean_time for r in self.current_results if r.status == TestStatus.PASSED]
        memory_usage = [r.memory_peak_mb for r in self.current_results if r.status == TestStatus.PASSED]
        
        performance_stats = {}
        if execution_times:
            performance_stats = {
                'total_execution_time': sum(execution_times),
                'average_execution_time': statistics.mean(execution_times),
                'fastest_test': min(execution_times),
                'slowest_test': max(execution_times),
            }
        
        memory_stats = {}
        if memory_usage:
            memory_stats = {
                'peak_memory_mb': max(memory_usage),
                'average_memory_mb': statistics.mean(memory_usage),
                'total_memory_allocated_mb': sum(memory_usage),
            }
        
        # Regression analysis
        regressions = [r for r in self.current_results if r.regression]
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            },
            'performance': performance_stats,
            'memory': memory_stats,
            'regressions': {
                'count': len(regressions),
                'tests': [r.test_name for r in regressions],
            },
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'status': r.status.value,
                    'execution_time': r.mean_time,
                    'throughput': r.throughput,
                    'memory_peak_mb': r.memory_peak_mb,
                    'speedup': r.speedup,
                    'regression': r.regression,
                }
                for r in self.current_results
            ]
        }
    
    def _create_test_circuits(self) -> List[Any]:
        """Create test circuits for benchmarking."""
        from ..benchmarks.circuits import create_benchmark_circuit
        
        circuits = []
        
        # Add various types of test circuits
        circuit_types = ["random", "quantum_volume", "algorithmic"]
        
        for circuit_type in circuit_types:
            try:
                circuit = create_benchmark_circuit(
                    circuit_type,
                    num_qubits=6,
                    depth=10
                )
                circuits.append(circuit)
            except Exception as e:
                self.logger.warning(f"Failed to create {circuit_type} circuit: {e}")
        
        return circuits or [None]  # Fallback
    
    def _generate_performance_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate performance summary from results."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.status == TestStatus.PASSED]
        
        if not successful_results:
            return {"status": "No successful results"}
        
        # Aggregate statistics
        execution_times = [r.mean_time for r in successful_results]
        throughputs = [r.throughput for r in successful_results]
        memory_usage = [r.memory_peak_mb for r in successful_results]
        
        return {
            'total_successful_tests': len(successful_results),
            'execution_time_stats': {
                'mean': statistics.mean(execution_times),
                'median': statistics.median(execution_times),
                'std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                'min': min(execution_times),
                'max': max(execution_times),
            },
            'throughput_stats': {
                'mean': statistics.mean(throughputs),
                'median': statistics.median(throughputs),
                'max': max(throughputs),
            },
            'memory_stats': {
                'peak_mb': max(memory_usage),
                'average_mb': statistics.mean(memory_usage),
                'total_mb': sum(memory_usage),
            },
            'performance_trends': self._analyze_performance_trends(),
        }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across historical results."""
        if len(self.historical_results) < 2:
            return {"status": "Insufficient historical data"}
        
        # Compare latest results with previous
        latest = self.historical_results[-1]
        previous = self.historical_results[-2]
        
        # Calculate overall trend
        latest_avg = statistics.mean([
            r.mean_time for r in latest.results 
            if r.status == TestStatus.PASSED
        ]) if latest.results else 0
        
        previous_avg = statistics.mean([
            r.mean_time for r in previous.results 
            if r.status == TestStatus.PASSED
        ]) if previous.results else 0
        
        if previous_avg > 0:
            performance_change = (previous_avg - latest_avg) / previous_avg
            trend = "improving" if performance_change > 0.05 else "stable" if abs(performance_change) <= 0.05 else "degrading"
        else:
            performance_change = 0
            trend = "stable"
        
        return {
            'trend': trend,
            'performance_change': performance_change,
            'latest_avg_time': latest_avg,
            'previous_avg_time': previous_avg,
        }


def create_benchmark_suite(
    include_mitigation_methods: bool = True,
    include_scalability_tests: bool = True,
    **config_kwargs
) -> QuantumBenchmarks:
    """
    Create a comprehensive benchmark suite.
    
    Args:
        include_mitigation_methods: Include mitigation method benchmarks
        include_scalability_tests: Include scalability tests
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured QuantumBenchmarks instance
    """
    config = BenchmarkConfig(**config_kwargs)
    benchmarks = QuantumBenchmarks(config)
    
    if include_mitigation_methods:
        # Add standard mitigation method benchmarks
        try:
            from ..mitigation.zne import ZeroNoiseExtrapolation
            zne = ZeroNoiseExtrapolation()
            benchmarks.add_mitigation_benchmark("ZNE", zne)
        except ImportError:
            pass
    
    # Add basic performance tests
    def cpu_intensive_test():
        """CPU intensive computation test."""
        return jnp.sum(jnp.array(range(100000)) ** 2)
    
    def memory_intensive_test():
        """Memory intensive test."""
        large_array = jnp.ones((1000, 1000))
        return jnp.sum(large_array @ large_array.T)
    
    benchmarks.add_test("cpu_intensive", cpu_intensive_test, BenchmarkType.THROUGHPUT)
    benchmarks.add_test("memory_intensive", memory_intensive_test, BenchmarkType.MEMORY)
    
    return benchmarks