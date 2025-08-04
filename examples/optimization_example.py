#!/usr/bin/env python3
"""
Comprehensive example demonstrating the QEM-Bench performance optimization framework.

This example shows how to use the optimization framework to significantly 
improve the performance of quantum error mitigation computations.
"""

import time
import numpy as np
import jax.numpy as jnp

# Import QEM-Bench optimization framework
from qem_bench.optimization import (
    PerformanceOptimizer, OptimizationConfig, OptimizationStrategy,
    CacheManager, JITCompiler, ParallelExecutor, MemoryManager,
    PerformanceProfiler, ProfilingMode, QuantumBenchmarks,
    create_optimized_mitigation
)

# Import performance annotations
from qem_bench.optimization.annotations import (
    performance_profile, quantum_operation, expensive_operation,
    mitigation_method, get_performance_registry
)

# Import core mitigation methods
from qem_bench.mitigation.zne import ZeroNoiseExtrapolation
from qem_bench.jax.simulator import JAXSimulator
from qem_bench.benchmarks.circuits import create_benchmark_circuit


def main():
    """Main demonstration of optimization framework."""
    print("=== QEM-Bench Performance Optimization Framework Demo ===\n")
    
    # 1. Basic Performance Optimization
    print("1. Setting up Performance Optimizer...")
    optimizer = PerformanceOptimizer(
        OptimizationConfig(
            strategy=OptimizationStrategy.ADAPTIVE,
            enable_jit=True,
            enable_caching=True,
            enable_parallel=True,
            target_speedup=2.0
        )
    )
    print(f"✓ Optimizer initialized with {optimizer.config.strategy.value} strategy\n")
    
    # 2. Demonstrate Function Optimization
    print("2. Function Optimization Demo...")
    
    @expensive_operation(cache_result=True, prefer_parallel=True)
    def expensive_computation(n: int) -> float:
        """Simulate expensive quantum computation."""
        result = 0.0
        for i in range(n):
            result += jnp.sin(i) * jnp.cos(i)
        return float(result)
    
    # Time original function
    start_time = time.time()
    result1 = expensive_computation(100000)
    original_time = time.time() - start_time
    
    # Time optimized function
    start_time = time.time()
    result2, opt_result = optimizer.optimize_function(expensive_computation, 100000)
    optimized_time = time.time() - start_time
    
    print(f"Original execution: {original_time:.4f}s")
    print(f"Optimized execution: {optimized_time:.4f}s")
    print(f"Speedup: {opt_result.speedup:.2f}x")
    print(f"Applied optimizations: {opt_result.applied_optimizations}\n")
    
    # 3. Quantum Circuit Optimization
    print("3. Quantum Circuit Optimization Demo...")
    
    # Create test circuit and backend
    circuit = create_benchmark_circuit("random", num_qubits=8, depth=10)
    backend = JAXSimulator(num_qubits=8)
    
    @quantum_operation(gate_count=80, qubit_count=8, depth=10)
    def simulate_circuit(circuit, backend, shots=1000):
        """Simulate quantum circuit."""
        return backend.run(circuit, shots=shots)
    
    # Optimize circuit execution
    result, opt_result = optimizer.optimize_circuit_execution([circuit], backend)
    print(f"Circuit execution optimized with {opt_result.speedup:.2f}x speedup")
    print(f"Memory usage: {opt_result.memory_usage_mb:.1f} MB\n")
    
    # 4. Mitigation Method Optimization
    print("4. Error Mitigation Optimization Demo...")
    
    # Create ZNE instance
    zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 1.5, 2.0, 2.5, 3.0])
    
    # Create optimized version
    optimized_zne = create_optimized_mitigation(zne, optimizer)
    
    print("Original ZNE:")
    start_time = time.time()
    try:
        result_orig = zne.mitigate(circuit, backend, shots=500)
        orig_time = time.time() - start_time
        print(f"  Execution time: {orig_time:.4f}s")
        print(f"  Mitigated value: {result_orig.mitigated_value:.4f}")
    except Exception as e:
        print(f"  Failed: {e}")
        orig_time = float('inf')
    
    print("Optimized ZNE:")
    start_time = time.time()
    try:
        result_opt = optimized_zne.mitigate(circuit, backend, shots=500)
        opt_time = time.time() - start_time
        print(f"  Execution time: {opt_time:.4f}s")
        print(f"  Mitigated value: {result_opt.mitigated_value:.4f}")
        if orig_time != float('inf'):
            print(f"  Speedup: {orig_time / opt_time:.2f}x")
    except Exception as e:
        print(f"  Failed: {e}")
    
    print()
    
    # 5. Memory Management Demo
    print("5. Memory Management Demo...")
    
    memory_manager = optimizer.memory_manager
    
    # Allocate large state vector
    state_id, state_vector = memory_manager.allocate_state_vector(
        num_qubits=12, initial_state="zero"
    )
    
    print(f"Allocated state vector for 12 qubits")
    print(f"State vector shape: {state_vector.shape}")
    print(f"Memory usage: {memory_manager.get_memory_usage()['pool_stats']['allocated_mb']:.1f} MB")
    
    # Clean up
    memory_manager.deallocate(state_id)
    print(f"Memory cleaned up\n")
    
    # 6. Performance Profiling Demo
    print("6. Performance Profiling Demo...")
    
    profiler = PerformanceProfiler(mode=ProfilingMode.DETAILED, enable=True)
    
    @profiler.profile(name="test_function")
    def test_function(size: int):
        """Test function for profiling."""
        data = jnp.ones((size, size))
        return jnp.sum(data @ data.T)
    
    # Run profiled function
    result = test_function(500)
    
    # Get profiling results
    profile_results = profiler.get_profile_results("test_function")
    if profile_results:
        result = profile_results[0]
        print(f"Function profiled: {result.function_name}")
        print(f"Execution time: {result.total_time:.4f}s")
        print(f"Memory peak: {result.memory_peak_mb:.1f} MB")
        print(f"Efficiency score: {result.efficiency_score:.3f}")
    
    print()
    
    # 7. Caching System Demo
    print("7. Intelligent Caching Demo...")
    
    cache_manager = optimizer.cache_manager
    
    def expensive_function(x: int) -> float:
        """Expensive function to demonstrate caching."""
        time.sleep(0.1)  # Simulate expensive computation
        return float(x ** 2 + 2 * x + 1)
    
    # First call (cache miss)
    start_time = time.time()
    result1 = cache_manager.get_or_compute("test_key", lambda: expensive_function(42))
    time1 = time.time() - start_time
    
    # Second call (cache hit)
    start_time = time.time()
    result2 = cache_manager.get_or_compute("test_key", lambda: expensive_function(42))
    time2 = time.time() - start_time
    
    print(f"First call (cache miss): {time1:.4f}s, result: {result1}")
    print(f"Second call (cache hit): {time2:.4f}s, result: {result2}")
    print(f"Cache speedup: {time1 / time2:.1f}x")
    
    cache_stats = cache_manager.get_usage_stats()
    print(f"Cache hit rate: {cache_stats['overall']['hit_rate']:.2%}\n")
    
    # 8. Parallel Execution Demo
    print("8. Parallel Execution Demo...")
    
    parallel_executor = optimizer.parallel_executor
    
    def parallel_task(task_id: int) -> float:
        """Task for parallel execution."""
        # Simulate computation
        return float(sum(i**2 for i in range(task_id * 1000)))
    
    # Execute tasks in parallel
    tasks = list(range(1, 9))  # 8 tasks
    
    start_time = time.time()
    result = parallel_executor.execute_batch(parallel_task, [(task,) for task in tasks])
    parallel_time = time.time() - start_time
    
    print(f"Executed {len(tasks)} tasks in parallel")
    print(f"Total time: {parallel_time:.4f}s")
    print(f"Strategy used: {result.strategy_used.value}")
    print(f"Workers used: {result.worker_count}")
    
    # Sequential comparison
    start_time = time.time()
    sequential_results = [parallel_task(task) for task in tasks]
    sequential_time = time.time() - start_time
    
    print(f"Sequential time: {sequential_time:.4f}s")
    print(f"Parallel speedup: {sequential_time / parallel_time:.2f}x\n")
    
    # 9. Performance Benchmarking
    print("9. Performance Benchmarking Demo...")
    
    benchmark_suite = QuantumBenchmarks()
    
    # Add some basic benchmarks
    def cpu_test():
        return jnp.sum(jnp.arange(10000) ** 2)
    
    def memory_test():
        array = jnp.ones((100, 100))
        return jnp.sum(array @ array.T)
    
    benchmark_suite.add_test("cpu_intensive", cpu_test)
    benchmark_suite.add_test("memory_intensive", memory_test)
    
    # Run benchmarks
    results = benchmark_suite.run_full_suite()
    
    print(f"Benchmark suite: {results.suite_name}")
    print(f"Tests passed: {results.passed_tests}/{results.total_tests}")
    print(f"Success rate: {results.success_rate:.2%}")
    print(f"Total execution time: {results.total_execution_time:.4f}s")
    
    for test_result in results.results:
        print(f"  {test_result.test_name}: {test_result.status.value} "
              f"({test_result.mean_time:.4f}s)")
    
    print()
    
    # 10. Performance Report
    print("10. Comprehensive Performance Report...")
    
    report = optimizer.get_performance_report()
    
    print("Performance Summary:")
    print(f"  Total optimizations: {report['performance_summary']['total_optimizations']}")
    print(f"  Average speedup: {report['performance_summary']['average_speedup']:.2f}x")
    print(f"  Maximum speedup: {report['performance_summary']['maximum_speedup']:.2f}x")
    print(f"  JIT success rate: {report['optimization_statistics']['jit_compilation_success_rate']:.2%}")
    print(f"  Cache hit rate: {report['optimization_statistics']['cache_hit_rate']:.2%}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    print("\n=== Demo Complete ===")
    print("The optimization framework provides comprehensive performance improvements")
    print("for quantum error mitigation computations through:")
    print("  • Automatic JIT compilation and caching")
    print("  • Intelligent parallel execution")
    print("  • Memory pooling and management")
    print("  • Performance profiling and analysis")
    print("  • Auto-scaling and load balancing")
    print("  • Comprehensive benchmarking and regression testing")


if __name__ == "__main__":
    main()