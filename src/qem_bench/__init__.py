"""
QEM-Bench: Comprehensive Quantum Error Mitigation Benchmarking Suite

A JAX-accelerated framework for implementing, evaluating, and comparing 
quantum error mitigation techniques on noisy quantum hardware.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"
__license__ = "MIT"

# Core mitigation techniques
from .mitigation.zne import ZeroNoiseExtrapolation
from .mitigation.pec import ProbabilisticErrorCancellation  
from .mitigation.vd import VirtualDistillation
from .mitigation.cdr import CliffordDataRegression

# Benchmarking tools
from .benchmarks.circuits import create_benchmark_circuit
from .benchmarks.metrics import compute_fidelity, compute_tvd

# Noise modeling
from .noise.models import NoiseModel
from .noise.characterization import NoiseProfiler

# Monitoring and health checking
from .monitoring import (
    SystemMonitor, PerformanceMonitor, QuantumResourceMonitor, AlertManager
)
from .health import HealthChecker, HealthStatus
from .metrics import MetricsCollector, CircuitMetrics, NoiseMetrics

# Enhanced mitigation with monitoring
from .monitoring.monitored_zne import MonitoredZeroNoiseExtrapolation
from .monitoring.dashboard import MonitoringDashboard

# Scaling and distributed computing framework
from .scaling import (
    AutoScaler, ScalingPolicy, WorkloadAnalyzer, ResourceScheduler, 
    CostOptimizer, DistributedExecutor, BackendBalancer, ResourceOptimizer,
    BackendOrchestrator, SpotInstanceManager
)
from .scaling.scaling_aware_zne import ScalingAwareZeroNoiseExtrapolation, ScalingAwareZNEConfig

# Security framework
from .security import (
    SecureConfig, CredentialManager, InputSanitizer, ResourceLimiter,
    AccessControl, AuditLogger, SecurityPolicy, get_default_policy
)

# Performance optimization framework
from .optimization import (
    PerformanceOptimizer, OptimizationConfig, CacheManager, CacheConfig,
    JITCompiler, CompilationConfig, ParallelExecutor, ExecutionStrategy,
    MemoryManager, PerformanceProfiler, AutoScaler, LoadBalancer,
    create_performance_optimizer, create_cache_manager, create_jit_compiler,
    create_parallel_executor, create_memory_manager, create_profiler
)

# Performance annotations
from .optimization.annotations import (
    performance_profile, performance_monitor, quantum_operation,
    expensive_operation, circuit_compilation, state_vector_operation,
    mitigation_method, get_performance_registry
)

# Optimized mitigation methods
from .optimization.optimized_mitigation import create_optimized_mitigation

# Benchmarking suite
from .optimization.benchmarks import QuantumBenchmarks, create_benchmark_suite

__all__ = [
    "__version__",
    # Core mitigation methods
    "ZeroNoiseExtrapolation",
    "ProbabilisticErrorCancellation", 
    "VirtualDistillation",
    "CliffordDataRegression",
    # Benchmarking
    "create_benchmark_circuit",
    "compute_fidelity",
    "compute_tvd", 
    # Noise modeling
    "NoiseModel",
    "NoiseProfiler",
    # Monitoring framework
    "SystemMonitor",
    "PerformanceMonitor", 
    "QuantumResourceMonitor",
    "AlertManager",
    # Health checking
    "HealthChecker",
    "HealthStatus",
    # Metrics collection
    "MetricsCollector",
    "CircuitMetrics",
    "NoiseMetrics",
    # Enhanced components
    "MonitoredZeroNoiseExtrapolation",
    "MonitoringDashboard",
    # Scaling and distributed computing
    "AutoScaler",
    "ScalingPolicy", 
    "WorkloadAnalyzer",
    "ResourceScheduler",
    "CostOptimizer",
    "DistributedExecutor",
    "BackendBalancer", 
    "ResourceOptimizer",
    "BackendOrchestrator",
    "SpotInstanceManager",
    "ScalingAwareZeroNoiseExtrapolation",
    "ScalingAwareZNEConfig",
    # Security framework
    "SecureConfig",
    "CredentialManager",
    "InputSanitizer",
    "ResourceLimiter",
    "AccessControl",
    "AuditLogger",
    "SecurityPolicy",
    "get_default_policy",
    # Performance optimization framework
    "PerformanceOptimizer",
    "OptimizationConfig",
    "CacheManager",
    "CacheConfig",
    "JITCompiler",
    "CompilationConfig", 
    "ParallelExecutor",
    "ExecutionStrategy",
    "MemoryManager",
    "PerformanceProfiler",
    "AutoScaler",
    "LoadBalancer",
    "create_performance_optimizer",
    "create_cache_manager",
    "create_jit_compiler",
    "create_parallel_executor",
    "create_memory_manager",
    "create_profiler",
    # Performance annotations
    "performance_profile",
    "performance_monitor",
    "quantum_operation",
    "expensive_operation", 
    "circuit_compilation",
    "state_vector_operation",
    "mitigation_method",
    "get_performance_registry",
    # Optimized mitigation
    "create_optimized_mitigation",
    # Benchmarking suite
    "QuantumBenchmarks",
    "create_benchmark_suite",
]