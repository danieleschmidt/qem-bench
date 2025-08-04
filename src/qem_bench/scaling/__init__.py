"""
Auto-scaling and distributed computing framework for QEM-Bench.

This module provides comprehensive auto-scaling capabilities including:
- Dynamic resource management and workload analysis
- Distributed computing with fault tolerance
- Load balancing across quantum backends
- Cloud integration and cost optimization
- Multi-backend orchestration
- Resource optimization algorithms
"""

from .auto_scaler import AutoScaler, ScalingPolicy, ScalingMetrics
from .workload_analyzer import WorkloadAnalyzer, WorkloadPredictor, WorkloadMetrics
from .resource_scheduler import ResourceScheduler, SchedulingPolicy, ResourceAllocation
from .cost_optimizer import CostOptimizer, CostModel, OptimizationStrategy
from .distributed_executor import (
    DistributedExecutor, 
    TaskScheduler, 
    ResultAggregator,
    FaultTolerantExecutor
)
from .load_balancer import (
    BackendBalancer,
    QueueManager, 
    PriorityScheduler,
    CapacityMonitor
)
from .cloud_providers import (
    CloudProvider,
    AWSProvider,
    GoogleCloudProvider,
    AzureProvider,
    SpotInstanceManager
)
from .backend_orchestrator import (
    BackendOrchestrator,
    CalibrationAwareScheduler,
    CrossBackendBenchmarker,
    FallbackStrategy
)
from .resource_optimizer import (
    ResourceOptimizer,
    CircuitBatcher,
    CompilationOptimizer,
    ShotAllocator
)

__all__ = [
    # Auto-scaling core
    "AutoScaler",
    "ScalingPolicy", 
    "ScalingMetrics",
    # Workload analysis
    "WorkloadAnalyzer",
    "WorkloadPredictor",
    "WorkloadMetrics",
    # Resource scheduling
    "ResourceScheduler",
    "SchedulingPolicy",
    "ResourceAllocation",
    # Cost optimization
    "CostOptimizer",
    "CostModel",
    "OptimizationStrategy",
    # Distributed computing
    "DistributedExecutor",
    "TaskScheduler",
    "ResultAggregator", 
    "FaultTolerantExecutor",
    # Load balancing
    "BackendBalancer",
    "QueueManager",
    "PriorityScheduler",
    "CapacityMonitor",
    # Cloud integration
    "CloudProvider",
    "AWSProvider",
    "GoogleCloudProvider", 
    "AzureProvider",
    "SpotInstanceManager",
    # Backend orchestration
    "BackendOrchestrator",
    "CalibrationAwareScheduler",
    "CrossBackendBenchmarker",
    "FallbackStrategy",
    # Resource optimization
    "ResourceOptimizer",
    "CircuitBatcher",
    "CompilationOptimizer",
    "ShotAllocator",
]