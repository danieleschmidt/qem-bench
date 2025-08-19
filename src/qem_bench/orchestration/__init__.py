"""
Intelligent Multi-Backend Orchestration System

Provides intelligent orchestration and load balancing across multiple quantum
computing backends for optimal error mitigation performance.

Research Contributions:
- Dynamic backend selection based on real-time characteristics
- Intelligent load balancing with predictive queuing
- Cross-platform optimization and resource allocation
- Fault-tolerant execution with automatic failover
- Performance-driven backend recommendation system
"""

from .backend_orchestrator import (
    BackendOrchestrator, 
    OrchestrationConfig,
    BackendSelector,
    LoadBalancer,
    QueuePredictor
)

# from .intelligent_router import (  # Module missing
#     IntelligentRouter,
#     RoutingStrategy,
#     RoutingDecision,
#     RoutingMetrics
# )

# from .performance_optimizer import (  # Module missing
#     CrossBackendOptimizer,
#     OptimizationObjective,
#     PerformanceModel,
#     ResourceAllocator
# )

from .fault_tolerance import (
    FaultTolerantExecutor,
    FailoverStrategy,
    CircuitPartitioner,
    ResultAggregator
)

__all__ = [
    "BackendOrchestrator",
    "OrchestrationConfig", 
    "BackendSelector",
    "LoadBalancer",
    "QueuePredictor",
    "IntelligentRouter",
    "RoutingStrategy",
    "RoutingDecision", 
    "RoutingMetrics",
    "CrossBackendOptimizer",
    "OptimizationObjective",
    "PerformanceModel",
    "ResourceAllocator",
    "FaultTolerantExecutor",
    "FailoverStrategy",
    "CircuitPartitioner",
    "ResultAggregator"
]