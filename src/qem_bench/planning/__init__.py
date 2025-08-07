"""
Quantum-Inspired Task Planning Module

Integrates quantum optimization principles with classical task planning,
leveraging the existing QEM infrastructure for enhanced optimization.
"""

from .core import QuantumInspiredPlanner, PlanningConfig, Task, TaskState
from .optimizer import QuantumTaskOptimizer, OptimizationStrategy
from .scheduler import QuantumScheduler, SchedulingPolicy
from .metrics import PlanningMetrics, TaskComplexity, PlanningAnalyzer, ComplexityMeasure
from .integration import QEMPlannerIntegration, QEMTask
from .validation import PlanningValidator, ValidationLevel, ResilientPlanningWrapper
from .recovery import QuantumPlanningRecovery, RecoveryStrategy, FaultType
from .performance import HighPerformancePlanner, PerformanceConfig, ComputeBackend

__all__ = [
    # Core components
    "QuantumInspiredPlanner",
    "PlanningConfig",
    "Task", 
    "TaskState",
    
    # Optimization
    "QuantumTaskOptimizer",
    "OptimizationStrategy",
    
    # Scheduling
    "QuantumScheduler",
    "SchedulingPolicy",
    
    # Metrics and analysis
    "PlanningMetrics",
    "TaskComplexity",
    "PlanningAnalyzer",
    "ComplexityMeasure",
    
    # QEM integration
    "QEMPlannerIntegration",
    "QEMTask",
    
    # Validation and robustness
    "PlanningValidator",
    "ValidationLevel", 
    "ResilientPlanningWrapper",
    
    # Recovery and fault tolerance
    "QuantumPlanningRecovery",
    "RecoveryStrategy",
    "FaultType",
    
    # High performance
    "HighPerformancePlanner",
    "PerformanceConfig",
    "ComputeBackend"
]