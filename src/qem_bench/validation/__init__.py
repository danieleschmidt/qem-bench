"""
Statistical Validation and Testing Framework

Provides comprehensive statistical validation, hypothesis testing, and 
reproducibility frameworks for quantum error mitigation research.

Research Contributions:
- Rigorous statistical hypothesis testing for error mitigation claims
- Multiple testing correction and power analysis
- Bootstrapping and permutation testing for robust inference
- Cross-validation frameworks for machine learning components
- Reproducibility and experiment versioning systems
- Publication-ready statistical reporting
"""

from .statistical_validator import (
    StatisticalValidator,
    HypothesisTest,
    TestResult,
    EffectSize,
    PowerAnalysis
)

from .experiment_framework import (
    ExperimentFramework,
    ExperimentConfig,
    ExperimentResult,
    ReplicationStudy,
    MetaAnalysis
)

from .reproducibility import (
    ReproducibilityManager,
    ExperimentVersion,
    ResultHash,
    EnvironmentCapture
)

from .cross_validation import (
    CrossValidator,
    ValidationStrategy,
    ValidationResult,
    LearningCurveAnalysis
)

from .benchmarking import (
    BenchmarkSuite,
    BenchmarkResult,
    PerformanceComparison,
    StatisticalSignificance
)

__all__ = [
    "StatisticalValidator",
    "HypothesisTest",
    "TestResult", 
    "EffectSize",
    "PowerAnalysis",
    "ExperimentFramework",
    "ExperimentConfig",
    "ExperimentResult",
    "ReplicationStudy",
    "MetaAnalysis",
    "ReproducibilityManager",
    "ExperimentVersion",
    "ResultHash",
    "EnvironmentCapture",
    "CrossValidator", 
    "ValidationStrategy",
    "ValidationResult",
    "LearningCurveAnalysis",
    "BenchmarkSuite",
    "BenchmarkResult",
    "PerformanceComparison",
    "StatisticalSignificance"
]