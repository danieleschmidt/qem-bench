"""Probabilistic Error Cancellation (PEC) implementation."""

from .core import (
    ProbabilisticErrorCancellation,
    DecompositionStrategy,
    PauliDecompositionStrategy,
    OptimalDecompositionStrategy,
    PECConfig,
    probabilistic_error_cancellation
)
from .result import PECResult, PECBatchResult

__all__ = [
    "ProbabilisticErrorCancellation",
    "DecompositionStrategy",
    "PauliDecompositionStrategy", 
    "OptimalDecompositionStrategy",
    "PECConfig",
    "PECResult",
    "PECBatchResult",
    "probabilistic_error_cancellation"
]