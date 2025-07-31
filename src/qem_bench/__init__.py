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

__all__ = [
    "__version__",
    "ZeroNoiseExtrapolation",
    "ProbabilisticErrorCancellation", 
    "VirtualDistillation",
    "CliffordDataRegression",
    "create_benchmark_circuit",
    "compute_fidelity",
    "compute_tvd", 
    "NoiseModel",
    "NoiseProfiler",
]