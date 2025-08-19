"""Performance metrics for quantum error mitigation benchmarking."""

from .fidelity import (
    compute_fidelity,
    compute_process_fidelity,
    compute_average_gate_fidelity,
    StateFidelityCalculator
)
from .distance import (
    compute_tvd,
    compute_trace_distance,
    compute_diamond_distance,
    compute_hellinger_distance
)
# from .entropy import (
#     compute_von_neumann_entropy,
#     compute_entanglement_entropy,
#     compute_mutual_information
# )
# from .benchmarking import (
#     compute_qem_efficiency,
#     compute_overhead_factor,
#     compute_success_probability,
#     BenchmarkMetrics
# )

__all__ = [
    # Fidelity metrics
    "compute_fidelity",
    "compute_process_fidelity", 
    "compute_average_gate_fidelity",
    "StateFidelityCalculator",
    
    # Distance metrics
    "compute_tvd",
    "compute_trace_distance",
    "compute_diamond_distance",
    "compute_hellinger_distance",
    
    # Entropy metrics (commented out - module missing)
    # "compute_von_neumann_entropy",
    # "compute_entanglement_entropy", 
    # "compute_mutual_information",
    
    # Benchmarking metrics (commented out - module missing)
    # "compute_qem_efficiency",
    # "compute_overhead_factor",
    # "compute_success_probability",
    # "BenchmarkMetrics"
]