"""Standard benchmark circuit implementations."""

from enum import Enum
from typing import Any, Dict, Optional, Union
import numpy as np

from ...jax.circuits import JAXCircuit


class BenchmarkCircuitType(Enum):
    """Enumeration of available benchmark circuit types."""
    QUANTUM_VOLUME = "quantum_volume"
    RANDOMIZED_BENCHMARKING = "randomized_benchmarking"
    QUANTUM_FOURIER_TRANSFORM = "quantum_fourier_transform"
    VQE_ANSATZ = "vqe_ansatz"
    QAOA = "qaoa"
    BELL_STATE = "bell_state"
    GHZ_STATE = "ghz_state"
    RANDOM_CIRCUIT = "random_circuit"


def create_benchmark_circuit(
    name: Union[str, BenchmarkCircuitType],
    qubits: int,
    depth: Optional[int] = None,
    **kwargs
) -> JAXCircuit:
    """
    Create a standard benchmark circuit.
    
    Args:
        name: Circuit type name or enum
        qubits: Number of qubits
        depth: Circuit depth (if applicable)
        **kwargs: Additional circuit-specific parameters
        
    Returns:
        JAXCircuit implementing the benchmark
        
    Example:
        >>> circuit = create_benchmark_circuit("quantum_volume", qubits=5, depth=10)
        >>> circuit = create_benchmark_circuit(BenchmarkCircuitType.QFT, qubits=4)
    """
    if isinstance(name, str):
        try:
            circuit_type = BenchmarkCircuitType(name)
        except ValueError:
            raise ValueError(f"Unknown benchmark circuit type: {name}")
    else:
        circuit_type = name
    
    if circuit_type == BenchmarkCircuitType.QUANTUM_VOLUME:
        from .quantum_volume import create_quantum_volume_circuit
        if depth is None:
            depth = qubits  # Standard QV depth equals number of qubits
        return create_quantum_volume_circuit(qubits, depth, **kwargs)
    
    elif circuit_type == BenchmarkCircuitType.RANDOMIZED_BENCHMARKING:
        from .randomized_benchmarking import create_rb_circuit
        if depth is None:
            depth = 10  # Default RB sequence length
        return create_rb_circuit(qubits, depth, **kwargs)
    
    elif circuit_type == BenchmarkCircuitType.QUANTUM_FOURIER_TRANSFORM:
        from .algorithmic import create_qft_circuit
        return create_qft_circuit(qubits, **kwargs)
    
    elif circuit_type == BenchmarkCircuitType.VQE_ANSATZ:
        from .algorithmic import create_vqe_ansatz
        if depth is None:
            depth = 2  # Default number of layers
        return create_vqe_ansatz(qubits, depth, **kwargs)
    
    elif circuit_type == BenchmarkCircuitType.QAOA:
        from .algorithmic import create_qaoa_circuit
        if depth is None:
            depth = 1  # Default p=1 QAOA
        return create_qaoa_circuit(qubits, depth, **kwargs)
    
    elif circuit_type == BenchmarkCircuitType.BELL_STATE:
        if qubits != 2:
            raise ValueError("Bell state requires exactly 2 qubits")
        circuit = JAXCircuit(2, name="bell_state")
        circuit.h(0)
        circuit.cx(0, 1)
        return circuit
    
    elif circuit_type == BenchmarkCircuitType.GHZ_STATE:
        circuit = JAXCircuit(qubits, name="ghz_state")
        circuit.h(0)
        for i in range(1, qubits):
            circuit.cx(0, i)
        return circuit
    
    elif circuit_type == BenchmarkCircuitType.RANDOM_CIRCUIT:
        from ...jax.circuits import create_random_circuit
        if depth is None:
            depth = qubits * 2  # Default depth
        return create_random_circuit(qubits, depth, **kwargs)
    
    else:
        raise ValueError(f"Unsupported circuit type: {circuit_type}")


def get_ideal_expectation_value(
    circuit_type: Union[str, BenchmarkCircuitType],
    observable_type: str = "all_z",
    **circuit_params
) -> Optional[float]:
    """
    Get the ideal (noiseless) expectation value for a benchmark circuit.
    
    Args:
        circuit_type: Type of benchmark circuit
        observable_type: Type of observable to measure
        **circuit_params: Parameters used to create the circuit
        
    Returns:
        Ideal expectation value if known, None otherwise
    """
    if isinstance(circuit_type, str):
        circuit_type = BenchmarkCircuitType(circuit_type)
    
    # For specific circuit types, we can calculate ideal values
    if circuit_type == BenchmarkCircuitType.BELL_STATE and observable_type == "all_z":
        return 0.0  # Bell state has equal superposition of |00⟩ and |11⟩
    
    elif circuit_type == BenchmarkCircuitType.GHZ_STATE and observable_type == "all_z":
        return 0.0  # GHZ state has equal superposition of |00...0⟩ and |11...1⟩
    
    # For most other cases, the ideal value depends on specific parameters
    # and would need to be computed numerically
    return None


def validate_benchmark_parameters(
    circuit_type: Union[str, BenchmarkCircuitType],
    qubits: int,
    depth: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Validate and normalize benchmark circuit parameters.
    
    Args:
        circuit_type: Type of benchmark circuit
        qubits: Number of qubits
        depth: Circuit depth
        **kwargs: Additional parameters
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ValueError: If parameters are invalid
    """
    if isinstance(circuit_type, str):
        circuit_type = BenchmarkCircuitType(circuit_type)
    
    if qubits < 1:
        raise ValueError("Number of qubits must be at least 1")
    
    params = {"qubits": qubits, "depth": depth}
    params.update(kwargs)
    
    # Circuit-specific validation
    if circuit_type == BenchmarkCircuitType.BELL_STATE:
        if qubits != 2:
            raise ValueError("Bell state requires exactly 2 qubits")
    
    elif circuit_type == BenchmarkCircuitType.QUANTUM_VOLUME:
        if depth is None:
            params["depth"] = qubits
        elif depth < 1:
            raise ValueError("QV depth must be at least 1")
    
    elif circuit_type == BenchmarkCircuitType.RANDOMIZED_BENCHMARKING:
        if depth is None:
            params["depth"] = 10
        elif depth < 1:
            raise ValueError("RB sequence length must be at least 1")
    
    elif circuit_type == BenchmarkCircuitType.VQE_ANSATZ:
        if depth is None:
            params["depth"] = 2
        elif depth < 1:
            raise ValueError("VQE ansatz layers must be at least 1")
        
        # Validate ansatz type
        ansatz_type = kwargs.get("ansatz_type", "hardware_efficient")
        valid_ansatz = ["hardware_efficient", "uccsd", "real_amplitudes"]
        if ansatz_type not in valid_ansatz:
            raise ValueError(f"Invalid ansatz type. Choose from: {valid_ansatz}")
        params["ansatz_type"] = ansatz_type
    
    elif circuit_type == BenchmarkCircuitType.QAOA:
        if depth is None:
            params["depth"] = 1  # p=1 QAOA
        elif depth < 1:
            raise ValueError("QAOA p parameter must be at least 1")
        
        # Validate problem type
        problem_type = kwargs.get("problem_type", "max_cut")
        valid_problems = ["max_cut", "portfolio_optimization", "vertex_cover"]
        if problem_type not in valid_problems:
            raise ValueError(f"Invalid problem type. Choose from: {valid_problems}")
        params["problem_type"] = problem_type
    
    return params


def get_benchmark_info(circuit_type: Union[str, BenchmarkCircuitType]) -> Dict[str, Any]:
    """
    Get information about a benchmark circuit type.
    
    Args:
        circuit_type: Type of benchmark circuit
        
    Returns:
        Dictionary with circuit information
    """
    if isinstance(circuit_type, str):
        circuit_type = BenchmarkCircuitType(circuit_type)
    
    info = {
        "name": circuit_type.value,
        "type": circuit_type,
        "description": "",
        "typical_use_cases": [],
        "required_parameters": ["qubits"],
        "optional_parameters": [],
        "complexity": "unknown"
    }
    
    if circuit_type == BenchmarkCircuitType.QUANTUM_VOLUME:
        info.update({
            "description": "Quantum volume circuits for benchmarking quantum computers",
            "typical_use_cases": ["Hardware benchmarking", "Error mitigation evaluation"],
            "optional_parameters": ["depth", "seed"],
            "complexity": "exponential in depth"
        })
    
    elif circuit_type == BenchmarkCircuitType.RANDOMIZED_BENCHMARKING:
        info.update({
            "description": "Randomized benchmarking sequences for gate fidelity measurement",
            "typical_use_cases": ["Gate error characterization", "Process tomography"],
            "optional_parameters": ["depth", "gate_set", "seed"],
            "complexity": "linear in sequence length"
        })
    
    elif circuit_type == BenchmarkCircuitType.QUANTUM_FOURIER_TRANSFORM:
        info.update({
            "description": "Quantum Fourier Transform implementation",
            "typical_use_cases": ["Algorithm benchmarking", "Phase estimation"],
            "optional_parameters": ["inverse"],
            "complexity": "O(n²) gates"
        })
    
    elif circuit_type == BenchmarkCircuitType.VQE_ANSATZ:
        info.update({
            "description": "Variational Quantum Eigensolver ansatz circuits",
            "typical_use_cases": ["Quantum chemistry", "Optimization problems"],
            "required_parameters": ["qubits", "depth"],
            "optional_parameters": ["ansatz_type", "parameters"],
            "complexity": "linear in layers"
        })
    
    elif circuit_type == BenchmarkCircuitType.QAOA:
        info.update({
            "description": "Quantum Approximate Optimization Algorithm circuits",
            "typical_use_cases": ["Combinatorial optimization", "Max-Cut problems"],
            "required_parameters": ["qubits", "depth"],
            "optional_parameters": ["problem_type", "graph", "parameters"],
            "complexity": "linear in p parameter"
        })
    
    return info