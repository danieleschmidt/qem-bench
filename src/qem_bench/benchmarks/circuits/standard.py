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
    W_STATE = "w_state"
    RANDOM_CIRCUIT = "random_circuit"
    BRICKWORK_CIRCUIT = "brickwork_circuit"
    HAAR_RANDOM = "haar_random"
    QUANTUM_SUPREMACY = "quantum_supremacy"
    CLIFFORD_CIRCUIT = "clifford_circuit"
    GROVER = "grover"
    QUANTUM_PHASE_ESTIMATION = "quantum_phase_estimation"


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
        >>> circuit = create_benchmark_circuit(BenchmarkCircuitType.QUANTUM_FOURIER_TRANSFORM, qubits=4)
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
        from .algorithmic import create_bell_circuit
        if qubits != 2:
            raise ValueError("Bell state requires exactly 2 qubits")
        bell_state = kwargs.get('bell_state', 0)
        return create_bell_circuit(bell_state)
    
    elif circuit_type == BenchmarkCircuitType.GHZ_STATE:
        from .algorithmic import create_ghz_circuit
        return create_ghz_circuit(qubits)
    
    elif circuit_type == BenchmarkCircuitType.W_STATE:
        from .algorithmic import create_w_state_circuit
        return create_w_state_circuit(qubits, **kwargs)
    
    elif circuit_type == BenchmarkCircuitType.RANDOM_CIRCUIT:
        from .random_circuits import create_random_circuit
        if depth is None:
            depth = qubits * 2  # Default depth
        return create_random_circuit(qubits, depth, **kwargs)
    
    elif circuit_type == BenchmarkCircuitType.BRICKWORK_CIRCUIT:
        from .random_circuits import create_brickwork_circuit
        if depth is None:
            depth = qubits  # Default depth
        return create_brickwork_circuit(qubits, depth, **kwargs)
    
    elif circuit_type == BenchmarkCircuitType.HAAR_RANDOM:
        from .random_circuits import create_haar_random_circuit
        num_gates = kwargs.get('num_gates', qubits * (qubits - 1) * 2)  # Default gate count
        return create_haar_random_circuit(qubits, num_gates, **kwargs)
    
    elif circuit_type == BenchmarkCircuitType.QUANTUM_SUPREMACY:
        from .random_circuits import create_quantum_supremacy_circuit
        if depth is None:
            depth = 10  # Default depth
        # Default to square grid
        grid_size = kwargs.get('grid_size', (int(np.ceil(np.sqrt(qubits))), int(np.ceil(np.sqrt(qubits)))))
        return create_quantum_supremacy_circuit(grid_size, depth, **kwargs)
    
    elif circuit_type == BenchmarkCircuitType.CLIFFORD_CIRCUIT:
        from .random_circuits import create_random_clifford_circuit
        if depth is None:
            depth = qubits * 2  # Default depth
        return create_random_clifford_circuit(qubits, depth, **kwargs)
    
    elif circuit_type == BenchmarkCircuitType.GROVER:
        from .algorithmic import create_grover_circuit
        marked_items = kwargs.get('marked_items', [0])  # Default marked item
        return create_grover_circuit(qubits, marked_items, **kwargs)
    
    elif circuit_type == BenchmarkCircuitType.QUANTUM_PHASE_ESTIMATION:
        from .algorithmic import create_quantum_phase_estimation_circuit
        unitary_qubits = kwargs.get('unitary_qubits', qubits // 2)  # Default half for unitary
        return create_quantum_phase_estimation_circuit(qubits, unitary_qubits, **kwargs)
    
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
    
    elif circuit_type == BenchmarkCircuitType.W_STATE and observable_type == "all_z":
        n_qubits = circuit_params.get('qubits', 3)
        return -1.0/n_qubits  # W state expectation value
    
    # Get expected values from algorithmic circuits module
    try:
        from .algorithmic import get_algorithmic_circuit_expected_values
        expected_values = get_algorithmic_circuit_expected_values()
        
        # Try to find expected value in the lookup table
        circuit_key = f"{circuit_type.value}_{circuit_params.get('qubits', '')}"
        if circuit_key in expected_values:
            return expected_values[circuit_key].get(observable_type, None)
    except ImportError:
        pass
    
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
        bell_state = kwargs.get("bell_state", 0)
        if not 0 <= bell_state <= 3:
            raise ValueError("Bell state index must be 0, 1, 2, or 3")
        params["bell_state"] = bell_state
    
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
        
        # Validate RB type
        rb_type = kwargs.get("rb_type", "single" if qubits == 1 else "simultaneous")
        valid_rb_types = ["single", "simultaneous", "purity"]
        if rb_type not in valid_rb_types:
            raise ValueError(f"Invalid RB type. Choose from: {valid_rb_types}")
        params["rb_type"] = rb_type
    
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
    
    elif circuit_type == BenchmarkCircuitType.W_STATE:
        if qubits < 2:
            raise ValueError("W state requires at least 2 qubits")
    
    elif circuit_type == BenchmarkCircuitType.QUANTUM_SUPREMACY:
        # Validate grid size
        grid_size = kwargs.get("grid_size", (int(np.ceil(np.sqrt(qubits))), int(np.ceil(np.sqrt(qubits)))))
        if grid_size[0] * grid_size[1] != qubits:
            raise ValueError(f"Grid size {grid_size} does not match {qubits} qubits")
        params["grid_size"] = grid_size
        
        if depth is None:
            params["depth"] = 10
        elif depth < 1:
            raise ValueError("Quantum supremacy circuit depth must be at least 1")
    
    elif circuit_type == BenchmarkCircuitType.GROVER:
        marked_items = kwargs.get("marked_items", [0])
        if not marked_items:
            raise ValueError("Grover algorithm requires at least one marked item")
        max_item = max(marked_items)
        if max_item >= 2**qubits:
            raise ValueError(f"Marked item {max_item} exceeds qubit capacity {2**qubits}")
        params["marked_items"] = marked_items
    
    elif circuit_type == BenchmarkCircuitType.QUANTUM_PHASE_ESTIMATION:
        unitary_qubits = kwargs.get("unitary_qubits", qubits // 2)
        if unitary_qubits >= qubits:
            raise ValueError("Unitary qubits must be less than total qubits")
        params["unitary_qubits"] = unitary_qubits
    
    elif circuit_type in [BenchmarkCircuitType.RANDOM_CIRCUIT, BenchmarkCircuitType.BRICKWORK_CIRCUIT, 
                         BenchmarkCircuitType.CLIFFORD_CIRCUIT]:
        if depth is None:
            params["depth"] = qubits * 2
        elif depth < 1:
            raise ValueError("Circuit depth must be at least 1")
    
    elif circuit_type == BenchmarkCircuitType.HAAR_RANDOM:
        num_gates = kwargs.get("num_gates", qubits * (qubits - 1) * 2)
        if num_gates < 1:
            raise ValueError("Number of gates must be at least 1")
        params["num_gates"] = num_gates
    
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
            "optional_parameters": ["depth", "seed", "permutation_type"],
            "complexity": "exponential in depth"
        })
    
    elif circuit_type == BenchmarkCircuitType.RANDOMIZED_BENCHMARKING:
        info.update({
            "description": "Randomized benchmarking sequences for gate fidelity measurement",
            "typical_use_cases": ["Gate error characterization", "Process tomography"],
            "optional_parameters": ["depth", "rb_type", "seed", "interleaved_gate"],
            "complexity": "linear in sequence length"
        })
    
    elif circuit_type == BenchmarkCircuitType.QUANTUM_FOURIER_TRANSFORM:
        info.update({
            "description": "Quantum Fourier Transform implementation",
            "typical_use_cases": ["Algorithm benchmarking", "Phase estimation"],
            "optional_parameters": ["inverse", "approximation_degree"],
            "complexity": "O(n²) gates"
        })
    
    elif circuit_type == BenchmarkCircuitType.VQE_ANSATZ:
        info.update({
            "description": "Variational Quantum Eigensolver ansatz circuits",
            "typical_use_cases": ["Quantum chemistry", "Optimization problems"],
            "required_parameters": ["qubits", "depth"],
            "optional_parameters": ["ansatz_type", "parameters", "entangling_gate"],
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
    
    elif circuit_type == BenchmarkCircuitType.BELL_STATE:
        info.update({
            "description": "Bell state preparation circuit",
            "typical_use_cases": ["Entanglement generation", "Quantum communication"],
            "optional_parameters": ["bell_state"],
            "complexity": "O(1) gates"
        })
    
    elif circuit_type == BenchmarkCircuitType.GHZ_STATE:
        info.update({
            "description": "GHZ state preparation circuit",
            "typical_use_cases": ["Multipartite entanglement", "Quantum sensing"],
            "optional_parameters": [],
            "complexity": "O(n) gates"
        })
    
    elif circuit_type == BenchmarkCircuitType.W_STATE:
        info.update({
            "description": "W state preparation circuit",
            "typical_use_cases": ["Multipartite entanglement", "Quantum sensing"],
            "required_parameters": ["qubits"],
            "optional_parameters": [],
            "complexity": "O(n) gates"
        })
    
    elif circuit_type == BenchmarkCircuitType.GROVER:
        info.update({
            "description": "Grover's quantum search algorithm",
            "typical_use_cases": ["Database search", "Amplitude amplification"],
            "required_parameters": ["qubits"],
            "optional_parameters": ["marked_items", "num_iterations"],
            "complexity": "O(√N) iterations"
        })
    
    elif circuit_type == BenchmarkCircuitType.QUANTUM_PHASE_ESTIMATION:
        info.update({
            "description": "Quantum phase estimation algorithm",
            "typical_use_cases": ["Eigenvalue estimation", "Quantum simulation"],
            "required_parameters": ["qubits"],
            "optional_parameters": ["unitary_qubits", "phase"],
            "complexity": "O(n²) gates"
        })
    
    elif circuit_type == BenchmarkCircuitType.RANDOM_CIRCUIT:
        info.update({
            "description": "Random quantum circuit with configurable gate set",
            "typical_use_cases": ["Random sampling", "Benchmarking"],
            "required_parameters": ["qubits", "depth"],
            "optional_parameters": ["gate_set", "two_qubit_gate_probability", "seed"],
            "complexity": "exponential in depth"
        })
    
    elif circuit_type == BenchmarkCircuitType.BRICKWORK_CIRCUIT:
        info.update({
            "description": "Brickwork random circuit with regular structure",
            "typical_use_cases": ["Random circuit sampling", "Benchmarking"],
            "required_parameters": ["qubits", "depth"],
            "optional_parameters": ["gate_type", "seed"],
            "complexity": "exponential in depth"
        })
    
    elif circuit_type == BenchmarkCircuitType.HAAR_RANDOM:
        info.update({
            "description": "Haar-random unitary approximation",
            "typical_use_cases": ["Random unitary sampling", "Quantum supremacy"],
            "required_parameters": ["qubits"],
            "optional_parameters": ["num_gates", "seed"],
            "complexity": "exponential simulation"
        })
    
    elif circuit_type == BenchmarkCircuitType.QUANTUM_SUPREMACY:
        info.update({
            "description": "Google quantum supremacy style circuit",
            "typical_use_cases": ["Quantum supremacy demonstration", "Benchmarking"],
            "required_parameters": ["qubits", "depth"],
            "optional_parameters": ["grid_size", "seed"],
            "complexity": "classically intractable"
        })
    
    elif circuit_type == BenchmarkCircuitType.CLIFFORD_CIRCUIT:
        info.update({
            "description": "Random Clifford circuit (efficiently simulable)",
            "typical_use_cases": ["Clifford benchmarking", "Error correction testing"],
            "required_parameters": ["qubits", "depth"],
            "optional_parameters": ["seed"],
            "complexity": "polynomial simulation"
        })
    
    return info