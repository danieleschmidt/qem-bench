"""Quantum Volume benchmark circuit implementation."""

import numpy as np
from typing import Optional, List
from ...jax.circuits import JAXCircuit


def create_quantum_volume_circuit(
    num_qubits: int,
    depth: int,
    seed: Optional[int] = None,
    permutation_type: str = "random"
) -> JAXCircuit:
    """
    Create a Quantum Volume benchmark circuit.
    
    Quantum Volume circuits consist of layers of random SU(4) gates
    applied to random pairs of qubits, following the IBM QV protocol.
    
    Args:
        num_qubits: Number of qubits (should be even for proper pairing)
        depth: Number of layers (typically equals num_qubits for QV)
        seed: Random seed for reproducibility
        permutation_type: Type of qubit permutation ("random", "linear", "brick")
        
    Returns:
        JAXCircuit implementing quantum volume benchmark
        
    Reference:
        Quantum Volume circuit definition from IBM's quantum volume paper
        arXiv:1811.12926
    """
    if seed is not None:
        np.random.seed(seed)
    
    circuit = JAXCircuit(num_qubits, name=f"quantum_volume_{num_qubits}x{depth}")
    
    for layer in range(depth):
        # Generate qubit permutation for this layer
        qubit_pairs = _generate_qubit_permutation(num_qubits, permutation_type, layer)
        
        # Apply random SU(4) gates to each pair
        for qubit1, qubit2 in qubit_pairs:
            _apply_random_su4_gate(circuit, qubit1, qubit2)
    
    return circuit


def _generate_qubit_permutation(
    num_qubits: int, 
    permutation_type: str, 
    layer: int
) -> List[tuple[int, int]]:
    """Generate qubit pairing for a layer."""
    
    if permutation_type == "random":
        # Random permutation of all qubits, then pair adjacent
        qubits = list(range(num_qubits))
        np.random.shuffle(qubits)
        pairs = [(qubits[i], qubits[i+1]) for i in range(0, num_qubits-1, 2)]
    
    elif permutation_type == "linear":
        # Linear nearest-neighbor pairing
        if layer % 2 == 0:
            # Even layers: (0,1), (2,3), (4,5), ...
            pairs = [(i, i+1) for i in range(0, num_qubits-1, 2)]
        else:
            # Odd layers: (1,2), (3,4), (5,6), ...
            pairs = [(i, i+1) for i in range(1, num_qubits-1, 2)]
    
    elif permutation_type == "brick":
        # Brick-like pattern for 2D connectivity
        if layer % 2 == 0:
            # Horizontal connections
            pairs = [(i, i+1) for i in range(0, num_qubits-1, 2)]
        else:
            # Shifted horizontal connections
            pairs = [(i, i+1) for i in range(1, num_qubits-1, 2)]
    
    else:
        raise ValueError(f"Unknown permutation type: {permutation_type}")
    
    return pairs


def _apply_random_su4_gate(circuit: JAXCircuit, qubit1: int, qubit2: int) -> None:
    """
    Apply a random SU(4) gate to two qubits.
    
    This is implemented as a sequence of single and two-qubit gates
    that can generate any SU(4) operation.
    """
    # Random single-qubit rotations before entangling gate
    circuit.ry(np.random.uniform(0, 2*np.pi), qubit1)
    circuit.rz(np.random.uniform(0, 2*np.pi), qubit1)
    circuit.ry(np.random.uniform(0, 2*np.pi), qubit2)
    circuit.rz(np.random.uniform(0, 2*np.pi), qubit2)
    
    # Entangling gate (CNOT)
    circuit.cx(qubit1, qubit2)
    
    # Random single-qubit rotations after entangling gate
    circuit.ry(np.random.uniform(0, 2*np.pi), qubit1)
    circuit.rz(np.random.uniform(0, 2*np.pi), qubit1)
    circuit.ry(np.random.uniform(0, 2*np.pi), qubit2)
    circuit.rz(np.random.uniform(0, 2*np.pi), qubit2)
    
    # Second entangling gate
    circuit.cx(qubit2, qubit1)
    
    # Final single-qubit rotations
    circuit.ry(np.random.uniform(0, 2*np.pi), qubit1)
    circuit.rz(np.random.uniform(0, 2*np.pi), qubit1)
    circuit.ry(np.random.uniform(0, 2*np.pi), qubit2)
    circuit.rz(np.random.uniform(0, 2*np.pi), qubit2)


def create_quantum_volume_model_circuit(
    num_qubits: int,
    depth: int,
    fidelity_target: float = 2/3
) -> JAXCircuit:
    """
    Create a model circuit for Quantum Volume benchmarking.
    
    This creates a specific instance that can be used to test
    whether a quantum computer can achieve the required fidelity
    threshold for a given quantum volume.
    
    Args:
        num_qubits: Number of qubits
        depth: Circuit depth
        fidelity_target: Target fidelity threshold (default 2/3)
        
    Returns:
        JAXCircuit for QV model circuit
    """
    # Use fixed seed for reproducible model circuit
    circuit = create_quantum_volume_circuit(
        num_qubits, 
        depth, 
        seed=42,  # Fixed seed for model circuit
        permutation_type="random"
    )
    
    circuit.name = f"qv_model_{num_qubits}x{depth}"
    return circuit


def calculate_quantum_volume_score(
    num_qubits: int,
    success_probability: float,
    threshold: float = 2/3
) -> int:
    """
    Calculate Quantum Volume score.
    
    Args:
        num_qubits: Number of qubits in the test
        success_probability: Measured success probability
        threshold: Required threshold (default 2/3)
        
    Returns:
        Quantum Volume score (2^num_qubits if successful, 0 if failed)
    """
    if success_probability >= threshold:
        return 2 ** num_qubits
    else:
        return 0


def estimate_quantum_volume_fidelity(
    ideal_counts: dict,
    measured_counts: dict,
    total_shots: int
) -> float:
    """
    Estimate fidelity from measurement counts.
    
    Args:
        ideal_counts: Ideal measurement counts (from noiseless simulation)
        measured_counts: Actual measurement counts
        total_shots: Total number of measurement shots
        
    Returns:
        Estimated fidelity
    """
    # Calculate heavy output probability
    ideal_probs = {k: v/sum(ideal_counts.values()) for k, v in ideal_counts.items()}
    
    # Find median probability
    sorted_probs = sorted(ideal_probs.values())
    median_prob = sorted_probs[len(sorted_probs)//2]
    
    # Heavy outputs are those with probability > median
    heavy_outputs = {k for k, p in ideal_probs.items() if p > median_prob}
    
    # Calculate measured heavy output probability
    heavy_counts = sum(measured_counts.get(k, 0) for k in heavy_outputs)
    measured_heavy_prob = heavy_counts / total_shots
    
    # Estimate fidelity (this is a simplified approximation)
    ideal_heavy_prob = sum(ideal_probs[k] for k in heavy_outputs)
    
    # Linear approximation: F ≈ 2 * P_heavy - 1
    # where P_heavy is the heavy output probability
    fidelity = 2 * measured_heavy_prob - 1
    fidelity = max(0, min(1, fidelity))  # Clamp to [0,1]
    
    return fidelity


def generate_quantum_volume_suite(
    max_qubits: int,
    min_qubits: int = 2
) -> List[JAXCircuit]:
    """
    Generate a suite of Quantum Volume circuits for benchmarking.
    
    Args:
        max_qubits: Maximum number of qubits
        min_qubits: Minimum number of qubits
        
    Returns:
        List of QV circuits with increasing size
    """
    circuits = []
    
    for n_qubits in range(min_qubits, max_qubits + 1):
        depth = n_qubits  # Standard QV depth
        circuit = create_quantum_volume_circuit(n_qubits, depth)
        circuits.append(circuit)
    
    return circuits