"""Random circuit generation utilities for benchmarking."""

import numpy as np
import jax.numpy as jnp
from typing import List, Optional, Dict, Any, Union, Tuple
from ...jax.circuits import JAXCircuit


def create_random_circuit(
    num_qubits: int,
    depth: int,
    gate_set: Optional[List[str]] = None,
    two_qubit_gate_probability: float = 0.1,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    seed: Optional[int] = None
) -> JAXCircuit:
    """
    Create a random quantum circuit for benchmarking.
    
    Args:
        num_qubits: Number of qubits
        depth: Circuit depth (number of gate layers)
        gate_set: List of gates to use (if None, uses default set)
        two_qubit_gate_probability: Probability of two-qubit gates
        connectivity: Allowed qubit connections (if None, all-to-all)
        seed: Random seed for reproducibility
        
    Returns:
        JAXCircuit with random gates
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    if gate_set is None:
        gate_set = ["h", "x", "y", "z", "rx", "ry", "rz", "cx", "cz"]
    
    if connectivity is None:
        # All-to-all connectivity
        connectivity = [(i, j) for i in range(num_qubits) 
                       for j in range(i+1, num_qubits)]
    
    circuit = JAXCircuit(num_qubits, name=f"random_{num_qubits}q_d{depth}")
    
    for layer in range(depth):
        # Determine which qubits get gates this layer
        active_qubits = set()
        
        for qubit in range(num_qubits):
            if np.random.random() < 0.8:  # 80% chance of gate on each qubit
                active_qubits.add(qubit)
        
        # Add random gates
        processed_qubits = set()
        
        for qubit in active_qubits:
            if qubit in processed_qubits:
                continue
            
            # Decide between single-qubit and two-qubit gate
            if (np.random.random() < two_qubit_gate_probability and 
                len(active_qubits - processed_qubits) > 1):
                
                # Two-qubit gate
                available_partners = [
                    q for q in active_qubits - processed_qubits - {qubit}
                    if (qubit, q) in connectivity or (q, qubit) in connectivity
                ]
                
                if available_partners:
                    partner = np.random.choice(available_partners)
                    gate = np.random.choice([g for g in gate_set if g in ["cx", "cz", "cphase"]])
                    
                    if gate == "cx":
                        circuit.cx(qubit, partner)
                    elif gate == "cz":
                        circuit.cz(qubit, partner)
                    elif gate == "cphase":
                        angle = np.random.uniform(0, 2*np.pi)
                        circuit.cphase(angle, qubit, partner)
                    
                    processed_qubits.add(qubit)
                    processed_qubits.add(partner)
                    continue
            
            # Single-qubit gate
            gate = np.random.choice([g for g in gate_set if g not in ["cx", "cz", "cphase"]])
            
            if gate == "h":
                circuit.h(qubit)
            elif gate == "x":
                circuit.x(qubit)
            elif gate == "y":
                circuit.y(qubit)
            elif gate == "z":
                circuit.z(qubit)
            elif gate == "s":
                circuit.s(qubit)
            elif gate == "t":
                circuit.t(qubit)
            elif gate in ["rx", "ry", "rz"]:
                angle = np.random.uniform(0, 2*np.pi)
                getattr(circuit, gate)(angle, qubit)
            
            processed_qubits.add(qubit)
    
    return circuit


def create_brickwork_circuit(
    num_qubits: int,
    depth: int,
    gate_type: str = "random_unitary",
    seed: Optional[int] = None
) -> JAXCircuit:
    """
    Create a brickwork random circuit.
    
    Brickwork circuits have a regular structure with alternating layers
    of two-qubit gates in a brick-like pattern.
    
    Args:
        num_qubits: Number of qubits
        depth: Number of two-qubit gate layers
        gate_type: Type of two-qubit gates ("random_unitary", "cx", "cz")
        seed: Random seed
        
    Returns:
        JAXCircuit with brickwork structure
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    circuit = JAXCircuit(num_qubits, name=f"brickwork_{gate_type}_{num_qubits}q_d{depth}")
    
    # Initial single-qubit rotations
    for qubit in range(num_qubits):
        circuit.ry(np.random.uniform(0, 2*np.pi), qubit)
        circuit.rz(np.random.uniform(0, 2*np.pi), qubit)
    
    for layer in range(depth):
        if layer % 2 == 0:
            # Even layers: (0,1), (2,3), (4,5), ...
            pairs = [(i, i+1) for i in range(0, num_qubits-1, 2)]
        else:
            # Odd layers: (1,2), (3,4), (5,6), ...
            pairs = [(i, i+1) for i in range(1, num_qubits-1, 2)]
        
        for q1, q2 in pairs:
            if gate_type == "random_unitary":
                _add_random_two_qubit_unitary(circuit, q1, q2)
            elif gate_type == "cx":
                circuit.cx(q1, q2)
            elif gate_type == "cz":
                circuit.cz(q1, q2)
            else:
                raise ValueError(f"Unknown gate type: {gate_type}")
        
        # Single-qubit rotations between layers
        if layer < depth - 1:
            for qubit in range(num_qubits):
                circuit.ry(np.random.uniform(0, 2*np.pi), qubit)
                circuit.rz(np.random.uniform(0, 2*np.pi), qubit)
    
    return circuit


def create_haar_random_circuit(
    num_qubits: int,
    num_gates: int,
    seed: Optional[int] = None
) -> JAXCircuit:
    """
    Create a circuit approximating a Haar-random unitary.
    
    Uses the approach of random two-qubit gates to approximate
    Haar-random unitaries on the full Hilbert space.
    
    Args:
        num_qubits: Number of qubits
        num_gates: Number of random two-qubit gates
        seed: Random seed
        
    Returns:
        JAXCircuit approximating Haar-random unitary
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    circuit = JAXCircuit(num_qubits, name=f"haar_random_{num_qubits}q_{num_gates}gates")
    
    # Initial random single-qubit unitaries
    for qubit in range(num_qubits):
        _add_random_single_qubit_unitary(circuit, qubit)
    
    # Random two-qubit gates
    for _ in range(num_gates):
        q1, q2 = np.random.choice(num_qubits, 2, replace=False)
        _add_random_two_qubit_unitary(circuit, q1, q2)
    
    return circuit


def _add_random_single_qubit_unitary(circuit: JAXCircuit, qubit: int) -> None:
    """Add a random single-qubit unitary (Haar-random SU(2))."""
    
    # Generate random parameters for SU(2)
    # Using ZYZ decomposition: U = Rz(α)Ry(β)Rz(γ)
    alpha = np.random.uniform(0, 2*np.pi)
    beta = np.random.uniform(0, np.pi)
    gamma = np.random.uniform(0, 2*np.pi)
    
    circuit.rz(alpha, qubit)
    circuit.ry(beta, qubit)
    circuit.rz(gamma, qubit)


def _add_random_two_qubit_unitary(circuit: JAXCircuit, q1: int, q2: int) -> None:
    """Add a random two-qubit unitary."""
    
    # Decomposition of random SU(4) using the KAK decomposition
    # This is a simplified version
    
    # Random single-qubit unitaries before
    _add_random_single_qubit_unitary(circuit, q1)
    _add_random_single_qubit_unitary(circuit, q2)
    
    # Random entangling operation
    circuit.cx(q1, q2)
    
    # Random phases
    theta1 = np.random.uniform(0, np.pi/2)
    theta2 = np.random.uniform(0, np.pi/2)
    theta3 = np.random.uniform(0, np.pi/2)
    
    circuit.rz(theta1, q1)
    circuit.ry(theta2, q1)
    circuit.cx(q2, q1)
    circuit.ry(theta3, q1)
    circuit.cx(q1, q2)
    
    # Random single-qubit unitaries after
    _add_random_single_qubit_unitary(circuit, q1)
    _add_random_single_qubit_unitary(circuit, q2)


def create_quantum_supremacy_circuit(
    grid_size: Tuple[int, int],
    depth: int,
    seed: Optional[int] = None
) -> JAXCircuit:
    """
    Create a quantum supremacy-style random circuit.
    
    Based on the Google quantum supremacy experiment design
    with random single-qubit gates and CZ gates on a 2D grid.
    
    Args:
        grid_size: Tuple of (rows, cols) for 2D qubit grid
        depth: Circuit depth
        seed: Random seed
        
    Returns:
        JAXCircuit for quantum supremacy benchmark
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    rows, cols = grid_size
    num_qubits = rows * cols
    
    circuit = JAXCircuit(
        num_qubits, 
        name=f"supremacy_{rows}x{cols}_d{depth}"
    )
    
    def qubit_index(row: int, col: int) -> int:
        """Convert 2D grid coordinates to qubit index."""
        return row * cols + col
    
    # Pattern of gates similar to Google's experiment
    gate_patterns = [
        [("sqrt_x", 0.5), ("sqrt_y", 0.5)],  # Pattern A
        [("sqrt_w", 1.0)],                   # Pattern B  
    ]
    
    for cycle in range(depth):
        # Single-qubit gates
        pattern = gate_patterns[cycle % len(gate_patterns)]
        
        for row in range(rows):
            for col in range(cols):
                qubit = qubit_index(row, col)
                gate_type, prob = np.random.choice(
                    len(pattern), p=[p[1] for p in pattern]
                ), pattern[np.random.choice(len(pattern), p=[p[1] for p in pattern])][0]
                
                if gate_type == "sqrt_x":
                    circuit.rx(np.pi/2, qubit)
                elif gate_type == "sqrt_y":
                    circuit.ry(np.pi/2, qubit)
                elif gate_type == "sqrt_w":
                    # √W gate (arbitrary single-qubit rotation)
                    circuit.rx(np.random.uniform(0, 2*np.pi), qubit)
                    circuit.ry(np.random.uniform(0, 2*np.pi), qubit)
        
        # Two-qubit CZ gates
        if cycle % 4 < 2:
            # Horizontal connections
            for row in range(rows):
                start_col = 0 if (cycle + row) % 2 == 0 else 1
                for col in range(start_col, cols-1, 2):
                    q1 = qubit_index(row, col)
                    q2 = qubit_index(row, col + 1)
                    circuit.cz(q1, q2)
        else:
            # Vertical connections
            for col in range(cols):
                start_row = 0 if (cycle + col) % 2 == 0 else 1
                for row in range(start_row, rows-1, 2):
                    q1 = qubit_index(row, col)
                    q2 = qubit_index(row + 1, col)
                    circuit.cz(q1, q2)
    
    return circuit


def create_random_clifford_circuit(
    num_qubits: int,
    depth: int,
    seed: Optional[int] = None
) -> JAXCircuit:
    """
    Create a random Clifford circuit.
    
    Uses only Clifford gates (H, S, CX) to create efficiently
    simulable random circuits.
    
    Args:
        num_qubits: Number of qubits
        depth: Circuit depth
        seed: Random seed
        
    Returns:
        JAXCircuit with only Clifford gates
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    circuit = JAXCircuit(num_qubits, name=f"random_clifford_{num_qubits}q_d{depth}")
    
    clifford_gates = ["h", "s", "x", "y", "z", "cx"]
    
    for _ in range(depth):
        gate = np.random.choice(clifford_gates)
        
        if gate in ["h", "s", "x", "y", "z"]:
            qubit = np.random.randint(num_qubits)
            getattr(circuit, gate)(qubit)
        elif gate == "cx":
            q1, q2 = np.random.choice(num_qubits, 2, replace=False)
            circuit.cx(q1, q2)
    
    return circuit


def create_parameterized_random_circuit(
    num_qubits: int,
    depth: int,
    param_density: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[JAXCircuit, jnp.ndarray]:
    """
    Create a parameterized random circuit for variational algorithms.
    
    Args:
        num_qubits: Number of qubits
        depth: Circuit depth
        param_density: Fraction of gates that are parameterized
        seed: Random seed
        
    Returns:
        Tuple of (circuit, parameters)
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    circuit = JAXCircuit(num_qubits, name=f"param_random_{num_qubits}q_d{depth}")
    parameters = []
    
    for layer in range(depth):
        # Random single-qubit rotations
        for qubit in range(num_qubits):
            if np.random.random() < param_density:
                # Parameterized gate
                gate_type = np.random.choice(["rx", "ry", "rz"])
                param = np.random.uniform(0, 2*np.pi)
                parameters.append(param)
                getattr(circuit, gate_type)(param, qubit)
            else:
                # Fixed gate
                gate = np.random.choice(["h", "x", "y", "z", "s"])
                getattr(circuit, gate)(qubit)
        
        # Entangling gates
        if layer < depth - 1:
            for qubit in range(0, num_qubits-1, 2):
                if np.random.random() < 0.7:  # 70% chance of entangling gate
                    circuit.cx(qubit, qubit + 1)
    
    return circuit, jnp.array(parameters)


def create_random_circuit_ensemble(
    num_qubits: int,
    depth: int,
    ensemble_size: int,
    circuit_type: str = "haar_random",
    **kwargs
) -> List[JAXCircuit]:
    """
    Create an ensemble of random circuits for statistical analysis.
    
    Args:
        num_qubits: Number of qubits
        depth: Circuit depth
        ensemble_size: Number of circuits to generate
        circuit_type: Type of random circuit
        **kwargs: Additional arguments for circuit generation
        
    Returns:
        List of random circuits
    """
    
    circuits = []
    
    for i in range(ensemble_size):
        seed = kwargs.get("seed", None)
        if seed is not None:
            seed = seed + i
        
        if circuit_type == "haar_random":
            circuit = create_haar_random_circuit(
                num_qubits, depth * num_qubits, seed=seed
            )
        elif circuit_type == "brickwork":
            circuit = create_brickwork_circuit(
                num_qubits, depth, seed=seed, **kwargs
            )
        elif circuit_type == "clifford":
            circuit = create_random_clifford_circuit(
                num_qubits, depth, seed=seed
            )
        elif circuit_type == "general":
            circuit = create_random_circuit(
                num_qubits, depth, seed=seed, **kwargs
            )
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
        
        circuit.name = f"{circuit.name}_#{i}"
        circuits.append(circuit)
    
    return circuits


def estimate_circuit_complexity(circuit: JAXCircuit) -> Dict[str, Any]:
    """
    Estimate the complexity metrics of a random circuit.
    
    Args:
        circuit: JAXCircuit to analyze
        
    Returns:
        Dictionary with complexity metrics
    """
    
    gate_counts = circuit.count_gates()
    
    # Count two-qubit gates
    two_qubit_gates = sum(
        count for gate, count in gate_counts.items() 
        if gate in ["cx", "cz", "controlled_phase", "swap"]
    )
    
    # Estimate entanglement depth (simplified)
    entanglement_depth = min(circuit.depth, circuit.num_qubits)
    
    # Estimate classical simulation complexity
    if two_qubit_gates == 0:
        simulation_complexity = "polynomial"
    elif circuit.num_qubits <= 10:
        simulation_complexity = "feasible"
    elif circuit.num_qubits <= 20:
        simulation_complexity = "challenging"
    else:
        simulation_complexity = "intractable"
    
    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth,
        "total_gates": circuit.size,
        "two_qubit_gates": two_qubit_gates,
        "entanglement_depth": entanglement_depth,
        "gate_counts": gate_counts,
        "simulation_complexity": simulation_complexity,
        "estimated_fidelity_bound": np.exp(-two_qubit_gates * 0.01)  # Rough estimate
    }