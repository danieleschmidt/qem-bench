"""Randomized Benchmarking circuit implementations."""

import numpy as np
import jax.numpy as jnp
from typing import List, Optional, Dict, Tuple, Union
from ...jax.circuits import JAXCircuit


# Clifford group generators for single-qubit RB
SINGLE_QUBIT_CLIFFORDS = [
    # Identity-equivalent
    [],  # I
    [("y", jnp.pi), ("x", jnp.pi)],  # X
    [("y", -jnp.pi/2), ("x", jnp.pi), ("y", jnp.pi/2)],  # Y  
    [("x", jnp.pi)],  # Z
    [("y", jnp.pi/2), ("x", jnp.pi)],  # -X
    [("y", jnp.pi/2)],  # -Y
    [("y", -jnp.pi/2), ("x", jnp.pi)],  # -Z
    [("x", jnp.pi/2), ("y", jnp.pi/2)],  # H
    [("x", -jnp.pi/2), ("y", -jnp.pi/2)],  # -H
    [("y", jnp.pi/2), ("x", jnp.pi/2)],  # P (S gate)
    [("y", -jnp.pi/2), ("x", -jnp.pi/2)],  # -P (-S gate)
    [("x", jnp.pi/2)],  # K
    [("x", -jnp.pi/2)],  # -K
    [("y", jnp.pi/2), ("x", jnp.pi), ("y", -jnp.pi/2)],  # Rotations
    [("y", -jnp.pi/2), ("x", jnp.pi), ("y", jnp.pi/2)],
    [("x", jnp.pi/2), ("y", jnp.pi), ("x", -jnp.pi/2)],
    [("x", -jnp.pi/2), ("y", jnp.pi), ("x", jnp.pi/2)],
    [("y", jnp.pi)],  # More rotations
    [("x", jnp.pi/2), ("y", -jnp.pi/2)],
    [("x", -jnp.pi/2), ("y", jnp.pi/2)],
    [("y", jnp.pi/2), ("x", -jnp.pi/2)],
    [("y", -jnp.pi/2), ("x", jnp.pi/2)],
    [("x", jnp.pi/2), ("y", jnp.pi/2), ("x", jnp.pi/2)],
    [("x", -jnp.pi/2), ("y", -jnp.pi/2), ("x", -jnp.pi/2)]
]


def create_rb_circuit(
    num_qubits: int,
    sequence_length: int,
    rb_type: str = "single",
    gate_set: Optional[List[str]] = None,
    seed: Optional[int] = None,
    interleaved_gate: Optional[str] = None
) -> JAXCircuit:
    """
    Create a Randomized Benchmarking circuit.
    
    Args:
        num_qubits: Number of qubits
        sequence_length: Length of the Clifford sequence
        rb_type: Type of RB ("single", "simultaneous", "purity")
        gate_set: Custom gate set (if None, uses Clifford gates)
        seed: Random seed for reproducibility
        interleaved_gate: Gate to interleave for interleaved RB
        
    Returns:
        JAXCircuit implementing randomized benchmarking
        
    Reference:
        Randomized benchmarking of quantum gates
        Emerson et al., Science 317, 1893 (2007)
        Magesan et al., Phys. Rev. Lett. 109, 080505 (2012)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if rb_type == "single" and num_qubits != 1:
        raise ValueError("Single-qubit RB requires exactly 1 qubit")
    
    circuit = JAXCircuit(
        num_qubits, 
        name=f"rb_{rb_type}_{num_qubits}q_m{sequence_length}"
    )
    
    if rb_type == "single":
        return _create_single_qubit_rb(circuit, sequence_length, interleaved_gate)
    elif rb_type == "simultaneous":
        return _create_simultaneous_rb(circuit, sequence_length, interleaved_gate)
    elif rb_type == "purity":
        return _create_purity_rb(circuit, sequence_length)
    else:
        raise ValueError(f"Unknown RB type: {rb_type}")


def _create_single_qubit_rb(
    circuit: JAXCircuit, 
    sequence_length: int, 
    interleaved_gate: Optional[str] = None
) -> JAXCircuit:
    """Create single-qubit randomized benchmarking sequence."""
    
    # Generate random Clifford sequence
    clifford_sequence = []
    for _ in range(sequence_length):
        clifford_idx = np.random.randint(len(SINGLE_QUBIT_CLIFFORDS))
        clifford_sequence.append(clifford_idx)
    
    # Apply Clifford gates
    for i, clifford_idx in enumerate(clifford_sequence):
        clifford_ops = SINGLE_QUBIT_CLIFFORDS[clifford_idx]
        
        # Apply the Clifford gate
        for gate_name, angle in clifford_ops:
            if gate_name == "x":
                circuit.rx(angle, 0)
            elif gate_name == "y":
                circuit.ry(angle, 0)
            elif gate_name == "z":
                circuit.rz(angle, 0)
        
        # Interleave gate if specified
        if interleaved_gate is not None and i < sequence_length - 1:
            _apply_interleaved_gate(circuit, interleaved_gate, [0])
    
    # Calculate and apply recovery operation
    recovery_clifford = _calculate_recovery_clifford_single(clifford_sequence)
    recovery_ops = SINGLE_QUBIT_CLIFFORDS[recovery_clifford]
    for gate_name, angle in recovery_ops:
        if gate_name == "x":
            circuit.rx(angle, 0)
        elif gate_name == "y":
            circuit.ry(angle, 0)
        elif gate_name == "z":
            circuit.rz(angle, 0)
    
    return circuit


def _create_simultaneous_rb(
    circuit: JAXCircuit, 
    sequence_length: int, 
    interleaved_gate: Optional[str] = None
) -> JAXCircuit:
    """Create simultaneous randomized benchmarking sequence."""
    
    num_qubits = circuit.num_qubits
    
    # Generate random Clifford sequence for each qubit
    clifford_sequences = []
    for _ in range(num_qubits):
        qubit_sequence = []
        for _ in range(sequence_length):
            clifford_idx = np.random.randint(len(SINGLE_QUBIT_CLIFFORDS))
            qubit_sequence.append(clifford_idx)
        clifford_sequences.append(qubit_sequence)
    
    # Apply Clifford gates simultaneously
    for i in range(sequence_length):
        for qubit in range(num_qubits):
            clifford_idx = clifford_sequences[qubit][i]
            clifford_ops = SINGLE_QUBIT_CLIFFORDS[clifford_idx]
            
            for gate_name, angle in clifford_ops:
                if gate_name == "x":
                    circuit.rx(angle, qubit)
                elif gate_name == "y":
                    circuit.ry(angle, qubit)
                elif gate_name == "z":
                    circuit.rz(angle, qubit)
        
        # Interleave gate if specified
        if interleaved_gate is not None and i < sequence_length - 1:
            if interleaved_gate == "cx":
                # Apply CNOT between adjacent qubits
                for qubit in range(0, num_qubits - 1, 2):
                    circuit.cx(qubit, qubit + 1)
            else:
                for qubit in range(num_qubits):
                    _apply_interleaved_gate(circuit, interleaved_gate, [qubit])
    
    # Apply recovery operations
    for qubit in range(num_qubits):
        recovery_clifford = _calculate_recovery_clifford_single(clifford_sequences[qubit])
        recovery_ops = SINGLE_QUBIT_CLIFFORDS[recovery_clifford]
        for gate_name, angle in recovery_ops:
            if gate_name == "x":
                circuit.rx(angle, qubit)
            elif gate_name == "y":
                circuit.ry(angle, qubit)
            elif gate_name == "z":
                circuit.rz(angle, qubit)
    
    return circuit


def _create_purity_rb(circuit: JAXCircuit, sequence_length: int) -> JAXCircuit:
    """Create purity randomized benchmarking sequence."""
    
    num_qubits = circuit.num_qubits
    
    # Purity RB uses random Pauli measurements
    pauli_gates = ["x", "y", "z"]
    
    for _ in range(sequence_length):
        # Apply random single-qubit Cliffords
        for qubit in range(num_qubits):
            clifford_idx = np.random.randint(len(SINGLE_QUBIT_CLIFFORDS))
            clifford_ops = SINGLE_QUBIT_CLIFFORDS[clifford_idx]
            
            for gate_name, angle in clifford_ops:
                if gate_name == "x":
                    circuit.rx(angle, qubit)
                elif gate_name == "y":
                    circuit.ry(angle, qubit)
                elif gate_name == "z":
                    circuit.rz(angle, qubit)
        
        # Add random two-qubit gates for multi-qubit purity RB
        if num_qubits > 1:
            for _ in range(num_qubits // 2):
                q1, q2 = np.random.choice(num_qubits, 2, replace=False)
                circuit.cx(q1, q2)
    
    return circuit


def _apply_interleaved_gate(circuit: JAXCircuit, gate_name: str, qubits: List[int]) -> None:
    """Apply an interleaved gate for interleaved RB."""
    
    if gate_name == "x":
        circuit.x(qubits[0])
    elif gate_name == "y":
        circuit.y(qubits[0])
    elif gate_name == "z":
        circuit.z(qubits[0])
    elif gate_name == "h":
        circuit.h(qubits[0])
    elif gate_name == "s":
        circuit.s(qubits[0])
    elif gate_name == "t":
        circuit.t(qubits[0])
    elif gate_name == "cx" and len(qubits) >= 2:
        circuit.cx(qubits[0], qubits[1])
    else:
        raise ValueError(f"Unsupported interleaved gate: {gate_name}")


def _calculate_recovery_clifford_single(clifford_sequence: List[int]) -> int:
    """Calculate the recovery Clifford for single-qubit RB."""
    
    # This is a simplified recovery calculation
    # In practice, this would involve proper Clifford group multiplication
    
    # For now, use identity as recovery (this is not physically correct
    # but serves as a placeholder for the proper implementation)
    
    # Proper implementation would:
    # 1. Convert each Clifford index to a matrix representation
    # 2. Multiply all matrices in sequence
    # 3. Find the inverse of the product
    # 4. Find the Clifford index corresponding to the inverse
    
    return 0  # Identity Clifford


def create_rb_sequence(
    num_qubits: int,
    sequence_lengths: List[int],
    rb_type: str = "single",
    num_samples: int = 10,
    seed: Optional[int] = None
) -> List[JAXCircuit]:
    """
    Create a full RB sequence with multiple lengths.
    
    Args:
        num_qubits: Number of qubits
        sequence_lengths: List of sequence lengths to test
        rb_type: Type of RB protocol
        num_samples: Number of random circuits per length
        seed: Random seed for reproducibility
        
    Returns:
        List of RB circuits for different sequence lengths
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    circuits = []
    
    for seq_length in sequence_lengths:
        for sample in range(num_samples):
            # Use different seed for each circuit
            circuit_seed = None if seed is None else seed + seq_length * 1000 + sample
            
            circuit = create_rb_circuit(
                num_qubits=num_qubits,
                sequence_length=seq_length,
                rb_type=rb_type,
                seed=circuit_seed
            )
            
            circuit.name = f"rb_{rb_type}_{num_qubits}q_m{seq_length}_s{sample}"
            circuits.append(circuit)
    
    return circuits


def calculate_rb_fidelity(
    sequence_lengths: List[int],
    survival_probabilities: List[float],
    num_qubits: int = 1
) -> Tuple[float, float]:
    """
    Calculate average gate fidelity from RB data.
    
    Args:
        sequence_lengths: List of sequence lengths used
        survival_probabilities: Measured survival probabilities
        num_qubits: Number of qubits (for multi-qubit RB)
        
    Returns:
        Tuple of (average_gate_fidelity, decay_parameter)
        
    Reference:
        F_avg = 1 - (1-p)*(d-1)/d
        where p is the decay parameter and d = 2^n is the dimension
    """
    
    # Fit exponential decay: P(m) = A * p^m + B
    # where m is sequence length, P(m) is survival probability
    
    # Convert to log scale for linear fit
    log_probs = np.log(np.array(survival_probabilities))
    
    # Linear fit: log(P) = log(A) + m*log(p)
    # We assume B ≈ 0 for simplicity
    fit_coeffs = np.polyfit(sequence_lengths, log_probs, 1)
    decay_param = np.exp(fit_coeffs[0])  # p = exp(slope)
    
    # Calculate average gate fidelity
    d = 2 ** num_qubits  # Hilbert space dimension
    avg_gate_fidelity = 1 - (1 - decay_param) * (d - 1) / d
    
    return avg_gate_fidelity, decay_param


def estimate_rb_error_bars(
    sequence_lengths: List[int],
    all_survival_probs: List[List[float]],
    num_qubits: int = 1
) -> Tuple[float, float, float, float]:
    """
    Estimate error bars on RB fidelity measurements.
    
    Args:
        sequence_lengths: List of sequence lengths
        all_survival_probs: List of survival probability lists (one per experiment)
        num_qubits: Number of qubits
        
    Returns:
        Tuple of (fidelity_mean, fidelity_std, decay_mean, decay_std)
    """
    
    fidelities = []
    decay_params = []
    
    for survival_probs in all_survival_probs:
        fidelity, decay = calculate_rb_fidelity(
            sequence_lengths, survival_probs, num_qubits
        )
        fidelities.append(fidelity)
        decay_params.append(decay)
    
    return (
        np.mean(fidelities), np.std(fidelities),
        np.mean(decay_params), np.std(decay_params)
    )


def create_interleaved_rb_pair(
    num_qubits: int,
    sequence_length: int,
    interleaved_gate: str,
    seed: Optional[int] = None
) -> Tuple[JAXCircuit, JAXCircuit]:
    """
    Create a pair of circuits for interleaved RB.
    
    Args:
        num_qubits: Number of qubits
        sequence_length: Length of Clifford sequence
        interleaved_gate: Gate to characterize
        seed: Random seed
        
    Returns:
        Tuple of (reference_circuit, interleaved_circuit)
    """
    
    # Reference circuit (standard RB)
    ref_circuit = create_rb_circuit(
        num_qubits=num_qubits,
        sequence_length=sequence_length,
        rb_type="single" if num_qubits == 1 else "simultaneous",
        seed=seed
    )
    
    # Interleaved circuit (with target gate interleaved)
    int_circuit = create_rb_circuit(
        num_qubits=num_qubits,
        sequence_length=sequence_length,
        rb_type="single" if num_qubits == 1 else "simultaneous",
        interleaved_gate=interleaved_gate,
        seed=seed
    )
    
    return ref_circuit, int_circuit


def analyze_interleaved_rb(
    ref_fidelity: float,
    int_fidelity: float,
    num_qubits: int = 1
) -> float:
    """
    Analyze interleaved RB results to extract gate fidelity.
    
    Args:
        ref_fidelity: Reference (standard RB) fidelity
        int_fidelity: Interleaved RB fidelity
        num_qubits: Number of qubits
        
    Returns:
        Gate fidelity of the interleaved gate
    """
    
    d = 2 ** num_qubits
    
    # Convert average gate fidelities to decay parameters
    p_ref = 1 - (1 - ref_fidelity) * d / (d - 1)
    p_int = 1 - (1 - int_fidelity) * d / (d - 1)
    
    # Gate fidelity is the ratio of decay parameters
    gate_fidelity = p_int / p_ref
    
    return gate_fidelity


def create_rb_benchmarking_suite(
    max_qubits: int = 5,
    max_sequence_length: int = 100,
    num_lengths: int = 10
) -> Dict[str, List[JAXCircuit]]:
    """
    Create a comprehensive RB benchmarking suite.
    
    Args:
        max_qubits: Maximum number of qubits to test
        max_sequence_length: Maximum sequence length
        num_lengths: Number of different sequence lengths
        
    Returns:
        Dictionary mapping RB types to circuit lists
    """
    
    suite = {}
    
    # Generate sequence lengths (exponentially spaced)
    sequence_lengths = np.logspace(
        0, np.log10(max_sequence_length), num_lengths, dtype=int
    ).tolist()
    
    # Single-qubit RB
    suite["single_qubit"] = create_rb_sequence(
        num_qubits=1,
        sequence_lengths=sequence_lengths,
        rb_type="single",
        num_samples=5
    )
    
    # Multi-qubit simultaneous RB
    for n_qubits in range(2, max_qubits + 1):
        key = f"simultaneous_{n_qubits}q"
        suite[key] = create_rb_sequence(
            num_qubits=n_qubits,
            sequence_lengths=sequence_lengths[:num_lengths//2],  # Shorter for multi-qubit
            rb_type="simultaneous",
            num_samples=3
        )
    
    return suite