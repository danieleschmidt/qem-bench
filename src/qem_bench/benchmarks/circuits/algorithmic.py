"""Algorithmic benchmark circuit implementations."""

import numpy as np
import jax.numpy as jnp
from typing import List, Optional, Dict, Any, Union, Tuple
from ...jax.circuits import JAXCircuit


def create_qft_circuit(
    num_qubits: int,
    inverse: bool = False,
    approximation_degree: Optional[int] = None
) -> JAXCircuit:
    """
    Create a Quantum Fourier Transform circuit.
    
    Args:
        num_qubits: Number of qubits
        inverse: If True, create inverse QFT
        approximation_degree: Degree of approximation (None for exact QFT)
        
    Returns:
        JAXCircuit implementing QFT
        
    Reference:
        Quantum Fourier Transform and its applications
        Nielsen & Chuang, Section 5.1
    """
    
    name = f"{'iqft' if inverse else 'qft'}_{num_qubits}q"
    if approximation_degree is not None:
        name += f"_approx{approximation_degree}"
    
    circuit = JAXCircuit(num_qubits, name=name)
    
    # QFT implementation
    for i in range(num_qubits):
        # Hadamard gate
        circuit.h(i)
        
        # Controlled phase rotations
        for j in range(i + 1, num_qubits):
            # Phase angle: 2π / 2^(j-i+1)
            k = j - i + 1
            
            # Apply approximation if specified
            if approximation_degree is not None and k > approximation_degree + 1:
                continue
            
            angle = 2 * jnp.pi / (2 ** k)
            if inverse:
                angle = -angle
            
            circuit.cphase(angle, j, i)
    
    # Swap qubits to reverse order (standard QFT convention)
    for i in range(num_qubits // 2):
        circuit.swap(i, num_qubits - 1 - i)
    
    return circuit


def create_vqe_ansatz(
    num_qubits: int,
    layers: int,
    ansatz_type: str = "hardware_efficient",
    parameters: Optional[jnp.ndarray] = None,
    entangling_gate: str = "cx"
) -> JAXCircuit:
    """
    Create a Variational Quantum Eigensolver ansatz circuit.
    
    Args:
        num_qubits: Number of qubits
        layers: Number of ansatz layers
        ansatz_type: Type of ansatz ("hardware_efficient", "uccsd", "real_amplitudes")
        parameters: Circuit parameters (if None, uses random parameters)
        entangling_gate: Entangling gate type ("cx", "cz", "iswap")
        
    Returns:
        JAXCircuit implementing VQE ansatz
        
    Reference:
        Variational Quantum Eigensolver
        Peruzzo et al., Nature Communications 5, 4213 (2014)
    """
    
    circuit = JAXCircuit(num_qubits, name=f"vqe_{ansatz_type}_{num_qubits}q_{layers}l")
    
    if ansatz_type == "hardware_efficient":
        return _create_hardware_efficient_ansatz(
            circuit, layers, parameters, entangling_gate
        )
    elif ansatz_type == "uccsd":
        return _create_uccsd_ansatz(circuit, layers, parameters)
    elif ansatz_type == "real_amplitudes":
        return _create_real_amplitudes_ansatz(
            circuit, layers, parameters, entangling_gate
        )
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")


def _create_hardware_efficient_ansatz(
    circuit: JAXCircuit,
    layers: int,
    parameters: Optional[jnp.ndarray],
    entangling_gate: str
) -> JAXCircuit:
    """Create hardware-efficient ansatz."""
    
    num_qubits = circuit.num_qubits
    
    # Calculate total number of parameters needed
    params_per_layer = 3 * num_qubits  # RY, RZ, RY for each qubit
    total_params = params_per_layer * layers
    
    if parameters is None:
        parameters = np.random.uniform(0, 2*np.pi, total_params)
    
    if len(parameters) != total_params:
        raise ValueError(f"Expected {total_params} parameters, got {len(parameters)}")
    
    param_idx = 0
    
    for layer in range(layers):
        # Single-qubit rotations
        for qubit in range(num_qubits):
            circuit.ry(parameters[param_idx], qubit)
            param_idx += 1
            circuit.rz(parameters[param_idx], qubit)
            param_idx += 1
            circuit.ry(parameters[param_idx], qubit)
            param_idx += 1
        
        # Entangling layer
        if layer < layers - 1:  # No entangling gates after last layer
            _add_entangling_layer(circuit, entangling_gate)
    
    return circuit


def _create_uccsd_ansatz(
    circuit: JAXCircuit,
    layers: int,
    parameters: Optional[jnp.ndarray]
) -> JAXCircuit:
    """Create Unitary Coupled Cluster Singles and Doubles ansatz."""
    
    num_qubits = circuit.num_qubits
    
    # UCCSD requires even number of qubits (pairs of spin orbitals)
    if num_qubits % 2 != 0:
        raise ValueError("UCCSD ansatz requires even number of qubits")
    
    num_electrons = num_qubits // 2  # Assume half-filling
    
    # Calculate number of parameters (singles + doubles excitations)
    num_singles = num_electrons * (num_qubits - num_electrons)
    num_doubles = (num_electrons * (num_electrons - 1) // 2) * \
                  ((num_qubits - num_electrons) * (num_qubits - num_electrons - 1) // 2)
    total_params = (num_singles + num_doubles) * layers
    
    if parameters is None:
        parameters = np.random.uniform(-0.1, 0.1, total_params)
    
    if len(parameters) != total_params:
        raise ValueError(f"Expected {total_params} parameters, got {len(parameters)}")
    
    # Initialize Hartree-Fock state
    for i in range(num_electrons):
        circuit.x(i)
    
    param_idx = 0
    
    for layer in range(layers):
        # Single excitations
        for i in range(num_electrons):
            for a in range(num_electrons, num_qubits):
                theta = parameters[param_idx]
                param_idx += 1
                
                # Single excitation: exp(θ(a†i - i†a))
                _add_single_excitation(circuit, theta, i, a)
        
        # Double excitations
        for i in range(num_electrons):
            for j in range(i + 1, num_electrons):
                for a in range(num_electrons, num_qubits):
                    for b in range(a + 1, num_qubits):
                        theta = parameters[param_idx]
                        param_idx += 1
                        
                        # Double excitation: exp(θ(a†b†ji - j†i†ba))
                        _add_double_excitation(circuit, theta, i, j, a, b)
    
    return circuit


def _create_real_amplitudes_ansatz(
    circuit: JAXCircuit,
    layers: int,
    parameters: Optional[jnp.ndarray],
    entangling_gate: str
) -> JAXCircuit:
    """Create real amplitudes ansatz (RY gates only)."""
    
    num_qubits = circuit.num_qubits
    total_params = num_qubits * layers
    
    if parameters is None:
        parameters = np.random.uniform(0, 2*np.pi, total_params)
    
    if len(parameters) != total_params:
        raise ValueError(f"Expected {total_params} parameters, got {len(parameters)}")
    
    param_idx = 0
    
    for layer in range(layers):
        # RY rotations
        for qubit in range(num_qubits):
            circuit.ry(parameters[param_idx], qubit)
            param_idx += 1
        
        # Entangling layer
        if layer < layers - 1:
            _add_entangling_layer(circuit, entangling_gate)
    
    return circuit


def _add_entangling_layer(circuit: JAXCircuit, entangling_gate: str) -> None:
    """Add entangling layer to circuit."""
    
    num_qubits = circuit.num_qubits
    
    if entangling_gate == "cx":
        # Linear entanglement
        for i in range(num_qubits - 1):
            circuit.cx(i, i + 1)
    elif entangling_gate == "cz":
        for i in range(num_qubits - 1):
            circuit.cz(i, i + 1)
    elif entangling_gate == "circular":
        # Circular entanglement
        for i in range(num_qubits):
            circuit.cx(i, (i + 1) % num_qubits)
    elif entangling_gate == "full":
        # Full entanglement (all pairs)
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                circuit.cx(i, j)
    else:
        raise ValueError(f"Unknown entangling gate: {entangling_gate}")


def _add_single_excitation(circuit: JAXCircuit, theta: float, i: int, a: int) -> None:
    """Add single excitation to UCCSD ansatz."""
    
    # Single excitation: exp(θ(a†i - i†a)) = cos(θ)I + i*sin(θ)(a†i - i†a)
    # Implemented using Givens rotation
    
    circuit.cx(a, i)
    circuit.ry(theta, i)
    circuit.cx(a, i)


def _add_double_excitation(
    circuit: JAXCircuit, 
    theta: float, 
    i: int, j: int, 
    a: int, b: int
) -> None:
    """Add double excitation to UCCSD ansatz."""
    
    # Simplified double excitation implementation
    # Full implementation would require more complex decomposition
    
    circuit.cx(i, j)
    circuit.cx(a, b)
    circuit.ry(theta, j)
    circuit.cx(a, b)
    circuit.cx(i, j)


def create_qaoa_circuit(
    num_qubits: int,
    p_layers: int,
    problem_type: str = "max_cut",
    graph: Optional[List[Tuple[int, int]]] = None,
    parameters: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
) -> JAXCircuit:
    """
    Create a Quantum Approximate Optimization Algorithm circuit.
    
    Args:
        num_qubits: Number of qubits
        p_layers: Number of QAOA layers (p parameter)
        problem_type: Type of problem ("max_cut", "portfolio_optimization")
        graph: Graph edges for Max-Cut (if None, creates complete graph)
        parameters: Tuple of (gamma, beta) parameter arrays
        
    Returns:
        JAXCircuit implementing QAOA
        
    Reference:
        A Quantum Approximate Optimization Algorithm
        Farhi et al., arXiv:1411.4028
    """
    
    circuit = JAXCircuit(num_qubits, name=f"qaoa_{problem_type}_{num_qubits}q_p{p_layers}")
    
    if parameters is None:
        # Random parameters
        gamma = np.random.uniform(0, 2*np.pi, p_layers)
        beta = np.random.uniform(0, np.pi, p_layers)
    else:
        gamma, beta = parameters
        if len(gamma) != p_layers or len(beta) != p_layers:
            raise ValueError(f"Parameter arrays must have length {p_layers}")
    
    # Create graph if not provided
    if graph is None:
        if problem_type == "max_cut":
            # Complete graph
            graph = [(i, j) for i in range(num_qubits) for j in range(i+1, num_qubits)]
        else:
            # Linear chain
            graph = [(i, i+1) for i in range(num_qubits-1)]
    
    # Initial state: equal superposition
    for qubit in range(num_qubits):
        circuit.h(qubit)
    
    # QAOA layers
    for layer in range(p_layers):
        # Problem Hamiltonian evolution (γ)
        if problem_type == "max_cut":
            _add_max_cut_evolution(circuit, graph, gamma[layer])
        elif problem_type == "portfolio_optimization":
            _add_portfolio_evolution(circuit, graph, gamma[layer])
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
        
        # Mixer Hamiltonian evolution (β)
        _add_mixer_evolution(circuit, beta[layer])
    
    return circuit


def _add_max_cut_evolution(
    circuit: JAXCircuit, 
    graph: List[Tuple[int, int]], 
    gamma: float
) -> None:
    """Add Max-Cut problem Hamiltonian evolution."""
    
    for i, j in graph:
        # ZZ interaction: exp(-iγ(1-ZiZj)/2) = exp(-iγ/2)exp(iγZiZj/2)
        circuit.cx(i, j)
        circuit.rz(gamma, j)
        circuit.cx(i, j)


def _add_portfolio_evolution(
    circuit: JAXCircuit, 
    edges: List[Tuple[int, int]], 
    gamma: float
) -> None:
    """Add portfolio optimization Hamiltonian evolution."""
    
    # Simplified portfolio optimization (quadratic terms)
    for i, j in edges:
        # Quadratic interaction term
        circuit.cx(i, j)
        circuit.rz(gamma, j)
        circuit.cx(i, j)
    
    # Linear terms (individual asset costs)
    for qubit in range(circuit.num_qubits):
        circuit.rz(-gamma, qubit)


def _add_mixer_evolution(circuit: JAXCircuit, beta: float) -> None:
    """Add mixer Hamiltonian evolution."""
    
    # Standard mixer: exp(-iβ∑Xi)
    for qubit in range(circuit.num_qubits):
        circuit.rx(2 * beta, qubit)


def create_ghz_circuit(num_qubits: int) -> JAXCircuit:
    """
    Create a GHZ state preparation circuit.
    
    Args:
        num_qubits: Number of qubits
        
    Returns:
        JAXCircuit that prepares |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2
    """
    
    circuit = JAXCircuit(num_qubits, name=f"ghz_{num_qubits}q")
    
    # Create GHZ state
    circuit.h(0)  # Create superposition
    for i in range(1, num_qubits):
        circuit.cx(0, i)  # Entangle with first qubit
    
    return circuit


def create_bell_circuit(bell_state: int = 0) -> JAXCircuit:
    """
    Create a Bell state preparation circuit.
    
    Args:
        bell_state: Bell state index (0-3)
                   0: |Φ+⟩ = (|00⟩ + |11⟩)/√2
                   1: |Φ-⟩ = (|00⟩ - |11⟩)/√2  
                   2: |Ψ+⟩ = (|01⟩ + |10⟩)/√2
                   3: |Ψ-⟩ = (|01⟩ - |10⟩)/√2
                   
    Returns:
        JAXCircuit that prepares the specified Bell state
    """
    
    if not 0 <= bell_state <= 3:
        raise ValueError("Bell state index must be 0, 1, 2, or 3")
    
    bell_names = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]
    circuit = JAXCircuit(2, name=f"bell_{bell_names[bell_state]}")
    
    # Base Bell state |Φ+⟩
    circuit.h(0)
    circuit.cx(0, 1)
    
    # Modify for other Bell states
    if bell_state == 1:  # |Φ-⟩: add phase
        circuit.z(0)
    elif bell_state == 2:  # |Ψ+⟩: flip second qubit
        circuit.x(1)
    elif bell_state == 3:  # |Ψ-⟩: flip and add phase
        circuit.x(1)
        circuit.z(0)
    
    return circuit


def create_w_state_circuit(num_qubits: int) -> JAXCircuit:
    """
    Create a W state preparation circuit.
    
    Args:
        num_qubits: Number of qubits
        
    Returns:
        JAXCircuit that prepares |W⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
    """
    
    if num_qubits < 2:
        raise ValueError("W state requires at least 2 qubits")
    
    circuit = JAXCircuit(num_qubits, name=f"w_{num_qubits}q")
    
    # Recursive W state preparation
    # Start with |1⟩|0...0⟩
    circuit.x(0)
    
    for k in range(1, num_qubits):
        # Rotation angle for uniform superposition
        theta = np.arccos(np.sqrt(1.0 / (k + 1)))
        
        # Controlled rotation
        circuit.ry(theta, k)
        circuit.cx(k, 0)
        
        # Multi-controlled X gate (simplified implementation)
        for j in range(k):
            circuit.cx(j, k)
    
    return circuit


def create_quantum_phase_estimation_circuit(
    num_qubits: int,
    unitary_qubits: int,
    phase: float = 0.0
) -> JAXCircuit:
    """
    Create a Quantum Phase Estimation circuit.
    
    Args:
        num_qubits: Total number of qubits (including ancilla)
        unitary_qubits: Number of qubits for the unitary operator
        phase: Phase to estimate (for testing)
        
    Returns:
        JAXCircuit implementing quantum phase estimation
    """
    
    if unitary_qubits >= num_qubits:
        raise ValueError("Unitary qubits must be less than total qubits")
    
    ancilla_qubits = num_qubits - unitary_qubits
    circuit = JAXCircuit(num_qubits, name=f"qpe_{num_qubits}q")
    
    # Initialize ancilla qubits in superposition
    for i in range(ancilla_qubits):
        circuit.h(i)
    
    # Initialize eigenstate (for testing, use |1⟩ state)
    circuit.x(ancilla_qubits)
    
    # Controlled unitary operations
    for i in range(ancilla_qubits):
        # Apply controlled-U^(2^i)
        power = 2 ** i
        angle = phase * power
        
        # Simplified: use controlled phase gate as unitary
        for target in range(ancilla_qubits, num_qubits):
            circuit.cphase(angle, i, target)
    
    # Inverse QFT on ancilla qubits
    iqft_circuit = create_qft_circuit(ancilla_qubits, inverse=True)
    
    # Apply inverse QFT gates
    for gate in iqft_circuit.gates:
        qubits = gate["qubits"]
        if gate["name"] == "hadamard":
            circuit.h(qubits[0])
        elif gate["name"] == "controlled_phase":
            # Extract angle from matrix (simplified)
            circuit.cphase(gate.get("angle", 0), qubits[0], qubits[1])
        elif gate["name"] == "swap":
            circuit.swap(qubits[0], qubits[1])
    
    return circuit


def create_grover_circuit(
    num_qubits: int,
    marked_items: List[int],
    num_iterations: Optional[int] = None
) -> JAXCircuit:
    """
    Create a Grover search circuit.
    
    Args:
        num_qubits: Number of qubits
        marked_items: List of marked item indices
        num_iterations: Number of Grover iterations (if None, uses optimal)
        
    Returns:
        JAXCircuit implementing Grover's algorithm
    """
    
    if not marked_items:
        raise ValueError("Must specify at least one marked item")
    
    N = 2 ** num_qubits
    M = len(marked_items)
    
    if num_iterations is None:
        # Optimal number of iterations
        num_iterations = int(np.pi * np.sqrt(N / M) / 4)
    
    circuit = JAXCircuit(num_qubits, name=f"grover_{num_qubits}q_{num_iterations}iter")
    
    # Initialize superposition
    for qubit in range(num_qubits):
        circuit.h(qubit)
    
    # Grover iterations
    for _ in range(num_iterations):
        # Oracle: mark target states
        for item in marked_items:
            _add_oracle_marking(circuit, item, num_qubits)
        
        # Diffusion operator (inversion about average)
        _add_diffusion_operator(circuit, num_qubits)
    
    return circuit


def _add_oracle_marking(circuit: JAXCircuit, marked_item: int, num_qubits: int) -> None:
    """Add oracle marking for Grover's algorithm."""
    
    # Convert item index to binary
    binary = format(marked_item, f'0{num_qubits}b')
    
    # Flip qubits that should be 0 in the target state
    for i, bit in enumerate(binary):
        if bit == '0':
            circuit.x(i)
    
    # Multi-controlled Z gate
    if num_qubits == 1:
        circuit.z(0)
    elif num_qubits == 2:
        circuit.cz(0, 1)
    else:
        # Simplified: use multiple CZ gates
        for i in range(num_qubits - 1):
            circuit.cz(i, num_qubits - 1)
    
    # Flip back
    for i, bit in enumerate(binary):
        if bit == '0':
            circuit.x(i)


def _add_diffusion_operator(circuit: JAXCircuit, num_qubits: int) -> None:
    """Add diffusion operator for Grover's algorithm."""
    
    # H† X† H = Z, so inversion about |+⟩ state
    for qubit in range(num_qubits):
        circuit.h(qubit)
        circuit.x(qubit)
    
    # Multi-controlled Z
    if num_qubits == 1:
        circuit.z(0)
    elif num_qubits == 2:
        circuit.cz(0, 1)
    else:
        # Simplified implementation
        for i in range(num_qubits - 1):
            circuit.cz(i, num_qubits - 1)
    
    for qubit in range(num_qubits):
        circuit.x(qubit)
        circuit.h(qubit)


def get_algorithmic_circuit_expected_values() -> Dict[str, Dict[str, float]]:
    """
    Get expected measurement outcomes for algorithmic benchmark circuits.
    
    Returns:
        Dictionary mapping circuit types to expected values
    """
    
    return {
        "bell_phi_plus": {
            "z0": 0.0,
            "z1": 0.0,
            "z0z1": 1.0,
            "x0x1": 1.0
        },
        "bell_phi_minus": {
            "z0": 0.0,
            "z1": 0.0,
            "z0z1": 1.0,
            "x0x1": -1.0
        },
        "ghz_3": {
            "z0": 0.0,
            "z1": 0.0,
            "z2": 0.0,
            "z0z1z2": 0.0,
            "x0x1x2": 1.0
        },
        "w_3": {
            "z0": 1.0/3,
            "z1": 1.0/3,
            "z2": 1.0/3,
            "all_z": -1.0/3
        }
    }