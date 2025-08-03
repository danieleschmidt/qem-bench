"""Quantum state utilities for JAX simulator."""

import jax.numpy as jnp
import numpy as np
from typing import List, Optional, Union


def zero_state(num_qubits: int) -> jnp.ndarray:
    """Create the |0...0⟩ state."""
    state = jnp.zeros(2 ** num_qubits, dtype=jnp.complex64)
    state = state.at[0].set(1.0)
    return state


def one_state(num_qubits: int) -> jnp.ndarray:
    """Create the |1...1⟩ state.""" 
    state = jnp.zeros(2 ** num_qubits, dtype=jnp.complex64)
    state = state.at[-1].set(1.0)
    return state


def plus_state(num_qubits: int) -> jnp.ndarray:
    """Create the |+...+⟩ state (uniform superposition)."""
    state = jnp.ones(2 ** num_qubits, dtype=jnp.complex64) / jnp.sqrt(2 ** num_qubits)
    return state


def minus_state(num_qubits: int) -> jnp.ndarray:
    """Create the |-...-⟩ state."""
    state = jnp.ones(2 ** num_qubits, dtype=jnp.complex64)
    # Apply alternating signs based on Hamming weight
    for i in range(2 ** num_qubits):
        hamming_weight = bin(i).count('1')
        if hamming_weight % 2 == 1:
            state = state.at[i].set(-1.0)
    
    state = state / jnp.sqrt(2 ** num_qubits)
    return state


def computational_basis_state(bitstring: str) -> jnp.ndarray:
    """
    Create computational basis state from bitstring.
    
    Args:
        bitstring: Binary string like "0101"
        
    Returns:
        Quantum state vector
    """
    num_qubits = len(bitstring)
    state_index = int(bitstring, 2)
    
    state = jnp.zeros(2 ** num_qubits, dtype=jnp.complex64)
    state = state.at[state_index].set(1.0)
    return state


def create_bell_state(bell_type: str = "00") -> jnp.ndarray:
    """
    Create Bell state.
    
    Args:
        bell_type: Type of Bell state
            - "00": |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
            - "01": |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2  
            - "10": |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
            - "11": |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
            
    Returns:
        Bell state vector
    """
    state = jnp.zeros(4, dtype=jnp.complex64)
    
    if bell_type == "00":  # |Φ⁺⟩
        state = state.at[0].set(1/jnp.sqrt(2))  # |00⟩
        state = state.at[3].set(1/jnp.sqrt(2))  # |11⟩
    elif bell_type == "01":  # |Ψ⁺⟩
        state = state.at[1].set(1/jnp.sqrt(2))  # |01⟩
        state = state.at[2].set(1/jnp.sqrt(2))  # |10⟩
    elif bell_type == "10":  # |Φ⁻⟩
        state = state.at[0].set(1/jnp.sqrt(2))   # |00⟩
        state = state.at[3].set(-1/jnp.sqrt(2))  # |11⟩
    elif bell_type == "11":  # |Ψ⁻⟩
        state = state.at[1].set(1/jnp.sqrt(2))   # |01⟩
        state = state.at[2].set(-1/jnp.sqrt(2))  # |10⟩
    else:
        raise ValueError(f"Unknown Bell state type: {bell_type}")
    
    return state


def create_ghz_state(num_qubits: int) -> jnp.ndarray:
    """
    Create GHZ state: (|0...0⟩ + |1...1⟩)/√2
    
    Args:
        num_qubits: Number of qubits
        
    Returns:
        GHZ state vector
    """
    state = jnp.zeros(2 ** num_qubits, dtype=jnp.complex64)
    state = state.at[0].set(1/jnp.sqrt(2))          # |0...0⟩
    state = state.at[-1].set(1/jnp.sqrt(2))         # |1...1⟩
    return state


def create_w_state(num_qubits: int) -> jnp.ndarray:
    """
    Create W state: (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
    
    Args:
        num_qubits: Number of qubits
        
    Returns:
        W state vector
    """
    state = jnp.zeros(2 ** num_qubits, dtype=jnp.complex64)
    amplitude = 1 / jnp.sqrt(num_qubits)
    
    # Add amplitude for each single-excitation state
    for i in range(num_qubits):
        state_index = 2 ** (num_qubits - 1 - i)  # 2^(n-1-i)
        state = state.at[state_index].set(amplitude)
    
    return state


def create_random_state(num_qubits: int, seed: Optional[int] = None) -> jnp.ndarray:
    """
    Create random quantum state using Haar measure.
    
    Args:
        num_qubits: Number of qubits
        seed: Random seed
        
    Returns:
        Random quantum state vector
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random complex amplitudes
    real_part = np.random.normal(0, 1, 2 ** num_qubits)
    imag_part = np.random.normal(0, 1, 2 ** num_qubits)
    state = real_part + 1j * imag_part
    
    # Normalize
    state = state / np.linalg.norm(state)
    
    return jnp.array(state, dtype=jnp.complex64)


def create_dicke_state(num_qubits: int, excitations: int) -> jnp.ndarray:
    """
    Create Dicke state |D_n^k⟩ with k excitations among n qubits.
    
    Args:
        num_qubits: Number of qubits (n)
        excitations: Number of excitations (k)
        
    Returns:
        Dicke state vector
    """
    if excitations > num_qubits:
        raise ValueError("Number of excitations cannot exceed number of qubits")
    
    state = jnp.zeros(2 ** num_qubits, dtype=jnp.complex64)
    
    # Find all computational basis states with exactly k ones
    valid_states = []
    for i in range(2 ** num_qubits):
        if bin(i).count('1') == excitations:
            valid_states.append(i)
    
    # Uniform superposition over valid states
    amplitude = 1 / jnp.sqrt(len(valid_states))
    for state_index in valid_states:
        state = state.at[state_index].set(amplitude)
    
    return state


def create_spin_coherent_state(num_qubits: int, theta: float, phi: float) -> jnp.ndarray:
    """
    Create spin coherent state on Bloch sphere.
    
    Args:
        num_qubits: Number of qubits
        theta: Polar angle (0 to π)
        phi: Azimuthal angle (0 to 2π)
        
    Returns:
        Spin coherent state
    """
    # Single-qubit state: cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    alpha = jnp.cos(theta / 2)
    beta = jnp.exp(1j * phi) * jnp.sin(theta / 2)
    
    # Tensor product of identical single-qubit states
    state = jnp.array([alpha, beta], dtype=jnp.complex64)
    
    # Tensor product num_qubits times
    for _ in range(num_qubits - 1):
        state = jnp.kron(state, jnp.array([alpha, beta], dtype=jnp.complex64))
    
    return state


def create_cat_state(num_qubits: int, alpha: complex = 1.0) -> jnp.ndarray:
    """
    Create cat state: (|0...0⟩ + α|1...1⟩)/√(1 + |α|²)
    
    Args:
        num_qubits: Number of qubits
        alpha: Complex amplitude for |1...1⟩ component
        
    Returns:
        Cat state vector
    """
    state = jnp.zeros(2 ** num_qubits, dtype=jnp.complex64)
    
    norm = jnp.sqrt(1 + jnp.abs(alpha) ** 2)
    state = state.at[0].set(1 / norm)            # |0...0⟩
    state = state.at[-1].set(alpha / norm)       # |1...1⟩
    
    return state


def create_thermal_state_approximation(
    num_qubits: int, 
    temperature: float,
    energy_levels: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Create approximation to thermal state using diagonal density matrix.
    
    Args:
        num_qubits: Number of qubits
        temperature: Temperature (in energy units)
        energy_levels: Energy levels (default: 0, 1, 2, ...)
        
    Returns:
        Purified thermal state vector
    """
    if energy_levels is None:
        energy_levels = jnp.arange(2 ** num_qubits, dtype=jnp.float32)
    
    # Boltzmann weights
    if temperature > 0:
        weights = jnp.exp(-energy_levels / temperature)
    else:
        # Ground state at zero temperature
        weights = jnp.zeros_like(energy_levels)
        weights = weights.at[0].set(1.0)
    
    # Normalize
    weights = weights / jnp.sum(weights)
    
    # Create purified state (assuming diagonal thermal state)
    state = jnp.sqrt(weights).astype(jnp.complex64)
    
    return state


# State analysis utilities
def fidelity(state1: jnp.ndarray, state2: jnp.ndarray) -> float:
    """
    Calculate fidelity between two quantum states.
    
    F = |⟨ψ₁|ψ₂⟩|²
    """
    overlap = jnp.conj(state1) @ state2
    return float(jnp.abs(overlap) ** 2)


def trace_distance(state1: jnp.ndarray, state2: jnp.ndarray) -> float:
    """
    Calculate trace distance between two quantum states.
    
    T = (1/2) ||ρ₁ - ρ₂||₁
    
    For pure states: T = √(1 - F)
    """
    fid = fidelity(state1, state2)
    return float(jnp.sqrt(1 - fid))


def von_neumann_entropy(state: jnp.ndarray, base: float = 2) -> float:
    """
    Calculate von Neumann entropy of a pure state (always 0).
    
    For mixed states, would need density matrix.
    """
    # Pure states have zero entropy
    return 0.0


def participation_ratio(state: jnp.ndarray) -> float:
    """
    Calculate participation ratio: 1 / Σᵢ |ψᵢ|⁴
    
    Measures the effective number of basis states.
    """
    probabilities = jnp.abs(state) ** 2
    inverse_pr = jnp.sum(probabilities ** 2)
    return float(1 / inverse_pr)


def entanglement_entropy_bipartite(
    state: jnp.ndarray, 
    subsystem_qubits: List[int],
    total_qubits: int
) -> float:
    """
    Calculate entanglement entropy for bipartite split.
    
    This is a simplified implementation that assumes
    computational basis measurements only.
    """
    # This would require computing reduced density matrix
    # For now, return placeholder
    return 0.0


def state_overlap(state1: jnp.ndarray, state2: jnp.ndarray) -> complex:
    """Calculate overlap ⟨ψ₁|ψ₂⟩ between two states."""
    return complex(jnp.conj(state1) @ state2)


def state_norm(state: jnp.ndarray) -> float:
    """Calculate norm of quantum state."""
    return float(jnp.linalg.norm(state))


def normalize_state(state: jnp.ndarray) -> jnp.ndarray:
    """Normalize quantum state to unit norm."""
    norm = jnp.linalg.norm(state)
    return state / norm if norm > 0 else state


def is_normalized(state: jnp.ndarray, tolerance: float = 1e-10) -> bool:
    """Check if state is normalized."""
    norm = jnp.linalg.norm(state)
    return bool(jnp.abs(norm - 1.0) < tolerance)