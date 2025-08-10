"""Distance metrics for quantum states and processes."""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

from .fidelity import compute_fidelity


def compute_tvd(
    rho: jnp.ndarray,
    sigma: jnp.ndarray,
    validate: bool = True
) -> float:
    """
    Compute trace distance (total variation distance) between quantum states.
    
    T(ρ,σ) = (1/2) * Tr(|ρ - σ|) = (1/2) * ||ρ - σ||₁
    
    Args:
        rho: First quantum state (density matrix)
        sigma: Second quantum state (density matrix)  
        validate: Whether to validate input states
        
    Returns:
        Trace distance between 0 and 1
        
    Reference:
        Nielsen & Chuang, "Quantum Computation and Quantum Information"
    """
    if validate:
        _validate_density_matrix(rho)
        _validate_density_matrix(sigma)
    
    # Compute difference matrix
    diff = rho - sigma
    
    # Compute eigenvalues of the difference
    eigenvals = jnp.linalg.eigvals(diff)
    
    # Trace distance is half the sum of absolute values of eigenvalues
    trace_distance = 0.5 * jnp.sum(jnp.abs(eigenvals))
    
    return float(jnp.real(trace_distance))


def compute_trace_distance(
    rho: jnp.ndarray,
    sigma: jnp.ndarray,
    validate: bool = True
) -> float:
    """Alias for compute_tvd for backward compatibility."""
    return compute_tvd(rho, sigma, validate)


def compute_bures_distance(
    rho: jnp.ndarray,
    sigma: jnp.ndarray,
    validate: bool = True
) -> float:
    """
    Compute Bures distance between quantum states.
    
    D_B(ρ,σ) = √(2(1 - √F(ρ,σ)))
    
    Args:
        rho: First quantum state (density matrix)
        sigma: Second quantum state (density matrix)
        validate: Whether to validate input states
        
    Returns:
        Bures distance between 0 and √2
    """
    fidelity = compute_fidelity(rho, sigma, validate)
    sqrt_fidelity = jnp.sqrt(fidelity)
    bures_distance = jnp.sqrt(2 * (1 - sqrt_fidelity))
    
    return float(bures_distance)


def compute_bures_angle(
    rho: jnp.ndarray,
    sigma: jnp.ndarray,
    validate: bool = True
) -> float:
    """
    Compute Bures angle between quantum states.
    
    A_B(ρ,σ) = arccos(√F(ρ,σ))
    
    Args:
        rho: First quantum state (density matrix)
        sigma: Second quantum state (density matrix)
        validate: Whether to validate input states
        
    Returns:
        Bures angle between 0 and π/2
    """
    fidelity = compute_fidelity(rho, sigma, validate)
    sqrt_fidelity = jnp.sqrt(jnp.clip(fidelity, 0, 1))
    bures_angle = jnp.arccos(sqrt_fidelity)
    
    return float(bures_angle)


def compute_hellinger_distance(
    rho: jnp.ndarray,
    sigma: jnp.ndarray,
    validate: bool = True
) -> float:
    """
    Compute Hellinger distance between quantum states.
    
    H(ρ,σ) = √(1 - Tr(√ρ√σ))
    
    Args:
        rho: First quantum state (density matrix)
        sigma: Second quantum state (density matrix)
        validate: Whether to validate input states
        
    Returns:
        Hellinger distance between 0 and 1
    """
    if validate:
        _validate_density_matrix(rho)
        _validate_density_matrix(sigma)
    
    # Compute square roots of density matrices
    sqrt_rho = _matrix_sqrt(rho)
    sqrt_sigma = _matrix_sqrt(sigma)
    
    # Compute Hellinger affinity
    affinity = jnp.real(jnp.trace(sqrt_rho @ sqrt_sigma))
    
    # Hellinger distance
    hellinger = jnp.sqrt(1 - jnp.clip(affinity, 0, 1))
    
    return float(hellinger)


def compute_quantum_js_divergence(
    rho: jnp.ndarray,
    sigma: jnp.ndarray,
    validate: bool = True
) -> float:
    """
    Compute quantum Jensen-Shannon divergence between quantum states.
    
    JS(ρ,σ) = S((ρ+σ)/2) - (S(ρ)+S(σ))/2
    
    Args:
        rho: First quantum state (density matrix)
        sigma: Second quantum state (density matrix)
        validate: Whether to validate input states
        
    Returns:
        Quantum JS divergence (non-negative)
    """
    if validate:
        _validate_density_matrix(rho)
        _validate_density_matrix(sigma)
    
    # Compute mixture
    mixture = (rho + sigma) / 2
    
    # Compute von Neumann entropies
    s_mixture = _von_neumann_entropy(mixture)
    s_rho = _von_neumann_entropy(rho)
    s_sigma = _von_neumann_entropy(sigma)
    
    js_divergence = s_mixture - (s_rho + s_sigma) / 2
    
    return float(jnp.maximum(js_divergence, 0.0))


def compute_diamond_distance(
    channel1: Any,
    channel2: Any,
    dimension: int,
    num_samples: int = 1000
) -> float:
    """
    Compute diamond distance between quantum channels (approximate).
    
    This is an approximate implementation using random state sampling.
    For exact computation, semidefinite programming is required.
    
    Args:
        channel1: First quantum channel
        channel2: Second quantum channel
        dimension: Hilbert space dimension
        num_samples: Number of random states to sample
        
    Returns:
        Approximate diamond distance
    """
    max_distance = 0.0
    
    for _ in range(num_samples):
        # Generate random pure state on extended space
        extended_dim = dimension ** 2
        psi = _random_pure_state(int(np.log2(extended_dim)))
        rho = jnp.outer(psi, jnp.conj(psi))
        
        # Partially trace to get input state
        input_state = _partial_trace(rho, [0], [2, 2])
        
        # Apply channels
        output1 = _apply_channel(channel1, input_state)
        output2 = _apply_channel(channel2, input_state)
        
        # Compute trace distance
        distance = compute_tvd(output1, output2, validate=False)
        max_distance = max(max_distance, distance)
    
    return max_distance


@dataclass
class DistanceCalculator:
    """
    Calculator for various quantum distance metrics.
    
    Provides methods for computing different distance measures
    between quantum states and processes.
    """
    
    validate_inputs: bool = True
    numerical_precision: float = 1e-12
    
    def trace_distance(
        self,
        state1: Union[jnp.ndarray, Dict[str, float]],
        state2: Union[jnp.ndarray, Dict[str, float]]
    ) -> float:
        """Compute trace distance between two quantum states."""
        rho1 = self._ensure_density_matrix(state1)
        rho2 = self._ensure_density_matrix(state2)
        
        return compute_tvd(rho1, rho2, validate=self.validate_inputs)
    
    def bures_distance(
        self,
        state1: Union[jnp.ndarray, Dict[str, float]],
        state2: Union[jnp.ndarray, Dict[str, float]]
    ) -> float:
        """Compute Bures distance between two quantum states."""
        rho1 = self._ensure_density_matrix(state1)
        rho2 = self._ensure_density_matrix(state2)
        
        return compute_bures_distance(rho1, rho2, validate=self.validate_inputs)
    
    def bures_angle(
        self,
        state1: Union[jnp.ndarray, Dict[str, float]],
        state2: Union[jnp.ndarray, Dict[str, float]]
    ) -> float:
        """Compute Bures angle between two quantum states."""
        rho1 = self._ensure_density_matrix(state1)
        rho2 = self._ensure_density_matrix(state2)
        
        return compute_bures_angle(rho1, rho2, validate=self.validate_inputs)
    
    def hellinger_distance(
        self,
        state1: Union[jnp.ndarray, Dict[str, float]],
        state2: Union[jnp.ndarray, Dict[str, float]]
    ) -> float:
        """Compute Hellinger distance between two quantum states."""
        rho1 = self._ensure_density_matrix(state1)
        rho2 = self._ensure_density_matrix(state2)
        
        return compute_hellinger_distance(rho1, rho2, validate=self.validate_inputs)
    
    def js_divergence(
        self,
        state1: Union[jnp.ndarray, Dict[str, float]],
        state2: Union[jnp.ndarray, Dict[str, float]]
    ) -> float:
        """Compute quantum Jensen-Shannon divergence between two quantum states."""
        rho1 = self._ensure_density_matrix(state1)
        rho2 = self._ensure_density_matrix(state2)
        
        return compute_quantum_js_divergence(rho1, rho2, validate=self.validate_inputs)
    
    def compute_batch_distances(
        self,
        reference_state: Union[jnp.ndarray, Dict[str, float]],
        test_states: list,
        metric: str = "trace_distance"
    ) -> jnp.ndarray:
        """
        Compute distances between reference state and batch of test states.
        
        Args:
            reference_state: Reference quantum state
            test_states: List of test states
            metric: Distance metric to compute
            
        Returns:
            Array of computed distances
        """
        distances = []
        
        for test_state in test_states:
            if metric == "trace_distance":
                value = self.trace_distance(reference_state, test_state)
            elif metric == "bures_distance":
                value = self.bures_distance(reference_state, test_state)
            elif metric == "bures_angle":
                value = self.bures_angle(reference_state, test_state)
            elif metric == "hellinger_distance":
                value = self.hellinger_distance(reference_state, test_state)
            elif metric == "js_divergence":
                value = self.js_divergence(reference_state, test_state)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            distances.append(value)
        
        return jnp.array(distances)
    
    def _ensure_density_matrix(
        self,
        state: Union[jnp.ndarray, Dict[str, float]]
    ) -> jnp.ndarray:
        """Convert various state representations to density matrix."""
        if isinstance(state, dict):
            # Measurement counts dictionary
            return self._counts_to_density_matrix(state)
        elif isinstance(state, (np.ndarray, jnp.ndarray)):
            state = jnp.array(state)
            if len(state.shape) == 1:
                # State vector
                return jnp.outer(state, jnp.conj(state))
            else:
                # Density matrix
                return state
        else:
            raise ValueError(f"Unknown state format: {type(state)}")
    
    def _counts_to_density_matrix(self, counts: Dict[str, float]) -> jnp.ndarray:
        """Convert measurement counts to density matrix."""
        # Determine number of qubits
        if not counts:
            raise ValueError("Empty counts dictionary")
        
        n_qubits = len(list(counts.keys())[0])
        dim = 2 ** n_qubits
        total_shots = sum(counts.values())
        
        # Build density matrix from measurement statistics
        rho = jnp.zeros((dim, dim), dtype=jnp.complex64)
        
        for bitstring, count in counts.items():
            # Convert bitstring to computational basis state index
            state_index = int(bitstring, 2)
            probability = count / total_shots
            
            # Add contribution to density matrix (assuming computational basis measurement)
            rho = rho.at[state_index, state_index].add(probability)
        
        return rho


# Helper functions
def _validate_density_matrix(rho: jnp.ndarray, tol: float = 1e-8) -> None:
    """Validate that matrix is a valid density matrix."""
    # Check if Hermitian
    if not jnp.allclose(rho, jnp.conj(rho.T), atol=tol):
        raise ValueError("Density matrix must be Hermitian")
    
    # Check if positive semidefinite
    eigenvals = jnp.real(jnp.linalg.eigvals(rho))
    if jnp.any(eigenvals < -tol):
        raise ValueError("Density matrix must be positive semidefinite")
    
    # Check if trace is 1
    trace = jnp.real(jnp.trace(rho))
    if not jnp.allclose(trace, 1.0, atol=tol):
        raise ValueError(f"Density matrix trace must be 1, got {trace}")


def _matrix_sqrt(A: jnp.ndarray) -> jnp.ndarray:
    """Compute principal square root of positive semidefinite matrix."""
    eigenvals, eigenvecs = jnp.linalg.eigh(A)
    
    # Ensure non-negative eigenvalues
    eigenvals = jnp.maximum(eigenvals, 0.0)
    sqrt_eigenvals = jnp.sqrt(eigenvals)
    
    return eigenvecs @ jnp.diag(sqrt_eigenvals) @ jnp.conj(eigenvecs.T)


def _von_neumann_entropy(rho: jnp.ndarray, base: float = 2) -> float:
    """Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ)."""
    eigenvals = jnp.real(jnp.linalg.eigvals(rho))
    
    # Remove zero eigenvalues to avoid log(0)
    eigenvals = eigenvals[eigenvals > 1e-12]
    
    if len(eigenvals) == 0:
        return 0.0
    
    entropy = -jnp.sum(eigenvals * jnp.log(eigenvals) / jnp.log(base))
    
    return float(jnp.maximum(entropy, 0.0))


def _random_pure_state(n_qubits: int) -> jnp.ndarray:
    """Generate random pure state using Haar measure."""
    dim = 2 ** n_qubits
    
    # Generate random complex vector
    real_part = np.random.normal(0, 1, dim)
    imag_part = np.random.normal(0, 1, dim)
    psi = real_part + 1j * imag_part
    
    # Normalize
    psi = psi / np.linalg.norm(psi)
    
    return jnp.array(psi, dtype=jnp.complex64)


def _partial_trace(rho: jnp.ndarray, traced_systems: list, dimensions: list) -> jnp.ndarray:
    """Compute partial trace of density matrix (simplified implementation)."""
    # This is a simplified implementation for demonstration
    # In practice, would need proper tensor reshaping and tracing
    return rho


def _apply_channel(channel: Any, state: jnp.ndarray) -> jnp.ndarray:
    """Apply quantum channel to state (simplified implementation)."""
    # This would need proper implementation based on channel representation
    if hasattr(channel, 'shape') and len(channel.shape) == 2:
        # Treat as unitary operation
        if channel.shape[0] == state.shape[0]:
            return channel @ state @ jnp.conj(channel.T)
    
    # Fallback: return state unchanged
    return state


# Convenience functions for backward compatibility
def tvd(rho: jnp.ndarray, sigma: jnp.ndarray) -> float:
    """Shorthand for compute_tvd."""
    return compute_tvd(rho, sigma)


def trace_distance(rho: jnp.ndarray, sigma: jnp.ndarray) -> float:
    """Shorthand for compute_trace_distance."""
    return compute_trace_distance(rho, sigma)


def bures_distance(rho: jnp.ndarray, sigma: jnp.ndarray) -> float:
    """Shorthand for compute_bures_distance."""
    return compute_bures_distance(rho, sigma)


def hellinger_distance(rho: jnp.ndarray, sigma: jnp.ndarray) -> float:
    """Shorthand for compute_hellinger_distance."""
    return compute_hellinger_distance(rho, sigma)