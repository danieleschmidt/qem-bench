"""Fidelity metrics for quantum state and process comparison."""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass


def compute_fidelity(
    rho: jnp.ndarray,
    sigma: jnp.ndarray,
    validate: bool = True
) -> float:
    """
    Compute quantum state fidelity F(ρ,σ) = Tr(√(√ρ σ √ρ)).
    
    Args:
        rho: First quantum state (density matrix)
        sigma: Second quantum state (density matrix)  
        validate: Whether to validate input states
        
    Returns:
        Fidelity between 0 and 1
        
    Reference:
        Nielsen & Chuang, "Quantum Computation and Quantum Information"
    """
    if validate:
        _validate_density_matrix(rho)
        _validate_density_matrix(sigma)
    
    # For pure states, fidelity simplifies
    if _is_pure_state(rho) and _is_pure_state(sigma):
        return float(jnp.abs(jnp.trace(rho @ sigma))**2)
    
    # General fidelity formula
    sqrt_rho = _matrix_sqrt(rho)
    M = sqrt_rho @ sigma @ sqrt_rho
    sqrt_M = _matrix_sqrt(M)
    fidelity = jnp.real(jnp.trace(sqrt_M))
    
    return float(jnp.clip(fidelity, 0.0, 1.0))


def compute_process_fidelity(
    chi1: jnp.ndarray,
    chi2: jnp.ndarray,
    representation: str = "chi"
) -> float:
    """
    Compute process fidelity between two quantum channels.
    
    Args:
        chi1: First process matrix
        chi2: Second process matrix
        representation: Process representation ("chi", "kraus", "choi")
        
    Returns:
        Process fidelity between 0 and 1
    """
    if representation == "chi":
        # χ-matrix representation: F = Tr(χ₁χ₂)
        fidelity = jnp.real(jnp.trace(chi1 @ chi2))
    elif representation == "choi":
        # Choi matrix representation
        fidelity = compute_fidelity(chi1, chi2, validate=False)
    else:
        raise ValueError(f"Unknown representation: {representation}")
    
    return float(jnp.clip(fidelity, 0.0, 1.0))


def compute_average_gate_fidelity(
    ideal_channel: Any,
    noisy_channel: Any,
    num_samples: int = 1000,
    seed: Optional[int] = None
) -> float:
    """
    Compute average gate fidelity over Haar-random input states.
    
    F_avg = ∫ F(E(ψ), E_ideal(ψ)) dψ
    
    Args:
        ideal_channel: Ideal quantum channel
        noisy_channel: Noisy quantum channel
        num_samples: Number of random states to sample
        seed: Random seed for reproducibility
        
    Returns:
        Average gate fidelity
    """
    if seed is not None:
        np.random.seed(seed)
    
    total_fidelity = 0.0
    n_qubits = int(np.log2(ideal_channel.shape[0])) if hasattr(ideal_channel, 'shape') else 1
    
    for _ in range(num_samples):
        # Generate random pure state
        psi = _random_pure_state(n_qubits)
        rho = jnp.outer(psi, jnp.conj(psi))
        
        # Apply channels
        ideal_output = _apply_channel(ideal_channel, rho)
        noisy_output = _apply_channel(noisy_channel, rho)
        
        # Compute fidelity
        fidelity = compute_fidelity(ideal_output, noisy_output, validate=False)
        total_fidelity += fidelity
    
    return total_fidelity / num_samples


@dataclass
class StateFidelityCalculator:
    """
    Calculator for various state fidelity metrics.
    
    Provides methods for computing different types of fidelities
    and related metrics for quantum states.
    """
    
    validate_inputs: bool = True
    numerical_precision: float = 1e-12
    
    def fidelity(
        self,
        state1: Union[jnp.ndarray, Dict[str, float]],
        state2: Union[jnp.ndarray, Dict[str, float]]
    ) -> float:
        """Compute fidelity between two quantum states."""
        rho1 = self._ensure_density_matrix(state1)
        rho2 = self._ensure_density_matrix(state2)
        
        return compute_fidelity(rho1, rho2, validate=self.validate_inputs)
    
    def root_fidelity(
        self,
        state1: Union[jnp.ndarray, Dict[str, float]],
        state2: Union[jnp.ndarray, Dict[str, float]]
    ) -> float:
        """Compute square root of fidelity."""
        return jnp.sqrt(self.fidelity(state1, state2))
    
    def infidelity(
        self,
        state1: Union[jnp.ndarray, Dict[str, float]],
        state2: Union[jnp.ndarray, Dict[str, float]]
    ) -> float:
        """Compute infidelity (1 - fidelity)."""
        return 1.0 - self.fidelity(state1, state2)
    
    def bures_distance(
        self,
        state1: Union[jnp.ndarray, Dict[str, float]],
        state2: Union[jnp.ndarray, Dict[str, float]]
    ) -> float:
        """Compute Bures distance between states."""
        F = self.fidelity(state1, state2)
        return jnp.sqrt(2 * (1 - jnp.sqrt(F)))
    
    def bures_angle(
        self,
        state1: Union[jnp.ndarray, Dict[str, float]],
        state2: Union[jnp.ndarray, Dict[str, float]]
    ) -> float:
        """Compute Bures angle between states."""
        F = self.fidelity(state1, state2)
        return jnp.arccos(jnp.sqrt(F))
    
    def compute_batch_fidelities(
        self,
        reference_state: Union[jnp.ndarray, Dict[str, float]],
        test_states: list,
        metric: str = "fidelity"
    ) -> jnp.ndarray:
        """
        Compute fidelities between reference state and batch of test states.
        
        Args:
            reference_state: Reference quantum state
            test_states: List of test states
            metric: Metric to compute ("fidelity", "infidelity", "bures_distance")
            
        Returns:
            Array of computed metrics
        """
        metrics = []
        
        for test_state in test_states:
            if metric == "fidelity":
                value = self.fidelity(reference_state, test_state)
            elif metric == "infidelity":
                value = self.infidelity(reference_state, test_state)
            elif metric == "bures_distance":
                value = self.bures_distance(reference_state, test_state)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            metrics.append(value)
        
        return jnp.array(metrics)
    
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


def _is_pure_state(rho: jnp.ndarray, tol: float = 1e-8) -> bool:
    """Check if state is pure (Tr(ρ²) ≈ 1)."""
    purity = jnp.real(jnp.trace(rho @ rho))
    return jnp.abs(purity - 1.0) < tol


def _matrix_sqrt(A: jnp.ndarray) -> jnp.ndarray:
    """Compute principal square root of positive semidefinite matrix."""
    eigenvals, eigenvecs = jnp.linalg.eigh(A)
    
    # Ensure non-negative eigenvalues
    eigenvals = jnp.maximum(eigenvals, 0.0)
    sqrt_eigenvals = jnp.sqrt(eigenvals)
    
    return eigenvecs @ jnp.diag(sqrt_eigenvals) @ jnp.conj(eigenvecs.T)


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


def _apply_channel(channel: Any, state: jnp.ndarray) -> jnp.ndarray:
    """Apply quantum channel to state (simplified implementation)."""
    # This would need proper implementation based on channel representation
    if hasattr(channel, 'shape') and len(channel.shape) == 2:
        # Treat as unitary operation
        if channel.shape[0] == state.shape[0]:
            return channel @ state @ jnp.conj(channel.T)
    
    # Fallback: return state unchanged
    return state