"""Core Probabilistic Error Cancellation implementation."""

import jax.numpy as jnp
import numpy as np
from typing import List, Union, Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass
import warnings
from abc import ABC, abstractmethod

from ...noise.models.base import NoiseModel, NoiseChannel
from .result import PECResult


@dataclass
class PECConfig:
    """Configuration for Probabilistic Error Cancellation."""
    num_samples: int = 1000
    decomposition_method: str = "pauli"
    optimization_method: str = "least_squares"
    max_quasi_prob: float = 100.0
    convergence_threshold: float = 1e-6
    max_iterations: int = 1000
    sampling_overhead_threshold: float = 1000.0


class DecompositionStrategy(ABC):
    """Abstract base class for quasi-probability decomposition strategies."""
    
    @abstractmethod
    def decompose(
        self, 
        noise_channel: NoiseChannel,
        implementable_channels: List[NoiseChannel]
    ) -> Dict[str, float]:
        """
        Decompose a noise channel into implementable channels with quasi-probabilities.
        
        Args:
            noise_channel: The target noise channel to decompose
            implementable_channels: List of channels that can be implemented
            
        Returns:
            Dictionary mapping channel names to quasi-probabilities
        """
        pass
    
    @abstractmethod 
    def sampling_overhead(self, quasi_probs: Dict[str, float]) -> float:
        """Calculate the sampling overhead for given quasi-probabilities."""
        pass


class PauliDecompositionStrategy(DecompositionStrategy):
    """Decomposition strategy using Pauli channel representation."""
    
    def decompose(
        self,
        noise_channel: NoiseChannel, 
        implementable_channels: List[NoiseChannel]
    ) -> Dict[str, float]:
        """
        Decompose noise channel using Pauli basis.
        
        This implements the standard PEC decomposition where any Pauli channel
        can be decomposed as a linear combination of implementable operations.
        """
        # Extract Pauli transfer matrix representation
        target_ptm = self._kraus_to_pauli_transfer_matrix(noise_channel.kraus_operators)
        
        # Build matrix of implementable channel PTMs
        implementable_ptms = []
        channel_names = []
        
        for channel in implementable_channels:
            ptm = self._kraus_to_pauli_transfer_matrix(channel.kraus_operators)
            implementable_ptms.append(ptm.flatten())
            channel_names.append(channel.name)
        
        # Add identity channel if not present
        if not any("identity" in name.lower() for name in channel_names):
            identity_ptm = jnp.eye(target_ptm.shape[0])
            implementable_ptms.append(identity_ptm.flatten())
            channel_names.append("identity")
        
        # Solve linear system: A @ x = b
        A = jnp.array(implementable_ptms).T
        b = target_ptm.flatten()
        
        try:
            # Use least squares to find quasi-probabilities
            quasi_probs_array = jnp.linalg.lstsq(A, b, rcond=None)[0]
            
            # Convert to dictionary
            quasi_probs = {
                name: float(prob) for name, prob in zip(channel_names, quasi_probs_array)
            }
            
            return quasi_probs
            
        except Exception as e:
            raise ValueError(f"Failed to decompose noise channel: {e}")
    
    def sampling_overhead(self, quasi_probs: Dict[str, float]) -> float:
        """Calculate sampling overhead as sum of absolute quasi-probabilities."""
        return sum(abs(prob) for prob in quasi_probs.values())
    
    def _kraus_to_pauli_transfer_matrix(self, kraus_ops: List[jnp.ndarray]) -> jnp.ndarray:
        """Convert Kraus operators to Pauli Transfer Matrix representation."""
        n_qubits = int(jnp.log2(kraus_ops[0].shape[0]))
        
        # Generate Pauli basis
        pauli_basis = self._generate_pauli_basis(n_qubits)
        
        # Compute PTM elements
        ptm_size = 4**n_qubits
        ptm = jnp.zeros((ptm_size, ptm_size), dtype=jnp.complex64)
        
        for i, pauli_i in enumerate(pauli_basis):
            for j, pauli_j in enumerate(pauli_basis):
                # Compute <P_i | E(P_j) >
                result = jnp.zeros_like(pauli_i)
                
                for K in kraus_ops:
                    result += K @ pauli_j @ jnp.conj(K).T
                
                ptm = ptm.at[i, j].set(jnp.trace(pauli_i @ result) / (2**n_qubits))
        
        return jnp.real(ptm)
    
    def _generate_pauli_basis(self, n_qubits: int) -> List[jnp.ndarray]:
        """Generate n-qubit Pauli basis."""
        pauli_matrices = [
            jnp.array([[1, 0], [0, 1]], dtype=jnp.complex64),  # I
            jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64),  # X
            jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64),  # Y
            jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)   # Z
        ]
        
        basis = []
        for i in range(4**n_qubits):
            # Convert i to base-4 representation
            indices = []
            temp = i
            for _ in range(n_qubits):
                indices.append(temp % 4)
                temp //= 4
            
            # Construct tensor product
            result = pauli_matrices[indices[0]]
            for idx in indices[1:]:
                result = jnp.kron(result, pauli_matrices[idx])
            
            basis.append(result)
        
        return basis


class OptimalDecompositionStrategy(DecompositionStrategy):
    """Optimal decomposition strategy that minimizes sampling overhead."""
    
    def __init__(self, method: str = "least_squares"):
        self.method = method
    
    def decompose(
        self,
        noise_channel: NoiseChannel,
        implementable_channels: List[NoiseChannel]
    ) -> Dict[str, float]:
        """
        Find optimal quasi-probability decomposition minimizing sampling overhead.
        """
        if self.method == "least_squares":
            return self._least_squares_decomposition(noise_channel, implementable_channels)
        elif self.method == "linear_programming":
            return self._linear_programming_decomposition(noise_channel, implementable_channels)
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
    
    def sampling_overhead(self, quasi_probs: Dict[str, float]) -> float:
        """Calculate sampling overhead."""
        return sum(abs(prob) for prob in quasi_probs.values())
    
    def _least_squares_decomposition(
        self,
        noise_channel: NoiseChannel,
        implementable_channels: List[NoiseChannel]
    ) -> Dict[str, float]:
        """Use least squares to find decomposition."""
        # Use Pauli decomposition as starting point
        pauli_strategy = PauliDecompositionStrategy()
        return pauli_strategy.decompose(noise_channel, implementable_channels)
    
    def _linear_programming_decomposition(
        self,
        noise_channel: NoiseChannel, 
        implementable_channels: List[NoiseChannel]
    ) -> Dict[str, float]:
        """Use linear programming to minimize L1 norm (sampling overhead)."""
        # This would require scipy.optimize or similar
        # For now, fall back to least squares
        warnings.warn("Linear programming decomposition not implemented, using least squares")
        return self._least_squares_decomposition(noise_channel, implementable_channels)


class ProbabilisticErrorCancellation:
    """
    Probabilistic Error Cancellation for quantum error mitigation.
    
    PEC works by decomposing noise channels into implementable operations
    with quasi-probabilities, then using importance sampling to estimate
    the noiseless expectation value.
    
    Args:
        noise_model: Noise model characterizing the quantum device
        decomposition_strategy: Strategy for decomposing noise channels
        config: Configuration parameters for PEC
    
    Example:
        >>> noise_model = DepolarizingNoiseModel(p=0.01)
        >>> pec = ProbabilisticErrorCancellation(noise_model=noise_model)
        >>> result = pec.mitigate(circuit, backend, observable)
        >>> print(f"Mitigated value: {result.mitigated_value:.4f}")
    """
    
    def __init__(
        self,
        noise_model: NoiseModel,
        decomposition_strategy: Optional[DecompositionStrategy] = None,
        config: Optional[PECConfig] = None,
        **kwargs
    ):
        self.noise_model = noise_model
        self.decomposition_strategy = decomposition_strategy or PauliDecompositionStrategy()
        self.config = config or PECConfig()
        
        # Additional kwargs for backward compatibility
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Cache for decompositions
        self._decomposition_cache: Dict[str, Dict[str, float]] = {}
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate PEC configuration parameters."""
        if self.config.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        
        if self.config.max_quasi_prob <= 0:
            raise ValueError("max_quasi_prob must be positive")
        
        if not (0 < self.config.convergence_threshold < 1):
            raise ValueError("convergence_threshold must be between 0 and 1")
    
    def mitigate(
        self,
        circuit: Any,
        backend: Any,
        observable: Optional[Any] = None,
        shots: int = 1024,
        **execution_kwargs
    ) -> PECResult:
        """
        Apply probabilistic error cancellation to mitigate errors.
        
        Args:
            circuit: Quantum circuit to execute
            backend: Quantum backend for execution
            observable: Observable to measure (if None, use all-Z)
            shots: Number of measurement shots for raw execution
            **execution_kwargs: Additional arguments for circuit execution
            
        Returns:
            PECResult containing raw and mitigated expectation values
        """
        # Execute raw circuit for comparison
        raw_value = self._execute_circuit(
            circuit, backend, observable, shots, **execution_kwargs
        )
        
        # Get noise channels for the circuit
        noise_channels = self.noise_model.get_noise_channels(circuit)
        
        # Compute quasi-probability decompositions
        decompositions = self._compute_decompositions(noise_channels)
        
        # Check sampling overhead
        total_overhead = self._compute_total_sampling_overhead(decompositions)
        
        if total_overhead > self.config.sampling_overhead_threshold:
            warnings.warn(
                f"High sampling overhead ({total_overhead:.1f}). "
                f"Consider using fewer noise channels or different decomposition strategy."
            )
        
        # Perform importance sampling
        mitigated_value, sampling_data = self._importance_sampling(
            circuit, backend, observable, decompositions, shots, **execution_kwargs
        )
        
        # Calculate error metrics
        error_reduction = self._calculate_error_reduction(
            raw_value, mitigated_value, sampling_data.get("ideal_value")
        )
        
        return PECResult(
            raw_value=raw_value,
            mitigated_value=mitigated_value,
            decompositions=decompositions,
            sampling_overhead=total_overhead,
            sampling_data=sampling_data,
            error_reduction=error_reduction,
            config=self.config
        )
    
    def _execute_circuit(
        self,
        circuit: Any,
        backend: Any,
        observable: Optional[Any],
        shots: int,
        **execution_kwargs
    ) -> float:
        """Execute a single circuit and return expectation value."""
        if hasattr(backend, 'run_with_observable'):
            result = backend.run_with_observable(
                circuit, observable, shots=shots, **execution_kwargs
            )
            return result.expectation_value
        else:
            result = backend.run(circuit, shots=shots, **execution_kwargs)
            return self._extract_expectation_value(result, observable)
    
    def _extract_expectation_value(
        self, 
        result: Any, 
        observable: Optional[Any]
    ) -> float:
        """Extract expectation value from measurement result."""
        if observable is None:
            # Default to all-Z measurement (computational basis)
            if hasattr(result, 'get_counts'):
                counts = result.get_counts()
                total_shots = sum(counts.values())
                
                # Calculate <Z⊗Z⊗...⊗Z> expectation value
                expectation = 0.0
                for bitstring, count in counts.items():
                    # Count number of |1⟩ states
                    num_ones = bitstring.count('1')
                    parity = (-1) ** num_ones
                    expectation += parity * count / total_shots
                
                return expectation
            else:
                raise ValueError("Cannot extract expectation value from result")
        else:
            # Use provided observable
            return observable.expectation_value(result)
    
    def _compute_decompositions(
        self, 
        noise_channels: List[NoiseChannel]
    ) -> Dict[str, Dict[str, float]]:
        """Compute quasi-probability decompositions for all noise channels."""
        decompositions = {}
        
        # Get implementable channels (gates available on the device)
        implementable_channels = self._get_implementable_channels()
        
        for channel in noise_channels:
            channel_key = f"{channel.name}_{hash(str(channel.kraus_operators))}"
            
            if channel_key not in self._decomposition_cache:
                decomposition = self.decomposition_strategy.decompose(
                    channel, implementable_channels
                )
                self._decomposition_cache[channel_key] = decomposition
            
            decompositions[channel.name] = self._decomposition_cache[channel_key]
        
        return decompositions
    
    def _get_implementable_channels(self) -> List[NoiseChannel]:
        """Get list of implementable channels (device gates)."""
        # For now, assume we can implement Pauli gates and identity
        from ...noise.models.base import pauli_kraus_operators
        
        paulis = pauli_kraus_operators()
        implementable = []
        
        for name, matrix in paulis.items():
            channel = NoiseChannel(
                name=name.lower(),
                kraus_operators=[matrix],
                probability=1.0
            )
            implementable.append(channel)
        
        return implementable
    
    def _compute_total_sampling_overhead(
        self, 
        decompositions: Dict[str, Dict[str, float]]
    ) -> float:
        """Compute total sampling overhead for all decompositions."""
        total_overhead = 1.0
        
        for channel_name, quasi_probs in decompositions.items():
            channel_overhead = self.decomposition_strategy.sampling_overhead(quasi_probs)
            total_overhead *= channel_overhead
        
        return total_overhead
    
    def _importance_sampling(
        self,
        circuit: Any,
        backend: Any,
        observable: Optional[Any],
        decompositions: Dict[str, Dict[str, float]],
        shots: int,
        **execution_kwargs
    ) -> Tuple[float, Dict[str, Any]]:
        """Perform importance sampling to estimate mitigated expectation value."""
        estimates = []
        weights = []
        
        for sample_idx in range(self.config.num_samples):
            # Sample implementation sequence
            implementation_sequence, weight = self._sample_implementation_sequence(
                decompositions
            )
            
            # Construct circuit with sampled implementations
            modified_circuit = self._apply_implementation_sequence(
                circuit, implementation_sequence
            )
            
            # Execute modified circuit
            expectation_value = self._execute_circuit(
                modified_circuit, backend, observable, shots, **execution_kwargs
            )
            
            # Store weighted estimate
            estimates.append(expectation_value * weight)
            weights.append(weight)
        
        # Compute final estimate
        estimates = jnp.array(estimates)
        weights = jnp.array(weights)
        
        # Importance sampling estimator
        mitigated_value = float(jnp.mean(estimates))
        
        # Compute statistics
        variance = float(jnp.var(estimates))
        std_error = float(jnp.sqrt(variance / len(estimates)))
        
        sampling_data = {
            "estimates": estimates.tolist(),
            "weights": weights.tolist(),
            "variance": variance,
            "std_error": std_error,
            "num_samples": len(estimates),
            "effective_samples": self._compute_effective_sample_size(weights)
        }
        
        return mitigated_value, sampling_data
    
    def _sample_implementation_sequence(
        self,
        decompositions: Dict[str, Dict[str, float]]
    ) -> Tuple[Dict[str, str], float]:
        """Sample an implementation sequence according to quasi-probabilities."""
        implementation_sequence = {}
        total_weight = 1.0
        
        for channel_name, quasi_probs in decompositions.items():
            # Convert to sampling probabilities (absolute values, normalized)
            abs_probs = {k: abs(v) for k, v in quasi_probs.items()}
            total_abs_prob = sum(abs_probs.values())
            
            if total_abs_prob == 0:
                # Skip if all probabilities are zero
                continue
            
            sampling_probs = {k: v / total_abs_prob for k, v in abs_probs.items()}
            
            # Sample implementation
            implementations = list(sampling_probs.keys())
            probabilities = list(sampling_probs.values())
            
            sampled_impl = np.random.choice(implementations, p=probabilities)
            implementation_sequence[channel_name] = sampled_impl
            
            # Update weight with sign and normalization
            original_prob = quasi_probs[sampled_impl]
            sampling_prob = sampling_probs[sampled_impl]
            
            weight_contribution = original_prob / sampling_prob if sampling_prob > 0 else 0
            total_weight *= weight_contribution
        
        return implementation_sequence, total_weight
    
    def _apply_implementation_sequence(
        self,
        circuit: Any,
        implementation_sequence: Dict[str, str]
    ) -> Any:
        """Apply implementation sequence to modify the circuit."""
        # This is a placeholder - actual implementation depends on circuit representation
        # For now, return the original circuit
        # In practice, this would insert the sampled gates/operations
        modified_circuit = circuit  # Copy and modify
        
        # TODO: Implement actual circuit modification based on implementation_sequence
        # This would involve:
        # 1. Identifying locations where noise channels apply
        # 2. Replacing them with sampled implementations
        # 3. Returning modified circuit
        
        return modified_circuit
    
    def _compute_effective_sample_size(self, weights: jnp.ndarray) -> float:
        """Compute effective sample size for importance sampling."""
        sum_weights = jnp.sum(weights)
        sum_weights_squared = jnp.sum(weights**2)
        
        if sum_weights_squared == 0:
            return 0.0
        
        return float(sum_weights**2 / sum_weights_squared)
    
    def _calculate_error_reduction(
        self,
        raw_value: float,
        mitigated_value: float,
        ideal_value: Optional[float] = None
    ) -> Optional[float]:
        """Calculate error reduction percentage."""
        if ideal_value is None:
            return None
        
        raw_error = abs(raw_value - ideal_value)
        mitigated_error = abs(mitigated_value - ideal_value)
        
        if raw_error == 0:
            return 1.0 if mitigated_error == 0 else 0.0
        
        return (raw_error - mitigated_error) / raw_error


# Convenience function for quick PEC
def probabilistic_error_cancellation(
    circuit: Any,
    backend: Any,
    noise_model: NoiseModel,
    observable: Optional[Any] = None,
    num_samples: int = 1000,
    shots: int = 1024,
    **kwargs
) -> PECResult:
    """
    Convenience function for quick probabilistic error cancellation.
    
    Args:
        circuit: Quantum circuit to execute
        backend: Quantum backend
        noise_model: Noise model for the device
        observable: Observable to measure
        num_samples: Number of importance sampling samples
        shots: Number of shots per circuit execution
        **kwargs: Additional arguments
        
    Returns:
        PECResult with mitigation results
    """
    config = PECConfig(num_samples=num_samples, **kwargs)
    pec = ProbabilisticErrorCancellation(noise_model=noise_model, config=config)
    return pec.mitigate(circuit, backend, observable, shots=shots)