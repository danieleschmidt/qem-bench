"""Readout error noise model implementation."""

import jax.numpy as jnp
import numpy as np
from typing import List, Any, Dict, Optional, Tuple
from .base import NoiseModel, NoiseChannel


class ReadoutErrorNoiseModel(NoiseModel):
    """
    Readout error noise model for measurement errors.
    
    Models classification errors in quantum state measurements,
    including:
    - Bit-flip errors (0 measured as 1, 1 measured as 0)
    - State preparation and measurement (SPAM) errors
    - Correlated readout errors between qubits
    - Assignment fidelity matrix modeling
    """
    
    def __init__(
        self,
        error_probabilities: Dict[int, Dict[str, float]] = None,
        assignment_matrices: Dict[int, jnp.ndarray] = None,
        correlated_errors: Dict[Tuple[int, ...], float] = None
    ):
        """
        Initialize readout error noise model.
        
        Args:
            error_probabilities: Per-qubit error probabilities 
                                {'qubit': {'0->1': prob, '1->0': prob}}
            assignment_matrices: Per-qubit assignment probability matrices
                               Shape (2, 2) where [i,j] = P(measure j | prepared i)
            correlated_errors: Correlated readout errors between qubit groups
                             {(qubit1, qubit2, ...): error_probability}
        """
        super().__init__("readout_error")
        
        self.error_probabilities = error_probabilities or {}
        self.assignment_matrices = assignment_matrices or {}
        self.correlated_errors = correlated_errors or {}
        
        # Create readout error channels
        self._create_channels()
    
    def _create_channels(self) -> None:
        """Create readout error noise channels."""
        # Single-qubit readout errors
        for qubit in self.error_probabilities.keys():
            error_probs = self.error_probabilities[qubit]
            p01 = error_probs.get('0->1', 0.0)  # False positive
            p10 = error_probs.get('1->0', 0.0)  # False negative
            
            if p01 > 0 or p10 > 0:
                kraus_ops = self._create_readout_kraus_operators(p01, p10)
                channel = NoiseChannel(
                    name=f"readout_error_q{qubit}",
                    kraus_operators=kraus_ops,
                    probability=1.0,
                    qubits=[qubit]
                )
                self.add_channel(channel)
        
        # Assignment matrix based errors
        for qubit, matrix in self.assignment_matrices.items():
            if not jnp.allclose(matrix, jnp.eye(2)):
                kraus_ops = self._assignment_matrix_to_kraus(matrix)
                channel = NoiseChannel(
                    name=f"assignment_error_q{qubit}",
                    kraus_operators=kraus_ops,
                    probability=1.0,
                    qubits=[qubit]
                )
                self.add_channel(channel)
        
        # Correlated readout errors
        for qubits, error_prob in self.correlated_errors.items():
            if error_prob > 0:
                kraus_ops = self._create_correlated_readout_kraus(qubits, error_prob)
                channel = NoiseChannel(
                    name=f"correlated_readout_{'_'.join(map(str, qubits))}",
                    kraus_operators=kraus_ops,
                    probability=1.0,
                    qubits=list(qubits)
                )
                self.add_channel(channel)
    
    def _create_readout_kraus_operators(self, p01: float, p10: float) -> List[jnp.ndarray]:
        """
        Create Kraus operators for single-qubit readout errors.
        
        Args:
            p01: Probability of measuring 1 when prepared in 0
            p10: Probability of measuring 0 when prepared in 1
            
        Returns:
            List of Kraus operators
        """
        # Readout error as quantum channel
        # E0: no error
        # E1: bit flip error
        
        # Identity component (no error)
        K0 = jnp.sqrt(1 - p01 - p10) * jnp.eye(2, dtype=jnp.complex64)
        
        # Error components
        kraus_ops = [K0]
        
        if p01 > 0:
            # |0⟩ measured as |1⟩
            K01 = jnp.sqrt(p01) * jnp.array([
                [0, 0],
                [1, 0]
            ], dtype=jnp.complex64)
            kraus_ops.append(K01)
        
        if p10 > 0:
            # |1⟩ measured as |0⟩ 
            K10 = jnp.sqrt(p10) * jnp.array([
                [0, 1],
                [0, 0]
            ], dtype=jnp.complex64)
            kraus_ops.append(K10)
        
        return kraus_ops
    
    def _assignment_matrix_to_kraus(self, assignment_matrix: jnp.ndarray) -> List[jnp.ndarray]:
        """
        Convert assignment probability matrix to Kraus operators.
        
        Args:
            assignment_matrix: 2x2 matrix where [i,j] = P(measure j | prepared i)
            
        Returns:
            List of Kraus operators
        """
        # For a 2x2 assignment matrix:
        # [[P(0|0), P(1|0)],
        #  [P(0|1), P(1|1)]]
        
        A = assignment_matrix
        
        # Decompose into Kraus operators
        # This is a simplified approach - in general, need proper CPTP decomposition
        K0 = jnp.sqrt(A[0, 0]) * jnp.array([[1, 0], [0, 0]], dtype=jnp.complex64)
        K1 = jnp.sqrt(A[0, 1]) * jnp.array([[0, 0], [1, 0]], dtype=jnp.complex64)
        K2 = jnp.sqrt(A[1, 0]) * jnp.array([[0, 1], [0, 0]], dtype=jnp.complex64)
        K3 = jnp.sqrt(A[1, 1]) * jnp.array([[0, 0], [0, 1]], dtype=jnp.complex64)
        
        return [K0, K1, K2, K3]
    
    def _create_correlated_readout_kraus(
        self, 
        qubits: Tuple[int, ...], 
        error_prob: float
    ) -> List[jnp.ndarray]:
        """
        Create Kraus operators for correlated readout errors.
        
        Args:
            qubits: Tuple of qubit indices involved in correlation
            error_prob: Probability of correlated error
            
        Returns:
            List of Kraus operators for multi-qubit system
        """
        n_qubits = len(qubits)
        dim = 2 ** n_qubits
        
        # Identity (no correlated error)
        K0 = jnp.sqrt(1 - error_prob) * jnp.eye(dim, dtype=jnp.complex64)
        
        # Correlated bit flip (simplified model)
        # Flips all qubits simultaneously
        flip_matrix = jnp.eye(dim, dtype=jnp.complex64)
        
        # Create bit flip pattern for all qubits
        for i in range(dim):
            # Flip all bits
            flipped_i = i ^ ((1 << n_qubits) - 1)  # XOR with all 1s
            flip_matrix = flip_matrix.at[flipped_i, i].set(1.0)
            flip_matrix = flip_matrix.at[i, i].set(0.0)
        
        K1 = jnp.sqrt(error_prob) * flip_matrix
        
        return [K0, K1]
    
    def get_noise_channels(self, circuit: Any) -> List[NoiseChannel]:
        """
        Get readout error channels for circuit measurements.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            List of noise channels to apply to measurements
        """
        channels = []
        
        # Apply readout errors only to measurements
        if hasattr(circuit, 'measurements') and circuit.measurements:
            for measurement in circuit.measurements:
                qubit = measurement.get('qubit')
                if qubit is not None:
                    # Single-qubit readout error
                    if f"readout_error_q{qubit}" in self.channels:
                        channel = self.channels[f"readout_error_q{qubit}"]
                        readout_channel = NoiseChannel(
                            name=f"{channel.name}_meas_{id(measurement)}",
                            kraus_operators=channel.kraus_operators,
                            probability=channel.probability,
                            qubits=[qubit]
                        )
                        channels.append(readout_channel)
                    
                    # Assignment matrix error
                    if f"assignment_error_q{qubit}" in self.channels:
                        channel = self.channels[f"assignment_error_q{qubit}"]
                        assignment_channel = NoiseChannel(
                            name=f"{channel.name}_meas_{id(measurement)}",
                            kraus_operators=channel.kraus_operators,
                            probability=channel.probability,
                            qubits=[qubit]
                        )
                        channels.append(assignment_channel)
            
            # Apply correlated readout errors
            measured_qubits = set()
            for measurement in circuit.measurements:
                qubit = measurement.get('qubit')
                if qubit is not None:
                    measured_qubits.add(qubit)
            
            for qubits, _ in self.correlated_errors.items():
                if all(q in measured_qubits for q in qubits):
                    channel_name = f"correlated_readout_{'_'.join(map(str, qubits))}"
                    if channel_name in self.channels:
                        channel = self.channels[channel_name]
                        corr_channel = NoiseChannel(
                            name=f"{channel.name}_measurement",
                            kraus_operators=channel.kraus_operators,
                            probability=channel.probability,
                            qubits=list(qubits)
                        )
                        channels.append(corr_channel)
        
        return channels
    
    def apply_noise(self, circuit: Any) -> Any:
        """
        Apply readout error noise to circuit.
        
        Args:
            circuit: Clean quantum circuit
            
        Returns:
            Circuit with readout errors applied
        """
        if hasattr(circuit, 'copy'):
            noisy_circuit = circuit.copy()
        else:
            import copy
            noisy_circuit = copy.deepcopy(circuit)
        
        # Add noise markers for simulation
        noisy_circuit._noise_model = self
        noisy_circuit._noise_channels = self.get_noise_channels(circuit)
        
        return noisy_circuit
    
    def set_error_probability(self, qubit: int, p01: float, p10: float) -> None:
        """
        Set readout error probabilities for a specific qubit.
        
        Args:
            qubit: Qubit index
            p01: Probability of measuring 1 when prepared in 0
            p10: Probability of measuring 0 when prepared in 1
        """
        self.error_probabilities[qubit] = {'0->1': p01, '1->0': p10}
        
        # Remove old channel and recreate
        old_channel = f"readout_error_q{qubit}"
        if old_channel in self.channels:
            del self.channels[old_channel]
        
        # Recreate channel
        kraus_ops = self._create_readout_kraus_operators(p01, p10)
        channel = NoiseChannel(
            name=old_channel,
            kraus_operators=kraus_ops,
            probability=1.0,
            qubits=[qubit]
        )
        self.add_channel(channel)
    
    def set_assignment_matrix(self, qubit: int, matrix: jnp.ndarray) -> None:
        """
        Set assignment probability matrix for a specific qubit.
        
        Args:
            qubit: Qubit index
            matrix: 2x2 assignment probability matrix
        """
        self.assignment_matrices[qubit] = matrix
        
        # Remove old channel and recreate
        old_channel = f"assignment_error_q{qubit}"
        if old_channel in self.channels:
            del self.channels[old_channel]
        
        # Recreate channel
        if not jnp.allclose(matrix, jnp.eye(2)):
            kraus_ops = self._assignment_matrix_to_kraus(matrix)
            channel = NoiseChannel(
                name=old_channel,
                kraus_operators=kraus_ops,
                probability=1.0,
                qubits=[qubit]
            )
            self.add_channel(channel)
    
    def get_assignment_fidelity(self, qubit: int) -> float:
        """
        Get assignment fidelity for a specific qubit.
        
        Args:
            qubit: Qubit index
            
        Returns:
            Assignment fidelity (average of diagonal elements)
        """
        if qubit in self.assignment_matrices:
            matrix = self.assignment_matrices[qubit]
            return float((matrix[0, 0] + matrix[1, 1]) / 2)
        elif qubit in self.error_probabilities:
            probs = self.error_probabilities[qubit]
            p01 = probs.get('0->1', 0.0)
            p10 = probs.get('1->0', 0.0)
            return (1 - p01 + 1 - p10) / 2
        else:
            return 1.0  # Perfect readout
    
    def scale_noise(self, factor: float) -> "ReadoutErrorNoiseModel":
        """
        Scale readout error noise by factor.
        
        Args:
            factor: Scaling factor
            
        Returns:
            New noise model with scaled error rates
        """
        # Scale error probabilities
        scaled_probs = {}
        for qubit, probs in self.error_probabilities.items():
            scaled_probs[qubit] = {
                '0->1': min(probs.get('0->1', 0.0) * factor, 1.0),
                '1->0': min(probs.get('1->0', 0.0) * factor, 1.0)
            }
        
        # Scale assignment matrices (move away from identity)
        scaled_matrices = {}
        for qubit, matrix in self.assignment_matrices.items():
            identity = jnp.eye(2)
            scaled_matrix = identity + factor * (matrix - identity)
            # Ensure proper probability matrix
            scaled_matrix = jnp.clip(scaled_matrix, 0, 1)
            # Renormalize rows
            row_sums = jnp.sum(scaled_matrix, axis=1, keepdims=True)
            scaled_matrices[qubit] = scaled_matrix / row_sums
        
        # Scale correlated errors
        scaled_corr = {
            qubits: min(prob * factor, 1.0) 
            for qubits, prob in self.correlated_errors.items()
        }
        
        return ReadoutErrorNoiseModel(
            error_probabilities=scaled_probs,
            assignment_matrices=scaled_matrices,
            correlated_errors=scaled_corr
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            "error_probabilities": {
                str(k): v for k, v in self.error_probabilities.items()
            },
            "assignment_matrices": {
                str(k): v.tolist() for k, v in self.assignment_matrices.items()
            },
            "correlated_errors": {
                "_".join(map(str, k)): float(v) 
                for k, v in self.correlated_errors.items()
            }
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReadoutErrorNoiseModel":
        """Create from dictionary representation."""
        error_probs = {int(k): v for k, v in data["error_probabilities"].items()}
        assignment_mats = {
            int(k): jnp.array(v) for k, v in data["assignment_matrices"].items()
        }
        corr_errors = {
            tuple(map(int, k.split("_"))): v 
            for k, v in data["correlated_errors"].items()
        }
        
        return cls(
            error_probabilities=error_probs,
            assignment_matrices=assignment_mats,
            correlated_errors=corr_errors
        )
    
    def __str__(self) -> str:
        """String representation."""
        lines = [f"ReadoutErrorNoiseModel"]
        if self.error_probabilities:
            lines.append(f"Error probabilities: {self.error_probabilities}")
        if self.assignment_matrices:
            lines.append(f"Assignment matrices: {len(self.assignment_matrices)} qubits")
        if self.correlated_errors:
            lines.append(f"Correlated errors: {self.correlated_errors}")
        lines.append(f"Channels: {len(self.channels)}")
        return "\n".join(lines)


class UniformReadoutErrorModel(ReadoutErrorNoiseModel):
    """
    Simplified readout error model with uniform error rates.
    
    Convenience class for common use cases.
    """
    
    def __init__(
        self,
        num_qubits: int,
        error_rate: float = 0.02,
        asymmetric: bool = False,
        p01_rate: Optional[float] = None,
        p10_rate: Optional[float] = None
    ):
        """
        Initialize uniform readout error model.
        
        Args:
            num_qubits: Number of qubits
            error_rate: Uniform error rate for symmetric errors
            asymmetric: If True, use different rates for 0->1 and 1->0
            p01_rate: Specific rate for 0->1 errors (if asymmetric)
            p10_rate: Specific rate for 1->0 errors (if asymmetric)
        """
        if asymmetric:
            p01 = p01_rate if p01_rate is not None else error_rate
            p10 = p10_rate if p10_rate is not None else error_rate * 0.5
        else:
            p01 = p10 = error_rate
        
        error_probabilities = {}
        for i in range(num_qubits):
            error_probabilities[i] = {'0->1': p01, '1->0': p10}
        
        super().__init__(error_probabilities=error_probabilities)
        
        self.name = "uniform_readout_error"
    
    @classmethod
    def from_fidelity(cls, num_qubits: int, fidelity: float) -> "UniformReadoutErrorModel":
        """
        Create readout error model from assignment fidelity.
        
        Args:
            num_qubits: Number of qubits
            fidelity: Assignment fidelity (0 to 1)
            
        Returns:
            UniformReadoutErrorModel instance
        """
        error_rate = (1 - fidelity) / 2  # Symmetric errors
        return cls(num_qubits=num_qubits, error_rate=error_rate)


class SPAMErrorModel(ReadoutErrorNoiseModel):
    """
    State Preparation and Measurement (SPAM) error model.
    
    Models both state preparation errors and measurement errors.
    """
    
    def __init__(
        self,
        num_qubits: int,
        prep_error_rate: float = 0.01,
        meas_error_rate: float = 0.02,
        correlated_prep_errors: bool = False
    ):
        """
        Initialize SPAM error model.
        
        Args:
            num_qubits: Number of qubits
            prep_error_rate: State preparation error rate
            meas_error_rate: Measurement error rate  
            correlated_prep_errors: Include correlated preparation errors
        """
        # Measurement errors
        error_probabilities = {}
        for i in range(num_qubits):
            error_probabilities[i] = {
                '0->1': meas_error_rate,
                '1->0': meas_error_rate
            }
        
        # Correlated errors (optional)
        correlated_errors = {}
        if correlated_prep_errors:
            # Add pairwise correlations
            for i in range(num_qubits - 1):
                for j in range(i + 1, num_qubits):
                    correlated_errors[(i, j)] = prep_error_rate * 0.1
        
        super().__init__(
            error_probabilities=error_probabilities,
            correlated_errors=correlated_errors
        )
        
        self.name = "spam_error"
        self.prep_error_rate = prep_error_rate
        self.meas_error_rate = meas_error_rate