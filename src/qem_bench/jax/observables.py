"""Observable definitions for expectation value measurements."""

import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Union, Optional
from abc import ABC, abstractmethod

from .gates import pauli_gate


class Observable(ABC):
    """Abstract base class for quantum observables."""
    
    def __init__(self, name: str):
        self.name = name
    
    @property
    @abstractmethod
    def matrix(self) -> jnp.ndarray:
        """Observable matrix representation."""
        pass
    
    @abstractmethod
    def expectation_value(self, statevector: jnp.ndarray) -> float:
        """Calculate expectation value ⟨ψ|O|ψ⟩."""
        pass


class PauliObservable(Observable):
    """
    Pauli string observable (tensor product of Pauli operators).
    
    Example:
        >>> obs = PauliObservable("XYZ", [0, 1, 2])  # X₀ ⊗ Y₁ ⊗ Z₂
        >>> obs = PauliObservable("ZZ", [0, 2])      # Z₀ ⊗ I₁ ⊗ Z₂
    """
    
    def __init__(
        self, 
        pauli_string: str, 
        qubits: List[int], 
        coefficient: complex = 1.0,
        name: Optional[str] = None
    ):
        """
        Initialize Pauli observable.
        
        Args:
            pauli_string: String of Pauli operators (e.g., "XYZ", "ZZ")
            qubits: Qubits on which Pauli operators act
            coefficient: Complex coefficient for the observable
            name: Optional name for the observable
        """
        if len(pauli_string) != len(qubits):
            raise ValueError("Length of pauli_string must match length of qubits")
        
        self.pauli_string = pauli_string.upper()
        self.qubits = qubits
        self.coefficient = coefficient
        self.num_qubits = max(qubits) + 1 if qubits else 1
        
        # Validate Pauli string
        valid_paulis = set("IXYZ")
        if not all(p in valid_paulis for p in self.pauli_string):
            raise ValueError(f"Invalid Pauli string: {pauli_string}")
        
        if name is None:
            name = f"{self.pauli_string}_{''.join(map(str, qubits))}"
        
        super().__init__(name)
        
        # Cache the matrix
        self._matrix = None
    
    @property
    def matrix(self) -> jnp.ndarray:
        """Construct the full observable matrix."""
        if self._matrix is None:
            self._matrix = self._construct_matrix()
        return self._matrix
    
    def _construct_matrix(self) -> jnp.ndarray:
        """Construct the tensor product matrix for the Pauli observable."""
        # Create identity matrix for all qubits
        full_matrix = jnp.eye(2 ** self.num_qubits, dtype=jnp.complex64)
        
        # Apply each Pauli operator
        for pauli_char, qubit in zip(self.pauli_string, self.qubits):
            if pauli_char != "I":  # Skip identity
                pauli_op = pauli_gate(pauli_char)
                pauli_matrix = pauli_op.matrix
                
                # Apply single-qubit Pauli to full matrix
                full_matrix = self._apply_single_qubit_to_full(
                    full_matrix, pauli_matrix, qubit
                )
        
        return self.coefficient * full_matrix
    
    def _apply_single_qubit_to_full(
        self, 
        full_matrix: jnp.ndarray, 
        single_qubit_op: jnp.ndarray, 
        target_qubit: int
    ) -> jnp.ndarray:
        """Apply single-qubit operator to specific qubit in full matrix."""
        # Reshape for tensor operations
        shape = [2] * self.num_qubits
        full_reshaped = full_matrix.reshape(shape + shape)
        
        # Apply operator using tensor contraction
        full_reshaped = jnp.tensordot(
            single_qubit_op, full_reshaped,
            axes=([1], [target_qubit])
        )
        
        # Move result axis back to correct position
        full_reshaped = jnp.moveaxis(full_reshaped, 0, target_qubit)
        
        return full_reshaped.reshape(full_matrix.shape)
    
    def expectation_value(self, statevector: jnp.ndarray) -> float:
        """Calculate expectation value ⟨ψ|O|ψ⟩."""
        result = jnp.conj(statevector) @ self.matrix @ statevector
        return float(result.real)
    
    def __mul__(self, other: Union[complex, "PauliObservable"]) -> "PauliObservable":
        """Multiply observable by scalar or another Pauli observable."""
        if isinstance(other, (int, float, complex)):
            return PauliObservable(
                self.pauli_string,
                self.qubits,
                self.coefficient * other,
                name=f"{other}*{self.name}"
            )
        elif isinstance(other, PauliObservable):
            # Tensor product of Pauli observables
            if set(self.qubits) & set(other.qubits):
                raise ValueError("Cannot multiply overlapping Pauli observables")
            
            combined_string = self.pauli_string + other.pauli_string
            combined_qubits = self.qubits + other.qubits
            combined_coeff = self.coefficient * other.coefficient
            
            return PauliObservable(
                combined_string,
                combined_qubits,
                combined_coeff,
                name=f"{self.name}*{other.name}"
            )
        else:
            raise TypeError(f"Cannot multiply PauliObservable by {type(other)}")
    
    def __rmul__(self, other: Union[complex, int, float]) -> "PauliObservable":
        """Right multiplication by scalar."""
        return self.__mul__(other)
    
    def __add__(self, other: "PauliObservable") -> "PauliSumObservable":
        """Add two Pauli observables."""
        if isinstance(other, PauliObservable):
            return PauliSumObservable([self, other])
        else:
            raise TypeError(f"Cannot add PauliObservable and {type(other)}")
    
    def __str__(self) -> str:
        """String representation."""
        if self.coefficient == 1.0:
            return f"{self.pauli_string}_{''.join(map(str, self.qubits))}"
        else:
            return f"({self.coefficient})*{self.pauli_string}_{''.join(map(str, self.qubits))}"


class PauliSumObservable(Observable):
    """
    Sum of Pauli observables (Hamiltonian).
    
    Example:
        >>> h = PauliSumObservable([
        ...     PauliObservable("Z", [0], 0.5),
        ...     PauliObservable("Z", [1], 0.5),
        ...     PauliObservable("XX", [0, 1], 0.3)
        ... ])
    """
    
    def __init__(self, terms: List[PauliObservable], name: Optional[str] = None):
        """
        Initialize sum of Pauli observables.
        
        Args:
            terms: List of PauliObservable terms
            name: Optional name for the observable
        """
        self.terms = terms
        
        if not terms:
            raise ValueError("At least one term required")
        
        # Determine number of qubits
        all_qubits = set()
        for term in terms:
            all_qubits.update(term.qubits)
        self.num_qubits = max(all_qubits) + 1 if all_qubits else 1
        
        if name is None:
            name = f"PauliSum_{len(terms)}_terms"
        
        super().__init__(name)
        
        # Cache the matrix
        self._matrix = None
    
    @property
    def matrix(self) -> jnp.ndarray:
        """Construct the full observable matrix as sum of Pauli terms."""
        if self._matrix is None:
            self._matrix = jnp.zeros((2 ** self.num_qubits, 2 ** self.num_qubits), dtype=jnp.complex64)
            for term in self.terms:
                # Ensure each term has the correct number of qubits
                term_matrix = self._extend_term_matrix(term)
                self._matrix += term_matrix
        return self._matrix
    
    def _extend_term_matrix(self, term: PauliObservable) -> jnp.ndarray:
        """Extend a Pauli term to full system size."""
        if term.num_qubits == self.num_qubits:
            return term.matrix
        
        # Create extended Pauli string with identities
        extended_string = "I" * self.num_qubits
        extended_qubits = list(range(self.num_qubits))
        
        # Replace with actual Pauli operators
        for pauli_char, qubit in zip(term.pauli_string, term.qubits):
            extended_string = extended_string[:qubit] + pauli_char + extended_string[qubit+1:]
        
        # Create extended observable
        extended_obs = PauliObservable(extended_string, extended_qubits, term.coefficient)
        return extended_obs.matrix
    
    def expectation_value(self, statevector: jnp.ndarray) -> float:
        """Calculate expectation value ⟨ψ|H|ψ⟩."""
        result = jnp.conj(statevector) @ self.matrix @ statevector
        return float(result.real)
    
    def __add__(self, other: Union[PauliObservable, "PauliSumObservable"]) -> "PauliSumObservable":
        """Add observables."""
        if isinstance(other, PauliObservable):
            return PauliSumObservable(self.terms + [other])
        elif isinstance(other, PauliSumObservable):
            return PauliSumObservable(self.terms + other.terms)
        else:
            raise TypeError(f"Cannot add PauliSumObservable and {type(other)}")
    
    def __mul__(self, scalar: Union[complex, int, float]) -> "PauliSumObservable":
        """Multiply by scalar."""
        scaled_terms = [scalar * term for term in self.terms]
        return PauliSumObservable(scaled_terms, name=f"{scalar}*{self.name}")
    
    def __rmul__(self, scalar: Union[complex, int, float]) -> "PauliSumObservable":
        """Right multiplication by scalar."""
        return self.__mul__(scalar)


class ZObservable(PauliObservable):
    """Convenience class for Z observable on single qubit."""
    
    def __init__(self, qubit: int):
        super().__init__("Z", [qubit], name=f"Z_{qubit}")


class XObservable(PauliObservable):
    """Convenience class for X observable on single qubit."""
    
    def __init__(self, qubit: int):
        super().__init__("X", [qubit], name=f"X_{qubit}")


class YObservable(PauliObservable):
    """Convenience class for Y observable on single qubit."""
    
    def __init__(self, qubit: int):
        super().__init__("Y", [qubit], name=f"Y_{qubit}")


# Utility functions
def create_pauli_observable(
    pauli_string: str, 
    qubits: Optional[List[int]] = None,
    coefficient: complex = 1.0
) -> PauliObservable:
    """
    Create Pauli observable from string.
    
    Args:
        pauli_string: Pauli string (e.g., "XYZ")
        qubits: Qubits to act on (default: [0, 1, 2, ...])
        coefficient: Observable coefficient
        
    Returns:
        PauliObservable
    """
    if qubits is None:
        qubits = list(range(len(pauli_string)))
    
    return PauliObservable(pauli_string, qubits, coefficient)


def create_z_observable(qubits: List[int]) -> PauliSumObservable:
    """Create sum of Z observables on specified qubits."""
    terms = [ZObservable(q) for q in qubits]
    return PauliSumObservable(terms, name=f"Z_sum_{len(qubits)}")


def create_transverse_field_ising_model(
    num_qubits: int,
    J: float = 1.0,
    h: float = 1.0
) -> PauliSumObservable:
    """
    Create Transverse Field Ising Model Hamiltonian.
    
    H = -J ∑ᵢ ZᵢZᵢ₊₁ - h ∑ᵢ Xᵢ
    
    Args:
        num_qubits: Number of qubits
        J: Coupling strength
        h: Transverse field strength
        
    Returns:
        PauliSumObservable representing TFIM Hamiltonian
    """
    terms = []
    
    # ZZ coupling terms
    for i in range(num_qubits - 1):
        zz_term = PauliObservable("ZZ", [i, i + 1], -J)
        terms.append(zz_term)
    
    # X field terms
    for i in range(num_qubits):
        x_term = PauliObservable("X", [i], -h)
        terms.append(x_term)
    
    return PauliSumObservable(terms, name=f"TFIM_{num_qubits}")


def create_heisenberg_model(
    num_qubits: int,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0
) -> PauliSumObservable:
    """
    Create Heisenberg model Hamiltonian.
    
    H = ∑ᵢ (Jₓ XᵢXᵢ₊₁ + Jᵧ YᵢYᵢ₊₁ + Jᵤ ZᵢZᵢ₊₁)
    
    Args:
        num_qubits: Number of qubits
        Jx, Jy, Jz: Coupling strengths
        
    Returns:
        PauliSumObservable representing Heisenberg Hamiltonian
    """
    terms = []
    
    for i in range(num_qubits - 1):
        # XX coupling
        xx_term = PauliObservable("XX", [i, i + 1], Jx)
        terms.append(xx_term)
        
        # YY coupling
        yy_term = PauliObservable("YY", [i, i + 1], Jy)
        terms.append(yy_term)
        
        # ZZ coupling
        zz_term = PauliObservable("ZZ", [i, i + 1], Jz)
        terms.append(zz_term)
    
    return PauliSumObservable(terms, name=f"Heisenberg_{num_qubits}")


def create_all_z_observable(num_qubits: int) -> PauliObservable:
    """Create tensor product of Z operators on all qubits."""
    pauli_string = "Z" * num_qubits
    qubits = list(range(num_qubits))
    return PauliObservable(pauli_string, qubits, name=f"AllZ_{num_qubits}")


def parse_pauli_string(pauli_string: str) -> PauliObservable:
    """
    Parse Pauli string notation like "X_0 Y_1 Z_2" or "XYZ".
    
    Args:
        pauli_string: String representation of Pauli observable
        
    Returns:
        PauliObservable
    """
    # Simple case: just Pauli letters
    if all(c in "IXYZ" for c in pauli_string.upper()):
        qubits = list(range(len(pauli_string)))
        return PauliObservable(pauli_string.upper(), qubits)
    
    # More complex parsing for "X_0 Y_1" format
    terms = pauli_string.replace(" ", "").split("*")
    
    if len(terms) == 1:
        # Single term like "X_0" or "Y_1"
        parts = terms[0].split("_")
        if len(parts) == 2:
            pauli_char = parts[0].upper()
            qubit = int(parts[1])
            return PauliObservable(pauli_char, [qubit])
    
    raise ValueError(f"Cannot parse Pauli string: {pauli_string}")