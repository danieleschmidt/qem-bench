"""Quantum gate definitions for JAX simulator."""

import jax.numpy as jnp
import numpy as np
from typing import Optional
from abc import ABC, abstractmethod


class QuantumGate(ABC):
    """Abstract base class for quantum gates."""
    
    def __init__(self, name: str):
        self.name = name
    
    @property
    @abstractmethod
    def matrix(self) -> jnp.ndarray:
        """Gate matrix representation."""
        pass


# Pauli gates
class PauliX(QuantumGate):
    """Pauli-X gate (NOT gate)."""
    
    def __init__(self):
        super().__init__("X")
    
    @property
    def matrix(self) -> jnp.ndarray:
        return jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)


class PauliY(QuantumGate):
    """Pauli-Y gate."""
    
    def __init__(self):
        super().__init__("Y")
    
    @property
    def matrix(self) -> jnp.ndarray:
        return jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)


class PauliZ(QuantumGate):
    """Pauli-Z gate."""
    
    def __init__(self):
        super().__init__("Z")
    
    @property
    def matrix(self) -> jnp.ndarray:
        return jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)


# Hadamard gate
class Hadamard(QuantumGate):
    """Hadamard gate."""
    
    def __init__(self):
        super().__init__("H")
    
    @property
    def matrix(self) -> jnp.ndarray:
        return jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / jnp.sqrt(2)


# Phase gates
class Phase(QuantumGate):
    """S gate (phase gate)."""
    
    def __init__(self):
        super().__init__("S")
    
    @property
    def matrix(self) -> jnp.ndarray:
        return jnp.array([[1, 0], [0, 1j]], dtype=jnp.complex64)


class TGate(QuantumGate):
    """T gate (π/8 gate)."""
    
    def __init__(self):
        super().__init__("T")
    
    @property
    def matrix(self) -> jnp.ndarray:
        return jnp.array([[1, 0], [0, jnp.exp(1j * jnp.pi / 4)]], dtype=jnp.complex64)


# Rotation gates
class RX(QuantumGate):
    """Rotation around X-axis."""
    
    def __init__(self, angle: float):
        super().__init__("RX")
        self.angle = angle
    
    @property
    def matrix(self) -> jnp.ndarray:
        cos_half = jnp.cos(self.angle / 2)
        sin_half = jnp.sin(self.angle / 2)
        return jnp.array([
            [cos_half, -1j * sin_half],
            [-1j * sin_half, cos_half]
        ], dtype=jnp.complex64)


class RY(QuantumGate):
    """Rotation around Y-axis."""
    
    def __init__(self, angle: float):
        super().__init__("RY")
        self.angle = angle
    
    @property
    def matrix(self) -> jnp.ndarray:
        cos_half = jnp.cos(self.angle / 2)
        sin_half = jnp.sin(self.angle / 2)
        return jnp.array([
            [cos_half, -sin_half],
            [sin_half, cos_half]
        ], dtype=jnp.complex64)


class RZ(QuantumGate):
    """Rotation around Z-axis."""
    
    def __init__(self, angle: float):
        super().__init__("RZ")
        self.angle = angle
    
    @property
    def matrix(self) -> jnp.ndarray:
        exp_neg = jnp.exp(-1j * self.angle / 2)
        exp_pos = jnp.exp(1j * self.angle / 2)
        return jnp.array([
            [exp_neg, 0],
            [0, exp_pos]
        ], dtype=jnp.complex64)


# Two-qubit gates
class CX(QuantumGate):
    """CNOT gate (controlled-X)."""
    
    def __init__(self):
        super().__init__("CX")
    
    @property
    def matrix(self) -> jnp.ndarray:
        return jnp.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=jnp.complex64)


class CZ(QuantumGate):
    """Controlled-Z gate."""
    
    def __init__(self):
        super().__init__("CZ")
    
    @property
    def matrix(self) -> jnp.ndarray:
        return jnp.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=jnp.complex64)


class SWAP(QuantumGate):
    """SWAP gate."""
    
    def __init__(self):
        super().__init__("SWAP")
    
    @property
    def matrix(self) -> jnp.ndarray:
        return jnp.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=jnp.complex64)


class ControlledPhase(QuantumGate):
    """Controlled phase gate."""
    
    def __init__(self, angle: float):
        super().__init__("CP")
        self.angle = angle
    
    @property
    def matrix(self) -> jnp.ndarray:
        return jnp.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, jnp.exp(1j * self.angle)]
        ], dtype=jnp.complex64)


# Three-qubit gates
class Toffoli(QuantumGate):
    """Toffoli gate (CCX)."""
    
    def __init__(self):
        super().__init__("CCX")
    
    @property
    def matrix(self) -> jnp.ndarray:
        matrix = jnp.eye(8, dtype=jnp.complex64)
        # Swap last two elements (|110⟩ ↔ |111⟩)
        matrix = matrix.at[6, 6].set(0)
        matrix = matrix.at[6, 7].set(1)
        matrix = matrix.at[7, 6].set(1)
        matrix = matrix.at[7, 7].set(0)
        return matrix


class Fredkin(QuantumGate):
    """Fredkin gate (controlled-SWAP)."""
    
    def __init__(self):
        super().__init__("CSWAP")
    
    @property
    def matrix(self) -> jnp.ndarray:
        matrix = jnp.eye(8, dtype=jnp.complex64)
        # Swap |101⟩ ↔ |110⟩ when control is |1⟩
        matrix = matrix.at[5, 5].set(0)
        matrix = matrix.at[5, 6].set(1)
        matrix = matrix.at[6, 5].set(1)
        matrix = matrix.at[6, 6].set(0)
        return matrix


# Identity gate
class Identity(QuantumGate):
    """Identity gate."""
    
    def __init__(self, num_qubits: int = 1):
        super().__init__("I")
        self.num_qubits = num_qubits
    
    @property
    def matrix(self) -> jnp.ndarray:
        size = 2 ** self.num_qubits
        return jnp.eye(size, dtype=jnp.complex64)


# Utility functions
def create_controlled_gate(base_gate: QuantumGate) -> QuantumGate:
    """Create a controlled version of a single-qubit gate."""
    
    class ControlledGate(QuantumGate):
        def __init__(self, base: QuantumGate):
            super().__init__(f"C{base.name}")
            self.base_gate = base
        
        @property
        def matrix(self) -> jnp.ndarray:
            # Controlled gate matrix: |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U
            base_matrix = self.base_gate.matrix
            identity = jnp.eye(2, dtype=jnp.complex64)
            
            # |0⟩⟨0| ⊗ I
            proj_0 = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex64)
            term1 = jnp.kron(proj_0, identity)
            
            # |1⟩⟨1| ⊗ U  
            proj_1 = jnp.array([[0, 0], [0, 1]], dtype=jnp.complex64)
            term2 = jnp.kron(proj_1, base_matrix)
            
            return term1 + term2
    
    return ControlledGate(base_gate)


def rotation_gate(axis: str, angle: float) -> QuantumGate:
    """Create rotation gate around specified axis."""
    axis = axis.upper()
    if axis == "X":
        return RX(angle)
    elif axis == "Y":
        return RY(angle)
    elif axis == "Z":
        return RZ(angle)
    else:
        raise ValueError(f"Unknown rotation axis: {axis}")


def pauli_gate(pauli: str) -> QuantumGate:
    """Create Pauli gate from string."""
    pauli = pauli.upper()
    if pauli == "X":
        return PauliX()
    elif pauli == "Y":
        return PauliY()
    elif pauli == "Z":
        return PauliZ()
    elif pauli == "I":
        return Identity()
    else:
        raise ValueError(f"Unknown Pauli operator: {pauli}")


# Gate decompositions
def decompose_arbitrary_single_qubit(matrix: jnp.ndarray) -> list[QuantumGate]:
    """
    Decompose arbitrary single-qubit unitary into RZ-RY-RZ sequence.
    
    Uses ZYZ decomposition: U = RZ(α) RY(β) RZ(γ)
    """
    # This is a simplified implementation
    # In practice, would use proper ZYZ decomposition algorithm
    
    # For now, return the matrix as a custom gate
    class CustomSingleQubit(QuantumGate):
        def __init__(self, mat):
            super().__init__("Custom")
            self._matrix = mat
        
        @property
        def matrix(self):
            return self._matrix
    
    return [CustomSingleQubit(matrix)]


def decompose_two_qubit_gate(matrix: jnp.ndarray) -> list[QuantumGate]:
    """
    Decompose arbitrary two-qubit unitary into elementary gates.
    
    This is a placeholder - full implementation would use KAK decomposition.
    """
    class CustomTwoQubit(QuantumGate):
        def __init__(self, mat):
            super().__init__("Custom2Q")
            self._matrix = mat
        
        @property
        def matrix(self):
            return self._matrix
    
    return [CustomTwoQubit(matrix)]