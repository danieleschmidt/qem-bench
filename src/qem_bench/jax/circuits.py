"""JAX quantum circuit representation and manipulation."""

import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field

from .gates import *


@dataclass
class JAXCircuit:
    """
    JAX-compatible quantum circuit representation.
    
    Stores quantum gates as a list of dictionaries for efficient
    JAX compilation and execution.
    
    Attributes:
        num_qubits: Number of qubits in the circuit
        gates: List of gate dictionaries
        measurements: Measurement instructions
        name: Optional circuit name
    """
    
    num_qubits: int
    gates: List[Dict[str, Any]] = field(default_factory=list)
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    name: Optional[str] = None
    
    def __post_init__(self):
        """Validate circuit after initialization."""
        if self.num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")
    
    # Single-qubit gates
    def x(self, qubit: int) -> "JAXCircuit":
        """Apply Pauli-X gate."""
        self._add_gate("single", PauliX(), [qubit])
        return self
    
    def y(self, qubit: int) -> "JAXCircuit":
        """Apply Pauli-Y gate.""" 
        self._add_gate("single", PauliY(), [qubit])
        return self
    
    def z(self, qubit: int) -> "JAXCircuit":
        """Apply Pauli-Z gate."""
        self._add_gate("single", PauliZ(), [qubit])
        return self
    
    def h(self, qubit: int) -> "JAXCircuit":
        """Apply Hadamard gate."""
        self._add_gate("single", Hadamard(), [qubit])
        return self
    
    def s(self, qubit: int) -> "JAXCircuit":
        """Apply S gate (phase gate)."""
        self._add_gate("single", Phase(), [qubit])
        return self
    
    def t(self, qubit: int) -> "JAXCircuit":
        """Apply T gate (π/8 gate)."""
        t_gate = RZ(jnp.pi / 4)
        self._add_gate("single", t_gate, [qubit])
        return self
    
    def rx(self, angle: float, qubit: int) -> "JAXCircuit":
        """Apply RX rotation gate."""
        self._add_gate("single", RX(angle), [qubit])
        return self
    
    def ry(self, angle: float, qubit: int) -> "JAXCircuit":
        """Apply RY rotation gate."""
        self._add_gate("single", RY(angle), [qubit])
        return self
    
    def rz(self, angle: float, qubit: int) -> "JAXCircuit":
        """Apply RZ rotation gate."""
        self._add_gate("single", RZ(angle), [qubit])
        return self
    
    # Two-qubit gates
    def cx(self, control: int, target: int) -> "JAXCircuit":
        """Apply CNOT gate."""
        self._add_gate("two", CX(), [control, target])
        return self
    
    def cnot(self, control: int, target: int) -> "JAXCircuit":
        """Apply CNOT gate (alias for cx)."""
        return self.cx(control, target)
    
    def cz(self, control: int, target: int) -> "JAXCircuit":
        """Apply controlled-Z gate."""
        self._add_gate("two", CZ(), [control, target])
        return self
    
    def swap(self, qubit1: int, qubit2: int) -> "JAXCircuit":
        """Apply SWAP gate."""
        # SWAP = CNOT(q1,q2) · CNOT(q2,q1) · CNOT(q1,q2)
        self.cx(qubit1, qubit2)
        self.cx(qubit2, qubit1)
        self.cx(qubit1, qubit2)
        return self
    
    def cphase(self, angle: float, control: int, target: int) -> "JAXCircuit":
        """Apply controlled phase gate."""
        cp_gate = ControlledPhase(angle)
        self._add_gate("two", cp_gate, [control, target])
        return self
    
    # Multi-qubit gates
    def ccx(self, control1: int, control2: int, target: int) -> "JAXCircuit":
        """Apply Toffoli (CCX) gate."""
        toffoli_gate = Toffoli()
        self._add_gate("multi", toffoli_gate, [control1, control2, target])
        return self
    
    def toffoli(self, control1: int, control2: int, target: int) -> "JAXCircuit":
        """Apply Toffoli gate (alias for ccx)."""
        return self.ccx(control1, control2, target)
    
    # Custom gates
    def custom_gate(self, matrix: Union[np.ndarray, jnp.ndarray], qubits: List[int]) -> "JAXCircuit":
        """Apply custom gate matrix."""
        matrix = jnp.array(matrix)
        n_qubits = len(qubits)
        expected_size = 2 ** n_qubits
        
        if matrix.shape != (expected_size, expected_size):
            raise ValueError(f"Matrix shape {matrix.shape} doesn't match {n_qubits} qubits")
        
        gate_type = "single" if n_qubits == 1 else "two" if n_qubits == 2 else "multi"
        
        gate_dict = {
            "type": gate_type,
            "matrix": matrix,
            "qubits": qubits,
            "name": "custom"
        }
        
        self.gates.append(gate_dict)
        return self
    
    # Measurement
    def measure(self, qubit: int, classical_bit: Optional[int] = None) -> "JAXCircuit":
        """Add measurement instruction."""
        if classical_bit is None:
            classical_bit = qubit
        
        measurement = {
            "type": "measure",
            "qubit": qubit,
            "classical_bit": classical_bit
        }
        
        self.measurements.append(measurement)
        return self
    
    def measure_all(self) -> "JAXCircuit":
        """Measure all qubits."""
        for i in range(self.num_qubits):
            self.measure(i)
        return self
    
    # Circuit manipulation
    def inverse(self) -> "JAXCircuit":
        """Return the inverse (adjoint) of the circuit."""
        inverse_circuit = JAXCircuit(self.num_qubits, name=f"{self.name}_inv")
        
        # Reverse order of gates and take adjoint
        for gate in reversed(self.gates):
            inv_matrix = jnp.conj(gate["matrix"]).T
            inv_gate = {
                "type": gate["type"],
                "matrix": inv_matrix,
                "qubits": gate["qubits"],
                "name": f"{gate.get('name', 'gate')}_inv"
            }
            inverse_circuit.gates.append(inv_gate)
        
        return inverse_circuit
    
    def compose(self, other: "JAXCircuit") -> "JAXCircuit":
        """Compose this circuit with another circuit."""
        if other.num_qubits != self.num_qubits:
            raise ValueError("Circuits must have same number of qubits to compose")
        
        composed = JAXCircuit(
            self.num_qubits, 
            name=f"{self.name}_composed_{other.name}"
        )
        
        # Add all gates from both circuits
        composed.gates.extend(self.gates)
        composed.gates.extend(other.gates)
        
        # Add measurements
        composed.measurements.extend(self.measurements)
        composed.measurements.extend(other.measurements)
        
        return composed
    
    def tensor(self, other: "JAXCircuit") -> "JAXCircuit":
        """Tensor product with another circuit."""
        total_qubits = self.num_qubits + other.num_qubits
        tensored = JAXCircuit(
            total_qubits,
            name=f"{self.name}_tensor_{other.name}"
        )
        
        # Add gates from first circuit (qubits 0 to num_qubits-1)
        tensored.gates.extend(self.gates)
        
        # Add gates from second circuit (qubits num_qubits to total_qubits-1)
        for gate in other.gates:
            shifted_gate = gate.copy()
            shifted_gate["qubits"] = [q + self.num_qubits for q in gate["qubits"]]
            tensored.gates.append(shifted_gate)
        
        return tensored
    
    def repeat(self, times: int) -> "JAXCircuit":
        """Repeat the circuit multiple times."""
        repeated = JAXCircuit(
            self.num_qubits,
            name=f"{self.name}_x{times}"
        )
        
        for _ in range(times):
            repeated.gates.extend(self.gates)
        
        repeated.measurements.extend(self.measurements)
        return repeated
    
    def copy(self) -> "JAXCircuit":
        """Create a deep copy of the circuit."""
        import copy
        return copy.deepcopy(self)
    
    # Circuit properties
    @property
    def depth(self) -> int:
        """Calculate circuit depth (number of gate layers)."""
        # Simplified depth calculation - assumes gates can be parallelized
        # if they act on different qubits
        if not self.gates:
            return 0
        
        # Track when each qubit is last used
        qubit_times = [0] * self.num_qubits
        current_time = 0
        
        for gate in self.gates:
            # Find the latest time among qubits used by this gate
            gate_qubits = gate["qubits"]
            start_time = max(qubit_times[q] for q in gate_qubits)
            end_time = start_time + 1
            
            # Update times for all qubits used by this gate
            for q in gate_qubits:
                qubit_times[q] = end_time
        
        return max(qubit_times)
    
    @property
    def size(self) -> int:
        """Number of gates in the circuit."""
        return len(self.gates)
    
    def count_gates(self) -> Dict[str, int]:
        """Count gates by type."""
        counts = {}
        for gate in self.gates:
            gate_name = gate.get("name", "unknown")
            counts[gate_name] = counts.get(gate_name, 0) + 1
        return counts
    
    # Helper methods
    def _add_gate(self, gate_type: str, gate_obj: Any, qubits: List[int]) -> None:
        """Add gate to circuit."""
        self._validate_qubits(qubits)
        
        gate_dict = {
            "type": gate_type,
            "matrix": gate_obj.matrix,
            "qubits": qubits,
            "name": gate_obj.name
        }
        
        self.gates.append(gate_dict)
    
    def _validate_qubits(self, qubits: List[int]) -> None:
        """Validate qubit indices."""
        for qubit in qubits:
            if not (0 <= qubit < self.num_qubits):
                raise ValueError(f"Qubit {qubit} out of range [0, {self.num_qubits-1}]")
    
    # String representation
    def __str__(self) -> str:
        """String representation of the circuit."""
        lines = [f"JAXCircuit({self.num_qubits} qubits)"]
        
        if self.name:
            lines.append(f"Name: {self.name}")
        
        lines.append(f"Depth: {self.depth}")
        lines.append(f"Gates: {self.size}")
        
        if self.gates:
            lines.append("Gate sequence:")
            for i, gate in enumerate(self.gates):
                gate_name = gate.get("name", "unknown")
                qubits = gate["qubits"]
                lines.append(f"  {i}: {gate_name} on qubits {qubits}")
        
        if self.measurements:
            lines.append("Measurements:")
            for meas in self.measurements:
                lines.append(f"  qubit {meas['qubit']} -> bit {meas['classical_bit']}")
        
        return "\\n".join(lines)
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"JAXCircuit(num_qubits={self.num_qubits}, gates={len(self.gates)}, name='{self.name}')"


# Convenience functions for creating circuits
def create_jax_circuit(num_qubits: int, name: Optional[str] = None) -> JAXCircuit:
    """Create an empty JAX circuit."""
    return JAXCircuit(num_qubits, name=name)


def create_bell_circuit(qubit1: int = 0, qubit2: int = 1) -> JAXCircuit:
    """Create a Bell state preparation circuit."""
    circuit = JAXCircuit(max(qubit1, qubit2) + 1, name="bell")
    circuit.h(qubit1)
    circuit.cx(qubit1, qubit2)
    return circuit


def create_ghz_circuit(num_qubits: int) -> JAXCircuit:
    """Create a GHZ state preparation circuit."""
    circuit = JAXCircuit(num_qubits, name="ghz")
    circuit.h(0)
    for i in range(1, num_qubits):
        circuit.cx(0, i)
    return circuit


def create_quantum_fourier_transform(num_qubits: int) -> JAXCircuit:
    """Create quantum Fourier transform circuit."""
    circuit = JAXCircuit(num_qubits, name="qft")
    
    for i in range(num_qubits):
        # Hadamard on qubit i
        circuit.h(i)
        
        # Controlled phase rotations
        for j in range(i + 1, num_qubits):
            angle = 2 * jnp.pi / (2 ** (j - i + 1))
            circuit.cphase(angle, j, i)
    
    # Reverse qubit order
    for i in range(num_qubits // 2):
        circuit.swap(i, num_qubits - 1 - i)
    
    return circuit


def create_random_circuit(
    num_qubits: int,
    depth: int,
    gate_set: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> JAXCircuit:
    """Create a random quantum circuit."""
    if gate_set is None:
        gate_set = ["h", "x", "y", "z", "rx", "ry", "rz", "cx"]
    
    np.random.seed(seed)
    circuit = JAXCircuit(num_qubits, name="random")
    
    for _ in range(depth):
        gate = np.random.choice(gate_set)
        
        if gate in ["h", "x", "y", "z"]:
            qubit = np.random.randint(num_qubits)
            getattr(circuit, gate)(qubit)
        elif gate in ["rx", "ry", "rz"]:
            qubit = np.random.randint(num_qubits)
            angle = np.random.uniform(0, 2 * np.pi)
            getattr(circuit, gate)(angle, qubit)
        elif gate == "cx":
            qubits = np.random.choice(num_qubits, 2, replace=False)
            circuit.cx(qubits[0], qubits[1])
    
    return circuit