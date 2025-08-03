"""JAX-based quantum simulator for high-performance quantum circuit simulation."""

import jax
import jax.numpy as jnp
from jax import random, vmap, jit
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

from .circuits import JAXCircuit
from .observables import PauliObservable


@dataclass
class SimulationResult:
    """Result from quantum circuit simulation."""
    
    statevector: jnp.ndarray
    measurement_counts: Optional[Dict[str, int]] = None
    expectation_values: Optional[Dict[str, float]] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[int] = None


class JAXSimulator:
    """
    High-performance quantum circuit simulator using JAX.
    
    Features:
    - GPU/TPU acceleration via JAX
    - JIT compilation for fast execution
    - Vectorized operations for batched simulation
    - Automatic differentiation support
    - Memory-efficient statevector simulation
    
    Example:
        >>> simulator = JAXSimulator(num_qubits=5)
        >>> circuit = create_jax_circuit(5)
        >>> circuit.h(0)
        >>> circuit.cx(0, 1)
        >>> result = simulator.run(circuit, shots=1024)
        >>> print(result.measurement_counts)
    """
    
    def __init__(
        self,
        num_qubits: int,
        precision: str = "float32",
        backend: str = "auto",
        seed: Optional[int] = None
    ):
        """
        Initialize JAX quantum simulator.
        
        Args:
            num_qubits: Number of qubits to simulate
            precision: Numerical precision ("float32" or "float64")
            backend: JAX backend ("cpu", "gpu", "tpu", "auto")
            seed: Random seed for reproducible results
        """
        self.num_qubits = num_qubits
        self.hilbert_space_size = 2 ** num_qubits
        self.precision = precision
        
        # Set up JAX configuration
        if backend != "auto":
            jax.config.update("jax_platform_name", backend)
        
        if precision == "float64":
            jax.config.update("jax_enable_x64", True)
        
        # Initialize random key
        self.key = random.PRNGKey(seed if seed is not None else 42)
        
        # Setup dtype
        self.dtype = jnp.float32 if precision == "float32" else jnp.float64
        self.complex_dtype = jnp.complex64 if precision == "float32" else jnp.complex128
        
        # Pre-allocate state vector
        self.initial_state = self._create_zero_state()
        
        # JIT-compiled functions for performance
        self._apply_gate_jit = jit(self._apply_gate)
        self._measure_all_jit = jit(self._measure_all)
        self._expectation_value_jit = jit(self._expectation_value)
    
    def _create_zero_state(self) -> jnp.ndarray:
        """Create the |0...0⟩ initial state."""
        state = jnp.zeros(self.hilbert_space_size, dtype=self.complex_dtype)
        state = state.at[0].set(1.0)
        return state
    
    def run(
        self,
        circuit: JAXCircuit,
        shots: Optional[int] = None,
        observables: Optional[List[PauliObservable]] = None,
        initial_state: Optional[jnp.ndarray] = None
    ) -> SimulationResult:
        """
        Run quantum circuit simulation.
        
        Args:
            circuit: JAX quantum circuit to simulate
            shots: Number of measurement shots (None for statevector only)
            observables: List of observables to measure
            initial_state: Custom initial state (default: |0...0⟩)
            
        Returns:
            SimulationResult containing statevector and measurement data
        """
        import time
        start_time = time.time()
        
        # Use provided initial state or default zero state
        if initial_state is not None:
            state = initial_state.astype(self.complex_dtype)
        else:
            state = self.initial_state
        
        # Apply circuit gates
        for gate in circuit.gates:
            state = self._apply_gate_jit(state, gate)
        
        # Normalize state (in case of numerical errors)
        state = state / jnp.linalg.norm(state)
        
        # Perform measurements if requested
        measurement_counts = None
        if shots is not None:
            measurement_counts = self._sample_measurements(state, shots)
        
        # Calculate expectation values if observables provided
        expectation_values = None
        if observables is not None:
            expectation_values = {}
            for obs in observables:
                exp_val = self._expectation_value_jit(state, obs.matrix)
                expectation_values[obs.name] = float(exp_val.real)
        
        execution_time = time.time() - start_time
        
        return SimulationResult(
            statevector=state,
            measurement_counts=measurement_counts,
            expectation_values=expectation_values,
            execution_time=execution_time
        )
    
    def _apply_gate(self, state: jnp.ndarray, gate: Dict[str, Any]) -> jnp.ndarray:
        """Apply a quantum gate to the state vector."""
        gate_type = gate["type"]
        qubits = gate["qubits"]
        
        if gate_type == "single":
            return self._apply_single_qubit_gate(state, gate["matrix"], qubits[0])
        elif gate_type == "two":
            return self._apply_two_qubit_gate(state, gate["matrix"], qubits[0], qubits[1])
        elif gate_type == "multi":
            return self._apply_multi_qubit_gate(state, gate["matrix"], qubits)
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
    
    def _apply_single_qubit_gate(
        self, 
        state: jnp.ndarray, 
        gate_matrix: jnp.ndarray, 
        qubit: int
    ) -> jnp.ndarray:
        """Apply single-qubit gate using tensor product structure."""
        # Reshape state for matrix multiplication
        state_reshaped = state.reshape([2] * self.num_qubits)
        
        # Apply gate to specific qubit
        gate_matrix = gate_matrix.astype(self.complex_dtype)
        state_reshaped = jnp.tensordot(
            gate_matrix, state_reshaped, 
            axes=([1], [qubit])
        )
        
        # Move the result axis back to correct position
        state_reshaped = jnp.moveaxis(state_reshaped, 0, qubit)
        
        return state_reshaped.flatten()
    
    def _apply_two_qubit_gate(
        self,
        state: jnp.ndarray,
        gate_matrix: jnp.ndarray,
        control: int,
        target: int
    ) -> jnp.ndarray:
        """Apply two-qubit gate using tensor product structure."""
        # Ensure control < target for consistent indexing
        if control > target:
            control, target = target, control
            # Swap gate matrix accordingly (CNOT(1,0) = SWAP · CNOT(0,1) · SWAP)
            gate_matrix = self._swap_gate_qubits(gate_matrix)
        
        # Reshape state
        state_reshaped = state.reshape([2] * self.num_qubits)
        
        # Apply two-qubit gate
        gate_matrix = gate_matrix.astype(self.complex_dtype)
        
        # Contract over control and target qubits
        axes_state = [control, target]
        axes_gate = [2, 3]  # Input indices of 4x4 gate matrix
        
        state_reshaped = jnp.tensordot(
            gate_matrix.reshape(2, 2, 2, 2), 
            state_reshaped,
            axes=(axes_gate, axes_state)
        )
        
        # Move result axes back
        state_reshaped = jnp.moveaxis(state_reshaped, [0, 1], [control, target])
        
        return state_reshaped.flatten()
    
    def _apply_multi_qubit_gate(
        self,
        state: jnp.ndarray,
        gate_matrix: jnp.ndarray,
        qubits: List[int]
    ) -> jnp.ndarray:
        """Apply multi-qubit gate."""
        n_gate_qubits = len(qubits)
        gate_size = 2 ** n_gate_qubits
        
        if gate_matrix.shape != (gate_size, gate_size):
            raise ValueError(f"Gate matrix shape {gate_matrix.shape} doesn't match {n_gate_qubits} qubits")
        
        # Reshape state and gate matrix
        state_reshaped = state.reshape([2] * self.num_qubits)
        gate_reshaped = gate_matrix.reshape([2] * (2 * n_gate_qubits))
        
        # Apply gate
        gate_reshaped = gate_reshaped.astype(self.complex_dtype)
        
        input_axes = list(range(n_gate_qubits, 2 * n_gate_qubits))
        state_axes = qubits
        
        state_reshaped = jnp.tensordot(
            gate_reshaped,
            state_reshaped,
            axes=(input_axes, state_axes)
        )
        
        # Move result axes back
        output_axes = list(range(n_gate_qubits))
        state_reshaped = jnp.moveaxis(state_reshaped, output_axes, qubits)
        
        return state_reshaped.flatten()
    
    def _swap_gate_qubits(self, gate_matrix: jnp.ndarray) -> jnp.ndarray:
        """Swap the qubits in a two-qubit gate matrix."""
        # Convert 4x4 matrix to 2x2x2x2 tensor
        gate_tensor = gate_matrix.reshape(2, 2, 2, 2)
        # Swap qubit order: (q0_out, q1_out, q0_in, q1_in) -> (q1_out, q0_out, q1_in, q0_in)
        gate_tensor = jnp.transpose(gate_tensor, (1, 0, 3, 2))
        return gate_tensor.reshape(4, 4)
    
    def _measure_all(self, state: jnp.ndarray) -> int:
        """Measure all qubits and return measurement outcome as integer."""
        probabilities = jnp.abs(state) ** 2
        self.key, subkey = random.split(self.key)
        return random.choice(subkey, self.hilbert_space_size, p=probabilities)
    
    def _sample_measurements(self, state: jnp.ndarray, shots: int) -> Dict[str, int]:
        """Sample multiple measurements from the quantum state."""
        # Generate all measurement outcomes
        outcomes = []
        for _ in range(shots):
            outcome = self._measure_all_jit(state)
            outcomes.append(int(outcome))
        
        # Convert to bitstring counts
        counts = {}
        for outcome in outcomes:
            bitstring = format(outcome, f'0{self.num_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return counts
    
    def _expectation_value(self, state: jnp.ndarray, observable: jnp.ndarray) -> jnp.ndarray:
        """Calculate expectation value ⟨ψ|O|ψ⟩."""
        return jnp.conj(state) @ observable @ state
    
    def run_batch(
        self,
        circuits: List[JAXCircuit],
        shots: Optional[int] = None,
        observables: Optional[List[PauliObservable]] = None
    ) -> List[SimulationResult]:
        """
        Run batch of circuits in parallel using vectorization.
        
        Args:
            circuits: List of JAX circuits to simulate
            shots: Number of shots per circuit
            observables: Observables to measure
            
        Returns:
            List of simulation results
        """
        # For now, run sequentially (could be optimized with vmap)
        results = []
        for circuit in circuits:
            result = self.run(circuit, shots=shots, observables=observables)
            results.append(result)
        
        return results
    
    def get_statevector(self, circuit: JAXCircuit) -> jnp.ndarray:
        """Get the final statevector after applying circuit."""
        result = self.run(circuit)
        return result.statevector
    
    def get_probabilities(self, circuit: JAXCircuit) -> jnp.ndarray:
        """Get measurement probabilities for all computational basis states."""
        statevector = self.get_statevector(circuit)
        return jnp.abs(statevector) ** 2
    
    def get_counts(self, circuit: JAXCircuit, shots: int) -> Dict[str, int]:
        """Get measurement counts dictionary."""
        result = self.run(circuit, shots=shots)
        return result.measurement_counts
    
    def measure_observable(
        self, 
        circuit: JAXCircuit, 
        observable: PauliObservable
    ) -> float:
        """Measure expectation value of an observable."""
        result = self.run(circuit, observables=[observable])
        return result.expectation_values[observable.name]
    
    def reset(self) -> None:
        """Reset simulator to initial state."""
        self.key = random.PRNGKey(42)  # Reset to default seed
    
    @property
    def backend_info(self) -> Dict[str, Any]:
        """Get information about the JAX backend."""
        return {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "num_qubits": self.num_qubits,
            "precision": self.precision,
            "hilbert_space_size": self.hilbert_space_size,
            "memory_usage_gb": (self.hilbert_space_size * 16) / (1024**3)  # Complex128 = 16 bytes
        }


# Utility functions for creating simulators
def create_gpu_simulator(num_qubits: int, **kwargs) -> JAXSimulator:
    """Create a GPU-accelerated JAX simulator."""
    return JAXSimulator(num_qubits, backend="gpu", **kwargs)


def create_cpu_simulator(num_qubits: int, **kwargs) -> JAXSimulator:
    """Create a CPU-based JAX simulator.""" 
    return JAXSimulator(num_qubits, backend="cpu", **kwargs)


def create_tpu_simulator(num_qubits: int, **kwargs) -> JAXSimulator:
    """Create a TPU-accelerated JAX simulator."""
    return JAXSimulator(num_qubits, backend="tpu", **kwargs)