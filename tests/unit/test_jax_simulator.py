"""Unit tests for JAX quantum simulator."""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import patch

from qem_bench.jax.simulator import JAXSimulator, SimulationResult
from qem_bench.jax.circuits import JAXCircuit


class TestJAXSimulator:
    """Test suite for JAXSimulator class."""
    
    def test_initialization_default(self):
        """Test simulator initialization with default parameters."""
        sim = JAXSimulator(num_qubits=3)
        
        assert sim.num_qubits == 3
        assert sim.hilbert_space_size == 8
        assert sim.precision == "float32"
        assert sim.dtype == jnp.float32
        assert sim.complex_dtype == jnp.complex64
    
    def test_initialization_custom_parameters(self):
        """Test simulator initialization with custom parameters."""
        sim = JAXSimulator(
            num_qubits=4,
            precision="float64",
            backend="cpu",
            seed=123
        )
        
        assert sim.num_qubits == 4
        assert sim.hilbert_space_size == 16
        assert sim.precision == "float64"
        assert sim.dtype == jnp.float64
        assert sim.complex_dtype == jnp.complex128
    
    def test_create_zero_state(self):
        """Test creation of zero state."""
        sim = JAXSimulator(num_qubits=2)
        state = sim._create_zero_state()
        
        expected = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.complex64)
        np.testing.assert_array_almost_equal(state, expected)
        
        # Check normalization
        assert np.isclose(jnp.linalg.norm(state), 1.0)
    
    def test_run_empty_circuit(self, sample_bell_circuit):
        """Test running circuit with no gates."""
        sim = JAXSimulator(num_qubits=2)
        empty_circuit = JAXCircuit(num_qubits=2, name="empty")
        
        result = sim.run(empty_circuit)
        
        assert isinstance(result, SimulationResult)
        # Should be |00⟩ state
        expected_state = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.complex64)
        np.testing.assert_array_almost_equal(result.statevector, expected_state)
    
    def test_run_bell_circuit(self, sample_bell_circuit):
        """Test running Bell state preparation circuit."""
        sim = JAXSimulator(num_qubits=2)
        
        result = sim.run(sample_bell_circuit)
        
        # Bell state: (|00⟩ + |11⟩)/√2
        expected_amplitude = 1.0 / np.sqrt(2)
        expected_state = jnp.array([
            expected_amplitude, 0.0, 0.0, expected_amplitude
        ], dtype=jnp.complex64)
        
        np.testing.assert_array_almost_equal(result.statevector, expected_state, decimal=6)
        
        # Check normalization
        assert np.isclose(jnp.linalg.norm(result.statevector), 1.0)
    
    def test_run_with_measurements(self, sample_bell_circuit):
        """Test circuit execution with shot-based measurements."""
        sim = JAXSimulator(num_qubits=2, seed=42)
        
        result = sim.run(sample_bell_circuit, shots=1000)
        
        assert result.measurement_counts is not None
        
        # Bell state should give roughly equal |00⟩ and |11⟩ counts
        counts = result.measurement_counts
        total_shots = sum(counts.values())
        assert total_shots == 1000
        
        # Should only have |00⟩ and |11⟩ outcomes
        assert set(counts.keys()).issubset({"00", "11"})
        
        # Check that both outcomes occur (with some statistical tolerance)
        if "00" in counts and "11" in counts:
            assert counts["00"] > 100  # At least 10% each outcome
            assert counts["11"] > 100
    
    def test_run_with_observables(self, sample_bell_circuit, sample_pauli_observable):
        """Test circuit execution with observable measurement."""
        sim = JAXSimulator(num_qubits=2)
        
        result = sim.run(sample_bell_circuit, observables=[sample_pauli_observable])
        
        assert result.expectation_values is not None
        assert "ZZ" in result.expectation_values
        
        # For Bell state |00⟩ + |11⟩, ZZ expectation should be 1
        # because Z⊗Z(|00⟩) = |00⟩ and Z⊗Z(|11⟩) = |11⟩
        expected_zz = 1.0
        assert np.isclose(result.expectation_values["ZZ"], expected_zz, atol=1e-6)
    
    def test_run_with_custom_initial_state(self):
        """Test circuit execution with custom initial state."""
        sim = JAXSimulator(num_qubits=2)
        
        # Custom initial state: |01⟩
        custom_state = jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.complex64)
        
        # Apply just Hadamard on first qubit
        circuit = JAXCircuit(num_qubits=2)
        circuit.h(0)
        
        result = sim.run(circuit, initial_state=custom_state)
        
        # Should get (|01⟩ + |11⟩)/√2
        expected_amplitude = 1.0 / np.sqrt(2)
        expected_state = jnp.array([
            0.0, expected_amplitude, 0.0, expected_amplitude
        ], dtype=jnp.complex64)
        
        np.testing.assert_array_almost_equal(result.statevector, expected_state, decimal=6)
    
    def test_apply_single_qubit_gate(self):
        """Test application of single-qubit gates."""
        sim = JAXSimulator(num_qubits=2)
        
        # Test Pauli-X gate
        gate_dict = {
            "type": "single",
            "matrix": jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64),
            "qubits": [0]
        }
        
        initial_state = sim.initial_state  # |00⟩
        result_state = sim._apply_gate(initial_state, gate_dict)
        
        # Should get |10⟩
        expected_state = jnp.array([0.0, 0.0, 1.0, 0.0], dtype=jnp.complex64)
        np.testing.assert_array_almost_equal(result_state, expected_state)
    
    def test_apply_two_qubit_gate(self):
        """Test application of two-qubit gates."""
        sim = JAXSimulator(num_qubits=2)
        
        # Test CNOT gate
        cnot_matrix = jnp.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=jnp.complex64)
        
        gate_dict = {
            "type": "two",
            "matrix": cnot_matrix,
            "qubits": [0, 1]
        }
        
        # Start with |10⟩ state
        initial_state = jnp.array([0.0, 0.0, 1.0, 0.0], dtype=jnp.complex64)
        result_state = sim._apply_gate(initial_state, gate_dict)
        
        # CNOT(|10⟩) = |11⟩
        expected_state = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.complex64)
        np.testing.assert_array_almost_equal(result_state, expected_state)
    
    def test_measure_all(self):
        """Test measurement of all qubits."""
        sim = JAXSimulator(num_qubits=2, seed=42)
        
        # Bell state
        state = jnp.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=jnp.complex64)
        
        outcomes = []
        for _ in range(100):
            outcome = sim._measure_all(state)  
            outcomes.append(outcome)
        
        # Should only get outcomes 0 (|00⟩) and 3 (|11⟩)
        unique_outcomes = set(outcomes)
        assert unique_outcomes.issubset({0, 3})
        
        # Both outcomes should occur
        assert len(unique_outcomes) > 1
    
    def test_sample_measurements(self):
        """Test sampling multiple measurements."""
        sim = JAXSimulator(num_qubits=2, seed=42)
        
        # Bell state
        state = jnp.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=jnp.complex64)
        
        counts = sim._sample_measurements(state, shots=1000)
        
        assert isinstance(counts, dict)
        assert sum(counts.values()) == 1000
        
        # Should only have |00⟩ and |11⟩
        assert set(counts.keys()).issubset({"00", "11"})
    
    def test_expectation_value_calculation(self):
        """Test expectation value calculation."""
        sim = JAXSimulator(num_qubits=1)
        
        # |+⟩ state: (|0⟩ + |1⟩)/√2
        state = jnp.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=jnp.complex64)
        
        # X observable
        x_observable = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
        
        expectation = sim._expectation_value(state, x_observable)
        
        # ⟨+|X|+⟩ = 1
        assert np.isclose(expectation, 1.0, atol=1e-6)
    
    def test_run_batch(self, benchmark_circuits):
        """Test batch circuit execution."""
        sim = JAXSimulator(num_qubits=3)
        
        results = sim.run_batch(benchmark_circuits, shots=100)
        
        assert len(results) == len(benchmark_circuits)
        for result in results:
            assert isinstance(result, SimulationResult)
            assert result.measurement_counts is not None
            assert sum(result.measurement_counts.values()) == 100
    
    def test_get_statevector(self, sample_bell_circuit):
        """Test statevector retrieval."""
        sim = JAXSimulator(num_qubits=2)
        
        statevector = sim.get_statevector(sample_bell_circuit)
        
        # Bell state: (|00⟩ + |11⟩)/√2
        expected_amplitude = 1.0 / np.sqrt(2)
        expected_state = jnp.array([
            expected_amplitude, 0.0, 0.0, expected_amplitude
        ], dtype=jnp.complex64)
        
        np.testing.assert_array_almost_equal(statevector, expected_state, decimal=6)
    
    def test_get_probabilities(self, sample_bell_circuit):
        """Test probability calculation."""
        sim = JAXSimulator(num_qubits=2)
        
        probabilities = sim.get_probabilities(sample_bell_circuit)
        
        # Bell state should have equal probabilities for |00⟩ and |11⟩
        expected_probs = jnp.array([0.5, 0.0, 0.0, 0.5])
        np.testing.assert_array_almost_equal(probabilities, expected_probs, decimal=6)
        
        # Check normalization
        assert np.isclose(jnp.sum(probabilities), 1.0)
    
    def test_get_counts(self, sample_bell_circuit):
        """Test measurement counts retrieval."""
        sim = JAXSimulator(num_qubits=2, seed=42)
        
        counts = sim.get_counts(sample_bell_circuit, shots=1000)
        
        assert isinstance(counts, dict)
        assert sum(counts.values()) == 1000
        assert set(counts.keys()).issubset({"00", "01", "10", "11"})
    
    def test_measure_observable(self, sample_bell_circuit, sample_pauli_observable):
        """Test single observable measurement."""
        sim = JAXSimulator(num_qubits=2)
        
        expectation = sim.measure_observable(sample_bell_circuit, sample_pauli_observable)
        
        # For Bell state, ZZ expectation should be 1
        assert np.isclose(expectation, 1.0, atol=1e-6)
    
    def test_reset_simulator(self):
        """Test simulator reset functionality."""
        sim = JAXSimulator(num_qubits=2, seed=42)
        
        # Run a circuit to change internal state
        circuit = JAXCircuit(num_qubits=2)
        circuit.h(0)
        sim.run(circuit, shots=100)
        
        # Reset simulator
        sim.reset()
        
        # Key should be reset to default
        # Note: This is a simplified test as we can't easily verify key state
        assert sim.key is not None
    
    def test_backend_info(self):
        """Test backend information retrieval."""
        sim = JAXSimulator(num_qubits=3, precision="float32")
        
        info = sim.backend_info
        
        assert isinstance(info, dict)
        assert "backend" in info
        assert "devices" in info
        assert info["num_qubits"] == 3
        assert info["precision"] == "float32"
        assert info["hilbert_space_size"] == 8
        assert "memory_usage_gb" in info
    
    def test_simulator_utilities(self):
        """Test utility functions for creating simulators."""
        from qem_bench.jax.simulator import (
            create_cpu_simulator,
            create_gpu_simulator,
            create_tpu_simulator
        )
        
        # Test CPU simulator creation
        cpu_sim = create_cpu_simulator(num_qubits=2)
        assert isinstance(cpu_sim, JAXSimulator)
        assert cpu_sim.num_qubits == 2
        
        # Test GPU simulator creation (may fallback to CPU)
        gpu_sim = create_gpu_simulator(num_qubits=2)
        assert isinstance(gpu_sim, JAXSimulator)
        
        # Test TPU simulator creation (may fallback to CPU)
        tpu_sim = create_tpu_simulator(num_qubits=2)
        assert isinstance(tpu_sim, JAXSimulator)


class TestSimulationResult:
    """Test suite for SimulationResult dataclass."""
    
    def test_simulation_result_creation(self):
        """Test SimulationResult creation and attributes."""
        statevector = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.complex64)
        counts = {"00": 800, "01": 200}
        expectation_values = {"Z": 0.6}
        
        result = SimulationResult(
            statevector=statevector,
            measurement_counts=counts,
            expectation_values=expectation_values,
            execution_time=1.5,
            memory_usage=1024
        )
        
        np.testing.assert_array_equal(result.statevector, statevector)
        assert result.measurement_counts == counts
        assert result.expectation_values == expectation_values
        assert result.execution_time == 1.5
        assert result.memory_usage == 1024
    
    def test_simulation_result_optional_fields(self):
        """Test SimulationResult with optional fields."""
        statevector = jnp.array([1.0, 0.0], dtype=jnp.complex64)
        
        result = SimulationResult(statevector=statevector)
        
        np.testing.assert_array_equal(result.statevector, statevector)
        assert result.measurement_counts is None
        assert result.expectation_values is None
        assert result.execution_time is None
        assert result.memory_usage is None