"""Integration test for Generation 1 basic functionality."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
import jax.numpy as jnp

# Import core components
from qem_bench.jax.circuits import JAXCircuit, create_bell_circuit, create_ghz_circuit
from qem_bench.jax.simulator import JAXSimulator
from qem_bench.jax.observables import PauliObservable, ZObservable, create_pauli_observable
from qem_bench.mitigation.zne.core import ZeroNoiseExtrapolation
from qem_bench.benchmarks.circuits.standard import create_benchmark_circuit
from qem_bench.benchmarks.metrics.fidelity import compute_fidelity
from qem_bench.benchmarks.metrics.distance import compute_tvd


class TestGeneration1BasicFunctionality:
    """Test Generation 1 basic functionality end-to-end."""
    
    def test_jax_circuit_creation(self):
        """Test basic JAX circuit creation and manipulation."""
        # Create a simple circuit
        circuit = JAXCircuit(2, name="test_circuit")
        circuit.h(0)
        circuit.cx(0, 1)
        
        # Verify circuit properties
        assert circuit.num_qubits == 2
        assert circuit.size == 2  # H + CX
        assert circuit.name == "test_circuit"
        assert len(circuit.gates) == 2
        
        # Check gate types
        assert circuit.gates[0]["name"] == "H"
        assert circuit.gates[1]["name"] == "CX"
    
    def test_bell_circuit_creation(self):
        """Test Bell state circuit creation."""
        circuit = create_bell_circuit()
        
        assert circuit.num_qubits == 2
        assert circuit.size == 2  # H + CX
        assert "bell" in circuit.name.lower()
    
    def test_jax_simulator_basic(self):
        """Test basic JAX simulator functionality."""
        # Create simulator
        simulator = JAXSimulator(num_qubits=2, seed=42)
        
        # Create simple circuit
        circuit = JAXCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)  # Bell state
        
        # Run simulation
        result = simulator.run(circuit, shots=1000)
        
        # Verify results
        assert result.statevector is not None
        assert result.measurement_counts is not None
        assert len(result.measurement_counts) <= 4  # At most 4 different outcomes
        
        # Check probabilities for Bell state
        probabilities = simulator.get_probabilities(circuit)
        assert abs(probabilities[0] - 0.5) < 0.1  # |00⟩
        assert abs(probabilities[3] - 0.5) < 0.1  # |11⟩
        assert abs(probabilities[1]) < 0.1       # |01⟩
        assert abs(probabilities[2]) < 0.1       # |10⟩
    
    def test_observable_measurement(self):
        """Test observable expectation value measurement."""
        simulator = JAXSimulator(num_qubits=1, seed=42)
        
        # Test Z measurement on |0⟩
        circuit = JAXCircuit(1)
        z_obs = ZObservable(0)
        exp_val = simulator.measure_observable(circuit, z_obs)
        assert abs(exp_val - 1.0) < 1e-6  # Should be +1
        
        # Test Z measurement on |1⟩
        circuit.x(0)
        exp_val = simulator.measure_observable(circuit, z_obs)
        assert abs(exp_val - (-1.0)) < 1e-6  # Should be -1
        
        # Test X measurement on |+⟩
        circuit = JAXCircuit(1)
        circuit.h(0)
        x_obs = create_pauli_observable("X", [0])
        exp_val = simulator.measure_observable(circuit, x_obs)
        assert abs(exp_val - 1.0) < 1e-6  # Should be +1
    
    def test_benchmark_circuit_creation(self):
        """Test benchmark circuit creation."""
        # Test quantum volume circuit
        qv_circuit = create_benchmark_circuit("quantum_volume", qubits=3, depth=3)
        assert qv_circuit.num_qubits == 3
        assert "quantum_volume" in qv_circuit.name
        
        # Test Bell state benchmark
        bell_circuit = create_benchmark_circuit("bell_state", qubits=2)
        assert bell_circuit.num_qubits == 2
        
        # Test GHZ state
        ghz_circuit = create_benchmark_circuit("ghz_state", qubits=3)
        assert ghz_circuit.num_qubits == 3
    
    def test_zne_basic_functionality(self):
        """Test basic ZNE functionality."""
        # Create a simple test setup
        simulator = JAXSimulator(num_qubits=2, seed=42)
        
        # Create Bell state circuit
        circuit = create_bell_circuit()
        
        # Define observable (Z⊗Z)
        observable = create_pauli_observable("ZZ", [0, 1])
        
        # Create mock backend that implements run_with_observable
        class MockBackend:
            def __init__(self, sim):
                self.sim = sim
            
            def run_with_observable(self, circuit, observable, shots=1024, **kwargs):
                # Apply noise scaling factor if present
                noise_factor = getattr(circuit, '_zne_noise_factor', 1.0)
                
                # Simulate the circuit
                result = self.sim.run(circuit, observables=[observable])
                exp_val = result.expectation_values[observable.name]
                
                # Add simulated noise proportional to noise factor
                noise = 0.1 * (noise_factor - 1.0) * np.random.normal()
                exp_val += noise
                
                # Mock result object
                class MockResult:
                    def __init__(self, exp_val):
                        self.expectation_value = exp_val
                
                return MockResult(exp_val)
        
        backend = MockBackend(simulator)
        
        # Create ZNE instance
        zne = ZeroNoiseExtrapolation(
            noise_factors=[1.0, 1.5, 2.0],
            extrapolator="richardson"
        )
        
        # Run ZNE mitigation
        result = zne.mitigate(circuit, backend, observable)
        
        # Verify result structure
        assert result.raw_value is not None
        assert result.mitigated_value is not None
        assert len(result.noise_factors) == 3
        assert len(result.expectation_values) == 3
        assert result.extrapolation_data is not None
        assert result.extrapolation_method in ["richardson_linear", "richardson_quadratic"]
    
    def test_fidelity_computation(self):
        """Test quantum state fidelity computation."""
        # Create two identical Bell states
        state1 = jnp.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=jnp.complex64)
        state2 = jnp.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=jnp.complex64)
        
        # Convert to density matrices
        rho1 = jnp.outer(state1, jnp.conj(state1))
        rho2 = jnp.outer(state2, jnp.conj(state2))
        
        # Compute fidelity
        fidelity = compute_fidelity(rho1, rho2)
        assert abs(fidelity - 1.0) < 1e-6  # Should be 1.0 for identical states
        
        # Test orthogonal states
        state3 = jnp.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)], dtype=jnp.complex64)
        rho3 = jnp.outer(state3, jnp.conj(state3))
        
        fidelity_orth = compute_fidelity(rho1, rho3)
        assert abs(fidelity_orth) < 1e-6  # Should be 0.0 for orthogonal states
    
    def test_trace_distance_computation(self):
        """Test trace distance computation."""
        # Create two identical Bell states
        state1 = jnp.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=jnp.complex64)
        state2 = jnp.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=jnp.complex64)
        
        # Convert to density matrices
        rho1 = jnp.outer(state1, jnp.conj(state1))
        rho2 = jnp.outer(state2, jnp.conj(state2))
        
        # Compute trace distance
        tvd = compute_tvd(rho1, rho2)
        assert abs(tvd) < 1e-6  # Should be 0.0 for identical states
        
        # Test orthogonal states
        state3 = jnp.array([1, 0, 0, 0], dtype=jnp.complex64)  # |00⟩
        state4 = jnp.array([0, 1, 0, 0], dtype=jnp.complex64)  # |01⟩
        
        rho3 = jnp.outer(state3, jnp.conj(state3))
        rho4 = jnp.outer(state4, jnp.conj(state4))
        
        tvd_orth = compute_tvd(rho3, rho4)
        assert abs(tvd_orth - 1.0) < 1e-6  # Should be 1.0 for orthogonal states
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. Create benchmark circuit
        circuit = create_benchmark_circuit("bell_state", qubits=2, bell_state=0)
        
        # 2. Create simulator
        simulator = JAXSimulator(num_qubits=2, seed=42)
        
        # 3. Create observable
        observable = create_pauli_observable("ZZ", [0, 1])
        
        # 4. Simulate ideal circuit
        ideal_result = simulator.run(circuit, observables=[observable])
        ideal_exp_val = ideal_result.expectation_values[observable.name]
        
        # 5. Simulate noisy circuit (add some artificial noise)
        noisy_state = ideal_result.statevector
        # Add small amount of noise
        noise = 0.01 * jnp.array([0.1, -0.05, 0.08, -0.03], dtype=jnp.complex64)
        noisy_state = noisy_state + noise
        noisy_state = noisy_state / jnp.linalg.norm(noisy_state)
        
        noisy_exp_val = jnp.real(jnp.conj(noisy_state) @ observable.matrix @ noisy_state)
        
        # 6. Verify we introduced error
        error = abs(float(ideal_exp_val) - float(noisy_exp_val))
        assert error > 1e-6  # Should have introduced some error
        
        # 7. Test fidelity between ideal and noisy states
        ideal_dm = jnp.outer(ideal_result.statevector, jnp.conj(ideal_result.statevector))
        noisy_dm = jnp.outer(noisy_state, jnp.conj(noisy_state))
        
        fidelity = compute_fidelity(ideal_dm, noisy_dm)
        assert 0.9 < fidelity < 1.0  # Should be high but less than 1
        
        print(f"End-to-end test passed:")
        print(f"  Ideal expectation value: {ideal_exp_val:.6f}")
        print(f"  Noisy expectation value: {noisy_exp_val:.6f}")
        print(f"  Error introduced: {error:.6f}")
        print(f"  Fidelity: {fidelity:.6f}")


def test_generation1_integration():
    """Run all Generation 1 integration tests."""
    test_suite = TestGeneration1BasicFunctionality()
    
    print("Running Generation 1 Basic Functionality Tests...")
    
    test_suite.test_jax_circuit_creation()
    print("✓ JAX circuit creation")
    
    test_suite.test_bell_circuit_creation() 
    print("✓ Bell circuit creation")
    
    test_suite.test_jax_simulator_basic()
    print("✓ JAX simulator basic")
    
    test_suite.test_observable_measurement()
    print("✓ Observable measurement")
    
    test_suite.test_benchmark_circuit_creation()
    print("✓ Benchmark circuit creation")
    
    test_suite.test_zne_basic_functionality()
    print("✓ ZNE basic functionality")
    
    test_suite.test_fidelity_computation()
    print("✓ Fidelity computation")
    
    test_suite.test_trace_distance_computation()
    print("✓ Trace distance computation")
    
    test_suite.test_end_to_end_workflow()
    print("✓ End-to-end workflow")
    
    print("\n🎉 All Generation 1 tests passed!")
    print("✨ Basic functionality is working correctly")
    
    return True


if __name__ == "__main__":
    test_generation1_integration()