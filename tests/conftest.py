"""Pytest configuration and shared fixtures for QEM-Bench testing."""

import pytest
import numpy as np
import jax.numpy as jnp
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

# Import QEM-Bench modules for fixtures
from qem_bench.jax.circuits import JAXCircuit
from qem_bench.jax.simulator import JAXSimulator
from qem_bench.noise.models import DepolarizingNoise
from qem_bench.data.storage import InMemoryStorageBackend
from qem_bench.data.models import ExperimentResult


@pytest.fixture(scope="session", autouse=True)
def configure_jax():
    """Configure JAX for testing environment."""
    import jax
    # Use CPU for tests to avoid GPU memory issues
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", False)  # Use float32 for speed


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_jax_circuit():
    """Create a sample JAX quantum circuit for testing."""
    circuit = JAXCircuit(num_qubits=3, name="test_circuit")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.ry(np.pi/4, 2)
    circuit.cx(1, 2)
    return circuit


@pytest.fixture
def sample_bell_circuit():
    """Create a Bell state preparation circuit."""
    circuit = JAXCircuit(num_qubits=2, name="bell_circuit")
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


@pytest.fixture
def sample_ghz_circuit():
    """Create a GHZ state preparation circuit."""
    circuit = JAXCircuit(num_qubits=3, name="ghz_circuit")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    return circuit


@pytest.fixture
def jax_simulator():
    """Create a JAX quantum simulator for testing."""
    return JAXSimulator(num_qubits=3, precision="float32", seed=42)


@pytest.fixture
def large_jax_simulator():
    """Create a larger JAX simulator for performance tests."""
    return JAXSimulator(num_qubits=5, precision="float32", seed=42)


@pytest.fixture
def mock_hardware_backend():
    """Create a mock hardware backend for testing."""
    backend = Mock()
    backend.name = "mock_hardware"
    backend.num_qubits = 5
    backend.run.return_value = Mock(
        get_counts=Mock(return_value={"00000": 512, "11111": 512})
    )
    backend.configuration.return_value = Mock(
        n_qubits=5,
        coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]],
        basis_gates=["cx", "u1", "u2", "u3"]
    )
    return backend


@pytest.fixture
def depolarizing_noise_model():
    """Create a depolarizing noise model for testing."""
    return DepolarizingNoise(
        single_qubit_error_rate=0.001,
        two_qubit_error_rate=0.01,
        readout_error_rate=0.02
    )


@pytest.fixture
def sample_experiment_result():
    """Create a sample experiment result for testing."""
    return ExperimentResult(
        experiment_name="test_experiment",
        circuit_name="test_circuit", 
        circuit_qubits=3,
        circuit_depth=5,
        mitigation_method="ZNE",
        backend_name="simulator",
        shots=1024,
        noise_factors=[1.0, 1.5, 2.0],
        raw_expectation_values={"1.0": 0.8, "1.5": 0.7, "2.0": 0.6},
        mitigated_expectation_value=0.85,
        fidelity=0.92,
        error_reduction=0.15,
        execution_time=5.2
    )


@pytest.fixture
def storage_backend():
    """Create an in-memory storage backend for testing."""
    return InMemoryStorageBackend()


@pytest.fixture
def sample_measurement_counts():
    """Generate sample measurement counts for testing."""
    return {
        "000": 256,
        "001": 128,
        "010": 64,
        "011": 32,
        "100": 16,
        "101": 8,
        "110": 4,
        "111": 2
    }


@pytest.fixture
def sample_statevector():
    """Create a sample quantum state vector."""
    # Normalized random state vector
    state = np.random.random(8) + 1j * np.random.random(8)
    state = state / np.linalg.norm(state)
    return jnp.array(state, dtype=jnp.complex64)


@pytest.fixture
def sample_density_matrix():
    """Create a sample density matrix."""
    # Random density matrix
    n = 4  # 2-qubit system
    A = np.random.random((n, n)) + 1j * np.random.random((n, n))
    rho = A @ A.conj().T
    rho = rho / np.trace(rho)  # Normalize
    return jnp.array(rho, dtype=jnp.complex64)


@pytest.fixture 
def sample_pauli_observable():
    """Create sample Pauli observable matrices."""
    from qem_bench.jax.observables import PauliObservable
    
    # Z ⊗ Z observable for 2-qubit system
    zz_matrix = jnp.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0], 
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=jnp.complex64)
    
    return PauliObservable("ZZ", zz_matrix)


@pytest.fixture
def benchmark_circuits():
    """Create a collection of benchmark circuits."""
    circuits = []
    
    # Bell state
    bell = JAXCircuit(2, name="bell")
    bell.h(0).cx(0, 1)
    circuits.append(bell)
    
    # GHZ state
    ghz = JAXCircuit(3, name="ghz")
    ghz.h(0).cx(0, 1).cx(0, 2)
    circuits.append(ghz)
    
    # Random circuit
    random_circuit = JAXCircuit(3, name="random")
    random_circuit.h(0).ry(np.pi/3, 1).cx(0, 2).rz(np.pi/4, 1)
    circuits.append(random_circuit)
    
    return circuits


@pytest.fixture(params=[1, 2, 3])
def parameterized_qubit_count(request):
    """Parametrized fixture for different qubit counts."""
    return request.param


@pytest.fixture(params=["float32", "float64"])
def precision_type(request):
    """Parametrized fixture for different precision types."""
    return request.param


@pytest.fixture(params=[1.0, 1.5, 2.0, 2.5])
def noise_factor(request):
    """Parametrized fixture for different noise factors."""
    return request.param


# Test data generators
def generate_random_circuits(num_circuits: int = 5) -> List[JAXCircuit]:
    """Generate random test circuits."""
    circuits = []
    for i in range(num_circuits):
        n_qubits = np.random.randint(2, 5)
        depth = np.random.randint(3, 10)
        
        circuit = JAXCircuit(n_qubits, name=f"random_{i}")
        
        for _ in range(depth):
            # Random single-qubit gate
            qubit = np.random.randint(n_qubits)
            gate_type = np.random.choice(["h", "x", "y", "z", "ry"])
            
            if gate_type == "ry":
                angle = np.random.uniform(0, 2*np.pi)
                circuit.ry(angle, qubit)
            else:
                getattr(circuit, gate_type)(qubit)
            
            # Random two-qubit gate
            if np.random.random() > 0.5 and n_qubits > 1:
                qubits = np.random.choice(n_qubits, 2, replace=False)
                circuit.cx(qubits[0], qubits[1])
        
        circuits.append(circuit)
    
    return circuits


# Test markers and configuration  
def pytest_configure(config):
    """Configure custom markers for test categorization."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests that require quantum hardware"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU acceleration" 
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests that measure performance"
    )
    config.addinivalue_line(
        "markers", "regression: marks tests for regression testing"
    )


@pytest.fixture(autouse=True)
def cleanup_jax():
    """Clean up JAX state after each test."""
    yield
    # Clear any JAX compilation cache to prevent memory issues
    import jax
    try:
        jax.clear_caches()
    except:
        pass  # Ignore if not available in this JAX version


# Mock fixtures for external dependencies
@pytest.fixture
def mock_ibm_backend():
    """Mock IBM Quantum backend for testing.""" 
    backend = Mock()
    backend.name = "ibmq_qasm_simulator"
    backend.status.return_value = Mock(operational=True, pending_jobs=0)
    backend.configuration.return_value = Mock(
        n_qubits=32,
        coupling_map=[[i, i+1] for i in range(31)],
        basis_gates=["id", "rz", "sx", "x", "cx"]
    )
    backend.properties.return_value = Mock(
        gate_error=Mock(return_value=0.001),
        readout_error=Mock(return_value=0.02)
    )
    return backend


@pytest.fixture
def mock_aws_braket_device():
    """Mock AWS Braket device for testing."""
    device = Mock()
    device.name = "SV1"
    device.type = "SIMULATOR"
    device.provider_name = "Amazon Braket"
    device.properties = Mock(
        paradigm=Mock(qubitCount=34),
        service=Mock(shotsRange=[1, 100000])
    )
    return device