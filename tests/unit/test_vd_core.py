"""Tests for Virtual Distillation core functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

# Mock JAX components for testing without dependencies
class MockJAXCircuit:
    def __init__(self, num_qubits, name=None):
        self.num_qubits = num_qubits
        self.name = name
        self.gates = []
        self.measurements = []
    
    def copy(self):
        return MockJAXCircuit(self.num_qubits, self.name)

class MockBackend:
    def __init__(self, expectation_value=0.5):
        self.expectation_value = expectation_value
    
    def run(self, circuit, shots=1024):
        result = Mock()
        result.get_counts.return_value = {"0": shots//2, "1": shots//2}
        return result
    
    def run_with_observable(self, circuit, observable, shots=1024):
        result = Mock()
        result.expectation_value = self.expectation_value
        return result

def test_vd_config():
    """Test VDConfig dataclass."""
    from src.qem_bench.mitigation.vd.core import VDConfig
    
    config = VDConfig(num_copies=3)
    assert config.num_copies == 3
    assert config.verification_strategy == "bell"
    assert config.confidence_threshold == 0.8
    assert config.verification_kwargs == {}

def test_virtual_distillation_init():
    """Test VirtualDistillation initialization."""
    from src.qem_bench.mitigation.vd.core import VirtualDistillation
    
    # Test basic initialization
    vd = VirtualDistillation(num_copies=2)
    assert vd.num_copies == 2
    assert vd.verification_strategy_name == "bell"
    
    # Test with custom parameters
    vd = VirtualDistillation(
        num_copies=3,
        verification_strategy="ghz",
        verification_kwargs={"include_variations": False}
    )
    assert vd.num_copies == 3
    assert vd.verification_strategy_name == "ghz"
    assert vd.verification_kwargs == {"include_variations": False}
    
    # Test invalid num_copies
    with pytest.raises(ValueError, match="Number of copies must be at least 1"):
        VirtualDistillation(num_copies=0)

def test_virtual_distillation_mitigate():
    """Test VirtualDistillation.mitigate() method."""
    from src.qem_bench.mitigation.vd.core import VirtualDistillation
    
    # Mock the dependencies to avoid import issues
    import sys
    from unittest.mock import patch
    
    # Create mocks for JAX modules
    mock_jnp = Mock()
    mock_jnp.array.side_effect = lambda x: np.array(x)
    mock_jnp.zeros.side_effect = lambda *args, **kwargs: np.zeros(*args[:1])
    mock_jnp.ones.side_effect = lambda *args, **kwargs: np.ones(*args[:1])
    mock_jnp.mean.side_effect = lambda x: np.mean(x)
    mock_jnp.sum.side_effect = lambda x: np.sum(x)
    mock_jnp.linalg.norm.side_effect = lambda x: np.linalg.norm(x)
    mock_jnp.sqrt.side_effect = lambda x: np.sqrt(x)
    
    with patch.dict('sys.modules', {
        'jax.numpy': mock_jnp,
        'src.qem_bench.jax.states': Mock(),
        'src.qem_bench.jax.circuits': Mock()
    }):
        # Create VD instance
        vd = VirtualDistillation(num_copies=2, verification_strategy="product")
        
        # Mock circuit and backend
        circuit = MockJAXCircuit(2)
        backend = MockBackend(expectation_value=0.7)
        
        # Mock the internal methods to avoid complex dependencies
        vd._initialize_verification_strategy = Mock()
        vd._execute_single_circuit = Mock(return_value=0.5)
        vd._perform_mcopy_distillation = Mock(return_value=(0.8, {"test": "data"}))
        vd._run_verification_circuits = Mock(return_value=(0.9, {"verification": "data"}))
        vd._calculate_error_reduction = Mock(return_value=0.6)
        vd._calculate_error_suppression_factor = Mock(return_value=10.0)
        
        # Run mitigation
        result = vd.mitigate(circuit, backend)
        
        # Verify result
        assert result.raw_value == 0.5
        assert result.mitigated_value == 0.8
        assert result.num_copies == 2
        assert result.verification_fidelity == 0.9
        assert result.error_reduction == 0.6
        
        # Verify methods were called
        vd._initialize_verification_strategy.assert_called_once()
        vd._execute_single_circuit.assert_called_once()
        vd._perform_mcopy_distillation.assert_called_once()
        vd._run_verification_circuits.assert_called_once()

def test_vd_result():
    """Test VDResult class."""
    from src.qem_bench.mitigation.vd.result import VDResult
    
    result = VDResult(
        raw_value=0.5,
        mitigated_value=0.8,
        num_copies=2,
        verification_fidelity=0.9,
        distillation_data={"method": "mcopy"},
        error_reduction=0.6
    )
    
    assert result.raw_value == 0.5
    assert result.mitigated_value == 0.8
    assert result.num_copies == 2
    assert result.verification_fidelity == 0.9
    assert result.error_reduction == 0.6
    assert result.improvement_factor == 2.5  # 1 / (1 - 0.6)
    
    # Test summary
    summary = result.summary()
    assert "Raw value:" in summary
    assert "Mitigated value:" in summary
    assert "Number of copies (M):" in summary

def test_vd_result_validation():
    """Test VDResult validation."""
    from src.qem_bench.mitigation.vd.result import VDResult
    
    # Test invalid num_copies
    with pytest.raises(ValueError, match="Number of copies must be at least 1"):
        VDResult(
            raw_value=0.5,
            mitigated_value=0.8,
            num_copies=0,
            verification_fidelity=0.9,
            distillation_data={}
        )
    
    # Test invalid verification_fidelity
    with pytest.raises(ValueError, match="Verification fidelity must be between 0 and 1"):
        VDResult(
            raw_value=0.5,
            mitigated_value=0.8,
            num_copies=2,
            verification_fidelity=1.5,
            distillation_data={}
        )

def test_bell_state_verification():
    """Test BellStateVerification strategy."""
    from src.qem_bench.mitigation.vd.verification import BellStateVerification
    
    strategy = BellStateVerification(bell_types=["00", "01"])
    
    # Test with 2 qubits (valid)
    circuits = strategy.generate_verification_circuits(num_qubits=2, num_copies=1)
    assert len(circuits) == 2  # One for each Bell type
    
    expected_states = strategy.get_expected_states(num_qubits=2, num_copies=1)
    assert len(expected_states) == 2
    
    # Test with wrong number of qubits
    with pytest.raises(ValueError, match="Bell state verification requires exactly 2 qubits"):
        strategy.generate_verification_circuits(num_qubits=3, num_copies=1)

def test_ghz_state_verification():
    """Test GHZStateVerification strategy."""
    from src.qem_bench.mitigation.vd.verification import GHZStateVerification
    
    strategy = GHZStateVerification(include_variations=True)
    
    # Test with 3 qubits
    circuits = strategy.generate_verification_circuits(num_qubits=3, num_copies=1)
    assert len(circuits) == 2  # Standard + phase variation
    
    expected_states = strategy.get_expected_states(num_qubits=3, num_copies=1)
    assert len(expected_states) == 2
    
    # Test with 2 qubits (no variations)
    circuits = strategy.generate_verification_circuits(num_qubits=2, num_copies=1)
    assert len(circuits) == 1  # Only standard GHZ
    
    # Test with invalid qubits
    with pytest.raises(ValueError, match="GHZ state verification requires at least 2 qubits"):
        strategy.generate_verification_circuits(num_qubits=1, num_copies=1)

def test_product_state_verification():
    """Test ProductStateVerification strategy."""
    from src.qem_bench.mitigation.vd.verification import ProductStateVerification
    
    strategy = ProductStateVerification(state_types=["zero", "plus"])
    
    # Test with any number of qubits
    circuits = strategy.generate_verification_circuits(num_qubits=3, num_copies=1)
    assert len(circuits) == 2  # One for each state type
    
    expected_states = strategy.get_expected_states(num_qubits=3, num_copies=1)
    assert len(expected_states) == 2

def test_create_verification_strategy():
    """Test verification strategy factory function."""
    from src.qem_bench.mitigation.vd.verification import create_verification_strategy
    
    # Test Bell strategy
    strategy = create_verification_strategy("bell", num_qubits=2)
    from src.qem_bench.mitigation.vd.verification import BellStateVerification
    assert isinstance(strategy, BellStateVerification)
    
    # Test GHZ strategy
    strategy = create_verification_strategy("ghz", num_qubits=3)
    from src.qem_bench.mitigation.vd.verification import GHZStateVerification
    assert isinstance(strategy, GHZStateVerification)
    
    # Test invalid strategy
    with pytest.raises(ValueError, match="Unknown verification strategy"):
        create_verification_strategy("invalid", num_qubits=2)
    
    # Test incompatible qubits
    with pytest.raises(ValueError, match="Bell state verification requires exactly 2 qubits"):
        create_verification_strategy("bell", num_qubits=3)

def test_convenience_function():
    """Test convenience function for VD."""
    from unittest.mock import patch
    
    # Mock the VirtualDistillation class to avoid complex dependencies
    with patch('src.qem_bench.mitigation.vd.core.VirtualDistillation') as MockVD:
        mock_instance = Mock()
        mock_result = Mock()
        mock_instance.mitigate.return_value = mock_result
        MockVD.return_value = mock_instance
        
        from src.qem_bench.mitigation.vd.core import virtual_distillation
        
        circuit = MockJAXCircuit(2)
        backend = MockBackend()
        
        result = virtual_distillation(circuit, backend, num_copies=3)
        
        # Verify VirtualDistillation was called correctly
        MockVD.assert_called_once_with(
            num_copies=3,
            verification_strategy="bell"
        )
        mock_instance.mitigate.assert_called_once_with(circuit, backend, shots=1024)
        assert result == mock_result

if __name__ == "__main__":
    pytest.main([__file__])