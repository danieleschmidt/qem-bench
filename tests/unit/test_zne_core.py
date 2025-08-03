"""Unit tests for Zero-Noise Extrapolation core functionality."""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock, patch

from qem_bench.mitigation.zne import ZeroNoiseExtrapolation, ZNEConfig
from qem_bench.mitigation.zne.extrapolation import RichardsonExtrapolator
from qem_bench.mitigation.zne.scaling import UnitaryFoldingScaler


class TestZeroNoiseExtrapolation:
    """Test suite for ZeroNoiseExtrapolation class."""
    
    def test_initialization_default(self):
        """Test ZNE initialization with default parameters."""
        zne = ZeroNoiseExtrapolation()
        
        assert zne.noise_factors == [1.0, 1.5, 2.0, 2.5, 3.0]
        assert isinstance(zne.noise_scaler, UnitaryFoldingScaler)
        assert isinstance(zne.extrapolator, RichardsonExtrapolator)
        assert isinstance(zne.config, ZNEConfig)
    
    def test_initialization_custom_parameters(self):
        """Test ZNE initialization with custom parameters."""
        noise_factors = [1.0, 2.0, 3.0]
        config = ZNEConfig(
            noise_factors=noise_factors,
            extrapolator="exponential",
            fit_bootstrap=True
        )
        
        zne = ZeroNoiseExtrapolation(
            noise_factors=noise_factors,
            extrapolator="exponential",
            config=config
        )
        
        assert zne.noise_factors == noise_factors
        assert zne.config.extrapolator == "exponential"
        assert zne.config.fit_bootstrap is True
    
    def test_invalid_noise_factors(self):
        """Test that invalid noise factors raise ValueError."""
        with pytest.raises(ValueError, match="All noise factors must be ≥ 1.0"):
            ZeroNoiseExtrapolation(noise_factors=[0.5, 1.0, 1.5])
        
        with pytest.raises(ValueError, match="At least 2 noise factors required"):
            ZeroNoiseExtrapolation(noise_factors=[1.0])
    
    def test_create_extrapolator_valid_methods(self):
        """Test creation of different extrapolator types."""
        zne = ZeroNoiseExtrapolation()
        
        # Test Richardson extrapolator
        richardson = zne._create_extrapolator("richardson")
        assert richardson.__class__.__name__ == "RichardsonExtrapolator"
        
        # Test exponential extrapolator
        exponential = zne._create_extrapolator("exponential")
        assert exponential.__class__.__name__ == "ExponentialExtrapolator"
        
        # Test polynomial extrapolator
        polynomial = zne._create_extrapolator("polynomial")
        assert polynomial.__class__.__name__ == "PolynomialExtrapolator"
    
    def test_create_extrapolator_invalid_method(self):
        """Test that invalid extrapolator method raises ValueError."""
        zne = ZeroNoiseExtrapolation()
        
        with pytest.raises(ValueError, match="Unknown extrapolator 'invalid'"):
            zne._create_extrapolator("invalid")
    
    @patch('qem_bench.mitigation.zne.core.ZeroNoiseExtrapolation._execute_noise_scaled_circuits')
    @patch('qem_bench.mitigation.zne.core.ZeroNoiseExtrapolation._extrapolate_to_zero_noise')
    def test_mitigate_basic_flow(self, mock_extrapolate, mock_execute):
        """Test basic mitigation flow."""
        # Setup mocks
        noise_factors = [1.0, 1.5, 2.0]
        expectation_values = [0.8, 0.7, 0.6]
        mock_execute.return_value = (noise_factors, expectation_values)
        mock_extrapolate.return_value = (0.85, {"method": "richardson", "r_squared": 0.95})
        
        # Create ZNE instance
        zne = ZeroNoiseExtrapolation(noise_factors=noise_factors)
        
        # Mock circuit and backend
        mock_circuit = Mock()
        mock_backend = Mock()
        
        # Execute mitigation
        result = zne.mitigate(mock_circuit, mock_backend, shots=1024)
        
        # Verify result
        assert result.raw_value == 0.8
        assert result.mitigated_value == 0.85
        assert result.noise_factors == noise_factors
        assert result.expectation_values == expectation_values
        assert result.extrapolation_data["method"] == "richardson"
        
        # Verify mock calls
        mock_execute.assert_called_once_with(
            mock_circuit, mock_backend, None, 1024
        )
        mock_extrapolate.assert_called_once_with(
            np.array(noise_factors), np.array(expectation_values)
        )
    
    def test_execute_noise_scaled_circuits_with_observable_support(self):
        """Test circuit execution with backend that supports observables."""
        zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 1.5])
        
        # Mock circuit
        mock_circuit = Mock()
        
        # Mock backend with observable support
        mock_backend = Mock()
        mock_backend.run_with_observable.side_effect = [
            Mock(expectation_value=0.8),  # noise factor 1.0
            Mock(expectation_value=0.7)   # noise factor 1.5
        ]
        
        # Mock observable
        mock_observable = Mock()
        
        # Mock noise scaler
        zne.noise_scaler.scale_noise = Mock(side_effect=lambda c, f: f"scaled_{f}")
        
        # Execute
        noise_values, expectation_values = zne._execute_noise_scaled_circuits(
            mock_circuit, mock_backend, mock_observable, 1024
        )
        
        # Verify results
        assert noise_values == [1.0, 1.5]
        assert expectation_values == [0.8, 0.7]
        
        # Verify backend calls
        assert mock_backend.run_with_observable.call_count == 2
        mock_backend.run_with_observable.assert_any_call(
            "scaled_1.0", mock_observable, shots=1024
        )
        mock_backend.run_with_observable.assert_any_call(
            "scaled_1.5", mock_observable, shots=1024
        )
    
    def test_execute_noise_scaled_circuits_without_observable_support(self):
        """Test circuit execution with backend that doesn't support observables."""
        zne = ZeroNoiseExtrapolation(noise_factors=[1.0, 1.5])
        
        # Mock circuit
        mock_circuit = Mock()
        
        # Mock backend without observable support
        mock_backend = Mock()
        del mock_backend.run_with_observable  # Remove the method
        mock_backend.run.side_effect = [
            Mock(get_counts=Mock(return_value={"00": 750, "11": 250})),  # factor 1.0
            Mock(get_counts=Mock(return_value={"00": 600, "11": 400}))   # factor 1.5
        ]
        
        # Mock noise scaler
        zne.noise_scaler.scale_noise = Mock(side_effect=lambda c, f: f"scaled_{f}")
        
        # Execute
        noise_values, expectation_values = zne._execute_noise_scaled_circuits(
            mock_circuit, mock_backend, None, 1000
        )
        
        # Verify results
        assert noise_values == [1.0, 1.5]
        # Check expectation values (Z⊗Z measurement)
        expected_1 = (750 - 250) / 1000  # 0.5
        expected_2 = (600 - 400) / 1000  # 0.2
        assert np.isclose(expectation_values[0], expected_1)
        assert np.isclose(expectation_values[1], expected_2)
    
    def test_extract_expectation_value_computational_basis(self):
        """Test expectation value extraction for computational basis measurement."""
        zne = ZeroNoiseExtrapolation()
        
        # Mock result with counts
        mock_result = Mock()
        mock_result.get_counts.return_value = {
            "000": 400,  # Even parity
            "001": 100,  # Odd parity  
            "010": 100,  # Odd parity
            "011": 50,   # Even parity
            "100": 100,  # Odd parity
            "101": 50,   # Even parity
            "110": 50,   # Even parity
            "111": 150   # Odd parity
        }
        
        expectation = zne._extract_expectation_value(mock_result, None)
        
        # Calculate expected value manually
        # Even parity: 400 + 50 + 50 + 50 = 550
        # Odd parity: 100 + 100 + 100 + 150 = 450
        # Total: 1000
        # Expectation: (550 - 450) / 1000 = 0.1
        assert np.isclose(expectation, 0.1)
    
    def test_extract_expectation_value_with_observable(self):
        """Test expectation value extraction with custom observable."""
        zne = ZeroNoiseExtrapolation()
        
        mock_result = Mock()
        mock_observable = Mock()
        mock_observable.expectation_value.return_value = 0.75
        
        expectation = zne._extract_expectation_value(mock_result, mock_observable)
        
        assert expectation == 0.75
        mock_observable.expectation_value.assert_called_once_with(mock_result)
    
    def test_extrapolate_to_zero_noise(self):
        """Test extrapolation to zero noise."""
        zne = ZeroNoiseExtrapolation()
        
        # Mock extrapolator
        mock_extrapolator = Mock()
        mock_extrapolator.extrapolate.return_value = (0.85, {"method": "test", "r_squared": 0.95})
        zne.extrapolator = mock_extrapolator
        
        noise_values = [1.0, 1.5, 2.0]
        expectation_values = [0.8, 0.7, 0.6]
        
        mitigated_value, fit_data = zne._extrapolate_to_zero_noise(
            noise_values, expectation_values
        )
        
        assert mitigated_value == 0.85
        assert fit_data["method"] == "test"
        assert fit_data["r_squared"] == 0.95
        
        # Verify extrapolator was called with numpy arrays
        mock_extrapolator.extrapolate.assert_called_once()
        args = mock_extrapolator.extrapolate.call_args[0]
        np.testing.assert_array_equal(args[0], np.array(noise_values))
        np.testing.assert_array_equal(args[1], np.array(expectation_values))
    
    def test_calculate_error_reduction_with_ideal_value(self):
        """Test error reduction calculation when ideal value is known."""
        zne = ZeroNoiseExtrapolation()
        
        raw_value = 0.7
        mitigated_value = 0.85
        ideal_value = 0.9
        
        error_reduction = zne._calculate_error_reduction(
            raw_value, mitigated_value, ideal_value
        )
        
        # Raw error: |0.7 - 0.9| = 0.2
        # Mitigated error: |0.85 - 0.9| = 0.05
        # Error reduction: (0.2 - 0.05) / 0.2 = 0.75
        expected_reduction = 0.75
        assert np.isclose(error_reduction, expected_reduction)
    
    def test_calculate_error_reduction_without_ideal_value(self):
        """Test error reduction calculation when ideal value is unknown."""
        zne = ZeroNoiseExtrapolation()
        
        error_reduction = zne._calculate_error_reduction(0.7, 0.85, None)
        
        assert error_reduction is None
    
    def test_calculate_error_reduction_edge_cases(self):
        """Test error reduction calculation edge cases."""
        zne = ZeroNoiseExtrapolation()
        
        # Perfect raw measurement
        error_reduction = zne._calculate_error_reduction(1.0, 0.9, 1.0)
        assert error_reduction == 0.0  # No improvement possible
        
        # Perfect mitigation
        error_reduction = zne._calculate_error_reduction(0.8, 1.0, 1.0)
        assert error_reduction == 1.0  # Complete error elimination
    
    def test_legacy_execute_method(self):
        """Test deprecated execute method for backward compatibility."""
        zne = ZeroNoiseExtrapolation()
        
        with patch.object(zne, 'mitigate') as mock_mitigate:
            mock_result = Mock()
            mock_mitigate.return_value = mock_result
            
            with pytest.warns(DeprecationWarning, match="execute\\(\\) is deprecated"):
                result = zne.execute("circuit", "backend", shots_per_factor=512)
            
            assert result == mock_result
            mock_mitigate.assert_called_once_with(
                "circuit", "backend", shots=512
            )


class TestZNEConfig:
    """Test suite for ZNEConfig dataclass."""
    
    def test_default_configuration(self):
        """Test default ZNE configuration."""
        config = ZNEConfig(noise_factors=[1.0, 1.5, 2.0])
        
        assert config.noise_factors == [1.0, 1.5, 2.0]
        assert config.extrapolator == "richardson"
        assert config.fit_bootstrap is False
        assert config.confidence_level == 0.95
        assert config.max_iterations == 1000
        assert config.convergence_threshold == 1e-6
    
    def test_custom_configuration(self):
        """Test custom ZNE configuration."""
        config = ZNEConfig(
            noise_factors=[1.0, 2.0, 3.0],
            extrapolator="exponential",
            fit_bootstrap=True,
            confidence_level=0.99,
            max_iterations=2000,
            convergence_threshold=1e-8
        )
        
        assert config.noise_factors == [1.0, 2.0, 3.0]
        assert config.extrapolator == "exponential"
        assert config.fit_bootstrap is True
        assert config.confidence_level == 0.99
        assert config.max_iterations == 2000
        assert config.convergence_threshold == 1e-8


def test_zero_noise_extrapolation_convenience_function():
    """Test the convenience function for ZNE."""
    from qem_bench.mitigation.zne.core import zero_noise_extrapolation
    
    with patch('qem_bench.mitigation.zne.core.ZeroNoiseExtrapolation') as mock_zne_class:
        mock_zne_instance = Mock()
        mock_result = Mock()
        mock_zne_instance.mitigate.return_value = mock_result
        mock_zne_class.return_value = mock_zne_instance
        
        result = zero_noise_extrapolation(
            "circuit", "backend", 
            noise_factors=[1.0, 1.5], 
            extrapolator="exponential",
            shots=2048
        )
        
        # Verify ZNE was created with correct parameters
        mock_zne_class.assert_called_once_with(
            noise_factors=[1.0, 1.5],
            extrapolator="exponential"
        )
        
        # Verify mitigate was called
        mock_zne_instance.mitigate.assert_called_once_with(
            "circuit", "backend", shots=2048
        )
        
        assert result == mock_result