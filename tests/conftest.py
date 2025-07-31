"""Pytest configuration and shared fixtures"""

import pytest
import numpy as np


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests"""
    np.random.seed(42)
    return 42


@pytest.fixture  
def sample_circuit():
    """Create a simple test circuit"""
    # Placeholder for quantum circuit creation
    return {"qubits": 2, "depth": 3}


@pytest.fixture
def mock_noise_model():
    """Create a mock noise model for testing"""
    return {
        "depolarizing_error": 0.01,
        "readout_error": 0.02,
        "coherence_time": 100.0
    }


@pytest.fixture
def sample_backend():
    """Create a mock quantum backend"""
    return "simulator"


# Test markers
def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests that require quantum hardware"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU acceleration"
    )