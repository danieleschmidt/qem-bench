"""Noise models for quantum error mitigation benchmarking."""

# Base classes
from .base import (
    NoiseModel, 
    NoiseChannel,
    NoiseChannelFactory,
    pauli_kraus_operators,
    depolarizing_kraus_operators,
    amplitude_damping_kraus_operators,
    phase_damping_kraus_operators,
    thermal_relaxation_kraus_operators,
    generalized_amplitude_damping_kraus_operators,
    bit_flip_kraus_operators,
    phase_flip_kraus_operators,
    bit_phase_flip_kraus_operators,
    pauli_channel_kraus_operators,
    reset_kraus_operators,
    two_qubit_depolarizing_kraus_operators
)

# Noise models
from .depolarizing import DepolarizingNoise, ThermalNoise
from .coherent import CoherentNoise
from .amplitude_damping import AmplitudeDampingNoiseModel, T1DecayModel
from .phase_damping import PhaseDampingNoiseModel, T2DephaseModel, RamseyDephaseModel
from .readout import ReadoutErrorNoiseModel, UniformReadoutErrorModel, SPAMErrorModel
from .crosstalk import CrosstalkNoiseModel, GridCrosstaIkModel
from .drift import DriftNoiseModel, DriftFunctions, AgingDriftModel
from .composite import CompositeNoiseModel

# Backwards compatibility
try:
    from .readout import ReadoutNoise
except ImportError:
    ReadoutNoise = ReadoutErrorNoiseModel

try:
    from .device_models import DeviceNoiseModel, create_device_noise_model
except ImportError:
    DeviceNoiseModel = CompositeNoiseModel
    def create_device_noise_model(*args, **kwargs):
        return CompositeNoiseModel([])

__all__ = [
    # Base classes
    "NoiseModel",
    "NoiseChannel",
    "NoiseChannelFactory",
    
    # Kraus operators
    "pauli_kraus_operators",
    "depolarizing_kraus_operators", 
    "amplitude_damping_kraus_operators",
    "phase_damping_kraus_operators",
    "thermal_relaxation_kraus_operators",
    "generalized_amplitude_damping_kraus_operators",
    "bit_flip_kraus_operators",
    "phase_flip_kraus_operators",
    "bit_phase_flip_kraus_operators",
    "pauli_channel_kraus_operators",
    "reset_kraus_operators",
    "two_qubit_depolarizing_kraus_operators",
    
    # Noise models
    "DepolarizingNoise",
    "ThermalNoise",
    "CoherentNoise",
    "AmplitudeDampingNoiseModel",
    "T1DecayModel",
    "PhaseDampingNoiseModel",
    "T2DephaseModel",
    "RamseyDephaseModel",
    "ReadoutErrorNoiseModel",
    "UniformReadoutErrorModel", 
    "SPAMErrorModel",
    "CrosstalkNoiseModel",
    "GridCrosstaIkModel",
    "DriftNoiseModel",
    "DriftFunctions",
    "AgingDriftModel",
    "CompositeNoiseModel",
    
    # Legacy/compatibility
    "ReadoutNoise",
    "DeviceNoiseModel",
    "create_device_noise_model"
]