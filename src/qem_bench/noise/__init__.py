"""Comprehensive noise modeling and characterization system for QEM-Bench.

This module provides a complete framework for modeling, characterizing, and replaying
quantum device noise for error mitigation benchmarking.

Key Components:
- Noise Models: Various physical noise models (T1, T2, readout, crosstalk, etc.)
- Characterization: Tools for measuring and profiling device noise
- Replay System: Record and reproduce real device noise patterns
- Factory Functions: Convenient constructors for common noise scenarios

Example:
    >>> from qem_bench.noise import create_device_noise_model, NoiseProfiler
    >>> 
    >>> # Create a realistic device noise model
    >>> noise_model = create_device_noise_model(
    ...     device_type="superconducting",
    ...     num_qubits=5,
    ...     t1_time=100.0,  # ¼s
    ...     t2_time=50.0,   # ¼s
    ...     readout_fidelity=0.98
    ... )
    >>> 
    >>> # Characterize a device
    >>> profiler = NoiseProfiler()
    >>> profile = profiler.characterize_device("my_device", num_qubits=5)
    >>> device_noise_model = profiler.create_noise_model(profile)
"""

from typing import List
import numpy as np

# Base classes
from .models.base import (
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
from .models.depolarizing import DepolarizingNoise, ThermalNoise
from .models.coherent import CoherentNoise
from .models.amplitude_damping import AmplitudeDampingNoiseModel, T1DecayModel
from .models.phase_damping import PhaseDampingNoiseModel, T2DephaseModel, RamseyDephaseModel
from .models.readout import (
    ReadoutErrorNoiseModel,
    UniformReadoutErrorModel,
    SPAMErrorModel
)
from .models.crosstalk import CrosstalkNoiseModel, GridCrosstaIkModel
from .models.drift import DriftNoiseModel, DriftFunctions, AgingDriftModel
from .models.composite import CompositeNoiseModel

# Characterization system
from .characterization.profiler import (
    NoiseProfiler,
    DeviceProfile,
    CharacterizationResult
)

# Replay system
from .replay.recorder import (
    NoiseRecorder,
    NoiseRecord,
    NoiseTrace
)
from .replay.replayer import (
    NoiseReplayer,
    ReplayConfig
)

# Import statements that need to be checked
try:
    from .models.readout import ReadoutNoise
except ImportError:
    # Handle case where ReadoutNoise doesn't exist yet
    ReadoutNoise = ReadoutErrorNoiseModel

try:
    from .models.device_models import DeviceNoiseModel, create_device_noise_model
except ImportError:
    # Create fallback device model functionality
    DeviceNoiseModel = CompositeNoiseModel
    
    def create_device_noise_model(*args, **kwargs):
        """Fallback device model creator."""
        return CompositeNoiseModel([])

# All exported classes and functions
__all__ = [
    # Base classes
    "NoiseModel",
    "NoiseChannel", 
    "NoiseChannelFactory",
    
    # Kraus operator generators
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
    
    # Characterization
    "NoiseProfiler",
    "DeviceProfile",
    "CharacterizationResult",
    
    # Replay system
    "NoiseRecorder",
    "NoiseRecord", 
    "NoiseTrace",
    "NoiseReplayer",
    "ReplayConfig",
    
    # Factory functions
    "create_device_noise_model",
    "create_superconducting_noise_model",
    "create_trapped_ion_noise_model",
    "create_photonic_noise_model",
    "create_neutral_atom_noise_model",
    "create_simple_noise_model",
    "create_realistic_noise_model",
    "create_benchmarking_noise_models"
]


# Factory functions for common noise model scenarios
def create_superconducting_noise_model(
    num_qubits: int,
    t1_time: float = 100.0,
    t2_time: float = 50.0,
    single_gate_error: float = 0.001,
    two_gate_error: float = 0.01,
    readout_fidelity: float = 0.98,
    crosstalk_strength: float = 0.001,
    include_drift: bool = False
) -> CompositeNoiseModel:
    """
    Create a realistic superconducting qubit noise model.
    
    Args:
        num_qubits: Number of qubits
        t1_time: T1 relaxation time (¼s)
        t2_time: T2 dephasing time (¼s)
        single_gate_error: Single-qubit gate error rate
        two_gate_error: Two-qubit gate error rate
        readout_fidelity: Readout fidelity
        crosstalk_strength: Crosstalk strength
        include_drift: Include temporal drift
        
    Returns:
        Composite noise model for superconducting device
    """
    noise_models = []
    
    # T1 relaxation
    t1_times = {i: t1_time + np.random.normal(0, t1_time * 0.1) for i in range(num_qubits)}
    t1_model = AmplitudeDampingNoiseModel(t1_times=t1_times)
    noise_models.append(t1_model)
    
    # T2 dephasing
    t2_times = {i: t2_time + np.random.normal(0, t2_time * 0.1) for i in range(num_qubits)}
    t2_model = PhaseDampingNoiseModel(t2_times=t2_times, t1_times=t1_times, pure_dephasing=False)
    noise_models.append(t2_model)
    
    # Gate errors
    depol_model = DepolarizingNoise(
        single_qubit_error_rate=single_gate_error,
        two_qubit_error_rate=two_gate_error,
        readout_error_rate=1 - readout_fidelity
    )
    noise_models.append(depol_model)
    
    # Readout errors
    readout_model = UniformReadoutErrorModel(
        num_qubits=num_qubits,
        error_rate=1 - readout_fidelity
    )
    noise_models.append(readout_model)
    
    # Crosstalk (nearest neighbor)
    if crosstalk_strength > 0:
        zz_couplings = {}
        for i in range(num_qubits - 1):
            zz_couplings[(i, i + 1)] = crosstalk_strength * 1e6  # Hz
        
        crosstalk_model = CrosstalkNoiseModel(
            coupling_map=zz_couplings,
            zz_couplings=zz_couplings
        )
        noise_models.append(crosstalk_model)
    
    # Temporal drift
    if include_drift:
        freq_drift = {
            i: DriftFunctions.linear_drift(1e3, 0.0)  # 1 kHz/s drift
            for i in range(num_qubits)
        }
        drift_model = DriftNoiseModel(
            num_qubits=num_qubits,
            frequency_drift=freq_drift
        )
        noise_models.append(drift_model)
    
    return CompositeNoiseModel(
        noise_models,
        name=f"superconducting_{num_qubits}q"
    )


def create_trapped_ion_noise_model(
    num_qubits: int,
    heating_rate: float = 10.0,  # quanta/s
    dephasing_rate: float = 0.1,  # 1/s
    single_gate_error: float = 0.0001,
    two_gate_error: float = 0.001,
    readout_fidelity: float = 0.995,
    motional_modes: bool = True
) -> CompositeNoiseModel:
    """
    Create a realistic trapped ion noise model.
    
    Args:
        num_qubits: Number of ions
        heating_rate: Motional heating rate (quanta/s)
        dephasing_rate: Magnetic field dephasing rate (1/s)
        single_gate_error: Single-qubit gate error rate
        two_gate_error: Two-qubit gate error rate
        readout_fidelity: State detection fidelity
        motional_modes: Include motional mode effects
        
    Returns:
        Composite noise model for trapped ion device
    """
    noise_models = []
    
    # Long coherence times typical of trapped ions
    t1_times = {i: 10000.0 + np.random.normal(0, 1000.0) for i in range(num_qubits)}  # ~10 s
    t2_times = {i: 1000.0 + np.random.normal(0, 100.0) for i in range(num_qubits)}   # ~1 s
    
    # T1 and T2 models
    t1_model = AmplitudeDampingNoiseModel(t1_times=t1_times)
    t2_model = PhaseDampingNoiseModel(t2_times=t2_times, pure_dephasing=True)
    noise_models.extend([t1_model, t2_model])
    
    # Gate errors (typically very low for trapped ions)
    gate_model = DepolarizingNoise(
        single_qubit_error_rate=single_gate_error,
        two_qubit_error_rate=two_gate_error,
        readout_error_rate=1 - readout_fidelity
    )
    noise_models.append(gate_model)
    
    # High-fidelity readout
    readout_model = UniformReadoutErrorModel(
        num_qubits=num_qubits,
        error_rate=1 - readout_fidelity
    )
    noise_models.append(readout_model)
    
    # Motional heating effects
    if motional_modes:
        heating_drift = {
            i: DriftFunctions.linear_drift(heating_rate * 1e-6, 0.0)  # Convert to amplitude error
            for i in range(num_qubits)
        }
        heating_model = DriftNoiseModel(
            num_qubits=num_qubits,
            amplitude_drift=heating_drift
        )
        noise_models.append(heating_model)
    
    return CompositeNoiseModel(
        noise_models,
        name=f"trapped_ion_{num_qubits}q"
    )


def create_photonic_noise_model(
    num_modes: int,
    loss_rate: float = 0.1,  # Photon loss probability
    detection_efficiency: float = 0.9,
    dark_count_rate: float = 0.001,
    gate_success_prob: float = 0.5,  # Two-photon gate success
    squeezing_dB: float = 10.0
) -> CompositeNoiseModel:
    """
    Create a photonic quantum computing noise model.
    
    Args:
        num_modes: Number of optical modes
        loss_rate: Photon loss probability
        detection_efficiency: Photodetector efficiency
        dark_count_rate: Dark count probability
        gate_success_prob: Two-photon gate success probability
        squeezing_dB: Squeezing parameter in dB
        
    Returns:
        Composite noise model for photonic device
    """
    noise_models = []
    
    # Photon loss (amplitude damping)
    loss_times = {i: -1.0 / np.log(1 - loss_rate) for i in range(num_modes)}
    loss_model = AmplitudeDampingNoiseModel(t1_times=loss_times)
    noise_models.append(loss_model)
    
    # Detection errors
    readout_error = 1 - detection_efficiency + dark_count_rate
    readout_model = UniformReadoutErrorModel(
        num_qubits=num_modes,
        error_rate=readout_error
    )
    noise_models.append(readout_model)
    
    # Gate success/failure
    gate_error = 1 - gate_success_prob
    gate_model = DepolarizingNoise(
        single_qubit_error_rate=0.001,  # Single-mode gates are typically good
        two_qubit_error_rate=gate_error,
        readout_error_rate=readout_error
    )
    noise_models.append(gate_model)
    
    return CompositeNoiseModel(
        noise_models,
        name=f"photonic_{num_modes}modes"
    )


def create_neutral_atom_noise_model(
    num_atoms: int,
    trap_lifetime: float = 30.0,  # s
    rydberg_decay: float = 1.0,   # ¼s
    single_gate_error: float = 0.002,
    two_gate_error: float = 0.02,
    readout_fidelity: float = 0.97,
    blockade_error: float = 0.01
) -> CompositeNoiseModel:
    """
    Create a neutral atom (Rydberg) noise model.
    
    Args:
        num_atoms: Number of atoms
        trap_lifetime: Atom loss time (s)
        rydberg_decay: Rydberg state decay time (¼s)
        single_gate_error: Single-atom gate error rate
        two_gate_error: Rydberg blockade gate error rate
        readout_fidelity: Fluorescence detection fidelity
        blockade_error: Rydberg blockade error rate
        
    Returns:
        Composite noise model for neutral atom device
    """
    noise_models = []
    
    # Atom loss
    t1_times = {i: trap_lifetime * 1e6 for i in range(num_atoms)}  # Convert to ¼s
    loss_model = AmplitudeDampingNoiseModel(t1_times=t1_times)
    noise_models.append(loss_model)
    
    # Rydberg state decay (for two-qubit gates)
    rydberg_t1 = {i: rydberg_decay for i in range(num_atoms)}
    rydberg_model = AmplitudeDampingNoiseModel(
        t1_times=rydberg_t1,
        gate_times={"single": 0.1, "two": 1.0, "readout": 10.0}  # Two-qubit gates take longer
    )
    noise_models.append(rydberg_model)
    
    # Gate errors
    gate_model = DepolarizingNoise(
        single_qubit_error_rate=single_gate_error,
        two_qubit_error_rate=two_gate_error,
        readout_error_rate=1 - readout_fidelity
    )
    noise_models.append(gate_model)
    
    # Readout errors
    readout_model = UniformReadoutErrorModel(
        num_qubits=num_atoms,
        error_rate=1 - readout_fidelity
    )
    noise_models.append(readout_model)
    
    # Blockade errors (specialized crosstalk)
    if blockade_error > 0:
        # Model as increased error rate for nearby atoms
        blockade_couplings = {}
        for i in range(num_atoms - 1):
            blockade_couplings[(i, i + 1)] = blockade_error * 1e3  # Arbitrary units
        
        blockade_model = CrosstalkNoiseModel(
            coupling_map=blockade_couplings,
            zz_couplings=blockade_couplings
        )
        noise_models.append(blockade_model)
    
    return CompositeNoiseModel(
        noise_models,
        name=f"neutral_atom_{num_atoms}q"
    )


def create_simple_noise_model(
    num_qubits: int,
    error_rate: float = 0.001
) -> DepolarizingNoise:
    """
    Create a simple uniform depolarizing noise model.
    
    Args:
        num_qubits: Number of qubits
        error_rate: Uniform error rate for all operations
        
    Returns:
        Simple depolarizing noise model
    """
    return DepolarizingNoise(
        single_qubit_error_rate=error_rate,
        two_qubit_error_rate=error_rate * 10,
        readout_error_rate=error_rate * 20
    )


def create_realistic_noise_model(
    device_type: str,
    num_qubits: int,
    **kwargs
) -> CompositeNoiseModel:
    """
    Create a realistic noise model for a specific device type.
    
    Args:
        device_type: Type of quantum device
        num_qubits: Number of qubits
        **kwargs: Device-specific parameters
        
    Returns:
        Realistic noise model for the device type
    """
    device_type = device_type.lower()
    
    if device_type in ["superconducting", "transmon", "ibm", "google"]:
        return create_superconducting_noise_model(num_qubits, **kwargs)
    elif device_type in ["trapped_ion", "ion", "ionq", "alpine"]:
        return create_trapped_ion_noise_model(num_qubits, **kwargs)
    elif device_type in ["photonic", "optical", "xanadu", "psiqu"]:
        return create_photonic_noise_model(num_qubits, **kwargs)
    elif device_type in ["neutral_atom", "rydberg", "quera", "pasqal"]:
        return create_neutral_atom_noise_model(num_qubits, **kwargs)
    else:
        raise ValueError(f"Unknown device type: {device_type}")


def create_benchmarking_noise_models(
    num_qubits: int,
    error_rates: List[float] = None
) -> List[NoiseModel]:
    """
    Create a suite of noise models for benchmarking.
    
    Args:
        num_qubits: Number of qubits
        error_rates: List of error rates to test
        
    Returns:
        List of noise models with varying error rates
    """
    if error_rates is None:
        error_rates = [0.0001, 0.001, 0.01, 0.1]
    
    models = []
    
    for error_rate in error_rates:
        # Simple depolarizing model
        depol_model = DepolarizingNoise(
            single_qubit_error_rate=error_rate,
            two_qubit_error_rate=error_rate * 10,
            readout_error_rate=error_rate * 20
        )
        depol_model.name = f"depolarizing_{error_rate}"
        models.append(depol_model)
        
        # T1/T2 model
        t1_time = -1.0 / np.log(1 - error_rate * 10) if error_rate > 0 else 1000.0
        t2_time = t1_time / 2
        
        t1_times = {i: t1_time for i in range(num_qubits)}
        t2_times = {i: t2_time for i in range(num_qubits)}
        
        decoherence_models = [
            AmplitudeDampingNoiseModel(t1_times=t1_times),
            PhaseDampingNoiseModel(t2_times=t2_times, pure_dephasing=True)
        ]
        
        composite_model = CompositeNoiseModel(
            decoherence_models,
            name=f"decoherence_{error_rate}"
        )
        models.append(composite_model)
    
    return models


# Backwards compatibility
create_device_noise_model = create_realistic_noise_model