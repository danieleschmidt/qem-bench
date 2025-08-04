"""Noise profiler for comprehensive device characterization."""

import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time

from ..models.base import NoiseModel
from ..models.amplitude_damping import AmplitudeDampingNoiseModel
from ..models.phase_damping import PhaseDampingNoiseModel
from ..models.readout import ReadoutErrorNoiseModel
from ..models.crosstalk import CrosstalkNoiseModel
from ..models.depolarizing import DepolarizingNoise


@dataclass
class CharacterizationResult:
    """Result from noise characterization measurement."""
    
    parameter_name: str
    qubit_index: Optional[int]
    value: float
    error: float
    unit: str
    measurement_time: float
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class DeviceProfile:
    """Complete noise profile of a quantum device."""
    
    device_name: str
    num_qubits: int
    characterization_time: float
    
    # Single-qubit parameters
    t1_times: Dict[int, float]
    t2_times: Dict[int, float]
    t2_star_times: Dict[int, float]
    frequencies: Dict[int, float]
    anharmonicities: Dict[int, float]
    
    # Gate fidelities
    single_qubit_fidelities: Dict[int, Dict[str, float]]
    two_qubit_fidelities: Dict[Tuple[int, int], Dict[str, float]]
    
    # Readout parameters
    readout_fidelities: Dict[int, float]
    readout_assignment_matrices: Dict[int, jnp.ndarray]
    
    # Crosstalk parameters
    zz_couplings: Dict[Tuple[int, int], float]
    crosstalk_matrix: jnp.ndarray
    
    # Error rates
    single_qubit_error_rates: Dict[int, float]
    two_qubit_error_rates: Dict[Tuple[int, int], float]
    
    # Temporal stability
    drift_rates: Dict[str, Dict[int, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "device_name": self.device_name,
            "num_qubits": self.num_qubits,
            "characterization_time": self.characterization_time,
            "t1_times": self.t1_times,
            "t2_times": self.t2_times,
            "t2_star_times": self.t2_star_times,
            "frequencies": self.frequencies,
            "anharmonicities": self.anharmonicities,
            "single_qubit_fidelities": self.single_qubit_fidelities,
            "two_qubit_fidelities": {
                f"{k[0]}_{k[1]}": v for k, v in self.two_qubit_fidelities.items()
            },
            "readout_fidelities": self.readout_fidelities,
            "readout_assignment_matrices": {
                str(k): v.tolist() for k, v in self.readout_assignment_matrices.items()
            },
            "zz_couplings": {
                f"{k[0]}_{k[1]}": v for k, v in self.zz_couplings.items()
            },
            "crosstalk_matrix": self.crosstalk_matrix.tolist(),
            "single_qubit_error_rates": self.single_qubit_error_rates,
            "two_qubit_error_rates": {
                f"{k[0]}_{k[1]}": v for k, v in self.two_qubit_error_rates.items()
            },
            "drift_rates": self.drift_rates
        }


class NoiseProfiler:
    """
    Comprehensive noise profiler for quantum devices.
    
    Performs systematic characterization of device noise parameters
    using various measurement protocols.
    """
    
    def __init__(
        self,
        backend: Any = None,
        num_shots: int = 8192,
        confidence_level: float = 0.95
    ):
        """
        Initialize noise profiler.
        
        Args:
            backend: Quantum backend for measurements
            num_shots: Number of shots per measurement
            confidence_level: Statistical confidence level
        """
        self.backend = backend
        self.num_shots = num_shots
        self.confidence_level = confidence_level
        
        # Measurement results storage
        self.results: List[CharacterizationResult] = []
        self.profile: Optional[DeviceProfile] = None
    
    def characterize_device(
        self,
        device_name: str,
        num_qubits: int,
        measurement_suite: Optional[List[str]] = None
    ) -> DeviceProfile:
        """
        Perform comprehensive device characterization.
        
        Args:
            device_name: Name of the device
            num_qubits: Number of qubits to characterize
            measurement_suite: List of measurements to perform
            
        Returns:
            Complete device noise profile
        """
        start_time = time.time()
        
        if measurement_suite is None:
            measurement_suite = [
                "t1_measurement",
                "t2_ramsey",
                "t2_echo", 
                "spectroscopy",
                "single_qubit_rb",
                "two_qubit_rb",
                "readout_calibration",
                "crosstalk_measurement",
                "process_tomography"
            ]
        
        print(f"Starting comprehensive characterization of {device_name}")
        print(f"Measurement suite: {measurement_suite}")
        
        # Initialize profile structure
        profile_data = {
            "device_name": device_name,
            "num_qubits": num_qubits,
            "t1_times": {},
            "t2_times": {},
            "t2_star_times": {},
            "frequencies": {},
            "anharmonicities": {},
            "single_qubit_fidelities": {},
            "two_qubit_fidelities": {},
            "readout_fidelities": {},
            "readout_assignment_matrices": {},
            "zz_couplings": {},
            "crosstalk_matrix": jnp.zeros((num_qubits, num_qubits)),
            "single_qubit_error_rates": {},
            "two_qubit_error_rates": {},
            "drift_rates": {}
        }
        
        # Perform measurements
        for measurement in measurement_suite:
            print(f"Running {measurement}...")
            
            if measurement == "t1_measurement":
                self._measure_t1_times(num_qubits, profile_data)
            elif measurement == "t2_ramsey":
                self._measure_t2_star_ramsey(num_qubits, profile_data)
            elif measurement == "t2_echo":
                self._measure_t2_echo(num_qubits, profile_data)
            elif measurement == "spectroscopy":
                self._measure_spectroscopy(num_qubits, profile_data)
            elif measurement == "single_qubit_rb":
                self._measure_single_qubit_rb(num_qubits, profile_data)
            elif measurement == "two_qubit_rb":
                self._measure_two_qubit_rb(num_qubits, profile_data)
            elif measurement == "readout_calibration":
                self._measure_readout_fidelity(num_qubits, profile_data)
            elif measurement == "crosstalk_measurement":
                self._measure_crosstalk(num_qubits, profile_data)
            elif measurement == "process_tomography":
                self._measure_process_tomography(num_qubits, profile_data)
        
        # Create device profile
        characterization_time = time.time() - start_time
        profile_data["characterization_time"] = characterization_time
        
        self.profile = DeviceProfile(**profile_data)
        
        print(f"Characterization complete in {characterization_time:.2f} seconds")
        return self.profile
    
    def _measure_t1_times(self, num_qubits: int, profile_data: Dict[str, Any]) -> None:
        """Measure T1 relaxation times for all qubits."""
        for qubit in range(num_qubits):
            # Simulate T1 measurement with varying delay times
            delay_times = jnp.logspace(-1, 2, 20)  # 0.1 to 100 μs
            populations = []
            
            for delay in delay_times:
                # Simulate decay: P(1) = exp(-delay/T1) 
                # Add noise for realistic measurement
                true_t1 = 50.0 + 20.0 * np.random.random()  # Random T1 between 50-70 μs
                theoretical_pop = np.exp(-delay / true_t1)
                measured_pop = theoretical_pop + np.random.normal(0, 0.02)  # 2% noise
                populations.append(max(0, min(1, measured_pop)))
            
            # Fit exponential decay
            try:
                # Simple exponential fit: y = exp(-x/T1)
                log_populations = np.log(np.array(populations) + 1e-10)
                coeffs = np.polyfit(delay_times, log_populations, 1)
                t1_measured = -1.0 / coeffs[0]
                
                # Estimate error from fit quality
                fit_error = np.std(log_populations - np.polyval(coeffs, delay_times))
                error = t1_measured * fit_error
                
            except:
                t1_measured = 50.0  # Default value
                error = 5.0
            
            profile_data["t1_times"][qubit] = float(t1_measured)
            
            # Store result
            result = CharacterizationResult(
                parameter_name="T1",
                qubit_index=qubit,
                value=float(t1_measured),
                error=float(error),
                unit="μs",
                measurement_time=time.time(),
                confidence=0.95,
                metadata={"delay_times": delay_times.tolist(), "populations": populations}
            )
            self.results.append(result)
    
    def _measure_t2_star_ramsey(self, num_qubits: int, profile_data: Dict[str, Any]) -> None:
        """Measure T2* times using Ramsey interferometry.""" 
        for qubit in range(num_qubits):
            # Simulate Ramsey measurement
            delay_times = jnp.linspace(0, 50, 25)  # 0 to 50 μs
            visibilities = []
            
            for delay in delay_times:
                # Simulate Ramsey decay with detuning
                true_t2_star = 20.0 + 10.0 * np.random.random()  # 20-30 μs
                detuning = 0.1 + 0.05 * np.random.random()  # MHz
                
                # Ramsey signal: V * exp(-delay/T2*) * cos(2π * detuning * delay)
                visibility = np.exp(-delay / true_t2_star)
                oscillation = np.cos(2 * np.pi * detuning * delay)
                signal = visibility * oscillation
                
                # Add measurement noise
                measured_signal = signal + np.random.normal(0, 0.05)
                visibilities.append(measured_signal)
            
            # Extract T2* from envelope decay
            try:
                # Fit envelope decay
                envelope = np.abs(np.array(visibilities))
                log_envelope = np.log(envelope + 1e-10)
                coeffs = np.polyfit(delay_times, log_envelope, 1)
                t2_star_measured = -1.0 / coeffs[0]
                
                fit_error = np.std(log_envelope - np.polyval(coeffs, delay_times))
                error = t2_star_measured * fit_error
                
            except:
                t2_star_measured = 20.0
                error = 2.0
            
            profile_data["t2_star_times"][qubit] = float(t2_star_measured)
            
            result = CharacterizationResult(
                parameter_name="T2*",
                qubit_index=qubit,
                value=float(t2_star_measured),
                error=float(error),
                unit="μs",
                measurement_time=time.time(),
                confidence=0.95,
                metadata={"delay_times": delay_times.tolist(), "visibilities": visibilities}
            )
            self.results.append(result)
    
    def _measure_t2_echo(self, num_qubits: int, profile_data: Dict[str, Any]) -> None:
        """Measure T2 times using spin echo."""
        for qubit in range(num_qubits):
            # Simulate echo measurement
            delay_times = jnp.linspace(0, 100, 20)  # 0 to 100 μs
            echo_amplitudes = []
            
            for delay in delay_times:
                # Echo decay: exp(-delay/T2)
                true_t2 = 40.0 + 20.0 * np.random.random()  # 40-60 μs
                theoretical_amp = np.exp(-delay / true_t2)
                measured_amp = theoretical_amp + np.random.normal(0, 0.03)
                echo_amplitudes.append(max(0, measured_amp))
            
            # Fit exponential decay
            try:
                log_amplitudes = np.log(np.array(echo_amplitudes) + 1e-10)
                coeffs = np.polyfit(delay_times, log_amplitudes, 1)
                t2_measured = -1.0 / coeffs[0]
                
                fit_error = np.std(log_amplitudes - np.polyval(coeffs, delay_times))
                error = t2_measured * fit_error
                
            except:
                t2_measured = 50.0
                error = 5.0
            
            profile_data["t2_times"][qubit] = float(t2_measured)
            
            result = CharacterizationResult(
                parameter_name="T2",
                qubit_index=qubit,
                value=float(t2_measured),
                error=float(error),
                unit="μs",
                measurement_time=time.time(),
                confidence=0.95,
                metadata={"delay_times": delay_times.tolist(), "amplitudes": echo_amplitudes}
            )
            self.results.append(result)
    
    def _measure_spectroscopy(self, num_qubits: int, profile_data: Dict[str, Any]) -> None:
        """Measure qubit frequencies and anharmonicities."""
        for qubit in range(num_qubits):
            # Simulate spectroscopy measurement
            base_freq = 5.0 + 0.5 * qubit  # GHz, spaced qubits
            anharmonicity = -200.0 - 50.0 * np.random.random()  # MHz
            
            # Add calibration uncertainty
            freq_error = np.random.normal(0, 1.0)  # MHz
            measured_freq = base_freq * 1000 + freq_error  # Convert to MHz
            
            profile_data["frequencies"][qubit] = float(measured_freq)
            profile_data["anharmonicities"][qubit] = float(anharmonicity)
            
            # Store frequency result
            freq_result = CharacterizationResult(
                parameter_name="frequency",
                qubit_index=qubit,
                value=float(measured_freq),
                error=1.0,
                unit="MHz",
                measurement_time=time.time(),
                confidence=0.95,
                metadata={"base_frequency": base_freq}
            )
            self.results.append(freq_result)
            
            # Store anharmonicity result
            anharm_result = CharacterizationResult(
                parameter_name="anharmonicity",
                qubit_index=qubit,
                value=float(anharmonicity),
                error=10.0,
                unit="MHz",
                measurement_time=time.time(),
                confidence=0.95,
                metadata={}
            )
            self.results.append(anharm_result)
    
    def _measure_single_qubit_rb(self, num_qubits: int, profile_data: Dict[str, Any]) -> None:
        """Measure single-qubit gate fidelities using randomized benchmarking."""
        for qubit in range(num_qubits):
            # Simulate RB measurement
            sequence_lengths = [1, 2, 4, 8, 16, 32, 64, 128]
            survival_probs = []
            
            for length in sequence_lengths:
                # RB decay: A * p^m + B where p is fidelity
                true_fidelity = 0.999 - 0.001 * np.random.random()
                A = 0.5  # Amplitude
                B = 0.5  # Offset
                
                theoretical_prob = A * (true_fidelity ** length) + B
                measured_prob = theoretical_prob + np.random.normal(0, 0.01)
                survival_probs.append(max(0, min(1, measured_prob)))
            
            # Fit RB curve
            try:
                # Fit A * p^m + B
                def rb_func(m, A, p, B):
                    return A * (p ** m) + B
                
                # Simple approximation for fidelity
                if len(survival_probs) >= 2:
                    decay_rate = (survival_probs[0] - survival_probs[-1]) / sequence_lengths[-1]
                    estimated_p = 1 - 2 * decay_rate
                    fidelity = max(0.9, min(1.0, estimated_p))
                else:
                    fidelity = 0.999
                
                error_rate = 1 - fidelity
                error = 0.001  # Typical RB error
                
            except:
                fidelity = 0.999
                error_rate = 0.001 
                error = 0.001
            
            # Store in profile
            if qubit not in profile_data["single_qubit_fidelities"]:
                profile_data["single_qubit_fidelities"][qubit] = {}
            
            profile_data["single_qubit_fidelities"][qubit]["average"] = float(fidelity)
            profile_data["single_qubit_error_rates"][qubit] = float(error_rate)
            
            result = CharacterizationResult(
                parameter_name="single_qubit_fidelity",
                qubit_index=qubit,
                value=float(fidelity),
                error=float(error),
                unit="",
                measurement_time=time.time(),
                confidence=0.95,
                metadata={"sequence_lengths": sequence_lengths, "survival_probs": survival_probs}
            )
            self.results.append(result)
    
    def _measure_two_qubit_rb(self, num_qubits: int, profile_data: Dict[str, Any]) -> None:
        """Measure two-qubit gate fidelities."""
        # Measure all adjacent pairs
        for qubit1 in range(num_qubits):
            for qubit2 in range(qubit1 + 1, min(qubit1 + 2, num_qubits)):  # Adjacent only
                # Simulate two-qubit RB
                sequence_lengths = [1, 2, 4, 8, 16, 32]
                survival_probs = []
                
                for length in sequence_lengths:
                    # Two-qubit gates typically have lower fidelity
                    true_fidelity = 0.99 - 0.02 * np.random.random()
                    A = 0.5
                    B = 0.5
                    
                    theoretical_prob = A * (true_fidelity ** length) + B
                    measured_prob = theoretical_prob + np.random.normal(0, 0.02)
                    survival_probs.append(max(0, min(1, measured_prob)))
                
                # Estimate fidelity
                if len(survival_probs) >= 2:
                    decay_rate = (survival_probs[0] - survival_probs[-1]) / sequence_lengths[-1]
                    estimated_p = 1 - 2 * decay_rate
                    fidelity = max(0.9, min(1.0, estimated_p))
                else:
                    fidelity = 0.99
                
                error_rate = 1 - fidelity
                
                # Store results
                qubit_pair = (qubit1, qubit2)
                profile_data["two_qubit_fidelities"][qubit_pair] = {"cnot": float(fidelity)}
                profile_data["two_qubit_error_rates"][qubit_pair] = float(error_rate)
                
                result = CharacterizationResult(
                    parameter_name="two_qubit_fidelity",
                    qubit_index=None,
                    value=float(fidelity),
                    error=0.002,
                    unit="",
                    measurement_time=time.time(),
                    confidence=0.95,
                    metadata={"qubit_pair": qubit_pair, "gate": "cnot"}
                )
                self.results.append(result)
    
    def _measure_readout_fidelity(self, num_qubits: int, profile_data: Dict[str, Any]) -> None:
        """Measure readout fidelities and assignment matrices."""
        for qubit in range(num_qubits):
            # Simulate readout calibration
            # Prepare |0⟩ and measure
            p00 = 0.98 + 0.015 * np.random.random()  # Probability of correctly reading 0
            p01 = 1 - p00  # Probability of reading 1 when prepared in 0
            
            # Prepare |1⟩ and measure  
            p11 = 0.96 + 0.02 * np.random.random()  # Probability of correctly reading 1
            p10 = 1 - p11  # Probability of reading 0 when prepared in 1
            
            # Assignment matrix
            assignment_matrix = jnp.array([
                [p00, p01],
                [p10, p11]
            ])
            
            # Readout fidelity
            readout_fidelity = (p00 + p11) / 2
            
            profile_data["readout_fidelities"][qubit] = float(readout_fidelity)
            profile_data["readout_assignment_matrices"][qubit] = assignment_matrix
            
            result = CharacterizationResult(
                parameter_name="readout_fidelity",
                qubit_index=qubit,
                value=float(readout_fidelity),
                error=0.005,
                unit="",
                measurement_time=time.time(),
                confidence=0.95,
                metadata={"assignment_matrix": assignment_matrix.tolist()}
            )
            self.results.append(result)
    
    def _measure_crosstalk(self, num_qubits: int, profile_data: Dict[str, Any]) -> None:
        """Measure crosstalk between qubits."""
        # Initialize crosstalk matrix
        crosstalk_matrix = jnp.zeros((num_qubits, num_qubits))
        
        for qubit1 in range(num_qubits):
            for qubit2 in range(num_qubits):
                if qubit1 != qubit2:
                    # Simulate crosstalk measurement
                    # Adjacent qubits have stronger crosstalk
                    if abs(qubit1 - qubit2) == 1:
                        # Adjacent qubits
                        crosstalk_strength = 0.01 + 0.005 * np.random.random()
                    else:
                        # Non-adjacent qubits
                        crosstalk_strength = 0.001 + 0.001 * np.random.random()
                    
                    crosstalk_matrix = crosstalk_matrix.at[qubit1, qubit2].set(crosstalk_strength)
                    
                    # Store ZZ couplings for adjacent qubits
                    if abs(qubit1 - qubit2) == 1:
                        qubit_pair = tuple(sorted([qubit1, qubit2]))
                        zz_coupling = crosstalk_strength * 1e6  # Convert to Hz
                        profile_data["zz_couplings"][qubit_pair] = float(zz_coupling)
        
        profile_data["crosstalk_matrix"] = crosstalk_matrix
    
    def _measure_process_tomography(self, num_qubits: int, profile_data: Dict[str, Any]) -> None:
        """Perform process tomography on key gates."""
        # This is a simplified version - full QPT is very resource intensive
        for qubit in range(min(num_qubits, 2)):  # Limit to first 2 qubits
            # Simulate process fidelity measurement for common gates
            gates = ["X", "Y", "Z", "H"]
            
            for gate in gates:
                # Simulate process fidelity
                ideal_fidelity = 0.999
                process_fidelity = ideal_fidelity - 0.005 * np.random.random()
                
                if qubit not in profile_data["single_qubit_fidelities"]:
                    profile_data["single_qubit_fidelities"][qubit] = {}
                
                profile_data["single_qubit_fidelities"][qubit][gate] = float(process_fidelity)
                
                result = CharacterizationResult(
                    parameter_name=f"{gate}_gate_fidelity",
                    qubit_index=qubit,
                    value=float(process_fidelity),
                    error=0.002,
                    unit="",
                    measurement_time=time.time(),
                    confidence=0.95,
                    metadata={"gate": gate, "method": "process_tomography"}
                )
                self.results.append(result)
    
    def create_noise_model(self, profile: Optional[DeviceProfile] = None) -> NoiseModel:
        """
        Create a comprehensive noise model from device profile.
        
        Args:
            profile: Device profile (uses self.profile if None)
            
        Returns:
            Composite noise model representing the device
        """
        if profile is None:
            if self.profile is None:
                raise ValueError("No device profile available. Run characterization first.")
            profile = self.profile
        
        from ..models.composite import CompositeNoiseModel
        
        noise_models = []
        
        # T1 relaxation model
        if profile.t1_times:
            t1_model = AmplitudeDampingNoiseModel(
                t1_times=profile.t1_times,
                gate_times={"single": 0.1, "two": 0.5, "readout": 1.0}
            )
            noise_models.append(t1_model)
        
        # T2 dephasing model
        if profile.t2_times:
            t2_model = PhaseDampingNoiseModel(
                t2_times=profile.t2_times,
                t1_times=profile.t1_times,
                gate_times={"single": 0.1, "two": 0.5, "readout": 1.0},
                pure_dephasing=False
            )
            noise_models.append(t2_model)
        
        # Readout error model
        if profile.readout_assignment_matrices:
            readout_model = ReadoutErrorNoiseModel(
                assignment_matrices=profile.readout_assignment_matrices
            )
            noise_models.append(readout_model)
        
        # Crosstalk model
        if profile.zz_couplings:
            coupling_map = {k: v for k, v in profile.zz_couplings.items()}
            crosstalk_model = CrosstalkNoiseModel(
                coupling_map=coupling_map,
                zz_couplings=profile.zz_couplings
            )
            noise_models.append(crosstalk_model)
        
        # Depolarizing model based on error rates
        if profile.single_qubit_error_rates or profile.two_qubit_error_rates:
            avg_single_error = np.mean(list(profile.single_qubit_error_rates.values())) if profile.single_qubit_error_rates else 0.001
            avg_two_error = np.mean(list(profile.two_qubit_error_rates.values())) if profile.two_qubit_error_rates else 0.01
            
            depolarizing_model = DepolarizingNoise(
                single_qubit_error_rate=avg_single_error,
                two_qubit_error_rate=avg_two_error,
                readout_error_rate=1 - np.mean(list(profile.readout_fidelities.values())) if profile.readout_fidelities else 0.02
            )
            noise_models.append(depolarizing_model)
        
        # Create composite model
        if noise_models:
            composite_model = CompositeNoiseModel(noise_models, name=f"{profile.device_name}_noise_model")
            return composite_model
        else:
            # Return empty depolarizing model as fallback
            return DepolarizingNoise(
                single_qubit_error_rate=0.001,
                two_qubit_error_rate=0.01,
                readout_error_rate=0.02
            )
    
    def get_characterization_summary(self) -> Dict[str, Any]:
        """
        Get summary of characterization results.
        
        Returns:
            Summary dictionary with key metrics
        """
        if not self.results:
            return {}
        
        summary = {
            "total_measurements": len(self.results),
            "parameters_measured": list(set(r.parameter_name for r in self.results)),
            "qubits_characterized": list(set(r.qubit_index for r in self.results if r.qubit_index is not None)),
            "measurement_time_range": (
                min(r.measurement_time for r in self.results),
                max(r.measurement_time for r in self.results)
            )
        }
        
        # Add parameter-specific summaries
        param_summaries = {}
        for param in summary["parameters_measured"]:
            param_results = [r for r in self.results if r.parameter_name == param]
            if param_results:
                values = [r.value for r in param_results]
                param_summaries[param] = {
                    "count": len(param_results),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "unit": param_results[0].unit
                }
        
        summary["parameter_summaries"] = param_summaries
        
        return summary
    
    def export_results(self, filename: str, format: str = "json") -> None:
        """
        Export characterization results to file.
        
        Args:
            filename: Output filename
            format: Export format ("json", "csv")
        """
        if format == "json":
            import json
            
            export_data = {
                "device_profile": self.profile.to_dict() if self.profile else None,
                "characterization_results": [
                    {
                        "parameter_name": r.parameter_name,
                        "qubit_index": r.qubit_index,
                        "value": float(r.value),
                        "error": float(r.error),
                        "unit": r.unit,
                        "measurement_time": float(r.measurement_time),
                        "confidence": float(r.confidence),
                        "metadata": r.metadata
                    }
                    for r in self.results
                ],
                "summary": self.get_characterization_summary()
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
                
        elif format == "csv":
            import csv
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "parameter_name", "qubit_index", "value", "error", 
                    "unit", "measurement_time", "confidence"
                ])
                
                for r in self.results:
                    writer.writerow([
                        r.parameter_name, r.qubit_index, r.value, r.error,
                        r.unit, r.measurement_time, r.confidence
                    ])
        
        print(f"Results exported to {filename}")
    
    def __str__(self) -> str:
        """String representation."""
        if self.profile:
            return f"NoiseProfiler for {self.profile.device_name} ({self.profile.num_qubits} qubits, {len(self.results)} measurements)"
        else:
            return f"NoiseProfiler ({len(self.results)} measurements)"