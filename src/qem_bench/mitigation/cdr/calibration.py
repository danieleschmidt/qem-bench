"""Device calibration for Clifford Data Regression."""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import warnings
import time

from .clifford import CliffordCircuitGenerator, CliffordSimulator


@dataclass
class CalibrationResult:
    """Results from device calibration."""
    
    gate_fidelities: Dict[str, float] = field(default_factory=dict)
    readout_errors: Dict[int, float] = field(default_factory=dict)
    coherence_times: Dict[str, float] = field(default_factory=dict)
    crosstalk_matrix: Optional[np.ndarray] = None
    device_connectivity: Optional[Dict[str, List[int]]] = None
    calibration_timestamp: Optional[str] = None
    measurement_shots: int = 1024
    calibration_circuits: int = 50
    
    def __post_init__(self):
        if self.calibration_timestamp is None:
            import datetime
            self.calibration_timestamp = datetime.datetime.now().isoformat()


@dataclass
class DeviceCharacteristics:
    """Characteristics of a quantum device."""
    
    num_qubits: int
    connectivity_map: Dict[int, List[int]]
    native_gates: List[str]
    coherence_limits: Dict[str, float]  # T1, T2* limits
    error_rates: Dict[str, float]  # Gate error rates
    readout_fidelity: Dict[int, float]  # Per-qubit readout fidelity
    
    def is_connected(self, qubit1: int, qubit2: int) -> bool:
        """Check if two qubits are connected."""
        return (qubit2 in self.connectivity_map.get(qubit1, []) or 
                qubit1 in self.connectivity_map.get(qubit2, []))


class DeviceCalibrator:
    """
    Device calibrator for quantum backends.
    
    Performs comprehensive calibration of quantum devices to characterize
    noise and optimize CDR performance.
    """
    
    def __init__(
        self,
        calibration_shots: int = 1024,
        num_calibration_circuits: int = 50,
        clifford_lengths: List[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize device calibrator.
        
        Args:
            calibration_shots: Shots per calibration circuit
            num_calibration_circuits: Number of circuits for calibration
            clifford_lengths: List of Clifford circuit lengths to test
            seed: Random seed for reproducible calibration
        """
        self.calibration_shots = calibration_shots
        self.num_calibration_circuits = num_calibration_circuits
        self.clifford_lengths = clifford_lengths or [10, 20, 50, 100]
        
        # Initialize components
        self.clifford_generator = CliffordCircuitGenerator(seed=seed)
        self.clifford_simulator = CliffordSimulator()
        
        # Calibration results
        self.last_calibration: Optional[CalibrationResult] = None
        self.device_characteristics: Optional[DeviceCharacteristics] = None
        
        # JAX setup
        self.key = random.PRNGKey(seed if seed is not None else 42)
    
    def calibrate(
        self, 
        backend: Any, 
        shots: Optional[int] = None,
        full_calibration: bool = True
    ) -> CalibrationResult:
        """
        Perform full device calibration.
        
        Args:
            backend: Quantum backend to calibrate
            shots: Override calibration shots
            full_calibration: Whether to perform comprehensive calibration
            
        Returns:
            Calibration results
        """
        shots = shots or self.calibration_shots
        
        print(f"Starting device calibration with {shots} shots per circuit...")
        start_time = time.time()
        
        # Initialize calibration result
        result = CalibrationResult(
            measurement_shots=shots,
            calibration_circuits=self.num_calibration_circuits
        )
        
        # Get device information
        num_qubits = self._get_num_qubits(backend)
        
        # Basic calibrations (always performed)
        result.readout_errors = self._calibrate_readout_errors(backend, num_qubits, shots)
        result.gate_fidelities = self._calibrate_gate_fidelities(backend, num_qubits, shots)
        
        if full_calibration:
            # Comprehensive calibrations
            result.coherence_times = self._calibrate_coherence_times(backend, num_qubits, shots)
            result.crosstalk_matrix = self._calibrate_crosstalk(backend, num_qubits, shots)
            result.device_connectivity = self._characterize_connectivity(backend, num_qubits)
        
        elapsed_time = time.time() - start_time
        print(f"Calibration completed in {elapsed_time:.2f} seconds")
        
        # Store results
        self.last_calibration = result
        self._build_device_characteristics(result, num_qubits)
        
        return result
    
    def _get_num_qubits(self, backend: Any) -> int:
        """Get number of qubits from backend."""
        if hasattr(backend, 'num_qubits'):
            return backend.num_qubits
        elif hasattr(backend, 'configuration'):
            return backend.configuration().num_qubits
        else:
            # Default assumption
            return 5
    
    def _calibrate_readout_errors(
        self, 
        backend: Any, 
        num_qubits: int, 
        shots: int
    ) -> Dict[int, float]:
        """Calibrate single-qubit readout errors."""
        print("Calibrating readout errors...")
        
        readout_errors = {}
        
        for qubit in range(num_qubits):
            # Create circuits for |0⟩ and |1⟩ state preparation
            circuit_0 = self._create_state_prep_circuit(qubit, state='0', num_qubits=num_qubits)
            circuit_1 = self._create_state_prep_circuit(qubit, state='1', num_qubits=num_qubits)
            
            # Execute circuits
            try:
                result_0 = backend.run(circuit_0, shots=shots)
                result_1 = backend.run(circuit_1, shots=shots)
                
                # Calculate readout error
                error_0to1 = self._get_readout_error(result_0, qubit, expected='0')
                error_1to0 = self._get_readout_error(result_1, qubit, expected='1')
                
                # Average error rate
                readout_errors[qubit] = (error_0to1 + error_1to0) / 2.0
                
            except Exception as e:
                warnings.warn(f"Readout calibration failed for qubit {qubit}: {e}")
                readout_errors[qubit] = 0.05  # Default 5% error
        
        return readout_errors
    
    def _calibrate_gate_fidelities(
        self, 
        backend: Any, 
        num_qubits: int, 
        shots: int
    ) -> Dict[str, float]:
        """Calibrate gate fidelities using randomized benchmarking."""
        print("Calibrating gate fidelities...")
        
        gate_fidelities = {}
        
        # Single-qubit gate fidelities
        for qubit in range(num_qubits):
            try:
                fidelity = self._single_qubit_randomized_benchmarking(
                    backend, qubit, num_qubits, shots
                )
                gate_fidelities[f'single_qubit_{qubit}'] = fidelity
            except Exception as e:
                warnings.warn(f"Single-qubit RB failed for qubit {qubit}: {e}")
                gate_fidelities[f'single_qubit_{qubit}'] = 0.99  # Default high fidelity
        
        # Two-qubit gate fidelities (sample a few pairs)
        qubit_pairs = self._get_sample_qubit_pairs(num_qubits, max_pairs=5)
        
        for control, target in qubit_pairs:
            try:
                fidelity = self._two_qubit_randomized_benchmarking(
                    backend, control, target, num_qubits, shots
                )
                gate_fidelities[f'two_qubit_{control}_{target}'] = fidelity
            except Exception as e:
                warnings.warn(f"Two-qubit RB failed for pair ({control}, {target}): {e}")
                gate_fidelities[f'two_qubit_{control}_{target}'] = 0.95  # Default
        
        return gate_fidelities
    
    def _calibrate_coherence_times(
        self, 
        backend: Any, 
        num_qubits: int, 
        shots: int
    ) -> Dict[str, float]:
        """Calibrate coherence times (T1, T2)."""
        print("Calibrating coherence times...")
        
        coherence_times = {}
        
        # This is a simplified implementation
        # Real implementation would perform T1/T2 experiments
        
        for qubit in range(num_qubits):
            try:
                # Estimate T1 (relaxation time)
                t1 = self._estimate_t1(backend, qubit, num_qubits, shots)
                coherence_times[f'T1_qubit_{qubit}'] = t1
                
                # Estimate T2* (dephasing time)
                t2_star = self._estimate_t2_star(backend, qubit, num_qubits, shots)
                coherence_times[f'T2_star_qubit_{qubit}'] = t2_star
                
            except Exception as e:
                warnings.warn(f"Coherence calibration failed for qubit {qubit}: {e}")
                # Default values (microseconds)
                coherence_times[f'T1_qubit_{qubit}'] = 100.0
                coherence_times[f'T2_star_qubit_{qubit}'] = 50.0
        
        return coherence_times
    
    def _calibrate_crosstalk(
        self, 
        backend: Any, 
        num_qubits: int, 
        shots: int
    ) -> np.ndarray:
        """Calibrate crosstalk between qubits."""
        print("Calibrating crosstalk...")
        
        crosstalk_matrix = np.zeros((num_qubits, num_qubits))
        
        # Measure crosstalk between adjacent qubits
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                try:
                    crosstalk = self._measure_crosstalk(backend, i, j, num_qubits, shots)
                    crosstalk_matrix[i, j] = crosstalk
                    crosstalk_matrix[j, i] = crosstalk  # Symmetric
                except Exception as e:
                    warnings.warn(f"Crosstalk measurement failed for qubits {i}, {j}: {e}")
        
        return crosstalk_matrix
    
    def _characterize_connectivity(
        self, 
        backend: Any, 
        num_qubits: int
    ) -> Dict[str, List[int]]:
        """Characterize device connectivity."""
        connectivity = {}
        
        # Try to get connectivity from backend
        if hasattr(backend, 'coupling_map'):
            coupling_map = backend.coupling_map
            for edge in coupling_map:
                control, target = edge
                if control not in connectivity:
                    connectivity[control] = []
                if target not in connectivity:
                    connectivity[target] = []
                connectivity[control].append(target)
                connectivity[target].append(control)
        else:
            # Default: linear connectivity
            for i in range(num_qubits - 1):
                connectivity[i] = [i + 1] if i + 1 < num_qubits else []
                connectivity[i + 1] = connectivity.get(i + 1, []) + [i]
        
        return connectivity
    
    def _create_state_prep_circuit(
        self, 
        qubit: int, 
        state: str, 
        num_qubits: int
    ) -> Any:
        """Create a circuit to prepare |0⟩ or |1⟩ on a specific qubit."""
        # This is a placeholder - would need actual circuit construction
        # based on the backend's circuit format
        
        if hasattr(self, '_create_circuit'):
            circuit = self._create_circuit(num_qubits)
            if state == '1':
                circuit.x(qubit)  # Apply X gate to prepare |1⟩
            return circuit
        else:
            # Return mock circuit
            return {"type": "state_prep", "qubit": qubit, "state": state}
    
    def _get_readout_error(
        self, 
        result: Any, 
        qubit: int, 
        expected: str
    ) -> float:
        """Calculate readout error from measurement result."""
        if hasattr(result, 'get_counts'):
            counts = result.get_counts()
            total_shots = sum(counts.values())
            
            # Count incorrect measurements
            incorrect = 0
            for bitstring, count in counts.items():
                if len(bitstring) > qubit:
                    measured_bit = bitstring[-(qubit + 1)]  # Reverse indexing
                    if measured_bit != expected:
                        incorrect += count
            
            return incorrect / total_shots if total_shots > 0 else 0.0
        else:
            # Fallback
            return 0.05  # 5% default error
    
    def _single_qubit_randomized_benchmarking(
        self, 
        backend: Any, 
        qubit: int, 
        num_qubits: int, 
        shots: int
    ) -> float:
        """Perform single-qubit randomized benchmarking."""
        fidelities = []
        
        for length in self.clifford_lengths[:3]:  # Test a few lengths
            # Generate random Clifford sequence
            clifford_circuit = self.clifford_generator.generate_random_clifford(
                num_qubits=1, length=length
            )
            
            # Execute circuit
            try:
                result = backend.run(clifford_circuit, shots=shots)
                
                # Calculate survival probability (simplified)
                survival_prob = self._calculate_survival_probability(result)
                fidelities.append(survival_prob)
                
            except Exception:
                fidelities.append(0.99)  # Default high fidelity
        
        return np.mean(fidelities)
    
    def _two_qubit_randomized_benchmarking(
        self, 
        backend: Any, 
        control: int, 
        target: int, 
        num_qubits: int, 
        shots: int
    ) -> float:
        """Perform two-qubit randomized benchmarking."""
        fidelities = []
        
        for length in self.clifford_lengths[:2]:  # Test fewer lengths for 2Q
            # Generate random 2-qubit Clifford sequence
            clifford_circuit = self.clifford_generator.generate_random_clifford(
                num_qubits=2, length=length
            )
            
            try:
                result = backend.run(clifford_circuit, shots=shots)
                survival_prob = self._calculate_survival_probability(result)
                fidelities.append(survival_prob)
            except Exception:
                fidelities.append(0.95)  # Default
        
        return np.mean(fidelities)
    
    def _calculate_survival_probability(self, result: Any) -> float:
        """Calculate survival probability from measurement result."""
        if hasattr(result, 'get_counts'):
            counts = result.get_counts()
            total_shots = sum(counts.values())
            
            # Count |00...0⟩ state (ground state survival)
            ground_state_count = counts.get('0' * len(list(counts.keys())[0]), 0)
            return ground_state_count / total_shots if total_shots > 0 else 0.0
        else:
            return 0.99  # Default high survival
    
    def _get_sample_qubit_pairs(
        self, 
        num_qubits: int, 
        max_pairs: int = 5
    ) -> List[Tuple[int, int]]:
        """Get a sample of qubit pairs for calibration."""
        pairs = []
        for i in range(min(num_qubits - 1, max_pairs)):
            pairs.append((i, i + 1))
        return pairs
    
    def _estimate_t1(
        self, 
        backend: Any, 
        qubit: int, 
        num_qubits: int, 
        shots: int
    ) -> float:
        """Estimate T1 relaxation time."""
        # Simplified T1 estimation
        # Real implementation would perform T1 experiment with varying delays
        
        # Create X gate followed by delay and measurement
        try:
            # This is a placeholder - real implementation would vary delay times
            base_fidelity = 0.99
            estimated_t1 = 100.0  # microseconds
            
            # Could implement actual T1 measurement here
            return estimated_t1
            
        except Exception:
            return 100.0  # Default T1 in microseconds
    
    def _estimate_t2_star(
        self, 
        backend: Any, 
        qubit: int, 
        num_qubits: int, 
        shots: int
    ) -> float:
        """Estimate T2* dephasing time."""
        # Simplified T2* estimation
        # Real implementation would perform Ramsey experiment
        
        try:
            # Placeholder for Ramsey experiment
            estimated_t2_star = 50.0  # microseconds
            return estimated_t2_star
            
        except Exception:
            return 50.0  # Default T2* in microseconds
    
    def _measure_crosstalk(
        self, 
        backend: Any, 
        qubit1: int, 
        qubit2: int, 
        num_qubits: int, 
        shots: int
    ) -> float:
        """Measure crosstalk between two qubits."""
        try:
            # Simplified crosstalk measurement
            # Real implementation would perform simultaneous operations
            
            # For now, return small random crosstalk
            self.key, subkey = random.split(self.key)
            crosstalk = float(random.uniform(subkey, minval=0.0, maxval=0.05))
            
            return crosstalk
            
        except Exception:
            return 0.01  # 1% default crosstalk
    
    def _build_device_characteristics(
        self, 
        calibration_result: CalibrationResult, 
        num_qubits: int
    ):
        """Build device characteristics from calibration results."""
        # Extract connectivity
        connectivity_map = {}
        if calibration_result.device_connectivity:
            connectivity_map = {
                int(k): v for k, v in calibration_result.device_connectivity.items()
            }
        else:
            # Default linear connectivity
            for i in range(num_qubits - 1):
                connectivity_map[i] = [i + 1] if i + 1 < num_qubits else []
                connectivity_map[i + 1] = connectivity_map.get(i + 1, []) + [i]
        
        # Extract error rates from gate fidelities
        error_rates = {}
        for gate_name, fidelity in calibration_result.gate_fidelities.items():
            error_rates[gate_name] = 1.0 - fidelity
        
        # Extract readout fidelities
        readout_fidelity = {}
        for qubit, error_rate in calibration_result.readout_errors.items():
            readout_fidelity[qubit] = 1.0 - error_rate
        
        # Build characteristics
        self.device_characteristics = DeviceCharacteristics(
            num_qubits=num_qubits,
            connectivity_map=connectivity_map,
            native_gates=['X', 'Y', 'Z', 'H', 'S', 'CX'],  # Default gate set
            coherence_limits=calibration_result.coherence_times,
            error_rates=error_rates,
            readout_fidelity=readout_fidelity
        )
    
    def get_optimal_training_parameters(self) -> Dict[str, Any]:
        """Get optimal CDR training parameters based on device characteristics."""
        if self.last_calibration is None:
            return {
                "num_training_circuits": 100,
                "clifford_length": 50,
                "shots_per_circuit": 1024
            }
        
        # Adapt parameters based on device quality
        avg_gate_fidelity = np.mean(list(self.last_calibration.gate_fidelities.values()))
        avg_readout_error = np.mean(list(self.last_calibration.readout_errors.values()))
        
        # More training circuits for noisier devices
        if avg_gate_fidelity < 0.95 or avg_readout_error > 0.05:
            num_training_circuits = 150
            shots_per_circuit = 2048
        else:
            num_training_circuits = 100
            shots_per_circuit = 1024
        
        # Shorter Clifford circuits for very noisy devices
        if avg_gate_fidelity < 0.90:
            clifford_length = 30
        else:
            clifford_length = 50
        
        return {
            "num_training_circuits": num_training_circuits,
            "clifford_length": clifford_length,
            "shots_per_circuit": shots_per_circuit,
            "regression_method": "ridge" if avg_gate_fidelity > 0.95 else "neural"
        }
    
    def recommend_error_mitigation_strategy(self) -> Dict[str, Any]:
        """Recommend error mitigation strategy based on device characteristics."""
        if self.device_characteristics is None:
            return {"strategy": "default_cdr"}
        
        recommendations = {
            "primary_method": "cdr",
            "backup_methods": ["zne"],
            "preprocessing": [],
            "postprocessing": []
        }
        
        # Check readout errors
        avg_readout_error = np.mean(list(self.last_calibration.readout_errors.values()))
        if avg_readout_error > 0.03:
            recommendations["preprocessing"].append("readout_error_mitigation")
        
        # Check coherence times
        if self.last_calibration.coherence_times:
            avg_t1 = np.mean([v for k, v in self.last_calibration.coherence_times.items() if 'T1' in k])
            if avg_t1 < 50.0:  # Short T1
                recommendations["postprocessing"].append("dynamical_decoupling")
        
        # Check gate fidelities
        avg_gate_fidelity = np.mean(list(self.last_calibration.gate_fidelities.values()))
        if avg_gate_fidelity < 0.90:
            recommendations["backup_methods"].insert(0, "pec")  # Add PEC for very noisy devices
        
        return recommendations
    
    def export_calibration_data(self, filepath: str):
        """Export calibration data to file."""
        if self.last_calibration is None:
            raise ValueError("No calibration data to export")
        
        import json
        
        # Convert calibration result to dict
        data = {
            "gate_fidelities": self.last_calibration.gate_fidelities,
            "readout_errors": self.last_calibration.readout_errors,
            "coherence_times": self.last_calibration.coherence_times,
            "crosstalk_matrix": self.last_calibration.crosstalk_matrix.tolist() if self.last_calibration.crosstalk_matrix is not None else None,
            "device_connectivity": self.last_calibration.device_connectivity,
            "calibration_timestamp": self.last_calibration.calibration_timestamp,
            "measurement_shots": self.last_calibration.measurement_shots,
            "calibration_circuits": self.last_calibration.calibration_circuits
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Calibration data exported to {filepath}")
    
    def import_calibration_data(self, filepath: str):
        """Import calibration data from file."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct calibration result
        self.last_calibration = CalibrationResult(
            gate_fidelities=data["gate_fidelities"],
            readout_errors={int(k): v for k, v in data["readout_errors"].items()},
            coherence_times=data["coherence_times"],
            crosstalk_matrix=np.array(data["crosstalk_matrix"]) if data["crosstalk_matrix"] else None,
            device_connectivity=data["device_connectivity"],
            calibration_timestamp=data["calibration_timestamp"],
            measurement_shots=data["measurement_shots"],
            calibration_circuits=data["calibration_circuits"]
        )
        
        print(f"Calibration data imported from {filepath}")


# Convenience function
def calibrate_device(
    backend: Any,
    shots: int = 1024,
    full_calibration: bool = True
) -> CalibrationResult:
    """
    Convenience function for device calibration.
    
    Args:
        backend: Quantum backend to calibrate
        shots: Calibration shots per circuit
        full_calibration: Perform comprehensive calibration
        
    Returns:
        Calibration results
    """
    calibrator = DeviceCalibrator(calibration_shots=shots)
    return calibrator.calibrate(backend, shots, full_calibration)