"""Crosstalk noise model implementation."""

import jax.numpy as jnp
import numpy as np
from typing import List, Any, Dict, Optional, Tuple, Set
from .base import NoiseModel, NoiseChannel


class CrosstalkNoiseModel(NoiseModel):
    """
    Crosstalk noise model for qubit-qubit interactions.
    
    Models unwanted interactions between qubits, including:
    - Static ZZ coupling between neighboring qubits
    - Gate-dependent crosstalk during two-qubit operations
    - Frequency crowding effects
    - Control line crosstalk
    - Parasitic coupling through shared resonators
    """
    
    def __init__(
        self,
        coupling_map: Dict[Tuple[int, int], float],
        zz_couplings: Optional[Dict[Tuple[int, int], float]] = None,
        gate_crosstalk: Optional[Dict[str, Dict[Tuple[int, int], float]]] = None,
        control_crosstalk: Optional[Dict[int, Dict[int, float]]] = None,
        frequency_crowding: Optional[Dict[int, float]] = None
    ):
        """
        Initialize crosstalk noise model.
        
        Args:
            coupling_map: Base coupling strengths between qubit pairs (Hz)
            zz_couplings: Static ZZ coupling strengths (Hz)
            gate_crosstalk: Gate-specific crosstalk parameters
                          {'gate_name': {(control, target): strength}}
            control_crosstalk: Control line crosstalk between qubits
                             {qubit: {neighbor: strength}}
            frequency_crowding: Frequency shift due to crowding effects (Hz)
        """
        super().__init__("crosstalk")
        
        self.coupling_map = coupling_map
        self.zz_couplings = zz_couplings or {}
        self.gate_crosstalk = gate_crosstalk or {}
        self.control_crosstalk = control_crosstalk or {}
        self.frequency_crowding = frequency_crowding or {}
        
        # Build adjacency information
        self.adjacent_qubits = self._build_adjacency_map()
        
        # Create crosstalk channels
        self._create_channels()
    
    def _build_adjacency_map(self) -> Dict[int, Set[int]]:
        """Build adjacency map from coupling information."""
        adjacency = {}
        
        # From coupling map
        for (q1, q2), _ in self.coupling_map.items():
            if q1 not in adjacency:
                adjacency[q1] = set()
            if q2 not in adjacency:
                adjacency[q2] = set()
            adjacency[q1].add(q2)
            adjacency[q2].add(q1)
        
        # From ZZ couplings
        for (q1, q2), _ in self.zz_couplings.items():
            if q1 not in adjacency:
                adjacency[q1] = set()
            if q2 not in adjacency:
                adjacency[q2] = set()
            adjacency[q1].add(q2)
            adjacency[q2].add(q1)
        
        return adjacency
    
    def _create_channels(self) -> None:
        """Create crosstalk noise channels."""
        # Static ZZ crosstalk channels
        for (q1, q2), coupling_strength in self.zz_couplings.items():
            if coupling_strength > 0:
                kraus_ops = self._create_zz_crosstalk_kraus(coupling_strength)
                channel = NoiseChannel(
                    name=f"zz_crosstalk_q{q1}_q{q2}",
                    kraus_operators=kraus_ops,
                    probability=1.0,
                    qubits=[q1, q2]
                )
                self.add_channel(channel)
        
        # Gate-dependent crosstalk channels
        for gate_name, crosstalk_params in self.gate_crosstalk.items():
            for (q1, q2), strength in crosstalk_params.items():
                if strength > 0:
                    kraus_ops = self._create_gate_crosstalk_kraus(gate_name, strength)
                    channel = NoiseChannel(
                        name=f"{gate_name}_crosstalk_q{q1}_q{q2}",
                        kraus_operators=kraus_ops,
                        probability=1.0,
                        qubits=[q1, q2]
                    )
                    self.add_channel(channel)
        
        # Control line crosstalk channels
        for qubit, neighbors in self.control_crosstalk.items():
            for neighbor, strength in neighbors.items():
                if strength > 0:
                    kraus_ops = self._create_control_crosstalk_kraus(strength)
                    channel = NoiseChannel(
                        name=f"control_crosstalk_q{qubit}_q{neighbor}",
                        kraus_operators=kraus_ops,
                        probability=1.0,
                        qubits=[qubit, neighbor]
                    )
                    self.add_channel(channel)
        
        # Frequency crowding channels (single-qubit)
        for qubit, freq_shift in self.frequency_crowding.items():
            if abs(freq_shift) > 0:
                kraus_ops = self._create_frequency_crowding_kraus(freq_shift)
                channel = NoiseChannel(
                    name=f"frequency_crowding_q{qubit}",
                    kraus_operators=kraus_ops,
                    probability=1.0,
                    qubits=[qubit]
                )
                self.add_channel(channel)
    
    def _create_zz_crosstalk_kraus(self, coupling_strength: float) -> List[jnp.ndarray]:
        """
        Create Kraus operators for static ZZ crosstalk.
        
        ZZ crosstalk: exp(-i * J * σ_z ⊗ σ_z * t)
        
        Args:
            coupling_strength: ZZ coupling strength (Hz)
            
        Returns:
            List of Kraus operators
        """
        # Assume small coupling for perturbative treatment
        # For a two-qubit ZZ interaction: H = J * Z ⊗ Z
        
        # Typical gate time for crosstalk effect
        gate_time = 0.1e-6  # 100 ns in seconds
        theta = coupling_strength * 2 * np.pi * gate_time
        
        # ZZ rotation matrix
        cos_theta = jnp.cos(theta / 2)
        sin_theta = jnp.sin(theta / 2)
        
        # Two-qubit ZZ rotation
        zz_matrix = jnp.array([
            [cos_theta, 0, 0, -1j * sin_theta],
            [0, cos_theta, 1j * sin_theta, 0],
            [0, 1j * sin_theta, cos_theta, 0],
            [-1j * sin_theta, 0, 0, cos_theta]
        ], dtype=jnp.complex64)
        
        return [zz_matrix]
    
    def _create_gate_crosstalk_kraus(self, gate_name: str, strength: float) -> List[jnp.ndarray]:
        """
        Create Kraus operators for gate-dependent crosstalk.
        
        Args:
            gate_name: Name of the gate causing crosstalk
            strength: Crosstalk strength
            
        Returns:
            List of Kraus operators
        """
        # Model as unwanted rotation on spectator qubits
        # Different gates cause different types of crosstalk
        
        if gate_name.upper() in ["CNOT", "CX"]:
            # CNOT crosstalk: unwanted ZX coupling
            theta = strength * 0.01  # Small angle
            
            cos_theta = jnp.cos(theta / 2)
            sin_theta = jnp.sin(theta / 2)
            
            # ZX crosstalk matrix
            crosstalk_matrix = jnp.array([
                [cos_theta, -1j * sin_theta, 0, 0],
                [-1j * sin_theta, cos_theta, 0, 0],
                [0, 0, cos_theta, 1j * sin_theta],
                [0, 0, 1j * sin_theta, cos_theta]
            ], dtype=jnp.complex64)
            
        elif gate_name.upper() in ["RX", "RY", "RZ"]:
            # Single-qubit rotation crosstalk
            theta = strength * 0.005  # Smaller for single-qubit
            
            cos_theta = jnp.cos(theta / 2)
            sin_theta = jnp.sin(theta / 2)
            
            # Simple two-qubit rotation
            crosstalk_matrix = jnp.array([
                [cos_theta, 0, 0, -1j * sin_theta],
                [0, cos_theta, -1j * sin_theta, 0],
                [0, -1j * sin_theta, cos_theta, 0],
                [-1j * sin_theta, 0, 0, cos_theta]
            ], dtype=jnp.complex64)
            
        else:
            # Generic crosstalk
            theta = strength * 0.001
            
            cos_theta = jnp.cos(theta / 2)
            sin_theta = jnp.sin(theta / 2)
            
            crosstalk_matrix = jnp.array([
                [cos_theta, -1j * sin_theta, 0, 0],
                [-1j * sin_theta, cos_theta, 0, 0],
                [0, 0, cos_theta, -1j * sin_theta],
                [0, 0, -1j * sin_theta, cos_theta]
            ], dtype=jnp.complex64)
        
        return [crosstalk_matrix]
    
    def _create_control_crosstalk_kraus(self, strength: float) -> List[jnp.ndarray]:
        """
        Create Kraus operators for control line crosstalk.
        
        Args:
            strength: Crosstalk strength
            
        Returns:
            List of Kraus operators
        """
        # Model as unwanted single-qubit rotation on neighbor
        theta = strength * 0.01
        
        cos_theta = jnp.cos(theta / 2)
        sin_theta = jnp.sin(theta / 2)
        
        # X rotation on second qubit due to control crosstalk
        control_crosstalk_matrix = jnp.array([
            [1, 0, 0, 0],
            [0, cos_theta, 0, -1j * sin_theta],
            [0, 0, 1, 0],
            [0, -1j * sin_theta, 0, cos_theta]
        ], dtype=jnp.complex64)
        
        return [control_crosstalk_matrix]
    
    def _create_frequency_crowding_kraus(self, freq_shift: float) -> List[jnp.ndarray]:
        """
        Create Kraus operators for frequency crowding effects.
        
        Args:
            freq_shift: Frequency shift due to crowding (Hz)
            
        Returns:
            List of Kraus operators (single-qubit)
        """
        # Model as Z rotation due to frequency shift
        gate_time = 0.1e-6  # 100 ns
        theta = 2 * np.pi * freq_shift * gate_time
        
        # Z rotation matrix
        z_rotation = jnp.array([
            [jnp.exp(-1j * theta / 2), 0],
            [0, jnp.exp(1j * theta / 2)]
        ], dtype=jnp.complex64)
        
        return [z_rotation]
    
    def get_noise_channels(self, circuit: Any) -> List[NoiseChannel]:
        """
        Get crosstalk channels for circuit gates.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            List of noise channels to apply
        """
        channels = []
        
        if hasattr(circuit, 'gates'):
            for gate_idx, gate in enumerate(circuit.gates):
                gate_name = gate.get("name", "unknown")
                gate_type = gate.get("type", "unknown")
                qubits = gate.get("qubits", [])
                
                # Apply static ZZ crosstalk between adjacent qubits
                active_qubits = set(qubits)
                for qubit in qubits:
                    if qubit in self.adjacent_qubits:
                        for neighbor in self.adjacent_qubits[qubit]:
                            if neighbor not in active_qubits:  # Only spectator qubits
                                # Check if we have ZZ coupling
                                coupling_key = tuple(sorted([qubit, neighbor]))
                                if coupling_key in self.zz_couplings:
                                    channel_name = f"zz_crosstalk_q{qubit}_q{neighbor}"
                                    if channel_name in self.channels:
                                        channel = self.channels[channel_name]
                                        crosstalk_channel = NoiseChannel(
                                            name=f"{channel.name}_gate_{gate_idx}",
                                            kraus_operators=channel.kraus_operators,
                                            probability=channel.probability,
                                            qubits=[qubit, neighbor]
                                        )
                                        channels.append(crosstalk_channel)
                
                # Apply gate-specific crosstalk
                if gate_name in self.gate_crosstalk:
                    gate_crosstalk_params = self.gate_crosstalk[gate_name]
                    for qubit in qubits:
                        if qubit in self.adjacent_qubits:
                            for neighbor in self.adjacent_qubits[qubit]:
                                if neighbor not in active_qubits:
                                    crosstalk_key = tuple(sorted([qubit, neighbor]))
                                    if crosstalk_key in gate_crosstalk_params:
                                        channel_name = f"{gate_name}_crosstalk_q{qubit}_q{neighbor}"
                                        if channel_name in self.channels:
                                            channel = self.channels[channel_name]
                                            gate_crosstalk_channel = NoiseChannel(
                                                name=f"{channel.name}_gate_{gate_idx}",
                                                kraus_operators=channel.kraus_operators,
                                                probability=channel.probability,
                                                qubits=[qubit, neighbor]
                                            )
                                            channels.append(gate_crosstalk_channel)
                
                # Apply control line crosstalk
                for qubit in qubits:
                    if qubit in self.control_crosstalk:
                        neighbors = self.control_crosstalk[qubit]
                        for neighbor, _ in neighbors.items():
                            if neighbor not in active_qubits:
                                channel_name = f"control_crosstalk_q{qubit}_q{neighbor}"
                                if channel_name in self.channels:
                                    channel = self.channels[channel_name]
                                    control_channel = NoiseChannel(
                                        name=f"{channel.name}_gate_{gate_idx}",
                                        kraus_operators=channel.kraus_operators,
                                        probability=channel.probability,
                                        qubits=[qubit, neighbor]
                                    )
                                    channels.append(control_channel)
                
                # Apply frequency crowding effects
                for qubit in qubits:
                    if qubit in self.frequency_crowding:
                        channel_name = f"frequency_crowding_q{qubit}"
                        if channel_name in self.channels:
                            channel = self.channels[channel_name]
                            crowding_channel = NoiseChannel(
                                name=f"{channel.name}_gate_{gate_idx}",
                                kraus_operators=channel.kraus_operators,
                                probability=channel.probability,
                                qubits=[qubit]
                            )
                            channels.append(crowding_channel)
        
        return channels
    
    def apply_noise(self, circuit: Any) -> Any:
        """
        Apply crosstalk noise to circuit.
        
        Args:
            circuit: Clean quantum circuit
            
        Returns:
            Circuit with crosstalk noise applied
        """
        if hasattr(circuit, 'copy'):
            noisy_circuit = circuit.copy()
        else:
            import copy
            noisy_circuit = copy.deepcopy(circuit)
        
        # Add noise markers for simulation
        noisy_circuit._noise_model = self
        noisy_circuit._noise_channels = self.get_noise_channels(circuit)
        
        return noisy_circuit
    
    def set_zz_coupling(self, qubit1: int, qubit2: int, strength: float) -> None:
        """
        Set ZZ coupling strength between two qubits.
        
        Args:
            qubit1: First qubit
            qubit2: Second qubit
            strength: Coupling strength (Hz)
        """
        coupling_key = tuple(sorted([qubit1, qubit2]))
        self.zz_couplings[coupling_key] = strength
        
        # Update adjacency
        if qubit1 not in self.adjacent_qubits:
            self.adjacent_qubits[qubit1] = set()
        if qubit2 not in self.adjacent_qubits:
            self.adjacent_qubits[qubit2] = set()
        self.adjacent_qubits[qubit1].add(qubit2)
        self.adjacent_qubits[qubit2].add(qubit1)
        
        # Recreate relevant channels
        channel_name = f"zz_crosstalk_q{qubit1}_q{qubit2}"
        if channel_name in self.channels:
            del self.channels[channel_name]
        
        if strength > 0:
            kraus_ops = self._create_zz_crosstalk_kraus(strength)
            channel = NoiseChannel(
                name=channel_name,
                kraus_operators=kraus_ops,
                probability=1.0,
                qubits=[qubit1, qubit2]
            )
            self.add_channel(channel)
    
    def get_coupling_strength(self, qubit1: int, qubit2: int) -> float:
        """
        Get coupling strength between two qubits.
        
        Args:
            qubit1: First qubit
            qubit2: Second qubit
            
        Returns:
            Coupling strength (Hz)
        """
        coupling_key = tuple(sorted([qubit1, qubit2]))
        return self.zz_couplings.get(coupling_key, 0.0)
    
    def scale_noise(self, factor: float) -> "CrosstalkNoiseModel":
        """
        Scale crosstalk noise by factor.
        
        Args:
            factor: Scaling factor
            
        Returns:
            New noise model with scaled crosstalk
        """
        scaled_coupling_map = {
            k: v * factor for k, v in self.coupling_map.items()
        }
        scaled_zz_couplings = {
            k: v * factor for k, v in self.zz_couplings.items()
        }
        scaled_gate_crosstalk = {
            gate: {k: v * factor for k, v in params.items()}
            for gate, params in self.gate_crosstalk.items()
        }
        scaled_control_crosstalk = {
            qubit: {neighbor: strength * factor for neighbor, strength in neighbors.items()}
            for qubit, neighbors in self.control_crosstalk.items()
        }
        scaled_frequency_crowding = {
            k: v * factor for k, v in self.frequency_crowding.items()
        }
        
        return CrosstalkNoiseModel(
            coupling_map=scaled_coupling_map,
            zz_couplings=scaled_zz_couplings,
            gate_crosstalk=scaled_gate_crosstalk,
            control_crosstalk=scaled_control_crosstalk,
            frequency_crowding=scaled_frequency_crowding
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            "coupling_map": {
                f"{k[0]}_{k[1]}": float(v) for k, v in self.coupling_map.items()
            },
            "zz_couplings": {
                f"{k[0]}_{k[1]}": float(v) for k, v in self.zz_couplings.items()
            },
            "gate_crosstalk": {
                gate: {f"{k[0]}_{k[1]}": float(v) for k, v in params.items()}
                for gate, params in self.gate_crosstalk.items()
            },
            "control_crosstalk": {
                str(qubit): {str(neighbor): float(strength) 
                           for neighbor, strength in neighbors.items()}
                for qubit, neighbors in self.control_crosstalk.items()
            },
            "frequency_crowding": {
                str(k): float(v) for k, v in self.frequency_crowding.items()
            }
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrosstalkNoiseModel":
        """Create from dictionary representation."""
        coupling_map = {
            tuple(map(int, k.split("_"))): v 
            for k, v in data["coupling_map"].items()
        }
        zz_couplings = {
            tuple(map(int, k.split("_"))): v 
            for k, v in data["zz_couplings"].items()
        }
        gate_crosstalk = {
            gate: {tuple(map(int, k.split("_"))): v for k, v in params.items()}
            for gate, params in data["gate_crosstalk"].items()
        }
        control_crosstalk = {
            int(qubit): {int(neighbor): strength 
                        for neighbor, strength in neighbors.items()}
            for qubit, neighbors in data["control_crosstalk"].items()
        }
        frequency_crowding = {
            int(k): v for k, v in data["frequency_crowding"].items()
        }
        
        return cls(
            coupling_map=coupling_map,
            zz_couplings=zz_couplings,
            gate_crosstalk=gate_crosstalk,
            control_crosstalk=control_crosstalk,
            frequency_crowding=frequency_crowding
        )
    
    def __str__(self) -> str:
        """String representation."""
        lines = [f"CrosstalkNoiseModel"]
        lines.append(f"Coupling map: {len(self.coupling_map)} pairs")
        lines.append(f"ZZ couplings: {len(self.zz_couplings)} pairs")
        lines.append(f"Gate crosstalk: {len(self.gate_crosstalk)} gate types")
        lines.append(f"Control crosstalk: {len(self.control_crosstalk)} qubits")
        lines.append(f"Frequency crowding: {len(self.frequency_crowding)} qubits")
        lines.append(f"Channels: {len(self.channels)}")
        return "\n".join(lines)


class GridCrosstaIkModel(CrosstalkNoiseModel):
    """
    Crosstalk model for grid/lattice qubit layouts.
    
    Convenience class for common hardware topologies.
    """
    
    def __init__(
        self,
        rows: int,
        cols: int,
        nearest_neighbor_coupling: float = 1e3,  # Hz
        next_nearest_neighbor_coupling: float = 1e2,  # Hz
        include_diagonal: bool = False
    ):
        """
        Initialize grid crosstalk model.
        
        Args:
            rows: Number of rows in grid
            cols: Number of columns in grid
            nearest_neighbor_coupling: NN coupling strength (Hz)
            next_nearest_neighbor_coupling: NNN coupling strength (Hz)
            include_diagonal: Include diagonal couplings
        """
        # Generate grid coupling map
        coupling_map = {}
        zz_couplings = {}
        
        for row in range(rows):
            for col in range(cols):
                qubit = row * cols + col
                
                # Nearest neighbors
                neighbors = []
                if row > 0:  # Up
                    neighbors.append((row - 1) * cols + col)
                if row < rows - 1:  # Down
                    neighbors.append((row + 1) * cols + col)
                if col > 0:  # Left
                    neighbors.append(row * cols + (col - 1))
                if col < cols - 1:  # Right
                    neighbors.append(row * cols + (col + 1))
                
                for neighbor in neighbors:
                    coupling_key = tuple(sorted([qubit, neighbor]))
                    coupling_map[coupling_key] = nearest_neighbor_coupling
                    zz_couplings[coupling_key] = nearest_neighbor_coupling * 0.1
                
                # Next-nearest neighbors (diagonal)
                if include_diagonal:
                    diag_neighbors = []
                    if row > 0 and col > 0:  # Up-left
                        diag_neighbors.append((row - 1) * cols + (col - 1))
                    if row > 0 and col < cols - 1:  # Up-right
                        diag_neighbors.append((row - 1) * cols + (col + 1))
                    if row < rows - 1 and col > 0:  # Down-left
                        diag_neighbors.append((row + 1) * cols + (col - 1))
                    if row < rows - 1 and col < cols - 1:  # Down-right
                        diag_neighbors.append((row + 1) * cols + (col + 1))
                    
                    for neighbor in diag_neighbors:
                        coupling_key = tuple(sorted([qubit, neighbor]))
                        coupling_map[coupling_key] = next_nearest_neighbor_coupling
                        zz_couplings[coupling_key] = next_nearest_neighbor_coupling * 0.1
        
        super().__init__(
            coupling_map=coupling_map,
            zz_couplings=zz_couplings
        )
        
        self.name = "grid_crosstalk"
        self.rows = rows
        self.cols = cols
    
    def get_qubit_position(self, qubit: int) -> Tuple[int, int]:
        """Get (row, col) position of qubit in grid."""
        row = qubit // self.cols
        col = qubit % self.cols
        return (row, col)
    
    def get_qubit_from_position(self, row: int, col: int) -> int:
        """Get qubit index from (row, col) position."""
        return row * self.cols + col