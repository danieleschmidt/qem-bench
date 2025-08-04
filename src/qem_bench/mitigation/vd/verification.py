"""Verification circuit generators for Virtual Distillation."""

import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

from ...jax.circuits import JAXCircuit, create_bell_circuit, create_ghz_circuit
from ...jax.states import create_bell_state, create_ghz_state, fidelity


class VerificationStrategy(ABC):
    """Abstract base class for verification strategies."""
    
    @abstractmethod
    def generate_verification_circuits(self, num_qubits: int, num_copies: int) -> List[JAXCircuit]:
        """
        Generate verification circuits for virtual distillation.
        
        Args:
            num_qubits: Number of qubits in the main circuit
            num_copies: Number of copies in the distillation
            
        Returns:
            List of verification circuits
        """
        pass
    
    @abstractmethod
    def get_expected_states(self, num_qubits: int, num_copies: int) -> List[jnp.ndarray]:
        """
        Get expected ideal states for verification circuits.
        
        Args:
            num_qubits: Number of qubits in the main circuit
            num_copies: Number of copies in the distillation
            
        Returns:
            List of expected quantum states
        """
        pass
    
    @abstractmethod
    def calculate_verification_fidelity(
        self, 
        measured_states: List[jnp.ndarray],
        expected_states: List[jnp.ndarray]
    ) -> float:
        """
        Calculate verification fidelity from measured and expected states.
        
        Args:
            measured_states: States measured from verification circuits
            expected_states: Expected ideal states
            
        Returns:
            Average verification fidelity
        """
        pass


class BellStateVerification(VerificationStrategy):
    """
    Bell state verification strategy.
    
    Uses Bell state preparation circuits to verify distillation quality.
    Suitable for 2-qubit systems.
    """
    
    def __init__(self, bell_types: Optional[List[str]] = None):
        """
        Initialize Bell state verification.
        
        Args:
            bell_types: Types of Bell states to use ("00", "01", "10", "11")
        """
        if bell_types is None:
            bell_types = ["00", "01"]  # Use Φ+ and Ψ+ by default
        
        self.bell_types = bell_types
        
        # Validate bell types
        valid_types = ["00", "01", "10", "11"]
        for bell_type in bell_types:
            if bell_type not in valid_types:
                raise ValueError(f"Invalid Bell type: {bell_type}. Valid types: {valid_types}")
    
    def generate_verification_circuits(self, num_qubits: int, num_copies: int) -> List[JAXCircuit]:
        """Generate Bell state verification circuits."""
        if num_qubits != 2:
            raise ValueError("Bell state verification requires exactly 2 qubits")
        
        circuits = []
        
        for i in range(num_copies):
            for bell_type in self.bell_types:
                # Create Bell circuit for this copy
                if bell_type == "00":  # |Φ+⟩ = (|00⟩ + |11⟩)/√2
                    circuit = create_bell_circuit(0, 1)
                elif bell_type == "01":  # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
                    circuit = JAXCircuit(2, name=f"bell_{bell_type}_copy_{i}")
                    circuit.h(0)
                    circuit.x(1)
                    circuit.cx(0, 1)
                elif bell_type == "10":  # |Φ-⟩ = (|00⟩ - |11⟩)/√2
                    circuit = create_bell_circuit(0, 1)
                    circuit.z(0)  # Add phase
                elif bell_type == "11":  # |Ψ-⟩ = (|01⟩ - |10⟩)/√2
                    circuit = JAXCircuit(2, name=f"bell_{bell_type}_copy_{i}")
                    circuit.h(0)
                    circuit.x(1)
                    circuit.cx(0, 1)
                    circuit.z(1)  # Add phase
                
                circuit.name = f"bell_{bell_type}_copy_{i}"
                circuits.append(circuit)
        
        return circuits
    
    def get_expected_states(self, num_qubits: int, num_copies: int) -> List[jnp.ndarray]:
        """Get expected Bell states."""
        if num_qubits != 2:
            raise ValueError("Bell state verification requires exactly 2 qubits")
        
        states = []
        
        for i in range(num_copies):
            for bell_type in self.bell_types:
                expected_state = create_bell_state(bell_type)
                states.append(expected_state)
        
        return states
    
    def calculate_verification_fidelity(
        self, 
        measured_states: List[jnp.ndarray],
        expected_states: List[jnp.ndarray]
    ) -> float:
        """Calculate average Bell state fidelity."""
        if len(measured_states) != len(expected_states):
            raise ValueError("Measured and expected states must have same length")
        
        fidelities = []
        for measured, expected in zip(measured_states, expected_states):
            fid = fidelity(measured, expected)
            fidelities.append(fid)
        
        return float(jnp.mean(jnp.array(fidelities)))


class GHZStateVerification(VerificationStrategy):
    """
    GHZ state verification strategy.
    
    Uses GHZ state preparation circuits to verify distillation quality.
    Suitable for multi-qubit systems.
    """
    
    def __init__(self, include_variations: bool = True):
        """
        Initialize GHZ state verification.
        
        Args:
            include_variations: Whether to include GHZ state variations with phase flips
        """
        self.include_variations = include_variations
    
    def generate_verification_circuits(self, num_qubits: int, num_copies: int) -> List[JAXCircuit]:
        """Generate GHZ state verification circuits."""
        if num_qubits < 2:
            raise ValueError("GHZ state verification requires at least 2 qubits")
        
        circuits = []
        
        for i in range(num_copies):
            # Standard GHZ: (|00...0⟩ + |11...1⟩)/√2
            circuit = create_ghz_circuit(num_qubits)
            circuit.name = f"ghz_standard_copy_{i}"
            circuits.append(circuit)
            
            # Include variations with phase flips
            if self.include_variations and num_qubits >= 3:
                # GHZ with phase flip: (|00...0⟩ - |11...1⟩)/√2
                circuit_phase = create_ghz_circuit(num_qubits)
                circuit_phase.z(0)  # Add phase flip
                circuit_phase.name = f"ghz_phase_copy_{i}"
                circuits.append(circuit_phase)
        
        return circuits
    
    def get_expected_states(self, num_qubits: int, num_copies: int) -> List[jnp.ndarray]:
        """Get expected GHZ states."""
        if num_qubits < 2:
            raise ValueError("GHZ state verification requires at least 2 qubits")
        
        states = []
        
        for i in range(num_copies):
            # Standard GHZ state
            ghz_state = create_ghz_state(num_qubits)
            states.append(ghz_state)
            
            # Include variations with phase flips
            if self.include_variations and num_qubits >= 3:
                # GHZ with phase flip
                ghz_phase_state = jnp.zeros(2 ** num_qubits, dtype=jnp.complex64)
                ghz_phase_state = ghz_phase_state.at[0].set(1/jnp.sqrt(2))          # |0...0⟩
                ghz_phase_state = ghz_phase_state.at[-1].set(-1/jnp.sqrt(2))        # -|1...1⟩
                states.append(ghz_phase_state)
        
        return states
    
    def calculate_verification_fidelity(
        self, 
        measured_states: List[jnp.ndarray],
        expected_states: List[jnp.ndarray]
    ) -> float:
        """Calculate average GHZ state fidelity."""
        if len(measured_states) != len(expected_states):
            raise ValueError("Measured and expected states must have same length")
        
        fidelities = []
        for measured, expected in zip(measured_states, expected_states):
            fid = fidelity(measured, expected)
            fidelities.append(fid)
        
        return float(jnp.mean(jnp.array(fidelities)))


class ProductStateVerification(VerificationStrategy):
    """
    Product state verification strategy.
    
    Uses simple product states (|+⟩⊗n, |0⟩⊗n, etc.) for verification.
    Suitable for any number of qubits, especially useful for benchmarking.
    """
    
    def __init__(self, state_types: Optional[List[str]] = None):
        """
        Initialize product state verification.
        
        Args:
            state_types: Types of product states ("zero", "plus", "minus")
        """
        if state_types is None:
            state_types = ["zero", "plus"]  # Use |0⟩⊗n and |+⟩⊗n by default
        
        self.state_types = state_types
        
        # Validate state types
        valid_types = ["zero", "plus", "minus", "one"]
        for state_type in state_types:
            if state_type not in valid_types:
                raise ValueError(f"Invalid state type: {state_type}. Valid types: {valid_types}")
    
    def generate_verification_circuits(self, num_qubits: int, num_copies: int) -> List[JAXCircuit]:
        """Generate product state verification circuits."""
        circuits = []
        
        for i in range(num_copies):
            for state_type in self.state_types:
                circuit = JAXCircuit(num_qubits, name=f"{state_type}_product_copy_{i}")
                
                if state_type == "zero":
                    # |0⟩⊗n - no gates needed
                    pass
                elif state_type == "one":
                    # |1⟩⊗n
                    for qubit in range(num_qubits):
                        circuit.x(qubit)
                elif state_type == "plus":
                    # |+⟩⊗n
                    for qubit in range(num_qubits):
                        circuit.h(qubit)
                elif state_type == "minus":
                    # |-⟩⊗n
                    for qubit in range(num_qubits):
                        circuit.x(qubit)
                        circuit.h(qubit)
                
                circuits.append(circuit)
        
        return circuits
    
    def get_expected_states(self, num_qubits: int, num_copies: int) -> List[jnp.ndarray]:
        """Get expected product states."""
        from ...jax.states import zero_state, one_state, plus_state, minus_state
        
        states = []
        
        for i in range(num_copies):
            for state_type in self.state_types:
                if state_type == "zero":
                    state = zero_state(num_qubits)
                elif state_type == "one":
                    state = one_state(num_qubits)
                elif state_type == "plus":
                    state = plus_state(num_qubits)
                elif state_type == "minus":
                    state = minus_state(num_qubits)
                
                states.append(state)
        
        return states
    
    def calculate_verification_fidelity(
        self, 
        measured_states: List[jnp.ndarray],
        expected_states: List[jnp.ndarray]
    ) -> float:
        """Calculate average product state fidelity."""
        if len(measured_states) != len(expected_states):
            raise ValueError("Measured and expected states must have same length")
        
        fidelities = []
        for measured, expected in zip(measured_states, expected_states):
            fid = fidelity(measured, expected)
            fidelities.append(fid)
        
        return float(jnp.mean(jnp.array(fidelities)))


class RandomStateVerification(VerificationStrategy):
    """
    Random state verification strategy.
    
    Uses random quantum states for verification. Useful for general
    benchmarking but may be less efficient than structured states.
    """
    
    def __init__(self, num_random_states: int = 2, seed: Optional[int] = None):
        """
        Initialize random state verification.
        
        Args:
            num_random_states: Number of different random states per copy
            seed: Random seed for reproducibility
        """
        self.num_random_states = num_random_states
        self.seed = seed
        self._random_states_cache = {}
    
    def generate_verification_circuits(self, num_qubits: int, num_copies: int) -> List[JAXCircuit]:
        """Generate random state verification circuits."""
        # This is a simplified implementation
        # In practice, would need to generate circuits that prepare the random states
        circuits = []
        
        if self.seed is not None:
            np.random.seed(self.seed)
        
        for i in range(num_copies):
            for j in range(self.num_random_states):
                circuit = JAXCircuit(num_qubits, name=f"random_{j}_copy_{i}")
                
                # Generate random circuit (simplified)
                depth = min(10, num_qubits * 2)  # Reasonable depth
                for _ in range(depth):
                    # Random single-qubit rotation
                    qubit = np.random.randint(num_qubits)
                    angle = np.random.uniform(0, 2 * np.pi)
                    gate_type = np.random.choice(['rx', 'ry', 'rz'])
                    getattr(circuit, gate_type)(angle, qubit)
                    
                    # Random two-qubit gate (with some probability)
                    if num_qubits > 1 and np.random.random() < 0.3:
                        qubits = np.random.choice(num_qubits, 2, replace=False)
                        circuit.cx(qubits[0], qubits[1])
                
                circuits.append(circuit)
        
        return circuits
    
    def get_expected_states(self, num_qubits: int, num_copies: int) -> List[jnp.ndarray]:
        """Get expected random states."""
        from ...jax.states import create_random_state
        
        states = []
        
        # Use cached states if available
        cache_key = (num_qubits, num_copies, self.seed)
        if cache_key in self._random_states_cache:
            return self._random_states_cache[cache_key]
        
        if self.seed is not None:
            np.random.seed(self.seed)
        
        for i in range(num_copies):
            for j in range(self.num_random_states):
                state = create_random_state(num_qubits, seed=None)  # Use global seed
                states.append(state)
        
        # Cache for future use
        self._random_states_cache[cache_key] = states
        return states
    
    def calculate_verification_fidelity(
        self, 
        measured_states: List[jnp.ndarray],
        expected_states: List[jnp.ndarray]
    ) -> float:
        """Calculate average random state fidelity."""
        if len(measured_states) != len(expected_states):
            raise ValueError("Measured and expected states must have same length")
        
        fidelities = []
        for measured, expected in zip(measured_states, expected_states):
            fid = fidelity(measured, expected)
            fidelities.append(fid)
        
        return float(jnp.mean(jnp.array(fidelities)))


# Factory function to create verification strategies
def create_verification_strategy(
    strategy_name: str,
    num_qubits: int,
    **kwargs
) -> VerificationStrategy:
    """
    Create a verification strategy.
    
    Args:
        strategy_name: Name of the strategy ("bell", "ghz", "product", "random")
        num_qubits: Number of qubits in the system
        **kwargs: Additional arguments for the strategy
        
    Returns:
        Verification strategy instance
    """
    strategies = {
        "bell": BellStateVerification,
        "ghz": GHZStateVerification,
        "product": ProductStateVerification,
        "random": RandomStateVerification
    }
    
    if strategy_name not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(f"Unknown verification strategy '{strategy_name}'. Available: {available}")
    
    strategy_class = strategies[strategy_name]
    
    # Validate strategy compatibility with number of qubits
    if strategy_name == "bell" and num_qubits != 2:
        raise ValueError("Bell state verification requires exactly 2 qubits")
    elif strategy_name == "ghz" and num_qubits < 2:
        raise ValueError("GHZ state verification requires at least 2 qubits")
    
    return strategy_class(**kwargs)


def estimate_verification_overhead(
    strategy: VerificationStrategy,
    num_qubits: int,
    num_copies: int
) -> Dict[str, Any]:
    """
    Estimate the computational overhead of verification.
    
    Args:
        strategy: Verification strategy
        num_qubits: Number of qubits
        num_copies: Number of copies
        
    Returns:
        Dictionary with overhead estimates
    """
    verification_circuits = strategy.generate_verification_circuits(num_qubits, num_copies)
    
    total_gates = sum(circuit.size for circuit in verification_circuits)
    total_depth = sum(circuit.depth for circuit in verification_circuits)
    num_circuits = len(verification_circuits)
    
    return {
        "num_verification_circuits": num_circuits,
        "total_gates": total_gates,
        "total_depth": total_depth,
        "average_gates_per_circuit": total_gates / num_circuits if num_circuits > 0 else 0,
        "average_depth_per_circuit": total_depth / num_circuits if num_circuits > 0 else 0,
        "overhead_factor": total_gates / (num_copies * num_qubits) if num_copies > 0 else 0
    }