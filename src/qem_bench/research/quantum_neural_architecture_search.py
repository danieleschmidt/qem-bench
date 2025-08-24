"""
Quantum Neural Architecture Search (QNAS)
Novel framework for automatically designing quantum neural network architectures
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import numpy as np
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import itertools
from functools import partial

logger = logging.getLogger(__name__)


@dataclass
class QuantumArchitectureGenome:
    """Genetic representation of a quantum neural network architecture"""
    num_qubits: int
    num_layers: int
    gate_sequence: List[str]
    parameter_sharing: Dict[str, List[int]]
    entanglement_pattern: str
    measurement_basis: List[str]
    fitness_score: float = 0.0
    complexity_score: float = 0.0
    efficiency_metric: float = 0.0
    
    def __post_init__(self):
        """Validate genome after initialization"""
        if not self.gate_sequence:
            self.gate_sequence = ['RY', 'RZ', 'CNOT']
        if not self.measurement_basis:
            self.measurement_basis = ['Z'] * self.num_qubits


@dataclass 
class QuantumNASConfig:
    """Configuration for Quantum Neural Architecture Search"""
    population_size: int = 50
    num_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_ratio: float = 0.2
    max_qubits: int = 16
    max_layers: int = 20
    fitness_criteria: List[str] = field(default_factory=lambda: ['accuracy', 'efficiency', 'expressivity'])
    search_space_constraints: Dict[str, Any] = field(default_factory=dict)


class QuantumArchitectureEvaluator(ABC):
    """Abstract base for evaluating quantum neural network architectures"""
    
    @abstractmethod
    def evaluate_architecture(
        self, 
        genome: QuantumArchitectureGenome,
        training_data: Any,
        validation_data: Any
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate architecture and return fitness score and metrics"""
        pass


class QuantumCircuitSimulator(QuantumArchitectureEvaluator):
    """Simulator-based architecture evaluator"""
    
    def __init__(self, max_simulation_qubits: int = 10):
        self.max_simulation_qubits = max_simulation_qubits
        self.gate_implementations = {
            'RX': self._apply_rx,
            'RY': self._apply_ry, 
            'RZ': self._apply_rz,
            'CNOT': self._apply_cnot,
            'CZ': self._apply_cz,
            'SWAP': self._apply_swap,
            'H': self._apply_hadamard
        }
        
    def evaluate_architecture(
        self,
        genome: QuantumArchitectureGenome,
        training_data: Tuple[jnp.ndarray, jnp.ndarray],
        validation_data: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate architecture using quantum circuit simulation"""
        
        if genome.num_qubits > self.max_simulation_qubits:
            # Use approximation for large circuits
            return self._approximate_evaluation(genome, training_data, validation_data)
        
        # Build and simulate quantum circuit
        circuit = self._build_quantum_circuit(genome)
        
        # Train and evaluate
        X_train, y_train = training_data
        X_val, y_val = validation_data
        
        # Simplified training (would use actual VQC training)
        parameters = self._train_vqc(circuit, X_train, y_train)
        
        # Evaluation metrics
        train_accuracy = self._evaluate_vqc(circuit, parameters, X_train, y_train)
        val_accuracy = self._evaluate_vqc(circuit, parameters, X_val, y_val)
        
        # Calculate additional metrics
        expressivity = self._calculate_expressivity(circuit, parameters)
        entanglement_capability = self._calculate_entanglement_capability(genome)
        parameter_efficiency = len(parameters) / max(train_accuracy, 1e-6)
        
        # Composite fitness score
        fitness = self._calculate_composite_fitness(
            val_accuracy, expressivity, entanglement_capability, parameter_efficiency
        )
        
        metrics = {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'expressivity': expressivity,
            'entanglement_capability': entanglement_capability,
            'parameter_efficiency': parameter_efficiency,
            'num_parameters': len(parameters)
        }
        
        return fitness, metrics
    
    def _build_quantum_circuit(self, genome: QuantumArchitectureGenome) -> Callable:
        """Build quantum circuit from genome"""
        
        @jax.jit
        def quantum_circuit(parameters: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
            # Initialize quantum state
            state = self._prepare_initial_state(genome.num_qubits, inputs)
            
            param_idx = 0
            
            # Apply layers
            for layer in range(genome.num_layers):
                # Apply parameterized gates
                for gate_name in genome.gate_sequence:
                    if gate_name in ['RX', 'RY', 'RZ']:
                        for qubit in range(genome.num_qubits):
                            if param_idx < len(parameters):
                                state = self.gate_implementations[gate_name](
                                    state, qubit, parameters[param_idx]
                                )
                                param_idx += 1
                    
                    elif gate_name == 'CNOT':
                        # Apply entanglement pattern
                        for control, target in self._get_entanglement_pairs(genome):
                            state = self.gate_implementations[gate_name](
                                state, control, target
                            )
                
                # Add variational layer
                if layer < genome.num_layers - 1:
                    state = self._apply_variational_layer(state, genome.num_qubits)
            
            # Measurement
            return self._measure_circuit(state, genome.measurement_basis)
        
        return quantum_circuit
    
    def _prepare_initial_state(self, num_qubits: int, inputs: jnp.ndarray) -> jnp.ndarray:
        """Prepare initial quantum state with classical input encoding"""
        # Initialize |0⟩^n state
        state = jnp.zeros(2**num_qubits, dtype=jnp.complex64)
        state = state.at[0].set(1.0 + 0.0j)
        
        # Amplitude encoding (simplified)
        if len(inputs) > 0:
            # Normalize inputs and encode
            normalized_inputs = inputs / (jnp.linalg.norm(inputs) + 1e-8)
            encoding_length = min(len(normalized_inputs), 2**num_qubits)
            state = state.at[:encoding_length].set(
                normalized_inputs[:encoding_length].astype(jnp.complex64)
            )
            # Renormalize
            state = state / (jnp.linalg.norm(state) + 1e-8)
        
        return state
    
    def _apply_rx(self, state: jnp.ndarray, qubit: int, angle: float) -> jnp.ndarray:
        """Apply RX rotation"""
        n_qubits = int(jnp.log2(len(state)))
        rx_gate = jnp.array([
            [jnp.cos(angle/2), -1j * jnp.sin(angle/2)],
            [-1j * jnp.sin(angle/2), jnp.cos(angle/2)]
        ], dtype=jnp.complex64)
        
        # Apply single-qubit gate (simplified tensor product)
        return self._apply_single_qubit_gate(state, rx_gate, qubit, n_qubits)
    
    def _apply_ry(self, state: jnp.ndarray, qubit: int, angle: float) -> jnp.ndarray:
        """Apply RY rotation"""
        n_qubits = int(jnp.log2(len(state)))
        ry_gate = jnp.array([
            [jnp.cos(angle/2), -jnp.sin(angle/2)],
            [jnp.sin(angle/2), jnp.cos(angle/2)]
        ], dtype=jnp.complex64)
        
        return self._apply_single_qubit_gate(state, ry_gate, qubit, n_qubits)
    
    def _apply_rz(self, state: jnp.ndarray, qubit: int, angle: float) -> jnp.ndarray:
        """Apply RZ rotation"""
        n_qubits = int(jnp.log2(len(state)))
        rz_gate = jnp.array([
            [jnp.exp(-1j * angle/2), 0],
            [0, jnp.exp(1j * angle/2)]
        ], dtype=jnp.complex64)
        
        return self._apply_single_qubit_gate(state, rz_gate, qubit, n_qubits)
    
    def _apply_cnot(self, state: jnp.ndarray, control: int, target: int) -> jnp.ndarray:
        """Apply CNOT gate"""
        n_qubits = int(jnp.log2(len(state)))
        new_state = state.copy()
        
        # Apply CNOT operation (simplified)
        for i in range(2**n_qubits):
            bit_string = [(i >> j) & 1 for j in range(n_qubits)]
            
            if bit_string[control] == 1:
                # Flip target qubit
                flipped_bit_string = bit_string.copy()
                flipped_bit_string[target] = 1 - flipped_bit_string[target]
                
                # Convert back to index
                flipped_index = sum(
                    bit * (2**j) for j, bit in enumerate(flipped_bit_string)
                )
                
                # Swap amplitudes
                new_state = new_state.at[i].set(state[flipped_index])
                new_state = new_state.at[flipped_index].set(state[i])
        
        return new_state
    
    def _apply_cz(self, state: jnp.ndarray, control: int, target: int) -> jnp.ndarray:
        """Apply CZ gate"""
        n_qubits = int(jnp.log2(len(state)))
        new_state = state.copy()
        
        # Apply phase flip when both qubits are |1⟩
        for i in range(2**n_qubits):
            bit_string = [(i >> j) & 1 for j in range(n_qubits)]
            
            if bit_string[control] == 1 and bit_string[target] == 1:
                new_state = new_state.at[i].set(-state[i])
        
        return new_state
    
    def _apply_swap(self, state: jnp.ndarray, qubit1: int, qubit2: int) -> jnp.ndarray:
        """Apply SWAP gate"""
        # Implement as three CNOT gates
        state = self._apply_cnot(state, qubit1, qubit2)
        state = self._apply_cnot(state, qubit2, qubit1)
        state = self._apply_cnot(state, qubit1, qubit2)
        return state
    
    def _apply_hadamard(self, state: jnp.ndarray, qubit: int, angle: float = 0.0) -> jnp.ndarray:
        """Apply Hadamard gate"""
        n_qubits = int(jnp.log2(len(state)))
        h_gate = jnp.array([
            [1, 1],
            [1, -1]
        ], dtype=jnp.complex64) / jnp.sqrt(2)
        
        return self._apply_single_qubit_gate(state, h_gate, qubit, n_qubits)
    
    def _apply_single_qubit_gate(
        self, 
        state: jnp.ndarray, 
        gate: jnp.ndarray, 
        qubit: int, 
        n_qubits: int
    ) -> jnp.ndarray:
        """Apply single-qubit gate using tensor product structure"""
        new_state = jnp.zeros_like(state)
        
        # For each computational basis state
        for i in range(2**n_qubits):
            bit_string = [(i >> j) & 1 for j in range(n_qubits)]
            qubit_state = bit_string[qubit]
            
            # Apply gate matrix
            for new_qubit_state in [0, 1]:
                amplitude = gate[new_qubit_state, qubit_state]
                
                if abs(amplitude) > 1e-10:  # Skip negligible amplitudes
                    # Create new bit string
                    new_bit_string = bit_string.copy()
                    new_bit_string[qubit] = new_qubit_state
                    
                    # Convert to index
                    new_index = sum(
                        bit * (2**j) for j, bit in enumerate(new_bit_string)
                    )
                    
                    new_state = new_state.at[new_index].add(amplitude * state[i])
        
        return new_state
    
    def _get_entanglement_pairs(self, genome: QuantumArchitectureGenome) -> List[Tuple[int, int]]:
        """Get entanglement pairs based on pattern"""
        pairs = []
        
        if genome.entanglement_pattern == "linear":
            pairs = [(i, i+1) for i in range(genome.num_qubits - 1)]
        elif genome.entanglement_pattern == "circular":
            pairs = [(i, (i+1) % genome.num_qubits) for i in range(genome.num_qubits)]
        elif genome.entanglement_pattern == "all_to_all":
            pairs = [(i, j) for i in range(genome.num_qubits) 
                    for j in range(i+1, genome.num_qubits)]
        elif genome.entanglement_pattern == "star":
            pairs = [(0, i) for i in range(1, genome.num_qubits)]
        else:  # random
            key = jax.random.PRNGKey(42)
            num_pairs = min(genome.num_qubits, 5)  # Limit for efficiency
            all_pairs = [(i, j) for i in range(genome.num_qubits) 
                        for j in range(i+1, genome.num_qubits)]
            selected_indices = jax.random.choice(
                key, len(all_pairs), (num_pairs,), replace=False
            )
            pairs = [all_pairs[int(idx)] for idx in selected_indices]
        
        return pairs
    
    def _apply_variational_layer(self, state: jnp.ndarray, num_qubits: int) -> jnp.ndarray:
        """Apply variational layer with fixed parameters"""
        # Simple variational layer with RY rotations
        for qubit in range(num_qubits):
            # Fixed variational parameter
            state = self._apply_ry(state, qubit, 0.1)
        return state
    
    def _measure_circuit(self, state: jnp.ndarray, measurement_basis: List[str]) -> jnp.ndarray:
        """Measure circuit in specified basis"""
        # For simplicity, return expectation values of Pauli measurements
        n_qubits = int(jnp.log2(len(state)))
        measurements = jnp.zeros(len(measurement_basis))
        
        for i, basis in enumerate(measurement_basis):
            if basis == 'Z':
                # Z measurement expectation
                expectation = 0.0
                for j in range(2**n_qubits):
                    bit_string = [(j >> k) & 1 for k in range(n_qubits)]
                    if i < len(bit_string):
                        sign = 1 if bit_string[i] == 0 else -1
                        expectation += sign * jnp.abs(state[j])**2
                measurements = measurements.at[i].set(expectation)
            elif basis == 'X':
                # X measurement (simplified)
                measurements = measurements.at[i].set(jnp.real(jnp.sum(state)))
            elif basis == 'Y':
                # Y measurement (simplified)
                measurements = measurements.at[i].set(jnp.imag(jnp.sum(state)))
        
        return measurements
    
    def _train_vqc(
        self, 
        circuit: Callable, 
        X_train: jnp.ndarray, 
        y_train: jnp.ndarray
    ) -> jnp.ndarray:
        """Train variational quantum circuit (simplified)"""
        
        # Estimate number of parameters needed
        num_params = 20  # Simplified estimate
        key = jax.random.PRNGKey(42)
        initial_params = jax.random.normal(key, (num_params,)) * 0.1
        
        # Simplified training (would use actual optimization)
        def loss_fn(params):
            predictions = jnp.array([
                circuit(params, x) for x in X_train[:min(10, len(X_train))]
            ])
            # Mean squared error (simplified)
            targets = y_train[:len(predictions)]
            if targets.ndim > 1:
                targets = targets[:, 0]  # Take first column
            
            # Ensure compatible shapes
            pred_values = jnp.sum(predictions, axis=1) if predictions.ndim > 1 else predictions
            return jnp.mean((pred_values - targets)**2)
        
        # Simple gradient descent (one step for efficiency)
        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(initial_params)
        updated_params = initial_params - 0.01 * gradient
        
        return updated_params
    
    def _evaluate_vqc(
        self,
        circuit: Callable,
        parameters: jnp.ndarray,
        X_test: jnp.ndarray,
        y_test: jnp.ndarray
    ) -> float:
        """Evaluate VQC performance"""
        
        predictions = jnp.array([
            circuit(parameters, x) for x in X_test[:min(5, len(X_test))]
        ])
        
        # Convert predictions to classification/regression output
        pred_values = jnp.sum(predictions, axis=1) if predictions.ndim > 1 else predictions
        targets = y_test[:len(pred_values)]
        if targets.ndim > 1:
            targets = targets[:, 0]
        
        # Calculate accuracy/error
        if jnp.all(jnp.isin(targets, [0, 1])):  # Classification
            pred_classes = (pred_values > 0.5).astype(int)
            accuracy = jnp.mean(pred_classes == targets)
            return float(accuracy)
        else:  # Regression
            mse = jnp.mean((pred_values - targets)**2)
            # Convert to accuracy-like metric
            return float(1.0 / (1.0 + mse))
    
    def _calculate_expressivity(self, circuit: Callable, parameters: jnp.ndarray) -> float:
        """Calculate circuit expressivity using parameter sensitivity"""
        
        # Test different input states
        key = jax.random.PRNGKey(123)
        test_inputs = jax.random.normal(key, (5, 4))  # 5 test inputs
        
        # Calculate output variance across parameter perturbations  
        param_variations = []
        for _ in range(10):  # 10 parameter variations
            key, subkey = jax.random.split(key)
            perturbed_params = parameters + jax.random.normal(subkey, parameters.shape) * 0.01
            
            outputs = []
            for test_input in test_inputs:
                try:
                    output = circuit(perturbed_params, test_input)
                    outputs.append(jnp.sum(output))  # Scalar summary
                except:
                    outputs.append(0.0)  # Handle errors
            
            param_variations.append(jnp.array(outputs))
        
        # Calculate variance across parameter variations
        param_variations = jnp.array(param_variations)
        expressivity = float(jnp.mean(jnp.var(param_variations, axis=0)))
        
        return min(expressivity, 1.0)  # Clip to reasonable range
    
    def _calculate_entanglement_capability(self, genome: QuantumArchitectureGenome) -> float:
        """Estimate entanglement capability of architecture"""
        
        # Count potential entangling gates
        entangling_gates = ['CNOT', 'CZ', 'SWAP']
        entangling_count = sum(1 for gate in genome.gate_sequence if gate in entangling_gates)
        
        # Entanglement pattern contribution
        pattern_scores = {
            'linear': 0.3,
            'circular': 0.4, 
            'star': 0.5,
            'all_to_all': 1.0,
            'random': 0.6
        }
        pattern_score = pattern_scores.get(genome.entanglement_pattern, 0.3)
        
        # Depth contribution
        depth_score = min(genome.num_layers / 10, 1.0)
        
        # Composite score
        capability = (entangling_count / len(genome.gate_sequence)) * pattern_score * depth_score
        
        return float(min(capability, 1.0))
    
    def _calculate_composite_fitness(
        self,
        accuracy: float,
        expressivity: float,
        entanglement: float,
        efficiency: float
    ) -> float:
        """Calculate composite fitness score"""
        
        # Weighted combination of metrics
        weights = [0.4, 0.2, 0.2, 0.2]  # Accuracy, expressivity, entanglement, efficiency
        metrics = [accuracy, expressivity, entanglement, 1.0 / (efficiency + 1e-6)]
        
        # Normalize efficiency term
        metrics[3] = min(metrics[3], 1.0)
        
        fitness = sum(w * m for w, m in zip(weights, metrics))
        return float(fitness)
    
    def _approximate_evaluation(
        self,
        genome: QuantumArchitectureGenome,
        training_data: Any,
        validation_data: Any
    ) -> Tuple[float, Dict[str, float]]:
        """Approximate evaluation for large circuits"""
        
        # Use heuristics for large circuits
        complexity_penalty = genome.num_qubits / self.max_simulation_qubits
        layer_bonus = min(genome.num_layers / 10, 1.0)
        
        # Estimate performance based on architecture properties
        estimated_accuracy = 0.7 * layer_bonus / complexity_penalty
        estimated_expressivity = min(genome.num_layers * len(genome.gate_sequence) / 100, 1.0)
        estimated_entanglement = self._calculate_entanglement_capability(genome)
        estimated_efficiency = 1.0 / (genome.num_qubits * genome.num_layers)
        
        fitness = self._calculate_composite_fitness(
            estimated_accuracy, estimated_expressivity, 
            estimated_entanglement, estimated_efficiency
        )
        
        metrics = {
            'train_accuracy': estimated_accuracy * 0.9,  # Assume overfitting
            'val_accuracy': estimated_accuracy,
            'expressivity': estimated_expressivity,
            'entanglement_capability': estimated_entanglement,
            'parameter_efficiency': estimated_efficiency,
            'num_parameters': genome.num_qubits * genome.num_layers
        }
        
        return fitness, metrics


class QuantumNeuralArchitectureSearch:
    """Main class for quantum neural architecture search"""
    
    def __init__(self, config: QuantumNASConfig, evaluator: QuantumArchitectureEvaluator):
        self.config = config
        self.evaluator = evaluator
        self.population = []
        self.generation = 0
        self.best_genomes = []
        self.search_history = []
        
    def initialize_population(self) -> List[QuantumArchitectureGenome]:
        """Initialize random population of quantum architectures"""
        
        population = []
        key = jax.random.PRNGKey(42)
        
        for i in range(self.config.population_size):
            key, *subkeys = jax.random.split(key, 8)
            
            # Random architecture parameters
            num_qubits = int(jax.random.randint(subkeys[0], (), 2, self.config.max_qubits + 1))
            num_layers = int(jax.random.randint(subkeys[1], (), 1, self.config.max_layers + 1))
            
            # Random gate sequence
            gate_options = ['RX', 'RY', 'RZ', 'CNOT', 'CZ', 'H']
            gate_sequence_length = int(jax.random.randint(subkeys[2], (), 3, 8))
            gate_indices = jax.random.randint(
                subkeys[3], (gate_sequence_length,), 0, len(gate_options)
            )
            gate_sequence = [gate_options[int(idx)] for idx in gate_indices]
            
            # Random entanglement pattern
            entanglement_options = ['linear', 'circular', 'star', 'all_to_all', 'random']
            ent_idx = int(jax.random.randint(subkeys[4], (), 0, len(entanglement_options)))
            entanglement_pattern = entanglement_options[ent_idx]
            
            # Random measurement basis
            basis_options = ['Z', 'X', 'Y']
            basis_indices = jax.random.randint(subkeys[5], (num_qubits,), 0, len(basis_options))
            measurement_basis = [basis_options[int(idx)] for idx in basis_indices]
            
            # Parameter sharing (simplified)
            parameter_sharing = {}
            
            genome = QuantumArchitectureGenome(
                num_qubits=num_qubits,
                num_layers=num_layers,
                gate_sequence=gate_sequence,
                parameter_sharing=parameter_sharing,
                entanglement_pattern=entanglement_pattern,
                measurement_basis=measurement_basis
            )
            
            population.append(genome)
        
        self.population = population
        return population
    
    def evaluate_population(
        self, 
        training_data: Any, 
        validation_data: Any
    ) -> List[Tuple[float, Dict[str, float]]]:
        """Evaluate fitness of entire population"""
        
        results = []
        
        for i, genome in enumerate(self.population):
            try:
                fitness, metrics = self.evaluator.evaluate_architecture(
                    genome, training_data, validation_data
                )
                
                # Update genome fitness
                genome.fitness_score = fitness
                genome.efficiency_metric = metrics.get('parameter_efficiency', 0.0)
                genome.complexity_score = metrics.get('num_parameters', 0)
                
                results.append((fitness, metrics))
                
                if i % 10 == 0:
                    logger.info(f"Evaluated {i+1}/{len(self.population)} architectures")
                    
            except Exception as e:
                logger.warning(f"Evaluation failed for genome {i}: {e}")
                results.append((0.0, {}))
                genome.fitness_score = 0.0
        
        return results
    
    def selection(self) -> List[QuantumArchitectureGenome]:
        """Select parents for reproduction using tournament selection"""
        
        tournament_size = max(2, self.config.population_size // 10)
        parents = []
        
        key = jax.random.PRNGKey(self.generation + 42)
        
        for _ in range(self.config.population_size):
            key, subkey = jax.random.split(key)
            
            # Tournament selection
            tournament_indices = jax.random.choice(
                subkey, len(self.population), (tournament_size,), replace=False
            )
            tournament_genomes = [self.population[int(idx)] for idx in tournament_indices]
            
            # Select winner (highest fitness)
            winner = max(tournament_genomes, key=lambda g: g.fitness_score)
            parents.append(winner)
        
        return parents
    
    def crossover(
        self, 
        parent1: QuantumArchitectureGenome, 
        parent2: QuantumArchitectureGenome
    ) -> Tuple[QuantumArchitectureGenome, QuantumArchitectureGenome]:
        """Create offspring through crossover"""
        
        # Create offspring by mixing parent properties
        child1_qubits = parent1.num_qubits if np.random.random() < 0.5 else parent2.num_qubits
        child1_layers = parent1.num_layers if np.random.random() < 0.5 else parent2.num_layers
        
        child2_qubits = parent2.num_qubits if np.random.random() < 0.5 else parent1.num_qubits
        child2_layers = parent2.num_layers if np.random.random() < 0.5 else parent1.num_layers
        
        # Crossover gate sequences
        crossover_point = len(parent1.gate_sequence) // 2
        child1_gates = parent1.gate_sequence[:crossover_point] + parent2.gate_sequence[crossover_point:]
        child2_gates = parent2.gate_sequence[:crossover_point] + parent1.gate_sequence[crossover_point:]
        
        # Mix other properties
        child1_entanglement = parent1.entanglement_pattern if np.random.random() < 0.5 else parent2.entanglement_pattern
        child2_entanglement = parent2.entanglement_pattern if np.random.random() < 0.5 else parent1.entanglement_pattern
        
        child1 = QuantumArchitectureGenome(
            num_qubits=child1_qubits,
            num_layers=child1_layers,
            gate_sequence=child1_gates,
            parameter_sharing={},
            entanglement_pattern=child1_entanglement,
            measurement_basis=['Z'] * child1_qubits
        )
        
        child2 = QuantumArchitectureGenome(
            num_qubits=child2_qubits,
            num_layers=child2_layers,
            gate_sequence=child2_gates,
            parameter_sharing={},
            entanglement_pattern=child2_entanglement,
            measurement_basis=['Z'] * child2_qubits
        )
        
        return child1, child2
    
    def mutate(self, genome: QuantumArchitectureGenome) -> QuantumArchitectureGenome:
        """Mutate genome with specified probability"""
        
        if np.random.random() > self.config.mutation_rate:
            return genome
        
        # Create mutated copy
        mutated = QuantumArchitectureGenome(
            num_qubits=genome.num_qubits,
            num_layers=genome.num_layers,
            gate_sequence=genome.gate_sequence.copy(),
            parameter_sharing=genome.parameter_sharing.copy(),
            entanglement_pattern=genome.entanglement_pattern,
            measurement_basis=genome.measurement_basis.copy()
        )
        
        # Random mutations
        if np.random.random() < 0.3:  # Mutate number of qubits
            mutated.num_qubits = max(2, min(self.config.max_qubits, 
                                           genome.num_qubits + np.random.randint(-2, 3)))
        
        if np.random.random() < 0.3:  # Mutate number of layers
            mutated.num_layers = max(1, min(self.config.max_layers,
                                          genome.num_layers + np.random.randint(-2, 3)))
        
        if np.random.random() < 0.4:  # Mutate gate sequence
            gate_options = ['RX', 'RY', 'RZ', 'CNOT', 'CZ', 'H']
            if mutated.gate_sequence:
                idx = np.random.randint(len(mutated.gate_sequence))
                mutated.gate_sequence[idx] = np.random.choice(gate_options)
        
        if np.random.random() < 0.3:  # Mutate entanglement pattern
            entanglement_options = ['linear', 'circular', 'star', 'all_to_all', 'random']
            mutated.entanglement_pattern = np.random.choice(entanglement_options)
        
        # Reset fitness (needs re-evaluation)
        mutated.fitness_score = 0.0
        
        return mutated
    
    def evolve_generation(self, training_data: Any, validation_data: Any) -> Dict[str, Any]:
        """Evolve one generation"""
        
        # Evaluate current population
        evaluation_results = self.evaluate_population(training_data, validation_data)
        
        # Sort by fitness
        fitness_scores = [result[0] for result in evaluation_results]
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
        sorted_population = [self.population[i] for i in sorted_indices]
        sorted_results = [evaluation_results[i] for i in sorted_indices]
        
        # Store best genomes
        num_elite = max(1, int(self.config.elitism_ratio * self.config.population_size))
        current_best = sorted_population[:num_elite]
        self.best_genomes.extend(current_best)
        
        # Selection
        parents = self.selection()
        
        # Create next generation
        next_generation = []
        
        # Elitism - keep best genomes
        next_generation.extend(current_best)
        
        # Fill rest with offspring
        while len(next_generation) < self.config.population_size:
            # Select two parents
            parent1 = np.random.choice(parents)
            parent2 = np.random.choice(parents)
            
            # Crossover
            if np.random.random() < self.config.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            next_generation.extend([child1, child2])
        
        # Trim to population size
        next_generation = next_generation[:self.config.population_size]
        self.population = next_generation
        
        # Generation statistics
        generation_stats = {
            'generation': self.generation,
            'best_fitness': fitness_scores[sorted_indices[0]],
            'mean_fitness': np.mean(fitness_scores),
            'std_fitness': np.std(fitness_scores),
            'best_genome': sorted_population[0],
            'evaluation_results': sorted_results[0][1]
        }
        
        self.search_history.append(generation_stats)
        self.generation += 1
        
        logger.info(f"Generation {self.generation}: Best fitness = {generation_stats['best_fitness']:.4f}")
        
        return generation_stats
    
    def search(self, training_data: Any, validation_data: Any) -> QuantumArchitectureGenome:
        """Run complete neural architecture search"""
        
        logger.info("Starting Quantum Neural Architecture Search...")
        
        # Initialize population
        self.initialize_population()
        logger.info(f"Initialized population of {len(self.population)} architectures")
        
        # Evolve for specified generations
        for generation in range(self.config.num_generations):
            generation_stats = self.evolve_generation(training_data, validation_data)
            
            # Early stopping if convergence is reached
            if generation > 10:
                recent_best = [stats['best_fitness'] for stats in self.search_history[-10:]]
                if max(recent_best) - min(recent_best) < 0.001:  # Converged
                    logger.info(f"Convergence reached at generation {generation}")
                    break
        
        # Return best architecture found
        all_best = self.best_genomes + self.population
        best_genome = max(all_best, key=lambda g: g.fitness_score)
        
        logger.info(f"Search completed. Best fitness: {best_genome.fitness_score:.4f}")
        logger.info(f"Best architecture: {best_genome.num_qubits} qubits, {best_genome.num_layers} layers")
        
        return best_genome
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get comprehensive search summary"""
        
        if not self.search_history:
            return {'error': 'No search history available'}
        
        best_fitnesses = [stats['best_fitness'] for stats in self.search_history]
        mean_fitnesses = [stats['mean_fitness'] for stats in self.search_history]
        
        summary = {
            'total_generations': len(self.search_history),
            'best_fitness_overall': max(best_fitnesses),
            'final_best_fitness': best_fitnesses[-1],
            'fitness_improvement': best_fitnesses[-1] - best_fitnesses[0],
            'convergence_generation': len(self.search_history),
            'population_diversity': np.mean([stats['std_fitness'] for stats in self.search_history]),
            'search_efficiency': max(best_fitnesses) / len(self.search_history),
            'fitness_progression': best_fitnesses,
            'mean_fitness_progression': mean_fitnesses
        }
        
        return summary


# Factory function
def create_quantum_nas(
    config: Optional[QuantumNASConfig] = None,
    evaluator_type: str = "simulator",
    **kwargs
) -> QuantumNeuralArchitectureSearch:
    """Factory for creating QNAS instances"""
    
    if config is None:
        config = QuantumNASConfig(**kwargs)
    
    if evaluator_type == "simulator":
        evaluator = QuantumCircuitSimulator()
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")
    
    return QuantumNeuralArchitectureSearch(config, evaluator)


# Export main components
__all__ = [
    'QuantumArchitectureGenome',
    'QuantumNASConfig', 
    'QuantumArchitectureEvaluator',
    'QuantumCircuitSimulator',
    'QuantumNeuralArchitectureSearch',
    'create_quantum_nas'
]