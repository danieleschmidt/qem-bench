"""
Quantum Task Optimizer

Advanced optimization strategies using quantum-inspired algorithms
and JAX acceleration for large-scale task planning.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Union
from enum import Enum
import numpy as np

from .core import Task, TaskState, PlanningConfig


class OptimizationStrategy(Enum):
    """Available optimization strategies"""
    QUANTUM_ANNEALING = "quantum_annealing"
    VARIATIONAL_QUANTUM = "variational_quantum" 
    ADIABATIC_QUANTUM = "adiabatic_quantum"
    QUANTUM_APPROXIMATE = "quantum_approximate"
    HYBRID_CLASSICAL = "hybrid_classical"


@dataclass
class OptimizationResult:
    """Result of quantum optimization"""
    optimal_solution: List[str]
    objective_value: float
    convergence_iterations: int
    quantum_fidelity: float
    classical_bound: Optional[float] = None
    computation_time: float = 0.0
    metadata: Dict[str, Any] = None


class QuantumTaskOptimizer:
    """
    Advanced quantum-inspired optimization for task scheduling
    
    Implements multiple quantum optimization strategies:
    - Quantum annealing for combinatorial optimization
    - Variational quantum eigensolvers for constrained problems  
    - Adiabatic quantum computation for continuous optimization
    """
    
    def __init__(self, config: PlanningConfig = None):
        self.config = config or PlanningConfig()
        self._compiled_functions = {}
        self._optimization_cache = {}
        
    @jax.jit
    def _quantum_annealing_step(self, state: jnp.ndarray, hamiltonian: jnp.ndarray, 
                              temperature: float, iteration: int) -> jnp.ndarray:
        """Single step of quantum annealing optimization"""
        key = jax.random.PRNGKey(iteration)
        
        # Apply Hamiltonian evolution with thermal fluctuations
        energy_gradient = -jnp.dot(hamiltonian, state)
        thermal_noise = jax.random.normal(key, state.shape) * jnp.sqrt(temperature)
        
        # Metropolis-Hastings-like update
        proposed_state = state + 0.01 * energy_gradient + 0.001 * thermal_noise
        proposed_state = proposed_state / jnp.linalg.norm(proposed_state)
        
        # Accept/reject based on energy difference
        current_energy = jnp.dot(state, jnp.dot(hamiltonian, state))
        proposed_energy = jnp.dot(proposed_state, jnp.dot(hamiltonian, proposed_state))
        
        accept_prob = jnp.exp(-(proposed_energy - current_energy) / (temperature + 1e-10))
        accept = jax.random.uniform(key) < accept_prob
        
        return jnp.where(accept, proposed_state, state)
    
    @jax.jit
    def _variational_quantum_step(self, params: jnp.ndarray, hamiltonian: jnp.ndarray,
                                ansatz: Callable) -> Tuple[jnp.ndarray, float]:
        """Variational quantum optimization step"""
        # Create ansatz state from parameters
        state = ansatz(params)
        
        # Compute expectation value
        energy = jnp.real(jnp.dot(jnp.conj(state), jnp.dot(hamiltonian, state)))
        
        # Compute gradient
        grad_fn = jax.grad(lambda p: jnp.real(jnp.dot(jnp.conj(ansatz(p)), 
                                                    jnp.dot(hamiltonian, ansatz(p)))))
        gradients = grad_fn(params)
        
        return gradients, energy
    
    @jax.jit  
    def _create_task_hamiltonian(self, task_data: jnp.ndarray, 
                               dependencies: jnp.ndarray) -> jnp.ndarray:
        """Create Hamiltonian encoding task scheduling problem"""
        n_tasks = task_data.shape[0]
        
        # Diagonal terms: individual task costs (complexity, resources)
        diagonal_terms = jnp.diag(task_data[:, 0] + task_data[:, 3])  # complexity + resources
        
        # Off-diagonal terms: dependency constraints and resource conflicts
        dependency_penalty = 1000.0  # Large penalty for dependency violations
        resource_conflict_penalty = 100.0
        
        # Dependency constraints
        dependency_terms = dependency_penalty * dependencies
        
        # Resource conflict terms (simplified)
        resource_matrix = jnp.outer(task_data[:, 3], task_data[:, 3]) * resource_conflict_penalty
        resource_conflicts = jnp.where(jnp.eye(n_tasks), 0, resource_matrix)
        
        return diagonal_terms + dependency_terms + resource_conflicts
    
    def optimize(self, tasks: Dict[str, Task], strategy: OptimizationStrategy = None,
                objective_function: Callable = None) -> OptimizationResult:
        """
        Optimize task scheduling using specified quantum strategy
        
        Args:
            tasks: Dictionary of tasks to optimize
            strategy: Optimization strategy to use
            objective_function: Custom objective function (optional)
            
        Returns:
            OptimizationResult with optimal solution
        """
        if not tasks:
            return OptimizationResult([], 0.0, 0, 0.0)
            
        strategy = strategy or OptimizationStrategy.QUANTUM_ANNEALING
        
        # Prepare task data for quantum optimization
        task_ids = list(tasks.keys())
        task_data = self._prepare_task_data(tasks)
        dependencies = self._create_dependency_matrix(tasks)
        
        # Create problem Hamiltonian
        hamiltonian = self._create_task_hamiltonian(task_data, dependencies)
        
        # Select optimization strategy
        if strategy == OptimizationStrategy.QUANTUM_ANNEALING:
            result = self._quantum_annealing_optimize(hamiltonian, task_ids)
        elif strategy == OptimizationStrategy.VARIATIONAL_QUANTUM:
            result = self._variational_quantum_optimize(hamiltonian, task_ids)
        elif strategy == OptimizationStrategy.ADIABATIC_QUANTUM:
            result = self._adiabatic_quantum_optimize(hamiltonian, task_ids)
        elif strategy == OptimizationStrategy.QUANTUM_APPROXIMATE:
            result = self._quantum_approximate_optimize(hamiltonian, task_ids)
        else:  # HYBRID_CLASSICAL
            result = self._hybrid_classical_optimize(hamiltonian, task_ids, tasks)
            
        return result
    
    def _quantum_annealing_optimize(self, hamiltonian: jnp.ndarray, 
                                  task_ids: List[str]) -> OptimizationResult:
        """Quantum annealing optimization implementation"""
        n_tasks = len(task_ids)
        
        # Initialize quantum state in superposition
        state = jnp.ones(n_tasks) / jnp.sqrt(n_tasks)
        
        # Annealing schedule: start hot, cool down
        initial_temp = 10.0
        final_temp = 0.01
        temperatures = jnp.geomspace(initial_temp, final_temp, self.config.max_iterations)
        
        best_state = state
        best_energy = float('inf')
        
        for iteration in range(self.config.max_iterations):
            temperature = temperatures[iteration]
            
            # Annealing step
            state = self._quantum_annealing_step(state, hamiltonian, temperature, iteration)
            
            # Track best solution
            current_energy = float(jnp.dot(state, jnp.dot(hamiltonian, state)))
            if current_energy < best_energy:
                best_energy = current_energy
                best_state = state.copy()
                
            # Check convergence
            if iteration > 100 and abs(current_energy - best_energy) < self.config.convergence_threshold:
                break
        
        # Convert quantum state to classical solution
        probabilities = jnp.abs(best_state) ** 2
        optimal_order = [task_ids[i] for i in jnp.argsort(-probabilities)]
        
        # Calculate quantum fidelity
        initial_state = jnp.ones(n_tasks) / jnp.sqrt(n_tasks)
        fidelity = float(jnp.abs(jnp.dot(jnp.conj(initial_state), best_state)) ** 2)
        
        return OptimizationResult(
            optimal_solution=optimal_order,
            objective_value=best_energy,
            convergence_iterations=iteration + 1,
            quantum_fidelity=fidelity
        )
    
    def _variational_quantum_optimize(self, hamiltonian: jnp.ndarray,
                                    task_ids: List[str]) -> OptimizationResult:
        """Variational quantum optimization (VQE-style)"""
        n_tasks = len(task_ids)
        n_params = n_tasks * 3  # Parameterized quantum circuit
        
        # Initialize variational parameters
        key = jax.random.PRNGKey(42)
        params = jax.random.normal(key, (n_params,)) * 0.1
        
        # Define parameterized quantum ansatz
        def ansatz(theta):
            # Simple parameterized circuit: rotations + entangling gates
            state = jnp.ones(n_tasks) / jnp.sqrt(n_tasks)  # |+⟩ state
            
            # Apply parameterized rotations
            for i in range(n_tasks):
                rotation_angle = theta[i]
                # Simplified rotation (in practice would use quantum gates)
                phase = jnp.exp(1j * rotation_angle)
                state = state.at[i].multiply(phase)
                
            # Apply entangling interactions
            for i in range(n_tasks - 1):
                coupling_strength = theta[n_tasks + i]
                # Simplified entangling operation
                state = state * jnp.exp(1j * coupling_strength * jnp.roll(state, 1))
                
            return state / jnp.linalg.norm(state)
        
        # Optimization loop
        learning_rate = 0.01
        best_params = params
        best_energy = float('inf')
        
        for iteration in range(self.config.max_iterations):
            gradients, energy = self._variational_quantum_step(params, hamiltonian, ansatz)
            
            # Gradient descent update
            params = params - learning_rate * gradients
            
            # Track best solution
            if energy < best_energy:
                best_energy = energy
                best_params = params.copy()
                
            # Adaptive learning rate
            if iteration % 100 == 0:
                learning_rate *= 0.95
                
            # Check convergence
            if iteration > 100 and abs(energy - best_energy) < self.config.convergence_threshold:
                break
        
        # Get final optimized state
        final_state = ansatz(best_params)
        probabilities = jnp.abs(final_state) ** 2
        optimal_order = [task_ids[i] for i in jnp.argsort(-probabilities)]
        
        # Calculate fidelity
        initial_state = jnp.ones(n_tasks) / jnp.sqrt(n_tasks)
        fidelity = float(jnp.abs(jnp.dot(jnp.conj(initial_state), final_state)) ** 2)
        
        return OptimizationResult(
            optimal_solution=optimal_order,
            objective_value=best_energy,
            convergence_iterations=iteration + 1,
            quantum_fidelity=fidelity
        )
    
    def _adiabatic_quantum_optimize(self, hamiltonian: jnp.ndarray,
                                  task_ids: List[str]) -> OptimizationResult:
        """Adiabatic quantum optimization"""
        n_tasks = len(task_ids)
        
        # Initial Hamiltonian (easy to solve)
        H_initial = jnp.eye(n_tasks)  # Simple identity
        
        # Final Hamiltonian (problem to solve)
        H_final = hamiltonian
        
        # Initialize in ground state of initial Hamiltonian
        eigenvals, eigenvecs = jnp.linalg.eigh(H_initial)
        state = eigenvecs[:, 0]  # Ground state
        
        # Adiabatic evolution
        dt = 1.0 / self.config.max_iterations
        
        for iteration in range(self.config.max_iterations):
            s = iteration / self.config.max_iterations  # Adiabatic parameter
            
            # Interpolate Hamiltonian
            H_t = (1 - s) * H_initial + s * H_final
            
            # Time evolution step (simplified Trotter)
            state = jnp.dot(jax.scipy.linalg.expm(-1j * dt * H_t), state)
            state = state / jnp.linalg.norm(state)
        
        # Measure final state
        probabilities = jnp.abs(state) ** 2
        optimal_order = [task_ids[i] for i in jnp.argsort(-probabilities)]
        
        final_energy = float(jnp.real(jnp.dot(jnp.conj(state), jnp.dot(H_final, state))))
        
        return OptimizationResult(
            optimal_solution=optimal_order,
            objective_value=final_energy,
            convergence_iterations=self.config.max_iterations,
            quantum_fidelity=1.0  # Perfect evolution (idealized)
        )
    
    def _quantum_approximate_optimize(self, hamiltonian: jnp.ndarray,
                                    task_ids: List[str]) -> OptimizationResult:
        """Quantum Approximate Optimization Algorithm (QAOA)"""
        n_tasks = len(task_ids)
        p_layers = 3  # QAOA depth
        
        # QAOA parameters: gamma (problem) and beta (mixing)
        key = jax.random.PRNGKey(123)
        gamma_params = jax.random.uniform(key, (p_layers,)) * jnp.pi
        beta_params = jax.random.uniform(key, (p_layers,)) * jnp.pi / 2
        
        def qaoa_circuit(gamma_vec, beta_vec):
            # Initialize in superposition
            state = jnp.ones(n_tasks) / jnp.sqrt(n_tasks)
            
            for p in range(p_layers):
                # Apply problem Hamiltonian
                state = jnp.dot(jax.scipy.linalg.expm(-1j * gamma_vec[p] * hamiltonian), state)
                
                # Apply mixing Hamiltonian (X rotations)
                mixing_H = jnp.eye(n_tasks)  # Simplified mixing
                state = jnp.dot(jax.scipy.linalg.expm(-1j * beta_vec[p] * mixing_H), state)
                
            return state
        
        # Optimize QAOA parameters
        params = jnp.concatenate([gamma_params, beta_params])
        
        def qaoa_objective(params_vec):
            gamma_vec = params_vec[:p_layers]
            beta_vec = params_vec[p_layers:]
            state = qaoa_circuit(gamma_vec, beta_vec)
            return jnp.real(jnp.dot(jnp.conj(state), jnp.dot(hamiltonian, state)))
        
        # Classical optimization of QAOA parameters
        from jax.example_libraries import optimizers
        
        opt_init, opt_update, get_params = optimizers.adam(0.01)
        opt_state = opt_init(params)
        
        best_energy = float('inf')
        best_state = None
        
        for iteration in range(self.config.max_iterations // 10):  # Fewer iterations for parameter optimization
            current_params = get_params(opt_state)
            
            # Compute gradient and update
            energy, gradients = jax.value_and_grad(qaoa_objective)(current_params)
            opt_state = opt_update(iteration, gradients, opt_state)
            
            if energy < best_energy:
                best_energy = energy
                gamma_best = current_params[:p_layers]
                beta_best = current_params[p_layers:]
                best_state = qaoa_circuit(gamma_best, beta_best)
        
        # Extract solution
        probabilities = jnp.abs(best_state) ** 2
        optimal_order = [task_ids[i] for i in jnp.argsort(-probabilities)]
        
        return OptimizationResult(
            optimal_solution=optimal_order,
            objective_value=best_energy,
            convergence_iterations=iteration + 1,
            quantum_fidelity=0.8  # Approximate
        )
    
    def _hybrid_classical_optimize(self, hamiltonian: jnp.ndarray, task_ids: List[str],
                                 tasks: Dict[str, Task]) -> OptimizationResult:
        """Hybrid quantum-classical optimization"""
        # Use quantum-inspired preprocessing + classical optimization
        n_tasks = len(task_ids)
        
        # Quantum preprocessing: identify clusters and critical paths
        eigenvals, eigenvecs = jnp.linalg.eigh(hamiltonian)
        quantum_features = eigenvecs[:, :min(5, n_tasks)]  # Top eigenvectors
        
        # Classical optimization with quantum features
        def classical_objective(order_indices):
            total_cost = 0.0
            current_time = 0.0
            completed = set()
            
            for idx in order_indices:
                task_id = task_ids[int(idx)]
                task = tasks[task_id]
                
                # Dependency penalty
                if not all(dep in completed for dep in task.dependencies):
                    total_cost += 1000.0  # Large penalty
                
                # Task cost
                duration = task.duration_estimate or 1.0
                total_cost += duration * task.complexity
                
                completed.add(task_id)
                current_time += duration
            
            return total_cost
        
        # Simulated annealing with quantum features
        from scipy.optimize import differential_evolution
        
        bounds = [(0, n_tasks - 1) for _ in range(n_tasks)]
        
        def objective_wrapper(x):
            # Use quantum features to guide classical search
            quantum_bias = jnp.dot(quantum_features.T, x[:quantum_features.shape[1]] if len(x) >= quantum_features.shape[1] else x)
            classical_cost = classical_objective(x)
            return classical_cost + 0.1 * jnp.sum(quantum_bias ** 2)
        
        # Classical optimization
        result = differential_evolution(
            objective_wrapper,
            bounds,
            maxiter=100,
            seed=42
        )
        
        # Convert to task order
        optimal_indices = jnp.argsort(result.x).astype(int)
        optimal_order = [task_ids[i] for i in optimal_indices]
        
        return OptimizationResult(
            optimal_solution=optimal_order,
            objective_value=result.fun,
            convergence_iterations=result.nit,
            quantum_fidelity=0.5,  # Hybrid approach
            classical_bound=result.fun
        )
    
    def _prepare_task_data(self, tasks: Dict[str, Task]) -> jnp.ndarray:
        """Prepare task data matrix for optimization"""
        task_data = []
        for task in tasks.values():
            row = [
                task.complexity,
                task.priority,
                len(task.dependencies),
                sum(task.resources.values()) if task.resources else 0.0
            ]
            task_data.append(row)
        return jnp.array(task_data)
    
    def _create_dependency_matrix(self, tasks: Dict[str, Task]) -> jnp.ndarray:
        """Create dependency constraint matrix"""
        task_ids = list(tasks.keys())
        n_tasks = len(task_ids)
        id_to_idx = {task_id: i for i, task_id in enumerate(task_ids)}
        
        dependency_matrix = jnp.zeros((n_tasks, n_tasks))
        
        for task_id, task in tasks.items():
            i = id_to_idx[task_id]
            for dep_id in task.dependencies:
                if dep_id in id_to_idx:
                    j = id_to_idx[dep_id]
                    dependency_matrix = dependency_matrix.at[i, j].set(1.0)
        
        return dependency_matrix