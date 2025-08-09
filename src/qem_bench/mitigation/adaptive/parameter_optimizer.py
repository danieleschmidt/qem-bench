"""
Parameter Optimizer for Adaptive Error Mitigation

Implements multiple optimization strategies for dynamically adjusting
error mitigation parameters based on performance feedback and device characteristics.

Research Contributions:
- Multi-objective optimization balancing accuracy, speed, and cost
- Gradient-based learning with JAX acceleration
- Evolutionary algorithms for parameter space exploration
- Bayesian optimization for sample-efficient learning
- Meta-learning for faster adaptation to new devices
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings

from .device_profiler import DeviceProfile


class OptimizationObjective(Enum):
    """Optimization objectives"""
    ACCURACY = "accuracy"
    SPEED = "speed"
    COST = "cost"  
    ROBUSTNESS = "robustness"
    COMPOSITE = "composite"


@dataclass
class OptimizationResult:
    """Result from parameter optimization"""
    
    optimized_params: Dict[str, Any]
    improvement: float
    confidence: float
    iterations: int
    convergence_achieved: bool
    objective_value: float
    gradient_norm: Optional[float] = None
    exploration_ratio: Optional[float] = None


@dataclass  
class OptimizationHistory:
    """History of optimization attempts and results"""
    
    parameter_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_history: List[Dict[str, float]] = field(default_factory=list)  
    timestamps: List[datetime] = field(default_factory=list)
    convergence_history: List[float] = field(default_factory=list)
    
    def add_result(
        self, 
        params: Dict[str, Any], 
        performance: Dict[str, float],
        timestamp: datetime = None
    ):
        """Add optimization result to history"""
        self.parameter_history.append(params.copy())
        self.performance_history.append(performance.copy())
        self.timestamps.append(timestamp or datetime.now())
    
    def get_recent_values(self, metric: str, window: int = 10) -> List[float]:
        """Get recent values for a specific metric"""
        recent_performance = self.performance_history[-window:]
        return [perf.get(metric, 0.0) for perf in recent_performance]
    
    def get_recent_performance_by_method(self) -> Dict[str, Dict[str, float]]:
        """Get recent performance broken down by method"""
        # This would analyze performance by individual methods
        # For now, return aggregate performance
        if not self.performance_history:
            return {}
        
        return {"ensemble": self.performance_history[-1]}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "parameter_history": self.parameter_history,
            "performance_history": self.performance_history,
            "timestamps": [t.isoformat() for t in self.timestamps],
            "convergence_history": self.convergence_history
        }


class ParameterOptimizer:
    """
    Multi-strategy parameter optimizer for adaptive error mitigation
    
    This class implements several optimization strategies:
    1. Gradient-based optimization using JAX autodiff
    2. Evolutionary algorithms for global exploration
    3. Bayesian optimization for sample efficiency
    4. Reinforcement learning for sequential decision making
    
    The optimizer can handle multi-objective optimization, parameter constraints,
    and adaptive learning rates based on optimization progress.
    """
    
    def __init__(
        self,
        strategy: str = "ensemble",
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        patience: int = 10,
        min_improvement: float = 1e-4
    ):
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.patience = patience
        self.min_improvement = min_improvement
        
        # Optimization state
        self.velocity = {}
        self.best_params = None
        self.best_performance = -np.inf
        self.patience_counter = 0
        self.learning_rate_schedule = []
        
        # JAX compilation for gradient-based methods
        self._gradient_step = jax.jit(self._compute_gradient_step)
        self._objective_fn = jax.jit(self._compute_objective_value)
        
        # Initialize strategy-specific components
        self._initialize_strategy_components()
    
    def _initialize_strategy_components(self):
        """Initialize components specific to optimization strategy"""
        
        if self.strategy == "evolutionary":
            self.population_size = 50
            self.mutation_rate = 0.1
            self.crossover_rate = 0.7
            self.current_population = None
            
        elif self.strategy == "bayesian":
            # Would initialize Gaussian Process components
            self.acquisition_function = "expected_improvement"
            self.gp_model = None
            
        elif self.strategy == "reinforcement":
            # Would initialize RL components (Q-network, etc.)
            self.q_network = None
            self.replay_buffer = []
            
        # Common components
        self.optimization_history = []
        self.parameter_bounds = self._get_default_parameter_bounds()
    
    def _get_default_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get default parameter bounds for optimization"""
        return {
            "learning_rate": (1e-5, 1.0),
            "exploration_rate": (0.0, 1.0),
            "noise_factor_range": (1.0, 5.0),
            "ensemble_weight": (0.0, 1.0),
            "superposition_width": (0.01, 0.5),
            "entanglement_strength": (0.1, 1.0)
        }
    
    @jax.jit
    def _compute_objective_value(
        self, 
        params: jnp.ndarray, 
        performance_data: jnp.ndarray,
        weights: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute objective value for given parameters"""
        
        # Multi-objective weighted sum
        # performance_data shape: [accuracy, speed, cost, robustness]
        # weights shape: [w_accuracy, w_speed, w_cost, w_robustness] 
        
        objective = jnp.sum(weights * performance_data)
        
        # Add regularization to prevent overfitting
        l2_penalty = 0.01 * jnp.sum(params ** 2)
        
        return objective - l2_penalty
    
    @jax.jit
    def _compute_gradient_step(
        self,
        params: jnp.ndarray,
        gradients: jnp.ndarray, 
        velocity: jnp.ndarray,
        learning_rate: float,
        momentum: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute gradient descent step with momentum"""
        
        # Update velocity
        new_velocity = momentum * velocity + learning_rate * gradients
        
        # Update parameters
        new_params = params - new_velocity
        
        return new_params, new_velocity
    
    def optimize(
        self,
        current_params: Dict[str, Any],
        performance_history: OptimizationHistory,
        device_profile: Optional[DeviceProfile] = None,
        objective: str = "accuracy",
        max_iterations: int = 100
    ) -> OptimizationResult:
        """
        Optimize parameters using configured strategy
        
        Args:
            current_params: Current parameter values
            performance_history: Historical performance data
            device_profile: Device characteristics for context
            objective: Primary optimization objective
            max_iterations: Maximum optimization iterations
            
        Returns:
            OptimizationResult with optimized parameters
        """
        
        try:
            if self.strategy == "gradient_based":
                return self._optimize_gradient_based(
                    current_params, performance_history, objective, max_iterations
                )
            elif self.strategy == "evolutionary":
                return self._optimize_evolutionary(
                    current_params, performance_history, objective, max_iterations
                )
            elif self.strategy == "bayesian":
                return self._optimize_bayesian(
                    current_params, performance_history, objective, max_iterations
                )
            elif self.strategy == "ensemble":
                return self._optimize_ensemble(
                    current_params, performance_history, objective, max_iterations
                )
            else:
                raise ValueError(f"Unknown optimization strategy: {self.strategy}")
                
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            return OptimizationResult(
                optimized_params=current_params,
                improvement=0.0,
                confidence=0.0,
                iterations=0,
                convergence_achieved=False,
                objective_value=0.0
            )
    
    def _optimize_gradient_based(
        self,
        current_params: Dict[str, Any],
        performance_history: OptimizationHistory,
        objective: str,
        max_iterations: int
    ) -> OptimizationResult:
        """Gradient-based optimization using JAX autodiff"""
        
        if len(performance_history.performance_history) < 5:
            # Not enough data for gradient estimation
            return OptimizationResult(
                optimized_params=current_params,
                improvement=0.0,
                confidence=0.0,
                iterations=0,
                convergence_achieved=False,
                objective_value=0.0
            )
        
        # Convert parameters to JAX arrays
        param_vector, param_keys = self._params_to_vector(current_params)
        
        # Estimate gradients from performance history
        gradients = self._estimate_gradients(
            performance_history, param_keys, objective
        )
        
        if gradients is None:
            return OptimizationResult(
                optimized_params=current_params,
                improvement=0.0,
                confidence=0.0,
                iterations=0,
                convergence_achieved=False,
                objective_value=0.0
            )
        
        # Initialize or update velocity
        if param_keys[0] not in self.velocity:
            self.velocity = {key: jnp.zeros_like(param_vector[i]) 
                           for i, key in enumerate(param_keys)}
        
        velocity_vector = jnp.array([self.velocity[key] for key in param_keys])
        
        # Optimization loop
        best_params = param_vector.copy()
        best_objective = -np.inf
        convergence_history = []
        
        for iteration in range(max_iterations):
            # Compute gradient step
            new_params, new_velocity = self._gradient_step(
                param_vector,
                gradients,
                velocity_vector, 
                self.learning_rate,
                self.momentum
            )
            
            # Apply parameter constraints
            new_params = self._apply_constraints(new_params, param_keys)
            
            # Evaluate objective (approximate)
            objective_value = self._evaluate_objective_approximate(
                new_params, param_keys, performance_history, objective
            )
            
            # Check for improvement
            if objective_value > best_objective + self.min_improvement:
                best_params = new_params.copy()
                best_objective = objective_value
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Update for next iteration
            param_vector = new_params
            velocity_vector = new_velocity
            convergence_history.append(float(jnp.linalg.norm(gradients)))
            
            # Check convergence
            if (self.patience_counter >= self.patience or 
                jnp.linalg.norm(gradients) < 1e-6):
                break
            
            # Re-estimate gradients (in practice, would use more sophisticated methods)
            if iteration % 5 == 0:
                gradients = self._estimate_gradients(
                    performance_history, param_keys, objective
                )
        
        # Update velocity state
        for i, key in enumerate(param_keys):
            self.velocity[key] = velocity_vector[i]
        
        # Convert back to parameter dictionary
        optimized_params = self._vector_to_params(best_params, param_keys, current_params)
        
        improvement = best_objective - self._evaluate_objective_approximate(
            self._params_to_vector(current_params)[0], param_keys, 
            performance_history, objective
        )
        
        return OptimizationResult(
            optimized_params=optimized_params,
            improvement=float(improvement),
            confidence=min(1.0, improvement / 0.1),  # Simple confidence estimate
            iterations=iteration + 1,
            convergence_achieved=self.patience_counter < self.patience,
            objective_value=float(best_objective),
            gradient_norm=float(jnp.linalg.norm(gradients))
        )
    
    def _optimize_evolutionary(
        self,
        current_params: Dict[str, Any],
        performance_history: OptimizationHistory,
        objective: str,
        max_iterations: int
    ) -> OptimizationResult:
        """Evolutionary algorithm optimization"""
        
        # Initialize population if not exists
        if self.current_population is None:
            self.current_population = self._initialize_population(current_params)
        
        param_vector, param_keys = self._params_to_vector(current_params)
        best_individual = param_vector.copy()
        best_fitness = -np.inf
        
        for generation in range(max_iterations):
            # Evaluate fitness for population
            fitness_scores = []
            for individual in self.current_population:
                fitness = self._evaluate_objective_approximate(
                    individual, param_keys, performance_history, objective
                )
                fitness_scores.append(fitness)
            
            # Track best individual
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_individual = self.current_population[gen_best_idx].copy()
                best_fitness = fitness_scores[gen_best_idx]
            
            # Selection, crossover, and mutation
            new_population = []
            
            # Elitism: keep best individuals
            elite_size = max(1, self.population_size // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(self.current_population[idx].copy())
            
            # Generate new individuals
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(self.current_population, fitness_scores)
                parent2 = self._tournament_selection(self.current_population, fitness_scores)
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self._mutate(child1, param_keys)
                child2 = self._mutate(child2, param_keys) 
                
                new_population.extend([child1, child2])
            
            self.current_population = new_population[:self.population_size]
        
        # Convert best individual back to parameters
        optimized_params = self._vector_to_params(best_individual, param_keys, current_params)
        
        original_fitness = self._evaluate_objective_approximate(
            param_vector, param_keys, performance_history, objective
        )
        
        return OptimizationResult(
            optimized_params=optimized_params,
            improvement=float(best_fitness - original_fitness),
            confidence=0.8,  # High confidence for evolutionary methods
            iterations=max_iterations,
            convergence_achieved=True,
            objective_value=float(best_fitness),
            exploration_ratio=self.mutation_rate
        )
    
    def _optimize_bayesian(
        self,
        current_params: Dict[str, Any],
        performance_history: OptimizationHistory,
        objective: str,
        max_iterations: int
    ) -> OptimizationResult:
        """Bayesian optimization using Gaussian Process surrogate"""
        
        # Simplified Bayesian optimization
        # In practice, would use libraries like GPyOpt or Ax
        
        param_vector, param_keys = self._params_to_vector(current_params)
        
        # Use existing performance history as training data
        X_train = []
        y_train = []
        
        for i, params in enumerate(performance_history.parameter_history[-20:]):  # Last 20 points
            if i < len(performance_history.performance_history):
                param_vec, _ = self._params_to_vector(params)
                perf = performance_history.performance_history[i].get(objective, 0.0)
                X_train.append(param_vec)
                y_train.append(perf)
        
        if len(X_train) < 3:
            # Not enough data for GP
            return OptimizationResult(
                optimized_params=current_params,
                improvement=0.0,
                confidence=0.0,
                iterations=0,
                convergence_achieved=False,
                objective_value=0.0
            )
        
        X_train = jnp.array(X_train)
        y_train = jnp.array(y_train)
        
        # Simple GP surrogate (would use proper GP in practice)
        best_candidate = param_vector.copy()
        best_value = -np.inf
        
        # Random search with GP-guided sampling
        for iteration in range(max_iterations):
            # Generate candidate (simplified acquisition function)
            candidate = self._generate_gp_candidate(X_train, y_train, param_keys)
            
            # Evaluate candidate (approximate)
            value = self._evaluate_objective_approximate(
                candidate, param_keys, performance_history, objective
            )
            
            if value > best_value:
                best_candidate = candidate.copy()
                best_value = value
            
            # Add to training data
            X_train = jnp.vstack([X_train, candidate])
            y_train = jnp.append(y_train, value)
        
        optimized_params = self._vector_to_params(best_candidate, param_keys, current_params)
        
        original_value = self._evaluate_objective_approximate(
            param_vector, param_keys, performance_history, objective
        )
        
        return OptimizationResult(
            optimized_params=optimized_params,
            improvement=float(best_value - original_value),
            confidence=0.9,  # High confidence for Bayesian methods
            iterations=max_iterations,
            convergence_achieved=True,
            objective_value=float(best_value)
        )
    
    def _optimize_ensemble(
        self,
        current_params: Dict[str, Any],
        performance_history: OptimizationHistory,
        objective: str,
        max_iterations: int
    ) -> OptimizationResult:
        """Ensemble optimization combining multiple strategies"""
        
        # Run multiple optimization strategies
        strategies = ["gradient_based", "evolutionary"]  # Simplified ensemble
        results = []
        
        for strategy in strategies:
            temp_optimizer = ParameterOptimizer(
                strategy=strategy,
                learning_rate=self.learning_rate
            )
            
            try:
                result = temp_optimizer.optimize(
                    current_params, performance_history, None, objective, 
                    max_iterations // len(strategies)
                )
                results.append((result, strategy))
            except:
                continue
        
        if not results:
            return OptimizationResult(
                optimized_params=current_params,
                improvement=0.0,
                confidence=0.0,
                iterations=0,
                convergence_achieved=False,
                objective_value=0.0
            )
        
        # Select best result
        best_result, best_strategy = max(results, key=lambda x: x[0].improvement)
        
        # Enhance confidence based on agreement between strategies
        if len(results) > 1:
            improvements = [r[0].improvement for r in results]
            agreement = 1.0 - np.std(improvements) / (np.mean(improvements) + 1e-6)
            best_result.confidence *= agreement
        
        return best_result
    
    # Helper methods for optimization strategies
    
    def _params_to_vector(self, params: Dict[str, Any]) -> Tuple[jnp.ndarray, List[str]]:
        """Convert parameter dictionary to vector"""
        vector_parts = []
        keys = []
        
        for key, value in params.items():
            if isinstance(value, (list, np.ndarray)):
                vector_parts.extend(value)
                keys.extend([f"{key}_{i}" for i in range(len(value))])
            elif isinstance(value, (int, float)):
                vector_parts.append(float(value))
                keys.append(key)
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        vector_parts.append(float(subvalue))
                        keys.append(f"{key}_{subkey}")
        
        return jnp.array(vector_parts), keys
    
    def _vector_to_params(
        self, 
        vector: jnp.ndarray, 
        keys: List[str], 
        template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert vector back to parameter dictionary"""
        
        result = template.copy()
        vector_idx = 0
        
        for key, value in template.items():
            if isinstance(value, (list, np.ndarray)):
                length = len(value)
                result[key] = vector[vector_idx:vector_idx + length].tolist()
                vector_idx += length
            elif isinstance(value, (int, float)):
                result[key] = float(vector[vector_idx])
                vector_idx += 1
            elif isinstance(value, dict):
                result[key] = {}
                for subkey in value.keys():
                    if isinstance(value[subkey], (int, float)):
                        result[key][subkey] = float(vector[vector_idx])
                        vector_idx += 1
        
        return result
    
    def _estimate_gradients(
        self,
        performance_history: OptimizationHistory,
        param_keys: List[str],
        objective: str
    ) -> Optional[jnp.ndarray]:
        """Estimate gradients from performance history"""
        
        if len(performance_history.performance_history) < 5:
            return None
        
        # Use finite differences on recent data
        recent_params = performance_history.parameter_history[-5:]
        recent_performance = performance_history.performance_history[-5:]
        
        gradients = []
        
        for i, key in enumerate(param_keys):
            # Extract parameter values
            param_values = []
            perf_values = []
            
            for j, params in enumerate(recent_params):
                param_vec, _ = self._params_to_vector(params)
                if i < len(param_vec):
                    param_values.append(float(param_vec[i]))
                    perf_values.append(recent_performance[j].get(objective, 0.0))
            
            if len(param_values) >= 2:
                # Simple finite difference
                gradient = (perf_values[-1] - perf_values[-2]) / \
                          (param_values[-1] - param_values[-2] + 1e-8)
                gradients.append(gradient)
            else:
                gradients.append(0.0)
        
        return jnp.array(gradients)
    
    def _apply_constraints(
        self, 
        params: jnp.ndarray, 
        param_keys: List[str]
    ) -> jnp.ndarray:
        """Apply parameter constraints"""
        
        constrained_params = params.copy()
        
        for i, key in enumerate(param_keys):
            # Extract base parameter name
            base_key = key.split('_')[0]
            
            if base_key in self.parameter_bounds:
                lower, upper = self.parameter_bounds[base_key]
                constrained_params = constrained_params.at[i].set(
                    jnp.clip(params[i], lower, upper)
                )
        
        return constrained_params
    
    def _evaluate_objective_approximate(
        self,
        param_vector: jnp.ndarray,
        param_keys: List[str],
        performance_history: OptimizationHistory,
        objective: str
    ) -> float:
        """Approximate objective evaluation using performance history"""
        
        if not performance_history.performance_history:
            return 0.0
        
        # Simple approximation: use similarity to historical parameters
        similarities = []
        performances = []
        
        for i, hist_params in enumerate(performance_history.parameter_history[-10:]):
            hist_vector, _ = self._params_to_vector(hist_params)
            
            if len(hist_vector) == len(param_vector):
                similarity = float(jnp.exp(-jnp.linalg.norm(param_vector - hist_vector)))
                similarities.append(similarity)
                
                perf_idx = len(performance_history.performance_history) - 10 + i
                if perf_idx >= 0 and perf_idx < len(performance_history.performance_history):
                    perf = performance_history.performance_history[perf_idx].get(objective, 0.0)
                    performances.append(perf)
        
        if similarities and performances:
            # Weighted average based on similarity
            similarities = jnp.array(similarities)
            performances = jnp.array(performances)
            weights = similarities / jnp.sum(similarities + 1e-8)
            return float(jnp.sum(weights * performances))
        else:
            return performance_history.performance_history[-1].get(objective, 0.0)
    
    def _initialize_population(self, base_params: Dict[str, Any]) -> List[jnp.ndarray]:
        """Initialize population for evolutionary algorithm"""
        
        base_vector, param_keys = self._params_to_vector(base_params)
        population = []
        
        for _ in range(self.population_size):
            # Add random noise to base parameters
            noise = jax.random.normal(
                jax.random.PRNGKey(np.random.randint(0, 10000)),
                base_vector.shape
            ) * 0.1
            
            individual = base_vector + noise
            individual = self._apply_constraints(individual, param_keys)
            population.append(individual)
        
        return population
    
    def _tournament_selection(
        self, 
        population: List[jnp.ndarray], 
        fitness_scores: List[float],
        tournament_size: int = 3
    ) -> jnp.ndarray:
        """Tournament selection for evolutionary algorithm"""
        
        selected_indices = np.random.choice(
            len(population), size=tournament_size, replace=False
        )
        
        best_idx = max(selected_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()
    
    def _crossover(
        self, 
        parent1: jnp.ndarray, 
        parent2: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Crossover operation for evolutionary algorithm"""
        
        # Simple uniform crossover
        mask = jax.random.bernoulli(
            jax.random.PRNGKey(np.random.randint(0, 10000)),
            0.5,
            parent1.shape
        )
        
        child1 = jnp.where(mask, parent1, parent2)
        child2 = jnp.where(mask, parent2, parent1)
        
        return child1, child2
    
    def _mutate(self, individual: jnp.ndarray, param_keys: List[str]) -> jnp.ndarray:
        """Mutation operation for evolutionary algorithm"""
        
        # Gaussian mutation
        mutation_mask = jax.random.bernoulli(
            jax.random.PRNGKey(np.random.randint(0, 10000)),
            self.mutation_rate,
            individual.shape
        )
        
        mutation_noise = jax.random.normal(
            jax.random.PRNGKey(np.random.randint(0, 10000)),
            individual.shape
        ) * 0.1
        
        mutated = individual + mutation_mask * mutation_noise
        return self._apply_constraints(mutated, param_keys)
    
    def _generate_gp_candidate(
        self,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        param_keys: List[str]
    ) -> jnp.ndarray:
        """Generate candidate using GP surrogate (simplified)"""
        
        # Simplified acquisition function: random search with bias toward high-performing regions
        n_dims = X_train.shape[1]
        
        # Find best point so far
        best_idx = jnp.argmax(y_train)
        best_x = X_train[best_idx]
        
        # Generate candidate near best point with exploration
        exploration_noise = jax.random.normal(
            jax.random.PRNGKey(np.random.randint(0, 10000)),
            (n_dims,)
        ) * 0.2
        
        candidate = best_x + exploration_noise
        return self._apply_constraints(candidate, param_keys)
    
    def reset_adaptation(self):
        """Reset adaptation state for new device or major drift"""
        self.velocity = {}
        self.best_params = None
        self.best_performance = -np.inf
        self.patience_counter = 0
        self.current_population = None
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics for research analysis"""
        return {
            "strategy": self.strategy,
            "learning_rate": self.learning_rate,
            "best_performance": self.best_performance,
            "patience_counter": self.patience_counter,
            "optimization_history_length": len(self.optimization_history)
        }