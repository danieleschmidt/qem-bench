"""
Learning Engine for Adaptive Error Mitigation

Implements the core learning algorithms and experience management for adaptive
error mitigation systems. Provides meta-learning, transfer learning, and
experience replay capabilities.

Research Contributions:
- Meta-learning algorithms for faster adaptation to new devices
- Experience replay with prioritized sampling
- Causal inference for understanding mitigation mechanisms
- Online learning with concept drift detection
- Multi-task learning across different error types
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from collections import deque
from enum import Enum

from .device_profiler import DeviceProfile


class LearningMode(Enum):
    """Learning modes for the engine"""
    ONLINE = "online"
    BATCH = "batch"
    META = "meta"
    TRANSFER = "transfer"


@dataclass
class Experience:
    """Single experience sample for learning"""
    
    state: Dict[str, Any]  # Circuit, device, parameters
    action: Dict[str, Any]  # Mitigation strategy chosen
    reward: float  # Performance improvement achieved
    next_state: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experience to dictionary"""
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority
        }


class ExperienceBuffer:
    """
    Prioritized experience replay buffer for learning
    
    Implements prioritized experience replay with temporal considerations
    for adaptive error mitigation learning.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        alpha: float = 0.6,  # Prioritization exponent
        beta: float = 0.4,   # Importance sampling exponent
        epsilon: float = 1e-6
    ):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        
        self.buffer: deque = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.position = 0
    
    def add(self, experience: Experience):
        """Add experience to buffer with priority"""
        
        # Calculate initial priority (high for new experiences)
        priority = max(self.priorities) if self.priorities else 1.0
        
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], List[int], jnp.ndarray]:
        """Sample batch with prioritized sampling"""
        
        if len(self.buffer) < batch_size:
            # Return all available experiences
            indices = list(range(len(self.buffer)))
            weights = jnp.ones(len(self.buffer))
            return list(self.buffer), indices, weights
        
        # Convert priorities to probabilities
        priorities = jnp.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities = probabilities / jnp.sum(probabilities)
        
        # Sample indices
        indices = jax.random.choice(
            jax.random.PRNGKey(np.random.randint(0, 10000)),
            len(self.buffer),
            shape=(batch_size,),
            replace=False,
            p=probabilities
        )
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / jnp.max(weights)  # Normalize
        
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, list(indices), weights
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities based on TD errors"""
        
        for i, td_error in zip(indices, td_errors):
            priority = abs(td_error) + self.epsilon
            if i < len(self.priorities):
                self.priorities[i] = priority
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            "size": len(self.buffer),
            "max_size": self.max_size,
            "avg_priority": float(np.mean(self.priorities)) if self.priorities else 0.0,
            "max_priority": float(np.max(self.priorities)) if self.priorities else 0.0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert buffer to dictionary for serialization"""
        return {
            "experiences": [exp.to_dict() for exp in list(self.buffer)[-100:]],  # Last 100 for size
            "statistics": self.get_statistics()
        }


class LearningEngine:
    """
    Core learning engine for adaptive error mitigation
    
    This class implements advanced learning algorithms for adaptive quantum
    error mitigation:
    
    1. Meta-learning for rapid adaptation to new devices
    2. Transfer learning across different quantum platforms
    3. Causal inference for understanding mitigation mechanisms
    4. Online learning with drift detection
    5. Multi-task learning for different error types
    6. Experience replay with prioritized sampling
    
    The engine serves as the intelligence layer that enables adaptive systems
    to learn from experience and improve over time.
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        min_data_points: int = 10,
        learning_modes: List[LearningMode] = None,
        meta_learning_rate: float = 0.001,
        transfer_learning_rate: float = 0.01
    ):
        self.buffer_size = buffer_size
        self.min_data_points = min_data_points
        self.learning_modes = learning_modes or [LearningMode.ONLINE, LearningMode.META]
        self.meta_learning_rate = meta_learning_rate
        self.transfer_learning_rate = transfer_learning_rate
        
        # Experience management
        self.experience_buffer = ExperienceBuffer(max_size=buffer_size)
        self.recent_experiences: List[Experience] = []
        
        # Meta-learning components
        self.meta_parameters: Dict[str, jnp.ndarray] = {}
        self.task_embeddings: Dict[str, jnp.ndarray] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Transfer learning components
        self.source_domains: Dict[str, Dict[str, Any]] = {}
        self.transfer_matrix: Optional[jnp.ndarray] = None
        
        # Causal inference components
        self.causal_graph: Dict[str, List[str]] = {}
        self.causal_strengths: Dict[Tuple[str, str], float] = {}
        
        # JAX compiled functions
        self._meta_gradient_step = jax.jit(self._compute_meta_gradient)
        self._transfer_weights = jax.jit(self._compute_transfer_weights)
        self._causal_inference = jax.jit(self._perform_causal_inference)
        
        # Performance tracking
        self.performance_history: List[Dict[str, float]] = []
        self.adaptation_times: List[float] = []
        self.transfer_gains: List[float] = []
        
        # Research metrics
        self._research_metrics = {
            "total_adaptations": 0,
            "successful_transfers": 0,
            "causal_relationships_discovered": 0,
            "meta_learning_improvements": [],
            "concept_drift_events": 0
        }
    
    def add_experience(self, experience_data: Dict[str, Any]):
        """Add new experience for learning"""
        
        try:
            # Convert to Experience object
            experience = Experience(
                state={
                    "circuit_features": experience_data.get("circuit_features", {}),
                    "backend_features": experience_data.get("backend_features", {}),
                    "device_profile": experience_data.get("device_profile")
                },
                action={
                    "parameters": experience_data.get("parameters", {}),
                    "method": experience_data.get("method", "zne")
                },
                reward=self._calculate_reward(experience_data),
                metadata=experience_data.get("metadata", {}),
                timestamp=experience_data.get("timestamp", datetime.now())
            )
            
            # Add to buffer
            self.experience_buffer.add(experience)
            self.recent_experiences.append(experience)
            
            # Maintain recent experiences limit
            if len(self.recent_experiences) > 100:
                self.recent_experiences.pop(0)
            
            # Trigger learning if conditions met
            if len(self.experience_buffer.buffer) % 10 == 0:  # Learn every 10 experiences
                self._trigger_learning()
                
        except Exception as e:
            warnings.warn(f"Failed to add experience: {e}")
    
    def _calculate_reward(self, experience_data: Dict[str, Any]) -> float:
        """Calculate reward signal from experience"""
        
        performance_metrics = experience_data.get("performance_metrics", {})
        
        # Multi-objective reward combining accuracy, speed, and cost
        accuracy = performance_metrics.get("accuracy", 0.5)
        speed = performance_metrics.get("speed", 0.5)
        cost = performance_metrics.get("cost", 0.5)
        
        # Weighted combination (can be learned/adapted)
        reward = 0.6 * accuracy + 0.2 * speed + 0.2 * (1 - cost)  # Cost is penalty
        
        return float(reward)
    
    def _trigger_learning(self):
        """Trigger learning based on available modes"""
        
        try:
            if LearningMode.ONLINE in self.learning_modes:
                self._perform_online_learning()
            
            if LearningMode.META in self.learning_modes:
                self._perform_meta_learning()
            
            if LearningMode.TRANSFER in self.learning_modes:
                self._perform_transfer_learning()
            
            # Update causal model
            self._update_causal_model()
            
        except Exception as e:
            warnings.warn(f"Learning trigger failed: {e}")
    
    def _perform_online_learning(self):
        """Perform online learning from recent experiences"""
        
        if len(self.recent_experiences) < 5:
            return
        
        # Sample recent experiences
        recent_batch = self.recent_experiences[-10:]
        
        # Extract patterns and update beliefs
        performance_trend = self._analyze_performance_trend(recent_batch)
        
        # Detect concept drift
        if self._detect_concept_drift(recent_batch):
            self._research_metrics["concept_drift_events"] += 1
            self._handle_concept_drift()
        
        # Update performance history
        avg_performance = np.mean([exp.reward for exp in recent_batch])
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "performance": avg_performance,
            "trend": performance_trend
        })
    
    def _perform_meta_learning(self):
        """Perform meta-learning for faster adaptation"""
        
        if len(self.experience_buffer.buffer) < 50:
            return
        
        try:
            # Sample diverse experiences for meta-learning
            experiences, indices, weights = self.experience_buffer.sample(20)
            
            # Group experiences by task/device
            task_groups = self._group_experiences_by_task(experiences)
            
            # Update meta-parameters using MAML-style learning
            for task_id, task_experiences in task_groups.items():
                if len(task_experiences) >= 3:
                    meta_gradient = self._compute_meta_learning_gradient(task_experiences)
                    self._update_meta_parameters(task_id, meta_gradient)
            
            self._research_metrics["total_adaptations"] += 1
            
        except Exception as e:
            warnings.warn(f"Meta-learning failed: {e}")
    
    def _perform_transfer_learning(self):
        """Perform transfer learning across domains"""
        
        if len(self.source_domains) < 2:
            return
        
        try:
            # Identify source and target domains
            domain_similarities = self._compute_domain_similarities()
            
            # Perform knowledge transfer
            for target_domain, source_domain in domain_similarities.items():
                if source_domain["similarity"] > 0.7:  # High similarity threshold
                    transfer_gain = self._transfer_knowledge(source_domain["id"], target_domain)
                    if transfer_gain > 0.1:  # Significant gain
                        self.transfer_gains.append(transfer_gain)
                        self._research_metrics["successful_transfers"] += 1
            
        except Exception as e:
            warnings.warn(f"Transfer learning failed: {e}")
    
    def _update_causal_model(self):
        """Update causal model for mechanism understanding"""
        
        if len(self.experience_buffer.buffer) < 20:
            return
        
        try:
            # Sample experiences for causal analysis
            experiences, _, _ = self.experience_buffer.sample(50)
            
            # Extract causal features
            causal_data = self._extract_causal_features(experiences)
            
            # Perform causal discovery (simplified)
            new_relationships = self._discover_causal_relationships(causal_data)
            
            # Update causal graph
            for cause, effect in new_relationships:
                if cause not in self.causal_graph:
                    self.causal_graph[cause] = []
                if effect not in self.causal_graph[cause]:
                    self.causal_graph[cause].append(effect)
                    self.causal_strengths[(cause, effect)] = new_relationships[(cause, effect)]
                    self._research_metrics["causal_relationships_discovered"] += 1
            
        except Exception as e:
            warnings.warn(f"Causal model update failed: {e}")
    
    def _analyze_performance_trend(self, experiences: List[Experience]) -> str:
        """Analyze performance trend from experiences"""
        
        rewards = [exp.reward for exp in experiences]
        
        if len(rewards) < 3:
            return "insufficient_data"
        
        # Simple trend analysis
        recent_avg = np.mean(rewards[-3:])
        earlier_avg = np.mean(rewards[:-3]) if len(rewards) > 3 else np.mean(rewards)
        
        if recent_avg > earlier_avg + 0.1:
            return "improving"
        elif recent_avg < earlier_avg - 0.1:
            return "degrading"
        else:
            return "stable"
    
    def _detect_concept_drift(self, experiences: List[Experience]) -> bool:
        """Detect concept drift in the data distribution"""
        
        if len(experiences) < 10:
            return False
        
        # Simple drift detection based on performance variance
        rewards = [exp.reward for exp in experiences]
        
        recent_variance = np.var(rewards[-5:])
        historical_variance = np.var(rewards[:-5]) if len(rewards) > 5 else 0.1
        
        # Drift detected if variance significantly increases
        return recent_variance > 2 * historical_variance and recent_variance > 0.1
    
    def _handle_concept_drift(self):
        """Handle detected concept drift"""
        
        # Reset recent learning to adapt to new distribution
        self.recent_experiences = self.recent_experiences[-5:]  # Keep only very recent
        
        # Increase learning rate temporarily
        self.meta_learning_rate *= 1.5
        self.transfer_learning_rate *= 1.5
        
        # Would implement more sophisticated drift adaptation in practice
    
    def _group_experiences_by_task(self, experiences: List[Experience]) -> Dict[str, List[Experience]]:
        """Group experiences by task/device for meta-learning"""
        
        task_groups = {}
        
        for exp in experiences:
            # Create task identifier from device and circuit characteristics
            device_name = exp.state.get("device_profile", {}).get("device_name", "unknown")
            circuit_type = self._classify_circuit_type(exp.state.get("circuit_features", {}))
            task_id = f"{device_name}_{circuit_type}"
            
            if task_id not in task_groups:
                task_groups[task_id] = []
            task_groups[task_id].append(exp)
        
        return task_groups
    
    def _classify_circuit_type(self, circuit_features: Dict[str, float]) -> str:
        """Classify circuit type for task grouping"""
        
        depth = circuit_features.get("depth", 10)
        entanglement = circuit_features.get("entanglement_measure", 0.5)
        
        if depth > 50:
            return "deep"
        elif entanglement > 0.7:
            return "highly_entangled"
        elif depth < 10:
            return "shallow"
        else:
            return "medium"
    
    @jax.jit
    def _compute_meta_gradient(
        self,
        task_params: jnp.ndarray,
        task_data: jnp.ndarray,
        meta_params: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute meta-gradient for MAML-style learning"""
        
        # Simplified meta-gradient computation
        # In practice, would use proper MAML implementation
        
        # Inner loop: adapt parameters to task
        adapted_params = task_params + self.meta_learning_rate * meta_params
        
        # Outer loop: compute meta-gradient
        task_loss = jnp.mean((adapted_params - task_data) ** 2)  # Simplified loss
        meta_gradient = jax.grad(lambda mp: jnp.mean((task_params + self.meta_learning_rate * mp - task_data) ** 2))(meta_params)
        
        return meta_gradient
    
    def _compute_meta_learning_gradient(self, task_experiences: List[Experience]) -> jnp.ndarray:
        """Compute meta-learning gradient from task experiences"""
        
        # Extract features and rewards
        features = []
        rewards = []
        
        for exp in task_experiences:
            feature_vec = self._experience_to_feature_vector(exp)
            features.append(feature_vec)
            rewards.append(exp.reward)
        
        features = jnp.array(features)
        rewards = jnp.array(rewards)
        
        # Initialize meta-parameters if not present
        if "default" not in self.meta_parameters:
            self.meta_parameters["default"] = jax.random.normal(
                jax.random.PRNGKey(42), (features.shape[1],)
            )
        
        # Compute meta-gradient
        meta_gradient = self._meta_gradient_step(
            features[0],  # Use first sample as task params
            rewards,
            self.meta_parameters["default"]
        )
        
        return meta_gradient
    
    def _update_meta_parameters(self, task_id: str, meta_gradient: jnp.ndarray):
        """Update meta-parameters for specific task"""
        
        if task_id not in self.meta_parameters:
            self.meta_parameters[task_id] = jax.random.normal(
                jax.random.PRNGKey(42), meta_gradient.shape
            )
        
        # Meta-parameter update
        self.meta_parameters[task_id] = (
            self.meta_parameters[task_id] - self.meta_learning_rate * meta_gradient
        )
        
        # Track improvement
        improvement = float(jnp.linalg.norm(meta_gradient))
        self._research_metrics["meta_learning_improvements"].append(improvement)
    
    def _compute_domain_similarities(self) -> Dict[str, Dict[str, Any]]:
        """Compute similarities between domains for transfer learning"""
        
        similarities = {}
        
        # Extract domain characteristics
        current_domain_features = self._extract_current_domain_features()
        
        for domain_id, domain_data in self.source_domains.items():
            similarity = self._compute_domain_similarity(
                current_domain_features, domain_data.get("features", [])
            )
            
            similarities[f"current_to_{domain_id}"] = {
                "id": domain_id,
                "similarity": similarity,
                "features": domain_data.get("features", [])
            }
        
        return similarities
    
    def _extract_current_domain_features(self) -> jnp.ndarray:
        """Extract features characterizing current domain"""
        
        if not self.recent_experiences:
            return jnp.zeros(10)
        
        # Aggregate features from recent experiences
        feature_vectors = [
            self._experience_to_feature_vector(exp) 
            for exp in self.recent_experiences[-10:]
        ]
        
        if feature_vectors:
            return jnp.mean(jnp.array(feature_vectors), axis=0)
        else:
            return jnp.zeros(10)
    
    def _compute_domain_similarity(
        self, 
        features1: jnp.ndarray, 
        features2: List[float]
    ) -> float:
        """Compute similarity between two domains"""
        
        if len(features2) == 0:
            return 0.0
        
        features2 = jnp.array(features2[:len(features1)])  # Match dimensions
        
        # Cosine similarity
        similarity = jnp.dot(features1, features2) / (
            jnp.linalg.norm(features1) * jnp.linalg.norm(features2) + 1e-8
        )
        
        return float(jnp.clip(similarity, 0.0, 1.0))
    
    def _transfer_knowledge(self, source_domain_id: str, target_domain: str) -> float:
        """Transfer knowledge from source to target domain"""
        
        if source_domain_id not in self.source_domains:
            return 0.0
        
        # Simple knowledge transfer (would be more sophisticated in practice)
        source_performance = self.source_domains[source_domain_id].get("avg_performance", 0.5)
        
        # Estimate performance gain from transfer
        current_performance = np.mean([exp.reward for exp in self.recent_experiences[-5:]])
        transfer_gain = max(0.0, source_performance - current_performance) * 0.5  # Conservative transfer
        
        return transfer_gain
    
    def _extract_causal_features(self, experiences: List[Experience]) -> Dict[str, List[float]]:
        """Extract features for causal analysis"""
        
        causal_data = {
            "circuit_depth": [],
            "device_error_rate": [],
            "noise_factors": [],
            "ensemble_weights": [],
            "performance": []
        }
        
        for exp in experiences:
            circuit_features = exp.state.get("circuit_features", {})
            backend_features = exp.state.get("backend_features", {})
            parameters = exp.action.get("parameters", {})
            
            causal_data["circuit_depth"].append(circuit_features.get("depth", 10))
            causal_data["device_error_rate"].append(backend_features.get("error_rate", 0.01))
            causal_data["noise_factors"].append(len(parameters.get("noise_factors", [1, 2, 3])))
            causal_data["ensemble_weights"].append(
                max(parameters.get("ensemble_weights", {}).values()) if parameters.get("ensemble_weights") else 0.5
            )
            causal_data["performance"].append(exp.reward)
        
        return causal_data
    
    def _discover_causal_relationships(self, causal_data: Dict[str, List[float]]) -> Dict[Tuple[str, str], float]:
        """Discover causal relationships (simplified implementation)"""
        
        relationships = {}
        variables = list(causal_data.keys())
        
        # Simple correlation-based causal discovery
        for i, var1 in enumerate(variables[:-1]):  # Exclude performance (outcome)
            for j, var2 in enumerate(variables):
                if i != j and len(causal_data[var1]) > 5:
                    correlation = np.corrcoef(causal_data[var1], causal_data[var2])[0, 1]
                    
                    # Simple heuristic for causal direction
                    if abs(correlation) > 0.3:  # Significant correlation
                        if var2 == "performance":  # Everything can cause performance
                            relationships[(var1, var2)] = abs(correlation)
                        elif "circuit" in var1 and "device" not in var1:  # Circuit causes device effects
                            relationships[(var1, var2)] = abs(correlation)
        
        return relationships
    
    def _experience_to_feature_vector(self, experience: Experience) -> jnp.ndarray:
        """Convert experience to feature vector"""
        
        features = []
        
        # Circuit features
        circuit_features = experience.state.get("circuit_features", {})
        features.extend([
            circuit_features.get("depth", 10) / 100.0,
            circuit_features.get("num_qubits", 5) / 20.0,
            circuit_features.get("entanglement_measure", 0.5)
        ])
        
        # Device features
        backend_features = experience.state.get("backend_features", {})
        features.extend([
            backend_features.get("error_rate", 0.01) * 100,
            backend_features.get("coherence_time", 100) / 1000.0
        ])
        
        # Action features
        parameters = experience.action.get("parameters", {})
        features.extend([
            len(parameters.get("noise_factors", [1, 2, 3])) / 10.0,
            parameters.get("learning_rate", 0.01) * 100
        ])
        
        # Reward
        features.append(experience.reward)
        
        # Pad to fixed size
        while len(features) < 10:
            features.append(0.0)
        
        return jnp.array(features[:10])
    
    @jax.jit
    def _compute_transfer_weights(
        self,
        source_features: jnp.ndarray,
        target_features: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute transfer weights between domains"""
        
        similarity = jnp.dot(source_features, target_features) / (
            jnp.linalg.norm(source_features) * jnp.linalg.norm(target_features) + 1e-8
        )
        
        # Convert similarity to transfer weights
        weights = jax.nn.softmax(jnp.array([similarity, 1 - similarity]))
        return weights
    
    @jax.jit
    def _perform_causal_inference(
        self,
        causal_matrix: jnp.ndarray,
        intervention: jnp.ndarray
    ) -> jnp.ndarray:
        """Perform causal inference given causal graph"""
        
        # Simplified causal inference
        causal_effect = jnp.dot(causal_matrix, intervention)
        return causal_effect
    
    def has_sufficient_data(self) -> bool:
        """Check if sufficient data available for learning"""
        return len(self.experience_buffer.buffer) >= self.min_data_points
    
    def get_recent_performance(self) -> Dict[str, float]:
        """Get recent performance metrics"""
        
        if not self.recent_experiences:
            return {"accuracy": 0.5, "speed": 0.5, "cost": 0.5}
        
        recent_rewards = [exp.reward for exp in self.recent_experiences[-5:]]
        
        return {
            "accuracy": np.mean(recent_rewards),
            "speed": np.std(recent_rewards),  # Low std = consistent speed
            "cost": 1.0 - np.mean(recent_rewards)  # Inverse of performance
        }
    
    def predict_next_performance(
        self,
        current_state: Dict[str, Any],
        proposed_action: Dict[str, Any]
    ) -> Dict[str, float]:
        """Predict performance for proposed action"""
        
        if not self.has_sufficient_data():
            return {"accuracy": 0.5, "speed": 0.5, "cost": 0.5, "confidence": 0.0}
        
        # Find similar experiences
        similar_experiences = self._find_similar_experiences(current_state, proposed_action)
        
        if similar_experiences:
            predicted_reward = np.mean([exp.reward for exp in similar_experiences])
            confidence = len(similar_experiences) / 10.0  # More similar experiences = higher confidence
        else:
            predicted_reward = 0.5
            confidence = 0.0
        
        return {
            "accuracy": predicted_reward,
            "speed": predicted_reward,
            "cost": 1.0 - predicted_reward,
            "confidence": min(confidence, 1.0)
        }
    
    def _find_similar_experiences(
        self,
        current_state: Dict[str, Any],
        proposed_action: Dict[str, Any],
        top_k: int = 5
    ) -> List[Experience]:
        """Find experiences similar to current state and proposed action"""
        
        similar_experiences = []
        
        for exp in list(self.experience_buffer.buffer)[-50:]:  # Check recent experiences
            state_similarity = self._compute_state_similarity(current_state, exp.state)
            action_similarity = self._compute_action_similarity(proposed_action, exp.action)
            
            total_similarity = 0.7 * state_similarity + 0.3 * action_similarity
            
            if total_similarity > 0.5:  # Similarity threshold
                exp.metadata["similarity"] = total_similarity
                similar_experiences.append(exp)
        
        # Return top-k most similar
        similar_experiences.sort(key=lambda x: x.metadata.get("similarity", 0), reverse=True)
        return similar_experiences[:top_k]
    
    def _compute_state_similarity(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
        """Compute similarity between states"""
        
        # Simple feature-based similarity
        features1 = self._state_to_features(state1)
        features2 = self._state_to_features(state2)
        
        similarity = jnp.dot(features1, features2) / (
            jnp.linalg.norm(features1) * jnp.linalg.norm(features2) + 1e-8
        )
        
        return float(jnp.clip(similarity, 0.0, 1.0))
    
    def _compute_action_similarity(self, action1: Dict[str, Any], action2: Dict[str, Any]) -> float:
        """Compute similarity between actions"""
        
        # Compare key parameters
        params1 = action1.get("parameters", {})
        params2 = action2.get("parameters", {})
        
        similarities = []
        
        # Compare noise factors
        nf1 = params1.get("noise_factors", [])
        nf2 = params2.get("noise_factors", [])
        
        if nf1 and nf2:
            nf_sim = 1.0 - abs(np.mean(nf1) - np.mean(nf2)) / max(np.mean(nf1), np.mean(nf2), 1.0)
            similarities.append(nf_sim)
        
        # Compare learning rate
        lr1 = params1.get("learning_rate", 0.01)
        lr2 = params2.get("learning_rate", 0.01)
        lr_sim = 1.0 - abs(lr1 - lr2) / max(lr1, lr2, 0.01)
        similarities.append(lr_sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _state_to_features(self, state: Dict[str, Any]) -> jnp.ndarray:
        """Convert state to feature vector"""
        
        features = []
        
        circuit_features = state.get("circuit_features", {})
        features.extend([
            circuit_features.get("depth", 10) / 100.0,
            circuit_features.get("num_qubits", 5) / 20.0
        ])
        
        backend_features = state.get("backend_features", {})
        features.extend([
            backend_features.get("error_rate", 0.01) * 100,
            backend_features.get("coherence_time", 100) / 1000.0
        ])
        
        # Pad to fixed size
        while len(features) < 5:
            features.append(0.0)
        
        return jnp.array(features[:5])
    
    def get_research_statistics(self) -> Dict[str, Any]:
        """Get learning statistics for research analysis"""
        
        return {
            "experience_buffer_size": len(self.experience_buffer.buffer),
            "total_adaptations": self._research_metrics["total_adaptations"],
            "successful_transfers": self._research_metrics["successful_transfers"],
            "causal_relationships": len(self.causal_strengths),
            "concept_drift_events": self._research_metrics["concept_drift_events"],
            "meta_learning_improvements": {
                "count": len(self._research_metrics["meta_learning_improvements"]),
                "mean": np.mean(self._research_metrics["meta_learning_improvements"]) if self._research_metrics["meta_learning_improvements"] else 0.0
            },
            "transfer_gains": {
                "count": len(self.transfer_gains),
                "mean": np.mean(self.transfer_gains) if self.transfer_gains else 0.0
            },
            "learning_modes": [mode.value for mode in self.learning_modes],
            "causal_graph_size": len(self.causal_graph)
        }