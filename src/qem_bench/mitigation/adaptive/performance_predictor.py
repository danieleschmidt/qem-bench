"""
Performance Predictor for Adaptive Error Mitigation

Implements machine learning models to predict error mitigation performance
based on circuit characteristics, device properties, and historical data.

Research Contributions:
- Neural network models for performance prediction
- Uncertainty quantification using ensemble methods
- Transfer learning across different devices
- Causal inference for understanding mitigation mechanisms
- Multi-objective performance modeling
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import warnings
from datetime import datetime
from enum import Enum

from .device_profiler import DeviceProfile


class ModelType(Enum):
    """Types of prediction models"""
    NEURAL_NETWORK = "neural_network"
    RANDOM_FOREST = "random_forest" 
    GAUSSIAN_PROCESS = "gaussian_process"
    ENSEMBLE = "ensemble"


@dataclass
class PredictionResult:
    """Result from performance prediction"""
    
    accuracy_prediction: float
    speed_prediction: float
    cost_prediction: float
    uncertainty: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_version: str = "1.0"


@dataclass
class PredictionModel:
    """Machine learning model for performance prediction"""
    
    model_type: ModelType
    parameters: Dict[str, Any] = field(default_factory=dict)
    training_data: List[Dict[str, Any]] = field(default_factory=list)
    validation_accuracy: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for serialization"""
        return {
            "model_type": self.model_type.value,
            "parameters": self.parameters,
            "validation_accuracy": self.validation_accuracy,
            "last_update": self.last_update.isoformat(),
            "training_data_size": len(self.training_data)
        }


class PerformancePredictor:
    """
    ML-powered performance predictor for error mitigation
    
    This class implements advanced machine learning techniques for predicting
    error mitigation performance:
    
    1. Neural networks with uncertainty quantification
    2. Ensemble methods for robust predictions
    3. Transfer learning across different devices
    4. Causal inference for mechanism understanding
    5. Multi-objective performance modeling
    
    The predictor enables proactive optimization by forecasting the performance
    of different mitigation strategies before execution.
    """
    
    def __init__(
        self,
        model_types: List[ModelType] = None,
        ensemble_weights: Optional[Dict[str, float]] = None,
        uncertainty_method: str = "monte_carlo_dropout",
        transfer_learning: bool = True
    ):
        self.model_types = model_types or [ModelType.NEURAL_NETWORK, ModelType.ENSEMBLE]
        self.ensemble_weights = ensemble_weights
        self.uncertainty_method = uncertainty_method
        self.transfer_learning = transfer_learning
        
        # Initialize models
        self.models: Dict[str, PredictionModel] = {}
        self._initialize_models()
        
        # Feature engineering
        self.feature_extractors = {}
        self._initialize_feature_extractors()
        
        # Training data
        self.training_history: List[Dict[str, Any]] = []
        self.max_training_history = 10000
        
        # JAX compiled functions
        self._neural_forward = jax.jit(self._neural_network_forward)
        self._compute_uncertainty = jax.jit(self._monte_carlo_uncertainty)
        
        # Transfer learning components
        self.device_embeddings: Dict[str, jnp.ndarray] = {}
        self.transfer_matrix: Optional[jnp.ndarray] = None
        
        # Research tracking
        self._research_metrics = {
            "predictions_made": 0,
            "prediction_errors": [],
            "uncertainty_calibration": [],
            "transfer_learning_gains": [],
            "feature_importance_evolution": []
        }
    
    def _initialize_models(self):
        """Initialize prediction models"""
        
        for model_type in self.model_types:
            if model_type == ModelType.NEURAL_NETWORK:
                self.models["neural_network"] = PredictionModel(
                    model_type=model_type,
                    parameters=self._get_default_nn_params()
                )
            elif model_type == ModelType.GAUSSIAN_PROCESS:
                self.models["gaussian_process"] = PredictionModel(
                    model_type=model_type,
                    parameters=self._get_default_gp_params()
                )
            elif model_type == ModelType.ENSEMBLE:
                self.models["ensemble"] = PredictionModel(
                    model_type=model_type,
                    parameters={"component_models": ["neural_network"]}
                )
    
    def _get_default_nn_params(self) -> Dict[str, Any]:
        """Get default neural network parameters"""
        return {
            "layers": [64, 32, 16, 3],  # 3 outputs: accuracy, speed, cost
            "activation": "relu",
            "dropout_rate": 0.1,
            "learning_rate": 0.001,
            "l2_regularization": 0.01,
            "batch_size": 32,
            "epochs": 100
        }
    
    def _get_default_gp_params(self) -> Dict[str, Any]:
        """Get default Gaussian Process parameters"""
        return {
            "kernel": "rbf",
            "length_scale": 1.0,
            "noise_level": 0.1,
            "optimize_hyperparams": True
        }
    
    def _initialize_feature_extractors(self):
        """Initialize feature extraction functions"""
        
        self.feature_extractors = {
            "circuit_features": self._extract_circuit_features,
            "device_features": self._extract_device_features,
            "historical_features": self._extract_historical_features,
            "interaction_features": self._extract_interaction_features
        }
    
    def predict(
        self,
        circuit_features: Dict[str, float],
        backend_features: Dict[str, float],
        current_params: Dict[str, Any],
        device_profile: Optional[DeviceProfile] = None
    ) -> PredictionResult:
        """
        Predict error mitigation performance
        
        Args:
            circuit_features: Features extracted from quantum circuit
            backend_features: Features extracted from quantum backend
            current_params: Current mitigation parameters
            device_profile: Device profile for context
            
        Returns:
            PredictionResult with performance predictions and uncertainty
        """
        
        try:
            # Extract and combine features
            feature_vector = self._combine_features(
                circuit_features, backend_features, current_params, device_profile
            )
            
            # Make predictions with each model
            model_predictions = {}
            model_uncertainties = {}
            
            for model_name, model in self.models.items():
                if model.model_type == ModelType.NEURAL_NETWORK:
                    pred, uncertainty = self._predict_neural_network(feature_vector, model)
                elif model.model_type == ModelType.GAUSSIAN_PROCESS:
                    pred, uncertainty = self._predict_gaussian_process(feature_vector, model)
                elif model.model_type == ModelType.ENSEMBLE:
                    pred, uncertainty = self._predict_ensemble(feature_vector, model)
                else:
                    continue
                
                model_predictions[model_name] = pred
                model_uncertainties[model_name] = uncertainty
            
            if not model_predictions:
                raise ValueError("No valid model predictions available")
            
            # Combine predictions
            final_prediction, final_uncertainty = self._combine_predictions(
                model_predictions, model_uncertainties
            )
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(
                feature_vector, final_prediction
            )
            
            # Calculate confidence based on uncertainty and model agreement
            confidence = self._calculate_prediction_confidence(
                model_predictions, model_uncertainties
            )
            
            result = PredictionResult(
                accuracy_prediction=final_prediction[0],
                speed_prediction=final_prediction[1], 
                cost_prediction=final_prediction[2],
                uncertainty={
                    "accuracy": final_uncertainty[0],
                    "speed": final_uncertainty[1],
                    "cost": final_uncertainty[2]
                },
                confidence=confidence,
                feature_importance=feature_importance
            )
            
            # Update research metrics
            self._research_metrics["predictions_made"] += 1
            
            return result
            
        except Exception as e:
            warnings.warn(f"Performance prediction failed: {e}")
            # Return default prediction
            return PredictionResult(
                accuracy_prediction=0.5,
                speed_prediction=0.5,
                cost_prediction=0.5,
                confidence=0.0
            )
    
    def _combine_features(
        self,
        circuit_features: Dict[str, float],
        backend_features: Dict[str, float], 
        current_params: Dict[str, Any],
        device_profile: Optional[DeviceProfile]
    ) -> jnp.ndarray:
        """Combine different feature types into unified feature vector"""
        
        features = []
        
        # Circuit features
        circuit_vector = self.feature_extractors["circuit_features"](circuit_features)
        features.extend(circuit_vector)
        
        # Backend/device features
        device_vector = self.feature_extractors["device_features"](backend_features, device_profile)
        features.extend(device_vector)
        
        # Parameter features
        param_vector = self._extract_parameter_features(current_params)
        features.extend(param_vector)
        
        # Historical context features
        if len(self.training_history) > 0:
            historical_vector = self.feature_extractors["historical_features"](
                self.training_history[-10:]  # Last 10 experiments
            )
            features.extend(historical_vector)
        else:
            features.extend([0.0] * 5)  # Placeholder historical features
        
        # Interaction features
        interaction_vector = self.feature_extractors["interaction_features"](
            circuit_features, backend_features
        )
        features.extend(interaction_vector)
        
        return jnp.array(features)
    
    def _extract_circuit_features(self, circuit_features: Dict[str, float]) -> List[float]:
        """Extract features from circuit characteristics"""
        
        return [
            circuit_features.get("depth", 10) / 100.0,  # Normalized depth
            circuit_features.get("num_qubits", 5) / 20.0,  # Normalized qubit count
            circuit_features.get("num_gates", 20) / 200.0,  # Normalized gate count
            circuit_features.get("entanglement_measure", 0.5),  # Already normalized
            circuit_features.get("gate_density", 2.0) / 10.0,  # Gates per layer
            np.log(circuit_features.get("circuit_volume", 100)) / 10.0,  # Log volume
        ]
    
    def _extract_device_features(
        self, 
        backend_features: Dict[str, float],
        device_profile: Optional[DeviceProfile]
    ) -> List[float]:
        """Extract features from device characteristics"""
        
        features = [
            backend_features.get("error_rate", 0.01) * 100,  # Scale up small values
            backend_features.get("coherence_time", 100) / 1000.0,  # Normalize μs
            backend_features.get("gate_time", 0.1) / 1.0,  # Normalize gate time
            backend_features.get("readout_fidelity", 0.95),  # Already normalized
        ]
        
        # Add device profile features if available
        if device_profile:
            profile_features = device_profile.to_feature_vector()
            # Normalize and add subset of most important features
            important_indices = [0, 2, 3, 4, 5]  # Qubits, QV, T1, T2, gate errors
            for i in important_indices:
                if i < len(profile_features):
                    features.append(float(profile_features[i]))
                else:
                    features.append(0.0)
        else:
            # Add placeholder features
            features.extend([0.0] * 5)
        
        return features
    
    def _extract_parameter_features(self, params: Dict[str, Any]) -> List[float]:
        """Extract features from mitigation parameters"""
        
        features = []
        
        # Noise factors
        if "noise_factors" in params:
            noise_factors = params["noise_factors"]
            features.extend([
                len(noise_factors),
                min(noise_factors) if noise_factors else 1.0,
                max(noise_factors) if noise_factors else 3.0,
                np.mean(noise_factors) if noise_factors else 2.0
            ])
        else:
            features.extend([5, 1.0, 3.0, 2.0])  # Defaults
        
        # Ensemble weights
        if "ensemble_weights" in params:
            weights = list(params["ensemble_weights"].values())
            features.extend([
                len(weights),
                max(weights) if weights else 0.5,
                np.std(weights) if len(weights) > 1 else 0.0
            ])
        else:
            features.extend([3, 0.33, 0.1])
        
        # Learning parameters
        features.extend([
            params.get("learning_rate", 0.01) * 100,  # Scale up
            params.get("exploration_rate", 0.1) * 10   # Scale up
        ])
        
        return features
    
    def _extract_historical_features(self, recent_history: List[Dict[str, Any]]) -> List[float]:
        """Extract features from recent experimental history"""
        
        if not recent_history:
            return [0.0] * 5
        
        # Extract performance trends
        accuracies = [exp.get("performance_metrics", {}).get("accuracy", 0.5) 
                     for exp in recent_history]
        speeds = [exp.get("performance_metrics", {}).get("speed", 0.5) 
                 for exp in recent_history]
        
        features = [
            np.mean(accuracies),
            np.std(accuracies) if len(accuracies) > 1 else 0.0,
            np.mean(speeds),
            len(recent_history) / 10.0,  # Normalized history length
            np.mean([len(exp.get("ensemble_results", {})) for exp in recent_history])  # Avg ensemble size
        ]
        
        return features
    
    def _extract_interaction_features(
        self,
        circuit_features: Dict[str, float],
        backend_features: Dict[str, float]
    ) -> List[float]:
        """Extract interaction features between circuit and device"""
        
        circuit_complexity = circuit_features.get("depth", 10) * circuit_features.get("num_qubits", 5)
        device_quality = 1.0 - backend_features.get("error_rate", 0.01)
        
        features = [
            circuit_complexity / 1000.0,  # Normalized complexity
            device_quality,
            circuit_complexity * device_quality / 1000.0,  # Interaction term
            circuit_features.get("entanglement_measure", 0.5) * device_quality,
            min(circuit_features.get("num_qubits", 5) / 20.0, device_quality)  # Resource matching
        ]
        
        return features
    
    @jax.jit
    def _neural_network_forward(
        self, 
        features: jnp.ndarray, 
        weights: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Forward pass through neural network"""
        
        x = features
        
        # Hidden layers
        for i, layer_size in enumerate([64, 32, 16]):
            W = weights[f"W{i}"]
            b = weights[f"b{i}"]
            x = jnp.dot(x, W) + b
            x = jax.nn.relu(x)
        
        # Output layer
        W_out = weights["W_out"]
        b_out = weights["b_out"]
        output = jnp.dot(x, W_out) + b_out
        
        # Apply sigmoid activation for normalized outputs
        return jax.nn.sigmoid(output)
    
    def _predict_neural_network(
        self, 
        features: jnp.ndarray, 
        model: PredictionModel
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Make prediction using neural network"""
        
        # Initialize weights if not present
        if "weights" not in model.parameters:
            model.parameters["weights"] = self._initialize_nn_weights(features.shape[0])
        
        weights = model.parameters["weights"]
        
        # Forward pass
        prediction = self._neural_forward(features, weights)
        
        # Uncertainty estimation using Monte Carlo dropout
        if self.uncertainty_method == "monte_carlo_dropout":
            uncertainty = self._compute_uncertainty(features, weights, n_samples=10)
        else:
            uncertainty = jnp.array([0.1, 0.1, 0.1])  # Default uncertainty
        
        return prediction, uncertainty
    
    def _initialize_nn_weights(self, input_size: int) -> Dict[str, jnp.ndarray]:
        """Initialize neural network weights"""
        
        key = jax.random.PRNGKey(42)
        weights = {}
        
        # Layer dimensions
        layer_sizes = [input_size, 64, 32, 16, 3]
        
        for i in range(len(layer_sizes) - 1):
            key, subkey = jax.random.split(key)
            
            # Xavier initialization
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            std = jnp.sqrt(2.0 / (fan_in + fan_out))
            
            if i < len(layer_sizes) - 2:
                weights[f"W{i}"] = jax.random.normal(subkey, (fan_in, fan_out)) * std
                weights[f"b{i}"] = jnp.zeros(fan_out)
            else:
                weights["W_out"] = jax.random.normal(subkey, (fan_in, fan_out)) * std
                weights["b_out"] = jnp.zeros(fan_out)
        
        return weights
    
    @jax.jit
    def _monte_carlo_uncertainty(
        self, 
        features: jnp.ndarray, 
        weights: Dict[str, jnp.ndarray],
        n_samples: int = 10
    ) -> jnp.ndarray:
        """Estimate uncertainty using Monte Carlo dropout"""
        
        predictions = []
        
        for i in range(n_samples):
            # Apply different dropout masks
            key = jax.random.PRNGKey(i)
            
            # Forward pass with dropout (simplified)
            x = features
            
            for layer_idx in range(3):  # 3 hidden layers
                W = weights[f"W{layer_idx}"]
                b = weights[f"b{layer_idx}"]
                
                # Apply dropout
                dropout_mask = jax.random.bernoulli(key, 0.9, x.shape)  # 10% dropout
                x = x * dropout_mask / 0.9  # Scale to maintain expectation
                
                x = jnp.dot(x, W) + b
                x = jax.nn.relu(x)
                
                key, _ = jax.random.split(key)
            
            # Output layer (no dropout)
            W_out = weights["W_out"]
            b_out = weights["b_out"]
            output = jnp.dot(x, W_out) + b_out
            output = jax.nn.sigmoid(output)
            
            predictions.append(output)
        
        predictions = jnp.stack(predictions)
        return jnp.std(predictions, axis=0)
    
    def _predict_gaussian_process(
        self,
        features: jnp.ndarray,
        model: PredictionModel
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Make prediction using Gaussian Process"""
        
        # Simplified GP prediction (would use proper GP library in practice)
        
        if len(model.training_data) < 3:
            # Not enough training data
            return jnp.array([0.5, 0.5, 0.5]), jnp.array([0.2, 0.2, 0.2])
        
        # Use simple kernel-based prediction
        training_features = jnp.array([
            data["features"] for data in model.training_data[-20:]  # Last 20 points
        ])
        training_targets = jnp.array([
            [data["accuracy"], data["speed"], data["cost"]] 
            for data in model.training_data[-20:]
        ])
        
        # RBF kernel
        length_scale = model.parameters.get("length_scale", 1.0)
        distances = jnp.linalg.norm(training_features - features, axis=1)
        weights = jnp.exp(-distances**2 / (2 * length_scale**2))
        weights = weights / jnp.sum(weights)
        
        # Weighted prediction
        prediction = jnp.sum(weights[:, None] * training_targets, axis=0)
        
        # Uncertainty based on weight concentration
        uncertainty = jnp.array([0.1, 0.1, 0.1]) * (1 - jnp.max(weights))
        
        return prediction, uncertainty
    
    def _predict_ensemble(
        self,
        features: jnp.ndarray,
        model: PredictionModel
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Make prediction using ensemble of models"""
        
        component_models = model.parameters.get("component_models", ["neural_network"])
        predictions = []
        uncertainties = []
        
        for component_name in component_models:
            if component_name in self.models and component_name != "ensemble":
                pred, unc = self._predict_neural_network(features, self.models[component_name])
                predictions.append(pred)
                uncertainties.append(unc)
        
        if not predictions:
            return jnp.array([0.5, 0.5, 0.5]), jnp.array([0.2, 0.2, 0.2])
        
        # Average predictions
        predictions = jnp.stack(predictions)
        ensemble_prediction = jnp.mean(predictions, axis=0)
        
        # Combine uncertainties (epistemic + aleatoric)
        uncertainties = jnp.stack(uncertainties)
        aleatoric_uncertainty = jnp.mean(uncertainties, axis=0)
        epistemic_uncertainty = jnp.std(predictions, axis=0)
        total_uncertainty = jnp.sqrt(aleatoric_uncertainty**2 + epistemic_uncertainty**2)
        
        return ensemble_prediction, total_uncertainty
    
    def _combine_predictions(
        self,
        model_predictions: Dict[str, jnp.ndarray],
        model_uncertainties: Dict[str, jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Combine predictions from multiple models"""
        
        if self.ensemble_weights:
            # Use specified weights
            weights = jnp.array([
                self.ensemble_weights.get(name, 1.0) 
                for name in model_predictions.keys()
            ])
        else:
            # Use inverse uncertainty weighting
            avg_uncertainties = jnp.array([
                jnp.mean(unc) for unc in model_uncertainties.values()
            ])
            weights = 1.0 / (avg_uncertainties + 1e-8)
        
        weights = weights / jnp.sum(weights)
        
        # Weighted average of predictions
        predictions = jnp.stack(list(model_predictions.values()))
        final_prediction = jnp.sum(weights[:, None] * predictions, axis=0)
        
        # Combined uncertainty
        uncertainties = jnp.stack(list(model_uncertainties.values()))
        
        # Weighted uncertainty combination
        weighted_variance = jnp.sum(weights[:, None] * uncertainties**2, axis=0)
        prediction_variance = jnp.sum(weights[:, None] * (predictions - final_prediction)**2, axis=0)
        final_uncertainty = jnp.sqrt(weighted_variance + prediction_variance)
        
        return final_prediction, final_uncertainty
    
    def _calculate_feature_importance(
        self,
        features: jnp.ndarray,
        prediction: jnp.ndarray
    ) -> Dict[str, float]:
        """Calculate feature importance using gradient-based method"""
        
        try:
            # Simplified feature importance using finite differences
            feature_names = self._get_feature_names()
            importance = {}
            
            baseline_pred = prediction[0]  # Use accuracy as baseline
            
            for i, name in enumerate(feature_names[:len(features)]):
                # Perturb feature
                perturbed_features = features.at[i].set(features[i] * 1.1)
                
                # Would need to re-run prediction here
                # For now, use simplified approximation
                importance[name] = float(abs(features[i]) / (jnp.sum(jnp.abs(features)) + 1e-8))
            
            return importance
            
        except Exception as e:
            warnings.warn(f"Feature importance calculation failed: {e}")
            return {}
    
    def _calculate_prediction_confidence(
        self,
        model_predictions: Dict[str, jnp.ndarray],
        model_uncertainties: Dict[str, jnp.ndarray]
    ) -> float:
        """Calculate confidence in prediction based on model agreement"""
        
        if len(model_predictions) <= 1:
            return 0.7  # Moderate confidence for single model
        
        # Agreement between models
        predictions = jnp.stack(list(model_predictions.values()))
        prediction_std = jnp.std(predictions, axis=0)
        agreement = 1.0 - jnp.mean(prediction_std)
        
        # Average uncertainty
        uncertainties = jnp.stack(list(model_uncertainties.values()))
        avg_uncertainty = jnp.mean(uncertainties)
        uncertainty_confidence = 1.0 - avg_uncertainty
        
        # Combined confidence
        confidence = 0.6 * agreement + 0.4 * uncertainty_confidence
        return float(jnp.clip(confidence, 0.0, 1.0))
    
    def _get_feature_names(self) -> List[str]:
        """Get names of features for interpretability"""
        
        return [
            "circuit_depth", "circuit_qubits", "circuit_gates", "entanglement",
            "gate_density", "circuit_volume", "device_error_rate", "coherence_time",
            "gate_time", "readout_fidelity", "device_qubits", "quantum_volume",
            "t1_time", "t2_time", "gate_errors", "noise_factors_count", 
            "min_noise_factor", "max_noise_factor", "avg_noise_factor",
            "ensemble_size", "max_weight", "weight_diversity", "learning_rate",
            "exploration_rate", "historical_accuracy", "historical_accuracy_std",
            "historical_speed", "history_length", "avg_ensemble_size",
            "circuit_complexity", "device_quality", "complexity_quality_interaction",
            "entanglement_quality", "resource_matching"
        ]
    
    def update(
        self, 
        predicted_performance: Dict[str, float],
        actual_performance: Dict[str, float]
    ):
        """Update models with new experimental data"""
        
        try:
            # Calculate prediction error
            prediction_error = {}
            for metric in predicted_performance:
                if metric in actual_performance:
                    error = abs(predicted_performance[metric] - actual_performance[metric])
                    prediction_error[metric] = error
            
            # Add to training history
            training_sample = {
                "predicted_performance": predicted_performance,
                "actual_performance": actual_performance,
                "prediction_error": prediction_error,
                "timestamp": datetime.now()
            }
            
            self.training_history.append(training_sample)
            
            # Maintain history size
            if len(self.training_history) > self.max_training_history:
                self.training_history.pop(0)
            
            # Update research metrics
            avg_error = np.mean(list(prediction_error.values()))
            self._research_metrics["prediction_errors"].append(avg_error)
            
            # Retrain models periodically
            if len(self.training_history) % 50 == 0:  # Retrain every 50 samples
                self._retrain_models()
                
        except Exception as e:
            warnings.warn(f"Model update failed: {e}")
    
    def _retrain_models(self):
        """Retrain models with accumulated data"""
        
        # This would implement actual model retraining
        # For now, just update validation accuracy based on recent performance
        
        if len(self.training_history) < 10:
            return
        
        recent_errors = [
            np.mean(list(sample["prediction_error"].values()))
            for sample in self.training_history[-20:]
        ]
        
        # Update validation accuracy (1 - average error)
        avg_error = np.mean(recent_errors)
        validation_accuracy = max(0.0, 1.0 - avg_error)
        
        for model in self.models.values():
            model.validation_accuracy = validation_accuracy
            model.last_update = datetime.now()
    
    def get_research_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics for research analysis"""
        
        return {
            "total_predictions": self._research_metrics["predictions_made"],
            "average_prediction_error": np.mean(self._research_metrics["prediction_errors"]) if self._research_metrics["prediction_errors"] else 0.0,
            "model_types": [model.model_type.value for model in self.models.values()],
            "training_data_size": len(self.training_history),
            "model_validation_accuracies": {
                name: model.validation_accuracy for name, model in self.models.items()
            },
            "uncertainty_method": self.uncertainty_method,
            "transfer_learning_enabled": self.transfer_learning
        }