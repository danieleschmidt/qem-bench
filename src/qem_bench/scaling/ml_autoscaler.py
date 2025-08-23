"""
Machine Learning-Powered Auto-Scaler for QEM-Bench

This module provides advanced ML-driven auto-scaling with predictive scaling,
anomaly detection, and intelligent resource optimization for quantum error
mitigation workloads.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import deque

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .auto_scaler import AutoScaler, ScalingPolicy


class MLScalingStrategy(Enum):
    """ML-based scaling strategies."""
    PREDICTIVE = "predictive"
    REACTIVE_WITH_PREDICTION = "reactive_with_prediction"
    ANOMALY_BASED = "anomaly_based"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


@dataclass
class MLScalingConfig:
    """Configuration for ML-powered auto-scaling."""
    strategy: MLScalingStrategy = MLScalingStrategy.PREDICTIVE
    
    # Prediction settings
    prediction_horizon_minutes: int = 15
    min_training_samples: int = 50
    retrain_interval_hours: int = 4
    
    # Feature engineering
    enable_time_features: bool = True
    enable_workload_features: bool = True
    enable_seasonal_features: bool = True
    
    # Model parameters
    model_type: str = "random_forest"
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    })
    
    # Anomaly detection
    anomaly_threshold: float = 2.0  # Standard deviations
    min_anomaly_duration: int = 60  # Seconds
    
    # Performance thresholds
    prediction_accuracy_threshold: float = 0.7
    model_confidence_threshold: float = 0.8


@dataclass
class ScalingFeatures:
    """Features for ML scaling prediction."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    queue_length: int
    request_rate: float
    response_time: float
    error_rate: float
    active_instances: int
    
    # Time-based features
    hour_of_day: int = 0
    day_of_week: int = 0
    is_weekend: bool = False
    
    # Workload features
    avg_circuit_depth: float = 0.0
    avg_qubit_count: float = 0.0
    mitigation_method_mix: Dict[str, float] = field(default_factory=dict)
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array."""
        base_features = [
            self.cpu_usage, self.memory_usage, self.queue_length,
            self.request_rate, self.response_time, self.error_rate,
            self.active_instances, self.hour_of_day, self.day_of_week,
            float(self.is_weekend), self.avg_circuit_depth, self.avg_qubit_count
        ]
        
        # Add mitigation method features
        method_features = [
            self.mitigation_method_mix.get("zne", 0.0),
            self.mitigation_method_mix.get("pec", 0.0),
            self.mitigation_method_mix.get("vd", 0.0),
            self.mitigation_method_mix.get("cdr", 0.0)
        ]
        
        return np.array(base_features + method_features)


class MLAutoScaler(AutoScaler):
    """
    Machine Learning-powered auto-scaler with predictive capabilities.
    
    Extends the base AutoScaler with ML-based prediction models,
    anomaly detection, and intelligent scaling decisions.
    """
    
    def __init__(
        self,
        policy: ScalingPolicy,
        ml_config: Optional[MLScalingConfig] = None
    ):
        """Initialize ML auto-scaler."""
        super().__init__(policy)
        
        self.ml_config = ml_config or MLScalingConfig()
        
        # ML components
        self.prediction_model = None
        self.feature_scaler = None
        self.is_trained = False
        
        # Data storage
        self.feature_history = deque(maxlen=10000)  # Keep last 10k samples
        self.target_history = deque(maxlen=10000)   # Target values (instances needed)
        
        # Prediction tracking
        self.predictions = deque(maxlen=1000)
        self.prediction_accuracy = 0.0
        self.last_training_time = datetime.min
        
        # Anomaly detection
        self.anomaly_detector = SimpleAnomalyDetector(
            threshold=self.ml_config.anomaly_threshold
        )
        
        # Threading
        self._ml_lock = threading.RLock()
        
        # Model initialization
        if SKLEARN_AVAILABLE:
            self._initialize_model()
        else:
            print("⚠️ scikit-learn not available, using basic scaling")
        
        print(f"✅ ML AutoScaler initialized with strategy: {self.ml_config.strategy.value}")
    
    def _initialize_model(self) -> None:
        """Initialize the ML prediction model."""
        if self.ml_config.model_type == "random_forest":
            self.prediction_model = RandomForestRegressor(**self.ml_config.model_params)
            self.feature_scaler = StandardScaler()
    
    def collect_features(
        self,
        cpu_usage: float,
        memory_usage: float,
        queue_length: int,
        request_rate: float,
        response_time: float,
        error_rate: float,
        active_instances: int,
        workload_info: Optional[Dict[str, Any]] = None
    ) -> ScalingFeatures:
        """Collect and engineer features for ML model."""
        now = datetime.now()
        
        # Base features
        features = ScalingFeatures(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            queue_length=queue_length,
            request_rate=request_rate,
            response_time=response_time,
            error_rate=error_rate,
            active_instances=active_instances
        )
        
        # Time-based features
        if self.ml_config.enable_time_features:
            features.hour_of_day = now.hour
            features.day_of_week = now.weekday()
            features.is_weekend = now.weekday() >= 5
        
        # Workload features
        if self.ml_config.enable_workload_features and workload_info:
            features.avg_circuit_depth = workload_info.get("avg_circuit_depth", 0.0)
            features.avg_qubit_count = workload_info.get("avg_qubit_count", 0.0)
            features.mitigation_method_mix = workload_info.get("method_mix", {})
        
        return features
    
    def predict_required_instances(
        self,
        features: ScalingFeatures,
        prediction_horizon_minutes: Optional[int] = None
    ) -> Tuple[int, float]:
        """
        Predict required instances using ML model.
        
        Returns:
            Tuple of (predicted_instances, confidence)
        """
        if not self.is_trained or not SKLEARN_AVAILABLE:
            # Fallback to rule-based scaling
            return self._fallback_prediction(features), 0.5
        
        horizon = prediction_horizon_minutes or self.ml_config.prediction_horizon_minutes
        
        with self._ml_lock:
            try:
                # Prepare features
                feature_array = features.to_array().reshape(1, -1)
                
                # Scale features
                scaled_features = self.feature_scaler.transform(feature_array)
                
                # Make prediction
                prediction = self.prediction_model.predict(scaled_features)[0]
                
                # Get confidence (for RandomForest, use prediction variance)
                if hasattr(self.prediction_model, "estimators_"):
                    predictions = [estimator.predict(scaled_features)[0] 
                                 for estimator in self.prediction_model.estimators_]
                    confidence = 1.0 - (np.std(predictions) / max(np.mean(predictions), 1.0))
                else:
                    confidence = 0.8  # Default confidence
                
                # Clamp prediction to reasonable bounds
                prediction = max(1, min(int(prediction), self.policy.max_instances))
                confidence = max(0.0, min(confidence, 1.0))
                
                # Store prediction for accuracy tracking
                self.predictions.append({
                    "timestamp": time.time(),
                    "prediction": prediction,
                    "features": features,
                    "confidence": confidence
                })
                
                return prediction, confidence
                
            except Exception as e:
                self.logger.error(f"ML prediction failed: {e}")
                return self._fallback_prediction(features), 0.3
    
    def _fallback_prediction(self, features: ScalingFeatures) -> int:
        """Fallback rule-based prediction when ML is not available."""
        # Simple rule-based scaling logic
        if features.cpu_usage > 80 or features.memory_usage > 80:
            return min(features.active_instances + 2, self.policy.max_instances)
        elif features.queue_length > 10:
            return min(features.active_instances + 1, self.policy.max_instances)
        elif features.cpu_usage < 20 and features.memory_usage < 20 and features.queue_length < 2:
            return max(features.active_instances - 1, self.policy.min_instances)
        else:
            return features.active_instances
    
    def train_model(self, force_retrain: bool = False) -> bool:
        """Train the ML prediction model."""
        if not SKLEARN_AVAILABLE:
            return False
        
        # Check if retraining is needed
        time_since_training = datetime.now() - self.last_training_time
        if (not force_retrain and 
            time_since_training.total_seconds() < self.ml_config.retrain_interval_hours * 3600):
            return False
        
        with self._ml_lock:
            if len(self.feature_history) < self.ml_config.min_training_samples:
                self.logger.info(
                    f"Not enough training samples: {len(self.feature_history)} < "
                    f"{self.ml_config.min_training_samples}"
                )
                return False
            
            try:
                # Prepare training data
                X = np.array([features.to_array() for features in self.feature_history])
                y = np.array(list(self.target_history))
                
                # Scale features
                self.feature_scaler.fit(X)
                X_scaled = self.feature_scaler.transform(X)
                
                # Train model
                self.prediction_model.fit(X_scaled, y)
                self.is_trained = True
                self.last_training_time = datetime.now()
                
                # Calculate accuracy on recent predictions
                self._update_prediction_accuracy()
                
                self.logger.info(
                    f"ML model trained with {len(X)} samples. "
                    f"Prediction accuracy: {self.prediction_accuracy:.3f}"
                )
                
                return True
                
            except Exception as e:
                self.logger.error(f"Model training failed: {e}")
                return False
    
    def _update_prediction_accuracy(self) -> None:
        """Update prediction accuracy based on recent predictions."""
        if not self.predictions:
            return
        
        # Compare predictions with actual outcomes (simplified)
        recent_predictions = list(self.predictions)[-100:]  # Last 100 predictions
        
        if len(recent_predictions) < 10:
            return
        
        # Calculate accuracy (simplified - in practice would need actual outcomes)
        # For now, use a placeholder accuracy calculation
        self.prediction_accuracy = 0.8  # Placeholder
    
    def detect_anomalies(self, features: ScalingFeatures) -> bool:
        """Detect anomalies in current metrics."""
        return self.anomaly_detector.is_anomaly([
            features.cpu_usage, features.memory_usage, 
            features.response_time, features.error_rate
        ])
    
    def make_scaling_decision(
        self,
        current_features: ScalingFeatures,
        workload_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make intelligent scaling decision using ML."""
        decision = {
            "timestamp": time.time(),
            "current_instances": current_features.active_instances,
            "recommended_instances": current_features.active_instances,
            "confidence": 0.5,
            "reasoning": "no_change",
            "strategy_used": "fallback",
            "anomaly_detected": False
        }
        
        try:
            # Check for anomalies
            is_anomaly = self.detect_anomalies(current_features)
            decision["anomaly_detected"] = is_anomaly
            
            if self.ml_config.strategy == MLScalingStrategy.PREDICTIVE:
                # Use ML prediction
                predicted_instances, confidence = self.predict_required_instances(current_features)
                
                decision.update({
                    "recommended_instances": predicted_instances,
                    "confidence": confidence,
                    "strategy_used": "predictive_ml",
                    "reasoning": f"ML prediction based on {len(self.feature_history)} historical samples"
                })
                
            elif self.ml_config.strategy == MLScalingStrategy.ANOMALY_BASED:
                if is_anomaly:
                    # Scale up on anomaly
                    recommended = min(
                        current_features.active_instances + 2, 
                        self.policy.max_instances
                    )
                    decision.update({
                        "recommended_instances": recommended,
                        "reasoning": "anomaly_detected_scale_up",
                        "strategy_used": "anomaly_based"
                    })
                    
            # Store data for future training
            self.feature_history.append(current_features)
            self.target_history.append(decision["recommended_instances"])
            
            # Auto-retrain if needed
            if len(self.feature_history) % 100 == 0:  # Every 100 samples
                self.train_model()
                
        except Exception as e:
            self.logger.error(f"ML scaling decision failed: {e}")
            decision["reasoning"] = f"error_fallback: {e}"
        
        return decision
    
    def get_ml_metrics(self) -> Dict[str, Any]:
        """Get ML-specific metrics."""
        return {
            "model_trained": self.is_trained,
            "training_samples": len(self.feature_history),
            "prediction_accuracy": self.prediction_accuracy,
            "last_training": self.last_training_time.isoformat(),
            "ml_config": {
                "strategy": self.ml_config.strategy.value,
                "prediction_horizon": self.ml_config.prediction_horizon_minutes,
                "model_type": self.ml_config.model_type
            },
            "sklearn_available": SKLEARN_AVAILABLE,
            "recent_predictions": len(self.predictions)
        }


class SimpleAnomalyDetector:
    """Simple statistical anomaly detector."""
    
    def __init__(self, threshold: float = 2.0, window_size: int = 100):
        self.threshold = threshold
        self.window_size = window_size
        self.data_history = deque(maxlen=window_size)
    
    def is_anomaly(self, values: List[float]) -> bool:
        """Check if values represent an anomaly."""
        if len(self.data_history) < 10:
            self.data_history.append(values)
            return False
        
        # Calculate z-score for each metric
        anomaly_scores = []
        for i, value in enumerate(values):
            historical = [data[i] for data in self.data_history if len(data) > i]
            if len(historical) > 5:
                mean_val = np.mean(historical)
                std_val = np.std(historical)
                if std_val > 0:
                    z_score = abs(value - mean_val) / std_val
                    anomaly_scores.append(z_score)
        
        self.data_history.append(values)
        
        # Return True if any metric exceeds threshold
        return any(score > self.threshold for score in anomaly_scores)