"""
Workload analysis and prediction for intelligent auto-scaling decisions.

This module provides sophisticated workload analysis capabilities to predict
resource needs and optimize scaling decisions based on historical patterns,
current trends, and quantum computation characteristics.
"""

import time
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import asyncio
from datetime import datetime, timedelta

from ..security import SecureConfig
from ..metrics import MetricsCollector


logger = logging.getLogger(__name__)


class WorkloadPattern(Enum):
    """Types of workload patterns."""
    CONSTANT = "constant"
    PERIODIC = "periodic"
    BURSTY = "bursty"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANDOM = "random"


class ResourceType(Enum):
    """Types of resources to analyze."""
    CPU = "cpu"
    MEMORY = "memory"
    QUANTUM_BACKENDS = "quantum_backends"
    NETWORK_BANDWIDTH = "network_bandwidth"
    STORAGE = "storage"


@dataclass
class WorkloadMetrics:
    """Comprehensive workload metrics for analysis."""
    timestamp: float = field(default_factory=time.time)
    
    # Computation metrics
    jobs_submitted: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    jobs_queued: int = 0
    
    # Resource utilization
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    network_utilization: float = 0.0
    storage_utilization: float = 0.0
    
    # Quantum-specific metrics
    circuits_executed: int = 0
    shots_executed: int = 0
    backend_utilization: Dict[str, float] = field(default_factory=dict)
    mitigation_overhead: float = 0.0
    
    # Performance metrics
    average_job_duration: float = 0.0
    queue_wait_time: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    
    # User patterns
    active_users: int = 0
    user_session_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp,
            "jobs_submitted": self.jobs_submitted,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "jobs_queued": self.jobs_queued,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "network_utilization": self.network_utilization,
            "storage_utilization": self.storage_utilization,
            "circuits_executed": self.circuits_executed,
            "shots_executed": self.shots_executed,
            "backend_utilization": self.backend_utilization,
            "mitigation_overhead": self.mitigation_overhead,
            "average_job_duration": self.average_job_duration,
            "queue_wait_time": self.queue_wait_time,
            "error_rate": self.error_rate,
            "throughput": self.throughput,
            "active_users": self.active_users,
            "user_session_duration": self.user_session_duration
        }


@dataclass
class WorkloadPrediction:
    """Prediction of future workload characteristics."""
    timestamp: float
    prediction_horizon: float  # seconds into the future
    
    # Predicted metrics
    predicted_cpu: float
    predicted_memory: float
    predicted_jobs: int
    predicted_throughput: float
    
    # Confidence metrics
    confidence: float  # 0.0 to 1.0
    prediction_error: float  # historical error rate
    
    # Pattern information  
    detected_pattern: WorkloadPattern
    pattern_strength: float  # how strong the pattern is
    
    # Resource recommendations
    recommended_instances: int
    recommended_backends: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to dictionary."""
        return {
            "timestamp": self.timestamp,
            "prediction_horizon": self.prediction_horizon,
            "predicted_cpu": self.predicted_cpu,
            "predicted_memory": self.predicted_memory,
            "predicted_jobs": self.predicted_jobs,
            "predicted_throughput": self.predicted_throughput,
            "confidence": self.confidence,
            "prediction_error": self.prediction_error,
            "detected_pattern": self.detected_pattern.value,
            "pattern_strength": self.pattern_strength,
            "recommended_instances": self.recommended_instances,
            "recommended_backends": self.recommended_backends
        }


class PatternDetector:
    """Detects patterns in workload time series data."""
    
    def __init__(self, min_data_points: int = 10):
        self.min_data_points = min_data_points
    
    def detect_pattern(
        self, 
        timestamps: np.ndarray, 
        values: np.ndarray
    ) -> Tuple[WorkloadPattern, float]:
        """
        Detect the dominant pattern in time series data.
        
        Returns:
            Tuple of (pattern, strength) where strength is 0.0-1.0
        """
        if len(values) < self.min_data_points:
            return WorkloadPattern.RANDOM, 0.0
        
        # Normalize timestamps to hours for analysis
        time_hours = (timestamps - timestamps[0]) / 3600.0
        
        # Test for different patterns
        patterns_scores = {}
        
        # 1. Test for constant pattern (low variance)
        variance = np.var(values)
        mean_value = np.mean(values)
        cv = variance / (mean_value + 1e-8)  # coefficient of variation
        if cv < 0.1:
            patterns_scores[WorkloadPattern.CONSTANT] = 1.0 - cv
        
        # 2. Test for trending patterns (linear regression)
        if len(values) >= 3:
            slope, _, r_value, _, _ = self._linear_regression(time_hours, values)
            r_squared = r_value ** 2
            
            if r_squared > 0.5:  # Strong linear relationship
                if slope > 0:
                    patterns_scores[WorkloadPattern.TRENDING_UP] = r_squared
                else:
                    patterns_scores[WorkloadPattern.TRENDING_DOWN] = r_squared
        
        # 3. Test for periodic patterns (FFT analysis)
        if len(values) >= 12:  # Need enough data for periodicity
            periodic_strength = self._detect_periodicity(time_hours, values)
            if periodic_strength > 0.3:
                patterns_scores[WorkloadPattern.PERIODIC] = periodic_strength
        
        # 4. Test for bursty patterns (high variance with spikes)
        if cv > 0.5:
            # Look for spike patterns
            spike_strength = self._detect_spikes(values)
            if spike_strength > 0.4:
                patterns_scores[WorkloadPattern.BURSTY] = spike_strength
        
        # Return the pattern with highest score
        if patterns_scores:
            best_pattern = max(patterns_scores.keys(), key=lambda k: patterns_scores[k])
            return best_pattern, patterns_scores[best_pattern]
        else:
            return WorkloadPattern.RANDOM, 0.1
    
    def _linear_regression(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float, float]:
        """Simple linear regression implementation."""
        from scipy import stats
        return stats.linregress(x, y)
    
    def _detect_periodicity(self, time_hours: np.ndarray, values: np.ndarray) -> float:
        """Detect periodic patterns using FFT."""
        try:
            # Remove trend first
            detrended = self._detrend(values)
            
            # Compute FFT
            fft = np.fft.fft(detrended)
            freqs = np.fft.fftfreq(len(detrended), d=np.mean(np.diff(time_hours)))
            
            # Find dominant frequency (excluding DC component)
            power_spectrum = np.abs(fft[1:len(fft)//2])
            if len(power_spectrum) == 0:
                return 0.0
            
            max_power = np.max(power_spectrum)
            total_power = np.sum(power_spectrum)
            
            # Periodicity strength is the ratio of max peak to total power
            if total_power > 0:
                return min(max_power / total_power * 2, 1.0)
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Periodicity detection failed: {e}")
            return 0.0
    
    def _detrend(self, values: np.ndarray) -> np.ndarray:
        """Remove linear trend from time series."""
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        trend = slope * x + intercept
        return values - trend
    
    def _detect_spikes(self, values: np.ndarray) -> float:
        """Detect spike patterns in data."""
        if len(values) < 3:
            return 0.0
        
        # Calculate z-scores
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return 0.0
        
        z_scores = np.abs((values - mean_val) / std_val)
        
        # Count spikes (z-score > 2)
        spike_count = np.sum(z_scores > 2)
        spike_ratio = spike_count / len(values)
        
        # Spike strength based on ratio and intensity
        max_z = np.max(z_scores)
        return min(spike_ratio * 2 + max_z / 10, 1.0)


class WorkloadPredictor:
    """Predicts future workload based on historical patterns."""
    
    def __init__(self, pattern_detector: Optional[PatternDetector] = None):
        self.pattern_detector = pattern_detector or PatternDetector()
        self.prediction_models = {}
        self.model_accuracy = {}
    
    def predict_workload(
        self,
        historical_metrics: List[WorkloadMetrics],
        prediction_horizon: float = 900.0  # 15 minutes
    ) -> WorkloadPrediction:
        """
        Predict future workload characteristics.
        
        Args:
            historical_metrics: List of historical workload metrics
            prediction_horizon: Time horizon for prediction (seconds)
            
        Returns:
            WorkloadPrediction with forecasted values
        """
        if len(historical_metrics) < 3:
            # Not enough data for prediction
            return self._default_prediction(prediction_horizon)
        
        # Extract time series data
        timestamps = np.array([m.timestamp for m in historical_metrics])
        cpu_values = np.array([m.cpu_utilization for m in historical_metrics])
        memory_values = np.array([m.memory_utilization for m in historical_metrics])
        job_counts = np.array([m.jobs_submitted for m in historical_metrics])
        throughput_values = np.array([m.throughput for m in historical_metrics])
        
        # Detect pattern in CPU utilization (primary metric)
        pattern, pattern_strength = self.pattern_detector.detect_pattern(
            timestamps, cpu_values
        )
        
        # Make predictions based on detected pattern
        prediction_time = time.time() + prediction_horizon
        
        predicted_cpu = self._predict_metric(
            timestamps, cpu_values, prediction_time, pattern
        )
        predicted_memory = self._predict_metric(
            timestamps, memory_values, prediction_time, pattern
        )
        predicted_jobs = int(self._predict_metric(
            timestamps, job_counts.astype(float), prediction_time, pattern
        ))
        predicted_throughput = self._predict_metric(
            timestamps, throughput_values, prediction_time, pattern
        )
        
        # Calculate confidence based on pattern strength and historical accuracy
        confidence = self._calculate_confidence(pattern, pattern_strength, cpu_values)
        
        # Calculate prediction error from historical data
        prediction_error = self._calculate_historical_error(
            timestamps, cpu_values, pattern
        )
        
        # Generate resource recommendations
        recommended_instances = self._recommend_instances(
            predicted_cpu, predicted_memory, predicted_jobs
        )
        recommended_backends = self._recommend_backends(
            predicted_jobs, predicted_throughput
        )
        
        return WorkloadPrediction(
            timestamp=time.time(),
            prediction_horizon=prediction_horizon,
            predicted_cpu=predicted_cpu,
            predicted_memory=predicted_memory,
            predicted_jobs=predicted_jobs,
            predicted_throughput=predicted_throughput,
            confidence=confidence,
            prediction_error=prediction_error,
            detected_pattern=pattern,
            pattern_strength=pattern_strength,
            recommended_instances=recommended_instances,
            recommended_backends=recommended_backends
        )
    
    def _predict_metric(
        self,
        timestamps: np.ndarray,
        values: np.ndarray,
        prediction_time: float,
        pattern: WorkloadPattern
    ) -> float:
        """Predict a single metric value based on pattern."""
        if len(values) == 0:
            return 0.0
        
        current_time = timestamps[-1]
        time_delta = prediction_time - current_time
        
        if pattern == WorkloadPattern.CONSTANT:
            return float(np.mean(values))
        
        elif pattern == WorkloadPattern.TRENDING_UP:
            # Linear extrapolation
            time_hours = (timestamps - timestamps[0]) / 3600.0
            slope, intercept, _, _, _ = self.pattern_detector._linear_regression(
                time_hours, values
            )
            future_time_hours = (prediction_time - timestamps[0]) / 3600.0
            return max(0.0, slope * future_time_hours + intercept)
        
        elif pattern == WorkloadPattern.TRENDING_DOWN:
            # Linear extrapolation with floor
            time_hours = (timestamps - timestamps[0]) / 3600.0
            slope, intercept, _, _, _ = self.pattern_detector._linear_regression(
                time_hours, values
            )
            future_time_hours = (prediction_time - timestamps[0]) / 3600.0
            return max(0.0, slope * future_time_hours + intercept)
        
        elif pattern == WorkloadPattern.PERIODIC:
            # Use recent values with periodic assumption
            # Simple approach: assume daily periodicity
            seconds_in_day = 24 * 3600
            phase_in_day = (prediction_time % seconds_in_day) / seconds_in_day
            
            # Find similar phase in historical data
            historical_phases = [(t % seconds_in_day) / seconds_in_day for t in timestamps]
            closest_idx = np.argmin([abs(p - phase_in_day) for p in historical_phases])
            
            return float(values[closest_idx])
        
        elif pattern == WorkloadPattern.BURSTY:
            # For bursty patterns, use recent average with some increase
            recent_values = values[-min(5, len(values)):]
            base_value = np.mean(recent_values)
            
            # Add some randomness for burst possibility
            return base_value * (1.0 + np.random.normal(0, 0.1))
        
        else:  # RANDOM
            # Use recent average
            recent_values = values[-min(3, len(values)):]
            return float(np.mean(recent_values))
    
    def _calculate_confidence(
        self,
        pattern: WorkloadPattern,
        pattern_strength: float,
        historical_values: np.ndarray
    ) -> float:
        """Calculate confidence in prediction."""
        # Base confidence on pattern strength
        base_confidence = pattern_strength
        
        # Adjust based on pattern type
        pattern_confidence_multipliers = {
            WorkloadPattern.CONSTANT: 1.0,
            WorkloadPattern.TRENDING_UP: 0.9,
            WorkloadPattern.TRENDING_DOWN: 0.9,
            WorkloadPattern.PERIODIC: 0.8,
            WorkloadPattern.BURSTY: 0.6,
            WorkloadPattern.RANDOM: 0.4
        }
        
        base_confidence *= pattern_confidence_multipliers.get(pattern, 0.5)
        
        # Adjust based on data quality
        if len(historical_values) > 20:
            base_confidence *= 1.1  # More data = higher confidence
        elif len(historical_values) < 5:
            base_confidence *= 0.7  # Less data = lower confidence
        
        return min(max(base_confidence, 0.1), 1.0)
    
    def _calculate_historical_error(
        self,
        timestamps: np.ndarray,
        values: np.ndarray,
        pattern: WorkloadPattern
    ) -> float:
        """Calculate historical prediction error."""
        if len(values) < 4:
            return 0.5  # Default error rate
        
        # Use last 50% of data to test predictions
        split_idx = len(values) // 2
        train_timestamps = timestamps[:split_idx]
        train_values = values[:split_idx]
        test_timestamps = timestamps[split_idx:]
        test_values = values[split_idx:]
        
        # Make predictions for test period
        errors = []
        for i, test_time in enumerate(test_timestamps):
            predicted = self._predict_metric(
                train_timestamps, train_values, test_time, pattern
            )
            actual = test_values[i]
            
            # Calculate relative error
            if actual != 0:
                error = abs(predicted - actual) / abs(actual)
            else:
                error = abs(predicted)
            
            errors.append(error)
        
        return float(np.mean(errors)) if errors else 0.5
    
    def _recommend_instances(
        self,
        predicted_cpu: float,
        predicted_memory: float,
        predicted_jobs: int
    ) -> int:
        """Recommend number of instances based on predictions."""
        # Simple algorithm: scale based on resource utilization
        cpu_instances_needed = max(1, int(np.ceil(predicted_cpu / 70.0)))  # 70% target
        memory_instances_needed = max(1, int(np.ceil(predicted_memory / 70.0)))
        job_instances_needed = max(1, int(np.ceil(predicted_jobs / 10)))  # 10 jobs per instance
        
        return max(cpu_instances_needed, memory_instances_needed, job_instances_needed)
    
    def _recommend_backends(
        self,
        predicted_jobs: int,
        predicted_throughput: float
    ) -> List[str]:
        """Recommend quantum backends based on predicted workload."""
        backends = []
        
        if predicted_jobs > 20:
            backends.append("high_capacity_simulator")
        
        if predicted_throughput > 100:
            backends.extend(["simulator_1", "simulator_2"])
        
        if predicted_jobs > 50:
            backends.append("hardware_backend_1")
        
        return backends or ["default_simulator"]
    
    def _default_prediction(self, prediction_horizon: float) -> WorkloadPrediction:
        """Return default prediction when insufficient data."""
        return WorkloadPrediction(
            timestamp=time.time(),
            prediction_horizon=prediction_horizon,
            predicted_cpu=30.0,  # Conservative estimate
            predicted_memory=40.0,
            predicted_jobs=5,
            predicted_throughput=10.0,
            confidence=0.3,  # Low confidence
            prediction_error=0.7,
            detected_pattern=WorkloadPattern.RANDOM,
            pattern_strength=0.1,
            recommended_instances=2,
            recommended_backends=["default_simulator"]
        )


class WorkloadAnalyzer:
    """
    Comprehensive workload analysis system for auto-scaling decisions.
    
    Features:
    - Real-time workload monitoring and pattern detection
    - Predictive analytics for resource planning
    - Quantum-specific workload characteristics analysis
    - Historical trend analysis and anomaly detection
    - Resource optimization recommendations
    
    Example:
        >>> analyzer = WorkloadAnalyzer()
        >>> await analyzer.start_monitoring()
        >>> prediction = await analyzer.get_workload_prediction(horizon=900)
        >>> print(f"Predicted CPU: {prediction.predicted_cpu:.1f}%")
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        pattern_detector: Optional[PatternDetector] = None,
        predictor: Optional[WorkloadPredictor] = None,
        config: Optional[SecureConfig] = None
    ):
        self.metrics_collector = metrics_collector
        self.pattern_detector = pattern_detector or PatternDetector()
        self.predictor = predictor or WorkloadPredictor(self.pattern_detector)
        self.config = config or SecureConfig()
        
        # Data storage
        self.workload_history: List[WorkloadMetrics] = []
        self.pattern_history: List[Tuple[WorkloadPattern, float, float]] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.collection_interval = 60.0  # 1 minute
        
        logger.info("WorkloadAnalyzer initialized")
    
    async def start_monitoring(self, collection_interval: float = 60.0) -> None:
        """Start continuous workload monitoring."""
        if self.is_monitoring:
            logger.warning("WorkloadAnalyzer is already monitoring")
            return
        
        self.is_monitoring = True
        self.collection_interval = collection_interval
        
        logger.info(f"Starting workload monitoring with {collection_interval}s interval")
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop workload monitoring."""
        self.is_monitoring = False
        logger.info("WorkloadAnalyzer monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect current workload metrics
                metrics = await self._collect_workload_metrics()
                
                # Store metrics
                self.workload_history.append(metrics)
                
                # Keep only recent history (last 24 hours)
                cutoff_time = time.time() - 24 * 3600
                self.workload_history = [
                    m for m in self.workload_history 
                    if m.timestamp > cutoff_time
                ]
                
                # Detect patterns if we have enough data
                if len(self.workload_history) >= 10:
                    await self._update_pattern_analysis()
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in workload monitoring loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_workload_metrics(self) -> WorkloadMetrics:
        """Collect comprehensive workload metrics."""
        # This would integrate with actual system metrics
        # For now, return simulated metrics
        
        metrics = WorkloadMetrics()
        
        if self.metrics_collector:
            try:
                system_metrics = await self.metrics_collector.collect_metrics()
                metrics.cpu_utilization = system_metrics.get("cpu_utilization", 0.0)
                metrics.memory_utilization = system_metrics.get("memory_utilization", 0.0)
                metrics.jobs_queued = system_metrics.get("queue_length", 0)
                metrics.throughput = system_metrics.get("throughput", 0.0)
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    async def _update_pattern_analysis(self) -> None:
        """Update pattern analysis with latest data."""
        try:
            # Analyze CPU utilization pattern
            timestamps = np.array([m.timestamp for m in self.workload_history])
            cpu_values = np.array([m.cpu_utilization for m in self.workload_history])
            
            pattern, strength = self.pattern_detector.detect_pattern(
                timestamps, cpu_values
            )
            
            # Store pattern information
            self.pattern_history.append((pattern, strength, time.time()))
            
            # Keep only recent pattern history
            cutoff_time = time.time() - 6 * 3600  # 6 hours
            self.pattern_history = [
                p for p in self.pattern_history if p[2] > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error updating pattern analysis: {e}")
    
    async def get_workload_prediction(
        self,
        horizon: float = 900.0
    ) -> WorkloadPrediction:
        """Get workload prediction for specified time horizon."""
        return self.predictor.predict_workload(
            self.workload_history, 
            prediction_horizon=horizon
        )
    
    def get_current_workload(self) -> Optional[WorkloadMetrics]:
        """Get the most recent workload metrics."""
        return self.workload_history[-1] if self.workload_history else None
    
    def get_workload_history(
        self, 
        duration_seconds: float = 3600.0
    ) -> List[WorkloadMetrics]:
        """Get workload history for specified duration."""
        cutoff_time = time.time() - duration_seconds
        return [m for m in self.workload_history if m.timestamp > cutoff_time]
    
    def get_dominant_pattern(self) -> Optional[Tuple[WorkloadPattern, float]]:
        """Get the most recent dominant workload pattern."""
        if not self.pattern_history:
            return None
        
        # Return most recent pattern
        pattern, strength, _ = self.pattern_history[-1]
        return pattern, strength
    
    def analyze_workload_trends(
        self, 
        duration_seconds: float = 3600.0
    ) -> Dict[str, Any]:
        """Analyze workload trends over specified duration."""
        recent_metrics = self.get_workload_history(duration_seconds)
        
        if len(recent_metrics) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Calculate trends for key metrics
        timestamps = np.array([m.timestamp for m in recent_metrics])
        time_hours = (timestamps - timestamps[0]) / 3600.0
        
        trends = {}
        
        for metric_name in ["cpu_utilization", "memory_utilization", "throughput"]:
            values = np.array([getattr(m, metric_name) for m in recent_metrics])
            
            if len(values) >= 3:
                try:
                    slope, _, r_value, _, _ = self.pattern_detector._linear_regression(
                        time_hours, values
                    )
                    trends[metric_name] = {
                        "slope": float(slope),
                        "r_squared": float(r_value ** 2),
                        "direction": "increasing" if slope > 0 else "decreasing",
                        "strength": abs(float(r_value))
                    }
                except Exception as e:
                    logger.debug(f"Trend analysis failed for {metric_name}: {e}")
                    trends[metric_name] = {"error": str(e)}
        
        return {
            "analysis_period": duration_seconds,
            "data_points": len(recent_metrics),
            "trends": trends,
            "dominant_pattern": self.get_dominant_pattern()
        }
    
    def detect_anomalies(
        self, 
        threshold_std: float = 2.0
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in recent workload data."""
        if len(self.workload_history) < 10:
            return []
        
        anomalies = []
        
        # Analyze recent data
        recent_metrics = self.workload_history[-20:]  # Last 20 data points
        
        for metric_name in ["cpu_utilization", "memory_utilization", "throughput"]:
            values = np.array([getattr(m, metric_name) for m in recent_metrics])
            
            if len(values) >= 5:
                mean_val = np.mean(values[:-1])  # Exclude latest value
                std_val = np.std(values[:-1])
                
                if std_val > 0:
                    latest_value = values[-1]
                    z_score = abs(latest_value - mean_val) / std_val
                    
                    if z_score > threshold_std:
                        anomalies.append({
                            "metric": metric_name,
                            "value": float(latest_value),
                            "expected_range": [
                                float(mean_val - threshold_std * std_val),
                                float(mean_val + threshold_std * std_val)
                            ],
                            "z_score": float(z_score),
                            "severity": "high" if z_score > 3.0 else "medium",
                            "timestamp": recent_metrics[-1].timestamp
                        })
        
        return anomalies