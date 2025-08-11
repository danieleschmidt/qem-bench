"""
Real-Time Adaptive Quantum Error Mitigation

Advanced adaptive error mitigation that responds to real-time device conditions,
drift patterns, and environmental factors to optimize mitigation strategies
dynamically during quantum computation execution.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import threading
import queue
import time
from collections import deque
import logging
from concurrent.futures import ThreadPoolExecutor

from ..mitigation.zne import ZeroNoiseExtrapolation
from ..monitoring import PerformanceMonitor
from ..jax.circuits import JAXCircuit

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveQEMConfig:
    """Configuration for real-time adaptive QEM."""
    # Real-time parameters
    monitoring_interval: float = 1.0  # seconds
    adaptation_frequency: int = 10  # adaptations per monitoring cycle
    max_adaptation_delay: float = 0.1  # maximum delay for adaptation (seconds)
    
    # Prediction parameters
    prediction_window: int = 50  # number of historical points for prediction
    prediction_horizon: int = 10  # steps ahead to predict
    confidence_threshold: float = 0.8  # minimum confidence for predictions
    
    # Adaptation parameters
    adaptation_aggressiveness: float = 0.5  # 0=conservative, 1=aggressive
    parameter_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'noise_factor_max': (1.5, 5.0),
        'extrapolation_order': (1, 4),
        'bootstrap_samples': (50, 500)
    })
    
    # Device monitoring
    monitor_device_drift: bool = True
    monitor_environmental_factors: bool = True
    monitor_queue_dynamics: bool = True
    monitor_crosstalk_patterns: bool = True
    
    # Learning parameters
    enable_online_learning: bool = True
    learning_rate: float = 0.01
    memory_decay_factor: float = 0.95
    experience_buffer_size: int = 1000


@dataclass
class DeviceState:
    """Real-time device state information."""
    timestamp: float
    gate_fidelities: Dict[str, float]
    readout_fidelities: Dict[int, float]
    coherence_times: Dict[str, float]
    temperature: float
    calibration_time: float
    queue_length: int
    error_rates: Dict[str, float]
    crosstalk_matrix: Optional[jnp.ndarray] = None
    
    
@dataclass
class AdaptationDecision:
    """Decision made by the adaptive system."""
    timestamp: float
    old_parameters: Dict[str, Any]
    new_parameters: Dict[str, Any]
    confidence: float
    predicted_improvement: float
    adaptation_reason: str
    execution_time: float


class DeviceDriftPredictor:
    """Predicts device parameter drift using time series analysis."""
    
    def __init__(self, config: AdaptiveQEMConfig):
        self.config = config
        self.drift_history = deque(maxlen=config.prediction_window)
        self.prediction_models = {}
        self.prediction_accuracy = {}
        
    def update_device_state(self, device_state: DeviceState):
        """Update device state history."""
        self.drift_history.append(device_state)
        
        # Update prediction models if we have enough data
        if len(self.drift_history) >= self.config.prediction_window:
            self._update_prediction_models()
    
    def _update_prediction_models(self):
        """Update time series prediction models for device parameters."""
        # Extract time series data
        timestamps = [state.timestamp for state in self.drift_history]
        
        # Model each parameter separately
        parameters_to_model = ['gate_fidelities', 'readout_fidelities', 'coherence_times', 'error_rates']
        
        for param_name in parameters_to_model:
            # Extract parameter values over time
            param_series = []
            for state in self.drift_history:
                param_dict = getattr(state, param_name)
                # Average all values in the dictionary for simplicity
                avg_value = np.mean(list(param_dict.values())) if param_dict else 0.0
                param_series.append(avg_value)
            
            # Fit simple linear trend model
            if len(param_series) >= 3:
                time_normalized = np.array(timestamps) - timestamps[0]
                coeffs = np.polyfit(time_normalized, param_series, deg=min(2, len(param_series)-1))
                self.prediction_models[param_name] = {
                    'coefficients': coeffs,
                    'last_timestamp': timestamps[-1],
                    'model_type': 'polynomial'
                }
    
    def predict_drift(self, steps_ahead: int = None) -> Dict[str, Dict[str, float]]:
        """Predict device parameter drift."""
        if steps_ahead is None:
            steps_ahead = self.config.prediction_horizon
            
        predictions = {}
        current_time = time.time()
        
        for param_name, model in self.prediction_models.items():
            if model['model_type'] == 'polynomial':
                # Predict future values
                time_delta = current_time - model['last_timestamp']
                future_times = [time_delta + i * self.config.monitoring_interval for i in range(1, steps_ahead + 1)]
                
                predicted_values = []
                confidences = []
                
                for t in future_times:
                    # Evaluate polynomial
                    predicted_value = sum(coeff * (t ** i) for i, coeff in enumerate(reversed(model['coefficients'])))
                    predicted_values.append(predicted_value)
                    
                    # Confidence decreases with prediction distance
                    confidence = max(0.1, self.config.confidence_threshold * np.exp(-t / (steps_ahead * 2)))
                    confidences.append(confidence)
                
                predictions[param_name] = {
                    'values': predicted_values,
                    'confidences': confidences,
                    'mean_confidence': np.mean(confidences)
                }
        
        return predictions
    
    def detect_drift_anomalies(self) -> List[Dict[str, Any]]:
        """Detect sudden changes or anomalies in device behavior."""
        if len(self.drift_history) < 5:
            return []
        
        anomalies = []
        recent_states = list(self.drift_history)[-5:]  # Last 5 measurements
        
        # Check for sudden changes in key parameters
        for i in range(1, len(recent_states)):
            prev_state = recent_states[i-1]
            curr_state = recent_states[i]
            
            # Check gate fidelity drops
            for gate_type in prev_state.gate_fidelities:
                if gate_type in curr_state.gate_fidelities:
                    prev_fidelity = prev_state.gate_fidelities[gate_type]
                    curr_fidelity = curr_state.gate_fidelities[gate_type]
                    
                    # Detect significant drops (>5% relative change)
                    if prev_fidelity > 0 and (prev_fidelity - curr_fidelity) / prev_fidelity > 0.05:
                        anomalies.append({
                            'type': 'gate_fidelity_drop',
                            'parameter': gate_type,
                            'timestamp': curr_state.timestamp,
                            'severity': (prev_fidelity - curr_fidelity) / prev_fidelity,
                            'details': f"Gate {gate_type} fidelity dropped from {prev_fidelity:.4f} to {curr_fidelity:.4f}"
                        })
            
            # Check coherence time degradation
            for coherence_type in ['t1', 't2']:
                if (coherence_type in prev_state.coherence_times and 
                    coherence_type in curr_state.coherence_times):
                    
                    prev_coherence = prev_state.coherence_times[coherence_type]
                    curr_coherence = curr_state.coherence_times[coherence_type]
                    
                    if prev_coherence > 0 and (prev_coherence - curr_coherence) / prev_coherence > 0.1:
                        anomalies.append({
                            'type': 'coherence_degradation',
                            'parameter': coherence_type,
                            'timestamp': curr_state.timestamp,
                            'severity': (prev_coherence - curr_coherence) / prev_coherence,
                            'details': f"Coherence {coherence_type} degraded from {prev_coherence:.2f}μs to {curr_coherence:.2f}μs"
                        })
        
        return anomalies


class AdaptiveNoisePredictor:
    """Predicts optimal noise scaling factors based on real-time conditions."""
    
    def __init__(self, config: AdaptiveQEMConfig):
        self.config = config
        self.noise_factor_history = deque(maxlen=config.prediction_window)
        self.performance_history = deque(maxlen=config.prediction_window)
        self.environmental_correlations = {}
        
    def update_performance_data(self, noise_factors: List[float], error_reduction: float, 
                               device_state: DeviceState, execution_context: Dict[str, Any]):
        """Update historical performance data."""
        entry = {
            'timestamp': time.time(),
            'noise_factors': noise_factors,
            'error_reduction': error_reduction,
            'device_state': device_state,
            'context': execution_context
        }
        
        self.performance_history.append(entry)
        self._update_correlations()
    
    def _update_correlations(self):
        """Update correlations between environmental factors and optimal parameters."""
        if len(self.performance_history) < 10:
            return
        
        # Extract features and performance metrics
        features = []
        performances = []
        
        for entry in self.performance_history:
            device_state = entry['device_state']
            feature_vector = [
                device_state.temperature,
                device_state.queue_length,
                np.mean(list(device_state.gate_fidelities.values())),
                np.mean(list(device_state.error_rates.values())),
                time.time() - device_state.calibration_time
            ]
            features.append(feature_vector)
            performances.append(entry['error_reduction'])
        
        # Compute correlations
        features_array = np.array(features)
        performances_array = np.array(performances)
        
        feature_names = ['temperature', 'queue_length', 'avg_gate_fidelity', 'avg_error_rate', 'time_since_calibration']
        
        for i, feature_name in enumerate(feature_names):
            if len(features_array) > 1:
                correlation = np.corrcoef(features_array[:, i], performances_array)[0, 1]
                if not np.isnan(correlation):
                    self.environmental_correlations[feature_name] = correlation
    
    def predict_optimal_noise_factors(self, device_state: DeviceState, 
                                    circuit_complexity: float) -> Tuple[List[float], float]:
        """Predict optimal noise factors based on current conditions."""
        # Base noise factors
        base_factors = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        # Adjust based on device state
        adjustments = self._compute_environmental_adjustments(device_state)
        
        # Adjust based on circuit complexity
        complexity_factor = 1.0 + 0.1 * (circuit_complexity - 1.0)  # Scale with complexity
        
        # Apply adjustments
        adjusted_factors = []
        for factor in base_factors:
            adjusted_factor = factor * complexity_factor * adjustments['scale_factor']
            adjusted_factor = max(1.0, min(5.0, adjusted_factor))  # Clamp to reasonable range
            adjusted_factors.append(adjusted_factor)
        
        # Remove duplicates and sort
        adjusted_factors = sorted(list(set(np.round(adjusted_factors, 2))))
        
        # Confidence based on correlation strength
        confidence = min(0.9, max(0.3, np.mean([abs(corr) for corr in self.environmental_correlations.values()])))
        
        return adjusted_factors, confidence
    
    def _compute_environmental_adjustments(self, device_state: DeviceState) -> Dict[str, float]:
        """Compute adjustments based on environmental correlations."""
        adjustments = {'scale_factor': 1.0}
        
        # Temperature adjustment
        if 'temperature' in self.environmental_correlations:
            temp_corr = self.environmental_correlations['temperature']
            if device_state.temperature > 0.02:  # High temperature
                if temp_corr < 0:  # Negative correlation means high temp hurts performance
                    adjustments['scale_factor'] *= 1.1  # Use higher noise factors
        
        # Queue length adjustment
        if 'queue_length' in self.environmental_correlations:
            queue_corr = self.environmental_correlations['queue_length']
            if device_state.queue_length > 50:  # High queue
                if queue_corr < 0:  # High queue hurts performance
                    adjustments['scale_factor'] *= 1.05
        
        # Gate fidelity adjustment
        if 'avg_gate_fidelity' in self.environmental_correlations:
            avg_fidelity = np.mean(list(device_state.gate_fidelities.values()))
            if avg_fidelity < 0.95:  # Low fidelity
                adjustments['scale_factor'] *= 1.15  # Use more aggressive mitigation
        
        return adjustments


class DynamicMitigationSelector:
    """Dynamically selects the best mitigation method based on real-time conditions."""
    
    def __init__(self, config: AdaptiveQEMConfig):
        self.config = config
        self.method_performance = {
            'zne': deque(maxlen=50),
            'pec': deque(maxlen=50),
            'vd': deque(maxlen=50),
            'cdr': deque(maxlen=50)
        }
        self.context_method_mapping = {}
        
    def update_method_performance(self, method: str, performance: float, context: Dict[str, Any]):
        """Update performance history for a mitigation method."""
        if method in self.method_performance:
            self.method_performance[method].append(performance)
            
            # Update context-method mapping
            context_key = self._create_context_key(context)
            if context_key not in self.context_method_mapping:
                self.context_method_mapping[context_key] = {}
            
            if method not in self.context_method_mapping[context_key]:
                self.context_method_mapping[context_key][method] = []
            
            self.context_method_mapping[context_key][method].append(performance)
    
    def select_optimal_method(self, circuit, device_state: DeviceState, 
                            execution_context: Dict[str, Any]) -> Tuple[str, float]:
        """Select the optimal mitigation method for current conditions."""
        context_key = self._create_context_key({
            'circuit_depth': circuit.depth,
            'num_qubits': circuit.num_qubits,
            'avg_gate_fidelity': np.mean(list(device_state.gate_fidelities.values())),
            'queue_length': device_state.queue_length
        })
        
        # If we have specific context history, use it
        if context_key in self.context_method_mapping:
            context_performance = self.context_method_mapping[context_key]
            if context_performance:
                # Find method with best average performance in this context
                best_method = max(context_performance.keys(), 
                                key=lambda m: np.mean(context_performance[m]))
                confidence = len(context_performance[best_method]) / 10.0  # More data = higher confidence
                return best_method, min(confidence, 0.9)
        
        # Fallback to overall performance
        method_scores = {}
        for method, performances in self.method_performance.items():
            if performances:
                # Weight recent performance more heavily
                weights = np.exp(-0.1 * np.arange(len(performances)))[::-1]  # Recent data weighted more
                weighted_performance = np.average(performances, weights=weights)
                method_scores[method] = weighted_performance
        
        if method_scores:
            best_method = max(method_scores.keys(), key=method_scores.get)
            confidence = len(self.method_performance[best_method]) / 50.0
            return best_method, min(confidence, 0.8)
        
        # Default fallback
        return 'zne', 0.3
    
    def _create_context_key(self, context: Dict[str, Any]) -> str:
        """Create a hashable context key for method selection."""
        # Discretize continuous values for context matching
        discretized = {}
        
        if 'circuit_depth' in context:
            discretized['depth_range'] = 'low' if context['circuit_depth'] < 10 else 'medium' if context['circuit_depth'] < 50 else 'high'
        
        if 'num_qubits' in context:
            discretized['qubit_range'] = 'small' if context['num_qubits'] < 5 else 'medium' if context['num_qubits'] < 15 else 'large'
        
        if 'avg_gate_fidelity' in context:
            discretized['fidelity_range'] = 'low' if context['avg_gate_fidelity'] < 0.9 else 'medium' if context['avg_gate_fidelity'] < 0.95 else 'high'
        
        if 'queue_length' in context:
            discretized['queue_range'] = 'short' if context['queue_length'] < 10 else 'medium' if context['queue_length'] < 50 else 'long'
        
        # Create string key
        return '_'.join(f"{k}:{v}" for k, v in sorted(discretized.items()))


class ContextAwareMitigation:
    """Context-aware mitigation that adapts to execution environment."""
    
    def __init__(self, config: AdaptiveQEMConfig):
        self.config = config
        self.context_history = deque(maxlen=200)
        self.adaptation_rules = {}
        
    def update_execution_context(self, context: Dict[str, Any], result: Dict[str, Any]):
        """Update context history with execution results."""
        entry = {
            'timestamp': time.time(),
            'context': context.copy(),
            'result': result.copy()
        }
        self.context_history.append(entry)
        
        # Update adaptation rules
        self._learn_adaptation_rules()
    
    def _learn_adaptation_rules(self):
        """Learn adaptation rules from historical context-result pairs."""
        if len(self.context_history) < 20:
            return
        
        # Group by similar contexts
        context_groups = {}
        for entry in self.context_history:
            context_key = self._create_context_signature(entry['context'])
            if context_key not in context_groups:
                context_groups[context_key] = []
            context_groups[context_key].append(entry)
        
        # Learn rules for each context group
        for context_key, entries in context_groups.items():
            if len(entries) >= 5:  # Need minimum data for reliable rules
                # Analyze what works best in this context
                best_performance = 0
                best_parameters = None
                
                for entry in entries:
                    if 'error_reduction' in entry['result']:
                        performance = entry['result']['error_reduction']
                        if performance > best_performance:
                            best_performance = performance
                            best_parameters = entry['result'].get('parameters', {})
                
                if best_parameters:
                    self.adaptation_rules[context_key] = {
                        'parameters': best_parameters,
                        'expected_performance': best_performance,
                        'confidence': min(0.9, len(entries) / 20.0)
                    }
    
    def adapt_to_context(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt mitigation parameters based on current execution context."""
        context_key = self._create_context_signature(current_context)
        
        # Check if we have a rule for this context
        if context_key in self.adaptation_rules:
            rule = self.adaptation_rules[context_key]
            if rule['confidence'] > 0.5:
                logger.info(f"Applying learned adaptation rule for context: {context_key}")
                return rule['parameters'].copy()
        
        # Fallback to heuristic adaptations
        adaptations = {}
        
        # Time-of-day adaptations
        current_hour = time.localtime().tm_hour
        if current_hour < 6 or current_hour > 22:  # Night/early morning
            # Devices may be colder, potentially better performance
            adaptations['noise_factor_max'] = 2.5  # Less aggressive mitigation
        else:  # Daytime
            # Devices warmer, potentially worse performance  
            adaptations['noise_factor_max'] = 3.5  # More aggressive mitigation
        
        # System load adaptations
        if current_context.get('system_load', 0.5) > 0.8:
            # High system load, prioritize speed
            adaptations['bootstrap_samples'] = 50
            adaptations['extrapolation_order'] = 1
        else:
            # Low system load, prioritize accuracy
            adaptations['bootstrap_samples'] = 200
            adaptations['extrapolation_order'] = 2
        
        return adaptations
    
    def _create_context_signature(self, context: Dict[str, Any]) -> str:
        """Create a signature string for context matching."""
        signature_parts = []
        
        # Time-based context (discretized)
        if 'timestamp' in context:
            hour = time.localtime(context['timestamp']).tm_hour
            time_period = 'night' if hour < 6 or hour > 22 else 'day'
            signature_parts.append(f"time:{time_period}")
        
        # System context
        if 'system_load' in context:
            load_level = 'high' if context['system_load'] > 0.7 else 'medium' if context['system_load'] > 0.3 else 'low'
            signature_parts.append(f"load:{load_level}")
        
        # Circuit context
        if 'circuit_complexity' in context:
            complexity = 'high' if context['circuit_complexity'] > 100 else 'medium' if context['circuit_complexity'] > 20 else 'low'
            signature_parts.append(f"complexity:{complexity}")
        
        return '_'.join(signature_parts)


class RealTimeQEMAdapter:
    """Main real-time adaptive QEM coordinator."""
    
    def __init__(self, config: AdaptiveQEMConfig):
        self.config = config
        
        # Components
        self.drift_predictor = DeviceDriftPredictor(config)
        self.noise_predictor = AdaptiveNoisePredictor(config)
        self.method_selector = DynamicMitigationSelector(config)
        self.context_adapter = ContextAwareMitigation(config)
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.adaptation_queue = queue.Queue()
        
        # State tracking
        self.current_device_state = None
        self.adaptation_history = deque(maxlen=500)
        self.performance_monitor = PerformanceMonitor()
        
        # Async support
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def start_adaptive_monitoring(self, device_interface):
        """Start real-time adaptive monitoring."""
        logger.info("Starting real-time adaptive QEM monitoring...")
        
        self.monitoring_active = True
        self.device_interface = device_interface
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Adaptive monitoring started")
    
    def stop_adaptive_monitoring(self):
        """Stop real-time adaptive monitoring."""
        logger.info("Stopping real-time adaptive QEM monitoring...")
        
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        logger.info("Adaptive monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Get current device state
                device_state = self._get_current_device_state()
                if device_state:
                    self.current_device_state = device_state
                    
                    # Update predictors
                    self.drift_predictor.update_device_state(device_state)
                    
                    # Check for anomalies
                    anomalies = self.drift_predictor.detect_drift_anomalies()
                    if anomalies:
                        logger.warning(f"Detected {len(anomalies)} device anomalies")
                        for anomaly in anomalies:
                            logger.warning(f"Anomaly: {anomaly['details']}")
                    
                    # Process any pending adaptations
                    self._process_adaptation_queue()
                
                # Sleep for remaining interval time
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.monitoring_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _get_current_device_state(self) -> Optional[DeviceState]:
        """Get current device state from device interface."""
        try:
            # This would interface with actual device APIs
            # For now, create synthetic state
            return DeviceState(
                timestamp=time.time(),
                gate_fidelities={'cx': 0.95, 'h': 0.98, 'rz': 0.99},
                readout_fidelities={i: 0.96 + 0.02 * np.random.random() for i in range(5)},
                coherence_times={'t1': 100.0, 't2': 50.0},
                temperature=0.01 + 0.005 * np.random.random(),
                calibration_time=time.time() - 3600,  # 1 hour ago
                queue_length=int(20 + 30 * np.random.random()),
                error_rates={'single_qubit': 0.001, 'two_qubit': 0.01}
            )
        except Exception as e:
            logger.error(f"Failed to get device state: {e}")
            return None
    
    def _process_adaptation_queue(self):
        """Process pending adaptation requests."""
        while not self.adaptation_queue.empty():
            try:
                adaptation_request = self.adaptation_queue.get_nowait()
                self._execute_adaptation(adaptation_request)
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing adaptation: {e}")
    
    def _execute_adaptation(self, adaptation_request: Dict[str, Any]):
        """Execute a specific adaptation."""
        start_time = time.time()
        
        # Extract adaptation details
        circuit = adaptation_request['circuit']
        context = adaptation_request['context']
        callback = adaptation_request.get('callback')
        
        try:
            # Select optimal method
            method, method_confidence = self.method_selector.select_optimal_method(
                circuit, self.current_device_state, context
            )
            
            # Predict optimal parameters
            noise_factors, noise_confidence = self.noise_predictor.predict_optimal_noise_factors(
                self.current_device_state, context.get('circuit_complexity', 1.0)
            )
            
            # Apply context-aware adaptations
            context_adaptations = self.context_adapter.adapt_to_context(context)
            
            # Combine adaptations
            adapted_parameters = {
                'method': method,
                'noise_factors': noise_factors,
                **context_adaptations
            }
            
            # Calculate overall confidence
            overall_confidence = (method_confidence + noise_confidence) / 2.0
            
            # Create adaptation decision
            decision = AdaptationDecision(
                timestamp=time.time(),
                old_parameters=adaptation_request.get('current_parameters', {}),
                new_parameters=adapted_parameters,
                confidence=overall_confidence,
                predicted_improvement=0.1 * overall_confidence,  # Heuristic
                adaptation_reason=f"Method: {method} (conf: {method_confidence:.2f}), Noise prediction (conf: {noise_confidence:.2f})",
                execution_time=time.time() - start_time
            )
            
            # Store adaptation history
            self.adaptation_history.append(decision)
            
            # Execute callback if provided
            if callback:
                callback(decision)
                
            logger.info(f"Adaptation completed in {decision.execution_time:.3f}s: {decision.adaptation_reason}")
            
        except Exception as e:
            logger.error(f"Adaptation execution failed: {e}")
    
    def request_adaptation(self, circuit, current_parameters: Dict[str, Any], 
                         execution_context: Dict[str, Any], 
                         callback: Optional[Callable] = None) -> bool:
        """Request adaptive parameter optimization."""
        if not self.monitoring_active:
            logger.warning("Adaptive monitoring not active, cannot process adaptation request")
            return False
        
        adaptation_request = {
            'circuit': circuit,
            'current_parameters': current_parameters,
            'context': execution_context,
            'callback': callback,
            'timestamp': time.time()
        }
        
        try:
            # Check if we can process immediately (low delay) or queue
            if self.adaptation_queue.qsize() < 10:  # Queue not full
                self.adaptation_queue.put(adaptation_request)
                return True
            else:
                logger.warning("Adaptation queue full, dropping request")
                return False
        except Exception as e:
            logger.error(f"Failed to queue adaptation request: {e}")
            return False
    
    def get_adaptation_performance(self) -> Dict[str, Any]:
        """Get performance analytics of the adaptive system."""
        if not self.adaptation_history:
            return {"status": "no_adaptations"}
        
        recent_adaptations = list(self.adaptation_history)[-50:]  # Last 50 adaptations
        
        analytics = {
            'total_adaptations': len(self.adaptation_history),
            'recent_adaptations': len(recent_adaptations),
            'average_confidence': np.mean([a.confidence for a in recent_adaptations]),
            'average_execution_time': np.mean([a.execution_time for a in recent_adaptations]),
            'predicted_improvements': [a.predicted_improvement for a in recent_adaptations],
            'adaptation_frequency': len(recent_adaptations) / max(1, (time.time() - recent_adaptations[0].timestamp) / 3600),  # per hour
            'most_common_methods': self._analyze_method_usage(),
            'adaptation_trends': self._analyze_adaptation_trends()
        }
        
        return analytics
    
    def _analyze_method_usage(self) -> Dict[str, int]:
        """Analyze which mitigation methods are used most frequently."""
        method_counts = {}
        for adaptation in self.adaptation_history:
            method = adaptation.new_parameters.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return dict(sorted(method_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_adaptation_trends(self) -> Dict[str, Any]:
        """Analyze trends in adaptation decisions."""
        if len(self.adaptation_history) < 10:
            return {"status": "insufficient_data"}
        
        recent = list(self.adaptation_history)[-20:]
        
        # Confidence trend
        confidences = [a.confidence for a in recent]
        confidence_trend = "improving" if confidences[-1] > confidences[0] else "declining"
        
        # Execution time trend
        exec_times = [a.execution_time for a in recent]
        speed_trend = "improving" if exec_times[-1] < exec_times[0] else "declining"
        
        return {
            'confidence_trend': confidence_trend,
            'speed_trend': speed_trend,
            'average_recent_confidence': np.mean(confidences),
            'average_recent_speed': np.mean(exec_times),
            'adaptation_stability': np.std(confidences) < 0.1  # Low variance = stable
        }