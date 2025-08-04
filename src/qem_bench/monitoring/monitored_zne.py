"""Zero-Noise Extrapolation with integrated monitoring."""

import time
import numpy as np
from typing import List, Union, Optional, Dict, Any, Callable
from dataclasses import dataclass
import warnings

# Import monitoring components
from ..monitoring import SystemMonitor, PerformanceMonitor, QuantumResourceMonitor, AlertManager
from ..metrics import MetricsCollector, CircuitMetrics, NoiseMetrics
from ..health import HealthChecker

# Import original ZNE components
from ..mitigation.zne.core import ZeroNoiseExtrapolation as BaseZNE, ZNEConfig
from ..mitigation.zne.scaling import NoiseScaler, UnitaryFoldingScaler
from ..mitigation.zne.extrapolation import Extrapolator, RichardsonExtrapolator
from ..mitigation.zne.result import ZNEResult


@dataclass
class MonitoredZNEConfig(ZNEConfig):
    """Extended configuration for monitored Zero-Noise Extrapolation."""
    # Monitoring settings
    enable_monitoring: bool = True
    enable_performance_monitoring: bool = True
    enable_resource_monitoring: bool = True
    enable_health_checks: bool = True
    enable_metrics_collection: bool = True
    enable_circuit_analysis: bool = True
    enable_noise_analysis: bool = True
    
    # Alert settings
    enable_alerts: bool = True
    max_execution_time_alert: float = 300.0  # 5 minutes
    min_fidelity_alert: float = 0.5
    max_error_rate_alert: float = 0.1
    
    # Export settings
    auto_export_metrics: bool = False
    export_directory: Optional[str] = None


class MonitoredZeroNoiseExtrapolation:
    """
    Zero-Noise Extrapolation with comprehensive monitoring capabilities.
    
    This class extends the base ZNE implementation with integrated monitoring,
    metrics collection, health checks, and performance analysis. It provides
    detailed insights into the mitigation process and can help identify
    performance bottlenecks and optimization opportunities.
    
    Example:
        >>> # Create monitored ZNE with default monitoring
        >>> zne = MonitoredZeroNoiseExtrapolation(
        ...     noise_factors=[1, 1.5, 2, 2.5, 3],
        ...     extrapolator="richardson"
        ... )
        >>> 
        >>> # Run mitigation with monitoring
        >>> result = zne.mitigate(circuit, backend, observable)
        >>> 
        >>> # Get monitoring insights
        >>> performance_stats = zne.get_performance_summary()
        >>> resource_usage = zne.get_resource_summary()
        >>> health_status = zne.get_health_summary()
    """
    
    def __init__(
        self,
        noise_scaler: Optional[NoiseScaler] = None,
        noise_factors: Optional[List[float]] = None,
        extrapolator: Union[str, Extrapolator] = "richardson",
        config: Optional[MonitoredZNEConfig] = None,
        **kwargs
    ):
        # Initialize configuration
        self.config = config or MonitoredZNEConfig(
            noise_factors=noise_factors or [1.0, 1.5, 2.0, 2.5, 3.0]
        )
        
        # Initialize base ZNE implementation
        self._base_zne = BaseZNE(
            noise_scaler=noise_scaler,
            noise_factors=noise_factors,
            extrapolator=extrapolator,
            config=self.config,  # Pass the base config fields
            **kwargs
        )
        
        # Initialize monitoring components
        self._init_monitoring_components()
        
        # Execution history
        self._execution_history = []
    
    def _init_monitoring_components(self):
        """Initialize all monitoring components."""
        # System monitoring
        if self.config.enable_monitoring:
            self.system_monitor = SystemMonitor()
            self.system_monitor.start()
        else:
            self.system_monitor = None
        
        # Performance monitoring
        if self.config.enable_performance_monitoring:
            self.performance_monitor = PerformanceMonitor()
        else:
            self.performance_monitor = None
        
        # Resource monitoring
        if self.config.enable_resource_monitoring:
            self.resource_monitor = QuantumResourceMonitor()
        else:
            self.resource_monitor = None
        
        # Health checking
        if self.config.enable_health_checks:
            self.health_checker = HealthChecker()
            self.health_checker.start_monitoring()
        else:
            self.health_checker = None
        
        # Metrics collection
        if self.config.enable_metrics_collection:
            self.metrics_collector = MetricsCollector()
        else:
            self.metrics_collector = None
        
        # Circuit analysis
        if self.config.enable_circuit_analysis:
            self.circuit_analyzer = CircuitMetrics()
        else:
            self.circuit_analyzer = None
        
        # Noise analysis
        if self.config.enable_noise_analysis:
            self.noise_analyzer = NoiseMetrics()
        else:
            self.noise_analyzer = None
        
        # Alert management
        if self.config.enable_alerts:
            self.alert_manager = AlertManager()
            self._setup_alert_rules()
        else:
            self.alert_manager = None
    
    def _setup_alert_rules(self):
        """Setup default alert rules."""
        if not self.alert_manager:
            return
        
        from ..monitoring.alert_manager import AlertRule, AlertSeverity, AlertType
        
        # Execution time alert
        self.alert_manager.add_rule(AlertRule(
            name="zne_execution_time",
            metric_name="zne_execution_duration",
            condition="greater_than",
            threshold=self.config.max_execution_time_alert,
            severity=AlertSeverity.HIGH,
            alert_type=AlertType.PERFORMANCE
        ))
        
        # Fidelity alert
        self.alert_manager.add_rule(AlertRule(
            name="zne_low_fidelity",
            metric_name="zne_fidelity_estimate",
            condition="less_than",
            threshold=self.config.min_fidelity_alert,
            severity=AlertSeverity.MEDIUM,
            alert_type=AlertType.PERFORMANCE
        ))
        
        # Error rate alert
        self.alert_manager.add_rule(AlertRule(
            name="zne_high_error_rate",
            metric_name="zne_error_rate",
            condition="greater_than",
            threshold=self.config.max_error_rate_alert,
            severity=AlertSeverity.HIGH,
            alert_type=AlertType.ERROR
        ))
    
    def mitigate(
        self,
        circuit: Any,
        backend: Any,
        observable: Optional[Any] = None,
        shots: int = 1024,
        **execution_kwargs
    ) -> ZNEResult:
        """
        Apply zero-noise extrapolation with comprehensive monitoring.
        
        Args:
            circuit: Quantum circuit to execute
            backend: Quantum backend for execution
            observable: Observable to measure (if None, use all-Z)
            shots: Number of measurement shots per noise factor
            **execution_kwargs: Additional arguments for circuit execution
            
        Returns:
            ZNEResult containing raw and mitigated expectation values plus monitoring data
        """
        execution_id = f"zne_{int(time.time())}_{id(circuit)}"
        
        # Start monitoring context
        with self._monitoring_context(execution_id, circuit, backend, shots):
            # Analyze circuit before execution
            circuit_analysis = self._analyze_circuit(circuit, execution_id)
            
            # Analyze noise model if available
            noise_analysis = self._analyze_noise_model(backend, execution_id)
            
            # Perform the actual ZNE mitigation
            with self._performance_context("zne_mitigation"):
                result = self._base_zne.mitigate(
                    circuit, backend, observable, shots, **execution_kwargs
                )
            
            # Enhance result with monitoring data
            result = self._enhance_result_with_monitoring(
                result, execution_id, circuit_analysis, noise_analysis
            )
            
            # Record execution in history
            self._record_execution(result, execution_id)
            
            # Check alerts
            self._check_alerts(result)
            
            return result
    
    def _monitoring_context(self, execution_id: str, circuit: Any, backend: Any, shots: int):
        """Context manager for comprehensive monitoring."""
        class MonitoringContext:
            def __init__(self, parent, execution_id, circuit, backend, shots):
                self.parent = parent
                self.execution_id = execution_id
                self.circuit = circuit
                self.backend = backend
                self.shots = shots
            
            def __enter__(self):
                # Start resource tracking
                if self.parent.resource_monitor:
                    self.resource_tracker = self.parent.resource_monitor.track_execution(
                        f"zne_execution_{self.execution_id}",
                        metadata={
                            'execution_id': self.execution_id,
                            'shots': self.shots,
                            'backend': str(self.backend)
                        }
                    )
                    return self.resource_tracker.__enter__()
                return None
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if hasattr(self, 'resource_tracker'):
                    self.resource_tracker.__exit__(exc_type, exc_val, exc_tb)
        
        return MonitoringContext(self, execution_id, circuit, backend, shots)
    
    def _performance_context(self, operation_name: str):
        """Context manager for performance monitoring."""
        if self.performance_monitor:
            return self.performance_monitor.time_operation(operation_name)
        else:
            # Dummy context manager
            class DummyContext:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
            return DummyContext()
    
    def _analyze_circuit(self, circuit: Any, execution_id: str) -> Optional[Any]:
        """Analyze the input circuit."""
        if not self.circuit_analyzer:
            return None
        
        try:
            analysis = self.circuit_analyzer.analyze_circuit(
                circuit, 
                circuit_id=f"zne_circuit_{execution_id}",
                metadata={'execution_id': execution_id}
            )
            
            # Record circuit metrics
            if self.metrics_collector:
                self.metrics_collector.set_gauge(
                    "zne_circuit_depth", analysis.circuit_depth,
                    labels={"execution_id": execution_id}
                )
                self.metrics_collector.set_gauge(
                    "zne_circuit_gates", analysis.num_gates,
                    labels={"execution_id": execution_id}
                )
                self.metrics_collector.set_gauge(
                    "zne_circuit_qubits", analysis.num_qubits,
                    labels={"execution_id": execution_id}
                )
                self.metrics_collector.set_gauge(
                    "zne_two_qubit_ratio", analysis.two_qubit_ratio,
                    labels={"execution_id": execution_id}
                )
            
            return analysis
            
        except Exception as e:
            warnings.warn(f"Circuit analysis failed: {e}")
            return None
    
    def _analyze_noise_model(self, backend: Any, execution_id: str) -> Optional[Any]:
        """Analyze the noise model from backend."""
        if not self.noise_analyzer:
            return None
        
        try:
            # Try to extract noise model from backend
            noise_model = None
            if hasattr(backend, 'noise_model'):
                noise_model = backend.noise_model
            elif hasattr(backend, 'properties'):
                # Create a simple noise model from backend properties
                noise_model = backend.properties()
            
            if noise_model:
                analysis = self.noise_analyzer.analyze_noise_model(
                    noise_model,
                    noise_model_id=f"zne_noise_{execution_id}",
                    metadata={'execution_id': execution_id}
                )
                
                # Record noise metrics
                if self.metrics_collector:
                    self.metrics_collector.set_gauge(
                        "zne_overall_error_rate", analysis.overall_error_rate,
                        labels={"execution_id": execution_id}
                    )
                    self.metrics_collector.set_gauge(
                        "zne_mitigation_difficulty", analysis.mitigation_difficulty,
                        labels={"execution_id": execution_id}
                    )
                
                return analysis
            
        except Exception as e:
            warnings.warn(f"Noise analysis failed: {e}")
        
        return None
    
    def _enhance_result_with_monitoring(self, result: ZNEResult, execution_id: str,
                                      circuit_analysis: Optional[Any] = None,
                                      noise_analysis: Optional[Any] = None) -> ZNEResult:
        """Enhance ZNE result with monitoring data."""
        # Add monitoring metadata to result
        monitoring_data = {
            'execution_id': execution_id,
            'timestamp': time.time(),
            'monitoring_enabled': True
        }
        
        # Add circuit analysis
        if circuit_analysis:
            monitoring_data['circuit_analysis'] = {
                'num_qubits': circuit_analysis.num_qubits,
                'num_gates': circuit_analysis.num_gates,
                'circuit_depth': circuit_analysis.circuit_depth,
                'two_qubit_ratio': circuit_analysis.two_qubit_ratio,
                'estimated_runtime': circuit_analysis.estimated_runtime,
                'noise_susceptibility': circuit_analysis.noise_susceptibility
            }
        
        # Add noise analysis
        if noise_analysis:
            monitoring_data['noise_analysis'] = {
                'overall_error_rate': noise_analysis.overall_error_rate,
                'mitigation_difficulty': noise_analysis.mitigation_difficulty,
                'recommended_methods': noise_analysis.recommended_methods,
                'coherence_quality': noise_analysis.coherence_quality
            }
        
        # Add performance metrics
        if self.performance_monitor:
            stats = self.performance_monitor.get_stats("zne_mitigation")
            if stats:
                monitoring_data['performance_stats'] = {
                    'execution_count': stats.count,
                    'avg_duration_ms': stats.avg_duration_ms,
                    'min_duration_ms': stats.min_duration * 1000,
                    'max_duration_ms': stats.max_duration * 1000
                }
        
        # Add system metrics
        if self.system_monitor:
            system_stats = self.system_monitor.get_current_stats()
            if system_stats:
                monitoring_data['system_stats'] = {
                    'cpu_percent': system_stats.cpu_percent,
                    'memory_percent': system_stats.memory_percent,
                    'memory_used_gb': system_stats.memory_used_gb
                }
        
        # Store monitoring data in result
        if not hasattr(result, 'monitoring_data'):
            result.monitoring_data = {}
        result.monitoring_data.update(monitoring_data)
        
        return result
    
    def _record_execution(self, result: ZNEResult, execution_id: str):
        """Record execution in history and metrics."""
        execution_record = {
            'execution_id': execution_id,
            'timestamp': time.time(),
            'raw_value': result.raw_value,
            'mitigated_value': result.mitigated_value,
            'error_reduction': result.error_reduction,
            'noise_factors': result.noise_factors,
            'monitoring_data': getattr(result, 'monitoring_data', {})
        }
        
        self._execution_history.append(execution_record)
        
        # Record core metrics
        if self.metrics_collector:
            self.metrics_collector.increment_counter(
                "zne_executions_total",
                labels={"execution_id": execution_id}
            )
            
            self.metrics_collector.set_gauge(
                "zne_raw_value", result.raw_value,
                labels={"execution_id": execution_id}
            )
            
            self.metrics_collector.set_gauge(
                "zne_mitigated_value", result.mitigated_value,
                labels={"execution_id": execution_id}
            )
            
            if result.error_reduction is not None:
                self.metrics_collector.set_gauge(
                    "zne_error_reduction", result.error_reduction,
                    labels={"execution_id": execution_id}
                )
    
    def _check_alerts(self, result: ZNEResult):
        """Check for alert conditions."""
        if not self.alert_manager:
            return
        
        monitoring_data = getattr(result, 'monitoring_data', {})
        
        # Check execution time
        performance_stats = monitoring_data.get('performance_stats', {})
        if 'avg_duration_ms' in performance_stats:
            duration_seconds = performance_stats['avg_duration_ms'] / 1000
            self.alert_manager.check_metric(
                "zne_execution_duration", duration_seconds, 
                source="monitored_zne"
            )
        
        # Check fidelity estimate
        if result.mitigated_value is not None:
            # Simple fidelity estimate (this would be more sophisticated in practice)
            fidelity_estimate = abs(result.mitigated_value)
            self.alert_manager.check_metric(
                "zne_fidelity_estimate", fidelity_estimate,
                source="monitored_zne"
            )
        
        # Check error rates from noise analysis
        noise_analysis = monitoring_data.get('noise_analysis', {})
        if 'overall_error_rate' in noise_analysis:
            self.alert_manager.check_metric(
                "zne_error_rate", noise_analysis['overall_error_rate'],
                source="monitored_zne"
            )
    
    def get_performance_summary(self, duration_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        if not self.performance_monitor:
            return {'error': 'Performance monitoring not enabled'}
        
        return self.performance_monitor.get_summary_report(duration_seconds)
    
    def get_resource_summary(self, duration_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Get resource usage summary."""
        if not self.resource_monitor:
            return {'error': 'Resource monitoring not enabled'}
        
        return self.resource_monitor.get_global_stats(duration_seconds)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health check summary.""" 
        if not self.health_checker:
            return {'error': 'Health checking not enabled'}
        
        return self.health_checker.get_health_summary()
    
    def get_metrics_overview(self) -> Dict[str, Any]:
        """Get metrics collection overview."""
        if not self.metrics_collector:
            return {'error': 'Metrics collection not enabled'}
        
        return self.metrics_collector.get_metrics_overview()
    
    def get_active_alerts(self) -> List[Any]:
        """Get active alerts."""
        if not self.alert_manager:
            return []
        
        return self.alert_manager.get_active_alerts()
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get execution history."""
        if limit:
            return self._execution_history[-limit:]
        return list(self._execution_history)
    
    def export_monitoring_data(self, filepath: str, format: str = "json"):
        """Export all monitoring data."""
        export_data = {
            'export_timestamp': time.time(),
            'config': {
                'enable_monitoring': self.config.enable_monitoring,
                'enable_performance_monitoring': self.config.enable_performance_monitoring,
                'enable_resource_monitoring': self.config.enable_resource_monitoring,
                'enable_health_checks': self.config.enable_health_checks,
                'enable_metrics_collection': self.config.enable_metrics_collection
            },
            'execution_history': self.get_execution_history(),
            'performance_summary': self.get_performance_summary(),
            'resource_summary': self.get_resource_summary(),
            'health_summary': self.get_health_summary(),
            'metrics_overview': self.get_metrics_overview(),
            'active_alerts': [
                {
                    'id': alert.id,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'message': alert.message,
                    'timestamp': alert.timestamp
                }
                for alert in self.get_active_alerts()
            ]
        }
        
        if format.lower() == "json":
            import json
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def cleanup(self):
        """Cleanup monitoring resources."""
        if self.system_monitor:
            self.system_monitor.stop()
        
        if self.health_checker:
            self.health_checker.stop_monitoring()
        
        if self.performance_monitor:
            self.performance_monitor.__exit__(None, None, None)
        
        if self.metrics_collector:
            self.metrics_collector.__exit__(None, None, None)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()