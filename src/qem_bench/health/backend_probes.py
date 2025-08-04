"""Backend health probes for QEM-Bench monitoring."""

import time
import logging
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import threading

from .health_checker import HealthCheckProvider, HealthCheck, HealthStatus


logger = logging.getLogger(__name__)


class BackendHealthProbe(HealthCheckProvider):
    """
    Health check provider for quantum backends.
    
    Monitors:
    - Backend connectivity and availability
    - Queue status and wait times
    - Calibration data freshness
    - Backend configuration validity
    - Performance metrics
    """
    
    def __init__(self, backends: Optional[List[Any]] = None, timeout: float = 30.0):
        """
        Initialize backend health probe.
        
        Args:
            backends: List of backends to monitor. If None, will auto-discover.
            timeout: Timeout for backend operations in seconds
        """
        self.backends = backends or []
        self.timeout = timeout
        self._backend_cache = {}
        self._last_check_time = {}
    
    def get_name(self) -> str:
        """Get the name of this health check provider."""
        return "backend_health_probe"
    
    def is_critical(self) -> bool:
        """Backend health is critical for quantum computations."""
        return True
    
    def get_check_interval(self) -> float:
        """Check backends every 2 minutes."""
        return 120.0
    
    def check_health(self) -> HealthCheck:
        """Perform backend health checks."""
        start_time = time.time()
        issues = []
        warnings = []
        details = {}
        recommendations = []
        
        # Discover backends if not provided
        if not self.backends:
            discovered_backends = self._discover_backends()
            details['backend_discovery'] = discovered_backends
            if discovered_backends.get('errors'):
                warnings.extend(discovered_backends['errors'])
        
        # Check each backend
        backend_results = {}
        
        if self.backends:
            # Use thread pool for parallel backend checks
            with ThreadPoolExecutor(max_workers=min(5, len(self.backends))) as executor:
                future_to_backend = {
                    executor.submit(self._check_single_backend, backend): backend
                    for backend in self.backends
                }
                
                for future in future_to_backend:
                    backend = future_to_backend[future]
                    backend_name = self._get_backend_name(backend)
                    
                    try:
                        result = future.result(timeout=self.timeout)
                        backend_results[backend_name] = result
                        
                        if result['status'] == 'error':
                            issues.append(f"Backend {backend_name}: {result['message']}")
                            if result.get('recommendations'):
                                recommendations.extend(result['recommendations'])
                        elif result['status'] == 'warning':
                            warnings.append(f"Backend {backend_name}: {result['message']}")
                            
                    except FutureTimeoutError:
                        backend_results[backend_name] = {
                            'status': 'error',
                            'message': f"Health check timed out after {self.timeout}s",
                            'recommendations': ['Check backend connectivity']
                        }
                        issues.append(f"Backend {backend_name} health check timed out")
                    except Exception as e:
                        backend_results[backend_name] = {
                            'status': 'error',
                            'message': f"Health check failed: {str(e)}",
                            'error': str(e)
                        }
                        issues.append(f"Backend {backend_name} health check failed: {str(e)}")
        else:
            warnings.append("No backends configured for health monitoring")
            recommendations.append("Configure quantum backends for monitoring")
        
        details['backends'] = backend_results
        
        # Overall status assessment
        total_backends = len(backend_results)
        healthy_backends = len([r for r in backend_results.values() if r['status'] == 'ok'])
        warning_backends = len([r for r in backend_results.values() if r['status'] == 'warning'])
        error_backends = len([r for r in backend_results.values() if r['status'] == 'error'])
        
        details['summary'] = {
            'total_backends': total_backends,
            'healthy_backends': healthy_backends,
            'warning_backends': warning_backends,
            'error_backends': error_backends,
            'health_percentage': (healthy_backends / total_backends * 100) if total_backends > 0 else 0
        }
        
        # Determine overall status
        if total_backends == 0:
            status = HealthStatus.WARNING
            message = "No backends available for health monitoring"
        elif error_backends > 0:
            if error_backends == total_backends:
                status = HealthStatus.UNHEALTHY
                message = f"All {total_backends} backends are unhealthy"
            else:
                status = HealthStatus.WARNING
                message = f"{error_backends}/{total_backends} backends are unhealthy"
        elif warning_backends > 0:
            status = HealthStatus.WARNING
            message = f"{warning_backends}/{total_backends} backends have warnings"
        else:
            status = HealthStatus.HEALTHY
            message = f"All {total_backends} backends are healthy"
        
        return HealthCheck(
            name=self.get_name(),
            status=status,
            timestamp=start_time,
            duration=time.time() - start_time,
            message=message,
            details=details,
            recommendations=list(set(recommendations))
        )
    
    def _discover_backends(self) -> Dict[str, Any]:
        """Discover available quantum backends."""
        discovery_result = {
            'backends_found': [],
            'errors': [],
            'providers_checked': []
        }
        
        # Try Qiskit backends
        try:
            from qiskit import IBMQ
            from qiskit.providers.aer import AerSimulator
            
            discovery_result['providers_checked'].append('qiskit')
            
            # Check for local simulators
            try:
                aer_sim = AerSimulator()
                discovery_result['backends_found'].append({
                    'name': 'aer_simulator',
                    'type': 'simulator',
                    'provider': 'qiskit_aer'
                })
            except Exception as e:
                discovery_result['errors'].append(f"Qiskit Aer error: {str(e)}")
            
            # Check for IBMQ backends (if configured)
            try:
                # This would require IBMQ account setup
                # providers = IBMQ.providers()
                # for provider in providers:
                #     for backend in provider.backends():
                #         discovery_result['backends_found'].append({
                #             'name': backend.name(),
                #             'type': 'hardware' if not backend.configuration().simulator else 'simulator',
                #             'provider': 'ibmq'
                #         })
                pass
            except Exception as e:
                discovery_result['errors'].append(f"IBMQ backends not available: {str(e)}")
                
        except ImportError:
            discovery_result['errors'].append("Qiskit not available")
        
        # Try Cirq backends
        try:
            import cirq
            discovery_result['providers_checked'].append('cirq')
            
            # Check for Google Quantum AI backends
            try:
                # This would require Google Quantum AI setup
                pass
            except Exception as e:
                discovery_result['errors'].append(f"Cirq/Google backends not available: {str(e)}")
                
        except ImportError:
            discovery_result['errors'].append("Cirq not available")
        
        # Try PennyLane backends
        try:
            import pennylane as qml
            discovery_result['providers_checked'].append('pennylane')
            
            # Check available devices
            available_devices = qml.about()
            if hasattr(qml, 'devices'):
                discovery_result['backends_found'].append({
                    'name': 'pennylane_default_qubit',
                    'type': 'simulator',
                    'provider': 'pennylane'
                })
        except ImportError:
            discovery_result['errors'].append("PennyLane not available")
        
        return discovery_result
    
    def _check_single_backend(self, backend: Any) -> Dict[str, Any]:
        """Check health of a single backend."""
        backend_name = self._get_backend_name(backend)
        
        try:
            # Basic connectivity check
            connectivity_result = self._check_connectivity(backend)
            if connectivity_result['status'] == 'error':
                return connectivity_result
            
            # Configuration check
            config_result = self._check_configuration(backend)
            if config_result['status'] == 'error':
                return config_result
            
            # Queue status check
            queue_result = self._check_queue_status(backend)
            
            # Calibration check
            calibration_result = self._check_calibration(backend)
            
            # Performance check
            performance_result = self._check_performance(backend)
            
            # Aggregate results
            all_results = [connectivity_result, config_result, queue_result, 
                          calibration_result, performance_result]
            
            warnings = []
            recommendations = []
            details = {}
            
            for result in all_results:
                if result.get('warnings'):
                    warnings.extend(result['warnings'])
                if result.get('recommendations'):
                    recommendations.extend(result['recommendations'])
                if result.get('details'):
                    details.update(result['details'])
            
            # Overall backend status
            if any(r['status'] == 'error' for r in all_results):
                status = 'error'
                message = "Backend has critical issues"
            elif any(r['status'] == 'warning' for r in all_results):
                status = 'warning'
                message = "Backend has performance or configuration warnings"
            else:
                status = 'ok'
                message = "Backend is healthy and operational"
            
            return {
                'status': status,
                'message': message,
                'warnings': warnings,
                'recommendations': recommendations,
                'details': details,
                'checks': {
                    'connectivity': connectivity_result,
                    'configuration': config_result,
                    'queue': queue_result,
                    'calibration': calibration_result,
                    'performance': performance_result
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Backend health check failed: {str(e)}",
                'error': str(e),
                'recommendations': ['Check backend configuration and connectivity']
            }
    
    def _get_backend_name(self, backend: Any) -> str:
        """Get the name of a backend."""
        if hasattr(backend, 'name'):
            return backend.name() if callable(backend.name) else backend.name
        elif hasattr(backend, '_name'):
            return backend._name
        elif hasattr(backend, '__class__'):
            return f"{backend.__class__.__name__}_{id(backend)}"
        else:
            return f"unknown_backend_{id(backend)}"
    
    def _check_connectivity(self, backend: Any) -> Dict[str, Any]:
        """Check backend connectivity."""
        try:
            # Try to access basic backend properties
            if hasattr(backend, 'status'):
                status = backend.status()
                if hasattr(status, 'operational') and not status.operational:
                    return {
                        'status': 'error',
                        'message': 'Backend is not operational',
                        'recommendations': ['Wait for backend to become operational']
                    }
            
            # Try to get backend configuration
            if hasattr(backend, 'configuration'):
                config = backend.configuration()
                # Successfully accessed configuration
            
            return {
                'status': 'ok',
                'message': 'Backend connectivity verified'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Connectivity check failed: {str(e)}',
                'error': str(e),
                'recommendations': ['Check network connection and backend credentials']
            }
    
    def _check_configuration(self, backend: Any) -> Dict[str, Any]:
        """Check backend configuration."""
        details = {}
        warnings = []
        
        try:
            if hasattr(backend, 'configuration'):
                config = backend.configuration()
                
                # Check basic configuration properties
                if hasattr(config, 'n_qubits'):
                    details['n_qubits'] = config.n_qubits
                    if config.n_qubits < 5:
                        warnings.append(f"Backend has only {config.n_qubits} qubits")
                
                if hasattr(config, 'simulator'):
                    details['is_simulator'] = config.simulator
                
                if hasattr(config, 'max_shots'):
                    details['max_shots'] = config.max_shots
                
                if hasattr(config, 'max_experiments'):
                    details['max_experiments'] = config.max_experiments
                
                # Check supported operations
                if hasattr(config, 'basis_gates'):
                    details['basis_gates'] = config.basis_gates
                
                if hasattr(config, 'coupling_map'):
                    details['has_coupling_map'] = config.coupling_map is not None
            
            return {
                'status': 'warning' if warnings else 'ok',
                'message': 'Configuration check completed',
                'warnings': warnings,
                'details': details
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Configuration check failed: {str(e)}',
                'error': str(e)
            }
    
    def _check_queue_status(self, backend: Any) -> Dict[str, Any]:
        """Check backend queue status."""
        details = {}
        warnings = []
        recommendations = []
        
        try:
            if hasattr(backend, 'status'):
                status = backend.status()
                
                if hasattr(status, 'pending_jobs'):
                    details['pending_jobs'] = status.pending_jobs
                    if status.pending_jobs > 100:
                        warnings.append(f"High queue load: {status.pending_jobs} pending jobs")
                        recommendations.append("Consider using a different backend or waiting")
                
                if hasattr(status, 'status_msg'):
                    details['status_message'] = status.status_msg
                
                # Estimate wait time based on queue
                if hasattr(status, 'pending_jobs') and status.pending_jobs > 0:
                    # Rough estimate: assume 1 minute per job
                    estimated_wait = status.pending_jobs * 1
                    details['estimated_wait_minutes'] = estimated_wait
                    
                    if estimated_wait > 30:
                        warnings.append(f"Long estimated wait time: {estimated_wait} minutes")
            
            return {
                'status': 'warning' if warnings else 'ok',
                'message': 'Queue status checked',
                'warnings': warnings,
                'recommendations': recommendations,
                'details': details
            }
            
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Queue status check failed: {str(e)}',
                'details': {'queue_check_error': str(e)}
            }
    
    def _check_calibration(self, backend: Any) -> Dict[str, Any]:
        """Check backend calibration data."""
        details = {}
        warnings = []
        
        try:
            if hasattr(backend, 'properties'):
                properties = backend.properties()
                
                if properties:
                    # Check calibration timestamp
                    if hasattr(properties, 'last_update_date'):
                        last_update = properties.last_update_date
                        details['last_calibration'] = str(last_update)
                        
                        # Check if calibration is recent (within 24 hours)
                        if hasattr(last_update, 'timestamp'):
                            hours_since_calibration = (time.time() - last_update.timestamp()) / 3600
                            details['hours_since_calibration'] = hours_since_calibration
                            
                            if hours_since_calibration > 24:
                                warnings.append(f"Calibration data is {hours_since_calibration:.1f} hours old")
                    
                    # Check error rates
                    if hasattr(properties, 'gate_errors'):
                        gate_errors = []
                        for gate_error in properties.gate_errors():
                            if hasattr(gate_error, 'value'):
                                gate_errors.append(gate_error.value)
                        
                        if gate_errors:
                            avg_gate_error = sum(gate_errors) / len(gate_errors)
                            details['avg_gate_error_rate'] = avg_gate_error
                            
                            if avg_gate_error > 0.01:  # 1% error rate threshold
                                warnings.append(f"High average gate error rate: {avg_gate_error:.4f}")
                    
                    # Check readout errors
                    if hasattr(properties, 'readout_errors'):
                        readout_errors = []
                        for readout_error in properties.readout_errors():
                            if hasattr(readout_error, 'value'):
                                readout_errors.append(readout_error.value)
                        
                        if readout_errors:
                            avg_readout_error = sum(readout_errors) / len(readout_errors)
                            details['avg_readout_error_rate'] = avg_readout_error
                            
                            if avg_readout_error > 0.05:  # 5% readout error threshold
                                warnings.append(f"High average readout error rate: {avg_readout_error:.4f}")
            
            return {
                'status': 'warning' if warnings else 'ok',
                'message': 'Calibration data checked',
                'warnings': warnings,
                'details': details
            }
            
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Calibration check failed: {str(e)}',
                'details': {'calibration_check_error': str(e)}
            }
    
    def _check_performance(self, backend: Any) -> Dict[str, Any]:
        """Check backend performance characteristics."""
        details = {}
        warnings = []
        
        try:
            # This is a placeholder for performance monitoring
            # In a real implementation, you might:
            # - Run a simple test circuit and measure execution time
            # - Check historical performance metrics
            # - Analyze success rates
            
            details['performance_check'] = 'placeholder'
            
            return {
                'status': 'ok',
                'message': 'Performance check completed',
                'warnings': warnings,
                'details': details
            }
            
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Performance check failed: {str(e)}',
                'details': {'performance_check_error': str(e)}
            }
    
    def add_backend(self, backend: Any):
        """Add a backend to monitor."""
        if backend not in self.backends:
            self.backends.append(backend)
    
    def remove_backend(self, backend: Any):
        """Remove a backend from monitoring."""
        if backend in self.backends:
            self.backends.remove(backend)