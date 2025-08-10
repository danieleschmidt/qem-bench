"""
Health monitoring and diagnostics for QEM-Bench.

Provides comprehensive health checks, system diagnostics, and proactive
monitoring to ensure optimal operation of all QEM components.
"""

import time
import threading
import psutil
import platform
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from pathlib import Path
import json
from enum import Enum

from .logger import get_logger


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    name: str
    status: HealthStatus
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    details: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "recommendations": self.recommendations
        }
    
    @property
    def is_healthy(self) -> bool:
        """Check if status indicates health."""
        return self.status == HealthStatus.HEALTHY
    
    @property
    def needs_attention(self) -> bool:
        """Check if status needs attention."""
        return self.status in (HealthStatus.WARNING, HealthStatus.CRITICAL)


class HealthChecker(ABC):
    """Abstract base class for health checkers."""
    
    def __init__(self, name: str, interval_seconds: int = 60):
        self.name = name
        self.interval_seconds = interval_seconds
        self.logger = get_logger(f"health.{name}")
        self.last_check_time: Optional[datetime] = None
        self.last_result: Optional[HealthCheckResult] = None
    
    @abstractmethod
    def check_health(self) -> HealthCheckResult:
        """Perform health check and return result."""
        pass
    
    def is_check_due(self) -> bool:
        """Check if health check is due."""
        if self.last_check_time is None:
            return True
        
        return datetime.now() - self.last_check_time >= timedelta(seconds=self.interval_seconds)
    
    def run_check(self) -> HealthCheckResult:
        """Run health check if due."""
        if not self.is_check_due():
            return self.last_result or HealthCheckResult(
                self.name, HealthStatus.UNKNOWN, "Check not due"
            )
        
        try:
            result = self.check_health()
            self.last_check_time = datetime.now()
            self.last_result = result
            
            # Log result
            if result.is_healthy:
                self.logger.info(f"Health check passed: {result.message}", 
                               "health_check", result.metrics)
            elif result.status == HealthStatus.WARNING:
                self.logger.warning(f"Health check warning: {result.message}", 
                                  "health_check", result.metrics)
            else:
                self.logger.error(f"Health check failed: {result.message}", 
                                "health_check", result.metrics)
            
            return result
        
        except Exception as e:
            error_result = HealthCheckResult(
                self.name, 
                HealthStatus.CRITICAL,
                f"Health check failed with exception: {str(e)}",
                details=str(e)
            )
            self.last_result = error_result
            self.last_check_time = datetime.now()
            
            self.logger.error(f"Health check exception: {str(e)}", e, "health_check")
            return error_result


class SystemResourceChecker(HealthChecker):
    """Check system resource utilization."""
    
    def __init__(self, cpu_warning_threshold: float = 80.0, 
                 cpu_critical_threshold: float = 95.0,
                 memory_warning_threshold: float = 85.0,
                 memory_critical_threshold: float = 95.0):
        super().__init__("system_resources", interval_seconds=30)
        self.cpu_warning = cpu_warning_threshold
        self.cpu_critical = cpu_critical_threshold
        self.memory_warning = memory_warning_threshold
        self.memory_critical = memory_critical_threshold
    
    def check_health(self) -> HealthCheckResult:
        """Check system resource health."""
        # Get current resource usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3)
        }
        
        # Determine overall status
        status = HealthStatus.HEALTHY
        message_parts = []
        recommendations = []
        
        # Check CPU
        if cpu_percent >= self.cpu_critical:
            status = HealthStatus.CRITICAL
            message_parts.append(f"CPU usage critical ({cpu_percent:.1f}%)")
            recommendations.append("Reduce computational load or scale resources")
        elif cpu_percent >= self.cpu_warning:
            status = max(status, HealthStatus.WARNING) if status != HealthStatus.CRITICAL else status
            message_parts.append(f"CPU usage high ({cpu_percent:.1f}%)")
            recommendations.append("Monitor CPU usage trends")
        
        # Check memory
        if memory.percent >= self.memory_critical:
            status = HealthStatus.CRITICAL
            message_parts.append(f"Memory usage critical ({memory.percent:.1f}%)")
            recommendations.append("Free memory or increase available RAM")
        elif memory.percent >= self.memory_warning:
            status = max(status, HealthStatus.WARNING) if status != HealthStatus.CRITICAL else status
            message_parts.append(f"Memory usage high ({memory.percent:.1f}%)")
            recommendations.append("Monitor memory usage and consider optimization")
        
        # Check disk space
        if disk.percent >= 95:
            status = HealthStatus.CRITICAL
            message_parts.append(f"Disk usage critical ({disk.percent:.1f}%)")
            recommendations.append("Free disk space immediately")
        elif disk.percent >= 85:
            status = max(status, HealthStatus.WARNING) if status != HealthStatus.CRITICAL else status
            message_parts.append(f"Disk usage high ({disk.percent:.1f}%)")
            recommendations.append("Clean up disk space")
        
        # Create message
        if status == HealthStatus.HEALTHY:
            message = f"System resources healthy (CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%, Disk: {disk.percent:.1f}%)"
        else:
            message = "; ".join(message_parts)
        
        return HealthCheckResult(
            self.name, status, message, metrics, 
            recommendations=recommendations
        )


class JAXBackendChecker(HealthChecker):
    """Check JAX backend availability and performance."""
    
    def __init__(self):
        super().__init__("jax_backend", interval_seconds=120)
    
    def check_health(self) -> HealthCheckResult:
        """Check JAX backend health."""
        try:
            import jax
            import jax.numpy as jnp
            
            # Basic functionality test
            start_time = time.time()
            x = jnp.array([1.0, 2.0, 3.0])
            y = jnp.dot(x, x)  # Simple operation
            result = float(y)  # Force computation
            duration_ms = (time.time() - start_time) * 1000
            
            # Get backend info
            backend = jax.default_backend()
            devices = jax.devices()
            
            metrics = {
                "backend": backend,
                "num_devices": len(devices),
                "devices": [str(d) for d in devices],
                "test_operation_ms": duration_ms,
                "test_result": result
            }
            
            # Determine status
            if duration_ms > 1000:  # > 1 second for simple operation
                status = HealthStatus.WARNING
                message = f"JAX backend slow ({duration_ms:.2f}ms for simple operation)"
                recommendations = ["Check JAX installation", "Verify hardware acceleration"]
            else:
                status = HealthStatus.HEALTHY
                message = f"JAX backend healthy ({backend}, {len(devices)} devices)"
                recommendations = []
            
            return HealthCheckResult(
                self.name, status, message, metrics,
                recommendations=recommendations
            )
        
        except ImportError:
            return HealthCheckResult(
                self.name, HealthStatus.CRITICAL,
                "JAX not available",
                recommendations=["Install JAX: pip install jax"]
            )
        
        except Exception as e:
            return HealthCheckResult(
                self.name, HealthStatus.CRITICAL,
                f"JAX backend test failed: {str(e)}",
                details=str(e),
                recommendations=["Check JAX installation", "Verify system compatibility"]
            )


class QuantumCircuitChecker(HealthChecker):
    """Check quantum circuit operations."""
    
    def __init__(self):
        super().__init__("quantum_circuits", interval_seconds=300)
    
    def check_health(self) -> HealthCheckResult:
        """Check quantum circuit functionality."""
        try:
            from ..jax.circuits import JAXCircuit
            from ..jax.simulator import JAXSimulator
            
            # Create test circuit
            circuit = JAXCircuit(2, name="health_check")
            circuit.h(0)
            circuit.cx(0, 1)
            
            # Test simulation
            start_time = time.time()
            simulator = JAXSimulator(2)
            result = simulator.run(circuit, shots=100)
            duration_ms = (time.time() - start_time) * 1000
            
            # Validate results
            if result.statevector is None:
                raise ValueError("Simulation returned no statevector")
            
            if result.measurement_counts is None:
                raise ValueError("Simulation returned no measurement counts")
            
            metrics = {
                "circuit_qubits": circuit.num_qubits,
                "circuit_gates": circuit.size,
                "simulation_time_ms": duration_ms,
                "measurement_outcomes": len(result.measurement_counts),
                "total_shots": sum(result.measurement_counts.values())
            }
            
            # Check performance
            if duration_ms > 5000:  # > 5 seconds
                status = HealthStatus.WARNING
                message = f"Circuit simulation slow ({duration_ms:.2f}ms for 2-qubit circuit)"
                recommendations = ["Optimize circuit operations", "Check system resources"]
            else:
                status = HealthStatus.HEALTHY
                message = f"Quantum circuits healthy (2-qubit simulation: {duration_ms:.2f}ms)"
                recommendations = []
            
            return HealthCheckResult(
                self.name, status, message, metrics,
                recommendations=recommendations
            )
        
        except ImportError as e:
            return HealthCheckResult(
                self.name, HealthStatus.CRITICAL,
                f"Quantum circuit modules not available: {str(e)}",
                recommendations=["Check QEM-Bench installation"]
            )
        
        except Exception as e:
            return HealthCheckResult(
                self.name, HealthStatus.CRITICAL,
                f"Quantum circuit test failed: {str(e)}",
                details=str(e),
                recommendations=["Check quantum circuit implementation", "Verify dependencies"]
            )


class ZNEChecker(HealthChecker):
    """Check ZNE error mitigation functionality."""
    
    def __init__(self):
        super().__init__("zne_mitigation", interval_seconds=600)  # 10 minutes
    
    def check_health(self) -> HealthCheckResult:
        """Check ZNE functionality."""
        try:
            from ..mitigation.zne.core import ZeroNoiseExtrapolation
            from ..jax.circuits import JAXCircuit
            from ..jax.simulator import JAXSimulator
            from ..jax.observables import ZObservable
            
            # Create test circuit
            circuit = JAXCircuit(1, name="zne_health_check")
            circuit.x(0)  # Simple |1⟩ state
            
            # Mock backend for testing
            class MockBackend:
                def __init__(self, simulator):
                    self.sim = simulator
                
                def run_with_observable(self, circuit, observable, shots=1000, **kwargs):
                    result = self.sim.run(circuit, observables=[observable])
                    
                    class MockResult:
                        def __init__(self, exp_val):
                            self.expectation_value = exp_val
                    
                    return MockResult(result.expectation_values[observable.name])
            
            # Test ZNE
            start_time = time.time()
            simulator = JAXSimulator(1)
            backend = MockBackend(simulator)
            observable = ZObservable(0)
            
            zne = ZeroNoiseExtrapolation(
                noise_factors=[1.0, 1.5, 2.0],
                extrapolator="richardson"
            )
            
            zne_result = zne.mitigate(circuit, backend, observable)
            duration_ms = (time.time() - start_time) * 1000
            
            # Validate results
            if zne_result.raw_value is None or zne_result.mitigated_value is None:
                raise ValueError("ZNE returned invalid results")
            
            metrics = {
                "zne_duration_ms": duration_ms,
                "raw_value": float(zne_result.raw_value),
                "mitigated_value": float(zne_result.mitigated_value),
                "noise_factors": len(zne_result.noise_factors),
                "extrapolation_method": zne_result.extrapolation_method,
                "fit_quality": float(zne_result.fit_quality) if hasattr(zne_result, 'fit_quality') else None
            }
            
            # Check performance and quality
            recommendations = []
            if duration_ms > 10000:  # > 10 seconds
                status = HealthStatus.WARNING
                message = f"ZNE mitigation slow ({duration_ms:.2f}ms)"
                recommendations.append("Optimize ZNE parameters")
            elif abs(zne_result.raw_value - (-1.0)) > 0.1:  # Should be close to -1 for |1⟩
                status = HealthStatus.WARNING
                message = f"ZNE raw value unexpected ({zne_result.raw_value:.3f})"
                recommendations.append("Check circuit simulation accuracy")
            else:
                status = HealthStatus.HEALTHY
                message = f"ZNE mitigation healthy ({duration_ms:.2f}ms, {zne_result.extrapolation_method})"
            
            return HealthCheckResult(
                self.name, status, message, metrics,
                recommendations=recommendations
            )
        
        except ImportError as e:
            return HealthCheckResult(
                self.name, HealthStatus.CRITICAL,
                f"ZNE modules not available: {str(e)}",
                recommendations=["Check QEM-Bench ZNE implementation"]
            )
        
        except Exception as e:
            return HealthCheckResult(
                self.name, HealthStatus.CRITICAL,
                f"ZNE test failed: {str(e)}",
                details=str(e),
                recommendations=["Check ZNE implementation", "Verify numerical stability"]
            )


class MemoryLeakChecker(HealthChecker):
    """Check for memory leaks during operations."""
    
    def __init__(self):
        super().__init__("memory_leaks", interval_seconds=900)  # 15 minutes
        self.baseline_memory = None
        self.memory_history = []
    
    def check_health(self) -> HealthCheckResult:
        """Check for memory leaks."""
        try:
            import gc
            
            # Force garbage collection
            gc.collect()
            
            # Get current memory usage
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024**2)  # MB
            
            # Initialize baseline if first run
            if self.baseline_memory is None:
                self.baseline_memory = current_memory
            
            # Store in history (keep last 10 measurements)
            self.memory_history.append(current_memory)
            if len(self.memory_history) > 10:
                self.memory_history.pop(0)
            
            # Calculate memory growth
            memory_growth = current_memory - self.baseline_memory
            
            # Calculate trend if enough history
            trend_mb_per_check = 0
            if len(self.memory_history) >= 5:
                # Simple linear regression on last 5 points
                n = len(self.memory_history)
                x = list(range(n))
                y = self.memory_history
                
                x_mean = sum(x) / n
                y_mean = sum(y) / n
                
                numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
                denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
                
                if denominator != 0:
                    trend_mb_per_check = numerator / denominator
            
            metrics = {
                "current_memory_mb": current_memory,
                "baseline_memory_mb": self.baseline_memory,
                "memory_growth_mb": memory_growth,
                "trend_mb_per_check": trend_mb_per_check,
                "history_points": len(self.memory_history)
            }
            
            # Determine status
            recommendations = []
            if memory_growth > 500:  # > 500 MB growth
                status = HealthStatus.CRITICAL
                message = f"Significant memory growth detected ({memory_growth:.1f} MB)"
                recommendations.extend([
                    "Check for memory leaks in quantum circuits",
                    "Review large array allocations",
                    "Consider restarting the process"
                ])
            elif memory_growth > 200:  # > 200 MB growth
                status = HealthStatus.WARNING
                message = f"Memory growth detected ({memory_growth:.1f} MB)"
                recommendations.append("Monitor memory usage closely")
            elif trend_mb_per_check > 10:  # Growing by >10 MB per check
                status = HealthStatus.WARNING
                message = f"Memory usage trending upward ({trend_mb_per_check:.1f} MB/check)"
                recommendations.append("Investigate potential memory leak")
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage stable ({current_memory:.1f} MB, growth: {memory_growth:.1f} MB)"
            
            return HealthCheckResult(
                self.name, status, message, metrics,
                recommendations=recommendations
            )
        
        except Exception as e:
            return HealthCheckResult(
                self.name, HealthStatus.CRITICAL,
                f"Memory leak check failed: {str(e)}",
                details=str(e)
            )


class HealthMonitor:
    """Main health monitoring system."""
    
    def __init__(self):
        self.checkers: Dict[str, HealthChecker] = {}
        self.logger = get_logger("health_monitor")
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.check_interval = 30  # seconds
        
        # Register default checkers
        self.register_checker(SystemResourceChecker())
        self.register_checker(JAXBackendChecker())
        self.register_checker(QuantumCircuitChecker())
        self.register_checker(ZNEChecker())
        self.register_checker(MemoryLeakChecker())
    
    def register_checker(self, checker: HealthChecker):
        """Register a health checker."""
        self.checkers[checker.name] = checker
        self.logger.info(f"Registered health checker: {checker.name}")
    
    def unregister_checker(self, name: str) -> bool:
        """Unregister a health checker."""
        if name in self.checkers:
            del self.checkers[name]
            self.logger.info(f"Unregistered health checker: {name}")
            return True
        return False
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}
        
        for name, checker in self.checkers.items():
            try:
                result = checker.run_check()
                results[name] = result
            except Exception as e:
                self.logger.error(f"Failed to run health check {name}: {str(e)}", e)
                results[name] = HealthCheckResult(
                    name, HealthStatus.CRITICAL,
                    f"Check execution failed: {str(e)}"
                )
        
        return results
    
    def run_specific_check(self, checker_name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        if checker_name not in self.checkers:
            return None
        
        try:
            return self.checkers[checker_name].run_check()
        except Exception as e:
            self.logger.error(f"Failed to run health check {checker_name}: {str(e)}", e)
            return HealthCheckResult(
                checker_name, HealthStatus.CRITICAL,
                f"Check execution failed: {str(e)}"
            )
    
    def get_overall_status(self) -> Tuple[HealthStatus, Dict[str, HealthCheckResult]]:
        """Get overall system health status."""
        results = self.run_all_checks()
        
        # Determine overall status
        if not results:
            return HealthStatus.UNKNOWN, results
        
        status_counts = {status: 0 for status in HealthStatus}
        for result in results.values():
            status_counts[result.status] += 1
        
        # Overall status logic
        if status_counts[HealthStatus.CRITICAL] > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.WARNING] > 0:
            overall_status = HealthStatus.WARNING
        elif status_counts[HealthStatus.HEALTHY] > 0:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        return overall_status, results
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Started health monitoring")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Stopped health monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                overall_status, results = self.get_overall_status()
                
                # Log overall status
                unhealthy_count = sum(1 for r in results.values() if not r.is_healthy)
                
                if overall_status == HealthStatus.HEALTHY:
                    self.logger.info(f"System health check passed ({len(results)} checks)")
                elif overall_status == HealthStatus.WARNING:
                    self.logger.warning(f"System health warnings ({unhealthy_count}/{len(results)} checks need attention)")
                else:
                    self.logger.error(f"System health critical ({unhealthy_count}/{len(results)} checks failed)")
                
                # Log individual issues
                for name, result in results.items():
                    if result.needs_attention:
                        self.logger.warning(f"Health issue in {name}: {result.message}")
                
                time.sleep(self.check_interval)
            
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {str(e)}", e)
                time.sleep(self.check_interval)
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        overall_status, results = self.get_overall_status()
        
        # System information
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time()  # Process uptime approximation
        }
        
        # Summary statistics
        status_counts = {status.value: 0 for status in HealthStatus}
        total_recommendations = 0
        
        for result in results.values():
            status_counts[result.status.value] += 1
            total_recommendations += len(result.recommendations)
        
        summary = {
            "overall_status": overall_status.value,
            "total_checks": len(results),
            "status_breakdown": status_counts,
            "total_recommendations": total_recommendations
        }
        
        # Convert results to dict format
        check_results = {name: result.to_dict() for name, result in results.items()}
        
        return {
            "system_info": system_info,
            "summary": summary,
            "checks": check_results
        }
    
    def save_health_report(self, filepath: Optional[str] = None):
        """Save health report to file."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"logs/health_report_{timestamp}.json"
        
        report = self.generate_health_report()
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Health report saved to {filepath}")


# Global health monitor instance
_health_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    return _health_monitor


def run_health_checks() -> Dict[str, HealthCheckResult]:
    """Run all health checks."""
    return _health_monitor.run_all_checks()


def get_system_health() -> Tuple[HealthStatus, Dict[str, HealthCheckResult]]:
    """Get overall system health."""
    return _health_monitor.get_overall_status()


def start_health_monitoring():
    """Start continuous health monitoring."""
    _health_monitor.start_monitoring()


def stop_health_monitoring():
    """Stop continuous health monitoring."""
    _health_monitor.stop_monitoring()


def generate_health_report() -> Dict[str, Any]:
    """Generate health report."""
    return _health_monitor.generate_health_report()


# Convenience functions for specific health checks
def check_system_resources() -> HealthCheckResult:
    """Check system resource health."""
    result = _health_monitor.run_specific_check("system_resources")
    return result or HealthCheckResult(
        "system_resources", HealthStatus.UNKNOWN, "Checker not available"
    )


def check_jax_backend() -> HealthCheckResult:
    """Check JAX backend health."""
    result = _health_monitor.run_specific_check("jax_backend")
    return result or HealthCheckResult(
        "jax_backend", HealthStatus.UNKNOWN, "Checker not available"
    )


def check_quantum_circuits() -> HealthCheckResult:
    """Check quantum circuit functionality."""
    result = _health_monitor.run_specific_check("quantum_circuits")
    return result or HealthCheckResult(
        "quantum_circuits", HealthStatus.UNKNOWN, "Checker not available"
    )


def check_zne_mitigation() -> HealthCheckResult:
    """Check ZNE error mitigation."""
    result = _health_monitor.run_specific_check("zne_mitigation")
    return result or HealthCheckResult(
        "zne_mitigation", HealthStatus.UNKNOWN, "Checker not available"
    )