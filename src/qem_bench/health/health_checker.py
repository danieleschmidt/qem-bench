"""Core health checking system for QEM-Bench."""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Result of a health check."""
    name: str
    status: HealthStatus
    timestamp: float
    duration: float  # seconds
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def age_seconds(self) -> float:
        """Age of the health check in seconds."""
        return time.time() - self.timestamp
    
    def is_stale(self, max_age_seconds: float) -> bool:
        """Check if the health check is stale."""
        return self.age_seconds > max_age_seconds


class HealthCheckProvider(ABC):
    """Abstract base class for health check providers."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this health check provider."""
        pass
    
    @abstractmethod
    def check_health(self) -> HealthCheck:
        """Perform the health check and return result."""
        pass
    
    def get_check_interval(self) -> float:
        """Get the interval between health checks in seconds."""
        return 60.0  # Default: 1 minute
    
    def is_critical(self) -> bool:
        """Whether this health check is critical for system operation."""
        return False


@dataclass
class HealthCheckerConfig:
    """Configuration for the health checking system."""
    enabled: bool = True
    check_interval: float = 60.0  # seconds
    stale_check_threshold: float = 300.0  # 5 minutes
    auto_remediation: bool = False
    max_check_history: int = 100
    parallel_checks: bool = True
    check_timeout: float = 30.0  # seconds


class HealthChecker:
    """
    Comprehensive health checking system for QEM-Bench.
    
    This system monitors the health of various components including backends,
    dependencies, hardware capabilities, and system resources. It can run
    continuous health checks and provide status reports.
    
    Example:
        >>> health_checker = HealthChecker()
        >>> 
        >>> # Add health check providers
        >>> health_checker.add_provider(DependencyChecker())
        >>> health_checker.add_provider(BackendHealthProbe())
        >>> 
        >>> # Run health checks
        >>> health_checker.start_monitoring()
        >>> 
        >>> # Get current health status
        >>> status = health_checker.get_overall_status()
        >>> print(f"System health: {status}")
    """
    
    def __init__(self, config: Optional[HealthCheckerConfig] = None):
        self.config = config or HealthCheckerConfig()
        self._providers: Dict[str, HealthCheckProvider] = {}
        self._check_results: Dict[str, HealthCheck] = {}
        self._check_history: Dict[str, List[HealthCheck]] = {}
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[str, HealthCheck], None]] = []
        
        # Initialize built-in health check providers
        self._init_builtin_providers()
    
    def _init_builtin_providers(self):
        """Initialize built-in health check providers."""
        try:
            from .dependency_checker import DependencyChecker
            from .hardware_detector import HardwareDetector
            
            self.add_provider(DependencyChecker())
            self.add_provider(HardwareDetector())
            
        except ImportError as e:
            logger.warning(f"Could not initialize some built-in health providers: {e}")
    
    def add_provider(self, provider: HealthCheckProvider):
        """
        Add a health check provider.
        
        Args:
            provider: HealthCheckProvider instance
        """
        name = provider.get_name()
        with self._lock:
            self._providers[name] = provider
            self._check_history[name] = []
        logger.info(f"Added health check provider: {name}")
    
    def remove_provider(self, provider_name: str):
        """
        Remove a health check provider.
        
        Args:
            provider_name: Name of the provider to remove
        """
        with self._lock:
            if provider_name in self._providers:
                del self._providers[provider_name]
                if provider_name in self._check_results:
                    del self._check_results[provider_name]
                if provider_name in self._check_history:
                    del self._check_history[provider_name]
        logger.info(f"Removed health check provider: {provider_name}")
    
    def get_providers(self) -> List[str]:
        """Get names of all registered health check providers."""
        with self._lock:
            return list(self._providers.keys())
    
    def run_check(self, provider_name: str) -> Optional[HealthCheck]:
        """
        Run a specific health check.
        
        Args:
            provider_name: Name of the provider to check
        
        Returns:
            HealthCheck result or None if provider not found
        """
        if not self.config.enabled:
            return None
        
        with self._lock:
            provider = self._providers.get(provider_name)
        
        if not provider:
            logger.warning(f"Health check provider not found: {provider_name}")
            return None
        
        try:
            start_time = time.perf_counter()
            
            # Run the health check with timeout
            if self.config.parallel_checks:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(provider.check_health)
                    try:
                        result = future.result(timeout=self.config.check_timeout)
                    except concurrent.futures.TimeoutError:
                        result = HealthCheck(
                            name=provider_name,
                            status=HealthStatus.UNHEALTHY,
                            timestamp=time.time(),
                            duration=self.config.check_timeout,
                            message=f"Health check timed out after {self.config.check_timeout}s",
                            recommendations=["Check if the health check provider is responsive"]
                        )
            else:
                result = provider.check_health()
            
            duration = time.perf_counter() - start_time
            result.duration = duration
            
            # Store result
            with self._lock:
                self._check_results[provider_name] = result
                self._check_history[provider_name].append(result)
                
                # Limit history size
                if len(self._check_history[provider_name]) > self.config.max_check_history:
                    self._check_history[provider_name].pop(0)
            
            # Trigger callbacks
            for callback in self._callbacks:
                try:
                    callback(provider_name, result)
                except Exception as e:
                    logger.error(f"Health check callback failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Health check failed for {provider_name}: {e}")
            error_result = HealthCheck(
                name=provider_name,
                status=HealthStatus.UNHEALTHY,
                timestamp=time.time(),
                duration=time.perf_counter() - start_time,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                recommendations=["Check the health check provider implementation"]
            )
            
            with self._lock:
                self._check_results[provider_name] = error_result
                self._check_history[provider_name].append(error_result)
            
            return error_result
    
    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary mapping provider names to HealthCheck results
        """
        if not self.config.enabled:
            return {}
        
        provider_names = self.get_providers()
        results = {}
        
        if self.config.parallel_checks and len(provider_names) > 1:
            # Run checks in parallel
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_name = {
                    executor.submit(self.run_check, name): name
                    for name in provider_names
                }
                
                for future in concurrent.futures.as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        result = future.result()
                        if result:
                            results[name] = result
                    except Exception as e:
                        logger.error(f"Parallel health check failed for {name}: {e}")
        else:
            # Run checks sequentially
            for name in provider_names:
                result = self.run_check(name)
                if result:
                    results[name] = result
        
        return results
    
    def get_check_result(self, provider_name: str) -> Optional[HealthCheck]:
        """
        Get the latest health check result for a provider.
        
        Args:
            provider_name: Name of the provider
        
        Returns:
            Latest HealthCheck result or None
        """
        with self._lock:
            return self._check_results.get(provider_name)
    
    def get_all_results(self, max_age_seconds: Optional[float] = None) -> Dict[str, HealthCheck]:
        """
        Get all latest health check results.
        
        Args:
            max_age_seconds: Only return results newer than this age
        
        Returns:
            Dictionary of health check results
        """
        with self._lock:
            results = dict(self._check_results)
        
        if max_age_seconds is not None:
            # Filter by age
            current_time = time.time()
            results = {
                name: result for name, result in results.items()
                if (current_time - result.timestamp) <= max_age_seconds
            }
        
        return results
    
    def get_check_history(self, provider_name: str, limit: Optional[int] = None) -> List[HealthCheck]:
        """
        Get health check history for a provider.
        
        Args:
            provider_name: Name of the provider
            limit: Maximum number of results to return (most recent first)
        
        Returns:
            List of HealthCheck results
        """
        with self._lock:
            history = self._check_history.get(provider_name, [])
            # Return most recent first
            history = list(reversed(history))
            if limit:
                history = history[:limit]
            return history
    
    def get_overall_status(self) -> HealthStatus:
        """
        Get overall system health status.
        
        Returns:
            Overall HealthStatus based on all providers
        """
        results = self.get_all_results(self.config.stale_check_threshold)
        
        if not results:
            return HealthStatus.UNKNOWN
        
        # Check for any critical failures
        critical_providers = []
        with self._lock:
            for name, provider in self._providers.items():
                if provider.is_critical():
                    critical_providers.append(name)
        
        # Determine overall status
        has_unhealthy = False
        has_warning = False
        
        for name, result in results.items():
            if result.status == HealthStatus.UNHEALTHY:
                if name in critical_providers:
                    return HealthStatus.UNHEALTHY  # Critical component is unhealthy
                has_unhealthy = True
            elif result.status == HealthStatus.WARNING:
                has_warning = True
        
        if has_unhealthy:
            return HealthStatus.WARNING  # Non-critical components unhealthy
        elif has_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive health summary.
        
        Returns:
            Dictionary with health summary information
        """
        results = self.get_all_results()
        overall_status = self.get_overall_status()
        
        # Status counts
        status_counts = {
            'healthy': 0,
            'warning': 0,
            'unhealthy': 0,
            'unknown': 0
        }
        
        for result in results.values():
            status_counts[result.status.value] += 1
        
        # Critical issues
        critical_issues = []
        warning_issues = []
        
        for name, result in results.items():
            if result.status == HealthStatus.UNHEALTHY:
                critical_issues.append({
                    'provider': name,
                    'message': result.message,
                    'recommendations': result.recommendations
                })
            elif result.status == HealthStatus.WARNING:
                warning_issues.append({
                    'provider': name,
                    'message': result.message,
                    'recommendations': result.recommendations
                })
        
        # Stale checks
        stale_checks = []
        for name, result in results.items():
            if result.is_stale(self.config.stale_check_threshold):
                stale_checks.append({
                    'provider': name,
                    'age_seconds': result.age_seconds,
                    'last_check': result.timestamp
                })
        
        return {
            'overall_status': overall_status.value,
            'timestamp': time.time(),
            'total_providers': len(self._providers),
            'status_counts': status_counts,
            'critical_issues': critical_issues,
            'warning_issues': warning_issues,
            'stale_checks': stale_checks,
            'monitoring_enabled': self._monitoring
        }
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if not self.config.enabled:
            logger.info("Health monitoring disabled")
            return
        
        if self._monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Started health monitoring with {self.config.check_interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Stopped health monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self.run_all_checks()
                time.sleep(self.config.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.config.check_interval)
    
    def add_callback(self, callback: Callable[[str, HealthCheck], None]):
        """
        Add a callback function to be called when health checks complete.
        
        Args:
            callback: Function that takes (provider_name, health_check) arguments
        """
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[str, HealthCheck], None]):
        """Remove a callback function."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def export_health_report(self, filepath: str, include_history: bool = False):
        """
        Export health report to a file.
        
        Args:
            filepath: Path to export file (JSON format)
            include_history: Whether to include check history
        """
        report_data = {
            'export_timestamp': time.time(),
            'summary': self.get_health_summary(),
            'current_results': {},
            'providers': []
        }
        
        # Current results
        results = self.get_all_results()
        for name, result in results.items():
            report_data['current_results'][name] = {
                'name': result.name,
                'status': result.status.value,
                'timestamp': result.timestamp,
                'duration': result.duration,
                'message': result.message,
                'details': result.details,
                'recommendations': result.recommendations,
                'age_seconds': result.age_seconds
            }
        
        # Provider information
        with self._lock:
            for name, provider in self._providers.items():
                provider_info = {
                    'name': name,
                    'check_interval': provider.get_check_interval(),
                    'is_critical': provider.is_critical()
                }
                
                if include_history:
                    history = self.get_check_history(name, limit=10)
                    provider_info['recent_history'] = [
                        {
                            'status': h.status.value,
                            'timestamp': h.timestamp,
                            'duration': h.duration,
                            'message': h.message
                        }
                        for h in history
                    ]
                
                report_data['providers'].append(provider_info)
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Exported health report to {filepath}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()