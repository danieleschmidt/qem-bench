"""
Comprehensive logging and monitoring framework for QEM-Bench.

Provides structured logging, performance monitoring, and diagnostic capabilities
for all QEM-Bench components.
"""

import logging
import logging.handlers
import json
import time
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from functools import wraps
import traceback
import sys
import os


@dataclass
class LogEntry:
    """Structured log entry for QEM-Bench."""
    
    timestamp: str
    level: str
    component: str
    operation: str
    message: str
    duration_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class QEMFormatter(logging.Formatter):
    """Custom formatter for QEM-Bench logs."""
    
    def __init__(self, include_metadata: bool = True):
        self.include_metadata = include_metadata
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        # Standard format
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        # Extract QEM-specific metadata
        component = getattr(record, 'component', 'unknown')
        operation = getattr(record, 'operation', 'unknown')
        duration_ms = getattr(record, 'duration_ms', None)
        metadata = getattr(record, 'metadata', {})
        
        # Create structured log entry
        log_entry = LogEntry(
            timestamp=timestamp,
            level=record.levelname,
            component=component,
            operation=operation,
            message=record.getMessage(),
            duration_ms=duration_ms,
            metadata=metadata if self.include_metadata else None,
            error=getattr(record, 'error', None),
            stack_trace=getattr(record, 'stack_trace', None)
        )
        
        return json.dumps(log_entry.to_dict(), default=str)


class QEMLogger:
    """Main logger class for QEM-Bench components."""
    
    def __init__(self, component: str, level: int = logging.INFO):
        self.component = component
        self.logger = logging.getLogger(f"qem_bench.{component}")
        self.logger.setLevel(level)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        self._operation_stack: List[str] = []
        self._timers: Dict[str, float] = {}
        self._metrics: Dict[str, List[float]] = {}
    
    def _setup_handlers(self):
        """Setup log handlers."""
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with JSON format
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{self.component}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(QEMFormatter())
        self.logger.addHandler(file_handler)
    
    def _create_log_record(self, level: int, message: str, operation: str = "unknown",
                          duration_ms: Optional[float] = None, 
                          metadata: Optional[Dict[str, Any]] = None,
                          error: Optional[Exception] = None):
        """Create a log record with QEM-specific fields."""
        # Get current operation from stack if not provided
        if operation == "unknown" and self._operation_stack:
            operation = self._operation_stack[-1]
        
        # Create record
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        
        # Add QEM-specific attributes
        record.component = self.component
        record.operation = operation
        record.duration_ms = duration_ms
        record.metadata = metadata or {}
        
        if error:
            record.error = str(error)
            record.stack_trace = traceback.format_exc()
        
        return record
    
    def info(self, message: str, operation: str = "unknown", 
             metadata: Optional[Dict[str, Any]] = None):
        """Log info message."""
        record = self._create_log_record(logging.INFO, message, operation, metadata=metadata)
        self.logger.handle(record)
    
    def warning(self, message: str, operation: str = "unknown", 
                metadata: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        record = self._create_log_record(logging.WARNING, message, operation, metadata=metadata)
        self.logger.handle(record)
    
    def error(self, message: str, error: Optional[Exception] = None, 
              operation: str = "unknown", metadata: Optional[Dict[str, Any]] = None):
        """Log error message."""
        record = self._create_log_record(logging.ERROR, message, operation, 
                                       metadata=metadata, error=error)
        self.logger.handle(record)
    
    def debug(self, message: str, operation: str = "unknown", 
              metadata: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        record = self._create_log_record(logging.DEBUG, message, operation, metadata=metadata)
        self.logger.handle(record)
    
    def start_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Start timing an operation."""
        self._operation_stack.append(operation)
        self._timers[operation] = time.time()
        
        self.info(f"Starting {operation}", operation, metadata)
    
    def end_operation(self, operation: str, success: bool = True, 
                     result_metadata: Optional[Dict[str, Any]] = None):
        """End timing an operation."""
        if operation in self._timers:
            duration_ms = (time.time() - self._timers[operation]) * 1000
            del self._timers[operation]
        else:
            duration_ms = None
        
        if operation in self._operation_stack:
            self._operation_stack.remove(operation)
        
        status = "completed" if success else "failed"
        message = f"Operation {operation} {status}"
        
        if success:
            record = self._create_log_record(logging.INFO, message, operation, 
                                           duration_ms, result_metadata)
        else:
            record = self._create_log_record(logging.WARNING, message, operation, 
                                           duration_ms, result_metadata)
        
        self.logger.handle(record)
        
        # Store performance metrics
        if duration_ms is not None:
            if operation not in self._metrics:
                self._metrics[operation] = []
            self._metrics[operation].append(duration_ms)
    
    def log_performance(self, operation: str, duration_ms: float, 
                       metadata: Optional[Dict[str, Any]] = None):
        """Log performance metrics."""
        perf_metadata = {"duration_ms": duration_ms, "operation": operation}
        if metadata:
            perf_metadata.update(metadata)
        
        self.info(f"Performance: {operation} took {duration_ms:.2f}ms", 
                 "performance", perf_metadata)
        
        # Store metrics
        if operation not in self._metrics:
            self._metrics[operation] = []
        self._metrics[operation].append(duration_ms)
    
    def log_quantum_circuit(self, circuit: Any, operation: str = "circuit_analysis"):
        """Log quantum circuit information."""
        metadata = {}
        
        try:
            if hasattr(circuit, 'num_qubits'):
                metadata['num_qubits'] = circuit.num_qubits
            if hasattr(circuit, 'size'):
                metadata['num_gates'] = circuit.size
            if hasattr(circuit, 'depth'):
                metadata['depth'] = circuit.depth
            if hasattr(circuit, 'name'):
                metadata['circuit_name'] = circuit.name
            
            self.info(f"Quantum circuit: {metadata.get('circuit_name', 'unnamed')}", 
                     operation, metadata)
        
        except Exception as e:
            self.error(f"Failed to log circuit information", e, operation)
    
    def log_zne_result(self, result: Any, operation: str = "zne_analysis"):
        """Log ZNE result information."""
        metadata = {}
        
        try:
            if hasattr(result, 'raw_value'):
                metadata['raw_value'] = float(result.raw_value)
            if hasattr(result, 'mitigated_value'):
                metadata['mitigated_value'] = float(result.mitigated_value)
            if hasattr(result, 'error_reduction'):
                if result.error_reduction is not None:
                    metadata['error_reduction'] = float(result.error_reduction)
            if hasattr(result, 'noise_factors'):
                metadata['num_noise_factors'] = len(result.noise_factors)
                metadata['noise_factor_range'] = [min(result.noise_factors), max(result.noise_factors)]
            if hasattr(result, 'extrapolation_method'):
                metadata['extrapolation_method'] = result.extrapolation_method
            if hasattr(result, 'fit_quality'):
                metadata['fit_quality'] = float(result.fit_quality)
            
            improvement = ""
            if 'raw_value' in metadata and 'mitigated_value' in metadata:
                diff = abs(metadata['mitigated_value'] - metadata['raw_value'])
                improvement = f" (improvement: {diff:.6f})"
            
            self.info(f"ZNE result: {metadata.get('extrapolation_method', 'unknown')} extrapolation{improvement}", 
                     operation, metadata)
        
        except Exception as e:
            self.error(f"Failed to log ZNE result information", e, operation)
    
    def get_performance_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        if operation:
            if operation not in self._metrics:
                return {}
            
            durations = self._metrics[operation]
            return {
                "operation": operation,
                "count": len(durations),
                "mean_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "total_ms": sum(durations)
            }
        else:
            stats = {}
            for op, durations in self._metrics.items():
                stats[op] = {
                    "count": len(durations),
                    "mean_ms": sum(durations) / len(durations),
                    "min_ms": min(durations),
                    "max_ms": max(durations),
                    "total_ms": sum(durations)
                }
            return stats


class MonitoredOperation:
    """Context manager for monitoring operations."""
    
    def __init__(self, logger: QEMLogger, operation: str, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.operation = operation
        self.metadata = metadata
        self.success = False
    
    def __enter__(self):
        self.logger.start_operation(self.operation, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.success = exc_type is None
        
        result_metadata = {}
        if not self.success and exc_val:
            result_metadata['error'] = str(exc_val)
            result_metadata['error_type'] = type(exc_val).__name__
        
        self.logger.end_operation(self.operation, self.success, result_metadata)
        return False  # Don't suppress exceptions


def monitored_operation(operation: str, component: str = "unknown", 
                       metadata: Optional[Dict[str, Any]] = None):
    """Decorator for monitoring function operations."""
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = QEMLogger(component)
            
            with MonitoredOperation(logger, operation, metadata):
                result = func(*args, **kwargs)
                return result
        
        return wrapper
    return decorator


class LoggingManager:
    """Global logging manager for QEM-Bench."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.loggers: Dict[str, QEMLogger] = {}
            self.initialized = True
            self._setup_global_logging()
    
    def _setup_global_logging(self):
        """Setup global logging configuration."""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set levels for external libraries
        logging.getLogger('jax').setLevel(logging.WARNING)
        logging.getLogger('numpy').setLevel(logging.WARNING)
    
    def get_logger(self, component: str) -> QEMLogger:
        """Get or create logger for component."""
        if component not in self.loggers:
            self.loggers[component] = QEMLogger(component)
        return self.loggers[component]
    
    def set_level(self, level: Union[int, str]):
        """Set logging level for all loggers."""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        for logger in self.loggers.values():
            logger.logger.setLevel(level)
    
    def get_all_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics from all loggers."""
        all_stats = {}
        for component, logger in self.loggers.items():
            stats = logger.get_performance_stats()
            if stats:
                all_stats[component] = stats
        return all_stats


# Global logging manager instance
_logging_manager = LoggingManager()


def get_logger(component: str) -> QEMLogger:
    """Get logger for component."""
    return _logging_manager.get_logger(component)


def set_global_log_level(level: Union[int, str]):
    """Set global logging level."""
    _logging_manager.set_level(level)


def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary across all components."""
    return _logging_manager.get_all_performance_stats()


# Convenience logging functions
def log_circuit_execution(circuit: Any, component: str = "simulation"):
    """Log circuit execution details."""
    logger = get_logger(component)
    logger.log_quantum_circuit(circuit, "circuit_execution")


def log_error_mitigation(method: str, result: Any, component: str = "mitigation"):
    """Log error mitigation results."""
    logger = get_logger(component)
    
    if method.lower() == "zne":
        logger.log_zne_result(result, "zne_mitigation")
    else:
        metadata = {"method": method}
        if hasattr(result, 'raw_value'):
            metadata['raw_value'] = float(result.raw_value)
        if hasattr(result, 'mitigated_value'):
            metadata['mitigated_value'] = float(result.mitigated_value)
        
        logger.info(f"Error mitigation completed using {method}", "error_mitigation", metadata)


# Example integration with existing classes
class LoggedZNE:
    """ZNE wrapper with automatic logging."""
    
    def __init__(self, zne_instance: Any):
        self.zne = zne_instance
        self.logger = get_logger("zne")
    
    def mitigate(self, *args, **kwargs):
        """Mitigate with logging."""
        with MonitoredOperation(self.logger, "zne_mitigation"):
            result = self.zne.mitigate(*args, **kwargs)
            self.logger.log_zne_result(result, "zne_mitigation")
            return result
    
    def __getattr__(self, name):
        """Delegate other attributes to wrapped instance."""
        return getattr(self.zne, name)


class LoggedSimulator:
    """Simulator wrapper with automatic logging."""
    
    def __init__(self, simulator_instance: Any):
        self.simulator = simulator_instance
        self.logger = get_logger("simulator")
    
    def run(self, circuit, *args, **kwargs):
        """Run simulation with logging."""
        self.logger.log_quantum_circuit(circuit, "simulation_preparation")
        
        with MonitoredOperation(self.logger, "circuit_simulation") as op:
            result = self.simulator.run(circuit, *args, **kwargs)
            
            # Log result metadata
            metadata = {}
            if hasattr(result, 'execution_time'):
                metadata['execution_time'] = result.execution_time
            if hasattr(result, 'measurement_counts') and result.measurement_counts:
                metadata['total_shots'] = sum(result.measurement_counts.values())
                metadata['num_outcomes'] = len(result.measurement_counts)
            
            self.logger.info("Circuit simulation completed", "circuit_simulation", metadata)
            return result
    
    def __getattr__(self, name):
        """Delegate other attributes to wrapped instance."""
        return getattr(self.simulator, name)


def add_logging_to_instance(instance: Any, component: str) -> Any:
    """Add logging capabilities to any instance."""
    if 'ZNE' in type(instance).__name__ or 'zne' in str(type(instance)).lower():
        return LoggedZNE(instance)
    elif 'Simulator' in type(instance).__name__ or 'simulator' in str(type(instance)).lower():
        return LoggedSimulator(instance)
    else:
        # Generic wrapper - add logging to all method calls
        class LoggedInstance:
            def __init__(self, wrapped, comp):
                self._wrapped = wrapped
                self._logger = get_logger(comp)
            
            def __getattr__(self, name):
                attr = getattr(self._wrapped, name)
                if callable(attr):
                    def logged_method(*args, **kwargs):
                        with MonitoredOperation(self._logger, f"{name}"):
                            return attr(*args, **kwargs)
                    return logged_method
                return attr
        
        return LoggedInstance(instance, component)


# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor performance across the entire QEM-Bench system."""
    
    def __init__(self):
        self.logger = get_logger("performance")
        self.start_time = time.time()
    
    def log_system_info(self):
        """Log system information."""
        import platform
        import psutil
        
        metadata = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3)
        }
        
        self.logger.info("System information logged", "system_info", metadata)
    
    def log_periodic_stats(self):
        """Log periodic performance statistics."""
        uptime = time.time() - self.start_time
        stats = get_performance_summary()
        
        metadata = {
            "uptime_seconds": uptime,
            "component_count": len(stats),
            "total_operations": sum(
                sum(op_stats["count"] for op_stats in comp_stats.values())
                for comp_stats in stats.values()
            )
        }
        
        self.logger.info("Periodic performance summary", "performance_summary", metadata)
        return stats


# Initialize performance monitor
_performance_monitor = PerformanceMonitor()