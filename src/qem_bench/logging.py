"""
Comprehensive logging framework for QEM-Bench.

This module provides structured logging with different levels, performance tracking,
progress monitoring, and customizable formatting for quantum error mitigation operations.
"""

import logging
import time
import json
import os
import sys
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager
from pathlib import Path
import threading
from functools import wraps
import warnings


class LogLevel(Enum):
    """Log levels for QEM-Bench operations."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Categories for different types of log messages."""
    GENERAL = "general"
    PERFORMANCE = "performance"
    VALIDATION = "validation"
    MITIGATION = "mitigation"
    NOISE = "noise"
    BACKEND = "backend"
    CIRCUIT = "circuit"
    SECURITY = "security"
    SYSTEM = "system"


@dataclass
class LogEntry:
    """Structured log entry for QEM-Bench operations."""
    timestamp: float
    level: LogLevel
    category: LogCategory
    module: str
    operation: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[float] = None
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    
    def __post_init__(self):
        """Set thread and process IDs if not provided."""
        if self.thread_id is None:
            self.thread_id = threading.get_ident()
        if self.process_id is None:
            self.process_id = os.getpid()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert log entry to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class PerformanceMetric:
    """Performance metric for operations."""
    operation: str
    module: str
    duration: float
    start_time: float
    end_time: float
    memory_delta: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class ProgressTracker:
    """Progress tracking for long-running operations."""
    
    def __init__(self, total_steps: int, operation: str, module: str):
        self.total_steps = total_steps
        self.current_step = 0
        self.operation = operation
        self.module = module
        self.start_time = time.time()
        self.step_times: List[float] = []
        self.completed = False
    
    def update(self, step: int = None, message: str = ""):
        """Update progress."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        current_time = time.time()
        self.step_times.append(current_time)
        
        progress_pct = (self.current_step / self.total_steps) * 100
        elapsed = current_time - self.start_time
        
        if self.current_step > 0:
            avg_step_time = elapsed / self.current_step
            eta = avg_step_time * (self.total_steps - self.current_step)
        else:
            eta = 0
        
        log_data = {
            "progress_pct": progress_pct,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "elapsed_time": elapsed,
            "eta": eta,
            "message": message
        }
        
        logger = get_logger()
        logger.info(
            f"Progress: {progress_pct:.1f}% ({self.current_step}/{self.total_steps})",
            category=LogCategory.GENERAL,
            module=self.module,
            operation=self.operation,
            data=log_data
        )
        
        if self.current_step >= self.total_steps:
            self.completed = True
    
    def finish(self, message: str = "Operation completed"):
        """Mark operation as finished."""
        if not self.completed:
            self.update(self.total_steps, message)


class QEMBenchLogger:
    """Main logger class for QEM-Bench operations."""
    
    def __init__(
        self,
        name: str = "qem_bench",
        level: LogLevel = LogLevel.INFO,
        handlers: Optional[List[logging.Handler]] = None,
        enable_performance_tracking: bool = True,
        log_file: Optional[str] = None,
        structured_format: bool = True
    ):
        self.name = name
        self.level = level
        self.enable_performance_tracking = enable_performance_tracking
        self.structured_format = structured_format
        
        # Create Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Add handlers
        if handlers:
            for handler in handlers:
                self.logger.addHandler(handler)
        else:
            # Default console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.value))
            
            if structured_format:
                formatter = StructuredFormatter()
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            self.add_file_handler(log_file)
        
        # Storage for metrics and entries
        self.log_entries: List[LogEntry] = []
        self.performance_metrics: List[PerformanceMetric] = []
        self.active_trackers: Dict[str, ProgressTracker] = {}
        
        # Thread lock for thread-safe logging
        self._lock = threading.Lock()
    
    def add_file_handler(
        self,
        filename: str,
        level: Optional[LogLevel] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """Add a rotating file handler."""
        from logging.handlers import RotatingFileHandler
        
        # Create directory if it doesn't exist
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            filename,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        
        if level:
            file_handler.setLevel(getattr(logging, level.value))
        else:
            file_handler.setLevel(getattr(logging, self.level.value))
        
        if self.structured_format:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log(
        self,
        level: LogLevel,
        message: str,
        category: LogCategory = LogCategory.GENERAL,
        module: str = "unknown",
        operation: str = "unknown",
        data: Optional[Dict[str, Any]] = None,
        duration: Optional[float] = None
    ):
        """Log a message with structured data."""
        with self._lock:
            entry = LogEntry(
                timestamp=time.time(),
                level=level,
                category=category,
                module=module,
                operation=operation,
                message=message,
                data=data or {},
                duration=duration
            )
            
            self.log_entries.append(entry)
            
            # Log to Python logger
            extra = {
                'category': category.value,
                'module': module,
                'operation': operation,
                'data': data or {},
                'duration': duration
            }
            
            getattr(self.logger, level.value.lower())(message, extra=extra)
    
    def debug(
        self,
        message: str,
        category: LogCategory = LogCategory.GENERAL,
        module: str = "unknown",
        operation: str = "unknown",
        data: Optional[Dict[str, Any]] = None
    ):
        """Log a debug message."""
        self.log(LogLevel.DEBUG, message, category, module, operation, data)
    
    def info(
        self,
        message: str,
        category: LogCategory = LogCategory.GENERAL,
        module: str = "unknown",
        operation: str = "unknown",
        data: Optional[Dict[str, Any]] = None
    ):
        """Log an info message."""
        self.log(LogLevel.INFO, message, category, module, operation, data)
    
    def warning(
        self,
        message: str,
        category: LogCategory = LogCategory.GENERAL,
        module: str = "unknown",
        operation: str = "unknown",
        data: Optional[Dict[str, Any]] = None
    ):
        """Log a warning message."""
        self.log(LogLevel.WARNING, message, category, module, operation, data)
    
    def error(
        self,
        message: str,
        category: LogCategory = LogCategory.GENERAL,
        module: str = "unknown",
        operation: str = "unknown",
        data: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ):
        """Log an error message."""
        self.log(LogLevel.ERROR, message, category, module, operation, data)
        if exc_info:
            self.logger.error(message, exc_info=True)
    
    def critical(
        self,
        message: str,
        category: LogCategory = LogCategory.GENERAL,
        module: str = "unknown",
        operation: str = "unknown",
        data: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ):
        """Log a critical message."""
        self.log(LogLevel.CRITICAL, message, category, module, operation, data)
        if exc_info:
            self.logger.critical(message, exc_info=True)
    
    def create_progress_tracker(
        self,
        total_steps: int,
        operation: str,
        module: str,
        tracker_id: Optional[str] = None
    ) -> ProgressTracker:
        """Create a progress tracker for long-running operations."""
        if tracker_id is None:
            tracker_id = f"{module}_{operation}_{int(time.time())}"
        
        tracker = ProgressTracker(total_steps, operation, module)
        self.active_trackers[tracker_id] = tracker
        
        self.info(
            f"Started operation: {operation}",
            category=LogCategory.GENERAL,
            module=module,
            operation=operation,
            data={"total_steps": total_steps, "tracker_id": tracker_id}
        )
        
        return tracker
    
    def remove_progress_tracker(self, tracker_id: str):
        """Remove a completed progress tracker."""
        if tracker_id in self.active_trackers:
            del self.active_trackers[tracker_id]
    
    @contextmanager
    def performance_timer(
        self,
        operation: str,
        module: str,
        parameters: Optional[Dict[str, Any]] = None,
        track_memory: bool = False
    ):
        """Context manager for measuring operation performance."""
        if not self.enable_performance_tracking:
            yield
            return
        
        import psutil
        process = psutil.Process() if track_memory else None
        start_memory = process.memory_info().rss if process else None
        
        start_time = time.time()
        error = None
        success = True
        
        self.debug(
            f"Starting operation: {operation}",
            category=LogCategory.PERFORMANCE,
            module=module,
            operation=operation,
            data=parameters or {}
        )
        
        try:
            yield
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            end_memory = process.memory_info().rss if process else None
            memory_delta = (end_memory - start_memory) if (start_memory and end_memory) else None
            
            metric = PerformanceMetric(
                operation=operation,
                module=module,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                memory_delta=memory_delta,
                parameters=parameters or {},
                success=success,
                error=error
            )
            
            with self._lock:
                self.performance_metrics.append(metric)
            
            log_data = {
                "duration": duration,
                "success": success,
                "memory_delta_mb": memory_delta / (1024 * 1024) if memory_delta else None,
                "parameters": parameters or {}
            }
            
            if error:
                log_data["error"] = error
            
            level = LogLevel.INFO if success else LogLevel.ERROR
            self.log(
                level,
                f"Completed operation: {operation} ({duration:.3f}s)",
                category=LogCategory.PERFORMANCE,
                module=module,
                operation=operation,
                data=log_data,
                duration=duration
            )
    
    def get_performance_summary(
        self,
        module: Optional[str] = None,
        operation: Optional[str] = None,
        time_window: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get performance summary statistics."""
        metrics = self.performance_metrics
        
        # Filter by criteria
        if module:
            metrics = [m for m in metrics if m.module == module]
        if operation:
            metrics = [m for m in metrics if m.operation == operation]
        if time_window:
            cutoff = time.time() - time_window
            metrics = [m for m in metrics if m.start_time >= cutoff]
        
        if not metrics:
            return {"total_operations": 0}
        
        durations = [m.duration for m in metrics]
        successful = [m for m in metrics if m.success]
        failed = [m for m in metrics if not m.success]
        
        summary = {
            "total_operations": len(metrics),
            "successful_operations": len(successful),
            "failed_operations": len(failed),
            "success_rate": len(successful) / len(metrics),
            "total_time": sum(durations),
            "average_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "operations_by_module": {},
            "operations_by_type": {}
        }
        
        # Group by module and operation
        for metric in metrics:
            module_name = metric.module
            if module_name not in summary["operations_by_module"]:
                summary["operations_by_module"][module_name] = {
                    "count": 0,
                    "total_time": 0,
                    "success_rate": 0
                }
            
            summary["operations_by_module"][module_name]["count"] += 1
            summary["operations_by_module"][module_name]["total_time"] += metric.duration
            
            operation_name = metric.operation
            if operation_name not in summary["operations_by_type"]:
                summary["operations_by_type"][operation_name] = {
                    "count": 0,
                    "total_time": 0,
                    "success_rate": 0
                }
            
            summary["operations_by_type"][operation_name]["count"] += 1
            summary["operations_by_type"][operation_name]["total_time"] += metric.duration
        
        # Calculate success rates
        for module_name, stats in summary["operations_by_module"].items():
            module_metrics = [m for m in metrics if m.module == module_name]
            successful_module = [m for m in module_metrics if m.success]
            stats["success_rate"] = len(successful_module) / len(module_metrics)
        
        for operation_name, stats in summary["operations_by_type"].items():
            operation_metrics = [m for m in metrics if m.operation == operation_name]
            successful_operation = [m for m in operation_metrics if m.success]
            stats["success_rate"] = len(successful_operation) / len(operation_metrics)
        
        return summary
    
    def export_logs(
        self,
        filename: str,
        format: str = "json",
        level_filter: Optional[LogLevel] = None,
        category_filter: Optional[LogCategory] = None,
        time_window: Optional[float] = None
    ):
        """Export logs to file."""
        entries = self.log_entries
        
        # Apply filters
        if level_filter:
            entries = [e for e in entries if e.level == level_filter]
        if category_filter:
            entries = [e for e in entries if e.category == category_filter]
        if time_window:
            cutoff = time.time() - time_window
            entries = [e for e in entries if e.timestamp >= cutoff]
        
        # Create directory if needed
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            if format.lower() == "json":
                json.dump([entry.to_dict() for entry in entries], f, indent=2, default=str)
            elif format.lower() == "csv":
                import csv
                if entries:
                    fieldnames = entries[0].to_dict().keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for entry in entries:
                        writer.writerow(entry.to_dict())
            else:
                # Plain text format
                for entry in entries:
                    f.write(f"{entry.timestamp} - {entry.level.value} - {entry.module} - {entry.message}\n")
    
    def clear_logs(self, keep_recent: Optional[float] = None):
        """Clear stored logs, optionally keeping recent entries."""
        with self._lock:
            if keep_recent:
                cutoff = time.time() - keep_recent
                self.log_entries = [e for e in self.log_entries if e.timestamp >= cutoff]
                self.performance_metrics = [m for m in self.performance_metrics if m.start_time >= cutoff]
            else:
                self.log_entries.clear()
                self.performance_metrics.clear()


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record):
        """Format log record with structured data."""
        log_data = {
            "timestamp": record.created,
            "level": record.levelname,
            "module": getattr(record, 'module', 'unknown'),
            "operation": getattr(record, 'operation', 'unknown'),
            "category": getattr(record, 'category', 'general'),
            "message": record.getMessage(),
        }
        
        # Add extra data if present
        extra_data = getattr(record, 'data', {})
        if extra_data:
            log_data["data"] = extra_data
        
        duration = getattr(record, 'duration', None)
        if duration:
            log_data["duration"] = duration
        
        return json.dumps(log_data, default=str)


# Global logger instance
_global_logger: Optional[QEMBenchLogger] = None
_logger_lock = threading.Lock()


def get_logger() -> QEMBenchLogger:
    """Get or create the global logger instance."""
    global _global_logger
    
    if _global_logger is None:
        with _logger_lock:
            if _global_logger is None:
                _global_logger = QEMBenchLogger()
    
    return _global_logger


def configure_logging(
    level: Union[str, LogLevel] = LogLevel.INFO,
    log_file: Optional[str] = None,
    structured_format: bool = True,
    enable_performance_tracking: bool = True,
    **kwargs
) -> QEMBenchLogger:
    """
    Configure the global logger.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        structured_format: Whether to use structured JSON format
        enable_performance_tracking: Whether to track performance metrics
        **kwargs: Additional arguments for QEMBenchLogger
        
    Returns:
        Configured logger instance
    """
    global _global_logger
    
    if isinstance(level, str):
        level = LogLevel(level.upper())
    
    with _logger_lock:
        _global_logger = QEMBenchLogger(
            level=level,
            log_file=log_file,
            structured_format=structured_format,
            enable_performance_tracking=enable_performance_tracking,
            **kwargs
        )
    
    return _global_logger


def reset_logging():
    """Reset the global logger instance."""
    global _global_logger
    with _logger_lock:
        _global_logger = None


# Logging decorators
def log_operation(
    operation: str,
    module: str,
    category: LogCategory = LogCategory.GENERAL,
    level: LogLevel = LogLevel.INFO,
    log_parameters: bool = False,
    log_result: bool = False,
    track_performance: bool = True
):
    """
    Decorator for logging function operations.
    
    Args:
        operation: Name of the operation
        module: Module name
        category: Log category
        level: Log level
        log_parameters: Whether to log function parameters
        log_result: Whether to log function result
        track_performance: Whether to track performance metrics
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            
            # Prepare parameter data
            param_data = {}
            if log_parameters:
                param_data = {
                    "args": [str(arg) for arg in args],
                    "kwargs": {k: str(v) for k, v in kwargs.items()}
                }
            
            # Log operation start
            logger.log(
                level,
                f"Starting {operation}",
                category=category,
                module=module,
                operation=operation,
                data=param_data
            )
            
            # Execute with performance tracking if enabled
            if track_performance:
                with logger.performance_timer(operation, module, param_data):
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Log result if requested
            if log_result:
                result_data = {"result": str(result)}
                logger.log(
                    level,
                    f"Completed {operation}",
                    category=category,
                    module=module,
                    operation=operation,
                    data=result_data
                )
            
            return result
        
        return wrapper
    return decorator


def log_errors(
    operation: str,
    module: str,
    category: LogCategory = LogCategory.GENERAL,
    reraise: bool = True
):
    """
    Decorator for logging errors in operations.
    
    Args:
        operation: Name of the operation
        module: Module name
        category: Log category
        reraise: Whether to re-raise the exception
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_data = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "args": [str(arg) for arg in args],
                    "kwargs": {k: str(v) for k, v in kwargs.items()}
                }
                
                logger.error(
                    f"Error in {operation}: {str(e)}",
                    category=category,
                    module=module,
                    operation=operation,
                    data=error_data,
                    exc_info=True
                )
                
                if reraise:
                    raise
                
                return None
        
        return wrapper
    return decorator


# Convenience functions
def debug(message: str, **kwargs):
    """Log a debug message."""
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs):
    """Log an info message."""
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs):
    """Log a warning message."""
    get_logger().warning(message, **kwargs)


def error(message: str, **kwargs):
    """Log an error message."""
    get_logger().error(message, **kwargs)


def critical(message: str, **kwargs):
    """Log a critical message."""
    get_logger().critical(message, **kwargs)


@contextmanager
def performance_timer(operation: str, module: str, **kwargs):
    """Context manager for performance timing."""
    with get_logger().performance_timer(operation, module, **kwargs):
        yield


def create_progress_tracker(total_steps: int, operation: str, module: str, **kwargs) -> ProgressTracker:
    """Create a progress tracker."""
    return get_logger().create_progress_tracker(total_steps, operation, module, **kwargs)


# Setup default configuration
def setup_default_logging():
    """Setup default logging configuration."""
    # Check for environment variables
    log_level = os.environ.get("QEM_BENCH_LOG_LEVEL", "INFO")
    log_file = os.environ.get("QEM_BENCH_LOG_FILE")
    structured = os.environ.get("QEM_BENCH_STRUCTURED_LOGS", "true").lower() == "true"
    
    configure_logging(
        level=log_level,
        log_file=log_file,
        structured_format=structured
    )


# Initialize default logging
setup_default_logging()


# Export all logging functionality
__all__ = [
    # Enums
    "LogLevel",
    "LogCategory",
    
    # Data classes
    "LogEntry",
    "PerformanceMetric",
    "ProgressTracker",
    
    # Main classes
    "QEMBenchLogger",
    "StructuredFormatter",
    
    # Global functions
    "get_logger",
    "configure_logging",
    "reset_logging",
    "setup_default_logging",
    
    # Decorators
    "log_operation",
    "log_errors",
    
    # Convenience functions
    "debug",
    "info", 
    "warning",
    "error",
    "critical",
    "performance_timer",
    "create_progress_tracker"
]