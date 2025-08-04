"""Performance monitoring and profiling for QEM-Bench operations."""

import time
import functools
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import json
import statistics


logger = logging.getLogger(__name__)


@dataclass
class TimingRecord:
    """Record of a timed operation."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    thread_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration * 1000


@dataclass
class PerformanceStats:
    """Performance statistics for an operation."""
    operation_name: str
    count: int
    total_duration: float
    avg_duration: float
    min_duration: float
    max_duration: float
    std_deviation: float
    p50_duration: float
    p95_duration: float
    p99_duration: float
    
    @property
    def avg_duration_ms(self) -> float:
        """Average duration in milliseconds."""
        return self.avg_duration * 1000
    
    @property
    def throughput_per_second(self) -> float:
        """Operations per second based on average duration."""
        return 1.0 / self.avg_duration if self.avg_duration > 0 else 0.0


@dataclass
class PerformanceMonitorConfig:
    """Configuration for performance monitoring."""
    enabled: bool = True
    max_records_per_operation: int = 1000
    enable_profiling: bool = False
    profile_memory: bool = False
    auto_export_interval: Optional[float] = None  # seconds
    export_directory: Optional[str] = None


class PerformanceMonitor:
    """
    Performance monitor for timing and profiling QEM-Bench operations.
    
    This monitor tracks execution times, provides statistical analysis,
    and can identify performance bottlenecks in quantum error mitigation
    experiments.
    
    Example:
        >>> monitor = PerformanceMonitor()
        >>> 
        >>> # Using context manager
        >>> with monitor.time_operation("zne_execution"):
        ...     zne.mitigate(circuit, backend)
        >>> 
        >>> # Using decorator
        >>> @monitor.profile("custom_function")
        >>> def my_function():
        ...     return some_computation()
        >>> 
        >>> stats = monitor.get_stats("zne_execution")
        >>> print(f"Average time: {stats.avg_duration_ms:.2f}ms")
    """
    
    def __init__(self, config: Optional[PerformanceMonitorConfig] = None):
        self.config = config or PerformanceMonitorConfig()
        self._records: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.max_records_per_operation)
        )
        self._lock = threading.Lock()
        self._active_operations: Dict[int, List[str]] = defaultdict(list)
        
        # Auto-export setup
        self._export_timer: Optional[threading.Timer] = None
        if self.config.auto_export_interval and self.config.export_directory:
            self._start_auto_export()
    
    def _start_auto_export(self):
        """Start automatic export of performance data."""
        if not self.config.enabled:
            return
            
        def auto_export():
            try:
                timestamp = int(time.time())
                filepath = f"{self.config.export_directory}/performance_{timestamp}.json"
                self.export_all_stats(filepath)
            except Exception as e:
                logger.error(f"Auto-export failed: {e}")
            finally:
                # Schedule next export
                self._export_timer = threading.Timer(
                    self.config.auto_export_interval, auto_export
                )
                self._export_timer.daemon = True
                self._export_timer.start()
        
        self._export_timer = threading.Timer(
            self.config.auto_export_interval, auto_export
        )
        self._export_timer.daemon = True
        self._export_timer.start()
    
    @contextmanager
    def time_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for timing an operation.
        
        Args:
            operation_name: Name of the operation being timed
            metadata: Additional metadata to store with the timing record
        
        Example:
            >>> with monitor.time_operation("circuit_execution", {"shots": 1024}):
            ...     result = backend.run(circuit, shots=1024)
        """
        if not self.config.enabled:
            yield
            return
        
        start_time = time.perf_counter()
        thread_id = threading.get_ident()
        
        # Track nested operations
        with self._lock:
            self._active_operations[thread_id].append(operation_name)
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            record = TimingRecord(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                thread_id=thread_id,
                metadata=metadata or {}
            )
            
            with self._lock:
                self._records[operation_name].append(record)
                if thread_id in self._active_operations:
                    self._active_operations[thread_id].pop()
    
    def profile(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Decorator for profiling function execution.
        
        Args:
            operation_name: Name of the operation being profiled
            metadata: Additional metadata to store with timing records
        
        Example:
            >>> @monitor.profile("zne_mitigation")
            >>> def run_zne(circuit, backend):
            ...     return zne.mitigate(circuit, backend)
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                func_metadata = metadata.copy() if metadata else {}
                func_metadata.update({
                    'function_name': func.__name__,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                })
                
                with self.time_operation(operation_name, func_metadata):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def record_duration(self, operation_name: str, duration: float, 
                       metadata: Optional[Dict[str, Any]] = None):
        """
        Manually record an operation duration.
        
        Args:
            operation_name: Name of the operation
            duration: Duration in seconds
            metadata: Additional metadata
        """
        if not self.config.enabled:
            return
        
        current_time = time.perf_counter()
        record = TimingRecord(
            operation_name=operation_name,
            start_time=current_time - duration,
            end_time=current_time,
            duration=duration,
            thread_id=threading.get_ident(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self._records[operation_name].append(record)
    
    def get_records(self, operation_name: str, 
                   duration_seconds: Optional[float] = None) -> List[TimingRecord]:
        """
        Get timing records for an operation.
        
        Args:
            operation_name: Name of the operation
            duration_seconds: If specified, only return records from this many
                            seconds ago
        
        Returns:
            List of TimingRecord objects
        """
        with self._lock:
            records = list(self._records[operation_name])
        
        if duration_seconds is None:
            return records
        
        cutoff_time = time.perf_counter() - duration_seconds
        return [r for r in records if r.end_time >= cutoff_time]
    
    def get_stats(self, operation_name: str, 
                 duration_seconds: Optional[float] = None) -> Optional[PerformanceStats]:
        """
        Get performance statistics for an operation.
        
        Args:
            operation_name: Name of the operation
            duration_seconds: If specified, only analyze records from this many
                            seconds ago
        
        Returns:
            PerformanceStats object or None if no records exist
        """
        records = self.get_records(operation_name, duration_seconds)
        if not records:
            return None
        
        durations = [r.duration for r in records]
        
        return PerformanceStats(
            operation_name=operation_name,
            count=len(durations),
            total_duration=sum(durations),
            avg_duration=statistics.mean(durations),
            min_duration=min(durations),
            max_duration=max(durations),
            std_deviation=statistics.stdev(durations) if len(durations) > 1 else 0.0,
            p50_duration=statistics.median(durations),
            p95_duration=self._percentile(durations, 0.95),
            p99_duration=self._percentile(durations, 0.99)
        )
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * percentile
        f = int(k)
        c = k - f
        
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        else:
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    
    def get_all_operation_names(self) -> List[str]:
        """Get names of all monitored operations."""
        with self._lock:
            return list(self._records.keys())
    
    def get_current_operations(self) -> Dict[int, List[str]]:
        """Get currently active operations by thread ID."""
        with self._lock:
            return dict(self._active_operations)
    
    def clear_records(self, operation_name: Optional[str] = None):
        """
        Clear timing records.
        
        Args:
            operation_name: If specified, clear only records for this operation.
                          If None, clear all records.
        """
        with self._lock:
            if operation_name:
                if operation_name in self._records:
                    self._records[operation_name].clear()
            else:
                self._records.clear()
    
    def get_summary_report(self, duration_seconds: Optional[float] = None) -> str:
        """
        Generate a text summary report of performance statistics.
        
        Args:
            duration_seconds: If specified, only analyze recent records
        
        Returns:
            Formatted text report
        """
        operation_names = self.get_all_operation_names()
        if not operation_names:
            return "No performance data available."
        
        report_lines = ["Performance Monitor Summary", "=" * 30, ""]
        
        for op_name in sorted(operation_names):
            stats = self.get_stats(op_name, duration_seconds)
            if not stats:
                continue
            
            report_lines.extend([
                f"Operation: {op_name}",
                f"  Count: {stats.count}",
                f"  Average: {stats.avg_duration_ms:.2f}ms",
                f"  Min/Max: {stats.min_duration*1000:.2f}ms / {stats.max_duration*1000:.2f}ms",
                f"  P50/P95/P99: {stats.p50_duration*1000:.2f}ms / {stats.p95_duration*1000:.2f}ms / {stats.p99_duration*1000:.2f}ms",
                f"  Throughput: {stats.throughput_per_second:.2f} ops/sec",
                ""
            ])
        
        # Add system information
        current_ops = self.get_current_operations()
        if current_ops:
            report_lines.extend([
                "Currently Active Operations:",
                "----------------------------"
            ])
            for thread_id, ops in current_ops.items():
                if ops:
                    report_lines.append(f"  Thread {thread_id}: {' -> '.join(ops)}")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def export_stats(self, operation_name: str, filepath: str, 
                    duration_seconds: Optional[float] = None):
        """
        Export performance statistics for an operation to a file.
        
        Args:
            operation_name: Name of the operation
            filepath: Path to export file (JSON format)
            duration_seconds: If specified, only export recent records
        """
        stats = self.get_stats(operation_name, duration_seconds)
        records = self.get_records(operation_name, duration_seconds)
        
        export_data = {
            'operation_name': operation_name,
            'export_timestamp': time.time(),
            'duration_filter_seconds': duration_seconds,
            'statistics': {
                'count': stats.count if stats else 0,
                'total_duration': stats.total_duration if stats else 0,
                'avg_duration': stats.avg_duration if stats else 0,
                'min_duration': stats.min_duration if stats else 0,
                'max_duration': stats.max_duration if stats else 0,
                'std_deviation': stats.std_deviation if stats else 0,
                'p50_duration': stats.p50_duration if stats else 0,
                'p95_duration': stats.p95_duration if stats else 0,
                'p99_duration': stats.p99_duration if stats else 0,
                'throughput_per_second': stats.throughput_per_second if stats else 0
            },
            'records': []
        }
        
        for record in records:
            record_data = {
                'start_time': record.start_time,
                'end_time': record.end_time,
                'duration': record.duration,
                'thread_id': record.thread_id,
                'metadata': record.metadata
            }
            export_data['records'].append(record_data)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported performance data for '{operation_name}' to {filepath}")
    
    def export_all_stats(self, filepath: str, duration_seconds: Optional[float] = None):
        """
        Export all performance statistics to a file.
        
        Args:
            filepath: Path to export file (JSON format) 
            duration_seconds: If specified, only export recent records
        """
        operation_names = self.get_all_operation_names()
        
        export_data = {
            'export_timestamp': time.time(),
            'duration_filter_seconds': duration_seconds,
            'operations': {}
        }
        
        for op_name in operation_names:
            stats = self.get_stats(op_name, duration_seconds)
            records = self.get_records(op_name, duration_seconds)
            
            operation_data = {
                'statistics': {
                    'count': stats.count if stats else 0,
                    'total_duration': stats.total_duration if stats else 0,
                    'avg_duration': stats.avg_duration if stats else 0,
                    'min_duration': stats.min_duration if stats else 0,
                    'max_duration': stats.max_duration if stats else 0,
                    'std_deviation': stats.std_deviation if stats else 0,
                    'p50_duration': stats.p50_duration if stats else 0,
                    'p95_duration': stats.p95_duration if stats else 0,
                    'p99_duration': stats.p99_duration if stats else 0,
                    'throughput_per_second': stats.throughput_per_second if stats else 0
                },
                'sample_records': records[-10:]  # Last 10 records as samples
            }
            
            # Convert TimingRecord objects to dictionaries
            operation_data['sample_records'] = [
                {
                    'start_time': r.start_time,
                    'end_time': r.end_time,
                    'duration': r.duration,
                    'thread_id': r.thread_id,
                    'metadata': r.metadata
                }
                for r in operation_data['sample_records']
            ]
            
            export_data['operations'][op_name] = operation_data
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported all performance data to {filepath}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._export_timer:
            self._export_timer.cancel()