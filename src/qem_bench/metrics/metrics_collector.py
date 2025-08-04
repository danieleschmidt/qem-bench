"""Core metrics collection system for QEM-Bench."""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import statistics


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


@dataclass
class MetricRecord:
    """A single metric measurement record."""
    name: str
    value: Union[float, int]
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_seconds(self) -> float:
        """Age of the metric record in seconds."""
        return time.time() - self.timestamp


@dataclass
class MetricSummary:
    """Summary statistics for a metric over time."""
    name: str
    metric_type: MetricType
    count: int
    sum_value: float
    min_value: float
    max_value: float
    avg_value: float
    std_deviation: float
    p50_value: float
    p95_value: float
    p99_value: float
    first_timestamp: float
    last_timestamp: float
    
    @property
    def duration_seconds(self) -> float:
        """Duration between first and last measurement."""
        return self.last_timestamp - self.first_timestamp
    
    @property
    def rate_per_second(self) -> float:
        """Rate of measurements per second."""
        if self.duration_seconds > 0:
            return self.count / self.duration_seconds
        return 0.0


@dataclass
class MetricsCollectorConfig:
    """Configuration for metrics collection."""
    enabled: bool = True
    max_records_per_metric: int = 10000
    retention_seconds: float = 3600  # 1 hour
    auto_cleanup_interval: float = 300  # 5 minutes
    enable_histograms: bool = True
    histogram_buckets: List[float] = field(default_factory=lambda: [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0
    ])


class MetricsCollector:
    """
    Comprehensive metrics collection system for QEM-Bench.
    
    Collects, stores, and analyzes various types of metrics from quantum
    error mitigation experiments. Supports counters, gauges, histograms,
    summaries, and timers with flexible labeling and metadata.
    
    Example:
        >>> collector = MetricsCollector()
        >>> 
        >>> # Record different types of metrics
        >>> collector.increment_counter("experiments_run", labels={"method": "zne"})
        >>> collector.set_gauge("current_fidelity", 0.95, labels={"circuit": "ghz"})
        >>> collector.record_histogram("execution_time", 2.5, labels={"backend": "ibm_perth"})
        >>> 
        >>> # Get metric summaries
        >>> summary = collector.get_summary("execution_time")
        >>> print(f"Average execution time: {summary.avg_value:.2f}s")
    """
    
    def __init__(self, config: Optional[MetricsCollectorConfig] = None):
        self.config = config or MetricsCollectorConfig()
        self._metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.max_records_per_metric)
        )
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[MetricRecord], None]] = []
        
        # Auto-cleanup timer
        self._cleanup_timer: Optional[threading.Timer] = None
        if self.config.auto_cleanup_interval > 0:
            self._start_cleanup_timer()
    
    def _start_cleanup_timer(self):
        """Start automatic cleanup of old metrics."""
        def cleanup():
            try:
                self.cleanup_old_metrics()
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
            finally:
                # Schedule next cleanup
                if self.config.auto_cleanup_interval > 0:
                    self._cleanup_timer = threading.Timer(
                        self.config.auto_cleanup_interval, cleanup
                    )
                    self._cleanup_timer.daemon = True
                    self._cleanup_timer.start()
        
        self._cleanup_timer = threading.Timer(
            self.config.auto_cleanup_interval, cleanup
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
    
    def increment_counter(self, name: str, value: float = 1.0, 
                         labels: Optional[Dict[str, str]] = None,
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Increment a counter metric.
        
        Args:
            name: Name of the counter
            value: Amount to increment by
            labels: Labels for the metric
            metadata: Additional metadata
        """
        if not self.config.enabled:
            return
        
        labels = labels or {}
        label_key = self._make_label_key(labels)
        full_name = f"{name}:{label_key}" if label_key else name
        
        with self._lock:
            self._counters[full_name] += value
        
        self._record_metric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            labels=labels,
            metadata=metadata
        )
    
    def set_gauge(self, name: str, value: Union[float, int],
                  labels: Optional[Dict[str, str]] = None,
                  metadata: Optional[Dict[str, Any]] = None):
        """
        Set a gauge metric value.
        
        Args:
            name: Name of the gauge
            value: Value to set
            labels: Labels for the metric
            metadata: Additional metadata
        """
        if not self.config.enabled:
            return
        
        labels = labels or {}
        label_key = self._make_label_key(labels)
        full_name = f"{name}:{label_key}" if label_key else name
        
        with self._lock:
            self._gauges[full_name] = float(value)
        
        self._record_metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels,
            metadata=metadata
        )
    
    def record_histogram(self, name: str, value: Union[float, int],
                        labels: Optional[Dict[str, str]] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        """
        Record a value in a histogram metric.
        
        Args:
            name: Name of the histogram
            value: Value to record
            labels: Labels for the metric
            metadata: Additional metadata
        """
        if not self.config.enabled:
            return
        
        self._record_metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            labels=labels or {},
            metadata=metadata
        )
    
    def record_timer(self, name: str, duration: float,
                    labels: Optional[Dict[str, str]] = None,
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Record a timer metric.
        
        Args:
            name: Name of the timer
            duration: Duration in seconds
            labels: Labels for the metric
            metadata: Additional metadata
        """
        if not self.config.enabled:
            return
        
        self._record_metric(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            labels=labels or {},
            metadata=metadata
        )
    
    def record_summary(self, name: str, value: Union[float, int],
                      labels: Optional[Dict[str, str]] = None,
                      metadata: Optional[Dict[str, Any]] = None):
        """
        Record a value in a summary metric.
        
        Args:
            name: Name of the summary
            value: Value to record
            labels: Labels for the metric
            metadata: Additional metadata
        """
        if not self.config.enabled:
            return
        
        self._record_metric(
            name=name,
            value=value,
            metric_type=MetricType.SUMMARY,
            labels=labels or {},
            metadata=metadata
        )
    
    def _record_metric(self, name: str, value: Union[float, int], 
                      metric_type: MetricType,
                      labels: Dict[str, str],
                      metadata: Optional[Dict[str, Any]] = None):
        """Internal method to record a metric."""
        record = MetricRecord(
            name=name,
            value=float(value),
            metric_type=metric_type,
            timestamp=time.time(),
            labels=labels,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._metrics[name].append(record)
        
        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback(record)
            except Exception as e:
                logger.error(f"Metrics callback failed: {e}")
    
    def _make_label_key(self, labels: Dict[str, str]) -> str:
        """Create a consistent key from labels."""
        if not labels:
            return ""
        
        # Sort labels for consistent keys
        sorted_items = sorted(labels.items())
        return ",".join(f"{k}={v}" for k, v in sorted_items)
    
    def get_records(self, name: str, 
                   duration_seconds: Optional[float] = None,
                   labels: Optional[Dict[str, str]] = None) -> List[MetricRecord]:
        """
        Get metric records for a specific metric.
        
        Args:
            name: Name of the metric
            duration_seconds: Only return records from this many seconds ago
            labels: Filter by labels
        
        Returns:
            List of MetricRecord objects
        """
        with self._lock:
            records = list(self._metrics[name])
        
        # Filter by time
        if duration_seconds is not None:
            cutoff_time = time.time() - duration_seconds
            records = [r for r in records if r.timestamp >= cutoff_time]
        
        # Filter by labels
        if labels:
            records = [
                r for r in records
                if all(r.labels.get(k) == v for k, v in labels.items())
            ]
        
        return records
    
    def get_summary(self, name: str, 
                   duration_seconds: Optional[float] = None,
                   labels: Optional[Dict[str, str]] = None) -> Optional[MetricSummary]:
        """
        Get summary statistics for a metric.
        
        Args:
            name: Name of the metric
            duration_seconds: Only analyze records from this many seconds ago
            labels: Filter by labels
        
        Returns:
            MetricSummary object or None if no records exist
        """
        records = self.get_records(name, duration_seconds, labels)
        if not records:
            return None
        
        values = [r.value for r in records]
        timestamps = [r.timestamp for r in records]
        
        return MetricSummary(
            name=name,
            metric_type=records[0].metric_type,  # Assume all records have same type
            count=len(values),
            sum_value=sum(values),
            min_value=min(values),
            max_value=max(values),
            avg_value=statistics.mean(values),
            std_deviation=statistics.stdev(values) if len(values) > 1 else 0.0,
            p50_value=statistics.median(values),
            p95_value=self._percentile(values, 0.95),
            p99_value=self._percentile(values, 0.99),
            first_timestamp=min(timestamps),
            last_timestamp=max(timestamps)
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
    
    def get_all_metric_names(self) -> List[str]:
        """Get names of all metrics with recorded data."""
        with self._lock:
            return list(self._metrics.keys())
    
    def get_counter_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """
        Get current value of a counter.
        
        Args:
            name: Counter name
            labels: Labels to match
        
        Returns:
            Current counter value
        """
        labels = labels or {}
        label_key = self._make_label_key(labels)
        full_name = f"{name}:{label_key}" if label_key else name
        
        with self._lock:
            return self._counters.get(full_name, 0.0)
    
    def get_gauge_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """
        Get current value of a gauge.
        
        Args:
            name: Gauge name
            labels: Labels to match
        
        Returns:
            Current gauge value or None if not set
        """
        labels = labels or {}
        label_key = self._make_label_key(labels)
        full_name = f"{name}:{label_key}" if label_key else name
        
        with self._lock:
            return self._gauges.get(full_name)
    
    def get_histogram_buckets(self, name: str, 
                             duration_seconds: Optional[float] = None,
                             labels: Optional[Dict[str, str]] = None) -> Dict[float, int]:
        """
        Get histogram bucket counts.
        
        Args:
            name: Histogram name
            duration_seconds: Time window for analysis
            labels: Labels to filter by
        
        Returns:
            Dictionary mapping bucket boundaries to counts
        """
        if not self.config.enable_histograms:
            return {}
        
        records = self.get_records(name, duration_seconds, labels)
        histogram_records = [r for r in records if r.metric_type == MetricType.HISTOGRAM]
        
        if not histogram_records:
            return {}
        
        values = [r.value for r in histogram_records]
        buckets = {}
        
        for bucket in self.config.histogram_buckets:
            count = sum(1 for v in values if v <= bucket)
            buckets[bucket] = count
        
        # Add infinity bucket
        buckets[float('inf')] = len(values)
        
        return buckets
    
    def cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        if self.config.retention_seconds <= 0:
            return
        
        cutoff_time = time.time() - self.config.retention_seconds
        metrics_cleaned = 0
        
        with self._lock:
            for metric_name, records in self._metrics.items():
                original_count = len(records)
                # Remove old records (deque doesn't support in-place filtering easily)
                fresh_records = [r for r in records if r.timestamp >= cutoff_time]
                records.clear()
                records.extend(fresh_records)
                metrics_cleaned += original_count - len(records)
        
        if metrics_cleaned > 0:
            logger.debug(f"Cleaned up {metrics_cleaned} old metric records")
    
    def reset_metric(self, name: str):
        """Reset all data for a specific metric."""
        with self._lock:
            if name in self._metrics:
                self._metrics[name].clear()
            
            # Reset counters and gauges with this name
            keys_to_remove = [k for k in self._counters.keys() if k.startswith(f"{name}:") or k == name]
            for key in keys_to_remove:
                del self._counters[key]
            
            keys_to_remove = [k for k in self._gauges.keys() if k.startswith(f"{name}:") or k == name]
            for key in keys_to_remove:
                del self._gauges[key]
    
    def reset_all_metrics(self):
        """Reset all collected metrics."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
    
    def get_metrics_overview(self) -> Dict[str, Any]:
        """Get an overview of all collected metrics."""
        metric_names = self.get_all_metric_names()
        overview = {
            'total_metrics': len(metric_names),
            'total_records': 0,
            'metrics_by_type': defaultdict(int),
            'oldest_record': None,
            'newest_record': None
        }
        
        all_timestamps = []
        
        for metric_name in metric_names:
            records = self.get_records(metric_name)
            overview['total_records'] += len(records)
            
            if records:
                metric_type = records[0].metric_type.value
                overview['metrics_by_type'][metric_type] += 1
                
                timestamps = [r.timestamp for r in records]
                all_timestamps.extend(timestamps)
        
        if all_timestamps:
            overview['oldest_record'] = min(all_timestamps)
            overview['newest_record'] = max(all_timestamps)
            overview['data_span_seconds'] = max(all_timestamps) - min(all_timestamps)
        
        overview['metrics_by_type'] = dict(overview['metrics_by_type'])
        return overview
    
    def add_callback(self, callback: Callable[[MetricRecord], None]):
        """Add a callback function to be called when metrics are recorded."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[MetricRecord], None]):
        """Remove a callback function."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def export_metrics(self, filepath: str, format: str = "json",
                      duration_seconds: Optional[float] = None):
        """
        Export metrics to a file.
        
        Args:
            filepath: Path to export file
            format: Export format ("json", "csv", "prometheus")
            duration_seconds: Only export recent metrics
        """
        if format.lower() == "json":
            self._export_json(filepath, duration_seconds)
        elif format.lower() == "csv":
            self._export_csv(filepath, duration_seconds)
        elif format.lower() == "prometheus":
            self._export_prometheus(filepath, duration_seconds)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, filepath: str, duration_seconds: Optional[float]):
        """Export metrics to JSON format."""
        export_data = {
            'export_timestamp': time.time(),
            'duration_filter_seconds': duration_seconds,
            'overview': self.get_metrics_overview(),
            'metrics': {}
        }
        
        for metric_name in self.get_all_metric_names():
            records = self.get_records(metric_name, duration_seconds)
            summary = self.get_summary(metric_name, duration_seconds)
            
            metric_data = {
                'summary': {
                    'count': summary.count if summary else 0,
                    'sum': summary.sum_value if summary else 0,
                    'avg': summary.avg_value if summary else 0,
                    'min': summary.min_value if summary else 0,
                    'max': summary.max_value if summary else 0,
                    'std_dev': summary.std_deviation if summary else 0,
                    'p50': summary.p50_value if summary else 0,
                    'p95': summary.p95_value if summary else 0,
                    'p99': summary.p99_value if summary else 0
                },
                'recent_records': []
            }
            
            # Include recent records (limit to avoid huge files)
            recent_records = records[-100:] if len(records) > 100 else records
            for record in recent_records:
                record_data = {
                    'timestamp': record.timestamp,
                    'value': record.value,
                    'type': record.metric_type.value,
                    'labels': record.labels,
                    'metadata': record.metadata
                }
                metric_data['recent_records'].append(record_data)
            
            export_data['metrics'][metric_name] = metric_data
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported metrics to {filepath} (JSON format)")
    
    def _export_csv(self, filepath: str, duration_seconds: Optional[float]):
        """Export metrics to CSV format."""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'metric_name', 'timestamp', 'value', 'type', 'labels', 'metadata'
            ])
            
            for metric_name in self.get_all_metric_names():
                records = self.get_records(metric_name, duration_seconds)
                for record in records:
                    writer.writerow([
                        record.name,
                        record.timestamp,
                        record.value,
                        record.metric_type.value,
                        json.dumps(record.labels),
                        json.dumps(record.metadata)
                    ])
        
        logger.info(f"Exported metrics to {filepath} (CSV format)")
    
    def _export_prometheus(self, filepath: str, duration_seconds: Optional[float]):
        """Export metrics to Prometheus format."""
        lines = []
        
        for metric_name in self.get_all_metric_names():
            summary = self.get_summary(metric_name, duration_seconds)
            if not summary:
                continue
            
            # Add metric help and type
            lines.append(f"# HELP {metric_name} QEM-Bench metric")
            
            if summary.metric_type == MetricType.COUNTER:
                lines.append(f"# TYPE {metric_name} counter")
                lines.append(f"{metric_name}_total {summary.sum_value}")
            elif summary.metric_type == MetricType.GAUGE:
                lines.append(f"# TYPE {metric_name} gauge")
                lines.append(f"{metric_name} {summary.avg_value}")
            elif summary.metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                lines.append(f"# TYPE {metric_name} histogram")
                buckets = self.get_histogram_buckets(metric_name, duration_seconds)
                for bucket, count in buckets.items():
                    le_value = "+Inf" if bucket == float('inf') else str(bucket)
                    lines.append(f'{metric_name}_bucket{{le="{le_value}"}} {count}')
                lines.append(f"{metric_name}_sum {summary.sum_value}")
                lines.append(f"{metric_name}_count {summary.count}")
            else:
                lines.append(f"# TYPE {metric_name} summary")
                lines.append(f"{metric_name}_sum {summary.sum_value}")
                lines.append(f"{metric_name}_count {summary.count}")
                lines.append(f'{metric_name}{{quantile="0.5"}} {summary.p50_value}')
                lines.append(f'{metric_name}{{quantile="0.95"}} {summary.p95_value}')
                lines.append(f'{metric_name}{{quantile="0.99"}} {summary.p99_value}')
            
            lines.append("")  # Empty line between metrics
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Exported metrics to {filepath} (Prometheus format)")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()