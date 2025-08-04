"""Metric exporters for different formats (Prometheus, JSON, CSV)."""

import time
import json
import csv
import logging
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from dataclasses import asdict

from .metrics_collector import MetricsCollector, MetricRecord, MetricSummary, MetricType


logger = logging.getLogger(__name__)


class MetricExporter(ABC):
    """Abstract base class for metric exporters."""
    
    @abstractmethod
    def export(self, collector: MetricsCollector, filepath: str, **kwargs):
        """Export metrics from collector to file."""
        pass
    
    @abstractmethod
    def get_format_name(self) -> str:
        """Get the name of the export format."""
        pass


class JSONExporter(MetricExporter):
    """Export metrics to JSON format."""
    
    def get_format_name(self) -> str:
        return "JSON"
    
    def export(self, collector: MetricsCollector, filepath: str, 
              duration_seconds: Optional[float] = None,
              include_raw_records: bool = False,
              max_records_per_metric: int = 100):
        """
        Export metrics to JSON format.
        
        Args:
            collector: MetricsCollector instance
            filepath: Output file path
            duration_seconds: Only export recent metrics
            include_raw_records: Whether to include raw metric records
            max_records_per_metric: Maximum records per metric to include
        """
        export_data = {
            'export_info': {
                'timestamp': time.time(),
                'format': self.get_format_name(),
                'duration_filter_seconds': duration_seconds,
                'exporter_config': {
                    'include_raw_records': include_raw_records,
                    'max_records_per_metric': max_records_per_metric
                }
            },
            'overview': collector.get_metrics_overview(),
            'metrics': {}
        }
        
        # Export each metric
        for metric_name in collector.get_all_metric_names():
            summary = collector.get_summary(metric_name, duration_seconds)
            
            metric_data = {
                'summary': self._serialize_summary(summary) if summary else None,
                'current_values': {}
            }
            
            # Get current counter/gauge values
            counter_value = collector.get_counter_value(metric_name)
            if counter_value > 0:
                metric_data['current_values']['counter'] = counter_value
            
            gauge_value = collector.get_gauge_value(metric_name)
            if gauge_value is not None:
                metric_data['current_values']['gauge'] = gauge_value
            
            # Include histogram buckets if applicable
            if summary and summary.metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                buckets = collector.get_histogram_buckets(metric_name, duration_seconds)
                metric_data['histogram_buckets'] = buckets
            
            # Include raw records if requested
            if include_raw_records:
                records = collector.get_records(metric_name, duration_seconds)
                # Limit number of records to avoid huge files
                if len(records) > max_records_per_metric:
                    records = records[-max_records_per_metric:]
                
                metric_data['records'] = [
                    self._serialize_record(record) for record in records
                ]
            
            export_data['metrics'][metric_name] = metric_data
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=self._json_serializer)
        
        logger.info(f"Exported metrics to {filepath} (JSON format)")
    
    def _serialize_summary(self, summary: MetricSummary) -> Dict[str, Any]:
        """Convert MetricSummary to serializable dict."""
        return {
            'name': summary.name,
            'metric_type': summary.metric_type.value,
            'count': summary.count,
            'sum': summary.sum_value,
            'avg': summary.avg_value,
            'min': summary.min_value,
            'max': summary.max_value,
            'std_dev': summary.std_deviation,
            'percentiles': {
                'p50': summary.p50_value,
                'p95': summary.p95_value,
                'p99': summary.p99_value
            },
            'time_info': {
                'first_timestamp': summary.first_timestamp,
                'last_timestamp': summary.last_timestamp,
                'duration_seconds': summary.duration_seconds,
                'rate_per_second': summary.rate_per_second
            }
        }
    
    def _serialize_record(self, record: MetricRecord) -> Dict[str, Any]:
        """Convert MetricRecord to serializable dict."""
        return {
            'name': record.name,
            'value': record.value,
            'metric_type': record.metric_type.value,
            'timestamp': record.timestamp,
            'labels': record.labels,
            'metadata': record.metadata,
            'age_seconds': record.age_seconds
        }
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for special objects."""
        if hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)


class CSVExporter(MetricExporter):
    """Export metrics to CSV format."""
    
    def get_format_name(self) -> str:
        return "CSV"
    
    def export(self, collector: MetricsCollector, filepath: str,
              duration_seconds: Optional[float] = None,
              export_mode: str = "records"):
        """
        Export metrics to CSV format.
        
        Args:
            collector: MetricsCollector instance
            filepath: Output file path
            duration_seconds: Only export recent metrics
            export_mode: "records" for raw records, "summary" for summaries
        """
        if export_mode == "records":
            self._export_records_csv(collector, filepath, duration_seconds)
        elif export_mode == "summary":
            self._export_summary_csv(collector, filepath, duration_seconds)
        else:
            raise ValueError(f"Unknown export mode: {export_mode}")
        
        logger.info(f"Exported metrics to {filepath} (CSV format, {export_mode} mode)")
    
    def _export_records_csv(self, collector: MetricsCollector, filepath: str,
                           duration_seconds: Optional[float]):
        """Export raw metric records to CSV."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'metric_name', 'timestamp', 'value', 'metric_type',
                'labels', 'metadata', 'age_seconds'
            ])
            
            # Write records for each metric
            for metric_name in collector.get_all_metric_names():
                records = collector.get_records(metric_name, duration_seconds)
                
                for record in records:
                    writer.writerow([
                        record.name,
                        record.timestamp,
                        record.value,
                        record.metric_type.value,
                        json.dumps(record.labels),
                        json.dumps(record.metadata),
                        record.age_seconds
                    ])
    
    def _export_summary_csv(self, collector: MetricsCollector, filepath: str,
                           duration_seconds: Optional[float]):
        """Export metric summaries to CSV."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'metric_name', 'metric_type', 'count', 'sum', 'avg', 'min', 'max',
                'std_dev', 'p50', 'p95', 'p99', 'duration_seconds', 'rate_per_second'
            ])
            
            # Write summaries for each metric
            for metric_name in collector.get_all_metric_names():
                summary = collector.get_summary(metric_name, duration_seconds)
                
                if summary:
                    writer.writerow([
                        summary.name,
                        summary.metric_type.value,
                        summary.count,
                        summary.sum_value,
                        summary.avg_value,
                        summary.min_value,
                        summary.max_value,
                        summary.std_deviation,
                        summary.p50_value,
                        summary.p95_value,
                        summary.p99_value,
                        summary.duration_seconds,
                        summary.rate_per_second
                    ])


class PrometheusExporter(MetricExporter):
    """Export metrics to Prometheus format."""
    
    def get_format_name(self) -> str:
        return "Prometheus"
    
    def export(self, collector: MetricsCollector, filepath: str,
              duration_seconds: Optional[float] = None,
              include_help: bool = True,
              include_type: bool = True):
        """
        Export metrics to Prometheus format.
        
        Args:
            collector: MetricsCollector instance
            filepath: Output file path
            duration_seconds: Only export recent metrics
            include_help: Include HELP comments
            include_type: Include TYPE comments
        """
        lines = []
        
        # Add export metadata as comments
        lines.append(f"# QEM-Bench metrics export")
        lines.append(f"# Generated at: {time.time()}")
        if duration_seconds:
            lines.append(f"# Duration filter: {duration_seconds} seconds")
        lines.append("")
        
        # Export each metric
        for metric_name in sorted(collector.get_all_metric_names()):
            summary = collector.get_summary(metric_name, duration_seconds)
            if not summary:
                continue
            
            # Sanitize metric name for Prometheus
            prom_name = self._sanitize_metric_name(metric_name)
            
            # Add help comment
            if include_help:
                lines.append(f"# HELP {prom_name} QEM-Bench metric: {metric_name}")
            
            # Export based on metric type
            if summary.metric_type == MetricType.COUNTER:
                if include_type:
                    lines.append(f"# TYPE {prom_name} counter")
                
                # Get current counter value
                counter_value = collector.get_counter_value(metric_name)
                lines.append(f"{prom_name}_total {counter_value}")
                
            elif summary.metric_type == MetricType.GAUGE:
                if include_type:
                    lines.append(f"# TYPE {prom_name} gauge")
                
                # Get current gauge value
                gauge_value = collector.get_gauge_value(metric_name)
                if gauge_value is not None:
                    lines.append(f"{prom_name} {gauge_value}")
                else:
                    lines.append(f"{prom_name} {summary.avg_value}")
                
            elif summary.metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                if include_type:
                    lines.append(f"# TYPE {prom_name} histogram")
                
                # Export histogram buckets
                buckets = collector.get_histogram_buckets(metric_name, duration_seconds)
                for bucket_le, count in sorted(buckets.items()):
                    if bucket_le == float('inf'):
                        lines.append(f'{prom_name}_bucket{{le="+Inf"}} {count}')
                    else:
                        lines.append(f'{prom_name}_bucket{{le="{bucket_le}"}} {count}')
                
                # Add sum and count
                lines.append(f"{prom_name}_sum {summary.sum_value}")
                lines.append(f"{prom_name}_count {summary.count}")
                
            elif summary.metric_type == MetricType.SUMMARY:
                if include_type:
                    lines.append(f"# TYPE {prom_name} summary")
                
                # Export quantiles
                quantiles = [
                    (0.5, summary.p50_value),
                    (0.95, summary.p95_value),
                    (0.99, summary.p99_value)
                ]
                
                for quantile, value in quantiles:
                    lines.append(f'{prom_name}{{quantile="{quantile}"}} {value}')
                
                # Add sum and count
                lines.append(f"{prom_name}_sum {summary.sum_value}")
                lines.append(f"{prom_name}_count {summary.count}")
            
            lines.append("")  # Empty line between metrics
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Exported metrics to {filepath} (Prometheus format)")
    
    def _sanitize_metric_name(self, name: str) -> str:
        """Sanitize metric name for Prometheus format."""
        # Replace invalid characters with underscores
        sanitized = ""
        for char in name:
            if char.isalnum() or char == '_':
                sanitized += char
            else:
                sanitized += '_'
        
        # Ensure it starts with a letter or underscore
        if sanitized and not (sanitized[0].isalpha() or sanitized[0] == '_'):
            sanitized = '_' + sanitized
        
        return sanitized.lower()


class MultiFormatExporter:
    """Export metrics to multiple formats simultaneously."""
    
    def __init__(self):
        self.exporters = {
            'json': JSONExporter(),
            'csv': CSVExporter(),
            'prometheus': PrometheusExporter()
        }
    
    def export_all_formats(self, collector: MetricsCollector, base_filepath: str,
                          duration_seconds: Optional[float] = None,
                          formats: Optional[List[str]] = None,
                          **kwargs):
        """
        Export metrics to multiple formats.
        
        Args:
            collector: MetricsCollector instance
            base_filepath: Base file path (extensions will be added)
            duration_seconds: Only export recent metrics
            formats: List of formats to export (default: all)
            **kwargs: Additional arguments passed to exporters
        """
        if formats is None:
            formats = list(self.exporters.keys())
        
        # Remove extension from base filepath if present
        if '.' in base_filepath:
            base_filepath = base_filepath.rsplit('.', 1)[0]
        
        results = {}
        
        for format_name in formats:
            if format_name not in self.exporters:
                logger.warning(f"Unknown export format: {format_name}")
                continue
            
            try:
                exporter = self.exporters[format_name]
                
                # Determine file extension
                if format_name == 'json':
                    filepath = f"{base_filepath}.json"
                elif format_name == 'csv':
                    filepath = f"{base_filepath}.csv"
                elif format_name == 'prometheus':
                    filepath = f"{base_filepath}.prom"
                else:
                    filepath = f"{base_filepath}.{format_name}"
                
                # Export with format-specific arguments
                format_kwargs = kwargs.copy()
                if format_name in kwargs:
                    format_kwargs.update(kwargs[format_name])
                
                exporter.export(collector, filepath, duration_seconds, **format_kwargs)
                results[format_name] = {'status': 'success', 'filepath': filepath}
                
            except Exception as e:
                logger.error(f"Failed to export {format_name} format: {e}")
                results[format_name] = {'status': 'error', 'error': str(e)}
        
        return results
    
    def add_exporter(self, name: str, exporter: MetricExporter):
        """Add a custom exporter."""
        self.exporters[name] = exporter
    
    def remove_exporter(self, name: str):
        """Remove an exporter."""
        if name in self.exporters:
            del self.exporters[name]
    
    def get_available_formats(self) -> List[str]:
        """Get list of available export formats."""
        return list(self.exporters.keys())


# Convenience functions for direct use
def export_to_json(collector: MetricsCollector, filepath: str, **kwargs):
    """Export metrics to JSON format."""
    exporter = JSONExporter()
    exporter.export(collector, filepath, **kwargs)


def export_to_csv(collector: MetricsCollector, filepath: str, **kwargs):
    """Export metrics to CSV format."""
    exporter = CSVExporter()
    exporter.export(collector, filepath, **kwargs)


def export_to_prometheus(collector: MetricsCollector, filepath: str, **kwargs):
    """Export metrics to Prometheus format."""
    exporter = PrometheusExporter()
    exporter.export(collector, filepath, **kwargs)