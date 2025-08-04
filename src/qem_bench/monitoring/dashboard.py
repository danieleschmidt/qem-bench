"""Dashboard and reporting system for QEM-Bench monitoring."""

import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import json


logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for the monitoring dashboard."""
    refresh_interval: float = 30.0  # seconds
    max_history_points: int = 100
    enable_trends: bool = True
    enable_alerts: bool = True
    enable_recommendations: bool = True


class MonitoringDashboard:
    """
    Text-based monitoring dashboard for QEM-Bench.
    
    Provides real-time status reports, performance analysis, and trend monitoring
    for quantum error mitigation experiments. Designed to be lightweight and
    work in command-line environments.
    
    Example:
        >>> # Create dashboard with monitoring components
        >>> dashboard = MonitoringDashboard(
        ...     system_monitor=system_monitor,
        ...     performance_monitor=performance_monitor,
        ...     health_checker=health_checker
        ... )
        >>> 
        >>> # Generate status report
        >>> report = dashboard.generate_status_report()
        >>> print(report)
        >>> 
        >>> # Get performance analysis
        >>> analysis = dashboard.analyze_performance_trends()
    """
    
    def __init__(
        self,
        system_monitor=None,
        performance_monitor=None,
        resource_monitor=None,
        health_checker=None,
        metrics_collector=None,
        alert_manager=None,
        config: Optional[DashboardConfig] = None
    ):
        self.config = config or DashboardConfig()
        
        # Monitoring components
        self.system_monitor = system_monitor
        self.performance_monitor = performance_monitor
        self.resource_monitor = resource_monitor
        self.health_checker = health_checker
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        
        # Dashboard state
        self._last_refresh = 0
        self._status_history = []
    
    def generate_status_report(self, detailed: bool = True) -> str:
        """
        Generate a comprehensive status report.
        
        Args:
            detailed: Whether to include detailed sections
        
        Returns:
            Formatted text report
        """
        report_lines = []
        current_time = time.time()
        
        # Header
        report_lines.extend([
            "=" * 60,
            "QEM-BENCH MONITORING DASHBOARD",
            "=" * 60,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}",
            ""
        ])
        
        # System Overview
        system_section = self._generate_system_overview()
        if system_section:
            report_lines.extend(system_section)
            report_lines.append("")
        
        # Health Status
        health_section = self._generate_health_status()
        if health_section:
            report_lines.extend(health_section)
            report_lines.append("")
        
        # Performance Summary
        performance_section = self._generate_performance_summary()
        if performance_section:
            report_lines.extend(performance_section)
            report_lines.append("")
        
        # Resource Usage
        resource_section = self._generate_resource_summary()
        if resource_section:
            report_lines.extend(resource_section)
            report_lines.append("")
        
        # Active Alerts
        if self.config.enable_alerts:
            alerts_section = self._generate_alerts_summary()
            if alerts_section:
                report_lines.extend(alerts_section)
                report_lines.append("")
        
        # Metrics Overview
        metrics_section = self._generate_metrics_overview()
        if metrics_section:
            report_lines.extend(metrics_section)
            report_lines.append("")
        
        # Detailed sections
        if detailed:
            # Performance Trends
            if self.config.enable_trends:
                trends_section = self._generate_trends_analysis()
                if trends_section:
                    report_lines.extend(trends_section)
                    report_lines.append("")
            
            # Recommendations
            if self.config.enable_recommendations:
                recommendations_section = self._generate_recommendations()
                if recommendations_section:
                    report_lines.extend(recommendations_section)
                    report_lines.append("")
        
        # Footer
        report_lines.extend([
            "=" * 60,
            "End of Report",
            "=" * 60
        ])
        
        return "\\n".join(report_lines)
    
    def _generate_system_overview(self) -> List[str]:
        """Generate system overview section."""
        if not self.system_monitor:
            return []
        
        lines = ["SYSTEM OVERVIEW", "-" * 15]
        
        try:
            current_stats = self.system_monitor.get_current_stats()
            if current_stats:
                lines.extend([
                    f"CPU Usage:      {current_stats.cpu_percent:6.1f}%",
                    f"Memory Usage:   {current_stats.memory_percent:6.1f}% ({current_stats.memory_used_gb:.1f}GB used)",
                    f"Disk Usage:     {current_stats.disk_usage_percent:6.1f}%",
                    f"Process Count:  {current_stats.process_count:6d}"
                ])
                
                if current_stats.load_average:
                    lines.append(f"Load Average:   {current_stats.load_average[0]:.2f}, {current_stats.load_average[1]:.2f}, {current_stats.load_average[2]:.2f}")
                
                if current_stats.gpu_info:
                    gpu_info = current_stats.gpu_info
                    lines.append(f"GPUs Available: {gpu_info.get('count', 0)}")
                    for i, gpu in enumerate(gpu_info.get('gpus', [])):
                        lines.append(f"  GPU {i}: {gpu['name']} ({gpu['memory_percent']:.1f}% memory, {gpu['gpu_percent']:.1f}% util)")
            else:
                lines.append("No current system data available")
            
            # Check for alerts
            alerts = self.system_monitor.check_alert_thresholds()
            if alerts:
                lines.append("")
                lines.append("SYSTEM ALERTS:")
                for alert in alerts:
                    lines.append(f"  ⚠ {alert['message']}")
        
        except Exception as e:
            lines.append(f"Error getting system overview: {e}")
        
        return lines
    
    def _generate_health_status(self) -> List[str]:
        """Generate health status section."""
        if not self.health_checker:
            return []
        
        lines = ["HEALTH STATUS", "-" * 13]
        
        try:
            health_summary = self.health_checker.get_health_summary()
            overall_status = health_summary.get('overall_status', 'unknown')
            
            # Status indicator
            status_icon = self._get_status_icon(overall_status)
            lines.append(f"Overall Health: {status_icon} {overall_status.upper()}")
            
            # Provider summary
            status_counts = health_summary.get('status_counts', {})
            total_providers = health_summary.get('total_providers', 0)
            
            lines.extend([
                f"Total Providers: {total_providers}",
                f"  Healthy:       {status_counts.get('healthy', 0)}",
                f"  Warning:       {status_counts.get('warning', 0)}",
                f"  Unhealthy:     {status_counts.get('unhealthy', 0)}",
                f"  Unknown:       {status_counts.get('unknown', 0)}"
            ])
            
            # Critical issues
            critical_issues = health_summary.get('critical_issues', [])
            if critical_issues:
                lines.append("")
                lines.append("CRITICAL ISSUES:")
                for issue in critical_issues[:3]:  # Show top 3
                    lines.append(f"  ✗ {issue['provider']}: {issue['message']}")
                if len(critical_issues) > 3:
                    lines.append(f"  ... and {len(critical_issues) - 3} more")
            
            # Warnings
            warning_issues = health_summary.get('warning_issues', [])
            if warning_issues:
                lines.append("")
                lines.append("WARNINGS:")
                for warning in warning_issues[:3]:  # Show top 3
                    lines.append(f"  ⚠ {warning['provider']}: {warning['message']}")
                if len(warning_issues) > 3:
                    lines.append(f"  ... and {len(warning_issues) - 3} more")
        
        except Exception as e:
            lines.append(f"Error getting health status: {e}")
        
        return lines
    
    def _generate_performance_summary(self) -> List[str]:
        """Generate performance summary section."""
        if not self.performance_monitor:
            return []
        
        lines = ["PERFORMANCE SUMMARY", "-" * 19]
        
        try:
            operation_names = self.performance_monitor.get_all_operation_names()
            if not operation_names:
                lines.append("No performance data available")
                return lines
            
            # Show top operations by execution count
            operation_stats = []
            for op_name in operation_names:
                stats = self.performance_monitor.get_stats(op_name, duration_seconds=3600)  # Last hour
                if stats:
                    operation_stats.append((op_name, stats))
            
            # Sort by count (most executed first)
            operation_stats.sort(key=lambda x: x[1].count, reverse=True)
            
            lines.append(f"{'Operation':<25} {'Count':<8} {'Avg(ms)':<10} {'P95(ms)':<10}")
            lines.append("-" * 55)
            
            for op_name, stats in operation_stats[:5]:  # Top 5
                lines.append(
                    f"{op_name:<25} {stats.count:<8} {stats.avg_duration_ms:<10.1f} {stats.p95_duration*1000:<10.1f}"
                )
            
            if len(operation_stats) > 5:
                lines.append(f"... and {len(operation_stats) - 5} more operations")
            
            # Current active operations
            current_ops = self.performance_monitor.get_current_operations()
            if current_ops:
                lines.append("")
                lines.append("ACTIVE OPERATIONS:")
                for thread_id, ops in current_ops.items():
                    if ops:
                        lines.append(f"  Thread {thread_id}: {' -> '.join(ops)}")
        
        except Exception as e:
            lines.append(f"Error getting performance summary: {e}")
        
        return lines
    
    def _generate_resource_summary(self) -> List[str]:
        """Generate resource usage summary section."""
        if not self.resource_monitor:
            return []
        
        lines = ["RESOURCE USAGE", "-" * 14]
        
        try:
            global_stats = self.resource_monitor.get_global_stats(duration_seconds=3600)  # Last hour
            if not global_stats:
                lines.append("No resource data available")
                return lines
            
            lines.extend([
                f"Total Executions: {global_stats.get('total_executions', 0)}",
                f"Total Shots:      {global_stats.get('total_shots', 0):,}",
                f"Total Circuits:   {global_stats.get('total_circuits', 0):,}",
                f"Total Gates:      {global_stats.get('total_gates', 0):,}",
                f"Avg Shots/Circuit: {global_stats.get('avg_shots_per_circuit', 0):.1f}",
                f"Avg Gates/Circuit: {global_stats.get('avg_gates_per_circuit', 0):.1f}"
            ])
            
            # Backend distribution
            backend_dist = global_stats.get('backend_distribution', {})
            if backend_dist:
                lines.append("")
                lines.append("BACKEND USAGE:")
                for backend, count in sorted(backend_dist.items(), key=lambda x: x[1], reverse=True)[:3]:
                    lines.append(f"  {backend}: {count} executions")
            
            # Top gates
            top_gates = global_stats.get('top_gates', {})
            if top_gates:
                lines.append("")
                lines.append("TOP GATES:")
                for gate, count in list(top_gates.items())[:5]:
                    lines.append(f"  {gate}: {count:,}")
        
        except Exception as e:
            lines.append(f"Error getting resource summary: {e}")
        
        return lines
    
    def _generate_alerts_summary(self) -> List[str]:
        """Generate alerts summary section."""
        if not self.alert_manager:
            return []
        
        lines = ["ACTIVE ALERTS", "-" * 13]
        
        try:
            active_alerts = self.alert_manager.get_active_alerts()
            if not active_alerts:
                lines.append("✓ No active alerts")
                return lines
            
            # Group by severity
            alerts_by_severity = defaultdict(list)
            for alert in active_alerts:
                alerts_by_severity[alert.severity.value].append(alert)
            
            # Show critical first, then high, medium, low
            for severity in ['critical', 'high', 'medium', 'low']:
                alerts = alerts_by_severity.get(severity, [])
                if alerts:
                    severity_icon = self._get_severity_icon(severity)
                    lines.append(f"{severity_icon} {severity.upper()} ({len(alerts)}):")
                    for alert in alerts[:3]:  # Show top 3 per severity
                        age_str = self._format_duration(alert.age_seconds)
                        lines.append(f"  {alert.title} ({age_str} ago)")
                        lines.append(f"    {alert.message}")
                    if len(alerts) > 3:
                        lines.append(f"    ... and {len(alerts) - 3} more")
                    lines.append("")
        
        except Exception as e:
            lines.append(f"Error getting alerts summary: {e}")
        
        return lines
    
    def _generate_metrics_overview(self) -> List[str]:
        """Generate metrics overview section."""
        if not self.metrics_collector:
            return []
        
        lines = ["METRICS OVERVIEW", "-" * 16]
        
        try:
            overview = self.metrics_collector.get_metrics_overview()
            
            lines.extend([
                f"Total Metrics:  {overview.get('total_metrics', 0)}",
                f"Total Records:  {overview.get('total_records', 0):,}",
            ])
            
            # Metrics by type
            metrics_by_type = overview.get('metrics_by_type', {})
            if metrics_by_type:
                lines.append("")
                lines.append("BY TYPE:")
                for metric_type, count in metrics_by_type.items():
                    lines.append(f"  {metric_type.title()}: {count}")
            
            # Data span
            if 'data_span_seconds' in overview:
                span_str = self._format_duration(overview['data_span_seconds'])
                lines.append(f"Data Span:      {span_str}")
        
        except Exception as e:
            lines.append(f"Error getting metrics overview: {e}")
        
        return lines
    
    def _generate_trends_analysis(self) -> List[str]:
        """Generate trends analysis section."""
        lines = ["PERFORMANCE TRENDS", "-" * 18]
        
        try:
            # Get recent performance data for key operations
            if self.performance_monitor:
                # Analyze ZNE performance trends
                zne_stats_1h = self.performance_monitor.get_stats("zne_mitigation", duration_seconds=3600)
                zne_stats_24h = self.performance_monitor.get_stats("zne_mitigation", duration_seconds=86400)
                
                if zne_stats_1h and zne_stats_24h:
                    if zne_stats_24h.avg_duration > 0:
                        performance_change = ((zne_stats_1h.avg_duration - zne_stats_24h.avg_duration) / zne_stats_24h.avg_duration) * 100
                        trend_icon = "↗" if performance_change > 5 else "↘" if performance_change < -5 else "→"
                        lines.append(f"ZNE Performance: {trend_icon} {performance_change:+.1f}% vs 24h avg")
                    
                    throughput_1h = zne_stats_1h.rate_per_second
                    throughput_24h = zne_stats_24h.rate_per_second
                    lines.append(f"ZNE Throughput:  {throughput_1h:.2f} ops/sec (1h) vs {throughput_24h:.2f} ops/sec (24h)")
            
            # System resource trends
            if self.system_monitor:
                recent_avg = self.system_monitor.get_average_stats(duration_seconds=3600)  # 1 hour
                if recent_avg:
                    lines.extend([
                        "",
                        "RESOURCE TRENDS (1h avg):",
                        f"  CPU:    {recent_avg.get('cpu_percent', 0):.1f}%",
                        f"  Memory: {recent_avg.get('memory_percent', 0):.1f}%",
                        f"  Disk:   {recent_avg.get('disk_usage_percent', 0):.1f}%"
                    ])
                    
                    # GPU trends if available
                    if any(k.startswith('gpu_') for k in recent_avg.keys()):
                        gpu_count = recent_avg.get('gpu_count', 0)
                        if gpu_count > 0:
                            lines.append("  GPUs:")
                            for i in range(int(gpu_count)):
                                mem_key = f'gpu_{i}_memory_percent'
                                util_key = f'gpu_{i}_utilization'
                                if mem_key in recent_avg:
                                    lines.append(f"    GPU {i}: {recent_avg[util_key]:.1f}% util, {recent_avg[mem_key]:.1f}% mem")
            
        except Exception as e:
            lines.append(f"Error generating trends analysis: {e}")
        
        return lines
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations section."""
        lines = ["RECOMMENDATIONS", "-" * 15]
        recommendations = []
        
        try:
            # System recommendations
            if self.system_monitor:
                current_stats = self.system_monitor.get_current_stats()
                if current_stats:
                    if current_stats.cpu_percent > 80:
                        recommendations.append("Consider reducing parallel workload or upgrading CPU")
                    if current_stats.memory_percent > 85:
                        recommendations.append("System memory usage is high - consider memory optimization")
                    if current_stats.disk_usage_percent > 90:
                        recommendations.append("Disk space is critically low - clean up or expand storage")
                    
                    if current_stats.gpu_info and current_stats.gpu_info.get('count', 0) == 0:
                        recommendations.append("No GPU detected - consider GPU acceleration for better performance")
            
            # Performance recommendations
            if self.performance_monitor:
                operation_names = self.performance_monitor.get_all_operation_names()
                slow_operations = []
                
                for op_name in operation_names:
                    stats = self.performance_monitor.get_stats(op_name, duration_seconds=3600)
                    if stats and stats.avg_duration > 10:  # Slower than 10 seconds
                        slow_operations.append((op_name, stats.avg_duration))
                
                if slow_operations:
                    slow_operations.sort(key=lambda x: x[1], reverse=True)
                    slowest_op = slow_operations[0]
                    recommendations.append(f"Operation '{slowest_op[0]}' is slow ({slowest_op[1]:.1f}s avg) - investigate optimization")
            
            # Health recommendations
            if self.health_checker:
                health_summary = self.health_checker.get_health_summary()
                critical_issues = health_summary.get('critical_issues', [])
                warning_issues = health_summary.get('warning_issues', [])
                
                if critical_issues:
                    recommendations.append(f"Address {len(critical_issues)} critical health issues")
                if warning_issues:
                    recommendations.append(f"Review {len(warning_issues)} health warnings")
            
            # Alert recommendations
            if self.alert_manager:
                active_alerts = self.alert_manager.get_active_alerts()
                if active_alerts:
                    critical_count = len([a for a in active_alerts if a.severity.value == 'critical'])
                    high_count = len([a for a in active_alerts if a.severity.value == 'high'])
                    
                    if critical_count > 0:
                        recommendations.append(f"Immediately address {critical_count} critical alerts")
                    if high_count > 0:
                        recommendations.append(f"Review {high_count} high-priority alerts")
            
            # Resource optimization recommendations
            if self.resource_monitor:
                global_stats = self.resource_monitor.get_global_stats(duration_seconds=3600)
                if global_stats:
                    avg_shots = global_stats.get('avg_shots_per_circuit', 0)
                    if avg_shots > 10000:
                        recommendations.append("Consider reducing shots per circuit for faster execution")
                    elif avg_shots < 100:
                        recommendations.append("Low shot count may affect result accuracy")
            
            # Display recommendations
            if recommendations:
                for i, rec in enumerate(recommendations[:5], 1):  # Top 5
                    lines.append(f"{i}. {rec}")
                if len(recommendations) > 5:
                    lines.append(f"... and {len(recommendations) - 5} more recommendations")
            else:
                lines.append("✓ No immediate recommendations")
        
        except Exception as e:
            lines.append(f"Error generating recommendations: {e}")
        
        return lines
    
    def _get_status_icon(self, status: str) -> str:
        """Get status icon for health status."""
        icons = {
            'healthy': '✓',
            'warning': '⚠',
            'unhealthy': '✗',
            'unknown': '?'
        }
        return icons.get(status.lower(), '?')
    
    def _get_severity_icon(self, severity: str) -> str:
        """Get icon for alert severity."""
        icons = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }
        return icons.get(severity.lower(), '⚪')
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"
    
    def analyze_performance_trends(self, duration_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Args:
            duration_hours: Hours of history to analyze
        
        Returns:
            Dictionary with trend analysis
        """
        if not self.performance_monitor:
            return {'error': 'Performance monitoring not available'}
        
        duration_seconds = duration_hours * 3600
        analysis = {
            'analysis_duration_hours': duration_hours,
            'timestamp': time.time(),
            'operations': {}
        }
        
        operation_names = self.performance_monitor.get_all_operation_names()
        
        for op_name in operation_names:
            stats = self.performance_monitor.get_stats(op_name, duration_seconds)
            if stats and stats.count >= 10:  # Need minimum data points
                # Simple trend analysis
                records = self.performance_monitor.get_records(op_name, duration_seconds)
                if len(records) >= 10:
                    # Split into first and second half for trend comparison
                    mid_point = len(records) // 2
                    first_half = records[:mid_point]
                    second_half = records[mid_point:]
                    
                    first_avg = sum(r.duration for r in first_half) / len(first_half)
                    second_avg = sum(r.duration for r in second_half) / len(second_half)
                    
                    trend_pct = ((second_avg - first_avg) / first_avg) * 100 if first_avg > 0 else 0
                    trend_direction = 'improving' if trend_pct < -5 else 'degrading' if trend_pct > 5 else 'stable'
                    
                    analysis['operations'][op_name] = {
                        'total_executions': stats.count,
                        'avg_duration_ms': stats.avg_duration_ms,
                        'trend_percentage': trend_pct,
                        'trend_direction': trend_direction,
                        'p95_duration_ms': stats.p95_duration * 1000,
                        'throughput_per_second': stats.throughput_per_second
                    }
        
        return analysis
    
    def generate_executive_summary(self) -> str:
        """Generate a brief executive summary."""
        lines = []
        current_time = time.time()
        
        lines.extend([
            "QEM-BENCH EXECUTIVE SUMMARY",
            "=" * 27,
            f"Report Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}",
            ""
        ])
        
        # Overall system health
        if self.health_checker:
            health_summary = self.health_checker.get_health_summary()
            overall_status = health_summary.get('overall_status', 'unknown')
            status_icon = self._get_status_icon(overall_status)
            lines.append(f"System Health: {status_icon} {overall_status.upper()}")
        
        # Key metrics
        if self.performance_monitor:
            operation_names = self.performance_monitor.get_all_operation_names()
            total_operations = sum(
                self.performance_monitor.get_stats(op, duration_seconds=86400).count
                for op in operation_names
                if self.performance_monitor.get_stats(op, duration_seconds=86400)
            )
            lines.append(f"Operations (24h): {total_operations}")
        
        if self.resource_monitor:
            global_stats = self.resource_monitor.get_global_stats(duration_seconds=86400)
            if global_stats:
                lines.append(f"Circuits Executed: {global_stats.get('total_circuits', 0):,}")
                lines.append(f"Total Shots: {global_stats.get('total_shots', 0):,}")
        
        # Alerts summary
        if self.alert_manager:
            active_alerts = self.alert_manager.get_active_alerts()
            critical_count = len([a for a in active_alerts if a.severity.value == 'critical'])
            high_count = len([a for a in active_alerts if a.severity.value == 'high'])
            
            if critical_count > 0:
                lines.append(f"🔴 Critical Alerts: {critical_count}")
            if high_count > 0:
                lines.append(f"🟠 High Priority Alerts: {high_count}")
            if critical_count == 0 and high_count == 0:
                lines.append("✓ No critical alerts")
        
        # Top recommendation
        recommendations = self._generate_recommendations()
        if len(recommendations) > 2:  # Skip header and separator
            lines.append("")
            lines.append("TOP RECOMMENDATION:")
            lines.append(f"• {recommendations[2]}")  # First actual recommendation
        
        return "\\n".join(lines)
    
    def export_dashboard_data(self, filepath: str, format: str = "json"):
        """Export dashboard data for external use."""
        dashboard_data = {
            'export_timestamp': time.time(),
            'config': {
                'refresh_interval': self.config.refresh_interval,
                'enable_trends': self.config.enable_trends,
                'enable_alerts': self.config.enable_alerts
            },
            'status_report': self.generate_status_report(detailed=False),
            'executive_summary': self.generate_executive_summary(),
            'performance_trends': self.analyze_performance_trends(),
            'component_status': {
                'system_monitor': self.system_monitor is not None,
                'performance_monitor': self.performance_monitor is not None,
                'resource_monitor': self.resource_monitor is not None,
                'health_checker': self.health_checker is not None,
                'metrics_collector': self.metrics_collector is not None,
                'alert_manager': self.alert_manager is not None
            }
        }
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported dashboard data to {filepath}")


def create_dashboard_from_monitors(**monitors) -> MonitoringDashboard:
    """
    Convenience function to create a dashboard from monitoring components.
    
    Args:
        **monitors: Keyword arguments with monitoring component instances
    
    Returns:
        MonitoringDashboard instance
    """
    return MonitoringDashboard(
        system_monitor=monitors.get('system_monitor'),
        performance_monitor=monitors.get('performance_monitor'),
        resource_monitor=monitors.get('resource_monitor'),
        health_checker=monitors.get('health_checker'),
        metrics_collector=monitors.get('metrics_collector'),
        alert_manager=monitors.get('alert_manager')
    )