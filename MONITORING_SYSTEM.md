# QEM-Bench Monitoring and Health Check System

## Overview

The QEM-Bench monitoring system provides comprehensive observability for quantum error mitigation experiments. It includes real-time monitoring, health checking, metrics collection, performance analysis, and alerting capabilities.

## Architecture

The monitoring system consists of several interconnected components:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   System        │    │  Performance     │    │  Quantum        │
│   Monitor       │    │  Monitor         │    │  Resource       │
│                 │    │                  │    │  Monitor        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Health        │    │  Metrics         │    │  Alert          │
│   Checker       │    │  Collector       │    │  Manager        │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌──────────────────┐
                    │   Monitoring     │
                    │   Dashboard      │
                    │                  │
                    └──────────────────┘
```

## Components

### 1. Monitoring Framework (`/src/qem_bench/monitoring/`)

#### SystemMonitor
- **Purpose**: Tracks system resource usage (CPU, memory, GPU, disk, network)
- **Features**:
  - Real-time resource monitoring with configurable sampling intervals
  - GPU detection and monitoring (NVIDIA/AMD support)
  - Threshold-based alerting for resource limits
  - Historical data retention with configurable limits
  - Thread-safe operation with background monitoring
- **Key Methods**:
  - `start()`: Begin monitoring in background thread
  - `get_current_stats()`: Get latest system snapshot
  - `get_average_stats(duration)`: Get averaged statistics over time period
  - `export_stats(filepath)`: Export monitoring data to file

#### PerformanceMonitor
- **Purpose**: Times and profiles mitigation method execution
- **Features**:
  - Context manager and decorator-based timing
  - Statistical analysis (mean, percentiles, std dev)
  - Nested operation tracking
  - Throughput calculation
  - Performance trend analysis
- **Key Methods**:
  - `time_operation(name)`: Context manager for timing operations
  - `profile(name)`: Decorator for automatic function profiling
  - `get_stats(operation, duration)`: Get performance statistics
  - `get_summary_report()`: Generate human-readable performance report

#### QuantumResourceMonitor
- **Purpose**: Tracks quantum-specific resources (shots, circuits, gates, qubits)
- **Features**:
  - Circuit analysis and gate counting
  - Shot and execution time tracking
  - Backend usage monitoring
  - Cost estimation (if configured)
  - Resource efficiency analysis
- **Key Methods**:
  - `track_execution(name)`: Context manager for resource tracking
  - `get_stats(operation)`: Get resource usage statistics
  - `get_global_stats()`: Get aggregated statistics across all operations

#### AlertManager
- **Purpose**: Threshold-based alerting and notification system
- **Features**:
  - Configurable alert rules with multiple conditions
  - Severity levels (low, medium, high, critical)
  - Cooldown periods to prevent alert spam
  - Alert resolution tracking
  - Statistical analysis of alert patterns
- **Key Methods**:
  - `add_rule(rule)`: Add new alert rule
  - `check_metric(name, value, source)`: Check metric against rules
  - `get_active_alerts()`: Get currently active alerts
  - `trigger_alert()`: Manually trigger alert

### 2. Health Check System (`/src/qem_bench/health/`)

#### HealthChecker
- **Purpose**: Comprehensive system health validation
- **Features**:
  - Pluggable health check providers
  - Parallel health check execution
  - Health status aggregation
  - Trend analysis and historical tracking
  - Automatic remediation hooks
- **Key Methods**:
  - `add_provider(provider)`: Add health check provider
  - `run_all_checks()`: Execute all health checks
  - `get_overall_status()`: Get system-wide health status
  - `start_monitoring()`: Begin continuous health monitoring

#### DependencyChecker
- **Purpose**: Validates Python dependencies and system requirements
- **Features**:
  - Required vs optional dependency checking
  - Version compatibility validation
  - JAX/GPU setup verification
  - Python version compatibility
  - Installation recommendations
- **Checks**:
  - Core dependencies (numpy, jax, scipy, etc.)
  - Optional packages (qiskit, cirq, pennylane, etc.)
  - GPU acceleration availability
  - System-level dependencies

#### HardwareDetector  
- **Purpose**: Detects and analyzes hardware capabilities
- **Features**:
  - CPU analysis (cores, frequency, architecture)
  - Memory configuration detection
  - GPU detection (NVIDIA/AMD)
  - Storage analysis (SSD vs HDD)
  - Performance tier classification
- **Capabilities**:
  - Quantum simulation readiness assessment
  - Hardware performance tier classification
  - Resource adequacy validation

#### BackendHealthProbe
- **Purpose**: Monitors quantum backend health and availability
- **Features**:
  - Backend connectivity validation
  - Queue status monitoring
  - Calibration data freshness checking
  - Performance characteristics analysis
  - Multi-backend parallel monitoring
- **Checks**:
  - Backend operational status
  - Queue length and estimated wait times
  - Calibration data age and quality
  - Error rates and fidelity metrics

### 3. Metrics Collection (`/src/qem_bench/metrics/`)

#### MetricsCollector
- **Purpose**: Collects and analyzes various types of metrics
- **Features**:
  - Multiple metric types (counter, gauge, histogram, summary, timer)
  - Label-based metric organization
  - Statistical analysis and percentile calculation
  - Time-series data retention
  - Export to multiple formats (JSON, CSV, Prometheus)
- **Metric Types**:
  - **Counters**: Monotonically increasing values
  - **Gauges**: Point-in-time measurements
  - **Histograms**: Distribution of values with bucketing
  - **Summaries**: Statistical summaries with quantiles
  - **Timers**: Duration measurements

#### CircuitMetrics
- **Purpose**: Analyzes quantum circuit properties and complexity
- **Features**:
  - Gate counting and categorization
  - Circuit depth and width analysis
  - Connectivity pattern analysis
  - Complexity metrics (volume, expressibility)
  - Performance impact estimation
- **Analysis Results**:
  - Basic properties (qubits, gates, depth)
  - Gate distribution and ratios
  - Topology and connectivity
  - Estimated runtime and noise susceptibility

#### NoiseMetrics
- **Purpose**: Analyzes noise models and error characteristics
- **Features**:
  - Error rate extraction and analysis
  - Coherence time analysis (T1, T2)
  - Noise correlation detection
  - Mitigation difficulty assessment
  - Method recommendation engine
- **Analysis Results**:
  - Error rate statistics
  - Noise structure classification
  - Quality metrics and fidelity estimates
  - Mitigation strategy recommendations

### 4. Integration Points

#### MonitoredZeroNoiseExtrapolation
- **Purpose**: ZNE implementation with integrated monitoring
- **Features**:
  - Transparent monitoring integration
  - Circuit and noise analysis
  - Performance tracking
  - Resource usage monitoring
  - Alert generation
- **Enhanced Results**:
  - Original ZNE results plus monitoring data
  - Circuit analysis information
  - Performance metrics
  - Resource usage statistics
  - System health indicators

### 5. Dashboard and Reporting (`/src/qem_bench/monitoring/dashboard.py`)

#### MonitoringDashboard
- **Purpose**: Unified monitoring interface and reporting
- **Features**:
  - Real-time status reports
  - Performance trend analysis
  - Resource utilization summaries
  - Health status overview
  - Executive summaries for stakeholders
- **Report Types**:
  - **Status Reports**: Comprehensive system overview
  - **Executive Summary**: High-level status for management
  - **Performance Trends**: Historical performance analysis
  - **Health Summary**: System health and recommendations

## Usage Examples

### Basic Monitoring Setup

```python
from qem_bench import (
    SystemMonitor, PerformanceMonitor, QuantumResourceMonitor,
    HealthChecker, MetricsCollector, AlertManager,
    MonitoringDashboard
)

# Initialize monitoring components
system_monitor = SystemMonitor()
performance_monitor = PerformanceMonitor()
resource_monitor = QuantumResourceMonitor()
health_checker = HealthChecker()
metrics_collector = MetricsCollector()
alert_manager = AlertManager()

# Start monitoring
system_monitor.start()
health_checker.start_monitoring()

# Create dashboard
dashboard = MonitoringDashboard(
    system_monitor=system_monitor,
    performance_monitor=performance_monitor,
    resource_monitor=resource_monitor,
    health_checker=health_checker,
    metrics_collector=metrics_collector,
    alert_manager=alert_manager
)
```

### Using Monitored ZNE

```python
from qem_bench import MonitoredZeroNoiseExtrapolation

# Create monitored ZNE with comprehensive monitoring
zne = MonitoredZeroNoiseExtrapolation(
    noise_factors=[1.0, 1.5, 2.0, 2.5, 3.0],
    extrapolator="richardson"
)

# Execute with monitoring
result = zne.mitigate(circuit, backend, observable, shots=1024)

# Access monitoring data
print(f"Raw value: {result.raw_value}")
print(f"Mitigated value: {result.mitigated_value}")
print(f"Monitoring data: {result.monitoring_data}")

# Get performance insights
performance_summary = zne.get_performance_summary()
resource_summary = zne.get_resource_summary()
health_summary = zne.get_health_summary()
```

### Dashboard Reporting

```python
# Generate status report
status_report = dashboard.generate_status_report()
print(status_report)

# Generate executive summary
executive_summary = dashboard.generate_executive_summary()
print(executive_summary)

# Analyze performance trends
trends = dashboard.analyze_performance_trends(duration_hours=24)
print(f"Performance trends: {trends}")
```

### Circuit and Noise Analysis

```python
from qem_bench import CircuitMetrics, NoiseMetrics

# Analyze circuit
circuit_analyzer = CircuitMetrics()
analysis = circuit_analyzer.analyze_circuit(circuit, "my_circuit")
print(f"Circuit depth: {analysis.circuit_depth}")
print(f"Gate distribution: {analysis.gate_distribution}")

# Analyze noise model
noise_analyzer = NoiseMetrics()
noise_analysis = noise_analyzer.analyze_noise_model(noise_model, "my_noise")
print(f"Overall error rate: {noise_analysis.overall_error_rate}")
print(f"Recommended methods: {noise_analysis.recommended_methods}")
```

## Configuration

### System Monitor Configuration

```python
from qem_bench.monitoring import SystemMonitorConfig

config = SystemMonitorConfig(
    enabled=True,
    sampling_interval=1.0,  # seconds
    max_history_size=1000,
    monitor_gpu=True,
    alert_thresholds={
        'cpu_percent': 90.0,
        'memory_percent': 85.0,
        'disk_usage_percent': 90.0
    }
)

system_monitor = SystemMonitor(config=config)
```

### Performance Monitor Configuration

```python
from qem_bench.monitoring import PerformanceMonitorConfig

config = PerformanceMonitorConfig(
    enabled=True,
    max_records_per_operation=1000,
    enable_profiling=True,
    auto_export_interval=3600,  # 1 hour
    export_directory="/path/to/exports"
)

performance_monitor = PerformanceMonitor(config=config)
```

### Alert Rules Configuration

```python
from qem_bench.monitoring import AlertRule, AlertSeverity, AlertType

# Create custom alert rules
alert_manager.add_rule(AlertRule(
    name="high_execution_time",
    metric_name="zne_execution_time",
    condition="greater_than",
    threshold=300.0,  # 5 minutes
    severity=AlertSeverity.HIGH,
    alert_type=AlertType.PERFORMANCE,
    consecutive_violations=2
))
```

## Export Formats

### Metrics Export

```python
# Export to JSON
metrics_collector.export_metrics("metrics.json", format="json")

# Export to CSV  
metrics_collector.export_metrics("metrics.csv", format="csv")

# Export to Prometheus format
metrics_collector.export_metrics("metrics.prom", format="prometheus")
```

### Dashboard Export

```python
# Export dashboard data
dashboard.export_dashboard_data("dashboard.json", format="json")

# Export health report
health_checker.export_health_report("health.json", include_history=True)
```

## Benefits

1. **Comprehensive Observability**: Complete visibility into system performance, resource usage, and health
2. **Proactive Issue Detection**: Early warning through health checks and alerting
3. **Performance Optimization**: Detailed timing and profiling data for optimization
4. **Resource Efficiency**: Track and optimize quantum resource usage
5. **Quality Assurance**: Automated validation of dependencies and system requirements
6. **Troubleshooting**: Rich diagnostic information for problem resolution
7. **Trend Analysis**: Historical data analysis for capacity planning
8. **Minimal Overhead**: Lightweight design with optional components
9. **Integration Friendly**: Non-intrusive integration with existing code
10. **Extensible**: Plugin architecture for custom monitoring components

## Key Features

- **Lightweight and Optional**: Can be disabled entirely or selectively
- **Non-intrusive**: Doesn't interfere with existing functionality
- **Thread-safe**: Safe for concurrent use
- **Configurable**: Extensive configuration options
- **Extensible**: Easy to add custom monitors and health checks
- **Export Ready**: Multiple export formats for integration
- **Real-time**: Live monitoring and alerting
- **Historical**: Time-series data for trend analysis
- **Comprehensive**: Covers all aspects of quantum computing workflows

This monitoring system transforms QEM-Bench from a simple mitigation library into a comprehensive, observable, and maintainable quantum computing platform suitable for production use.