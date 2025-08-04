"""
Example demonstrating the QEM-Bench monitoring and health check system.

This example shows how to:
1. Set up comprehensive monitoring for quantum error mitigation experiments
2. Use the MonitoredZeroNoiseExtrapolation with integrated monitoring
3. Generate dashboard reports and analyze performance
4. Export monitoring data for further analysis
"""

import time
import numpy as np
from typing import Optional

# Import QEM-Bench components
from qem_bench import (
    # Monitoring framework
    SystemMonitor, PerformanceMonitor, QuantumResourceMonitor, AlertManager,
    # Health checking
    HealthChecker, HealthStatus,
    # Metrics collection
    MetricsCollector, CircuitMetrics, NoiseMetrics,
    # Enhanced mitigation with monitoring
    MonitoredZeroNoiseExtrapolation,
    # Dashboard and reporting
    MonitoringDashboard
)

# Example quantum circuit and backend (simplified for demonstration)
class MockQuantumCircuit:
    """Mock quantum circuit for demonstration."""
    def __init__(self, num_qubits: int = 4, num_gates: int = 20):
        self.num_qubits = num_qubits
        self.num_gates = num_gates
        self.depth = max(1, num_gates // num_qubits)
        
        # Simulate gate structure
        self.gate_counts = {
            'cx': num_gates // 3,
            'h': num_gates // 4,
            'rz': num_gates // 3,
            'measure': num_qubits
        }
    
    def count_ops(self):
        return self.gate_counts


class MockQuantumBackend:
    """Mock quantum backend for demonstration."""
    def __init__(self, name: str = "mock_backend"):
        self.name = name
        self.noise_level = 0.01  # 1% error rate
    
    def run(self, circuit, shots: int = 1024):
        # Simulate execution time
        execution_time = 0.1 + np.random.exponential(0.5)
        time.sleep(execution_time)
        
        # Simulate noisy results
        ideal_value = 1.0
        noise = np.random.normal(0, self.noise_level * np.sqrt(shots))
        measured_value = ideal_value + noise
        
        return MockResult(measured_value, shots)


class MockResult:
    """Mock quantum result."""
    def __init__(self, expectation_value: float, shots: int):
        self.expectation_value = expectation_value
        self.shots = shots


def setup_comprehensive_monitoring():
    """Set up all monitoring components."""
    print("🔧 Setting up comprehensive monitoring system...")
    
    # System monitoring
    system_monitor = SystemMonitor()
    system_monitor.start()
    print("✓ System monitor started")
    
    # Performance monitoring
    performance_monitor = PerformanceMonitor()
    print("✓ Performance monitor initialized")
    
    # Resource monitoring
    resource_monitor = QuantumResourceMonitor()
    print("✓ Quantum resource monitor initialized")
    
    # Health checking
    health_checker = HealthChecker()
    health_checker.start_monitoring()
    print("✓ Health checker started")
    
    # Metrics collection
    metrics_collector = MetricsCollector()
    print("✓ Metrics collector initialized")
    
    # Alert management
    alert_manager = AlertManager()
    print("✓ Alert manager initialized")
    
    # Dashboard
    dashboard = MonitoringDashboard(
        system_monitor=system_monitor,
        performance_monitor=performance_monitor,
        resource_monitor=resource_monitor,
        health_checker=health_checker,
        metrics_collector=metrics_collector,
        alert_manager=alert_manager
    )
    print("✓ Monitoring dashboard created")
    
    return {
        'system_monitor': system_monitor,
        'performance_monitor': performance_monitor,
        'resource_monitor': resource_monitor,
        'health_checker': health_checker,
        'metrics_collector': metrics_collector,
        'alert_manager': alert_manager,
        'dashboard': dashboard
    }


def demonstrate_circuit_analysis():
    """Demonstrate circuit analysis capabilities."""
    print("\\n📊 Demonstrating circuit analysis...")
    
    circuit_analyzer = CircuitMetrics()
    
    # Analyze different types of circuits
    circuits = [
        MockQuantumCircuit(4, 15),   # Small circuit
        MockQuantumCircuit(8, 40),   # Medium circuit
        MockQuantumCircuit(12, 80),  # Large circuit
    ]
    
    analyses = []
    for i, circuit in enumerate(circuits):
        analysis = circuit_analyzer.analyze_circuit(
            circuit, 
            circuit_id=f"demo_circuit_{i+1}"
        )
        analyses.append(analysis)
        print(f"  Circuit {i+1}: {analysis.num_qubits} qubits, {analysis.num_gates} gates, depth {analysis.circuit_depth}")
    
    # Get batch summary
    summary = circuit_analyzer.get_batch_summary(analyses)
    print(f"  Batch summary: {summary['total_circuits']} circuits analyzed")
    print(f"  Average gates per circuit: {summary['gate_statistics']['avg']:.1f}")
    
    return analyses


def demonstrate_monitored_zne():
    """Demonstrate monitored Zero-Noise Extrapolation."""
    print("\\n🎯 Demonstrating monitored ZNE execution...")
    
    # Create monitored ZNE instance
    monitored_zne = MonitoredZeroNoiseExtrapolation(
        noise_factors=[1.0, 1.5, 2.0, 2.5, 3.0],
        extrapolator="richardson"
    )
    
    # Create test circuit and backend
    circuit = MockQuantumCircuit(num_qubits=6, num_gates=30)
    backend = MockQuantumBackend("demo_backend")
    
    print("  Executing ZNE with comprehensive monitoring...")
    
    # Execute with monitoring
    result = monitored_zne.mitigate(
        circuit=circuit,
        backend=backend,
        shots=1024
    )
    
    print(f"  Raw value: {result.raw_value:.4f}")
    print(f"  Mitigated value: {result.mitigated_value:.4f}")
    
    if hasattr(result, 'monitoring_data'):
        monitoring_data = result.monitoring_data
        print(f"  Execution ID: {monitoring_data.get('execution_id', 'N/A')}")
        
        circuit_info = monitoring_data.get('circuit_analysis', {})
        if circuit_info:
            print(f"  Circuit analysis: {circuit_info['num_qubits']} qubits, {circuit_info['num_gates']} gates")
        
        performance_info = monitoring_data.get('performance_stats', {})
        if performance_info:
            print(f"  Execution time: {performance_info.get('avg_duration_ms', 0):.1f}ms")
    
    # Get monitoring summaries
    print("\\n📈 Monitoring summaries:")
    performance_summary = monitored_zne.get_performance_summary()
    if isinstance(performance_summary, str):
        print("  Performance summary available")
    
    resource_summary = monitored_zne.get_resource_summary()
    if 'error' not in resource_summary:
        print(f"  Resource usage tracked: {resource_summary.get('total_executions', 0)} executions")
    
    health_summary = monitored_zne.get_health_summary()
    if 'error' not in health_summary:
        print(f"  System health: {health_summary.get('overall_status', 'unknown')}")
    
    return monitored_zne, result


def demonstrate_dashboard_reporting(monitors):
    """Demonstrate dashboard and reporting capabilities."""
    print("\\n📋 Generating dashboard reports...")
    
    dashboard = monitors['dashboard']
    
    # Generate executive summary
    print("\\n=== EXECUTIVE SUMMARY ===")
    executive_summary = dashboard.generate_executive_summary()
    print(executive_summary)
    
    # Generate detailed status report
    print("\\n=== DETAILED STATUS REPORT ===")
    status_report = dashboard.generate_status_report(detailed=False)  # Brief version for demo
    print(status_report[:1000] + "\\n... (truncated for demo)")
    
    # Analyze performance trends
    trends = dashboard.analyze_performance_trends(duration_hours=1)
    if 'error' not in trends:
        print(f"\\n📊 Performance trends analyzed over {trends['analysis_duration_hours']} hours")
        for op_name, op_data in trends.get('operations', {}).items():
            print(f"  {op_name}: {op_data['total_executions']} executions, {op_data['trend_direction']} trend")


def demonstrate_metrics_and_alerts(monitors):
    """Demonstrate metrics collection and alerting."""
    print("\\n📊 Demonstrating metrics and alerts...")
    
    metrics_collector = monitors['metrics_collector']
    alert_manager = monitors['alert_manager']
    
    # Record some example metrics
    for i in range(10):
        # Simulate varying performance metrics
        execution_time = 1.0 + np.random.exponential(0.5)
        fidelity = 0.9 + np.random.normal(0, 0.05)
        
        metrics_collector.record_histogram("execution_time", execution_time, 
                                         labels={"method": "zne", "iteration": str(i)})
        metrics_collector.set_gauge("current_fidelity", fidelity,
                                  labels={"experiment": "demo"})
        metrics_collector.increment_counter("experiments_completed",
                                          labels={"status": "success"})
        
        # Check for alerts
        alert_manager.check_metric("execution_time", execution_time, "demo_system")
        alert_manager.check_metric("current_fidelity", fidelity, "demo_system")
    
    # Get metrics overview
    overview = metrics_collector.get_metrics_overview()
    print(f"  Collected {overview['total_records']} metric records")
    print(f"  Tracking {overview['total_metrics']} different metrics")
    
    # Check for active alerts
    active_alerts = alert_manager.get_active_alerts()
    if active_alerts:
        print(f"  ⚠ {len(active_alerts)} active alerts")
        for alert in active_alerts[:3]:  # Show first 3
            print(f"    {alert.severity.value}: {alert.title}")
    else:
        print("  ✓ No active alerts")


def demonstrate_health_monitoring(monitors):
    """Demonstrate health monitoring capabilities."""
    print("\\n🏥 Demonstrating health monitoring...")
    
    health_checker = monitors['health_checker']
    
    # Run health checks
    health_results = health_checker.run_all_checks()
    print(f"  Ran health checks on {len(health_results)} providers")
    
    for provider_name, result in health_results.items():
        status_icon = "✓" if result.status.value == "healthy" else "⚠" if result.status.value == "warning" else "✗"
        print(f"    {status_icon} {provider_name}: {result.status.value}")
        if result.status.value != "healthy" and result.recommendations:
            print(f"      Recommendation: {result.recommendations[0]}")
    
    # Get overall health summary
    health_summary = health_checker.get_health_summary()
    print(f"  Overall system health: {health_summary['overall_status']}")


def demonstrate_data_export(monitors, monitored_zne):
    """Demonstrate data export capabilities."""
    print("\\n💾 Demonstrating data export...")
    
    try:
        # Export dashboard data
        dashboard = monitors['dashboard']
        dashboard.export_dashboard_data("/tmp/dashboard_export.json")
        print("  ✓ Dashboard data exported to /tmp/dashboard_export.json")
        
        # Export metrics in multiple formats
        metrics_collector = monitors['metrics_collector']
        metrics_collector.export_metrics("/tmp/metrics_export.json", format="json")
        print("  ✓ Metrics exported to JSON format")
        
        # Export monitoring data from ZNE
        monitored_zne.export_monitoring_data("/tmp/zne_monitoring.json")
        print("  ✓ ZNE monitoring data exported")
        
        # Export health report
        health_checker = monitors['health_checker']
        health_checker.export_health_report("/tmp/health_report.json", include_history=True)
        print("  ✓ Health report exported")
        
    except Exception as e:
        print(f"  ⚠ Export demo skipped (requires write permissions): {e}")


def cleanup_monitoring(monitors):
    """Clean up monitoring resources."""
    print("\\n🧹 Cleaning up monitoring resources...")
    
    try:
        if monitors['system_monitor']:
            monitors['system_monitor'].stop()
        
        if monitors['health_checker']:
            monitors['health_checker'].stop_monitoring()
        
        print("  ✓ Monitoring resources cleaned up")
    
    except Exception as e:
        print(f"  ⚠ Cleanup warning: {e}")


def main():
    """Main demonstration function."""
    print("🚀 QEM-Bench Monitoring System Demonstration")
    print("=" * 50)
    
    try:
        # Set up monitoring
        monitors = setup_comprehensive_monitoring()
        
        # Wait a moment for monitors to initialize
        time.sleep(2)
        
        # Demonstrate circuit analysis
        circuit_analyses = demonstrate_circuit_analysis()
        
        # Demonstrate monitored ZNE
        monitored_zne, zne_result = demonstrate_monitored_zne()
        
        # Wait for some monitoring data to accumulate
        time.sleep(1)
        
        # Demonstrate metrics and alerts
        demonstrate_metrics_and_alerts(monitors)
        
        # Demonstrate health monitoring
        demonstrate_health_monitoring(monitors)
        
        # Generate dashboard reports
        demonstrate_dashboard_reporting(monitors)
        
        # Demonstrate data export
        demonstrate_data_export(monitors, monitored_zne)
        
        print("\\n✅ Monitoring demonstration completed successfully!")
        print("\\nKey features demonstrated:")
        print("  • Comprehensive system monitoring")
        print("  • Performance profiling and analysis")
        print("  • Quantum resource tracking")
        print("  • Health checking and validation")
        print("  • Metrics collection and alerting")
        print("  • Circuit and noise model analysis")
        print("  • Integrated monitoring in ZNE")
        print("  • Dashboard reporting and trends")
        print("  • Data export in multiple formats")
        
    except KeyboardInterrupt:
        print("\\n⏹ Demonstration interrupted by user")
    
    except Exception as e:
        print(f"\\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up resources
        if 'monitors' in locals():
            cleanup_monitoring(monitors)
        if 'monitored_zne' in locals():
            monitored_zne.cleanup()


if __name__ == "__main__":
    main()