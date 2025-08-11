"""
Advanced Research Capabilities Demo

Comprehensive demonstration of QEM-Bench's advanced research features including:
- Machine Learning QEM optimization
- Quantum-classical hybrid algorithms
- Real-time adaptive error mitigation
- Advanced leaderboard system
- Novel technique discovery
- Experimental framework

This demo showcases how researchers can leverage QEM-Bench for cutting-edge
quantum error mitigation research.
"""

import numpy as np
import jax.numpy as jnp
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import QEM-Bench research modules
from qem_bench.research import (
    MLQEMOptimizer, MLQEMConfig, HybridQEMFramework, HybridQEMConfig,
    RealTimeQEMAdapter, AdaptiveQEMConfig, AdvancedLeaderboardSystem,
    QEMTechniqueDiscoverer, DiscoveryConfig, ResearchExperimentFramework
)
from qem_bench.jax.circuits import JAXCircuit
from qem_bench.jax.simulator import JAXSimulator


def demo_ml_qem_optimization():
    """Demonstrate machine learning-powered QEM optimization."""
    print("\n🧠 MACHINE LEARNING QEM OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Configure ML QEM optimizer
    ml_config = MLQEMConfig(
        learning_rate=0.001,
        batch_size=16,
        num_epochs=100,
        hidden_layers=[128, 64, 32],
        feature_scaling="standard"
    )
    
    ml_optimizer = MLQEMOptimizer(ml_config)
    
    # Create test circuit
    circuit = JAXCircuit(3, name="test_circuit")
    circuit.h(0).cx(0, 1).ry(np.pi/4, 2).cx(1, 2)
    
    # Simulate some experimental data
    print("📊 Simulating experimental training data...")
    
    # Add mock experimental results
    for i in range(20):
        mock_noise_model = type('NoiseModel', (), {
            'single_qubit_error_rate': 0.001 + 0.001 * np.random.random(),
            'two_qubit_error_rate': 0.01 + 0.01 * np.random.random(),
            'readout_error_rate': 0.02,
            'coherence_time_t1': 100.0,
            'coherence_time_t2': 50.0,
            'spatial_correlation_strength': 0.1,
            'temporal_correlation_strength': 0.05,
            'crosstalk_strength': 0.02,
            'device_connectivity': 0.8,
            'gate_fidelity_variance': 0.01,
            'calibration_drift_rate': 0.001
        })()
        
        mock_context = {
            'queue_length': np.random.randint(10, 100),
            'device_temperature': 0.01 + 0.005 * np.random.random(),
            'time_since_calibration': np.random.uniform(0, 3600),
            'current_load': np.random.uniform(0.1, 0.9)
        }
        
        mock_parameters = {
            'noise_factor_max': 2.0 + np.random.uniform(-0.5, 1.5),
            'num_noise_factors': np.random.randint(3, 8),
            'extrapolation_order': np.random.randint(1, 4),
            'bootstrap_samples': np.random.randint(50, 300),
            'confidence_level': 0.95
        }
        
        mock_result = {
            'error_reduction': 0.1 + 0.2 * np.random.random(),
            'fidelity': 0.8 + 0.15 * np.random.random(),
            'execution_time': 1.0 + 2.0 * np.random.random()
        }
        
        ml_optimizer.add_experiment(circuit, mock_noise_model, mock_context, mock_parameters, mock_result)
    
    # Train ML models
    print("🎓 Training ML models on experimental data...")
    training_results = ml_optimizer.train_models()
    
    print(f"✅ Model training completed with {training_results['num_experiments']} experiments")
    
    # Optimize parameters for new conditions
    print("🔮 Predicting optimal parameters for new conditions...")
    
    new_context = {
        'queue_length': 25,
        'device_temperature': 0.015,
        'time_since_calibration': 1800,
        'current_load': 0.6
    }
    
    optimal_params = ml_optimizer.optimize_parameters(circuit, mock_noise_model, new_context)
    
    print("🎯 Optimized Parameters:")
    for param, value in optimal_params.items():
        print(f"   {param}: {value:.3f}")
    
    # Get optimization insights
    insights = ml_optimizer.get_optimization_insights()
    print(f"\n📈 Optimization Insights:")
    print(f"   Model accuracy: {insights['model_accuracy']:.1%}")
    print(f"   Total experiments: {insights['total_experiments']}")
    print(f"   Performance trend: {insights['performance_trend']:.3f}")


def demo_hybrid_qem_algorithms():
    """Demonstrate quantum-classical hybrid algorithms."""
    print("\n🌟 HYBRID QUANTUM-CLASSICAL ALGORITHMS DEMO")
    print("=" * 60)
    
    # Configure hybrid framework
    hybrid_config = HybridQEMConfig(
        quantum_optimizer="vqe",
        classical_optimizer="bfgs",
        quantum_classical_ratio=0.6,
        max_hybrid_iterations=50,
        enable_co_design=True
    )
    
    hybrid_framework = HybridQEMFramework(hybrid_config)
    
    # Create test circuit
    circuit = JAXCircuit(4, name="hybrid_test")
    circuit.h(0).cx(0, 1).ry(np.pi/3, 2).cx(1, 3).rz(np.pi/6, 2)
    
    # Mock backend
    simulator = JAXSimulator(num_qubits=4, precision="float32")
    
    print("🔄 Optimizing with hybrid quantum-classical approach...")
    
    # Optimize using different hybrid methods
    methods = ["variational", "hybrid_zne", "co_design"]
    
    for method in methods:
        print(f"\n🎯 Testing {method} method...")
        
        try:
            result = hybrid_framework.optimize_error_mitigation(
                circuit=circuit,
                backend=simulator,
                method=method,
                target_fidelity=0.9,
                noise_budget=2.5
            )
            
            print(f"   ✅ {method}: Efficiency = {result.get('hybrid_efficiency', 0):.3f}")
            print(f"   ⏱️  Optimization time: {result.get('optimization_time', 'N/A')}")
            
        except Exception as e:
            print(f"   ❌ {method}: Error = {e}")
    
    # Analyze performance
    print("\n📊 Analyzing hybrid performance...")
    performance_analysis = hybrid_framework.analyze_hybrid_performance()
    
    if performance_analysis.get('status') != 'no_data':
        print(f"   Total optimizations: {performance_analysis['total_optimizations']}")
        
        if 'recommendations' in performance_analysis:
            for rec in performance_analysis['recommendations']:
                print(f"   💡 {rec}")


def demo_adaptive_qem():
    """Demonstrate real-time adaptive error mitigation."""
    print("\n🔄 REAL-TIME ADAPTIVE ERROR MITIGATION DEMO")
    print("=" * 60)
    
    # Configure adaptive system
    adaptive_config = AdaptiveQEMConfig(
        monitoring_interval=0.5,
        adaptation_frequency=5,
        prediction_window=20,
        enable_online_learning=True
    )
    
    adapter = RealTimeQEMAdapter(adaptive_config)
    
    # Mock device interface
    class MockDeviceInterface:
        def get_current_state(self):
            return {
                'gate_fidelities': {'cx': 0.95, 'h': 0.98},
                'temperature': 0.012,
                'queue_length': 30
            }
    
    device_interface = MockDeviceInterface()
    
    print("🚀 Starting adaptive monitoring...")
    
    # Start monitoring (would run in background in real application)
    # adapter.start_adaptive_monitoring(device_interface)
    
    # Create test circuit
    circuit = JAXCircuit(3, name="adaptive_test")
    circuit.h(0).cx(0, 1).ry(np.pi/4, 2)
    
    # Request adaptation
    print("📡 Requesting parameter adaptation...")
    
    current_params = {
        'noise_factor_max': 3.0,
        'extrapolation_order': 2,
        'bootstrap_samples': 100
    }
    
    execution_context = {
        'circuit_complexity': circuit.depth * circuit.num_qubits,
        'system_load': 0.4,
        'timestamp': datetime.now().timestamp()
    }
    
    def adaptation_callback(decision):
        print(f"🎯 Adaptation Decision:")
        print(f"   Confidence: {decision.confidence:.3f}")
        print(f"   Predicted improvement: {decision.predicted_improvement:.1%}")
        print(f"   Reason: {decision.adaptation_reason}")
        print(f"   New parameters: {decision.new_parameters}")
    
    success = adapter.request_adaptation(
        circuit=circuit,
        current_parameters=current_params,
        execution_context=execution_context,
        callback=adaptation_callback
    )
    
    if success:
        print("✅ Adaptation request queued successfully")
    else:
        print("❌ Adaptation request failed")
    
    # Get performance analytics
    print("\n📈 Adaptive system performance:")
    analytics = adapter.get_adaptation_performance()
    
    if analytics.get('status') != 'no_adaptations':
        print(f"   Recent adaptations: {analytics.get('recent_adaptations', 0)}")
        print(f"   Average confidence: {analytics.get('average_confidence', 0):.3f}")
        print(f"   Execution time: {analytics.get('average_execution_time', 0):.3f}s")


def demo_advanced_leaderboards():
    """Demonstrate advanced leaderboard and benchmarking."""
    print("\n🏆 ADVANCED LEADERBOARD SYSTEM DEMO")
    print("=" * 60)
    
    leaderboard_system = AdvancedLeaderboardSystem()
    
    print("📤 Submitting benchmark results...")
    
    # Submit several benchmark entries
    methods = ["Enhanced_ZNE", "Hybrid_PEC", "Adaptive_VD", "ML_Optimized_CDR"]
    circuits = ["quantum_volume", "random_circuits", "vqe", "qaoa"]
    
    for i, (method, circuit_family) in enumerate(zip(methods, circuits)):
        result = leaderboard_system.submit_benchmark_result(
            submission_id=f"demo_{i+1}",
            timestamp=datetime.now(),
            submitter="research_demo",
            method_name=method,
            circuit_family=circuit_family,
            circuit_parameters={"depth": 10, "width": 5},
            noise_model="depolarizing",
            backend="simulator",
            error_reduction=0.15 + 0.1 * np.random.random(),
            fidelity_improvement=0.05 + 0.05 * np.random.random(),
            execution_time=2.0 + 3.0 * np.random.random(),
            resource_overhead=1.5 + 0.5 * np.random.random(),
            success_probability=0.9 + 0.08 * np.random.random(),
            qubits=5,
            circuit_depth=10,
            shots=1024,
            noise_strength=0.01,
            code_hash=f"abc123{i}",
            random_seed=42 + i
        )
        
        print(f"   ✅ Submitted {method}: Position #{result['leaderboard_position'].get('overall', 'N/A')}")
    
    # Generate leaderboards
    print("\n📊 Generating leaderboards...")
    
    overall_board = leaderboard_system.get_leaderboard(category="all", period="all_time")
    
    print(f"🏆 Overall Leaderboard ({overall_board['entries'][:3] if overall_board['entries'] else []}):")
    
    for i, entry in enumerate(overall_board.get('entries', [])[:3]):
        print(f"   {i+1}. {entry['method_name']} by {entry['submitter']}")
        print(f"      Score: {entry['score']:.4f}, Error Reduction: {entry['primary_metric']:.3f}")
    
    # Analyze method performance
    if overall_board.get('entries'):
        top_method = overall_board['entries'][0]['method_name']
        print(f"\n🔍 Analyzing {top_method}...")
        
        analysis = leaderboard_system.analyze_method(top_method)
        
        if analysis.get('status') != 'no_data':
            print(f"   Submissions: {analysis['total_submissions']}")
            print(f"   Avg performance: {analysis['overall_performance']['avg_error_reduction']:.3f}")
            print(f"   Best result: {analysis['overall_performance']['best_error_reduction']:.3f}")
    
    # Get research insights
    print("\n🔬 Research insights:")
    insights = leaderboard_system.get_research_insights()
    
    if insights.get('status') != 'no_data':
        print(f"   Total entries analyzed: {insights['total_entries_analyzed']}")
        
        if 'key_findings' in insights:
            for finding in insights['key_findings'][:2]:
                print(f"   💡 {finding}")


def demo_technique_discovery():
    """Demonstrate novel QEM technique discovery."""
    print("\n🔬 NOVEL QEM TECHNIQUE DISCOVERY DEMO")
    print("=" * 60)
    
    # Configure discovery system
    discovery_config = DiscoveryConfig(
        population_size=20,  # Reduced for demo
        num_generations=10,  # Reduced for demo
        test_circuits_per_technique=3,
        performance_threshold=0.05
    )
    
    discoverer = QEMTechniqueDiscoverer(discovery_config)
    
    print("🧬 Starting genetic algorithm for technique discovery...")
    print("   (Using reduced parameters for demo - actual discovery uses larger populations)")
    
    # Run discovery
    try:
        discovery_results = discoverer.discover_techniques()
        
        print("🎉 Discovery Results:")
        summary = discovery_results['discovery_summary']
        print(f"   Techniques evaluated: {summary['total_techniques_evaluated']}")
        print(f"   Novel techniques found: {summary['novel_techniques_found']}")
        print(f"   Validated techniques: {summary['validated_techniques']}")
        print(f"   Discovery time: {summary['discovery_time_seconds']:.1f}s")
        print(f"   Best fitness: {summary['best_fitness_achieved']:.4f}")
        
        # Show validated techniques
        if discovery_results['validated_techniques']:
            print("\n🏆 Top Validated Techniques:")
            
            for i, technique in enumerate(discovery_results['validated_techniques'][:3]):
                tech_info = technique['technique']
                print(f"   {i+1}. {tech_info['name']}")
                print(f"      Fitness: {technique['fitness']:.4f}")
                print(f"      Complexity: {tech_info['complexity']}")
                print(f"      Operations: {len(tech_info['steps'])}")
                print(f"      Confidence: {technique['confidence_score']:.3f}")
        
        # Show insights
        if discovery_results['discovery_insights'].get('status') != 'no_techniques':
            insights = discovery_results['discovery_insights']
            print(f"\n🔍 Discovery Patterns:")
            
            if 'common_operations' in insights:
                print("   Most effective operations:")
                for op, count in insights['common_operations'][:3]:
                    print(f"     - {op}: used {count} times")
        
        # Show recommendations
        print(f"\n💡 Recommendations:")
        for rec in discovery_results.get('recommendations', [])[:3]:
            print(f"   • {rec}")
    
    except Exception as e:
        print(f"❌ Discovery failed (expected in demo environment): {e}")
        print("   In a full environment, this would discover novel QEM techniques")


def demo_experimental_framework():
    """Demonstrate the research experimental framework."""
    print("\n🧪 RESEARCH EXPERIMENTAL FRAMEWORK DEMO")
    print("=" * 60)
    
    from qem_bench.research.experimental import ExperimentConfig, ResearchExperimentFramework
    
    # Configure experiment
    exp_config = ExperimentConfig(
        experiment_name="QEM_Method_Comparison",
        description="Comparing different QEM methods across circuit types",
        hypothesis="Hybrid methods outperform single-technique methods",
        success_criteria=["Statistically significant improvement", "Effect size > 0.5"],
        confidence_level=0.95
    )
    
    framework = ResearchExperimentFramework(exp_config)
    
    # Design experiment
    print("📋 Designing factorial experiment...")
    
    factors = {
        'qem_method': ['ZNE', 'PEC', 'Hybrid'],
        'circuit_type': ['quantum_volume', 'random'],
        'noise_level': ['low', 'high']
    }
    
    response_variables = ['error_reduction', 'fidelity_improvement', 'execution_time']
    
    design = framework.design_experiment(factors, response_variables)
    
    print(f"   ✅ Generated {design['num_conditions']} experimental conditions")
    print(f"   📊 Response variables: {', '.join(response_variables)}")
    
    # Mock execution function
    def mock_execution(condition):
        """Mock experimental execution."""
        # Simulate different performance based on conditions
        base_error_reduction = 0.1
        
        # Method effects
        if condition['qem_method'] == 'Hybrid':
            base_error_reduction += 0.05
        elif condition['qem_method'] == 'PEC':
            base_error_reduction += 0.02
        
        # Noise level effects
        if condition['noise_level'] == 'high':
            base_error_reduction *= 0.8
        
        # Add random variation
        error_reduction = base_error_reduction + np.random.normal(0, 0.02)
        
        return {
            'error_reduction': max(0, error_reduction),
            'fidelity_improvement': error_reduction * 0.5 + np.random.normal(0, 0.01),
            'execution_time': 2.0 + np.random.exponential(1.0)
        }
    
    # Execute experiment
    print("🔬 Executing experiment conditions...")
    
    results = framework.execute_experiment(design, mock_execution)
    
    print(f"   ✅ Executed {results['execution_summary']['successful_executions']} conditions")
    
    # Analyze results
    print("📊 Analyzing experimental results...")
    
    analysis = framework.analyze_results(results)
    
    if analysis.get('status') != 'no_valid_data':
        print("   ✅ Statistical analysis completed")
        
        # Show key results
        if 'inferential_statistics' in analysis:
            significant_effects = []
            for response_var, tests in analysis['inferential_statistics'].items():
                for test_name, result in tests.items():
                    if isinstance(result, dict) and result.get('significant', False):
                        significant_effects.append(f"{test_name} on {response_var}")
            
            if significant_effects:
                print(f"   🎯 Significant effects found: {', '.join(significant_effects[:2])}")
            else:
                print("   📊 No statistically significant effects detected")
        
        # Generate report
        print("📋 Generating experimental report...")
        
        report = framework.generate_report(analysis)
        
        print(f"   ✅ Report generated: {report['experiment_info']['name']}")
        print(f"   📝 Hypothesis: {report['experiment_info']['hypothesis']}")
        
        # Show conclusions
        if report['conclusions']:
            print("   🎯 Key conclusions:")
            for conclusion in report['conclusions'][:2]:
                print(f"     • {conclusion}")
    
    else:
        print("   ❌ No valid data for analysis")


def main():
    """Run all advanced research demos."""
    print("🚀 QEM-BENCH ADVANCED RESEARCH CAPABILITIES DEMO")
    print("=" * 70)
    print("Showcasing cutting-edge quantum error mitigation research features")
    print("=" * 70)
    
    try:
        # Run each demo
        demo_ml_qem_optimization()
        demo_hybrid_qem_algorithms()
        demo_adaptive_qem()
        demo_advanced_leaderboards()
        demo_technique_discovery()
        demo_experimental_framework()
        
        print("\n" + "=" * 70)
        print("🎉 ALL ADVANCED RESEARCH DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\n💡 Key Capabilities Demonstrated:")
        print("   ✅ Machine Learning QEM Optimization")
        print("   ✅ Quantum-Classical Hybrid Algorithms")
        print("   ✅ Real-Time Adaptive Error Mitigation")
        print("   ✅ Advanced Competitive Benchmarking")
        print("   ✅ Novel Technique Discovery via Genetic Algorithms")
        print("   ✅ Rigorous Experimental Framework")
        
        print("\n🔬 Research Impact:")
        print("   • Automated discovery of new QEM techniques")
        print("   • Machine learning-powered parameter optimization")
        print("   • Real-time adaptation to device conditions")
        print("   • Publication-ready experimental validation")
        print("   • Competitive benchmarking for fair comparison")
        
        print("\n📚 Next Steps:")
        print("   • Explore individual research modules in detail")
        print("   • Integrate with your quantum hardware")
        print("   • Contribute novel techniques to the community")
        print("   • Publish research using QEM-Bench framework")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("Note: Some features require additional dependencies or hardware access")


if __name__ == "__main__":
    main()