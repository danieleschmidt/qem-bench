"""
Quantum-Inspired Task Planning Example

Demonstrates the integration of quantum-inspired planning with QEM infrastructure
for optimized quantum circuit execution scheduling.
"""

import numpy as np
import jax.numpy as jnp
from datetime import datetime, timedelta

# Import QEM-Planning integration
from qem_bench.planning import (
    QuantumInspiredPlanner, PlanningConfig, Task, TaskState,
    QuantumTaskOptimizer, OptimizationStrategy,
    QuantumScheduler, SchedulingPolicy,
    PlanningAnalyzer, ComplexityMeasure
)

from qem_bench.planning.integration import QEMPlannerIntegration, QEMTask


def create_sample_circuit_tasks():
    """Create sample quantum circuit tasks for planning"""
    tasks = []
    
    # Task 1: Quantum Volume benchmark
    qv_circuit = {
        'num_qubits': 5,
        'depth': 10,
        'gate_count': 50,
        'circuit_type': 'quantum_volume'
    }
    
    task1 = QEMTask(
        id='qv_benchmark_5q',
        name='Quantum Volume 5-qubit Benchmark',
        complexity=5.0,
        priority=0.8,
        duration_estimate=30.0,
        resources={'qubits': 5, 'memory': 1.0, 'compute': 2.0},
        requires_zne=True,
        requires_cdr=True,
        metadata={'circuit_spec': qv_circuit}
    )
    tasks.append(task1)
    
    # Task 2: VQE ansatz optimization
    vqe_circuit = {
        'num_qubits': 4,
        'depth': 20,
        'gate_count': 80,
        'circuit_type': 'vqe_ansatz'
    }
    
    task2 = QEMTask(
        id='vqe_h2_optimization',
        name='VQE H2 Molecule Optimization',
        complexity=8.0,
        priority=0.9,
        duration_estimate=60.0,
        resources={'qubits': 4, 'memory': 2.0, 'compute': 4.0},
        dependencies=['qv_benchmark_5q'],  # Depends on calibration
        requires_zne=True,
        requires_pec=True,
        metadata={'circuit_spec': vqe_circuit}
    )
    tasks.append(task2)
    
    # Task 3: Quantum Fourier Transform
    qft_circuit = {
        'num_qubits': 6,
        'depth': 15,
        'gate_count': 90,
        'circuit_type': 'qft'
    }
    
    task3 = QEMTask(
        id='qft_6qubit',
        name='6-Qubit Quantum Fourier Transform',
        complexity=6.5,
        priority=0.7,
        duration_estimate=45.0,
        resources={'qubits': 6, 'memory': 1.5, 'compute': 3.0},
        requires_vd=True,
        metadata={'circuit_spec': qft_circuit}
    )
    tasks.append(task3)
    
    # Task 4: Quantum error correction syndrome measurement
    qec_circuit = {
        'num_qubits': 9,
        'depth': 5,
        'gate_count': 36,
        'circuit_type': 'qec_syndrome'
    }
    
    task4 = QEMTask(
        id='qec_syndrome_measurement',
        name='9-Qubit QEC Syndrome Measurement',
        complexity=4.0,
        priority=1.0,  # High priority for error correction
        duration_estimate=20.0,
        resources={'qubits': 9, 'memory': 0.8, 'compute': 1.5},
        requires_cdr=True,
        metadata={'circuit_spec': qec_circuit}
    )
    tasks.append(task4)
    
    # Task 5: Quantum machine learning classifier
    qml_circuit = {
        'num_qubits': 8,
        'depth': 25,
        'gate_count': 150,
        'circuit_type': 'qml_classifier'
    }
    
    task5 = QEMTask(
        id='qml_classification',
        name='Quantum ML Binary Classifier',
        complexity=12.0,
        priority=0.6,
        duration_estimate=90.0,
        resources={'qubits': 8, 'memory': 3.0, 'compute': 6.0},
        dependencies=['qec_syndrome_measurement'],  # Needs error correction first
        requires_zne=True,
        requires_pec=True,
        requires_vd=True,
        metadata={'circuit_spec': qml_circuit}
    )
    tasks.append(task5)
    
    return tasks


def demonstrate_basic_planning():
    """Demonstrate basic quantum-inspired task planning"""
    print("🧠 Quantum-Inspired Task Planning Demo")
    print("=" * 50)
    
    # Create planning configuration
    config = PlanningConfig(
        max_iterations=500,
        convergence_threshold=1e-4,
        quantum_annealing_schedule="exponential",
        superposition_width=0.15,
        entanglement_strength=0.6,
        interference_factor=0.4,
        use_gpu=True,
        enable_monitoring=True
    )
    
    # Initialize planner
    planner = QuantumInspiredPlanner(config)
    
    # Create and add tasks
    tasks = create_sample_circuit_tasks()
    for task in tasks:
        planner.add_task(task)
        print(f"📋 Added task: {task.name} (complexity: {task.complexity:.1f})")
    
    print(f"\n🎯 Planning {len(tasks)} quantum circuit tasks...")
    
    # Generate optimal plan
    start_time = datetime.now()
    planning_result = planner.plan(objective="minimize_completion_time")
    planning_time = (datetime.now() - start_time).total_seconds()
    
    # Display results
    print(f"⚡ Planning completed in {planning_time:.3f} seconds")
    print(f"🔬 Quantum fidelity: {planning_result['quantum_fidelity']:.3f}")
    print(f"🎯 Convergence: {'✅' if planning_result['convergence_achieved'] else '❌'}")
    print(f"⏱️  Total completion time: {planning_result['total_time']:.1f} seconds")
    
    # Show schedule
    print(f"\n📅 Optimized Schedule:")
    for i, event in enumerate(planning_result['schedule']):
        print(f"  {i+1}. {event['task_name']}")
        print(f"     ⏰ {event['start_time']:.1f}s - {event['end_time']:.1f}s")
        print(f"     🔧 Resources: {event['resources']}")
        print(f"     ⭐ Priority: {event['priority']:.1f}")
        
    return planning_result


def demonstrate_optimization_strategies():
    """Demonstrate different quantum optimization strategies"""
    print("\n🚀 Quantum Optimization Strategies Comparison")
    print("=" * 50)
    
    # Create tasks
    tasks = create_sample_circuit_tasks()
    task_dict = {task.id: task for task in tasks}
    
    # Test different strategies
    strategies = [
        OptimizationStrategy.QUANTUM_ANNEALING,
        OptimizationStrategy.VARIATIONAL_QUANTUM,
        OptimizationStrategy.ADIABATIC_QUANTUM,
        OptimizationStrategy.QUANTUM_APPROXIMATE
    ]
    
    optimizer = QuantumTaskOptimizer()
    results = {}
    
    for strategy in strategies:
        print(f"\n🔬 Testing {strategy.value}...")
        
        start_time = datetime.now()
        result = optimizer.optimize(task_dict, strategy=strategy)
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        results[strategy.value] = {
            'objective_value': result.objective_value,
            'quantum_fidelity': result.quantum_fidelity,
            'convergence_iterations': result.convergence_iterations,
            'optimization_time': optimization_time,
            'solution_quality': 1.0 / (result.objective_value + 1e-6)
        }
        
        print(f"  ⚡ Time: {optimization_time:.3f}s")
        print(f"  🎯 Objective: {result.objective_value:.2f}")
        print(f"  🔬 Fidelity: {result.quantum_fidelity:.3f}")
        print(f"  🔄 Iterations: {result.convergence_iterations}")
    
    # Find best strategy
    best_strategy = min(results.keys(), key=lambda k: results[k]['objective_value'])
    print(f"\n🏆 Best strategy: {best_strategy}")
    print(f"   📊 Objective value: {results[best_strategy]['objective_value']:.2f}")
    
    return results


def demonstrate_real_time_scheduling():
    """Demonstrate real-time quantum scheduling"""
    print("\n⚡ Real-Time Quantum Scheduling Demo")
    print("=" * 50)
    
    # Create scheduler with quantum priority policy
    config = PlanningConfig(enable_monitoring=True)
    scheduler = QuantumScheduler(config, SchedulingPolicy.QUANTUM_PRIORITY)
    
    # Add quantum resources
    scheduler.add_resource('qubits', 20, quantum_efficiency=0.95)
    scheduler.add_resource('memory', 10.0, quantum_efficiency=0.9)
    scheduler.add_resource('compute', 15.0, quantum_efficiency=0.85)
    
    print("🔧 Added quantum resources:")
    print("   • 20 qubits (95% efficiency)")
    print("   • 10 GB memory (90% efficiency)")  
    print("   • 15 compute units (85% efficiency)")
    
    # Start real-time scheduler
    scheduler.start_scheduler()
    print("\n🚀 Started real-time quantum scheduler")
    
    # Submit tasks dynamically
    tasks = create_sample_circuit_tasks()
    for i, task in enumerate(tasks):
        task_id = scheduler.submit_task(task)
        print(f"📋 Submitted: {task.name} (ID: {task_id})")
        
        # Simulate dynamic submission
        import time
        time.sleep(0.5)
    
    # Let scheduler run for a bit
    print("\n⏳ Running scheduler for 5 seconds...")
    time.sleep(5)
    
    # Get status
    status = scheduler.get_schedule_status()
    resource_status = scheduler.get_resource_status()
    
    print(f"\n📊 Scheduler Status:")
    print(f"   Total tasks: {status['total_tasks']}")
    print(f"   Completed: {status['completed_tasks']}")
    print(f"   Pending: {status['pending_tasks']}")
    print(f"   Total quantum priority: {status['total_quantum_priority']:.2f}")
    
    print(f"\n🔧 Resource Status:")
    for resource_id, resource_info in resource_status.items():
        utilization = resource_info['utilization'] * 100
        efficiency = resource_info['quantum_efficiency'] * 100
        print(f"   {resource_id}: {utilization:.1f}% utilized, {efficiency:.1f}% efficient")
    
    # Stop scheduler
    scheduler.stop_scheduler()
    print("\n⏹️  Stopped quantum scheduler")
    
    return status, resource_status


def demonstrate_complexity_analysis():
    """Demonstrate quantum task complexity analysis"""
    print("\n🔬 Quantum Task Complexity Analysis")
    print("=" * 50)
    
    analyzer = PlanningAnalyzer()
    tasks = create_sample_circuit_tasks()
    
    # Analyze each task
    complexities = []
    for task in tasks:
        complexity = analyzer.analyze_task_complexity(task, ComplexityMeasure.QUANTUM_VOLUME)
        complexities.append(complexity)
        
        print(f"\n📋 {task.name}:")
        print(f"   Quantum Volume: {complexity.quantum_volume:.2f}")
        print(f"   Entanglement Entropy: {complexity.entanglement_entropy:.2f}")
        print(f"   Circuit Depth: {complexity.circuit_depth:.2f}")
        print(f"   Resource Complexity: {complexity.resource_complexity:.2f}")
        print(f"   Overall Complexity: {complexity.overall_complexity:.2f}")
    
    # Generate complexity report
    task_dict = {task.id: task for task in tasks}
    report = analyzer.generate_complexity_report(task_dict)
    
    print(f"\n📊 Complexity Report Summary:")
    stats = report['complexity_statistics']
    print(f"   Mean complexity: {stats['mean']:.2f}")
    print(f"   Std deviation: {stats['stdev']:.2f}")
    print(f"   Range: {stats['min']:.2f} - {stats['max']:.2f}")
    
    # Most complex task
    most_complex = report['most_complex_tasks'][0]
    print(f"   Most complex: {most_complex.task_id} ({most_complex.overall_complexity:.2f})")
    
    return report


def demonstrate_qem_integration():
    """Demonstrate full QEM integration with planning"""
    print("\n🔗 QEM-Planning Integration Demo")
    print("=" * 50)
    
    # Create integrated QEM planner
    config = PlanningConfig(
        max_iterations=300,
        use_gpu=True,
        enable_monitoring=True
    )
    
    qem_planner = QEMPlannerIntegration(config)
    
    # Create QEM-specific tasks
    tasks = []
    
    # Bell state preparation with ZNE
    bell_task = qem_planner.create_qem_task(
        task_id='bell_state_prep',
        name='Bell State Preparation',
        circuit_spec={'num_qubits': 2, 'depth': 3, 'gate_count': 6},
        mitigation_requirements={'zne': True, 'cdr': True}
    )
    tasks.append(bell_task)
    
    # GHZ state with comprehensive mitigation
    ghz_task = qem_planner.create_qem_task(
        task_id='ghz_state_8q',
        name='8-Qubit GHZ State',
        circuit_spec={'num_qubits': 8, 'depth': 8, 'gate_count': 15},
        mitigation_requirements={'zne': True, 'pec': True, 'vd': True}
    )
    tasks.append(ghz_task)
    
    # Add dependency
    ghz_task.dependencies = ['bell_state_prep']
    
    print(f"🎯 Created {len(tasks)} QEM tasks with mitigation requirements")
    
    # Plan execution
    execution_constraints = {
        'max_qubits': 10,
        'max_memory': 5.0,
        'max_compute': 8.0
    }
    
    print("\n📋 Planning QEM execution with constraints...")
    execution_plan = qem_planner.plan_qem_execution(tasks, execution_constraints)
    
    print(f"✅ Planning completed!")
    print(f"   Total time: {execution_plan['total_time']:.1f}s")
    print(f"   Quantum fidelity: {execution_plan['quantum_fidelity']:.3f}")
    print(f"   Expected error reduction: {execution_plan['qem_integration']['expected_error_reduction']:.1%}")
    
    # Execute plan
    print("\n🚀 Executing QEM plan...")
    execution_results = qem_planner.execute_qem_plan(execution_plan)
    
    print(f"✅ Execution completed!")
    print(f"   Successful tasks: {execution_results['successful_tasks']}/{len(tasks)}")
    print(f"   Average fidelity: {execution_results['average_fidelity']:.3f}")
    print(f"   Total error reduction: {execution_results['total_error_reduction']:.2f}")
    
    return execution_plan, execution_results


def main():
    """Run all quantum planning demonstrations"""
    print("🌟 QEM-Bench Quantum-Inspired Task Planning")
    print("=" * 60)
    print("Comprehensive demonstration of quantum optimization for task scheduling\n")
    
    try:
        # Basic planning
        planning_result = demonstrate_basic_planning()
        
        # Optimization strategies
        strategy_results = demonstrate_optimization_strategies()
        
        # Real-time scheduling
        scheduler_status, resource_status = demonstrate_real_time_scheduling()
        
        # Complexity analysis
        complexity_report = demonstrate_complexity_analysis()
        
        # QEM integration
        execution_plan, execution_results = demonstrate_qem_integration()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\nKey Results:")
        print(f"• Quantum fidelity achieved: {planning_result['quantum_fidelity']:.3f}")
        print(f"• Best optimization strategy: {min(strategy_results.keys(), key=lambda k: strategy_results[k]['objective_value'])}")
        print(f"• Resource utilization: {np.mean([r['utilization'] for r in resource_status.values()])*100:.1f}%")
        print(f"• QEM error reduction: {execution_results['total_error_reduction']:.2f}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()