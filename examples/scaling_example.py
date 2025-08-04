"""
Comprehensive example of QEM-Bench auto-scaling and distributed computing features.

This example demonstrates:
1. Setting up auto-scaling infrastructure
2. Configuring multiple quantum backends
3. Running distributed ZNE experiments
4. Resource optimization and cost management
5. Performance monitoring and analytics
"""

import asyncio
import numpy as np
from typing import List, Dict, Any

# Import QEM-Bench scaling components
from qem_bench.scaling import (
    AutoScaler, ScalingPolicy, WorkloadAnalyzer, ResourceScheduler, 
    CostOptimizer, Budget, DistributedExecutor, BackendBalancer,
    ResourceOptimizer, BackendOrchestrator, SpotInstanceManager
)
from qem_bench.scaling.scaling_aware_zne import (
    ScalingAwareZeroNoiseExtrapolation, ScalingAwareZNEConfig
)
from qem_bench.scaling.cloud_providers import (
    create_cloud_provider, CloudProvider, InstanceType, AutoScalingGroup
)
from qem_bench.scaling.backend_orchestrator import (
    QuantumBackendInfo, BackendType, CalibrationStatus, BackendHealth
)
from qem_bench.security import SecureConfig


async def setup_scaling_infrastructure():
    """Set up the complete auto-scaling infrastructure."""
    print("🚀 Setting up QEM-Bench scaling infrastructure...")
    
    # 1. Configure security
    secure_config = SecureConfig()
    
    # 2. Set up auto-scaling policy
    scaling_policy = ScalingPolicy(
        min_instances=2,
        max_instances=20,
        cpu_scale_up_threshold=75.0,
        cpu_scale_down_threshold=25.0,
        scale_up_cooldown=300.0,  # 5 minutes
        scale_down_cooldown=600.0,  # 10 minutes
        max_cost_per_hour=50.0,  # $50/hour budget
        cost_optimization_enabled=True
    )
    
    # 3. Initialize auto-scaler
    auto_scaler = AutoScaler(policy=scaling_policy, config=secure_config)
    await auto_scaler.start()
    
    # 4. Set up workload analyzer
    workload_analyzer = WorkloadAnalyzer(config=secure_config)
    await workload_analyzer.start_monitoring()
    
    # 5. Initialize resource scheduler
    resource_scheduler = ResourceScheduler(config=secure_config)
    await resource_scheduler.start()
    
    # 6. Set up cost optimizer with budget
    budget = Budget(
        total_budget=1000.0,  # $1000 monthly budget
        period_days=30,
        warning_threshold=0.8,
        critical_threshold=0.95
    )
    cost_optimizer = CostOptimizer(budget=budget, config=secure_config)
    await cost_optimizer.start()
    
    # 7. Initialize distributed executor
    distributed_executor = DistributedExecutor(config=secure_config)
    await distributed_executor.start()
    
    # 8. Set up backend balancer
    backend_balancer = BackendBalancer(config=secure_config)
    await backend_balancer.start()
    
    # 9. Initialize resource optimizer
    resource_optimizer = ResourceOptimizer(config=secure_config)
    
    # 10. Set up backend orchestrator
    backend_orchestrator = BackendOrchestrator(config=secure_config)
    await backend_orchestrator.start()
    
    print("✅ Scaling infrastructure ready!")
    
    return {
        "auto_scaler": auto_scaler,
        "workload_analyzer": workload_analyzer,
        "resource_scheduler": resource_scheduler,
        "cost_optimizer": cost_optimizer,
        "distributed_executor": distributed_executor,
        "backend_balancer": backend_balancer,
        "resource_optimizer": resource_optimizer,
        "backend_orchestrator": backend_orchestrator,
        "secure_config": secure_config
    }


def create_sample_quantum_backends() -> List[QuantumBackendInfo]:
    """Create sample quantum backends for demonstration."""
    print("🔧 Creating sample quantum backends...")
    
    backends = []
    
    # IBM Quantum backend
    ibm_backend = QuantumBackendInfo(
        id="ibm_quantum_5q",
        name="IBM Quantum 5-qubit",
        provider="IBM",
        backend_type=BackendType.HARDWARE,
        num_qubits=5,
        coupling_map=[(0, 1), (1, 2), (2, 3), (3, 4), (1, 0), (2, 1), (3, 2), (4, 3)],
        gate_set=["x", "sx", "rz", "cx", "id", "reset"],
        gate_fidelities={
            "x": 0.9995,
            "sx": 0.9993,
            "cx": 0.995,
            "rz": 0.9999
        },
        readout_fidelities=[0.97, 0.96, 0.98, 0.97, 0.96],
        calibration_status=CalibrationStatus.FRESH,
        last_calibration_time=1704067200.0,  # Recent calibration
        success_rate=0.98,
        cost_per_shot=0.0001
    )
    backends.append(ibm_backend)
    
    # Google Quantum AI backend
    google_backend = QuantumBackendInfo(
        id="google_sycamore_7q",
        name="Google Sycamore 7-qubit",
        provider="Google",
        backend_type=BackendType.HARDWARE,
        num_qubits=7,
        coupling_map=[(i, i+1) for i in range(6)] + [(i+1, i) for i in range(6)],
        gate_set=["x", "y", "z", "xy", "fsim", "sqrt_iswap"],
        gate_fidelities={
            "x": 0.9998,
            "y": 0.9998,
            "fsim": 0.996,
            "sqrt_iswap": 0.997
        },
        readout_fidelities=[0.98] * 7,
        calibration_status=CalibrationStatus.GOOD,
        success_rate=0.97,
        cost_per_shot=0.00008
    )
    backends.append(google_backend)
    
    # High-performance simulator
    simulator_backend = QuantumBackendInfo(
        id="qem_simulator_20q",
        name="QEM-Bench Simulator 20-qubit",
        provider="QEM-Bench",
        backend_type=BackendType.SIMULATOR,
        num_qubits=20,
        coupling_map=[],  # Fully connected
        gate_set=["x", "y", "z", "rx", "ry", "rz", "cx", "cy", "cz", "ccx"],
        gate_fidelities={gate: 1.0 for gate in ["x", "y", "z", "rx", "ry", "rz", "cx", "cy", "cz", "ccx"]},
        readout_fidelities=[1.0] * 20,
        calibration_status=CalibrationStatus.FRESH,
        success_rate=0.999,
        cost_per_shot=0.00001  # Very low cost for simulator
    )
    backends.append(simulator_backend)
    
    print(f"✅ Created {len(backends)} quantum backends")
    return backends


def create_sample_circuits() -> List[Dict[str, Any]]:
    """Create sample quantum circuits for ZNE experiments."""
    print("📊 Creating sample quantum circuits...")
    
    circuits = []
    
    # Circuit 1: Simple 2-qubit circuit
    circuit1 = {
        "id": "bell_state_circuit",
        "description": "Bell state preparation circuit",
        "num_qubits": 2,
        "depth": 3,
        "gates_used": ["x", "cx"],
        "gate_count": 3,
        "estimated_execution_time": 45.0,
        "shots": 1024
    }
    circuits.append(circuit1)
    
    # Circuit 2: Quantum volume circuit
    circuit2 = {
        "id": "qv_circuit_4q",
        "description": "Quantum Volume circuit (4 qubits)",
        "num_qubits": 4,
        "depth": 8,
        "gates_used": ["x", "y", "rz", "cx"],
        "gate_count": 20,
        "estimated_execution_time": 90.0,
        "shots": 2048
    }
    circuits.append(circuit2)
    
    # Circuit 3: Variational circuit
    circuit3 = {
        "id": "vqe_ansatz_3q",
        "description": "VQE ansatz circuit (3 qubits)",
        "num_qubits": 3,
        "depth": 6,
        "gates_used": ["ry", "rz", "cx"],
        "gate_count": 15,
        "estimated_execution_time": 60.0,
        "shots": 1024
    }
    circuits.append(circuit3)
    
    print(f"✅ Created {len(circuits)} sample circuits")
    return circuits


async def demonstrate_scaling_aware_zne(infrastructure, backends, circuits):
    """Demonstrate scaling-aware ZNE execution."""
    print("\n🧪 Demonstrating Scaling-Aware Zero-Noise Extrapolation...")
    
    # Configure scaling-aware ZNE
    zne_config = ScalingAwareZNEConfig(
        noise_factors=[1.0, 1.5, 2.0, 2.5, 3.0],
        extrapolator="richardson",
        enable_auto_scaling=True,
        enable_distributed_execution=True,
        enable_load_balancing=True,
        enable_resource_optimization=True,
        target_completion_time=120.0,  # 2 minutes target
        max_cost_per_experiment=5.0,   # $5 per experiment
        min_fidelity_threshold=0.85,
        preferred_backends=["qem_simulator_20q", "ibm_quantum_5q"],
        shot_distribution_strategy="quality_weighted",
        circuit_batching_enabled=True,
        adaptive_shot_allocation=True
    )
    
    # Initialize scaling-aware ZNE
    scaling_zne = ScalingAwareZeroNoiseExtrapolation(
        config=zne_config,
        secure_config=infrastructure["secure_config"]
    )
    
    # Start scaling services
    startup_result = await scaling_zne.start_scaling_services()
    print(f"📡 Scaling services started: {startup_result}")
    
    # Add backends to ZNE system
    for backend in backends:
        scaling_zne.add_backend(backend)
    
    # Run ZNE experiments for each circuit
    experiment_results = []
    
    for i, circuit in enumerate(circuits):
        print(f"\n🔬 Running ZNE experiment {i+1}/{len(circuits)}: {circuit['description']}")
        
        try:
            # Execute scaling-aware ZNE
            result = await scaling_zne.mitigate_scaled(
                circuit=circuit,  # In practice, this would be an actual quantum circuit
                backends=backends,
                observable=None,  # Would specify actual observable
                shots=circuit["shots"]
            )
            
            print(f"✅ Experiment completed successfully!")
            print(f"   - Raw value: {result.raw_value:.4f}")
            print(f"   - Mitigated value: {result.mitigated_value:.4f}")
            print(f"   - Backends used: {result.backends_used}")
            print(f"   - Execution time: {result.total_execution_time:.2f}s")
            print(f"   - Parallelization factor: {result.parallelization_factor:.2f}x")
            print(f"   - Total cost: ${result.total_cost:.4f}")
            print(f"   - Resource efficiency: {result.resource_efficiency:.2%}")
            
            experiment_results.append(result)
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            experiment_results.append(None)
    
    # Get scaling system status
    scaling_status = scaling_zne.get_scaling_status()
    print(f"\n📊 Final scaling system status:")
    print(f"   - Scaling enabled: {scaling_status['scaling_enabled']}")
    print(f"   - Active backends: {scaling_status['active_backends']}")
    print(f"   - Experiments executed: {scaling_status['scaling_statistics']['experiments_executed']}")
    print(f"   - Average parallelization: {scaling_status['scaling_statistics']['average_parallelization']:.2f}x")
    print(f"   - Cost savings: ${scaling_status['scaling_statistics']['cost_savings']:.2f}")
    
    # Stop scaling services
    await scaling_zne.stop_scaling_services()
    
    return experiment_results


async def demonstrate_cloud_integration():
    """Demonstrate cloud provider integration and spot instance management."""
    print("\n☁️  Demonstrating Cloud Integration...")
    
    # Create cloud providers
    aws_provider = create_cloud_provider(CloudProvider.AWS, {
        "region": "us-east-1",
        "access_key": "demo_key",
        "secret_key": "demo_secret"
    })
    
    gcp_provider = create_cloud_provider(CloudProvider.GOOGLE_CLOUD, {
        "project_id": "qem-bench-demo",
        "zone": "us-central1-a"
    })
    
    # Set up spot instance manager
    spot_manager = SpotInstanceManager({
        CloudProvider.AWS: aws_provider,
        CloudProvider.GOOGLE_CLOUD: gcp_provider
    })
    
    await spot_manager.start_monitoring()
    
    # Launch some spot instances for cost savings
    print("🏃 Launching cost-optimized spot instances...")
    
    aws_spot = await spot_manager.launch_spot_instance(
        CloudProvider.AWS,
        InstanceType.MEDIUM,
        max_price=0.05,  # $0.05/hour maximum
        availability_zone="us-east-1a"
    )
    
    if aws_spot:
        print(f"✅ Launched AWS spot instance: {aws_spot.id} at ${aws_spot.hourly_cost:.4f}/hour")
    
    gcp_spot = await spot_manager.launch_spot_instance(
        CloudProvider.GOOGLE_CLOUD,
        InstanceType.LARGE,
        max_price=0.08,  # $0.08/hour maximum
        availability_zone="us-central1-a"
    )
    
    if gcp_spot:
        print(f"✅ Launched GCP preemptible instance: {gcp_spot.id} at ${gcp_spot.hourly_cost:.4f}/hour")
    
    # Analyze spot savings potential
    savings_analysis = spot_manager.analyze_spot_savings(
        CloudProvider.AWS,
        InstanceType.MEDIUM,
        hours=24
    )
    
    print(f"💰 Spot savings analysis (AWS, 24h):")
    print(f"   - Average savings: {savings_analysis['savings_analysis']['average_savings']:.1%}")
    print(f"   - Monthly savings estimate: ${savings_analysis['savings_analysis']['monthly_savings_estimate']:.2f}")
    print(f"   - Recommendation: {savings_analysis['recommendation']}")
    
    # Get spot instance statistics
    spot_stats = spot_manager.get_spot_instance_stats()
    print(f"📈 Spot instance statistics:")
    print(f"   - Active instances: {spot_stats['active_spot_instances']}")
    print(f"   - Total runtime: {spot_stats['total_runtime_hours']:.1f} hours")
    print(f"   - Estimated savings: ${spot_stats['estimated_cost_savings']:.2f}")
    
    await spot_manager.stop_monitoring()


async def demonstrate_performance_analytics(infrastructure):
    """Demonstrate performance monitoring and analytics."""
    print("\n📊 Demonstrating Performance Analytics...")
    
    # Get workload analytics
    workload_analyzer = infrastructure["workload_analyzer"]
    workload_trends = workload_analyzer.analyze_workload_trends(duration_seconds=3600)
    print(f"📈 Workload trends analysis:")
    print(f"   - Data points: {workload_trends.get('data_points', 0)}")
    print(f"   - Dominant pattern: {workload_trends.get('dominant_pattern', 'Unknown')}")
    
    # Get cost optimization analytics
    cost_optimizer = infrastructure["cost_optimizer"]
    cost_analytics = cost_optimizer.get_cost_analytics(days=1)
    print(f"💰 Cost analytics:")
    print(f"   - Total cost: ${cost_analytics.get('total_cost', 0):.2f}")
    print(f"   - Cost savings: ${cost_analytics.get('cost_savings', 0):.2f}")
    
    # Get resource scheduling statistics
    resource_scheduler = infrastructure["resource_scheduler"]
    queue_status = resource_scheduler.get_queue_status()
    print(f"⚡ Resource scheduling status:")
    print(f"   - Pending tasks: {queue_status['pending_tasks']}")
    print(f"   - Running tasks: {queue_status['running_tasks']}")
    print(f"   - Completed tasks: {queue_status['completed_tasks']}")
    print(f"   - Average queue time: {queue_status['average_queue_time']:.2f}s")
    
    # Get backend orchestration status
    backend_orchestrator = infrastructure["backend_orchestrator"]
    orchestration_status = backend_orchestrator.get_orchestration_status()
    print(f"🎭 Backend orchestration status:")
    print(f"   - Available backends: {orchestration_status['available_backends']}")
    print(f"   - Queued jobs: {orchestration_status['queued_jobs']}")
    print(f"   - Active jobs: {orchestration_status['active_jobs']}")


async def cleanup_infrastructure(infrastructure):
    """Clean up all infrastructure components."""
    print("\n🧹 Cleaning up infrastructure...")
    
    cleanup_tasks = []
    
    for name, component in infrastructure.items():
        if hasattr(component, 'stop'):
            cleanup_tasks.append(component.stop())
    
    await asyncio.gather(*cleanup_tasks, return_exceptions=True)
    print("✅ Infrastructure cleanup completed")


async def main():
    """Main demonstration function."""
    print("🌟 QEM-Bench Auto-Scaling and Distributed Computing Demo")
    print("=" * 60)
    
    try:
        # Set up infrastructure
        infrastructure = await setup_scaling_infrastructure()
        
        # Create sample backends and circuits
        backends = create_sample_quantum_backends()
        circuits = create_sample_circuits()
        
        # Demonstrate scaling-aware ZNE
        experiment_results = await demonstrate_scaling_aware_zne(
            infrastructure, backends, circuits
        )
        
        # Demonstrate cloud integration
        await demonstrate_cloud_integration()
        
        # Show performance analytics
        await demonstrate_performance_analytics(infrastructure)
        
        # Summary
        successful_experiments = len([r for r in experiment_results if r is not None])
        print(f"\n🎉 Demo Summary:")
        print(f"   - Successful experiments: {successful_experiments}/{len(circuits)}")
        print(f"   - Backends utilized: {len(backends)}")
        print(f"   - Scaling features demonstrated: Auto-scaling, Distributed execution, Load balancing")
        print(f"   - Cloud integration: AWS, Google Cloud spot instances")
        print(f"   - Cost optimization: Budget tracking, Spot instances, Resource optimization")
        
        # Clean up
        await cleanup_infrastructure(infrastructure)
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())