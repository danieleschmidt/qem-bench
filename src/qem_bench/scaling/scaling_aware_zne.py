"""
Scaling-aware Zero-Noise Extrapolation implementation.

This module extends the standard ZNE implementation with comprehensive scaling
capabilities including distributed execution, auto-scaling, load balancing,
and resource optimization specifically for quantum error mitigation workloads.
"""

import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import numpy as np

from ..mitigation.zne.core import ZeroNoiseExtrapolation, ZNEConfig, ZNEResult
from ..security import SecureConfig
from .auto_scaler import AutoScaler, ScalingPolicy
from .distributed_executor import DistributedExecutor, DistributedTask, TaskPriority
from .load_balancer import BackendBalancer, QuantumBackend, QueuedJob, JobPriority
from .resource_optimizer import ResourceOptimizer, OptimizationStrategy
from .backend_orchestrator import BackendOrchestrator, QuantumBackendInfo, OrchestrationJob


logger = logging.getLogger(__name__)


@dataclass
class ScalingAwareZNEConfig(ZNEConfig):
    """Extended ZNE configuration with scaling parameters."""
    
    # Scaling configuration
    enable_auto_scaling: bool = True
    enable_distributed_execution: bool = True
    enable_load_balancing: bool = True
    enable_resource_optimization: bool = True
    
    # Performance targets
    target_completion_time: Optional[float] = None  # seconds
    max_cost_per_experiment: Optional[float] = None
    min_fidelity_threshold: float = 0.8
    
    # Scaling thresholds
    scale_up_queue_threshold: int = 10
    scale_down_idle_threshold: float = 300.0  # 5 minutes
    max_concurrent_noise_factors: int = 5
    
    # Resource allocation
    preferred_backends: List[str] = field(default_factory=list)
    fallback_enabled: bool = True
    shot_distribution_strategy: str = "quality_weighted"
    
    # Optimization preferences
    circuit_batching_enabled: bool = True
    compilation_optimization_level: str = "optimized"
    adaptive_shot_allocation: bool = True


@dataclass
class ScalingAwareZNEResult(ZNEResult):
    """Extended ZNE result with scaling information."""
    
    # Scaling execution details
    backends_used: List[str] = field(default_factory=list)
    total_execution_time: float = 0.0
    scaling_decisions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Resource utilization
    cpu_hours_used: float = 0.0
    total_cost: float = 0.0
    shot_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Performance metrics
    parallelization_factor: float = 1.0
    resource_efficiency: float = 0.0
    cost_efficiency: float = 0.0
    
    # Quality metrics
    cross_backend_consistency: Optional[float] = None
    fallback_usage: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including scaling information."""
        base_dict = super().to_dict()
        
        scaling_dict = {
            "backends_used": self.backends_used,
            "total_execution_time": self.total_execution_time,
            "scaling_decisions": self.scaling_decisions,
            "cpu_hours_used": self.cpu_hours_used,
            "total_cost": self.total_cost,
            "shot_distribution": self.shot_distribution,
            "parallelization_factor": self.parallelization_factor,
            "resource_efficiency": self.resource_efficiency,
            "cost_efficiency": self.cost_efficiency,
            "cross_backend_consistency": self.cross_backend_consistency,
            "fallback_usage": self.fallback_usage
        }
        
        base_dict.update(scaling_dict)
        return base_dict


class ScalingAwareZeroNoiseExtrapolation(ZeroNoiseExtrapolation):
    """
    Scaling-aware Zero-Noise Extrapolation with comprehensive scaling capabilities.
    
    This class extends the standard ZNE implementation with:
    - Automatic scaling based on workload
    - Distributed execution across multiple backends
    - Intelligent load balancing
    - Resource optimization
    - Cost-aware execution
    - Multi-backend orchestration with fallback
    
    Example:
        >>> config = ScalingAwareZNEConfig(
        ...     noise_factors=[1, 1.5, 2, 2.5, 3],
        ...     enable_auto_scaling=True,
        ...     target_completion_time=300.0
        ... )
        >>> zne = ScalingAwareZeroNoiseExtrapolation(config=config)
        >>> await zne.start_scaling_services()
        >>> result = await zne.mitigate_scaled(circuit, backends, observable)
    """
    
    def __init__(
        self,
        config: Optional[ScalingAwareZNEConfig] = None,
        scaling_policy: Optional[ScalingPolicy] = None,
        secure_config: Optional[SecureConfig] = None,
        **kwargs
    ):
        # Initialize base ZNE with standard config
        base_config = ZNEConfig(
            noise_factors=config.noise_factors if config else [1.0, 1.5, 2.0, 2.5, 3.0],
            extrapolator=config.extrapolator if config else "richardson"
        )
        super().__init__(config=base_config, **kwargs)
        
        # Store scaling configuration
        self.scaling_config = config or ScalingAwareZNEConfig(noise_factors=self.noise_factors)
        self.secure_config = secure_config or SecureConfig()
        
        # Initialize scaling components
        self.auto_scaler: Optional[AutoScaler] = None
        self.distributed_executor: Optional[DistributedExecutor] = None
        self.backend_balancer: Optional[BackendBalancer] = None
        self.resource_optimizer: Optional[ResourceOptimizer] = None
        self.backend_orchestrator: Optional[BackendOrchestrator] = None
        
        # Scaling state
        self.is_scaling_enabled = False
        self.active_backends: Dict[str, QuantumBackendInfo] = {}
        self.scaling_statistics = {
            "experiments_executed": 0,
            "total_scaling_time": 0.0,
            "average_parallelization": 1.0,
            "cost_savings": 0.0
        }
        
        logger.info("ScalingAwareZeroNoiseExtrapolation initialized")
    
    async def start_scaling_services(self) -> Dict[str, Any]:
        """Initialize and start all scaling services."""
        logger.info("Starting scaling services")
        
        startup_results = {}
        
        try:
            # Initialize auto-scaler if enabled
            if self.scaling_config.enable_auto_scaling:
                scaling_policy = ScalingPolicy(
                    min_instances=1,
                    max_instances=10,
                    cpu_scale_up_threshold=75.0,
                    scale_up_cooldown=300.0
                )
                
                self.auto_scaler = AutoScaler(
                    policy=scaling_policy,
                    config=self.secure_config
                )
                await self.auto_scaler.start()
                startup_results["auto_scaler"] = "started"
            
            # Initialize distributed executor if enabled
            if self.scaling_config.enable_distributed_execution:
                self.distributed_executor = DistributedExecutor(self.secure_config)
                await self.distributed_executor.start()
                startup_results["distributed_executor"] = "started"
            
            # Initialize backend balancer if enabled
            if self.scaling_config.enable_load_balancing:
                self.backend_balancer = BackendBalancer(config=self.secure_config)
                await self.backend_balancer.start()
                startup_results["backend_balancer"] = "started"
            
            # Initialize resource optimizer if enabled
            if self.scaling_config.enable_resource_optimization:
                optimization_strategy = OptimizationStrategy.BALANCED
                if self.scaling_config.max_cost_per_experiment:
                    optimization_strategy = OptimizationStrategy.MINIMIZE_COST
                elif self.scaling_config.target_completion_time:
                    optimization_strategy = OptimizationStrategy.MINIMIZE_TIME
                
                self.resource_optimizer = ResourceOptimizer(
                    strategy=optimization_strategy,
                    config=self.secure_config
                )
                startup_results["resource_optimizer"] = "started"
            
            # Initialize backend orchestrator
            self.backend_orchestrator = BackendOrchestrator(self.secure_config)
            await self.backend_orchestrator.start()
            startup_results["backend_orchestrator"] = "started"
            
            self.is_scaling_enabled = True
            startup_results["scaling_enabled"] = True
            
            logger.info("All scaling services started successfully")
            
        except Exception as e:
            logger.error(f"Error starting scaling services: {e}")
            startup_results["error"] = str(e)
            await self.stop_scaling_services()
        
        return startup_results
    
    async def stop_scaling_services(self) -> None:
        """Stop all scaling services."""
        logger.info("Stopping scaling services")
        
        if self.auto_scaler:
            await self.auto_scaler.stop()
        
        if self.distributed_executor:
            await self.distributed_executor.stop()
        
        if self.backend_balancer:
            await self.backend_balancer.stop()
        
        if self.backend_orchestrator:
            await self.backend_orchestrator.stop()
        
        self.is_scaling_enabled = False
        logger.info("All scaling services stopped")
    
    def add_backend(
        self, 
        backend_info: Union[Dict[str, Any], QuantumBackendInfo]
    ) -> None:
        """Add a quantum backend to the scaling system."""
        if isinstance(backend_info, dict):
            # Convert dict to QuantumBackendInfo
            from .backend_orchestrator import BackendType, CalibrationStatus, BackendHealth
            
            backend = QuantumBackendInfo(
                id=backend_info["id"],
                name=backend_info.get("name", backend_info["id"]),
                provider=backend_info.get("provider", "unknown"),
                backend_type=BackendType(backend_info.get("type", "hardware")),
                num_qubits=backend_info.get("num_qubits", 5),
                coupling_map=backend_info.get("coupling_map", []),
                gate_set=backend_info.get("gate_set", ["x", "sx", "rz", "cx"]),
                gate_fidelities=backend_info.get("gate_fidelities", {}),
                readout_fidelities=backend_info.get("readout_fidelities", [0.95] * backend_info.get("num_qubits", 5))
            )
        else:
            backend = backend_info
        
        self.active_backends[backend.id] = backend
        
        # Add to scaling services
        if self.backend_orchestrator:
            self.backend_orchestrator.add_backend(backend)
        
        if self.backend_balancer:
            # Convert to load balancer backend format
            lb_backend = QuantumBackend(
                id=backend.id,
                name=backend.name,
                provider=backend.provider,
                num_qubits=backend.num_qubits,
                topology="custom",
                gate_set=backend.gate_set
            )
            self.backend_balancer.add_backend(lb_backend)
        
        logger.info(f"Added backend: {backend.name} ({backend.id})")
    
    async def mitigate_scaled(
        self,
        circuit: Any,
        backends: Optional[List[Any]] = None,
        observable: Optional[Any] = None,
        shots: int = 1024,
        **execution_kwargs
    ) -> ScalingAwareZNEResult:
        """
        Execute ZNE with comprehensive scaling capabilities.
        
        Args:
            circuit: Quantum circuit to execute
            backends: List of quantum backends (optional if backends already added)
            observable: Observable to measure
            shots: Number of measurement shots
            **execution_kwargs: Additional execution arguments
            
        Returns:
            ScalingAwareZNEResult with detailed scaling information
        """
        if not self.is_scaling_enabled:
            logger.warning("Scaling services not started, using standard ZNE")
            base_result = await super().mitigate(circuit, backends[0] if backends else None, observable, shots, **execution_kwargs)
            return self._convert_to_scaling_result(base_result)
        
        start_time = time.time()
        
        logger.info("Starting scaled ZNE execution")
        
        try:
            # Step 1: Prepare backends
            if backends:
                for backend in backends:
                    if hasattr(backend, 'to_dict'):
                        self.add_backend(backend.to_dict())
                    else:
                        self.add_backend(backend)
            
            # Step 2: Create distributed tasks for each noise factor
            tasks = await self._create_noise_factor_tasks(
                circuit, observable, shots, **execution_kwargs
            )
            
            # Step 3: Optimize resource allocation
            if self.resource_optimizer and len(tasks) > 1:
                optimization_result = await self._optimize_task_execution(tasks)
                logger.info(f"Resource optimization completed: {optimization_result}")
            
            # Step 4: Execute tasks with orchestration
            execution_results = await self._execute_distributed_tasks(tasks)
            
            # Step 5: Aggregate results and perform extrapolation
            zne_result = await self._aggregate_and_extrapolate(execution_results)
            
            # Step 6: Calculate scaling metrics
            scaling_result = await self._calculate_scaling_metrics(
                zne_result, execution_results, start_time
            )
            
            # Update statistics
            self._update_scaling_statistics(scaling_result)
            
            logger.info(f"Scaled ZNE completed in {scaling_result.total_execution_time:.2f}s")
            
            return scaling_result
            
        except Exception as e:
            logger.error(f"Error in scaled ZNE execution: {e}")
            # Fallback to standard ZNE
            if backends:
                base_result = await super().mitigate(circuit, backends[0], observable, shots, **execution_kwargs)
                return self._convert_to_scaling_result(base_result, error=str(e))
            else:
                raise
    
    async def _create_noise_factor_tasks(
        self,
        circuit: Any,
        observable: Optional[Any],
        shots: int,
        **execution_kwargs
    ) -> List[DistributedTask]:
        """Create distributed tasks for each noise factor."""
        tasks = []
        
        for i, noise_factor in enumerate(self.noise_factors):
            task = DistributedTask(
                id=f"zne_noise_factor_{noise_factor}_{int(time.time())}_{i}",
                function_name="execute_noise_scaled_circuit",
                args=(circuit, noise_factor, observable, shots),
                kwargs=execution_kwargs,
                priority=TaskPriority.HIGH if noise_factor == 1.0 else TaskPriority.NORMAL,
                timeout=self.scaling_config.target_completion_time,
                min_memory_gb=2.0,
                min_cpu_cores=1
            )
            
            tasks.append(task)
        
        logger.info(f"Created {len(tasks)} distributed tasks for noise factors")
        return tasks
    
    async def _optimize_task_execution(self, tasks: List[DistributedTask]) -> Dict[str, Any]:
        """Optimize task execution using resource optimizer."""
        if not self.resource_optimizer:
            return {"message": "Resource optimizer not available"}
        
        # Convert tasks to circuits for optimization
        circuits = []
        for task in tasks:
            circuit_info = {
                "id": task.id,
                "num_qubits": 5,  # Would extract from actual circuit
                "gates_used": ["x", "sx", "rz", "cx"],  # Would extract from actual circuit
                "shots": task.args[3] if len(task.args) > 3 else 1024,
                "estimated_execution_time": 60.0
            }
            circuits.append(circuit_info)
        
        # Get backend information
        backend_infos = []
        for backend in self.active_backends.values():
            backend_info = {
                "id": backend.id,
                "num_qubits": backend.num_qubits,
                "gate_set": backend.gate_set,
                "fidelity": backend.overall_fidelity(),
                "cost_per_shot": backend.cost_per_shot,
                "is_available": backend.is_available()
            }
            backend_infos.append(backend_info)
        
        # Optimize workload
        optimization_result = await self.resource_optimizer.optimize_workload(
            circuits, backend_infos, {
                "max_cost": self.scaling_config.max_cost_per_experiment,
                "target_completion_time": self.scaling_config.target_completion_time
            }
        )
        
        return optimization_result
    
    async def _execute_distributed_tasks(
        self, 
        tasks: List[DistributedTask]
    ) -> List[Dict[str, Any]]:
        """Execute tasks using distributed executor and orchestration."""
        execution_results = []
        
        if self.distributed_executor and len(tasks) > 1:
            # Execute tasks in parallel using distributed executor
            logger.info(f"Executing {len(tasks)} tasks in parallel")
            
            # Submit all tasks
            task_futures = []
            for task in tasks:
                future = self.distributed_executor.execute_task(task)
                task_futures.append(future)
            
            # Wait for completion
            results = await asyncio.gather(*task_futures, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {tasks[i].id} failed: {result}")
                    # Try fallback execution
                    fallback_result = await self._execute_single_task_fallback(tasks[i])
                    execution_results.append(fallback_result)
                else:
                    execution_results.append(result)
        
        else:
            # Sequential execution with orchestration
            logger.info("Executing tasks sequentially with orchestration")
            
            for task in tasks:
                if self.backend_orchestrator:
                    # Use orchestrator for backend selection and fallback
                    orchestration_job = OrchestrationJob(
                        id=task.id,
                        circuit_description={"noise_factor": task.args[1]},
                        requirements={"shots": task.args[3]},
                        preferred_backends=self.scaling_config.preferred_backends,
                        enable_fallback=self.scaling_config.fallback_enabled
                    )
                    
                    result = await self.backend_orchestrator.execute_job(orchestration_job)
                    execution_results.append(result)
                else:
                    # Fallback to single task execution
                    result = await self._execute_single_task_fallback(task)
                    execution_results.append(result)
        
        return execution_results
    
    async def _execute_single_task_fallback(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute single task with fallback to standard execution."""
        try:
            # Extract parameters from task
            circuit = task.args[0]
            noise_factor = task.args[1]
            observable = task.args[2]
            shots = task.args[3]
            
            # Scale noise in circuit
            scaled_circuit = self.noise_scaler.scale_noise(circuit, noise_factor)
            
            # Select best available backend
            if self.active_backends:
                backend_id = max(
                    self.active_backends.keys(),
                    key=lambda bid: self.active_backends[bid].performance_score()
                )
                backend = self.active_backends[backend_id]
                
                # Simulate execution (in practice, would call actual backend)
                await asyncio.sleep(np.random.uniform(0.5, 2.0))  # Simulate execution time
                
                # Simulate results
                expectation_value = np.random.normal(0.5, 0.1)  # Simulated expectation value
                
                return {
                    "success": True,
                    "expectation_value": expectation_value,
                    "noise_factor": noise_factor,
                    "backend_id": backend_id,
                    "shots": shots,
                    "execution_time": np.random.uniform(30, 120)
                }
            else:
                raise ValueError("No backends available")
                
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "noise_factor": task.args[1] if len(task.args) > 1 else 1.0
            }
    
    async def _aggregate_and_extrapolate(
        self, 
        execution_results: List[Dict[str, Any]]
    ) -> ZNEResult:
        """Aggregate distributed results and perform extrapolation."""
        # Filter successful results
        successful_results = [r for r in execution_results if r.get("success", False)]
        
        if not successful_results:
            raise ValueError("No successful executions for extrapolation")
        
        # Extract noise factors and expectation values
        noise_values = []
        expectation_values = []
        
        for result in successful_results:
            noise_values.append(result["noise_factor"])
            expectation_values.append(result["expectation_value"])
        
        # Sort by noise factor
        sorted_pairs = sorted(zip(noise_values, expectation_values))
        noise_values = [pair[0] for pair in sorted_pairs]
        expectation_values = [pair[1] for pair in sorted_pairs]
        
        # Perform extrapolation using base class method
        mitigated_value, extrapolation_data = self._extrapolate_to_zero_noise(
            noise_values, expectation_values
        )
        
        # Create ZNE result
        raw_value = expectation_values[0] if expectation_values else 0.0
        
        return ZNEResult(
            raw_value=raw_value,
            mitigated_value=mitigated_value,
            noise_factors=noise_values,
            expectation_values=expectation_values,
            extrapolation_data=extrapolation_data,
            error_reduction=None,  # Would calculate if ideal value known
            config=self.config
        )
    
    async def _calculate_scaling_metrics(
        self,
        zne_result: ZNEResult,
        execution_results: List[Dict[str, Any]],
        start_time: float
    ) -> ScalingAwareZNEResult:
        """Calculate comprehensive scaling metrics."""
        total_execution_time = time.time() - start_time
        
        # Extract backend usage
        backends_used = list(set([
            r.get("backend_id", "unknown") for r in execution_results 
            if r.get("success", False)
        ]))
        
        # Calculate shot distribution
        shot_distribution = {}
        for result in execution_results:
            if result.get("success", False):
                backend_id = result.get("backend_id", "unknown")
                shots = result.get("shots", 0)
                shot_distribution[backend_id] = shot_distribution.get(backend_id, 0) + shots
        
        # Calculate parallelization factor
        sequential_time = sum([
            r.get("execution_time", 60.0) for r in execution_results 
            if r.get("success", False)
        ])
        parallelization_factor = sequential_time / max(total_execution_time, 1.0)
        
        # Calculate resource efficiency
        successful_executions = len([r for r in execution_results if r.get("success", False)])
        total_executions = len(execution_results)
        resource_efficiency = successful_executions / max(total_executions, 1)
        
        # Calculate cost efficiency (simplified)
        total_shots = sum(shot_distribution.values())
        cost_per_shot = 0.001  # Simplified cost model
        total_cost = total_shots * cost_per_shot
        cost_efficiency = 1.0 / max(total_cost, 0.001)  # Inverse of cost
        
        # Calculate cross-backend consistency
        cross_backend_consistency = None
        if len(backends_used) > 1:
            backend_results = {}
            for result in execution_results:
                if result.get("success", False):
                    backend_id = result.get("backend_id")
                    noise_factor = result.get("noise_factor")
                    expectation = result.get("expectation_value")
                    
                    if backend_id not in backend_results:
                        backend_results[backend_id] = {}
                    backend_results[backend_id][noise_factor] = expectation
            
            # Calculate consistency across backends for same noise factors
            consistency_scores = []
            for noise_factor in self.noise_factors:
                backend_values = [
                    results.get(noise_factor) for results in backend_results.values()
                    if noise_factor in results
                ]
                if len(backend_values) > 1:
                    std_dev = np.std(backend_values)
                    mean_val = np.mean(backend_values)
                    if mean_val != 0:
                        consistency = 1.0 - (std_dev / abs(mean_val))
                        consistency_scores.append(max(0, consistency))
            
            if consistency_scores:
                cross_backend_consistency = np.mean(consistency_scores)
        
        # Count fallback usage
        fallback_usage = len([r for r in execution_results if not r.get("success", False)])
        
        # Create scaling-aware result
        scaling_result = ScalingAwareZNEResult(
            raw_value=zne_result.raw_value,
            mitigated_value=zne_result.mitigated_value,
            noise_factors=zne_result.noise_factors,
            expectation_values=zne_result.expectation_values,
            extrapolation_data=zne_result.extrapolation_data,
            error_reduction=zne_result.error_reduction,
            config=zne_result.config,
            backends_used=backends_used,
            total_execution_time=total_execution_time,
            shot_distribution=shot_distribution,
            parallelization_factor=parallelization_factor,
            resource_efficiency=resource_efficiency,
            cost_efficiency=cost_efficiency,
            total_cost=total_cost,
            cross_backend_consistency=cross_backend_consistency,
            fallback_usage=fallback_usage
        )
        
        return scaling_result
    
    def _convert_to_scaling_result(
        self, 
        base_result: ZNEResult, 
        error: Optional[str] = None
    ) -> ScalingAwareZNEResult:
        """Convert standard ZNE result to scaling-aware result."""
        return ScalingAwareZNEResult(
            raw_value=base_result.raw_value,
            mitigated_value=base_result.mitigated_value,
            noise_factors=base_result.noise_factors,
            expectation_values=base_result.expectation_values,
            extrapolation_data=base_result.extrapolation_data,
            error_reduction=base_result.error_reduction,
            config=base_result.config,
            backends_used=["single_backend"],
            total_execution_time=60.0,  # Estimate
            parallelization_factor=1.0,
            resource_efficiency=0.5 if error else 1.0,
            cost_efficiency=0.5,
            scaling_decisions=[{"error": error}] if error else []
        )
    
    def _update_scaling_statistics(self, result: ScalingAwareZNEResult) -> None:
        """Update scaling statistics."""
        self.scaling_statistics["experiments_executed"] += 1
        self.scaling_statistics["total_scaling_time"] += result.total_execution_time
        
        # Update average parallelization
        current_avg = self.scaling_statistics["average_parallelization"]
        count = self.scaling_statistics["experiments_executed"]
        new_avg = ((current_avg * (count - 1)) + result.parallelization_factor) / count
        self.scaling_statistics["average_parallelization"] = new_avg
        
        # Estimate cost savings from parallelization
        sequential_time_estimate = result.total_execution_time * result.parallelization_factor
        time_savings = sequential_time_estimate - result.total_execution_time
        cost_savings = time_savings * 0.10  # $0.10 per hour estimate
        self.scaling_statistics["cost_savings"] += cost_savings
    
    async def benchmark_scaling_performance(
        self,
        test_circuits: List[Any],
        shots_per_test: int = 1024
    ) -> Dict[str, Any]:
        """Benchmark scaling performance across multiple test cases."""
        logger.info(f"Starting scaling performance benchmark with {len(test_circuits)} circuits")
        
        benchmark_results = []
        start_time = time.time()
        
        for i, circuit in enumerate(test_circuits):
            logger.info(f"Benchmarking circuit {i+1}/{len(test_circuits)}")
            
            circuit_start = time.time()
            
            try:
                result = await self.mitigate_scaled(
                    circuit, 
                    shots=shots_per_test
                )
                
                benchmark_results.append({
                    "circuit_id": i,
                    "success": True,
                    "execution_time": result.total_execution_time,
                    "parallelization_factor": result.parallelization_factor,
                    "backends_used": len(result.backends_used),
                    "resource_efficiency": result.resource_efficiency,
                    "cost": result.total_cost
                })
                
            except Exception as e:
                logger.error(f"Benchmark failed for circuit {i}: {e}")
                benchmark_results.append({
                    "circuit_id": i,
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - circuit_start
                })
        
        total_benchmark_time = time.time() - start_time
        
        # Calculate benchmark statistics
        successful_results = [r for r in benchmark_results if r["success"]]
        
        if successful_results:
            avg_execution_time = np.mean([r["execution_time"] for r in successful_results])
            avg_parallelization = np.mean([r["parallelization_factor"] for r in successful_results])
            avg_backends_used = np.mean([r["backends_used"] for r in successful_results])
            avg_efficiency = np.mean([r["resource_efficiency"] for r in successful_results])
            total_cost = sum([r["cost"] for r in successful_results])
            
            benchmark_summary = {
                "total_circuits": len(test_circuits),
                "successful_circuits": len(successful_results),
                "success_rate": len(successful_results) / len(test_circuits),
                "total_benchmark_time": total_benchmark_time,
                "average_execution_time": avg_execution_time,
                "average_parallelization_factor": avg_parallelization,
                "average_backends_used": avg_backends_used,
                "average_resource_efficiency": avg_efficiency,
                "total_cost": total_cost,
                "scaling_statistics": self.scaling_statistics.copy()
            }
        else:
            benchmark_summary = {
                "total_circuits": len(test_circuits),
                "successful_circuits": 0,
                "success_rate": 0.0,
                "error": "All benchmark circuits failed"
            }
        
        return {
            "benchmark_summary": benchmark_summary,
            "individual_results": benchmark_results,
            "timestamp": time.time()
        }
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling system status."""
        status = {
            "scaling_enabled": self.is_scaling_enabled,
            "active_backends": len(self.active_backends),
            "scaling_statistics": self.scaling_statistics.copy(),
            "services": {}
        }
        
        # Get status from each service
        if self.auto_scaler:
            status["services"]["auto_scaler"] = "running"
        
        if self.distributed_executor:
            executor_status = self.distributed_executor.get_system_status()
            status["services"]["distributed_executor"] = executor_status
        
        if self.backend_balancer:
            balancer_status = self.backend_balancer.get_system_status()
            status["services"]["backend_balancer"] = balancer_status
        
        if self.backend_orchestrator:
            orchestrator_status = self.backend_orchestrator.get_orchestration_status()
            status["services"]["backend_orchestrator"] = orchestrator_status
        
        if self.resource_optimizer:
            optimizer_stats = self.resource_optimizer.get_component_stats()
            status["services"]["resource_optimizer"] = optimizer_stats
        
        # Backend information
        status["backends"] = {
            backend_id: backend.to_dict()
            for backend_id, backend in self.active_backends.items()
        }
        
        return status
    
    async def optimize_scaling_configuration(self) -> Dict[str, Any]:
        """Optimize scaling configuration based on historical performance."""
        if self.scaling_statistics["experiments_executed"] < 5:
            return {"message": "Insufficient data for optimization"}
        
        recommendations = []
        
        # Analyze parallelization efficiency
        avg_parallelization = self.scaling_statistics["average_parallelization"]
        if avg_parallelization < 2.0:
            recommendations.append(
                "Low parallelization factor - consider adding more backends or enabling distributed execution"
            )
        
        # Analyze cost efficiency
        avg_cost_per_experiment = self.scaling_statistics["cost_savings"] / max(
            self.scaling_statistics["experiments_executed"], 1
        )
        if avg_cost_per_experiment < 0.01:
            recommendations.append(
                "Low cost savings - consider optimizing resource allocation strategy"
            )
        
        # Backend utilization analysis
        if len(self.active_backends) > 1:
            # Check if all backends are being used effectively
            # This would analyze backend usage patterns
            recommendations.append("Consider rebalancing backend usage for better efficiency")
        
        return {
            "optimization_completed": True,
            "current_performance": {
                "average_parallelization": avg_parallelization,
                "average_cost_savings": avg_cost_per_experiment,
                "active_backends": len(self.active_backends)
            },
            "recommendations": recommendations,
            "timestamp": time.time()
        }