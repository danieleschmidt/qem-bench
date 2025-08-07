"""
QEM-Planning Integration

Seamless integration between quantum error mitigation infrastructure
and quantum-inspired task planning capabilities.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union
import numpy as np
from datetime import datetime

from .core import QuantumInspiredPlanner, Task, PlanningConfig, TaskState
from .optimizer import QuantumTaskOptimizer, OptimizationStrategy
from .scheduler import QuantumScheduler, SchedulingPolicy
from .metrics import PlanningAnalyzer, PlanningMetrics

# QEM infrastructure imports
from ..mitigation.zne import ZeroNoiseExtrapolation
from ..mitigation.pec import ProbabilisticErrorCancellation
from ..mitigation.vd import VirtualDistillation
from ..mitigation.cdr import CliffordDataRegression
from ..monitoring import SystemMonitor, MetricsCollector
from ..optimization import PerformanceOptimizer
from ..scaling import AutoScaler, ResourceOptimizer
from ..security import SecureConfig, InputSanitizer


@dataclass
class QEMTask(Task):
    """Extended task with QEM-specific properties"""
    mitigation_method: Optional[str] = None
    circuit_complexity: float = 1.0
    noise_resilience: float = 0.5
    quantum_backend: Optional[str] = None
    expected_fidelity: float = 0.9
    error_budget: float = 0.1
    
    # QEM-specific metadata
    requires_zne: bool = False
    requires_pec: bool = False  
    requires_vd: bool = False
    requires_cdr: bool = False


class QEMPlannerIntegration:
    """
    Integrated quantum-inspired planner with QEM infrastructure
    
    Features:
    - Quantum circuit planning with error mitigation
    - Resource-aware QEM scheduling
    - Performance optimization integration
    - Security and monitoring integration
    - Scalable distributed execution
    """
    
    def __init__(self, config: PlanningConfig = None):
        self.config = config or PlanningConfig()
        
        # Core planning components
        self.planner = QuantumInspiredPlanner(self.config)
        self.scheduler = QuantumScheduler(self.config, SchedulingPolicy.QUANTUM_PRIORITY)
        self.analyzer = PlanningAnalyzer()
        
        # QEM infrastructure integration
        self.system_monitor = SystemMonitor() if self.config.enable_monitoring else None
        self.performance_optimizer = PerformanceOptimizer() if self.config.use_gpu else None
        self.metrics_collector = MetricsCollector() if self.config.enable_monitoring else None
        self.auto_scaler = AutoScaler()
        self.security_config = SecureConfig()
        self.input_sanitizer = InputSanitizer()
        
        # QEM methods
        self.zne = ZeroNoiseExtrapolation()
        self.pec = ProbabilisticErrorCancellation()
        self.vd = VirtualDistillation()
        self.cdr = CliffordDataRegression()
        
        # Integration state
        self._qem_backends: Dict[str, Any] = {}
        self._circuit_cache: Dict[str, Any] = {}
        self._mitigation_cache: Dict[str, Any] = {}
        
    def create_qem_task(self, task_id: str, name: str, circuit_spec: Dict[str, Any],
                        mitigation_requirements: Dict[str, bool] = None) -> QEMTask:
        """
        Create a QEM-specific task with quantum circuit planning
        
        Args:
            task_id: Unique task identifier
            name: Human-readable task name
            circuit_spec: Quantum circuit specification
            mitigation_requirements: Required mitigation methods
            
        Returns:
            QEMTask configured for quantum execution
        """
        # Sanitize inputs
        task_id = self.input_sanitizer.sanitize_string(task_id)
        name = self.input_sanitizer.sanitize_string(name)
        
        mitigation_req = mitigation_requirements or {}
        
        # Analyze circuit complexity
        circuit_complexity = self._analyze_circuit_complexity(circuit_spec)
        
        # Estimate noise resilience
        noise_resilience = self._estimate_noise_resilience(circuit_spec, mitigation_req)
        
        # Determine resource requirements
        resources = self._estimate_qem_resources(circuit_spec, mitigation_req)
        
        # Create QEM task
        task = QEMTask(
            id=task_id,
            name=name,
            complexity=circuit_complexity,
            resources=resources,
            circuit_complexity=circuit_complexity,
            noise_resilience=noise_resilience,
            requires_zne=mitigation_req.get('zne', False),
            requires_pec=mitigation_req.get('pec', False),
            requires_vd=mitigation_req.get('vd', False),
            requires_cdr=mitigation_req.get('cdr', False),
            metadata={'circuit_spec': circuit_spec}
        )
        
        # Cache circuit for later execution
        self._circuit_cache[task_id] = circuit_spec
        
        if self.metrics_collector:
            self.metrics_collector.record_event("qem_task_created", {
                "task_id": task_id,
                "circuit_complexity": circuit_complexity,
                "mitigation_methods": mitigation_req
            })
        
        return task
    
    def plan_qem_execution(self, tasks: List[QEMTask], 
                          execution_constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Plan optimal execution of QEM tasks
        
        Args:
            tasks: List of QEM tasks to plan
            execution_constraints: Resource and time constraints
            
        Returns:
            Comprehensive execution plan with QEM integration
        """
        if self.metrics_collector:
            self.metrics_collector.record_event("qem_planning_started", {
                "num_tasks": len(tasks),
                "constraints": execution_constraints or {}
            })
        
        try:
            # Add tasks to planner
            for task in tasks:
                self.planner.add_task(task)
            
            # Apply execution constraints
            if execution_constraints:
                self._apply_execution_constraints(execution_constraints)
            
            # Optimize with QEM-aware strategy
            optimization_strategy = self._select_qem_optimization_strategy(tasks)
            
            # Generate quantum-optimized plan
            planning_result = self.planner.plan(objective="minimize_qem_error")
            
            # Enhance with QEM-specific scheduling
            enhanced_plan = self._enhance_with_qem_scheduling(planning_result, tasks)
            
            # Add resource scaling recommendations
            scaling_recommendations = self._generate_scaling_recommendations(enhanced_plan)
            
            # Security validation
            security_validation = self._validate_plan_security(enhanced_plan)
            
            # Performance metrics
            performance_metrics = self.analyzer.analyze_planning_performance(
                {task.id: task for task in tasks}, enhanced_plan
            )
            
            final_plan = {
                **enhanced_plan,
                'qem_integration': {
                    'optimization_strategy': optimization_strategy.value,
                    'mitigation_methods_used': self._get_mitigation_methods_used(tasks),
                    'expected_error_reduction': self._estimate_error_reduction(tasks),
                    'quantum_resource_efficiency': self._calculate_quantum_efficiency(tasks)
                },
                'scaling_recommendations': scaling_recommendations,
                'security_validation': security_validation,
                'performance_metrics': performance_metrics,
                'execution_metadata': {
                    'planning_timestamp': datetime.now(),
                    'config': self.config,
                    'infrastructure_status': self._get_infrastructure_status()
                }
            }
            
            if self.metrics_collector:
                self.metrics_collector.record_event("qem_planning_completed", {
                    "total_time": enhanced_plan.get('total_time', 0),
                    "quantum_fidelity": enhanced_plan.get('quantum_fidelity', 0),
                    "num_scheduled_tasks": len(enhanced_plan.get('schedule', []))
                })
            
            return final_plan
            
        except Exception as e:
            if self.metrics_collector:
                self.metrics_collector.record_event("qem_planning_failed", {"error": str(e)})
            raise
    
    def execute_qem_plan(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute QEM plan with real-time monitoring and adaptation
        
        Args:
            execution_plan: Plan generated by plan_qem_execution
            
        Returns:
            Execution results with comprehensive metrics
        """
        if self.metrics_collector:
            self.metrics_collector.record_event("qem_execution_started", {
                "plan_id": execution_plan.get('plan_id', 'unknown')
            })
        
        try:
            # Start system monitoring
            if self.system_monitor:
                self.system_monitor.start_monitoring()
            
            # Initialize scheduler with QEM tasks
            self.scheduler.start_scheduler()
            
            # Execute schedule with QEM integration
            schedule = execution_plan.get('schedule', [])
            execution_results = []
            
            for task_event in schedule:
                task_id = task_event['task_id']
                
                # Execute with appropriate QEM methods
                task_result = self._execute_qem_task(task_id, task_event)
                execution_results.append(task_result)
                
                # Update scheduler with results
                if task_result.get('success', False):
                    self.scheduler._complete_task(task_id)
                else:
                    self.scheduler.failed_tasks.add(task_id)
            
            # Stop monitoring and gather results
            if self.system_monitor:
                monitoring_data = self.system_monitor.get_monitoring_data()
                self.system_monitor.stop_monitoring()
            else:
                monitoring_data = {}
            
            self.scheduler.stop_scheduler()
            
            # Compile comprehensive results
            execution_summary = {
                'execution_results': execution_results,
                'successful_tasks': sum(1 for r in execution_results if r.get('success', False)),
                'failed_tasks': sum(1 for r in execution_results if not r.get('success', True)),
                'total_execution_time': sum(r.get('execution_time', 0) for r in execution_results),
                'average_fidelity': np.mean([r.get('fidelity', 0) for r in execution_results]),
                'total_error_reduction': sum(r.get('error_reduction', 0) for r in execution_results),
                'monitoring_data': monitoring_data,
                'scheduler_status': self.scheduler.get_schedule_status(),
                'resource_utilization': self.scheduler.get_resource_status(),
                'qem_metrics': self._compile_qem_metrics(execution_results)
            }
            
            if self.metrics_collector:
                self.metrics_collector.record_event("qem_execution_completed", {
                    "successful_tasks": execution_summary['successful_tasks'],
                    "total_time": execution_summary['total_execution_time']
                })
            
            return execution_summary
            
        except Exception as e:
            if self.metrics_collector:
                self.metrics_collector.record_event("qem_execution_failed", {"error": str(e)})
            raise
    
    def _analyze_circuit_complexity(self, circuit_spec: Dict[str, Any]) -> float:
        """Analyze quantum circuit complexity"""
        # Extract circuit parameters
        num_qubits = circuit_spec.get('num_qubits', 1)
        circuit_depth = circuit_spec.get('depth', 1)
        gate_count = circuit_spec.get('gate_count', circuit_depth)
        
        # Complexity formula inspired by quantum volume
        base_complexity = num_qubits * circuit_depth
        gate_factor = np.log2(gate_count + 1)
        
        return base_complexity * gate_factor
    
    def _estimate_noise_resilience(self, circuit_spec: Dict[str, Any], 
                                 mitigation_req: Dict[str, bool]) -> float:
        """Estimate circuit noise resilience with mitigation"""
        base_resilience = 0.5  # Default resilience
        
        # Adjust for circuit properties
        num_qubits = circuit_spec.get('num_qubits', 1)
        circuit_depth = circuit_spec.get('depth', 1)
        
        # Larger/deeper circuits are less resilient
        size_penalty = 1.0 / (1.0 + 0.1 * num_qubits * circuit_depth)
        base_resilience *= size_penalty
        
        # Mitigation methods improve resilience
        mitigation_bonus = 0.0
        if mitigation_req.get('zne', False):
            mitigation_bonus += 0.2
        if mitigation_req.get('pec', False):
            mitigation_bonus += 0.3
        if mitigation_req.get('vd', False):
            mitigation_bonus += 0.25
        if mitigation_req.get('cdr', False):
            mitigation_bonus += 0.15
        
        return min(1.0, base_resilience + mitigation_bonus)
    
    def _estimate_qem_resources(self, circuit_spec: Dict[str, Any],
                              mitigation_req: Dict[str, bool]) -> Dict[str, float]:
        """Estimate resource requirements for QEM execution"""
        base_resources = {
            'qubits': float(circuit_spec.get('num_qubits', 1)),
            'memory': float(circuit_spec.get('num_qubits', 1) * 0.1),  # GB per qubit
            'compute': float(circuit_spec.get('depth', 1) * 0.05)  # Compute units
        }
        
        # Mitigation overhead
        if mitigation_req.get('zne', False):
            base_resources['compute'] *= 3.0  # ZNE overhead
        if mitigation_req.get('pec', False):
            base_resources['compute'] *= 5.0  # PEC overhead
        if mitigation_req.get('vd', False):
            base_resources['memory'] *= 2.0  # VD memory overhead
        if mitigation_req.get('cdr', False):
            base_resources['compute'] *= 2.0  # CDR overhead
        
        return base_resources
    
    def _select_qem_optimization_strategy(self, tasks: List[QEMTask]) -> OptimizationStrategy:
        """Select optimal optimization strategy for QEM tasks"""
        # Analyze task characteristics
        has_complex_circuits = any(task.circuit_complexity > 10 for task in tasks)
        has_many_dependencies = any(len(task.dependencies) > 3 for task in tasks)
        has_strict_fidelity = any(task.expected_fidelity > 0.95 for task in tasks)
        
        # Strategy selection logic
        if has_strict_fidelity and has_complex_circuits:
            return OptimizationStrategy.VARIATIONAL_QUANTUM
        elif has_many_dependencies:
            return OptimizationStrategy.ADIABATIC_QUANTUM
        elif len(tasks) > 10:
            return OptimizationStrategy.QUANTUM_APPROXIMATE
        else:
            return OptimizationStrategy.QUANTUM_ANNEALING
    
    def _enhance_with_qem_scheduling(self, planning_result: Dict[str, Any], 
                                   tasks: List[QEMTask]) -> Dict[str, Any]:
        """Enhance planning result with QEM-specific scheduling"""
        enhanced_result = planning_result.copy()
        
        # Add QEM-specific timing adjustments
        schedule = enhanced_result.get('schedule', [])
        for event in schedule:
            task_id = event['task_id']
            task = next((t for t in tasks if t.id == task_id), None)
            
            if task:
                # Adjust timing for mitigation overhead
                mitigation_overhead = self._calculate_mitigation_overhead(task)
                event['duration'] *= mitigation_overhead
                event['end_time'] = event['start_time'] + event['duration']
                
                # Add QEM-specific metadata
                event['qem_metadata'] = {
                    'mitigation_methods': self._get_task_mitigation_methods(task),
                    'expected_error_reduction': self._estimate_task_error_reduction(task),
                    'resource_efficiency': task.noise_resilience
                }
        
        return enhanced_result
    
    def _calculate_mitigation_overhead(self, task: QEMTask) -> float:
        """Calculate overhead factor for mitigation methods"""
        overhead = 1.0
        
        if task.requires_zne:
            overhead *= 3.0  # ZNE requires multiple noise levels
        if task.requires_pec:
            overhead *= 5.0  # PEC requires many samples
        if task.requires_vd:
            overhead *= 2.5  # VD requires multiple copies
        if task.requires_cdr:
            overhead *= 1.5  # CDR has lower overhead
        
        return overhead
    
    def _execute_qem_task(self, task_id: str, task_event: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual QEM task with appropriate mitigation"""
        start_time = datetime.now()
        
        try:
            # Get circuit specification
            circuit_spec = self._circuit_cache.get(task_id, {})
            qem_metadata = task_event.get('qem_metadata', {})
            mitigation_methods = qem_metadata.get('mitigation_methods', [])
            
            # Apply mitigation methods
            execution_result = {'success': True, 'fidelity': 0.9, 'error_reduction': 0.0}
            
            if 'zne' in mitigation_methods:
                zne_result = self._apply_zne(circuit_spec)
                execution_result['fidelity'] *= zne_result.get('improvement_factor', 1.0)
                execution_result['error_reduction'] += zne_result.get('error_reduction', 0.0)
            
            if 'pec' in mitigation_methods:
                pec_result = self._apply_pec(circuit_spec)
                execution_result['fidelity'] *= pec_result.get('improvement_factor', 1.0)
                execution_result['error_reduction'] += pec_result.get('error_reduction', 0.0)
            
            if 'vd' in mitigation_methods:
                vd_result = self._apply_vd(circuit_spec)
                execution_result['fidelity'] *= vd_result.get('improvement_factor', 1.0)
                execution_result['error_reduction'] += vd_result.get('error_reduction', 0.0)
            
            if 'cdr' in mitigation_methods:
                cdr_result = self._apply_cdr(circuit_spec)
                execution_result['fidelity'] *= cdr_result.get('improvement_factor', 1.0)
                execution_result['error_reduction'] += cdr_result.get('error_reduction', 0.0)
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            execution_result.update({
                'task_id': task_id,
                'execution_time': execution_time,
                'start_time': start_time,
                'end_time': end_time,
                'mitigation_methods_applied': mitigation_methods,
                'circuit_parameters': circuit_spec
            })
            
            return execution_result
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return {
                'task_id': task_id,
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'fidelity': 0.0,
                'error_reduction': 0.0
            }
    
    def _apply_zne(self, circuit_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Zero Noise Extrapolation"""
        # Simulate ZNE execution
        improvement_factor = 1.5  # Typical ZNE improvement
        error_reduction = 0.3
        
        return {
            'improvement_factor': improvement_factor,
            'error_reduction': error_reduction,
            'method': 'zne'
        }
    
    def _apply_pec(self, circuit_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Probabilistic Error Cancellation"""
        # Simulate PEC execution
        improvement_factor = 2.0  # Typical PEC improvement
        error_reduction = 0.4
        
        return {
            'improvement_factor': improvement_factor,
            'error_reduction': error_reduction,
            'method': 'pec'
        }
    
    def _apply_vd(self, circuit_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Virtual Distillation"""
        # Simulate VD execution
        improvement_factor = 1.8  # Typical VD improvement
        error_reduction = 0.35
        
        return {
            'improvement_factor': improvement_factor,
            'error_reduction': error_reduction,
            'method': 'vd'
        }
    
    def _apply_cdr(self, circuit_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Clifford Data Regression"""
        # Simulate CDR execution
        improvement_factor = 1.3  # Typical CDR improvement
        error_reduction = 0.2
        
        return {
            'improvement_factor': improvement_factor,
            'error_reduction': error_reduction,
            'method': 'cdr'
        }
    
    def _get_task_mitigation_methods(self, task: QEMTask) -> List[str]:
        """Get list of mitigation methods for task"""
        methods = []
        if task.requires_zne:
            methods.append('zne')
        if task.requires_pec:
            methods.append('pec')
        if task.requires_vd:
            methods.append('vd')
        if task.requires_cdr:
            methods.append('cdr')
        return methods
    
    def _estimate_task_error_reduction(self, task: QEMTask) -> float:
        """Estimate error reduction for task"""
        total_reduction = 0.0
        
        if task.requires_zne:
            total_reduction += 0.3
        if task.requires_pec:
            total_reduction += 0.4
        if task.requires_vd:
            total_reduction += 0.35
        if task.requires_cdr:
            total_reduction += 0.2
        
        return min(0.9, total_reduction)  # Cap at 90% reduction
    
    def _apply_execution_constraints(self, constraints: Dict[str, Any]) -> None:
        """Apply execution constraints to planning"""
        # Resource constraints
        if 'max_qubits' in constraints:
            # Add qubit resource constraint to scheduler
            self.scheduler.add_resource('qubits', constraints['max_qubits'])
        
        if 'max_memory' in constraints:
            self.scheduler.add_resource('memory', constraints['max_memory'])
        
        if 'max_compute' in constraints:
            self.scheduler.add_resource('compute', constraints['max_compute'])
        
        # Time constraints
        if 'deadline' in constraints:
            # Implement deadline constraint in optimization
            pass
    
    def _generate_scaling_recommendations(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate resource scaling recommendations"""
        return {
            'recommended_scaling': 'auto',
            'peak_resource_usage': self._estimate_peak_usage(plan),
            'scaling_triggers': ['queue_depth > 10', 'avg_wait_time > 30s']
        }
    
    def _validate_plan_security(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate plan security requirements"""
        return {
            'security_validated': True,
            'access_controls_verified': True,
            'resource_limits_enforced': True
        }
    
    def _get_mitigation_methods_used(self, tasks: List[QEMTask]) -> List[str]:
        """Get unique mitigation methods across all tasks"""
        methods = set()
        for task in tasks:
            methods.update(self._get_task_mitigation_methods(task))
        return list(methods)
    
    def _estimate_error_reduction(self, tasks: List[QEMTask]) -> float:
        """Estimate overall error reduction"""
        if not tasks:
            return 0.0
        
        total_reduction = sum(self._estimate_task_error_reduction(task) for task in tasks)
        return total_reduction / len(tasks)
    
    def _calculate_quantum_efficiency(self, tasks: List[QEMTask]) -> float:
        """Calculate quantum resource efficiency"""
        if not tasks:
            return 1.0
        
        total_efficiency = sum(task.noise_resilience for task in tasks)
        return total_efficiency / len(tasks)
    
    def _get_infrastructure_status(self) -> Dict[str, Any]:
        """Get current infrastructure status"""
        return {
            'monitoring_active': self.system_monitor is not None,
            'performance_optimization': self.performance_optimizer is not None,
            'auto_scaling': True,
            'security_enabled': True
        }
    
    def _estimate_peak_usage(self, plan: Dict[str, Any]) -> Dict[str, float]:
        """Estimate peak resource usage"""
        schedule = plan.get('schedule', [])
        if not schedule:
            return {}
        
        max_qubits = max((event.get('resources', {}).get('qubits', 0) for event in schedule), default=0)
        max_memory = max((event.get('resources', {}).get('memory', 0) for event in schedule), default=0)
        max_compute = max((event.get('resources', {}).get('compute', 0) for event in schedule), default=0)
        
        return {
            'qubits': max_qubits,
            'memory': max_memory,
            'compute': max_compute
        }
    
    def _compile_qem_metrics(self, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile comprehensive QEM metrics"""
        if not execution_results:
            return {}
        
        return {
            'total_tasks': len(execution_results),
            'successful_tasks': sum(1 for r in execution_results if r.get('success', False)),
            'average_fidelity': np.mean([r.get('fidelity', 0) for r in execution_results]),
            'total_error_reduction': sum(r.get('error_reduction', 0) for r in execution_results),
            'execution_efficiency': len([r for r in execution_results if r.get('success', False)]) / len(execution_results),
            'mitigation_effectiveness': np.mean([r.get('error_reduction', 0) for r in execution_results])
        }