"""
Quantum Planning Recovery and Fault Tolerance

Advanced error recovery, fault tolerance, and self-healing mechanisms
for quantum-inspired task planning systems.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import pickle

from .core import Task, TaskState, PlanningConfig
from .validation import ValidationError, ErrorSeverity
from ..monitoring import MetricsCollector


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK_ALGORITHM = "fallback_algorithm"  
    CHECKPOINT_RESTORE = "checkpoint_restore"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    QUANTUM_ERROR_CORRECTION = "quantum_error_correction"
    ADAPTIVE_RECONFIGURATION = "adaptive_reconfiguration"


class FaultType(Enum):
    """Types of faults that can occur"""
    CONVERGENCE_FAILURE = "convergence_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_VIOLATION = "dependency_violation"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    SCHEDULING_CONFLICT = "scheduling_conflict"
    SYSTEM_OVERLOAD = "system_overload"
    SECURITY_BREACH = "security_breach"


@dataclass
class RecoveryContext:
    """Context information for recovery operations"""
    fault_type: FaultType
    error_details: Dict[str, Any]
    original_config: PlanningConfig
    failed_tasks: List[str]
    attempt_count: int = 0
    last_checkpoint: Optional[datetime] = None
    recovery_start_time: datetime = field(default_factory=datetime.now)
    

@dataclass
class CheckpointData:
    """Checkpoint data for recovery"""
    timestamp: datetime
    planning_state: Dict[str, Any]
    task_states: Dict[str, TaskState]
    resource_states: Dict[str, Any]
    config_snapshot: PlanningConfig
    quantum_state_data: Optional[bytes] = None


class QuantumPlanningRecovery:
    """
    Advanced recovery and fault tolerance system for quantum planning
    
    Features:
    - Multiple recovery strategies
    - Checkpointing and state restoration
    - Adaptive reconfiguration
    - Quantum error correction analogies
    - Self-healing mechanisms
    - Performance degradation handling
    """
    
    def __init__(self, config: PlanningConfig = None):
        self.config = config or PlanningConfig()
        self.metrics = MetricsCollector() if config and config.enable_monitoring else None
        
        # Recovery state
        self.checkpoints: List[CheckpointData] = []
        self.recovery_history: List[RecoveryContext] = []
        self.fault_counters: Dict[FaultType, int] = {}
        
        # Recovery configuration
        self.max_recovery_attempts = 5
        self.checkpoint_interval = timedelta(minutes=5)
        self.max_checkpoints = 10
        self.backoff_base_delay = 1.0  # seconds
        self.backoff_max_delay = 60.0  # seconds
        
        # Self-healing
        self._healing_thread = None
        self._stop_healing = threading.Event()
        self._last_checkpoint = datetime.now()
        
        # Recovery strategies registry
        self.recovery_strategies = {
            RecoveryStrategy.RETRY_WITH_BACKOFF: self._retry_with_backoff,
            RecoveryStrategy.FALLBACK_ALGORITHM: self._fallback_algorithm,
            RecoveryStrategy.CHECKPOINT_RESTORE: self._checkpoint_restore,
            RecoveryStrategy.GRACEFUL_DEGRADATION: self._graceful_degradation,
            RecoveryStrategy.QUANTUM_ERROR_CORRECTION: self._quantum_error_correction,
            RecoveryStrategy.ADAPTIVE_RECONFIGURATION: self._adaptive_reconfiguration
        }
    
    def create_checkpoint(self, planner_instance, additional_data: Dict[str, Any] = None) -> str:
        """Create a recovery checkpoint"""
        try:
            # Collect current state
            planning_state = {
                'tasks': {tid: self._serialize_task(task) for tid, task in planner_instance.tasks.items()},
                'quantum_state_cache': getattr(planner_instance, '_quantum_state_cache', {}),
                'solution_cache': getattr(planner_instance, '_solution_cache', {}),
                'additional_data': additional_data or {}
            }
            
            # Collect task states
            task_states = {tid: task.state for tid, task in planner_instance.tasks.items()}
            
            # Resource states (if scheduler is available)
            resource_states = {}
            if hasattr(planner_instance, 'resources'):
                resource_states = {rid: self._serialize_resource_state(res) 
                                 for rid, res in planner_instance.resources.items()}
            
            # Create checkpoint
            checkpoint = CheckpointData(
                timestamp=datetime.now(),
                planning_state=planning_state,
                task_states=task_states,
                resource_states=resource_states,
                config_snapshot=self.config,
                quantum_state_data=self._serialize_quantum_state(planner_instance)
            )
            
            # Store checkpoint
            self.checkpoints.append(checkpoint)
            
            # Maintain checkpoint limit
            if len(self.checkpoints) > self.max_checkpoints:
                self.checkpoints.pop(0)
            
            self._last_checkpoint = datetime.now()
            
            checkpoint_id = f"checkpoint_{checkpoint.timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            if self.metrics:
                self.metrics.record_event("checkpoint_created", {
                    "checkpoint_id": checkpoint_id,
                    "state_size": len(planning_state),
                    "num_tasks": len(task_states)
                })
            
            return checkpoint_id
            
        except Exception as e:
            if self.metrics:
                self.metrics.record_event("checkpoint_failed", {"error": str(e)})
            raise RuntimeError(f"Checkpoint creation failed: {e}")
    
    def recover_from_fault(self, fault_type: FaultType, error_details: Dict[str, Any],
                          planner_instance, failed_operation: Callable = None) -> Dict[str, Any]:
        """
        Recover from a planning fault using appropriate strategy
        
        Args:
            fault_type: Type of fault encountered
            error_details: Details about the error
            planner_instance: The planner instance to recover
            failed_operation: The operation that failed (for retry)
            
        Returns:
            Recovery result with status and recovered data
        """
        # Create recovery context
        context = RecoveryContext(
            fault_type=fault_type,
            error_details=error_details,
            original_config=self.config,
            failed_tasks=error_details.get('failed_tasks', [])
        )
        
        # Track fault occurrence
        if fault_type not in self.fault_counters:
            self.fault_counters[fault_type] = 0
        self.fault_counters[fault_type] += 1
        
        if self.metrics:
            self.metrics.record_event("fault_detected", {
                "fault_type": fault_type.value,
                "fault_count": self.fault_counters[fault_type]
            })
        
        # Select recovery strategy
        recovery_strategy = self._select_recovery_strategy(context)
        
        # Attempt recovery
        recovery_result = None
        max_attempts = self.max_recovery_attempts
        
        for attempt in range(max_attempts):
            context.attempt_count = attempt + 1
            
            try:
                if self.metrics:
                    self.metrics.record_event("recovery_attempt", {
                        "fault_type": fault_type.value,
                        "strategy": recovery_strategy.value,
                        "attempt": attempt + 1
                    })
                
                # Execute recovery strategy
                recovery_function = self.recovery_strategies[recovery_strategy]
                recovery_result = recovery_function(context, planner_instance, failed_operation)
                
                if recovery_result and recovery_result.get('success', False):
                    break
                    
            except Exception as e:
                if attempt == max_attempts - 1:
                    # Last attempt failed
                    recovery_result = {
                        'success': False,
                        'error': f"Recovery failed after {max_attempts} attempts: {str(e)}",
                        'strategy_used': recovery_strategy.value,
                        'attempts': max_attempts
                    }
                else:
                    # Try next attempt with potentially different strategy
                    recovery_strategy = self._select_fallback_strategy(context, recovery_strategy)
        
        # Record recovery outcome
        self.recovery_history.append(context)
        
        if recovery_result and recovery_result.get('success', False):
            if self.metrics:
                self.metrics.record_event("recovery_successful", {
                    "fault_type": fault_type.value,
                    "strategy": recovery_strategy.value,
                    "attempts": context.attempt_count
                })
        else:
            if self.metrics:
                self.metrics.record_event("recovery_failed", {
                    "fault_type": fault_type.value,
                    "final_strategy": recovery_strategy.value,
                    "total_attempts": context.attempt_count
                })
        
        return recovery_result or {'success': False, 'error': 'Recovery exhausted all strategies'}
    
    def _select_recovery_strategy(self, context: RecoveryContext) -> RecoveryStrategy:
        """Select appropriate recovery strategy based on fault type"""
        fault_strategies = {
            FaultType.CONVERGENCE_FAILURE: RecoveryStrategy.ADAPTIVE_RECONFIGURATION,
            FaultType.RESOURCE_EXHAUSTION: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FaultType.DEPENDENCY_VIOLATION: RecoveryStrategy.CHECKPOINT_RESTORE,
            FaultType.QUANTUM_DECOHERENCE: RecoveryStrategy.QUANTUM_ERROR_CORRECTION,
            FaultType.SCHEDULING_CONFLICT: RecoveryStrategy.RETRY_WITH_BACKOFF,
            FaultType.SYSTEM_OVERLOAD: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FaultType.SECURITY_BREACH: RecoveryStrategy.CHECKPOINT_RESTORE
        }
        
        base_strategy = fault_strategies.get(context.fault_type, RecoveryStrategy.RETRY_WITH_BACKOFF)
        
        # Adjust based on fault frequency
        fault_count = self.fault_counters.get(context.fault_type, 0)
        if fault_count > 3:
            # Frequent faults - use more aggressive recovery
            return RecoveryStrategy.ADAPTIVE_RECONFIGURATION
        
        return base_strategy
    
    def _select_fallback_strategy(self, context: RecoveryContext, 
                                 failed_strategy: RecoveryStrategy) -> RecoveryStrategy:
        """Select fallback strategy when primary recovery fails"""
        fallback_hierarchy = {
            RecoveryStrategy.RETRY_WITH_BACKOFF: RecoveryStrategy.FALLBACK_ALGORITHM,
            RecoveryStrategy.FALLBACK_ALGORITHM: RecoveryStrategy.GRACEFUL_DEGRADATION,
            RecoveryStrategy.CHECKPOINT_RESTORE: RecoveryStrategy.ADAPTIVE_RECONFIGURATION,
            RecoveryStrategy.GRACEFUL_DEGRADATION: RecoveryStrategy.CHECKPOINT_RESTORE,
            RecoveryStrategy.QUANTUM_ERROR_CORRECTION: RecoveryStrategy.FALLBACK_ALGORITHM,
            RecoveryStrategy.ADAPTIVE_RECONFIGURATION: RecoveryStrategy.GRACEFUL_DEGRADATION
        }
        
        return fallback_hierarchy.get(failed_strategy, RecoveryStrategy.FALLBACK_ALGORITHM)
    
    def _retry_with_backoff(self, context: RecoveryContext, planner_instance,
                          failed_operation: Callable = None) -> Dict[str, Any]:
        """Retry operation with exponential backoff"""
        if not failed_operation:
            return {'success': False, 'error': 'No operation to retry'}
        
        # Calculate backoff delay
        delay = min(
            self.backoff_base_delay * (2 ** (context.attempt_count - 1)),
            self.backoff_max_delay
        )
        
        # Add jitter to avoid thundering herd
        jitter = np.random.uniform(0, 0.1) * delay
        total_delay = delay + jitter
        
        time.sleep(total_delay)
        
        try:
            # Retry the operation
            result = failed_operation()
            return {
                'success': True,
                'result': result,
                'strategy': 'retry_with_backoff',
                'delay_used': total_delay,
                'attempt': context.attempt_count
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'strategy': 'retry_with_backoff',
                'delay_used': total_delay
            }
    
    def _fallback_algorithm(self, context: RecoveryContext, planner_instance,
                          failed_operation: Callable = None) -> Dict[str, Any]:
        """Use simple fallback planning algorithm"""
        try:
            # Simple greedy scheduling based on priority
            if not hasattr(planner_instance, 'tasks'):
                return {'success': False, 'error': 'No tasks available for fallback'}
            
            tasks = list(planner_instance.tasks.values())
            
            # Filter out failed tasks if specified
            if context.failed_tasks:
                tasks = [t for t in tasks if t.id not in context.failed_tasks]
            
            if not tasks:
                return {'success': False, 'error': 'No valid tasks for fallback scheduling'}
            
            # Sort by priority and complexity
            tasks.sort(key=lambda t: (-t.priority, t.complexity))
            
            # Create simple schedule
            schedule = []
            current_time = 0.0
            
            for task in tasks:
                duration = task.duration_estimate or 1.0
                
                schedule.append({
                    'task_id': task.id,
                    'task_name': task.name,
                    'start_time': current_time,
                    'end_time': current_time + duration,
                    'duration': duration,
                    'resources': task.resources,
                    'priority': task.priority
                })
                
                current_time += duration
            
            fallback_result = {
                'schedule': schedule,
                'total_time': current_time,
                'quantum_fidelity': 0.7,  # Reasonable fallback fidelity
                'convergence_achieved': True,
                'method': 'fallback_greedy',
                'tasks_scheduled': len(schedule)
            }
            
            return {
                'success': True,
                'result': fallback_result,
                'strategy': 'fallback_algorithm',
                'tasks_recovered': len(tasks)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Fallback algorithm failed: {str(e)}",
                'strategy': 'fallback_algorithm'
            }
    
    def _checkpoint_restore(self, context: RecoveryContext, planner_instance,
                          failed_operation: Callable = None) -> Dict[str, Any]:
        """Restore from most recent checkpoint"""
        if not self.checkpoints:
            return {'success': False, 'error': 'No checkpoints available for restore'}
        
        try:
            # Get most recent checkpoint
            latest_checkpoint = self.checkpoints[-1]
            
            # Restore planning state
            self._restore_planning_state(planner_instance, latest_checkpoint)
            
            # If we have a failed operation, try it again after restore
            result = None
            if failed_operation:
                try:
                    result = failed_operation()
                except Exception as e:
                    return {
                        'success': False,
                        'error': f"Operation failed even after restore: {str(e)}",
                        'strategy': 'checkpoint_restore',
                        'checkpoint_age': (datetime.now() - latest_checkpoint.timestamp).total_seconds()
                    }
            
            return {
                'success': True,
                'result': result,
                'strategy': 'checkpoint_restore',
                'checkpoint_timestamp': latest_checkpoint.timestamp,
                'checkpoint_age': (datetime.now() - latest_checkpoint.timestamp).total_seconds()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Checkpoint restore failed: {str(e)}",
                'strategy': 'checkpoint_restore'
            }
    
    def _graceful_degradation(self, context: RecoveryContext, planner_instance,
                            failed_operation: Callable = None) -> Dict[str, Any]:
        """Implement graceful degradation strategy"""
        try:
            # Reduce problem complexity by filtering tasks
            if not hasattr(planner_instance, 'tasks'):
                return {'success': False, 'error': 'No tasks available for degradation'}
            
            original_tasks = dict(planner_instance.tasks)
            
            # Degradation strategies based on fault type
            if context.fault_type == FaultType.RESOURCE_EXHAUSTION:
                # Remove most resource-intensive tasks
                tasks_by_resources = sorted(
                    original_tasks.values(), 
                    key=lambda t: sum(t.resources.values()) if t.resources else 0,
                    reverse=True
                )
                # Keep only 70% of tasks (remove most resource-intensive 30%)
                keep_count = max(1, int(len(tasks_by_resources) * 0.7))
                kept_tasks = {t.id: t for t in tasks_by_resources[:keep_count]}
                
            elif context.fault_type == FaultType.SYSTEM_OVERLOAD:
                # Remove lowest priority tasks
                tasks_by_priority = sorted(
                    original_tasks.values(),
                    key=lambda t: t.priority,
                    reverse=True
                )
                # Keep only 80% of tasks (remove lowest priority 20%)
                keep_count = max(1, int(len(tasks_by_priority) * 0.8))
                kept_tasks = {t.id: t for t in tasks_by_priority[:keep_count]}
                
            else:
                # Generic degradation: remove most complex tasks
                tasks_by_complexity = sorted(
                    original_tasks.values(),
                    key=lambda t: t.complexity
                )
                # Keep only 75% of tasks (remove most complex 25%)
                keep_count = max(1, int(len(tasks_by_complexity) * 0.75))
                kept_tasks = {t.id: t for t in tasks_by_complexity[:keep_count]}
            
            # Update planner with degraded task set
            planner_instance.tasks = kept_tasks
            
            # Try the operation with reduced complexity
            if failed_operation:
                try:
                    result = failed_operation()
                    
                    # Enhance result with degradation info
                    if isinstance(result, dict):
                        result['degradation_applied'] = True
                        result['original_task_count'] = len(original_tasks)
                        result['degraded_task_count'] = len(kept_tasks)
                        result['degradation_ratio'] = len(kept_tasks) / len(original_tasks)
                    
                    return {
                        'success': True,
                        'result': result,
                        'strategy': 'graceful_degradation',
                        'tasks_removed': len(original_tasks) - len(kept_tasks),
                        'degradation_type': context.fault_type.value
                    }
                    
                except Exception as e:
                    # Restore original tasks if degradation didn't help
                    planner_instance.tasks = original_tasks
                    return {
                        'success': False,
                        'error': f"Degradation failed: {str(e)}",
                        'strategy': 'graceful_degradation'
                    }
            else:
                return {
                    'success': True,
                    'result': {'tasks_degraded': True, 'remaining_tasks': len(kept_tasks)},
                    'strategy': 'graceful_degradation',
                    'tasks_removed': len(original_tasks) - len(kept_tasks)
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Graceful degradation failed: {str(e)}",
                'strategy': 'graceful_degradation'
            }
    
    def _quantum_error_correction(self, context: RecoveryContext, planner_instance,
                                failed_operation: Callable = None) -> Dict[str, Any]:
        """Apply quantum error correction analogies"""
        try:
            # Quantum error correction inspired recovery
            # Use redundancy and error detection/correction principles
            
            # Create multiple "logical" solutions and vote
            solutions = []
            num_trials = 3  # Redundancy factor
            
            for trial in range(num_trials):
                try:
                    # Slightly perturb configuration for diversity
                    perturbed_config = self._create_perturbed_config(planner_instance.config, trial)
                    
                    # Store original config
                    original_config = planner_instance.config
                    
                    # Apply perturbation
                    planner_instance.config = perturbed_config
                    
                    # Try operation with perturbed config
                    if failed_operation:
                        trial_result = failed_operation()
                        solutions.append({
                            'result': trial_result,
                            'trial': trial,
                            'config_perturbation': f"trial_{trial}",
                            'success': True
                        })
                    
                    # Restore original config
                    planner_instance.config = original_config
                    
                except Exception as e:
                    # This trial failed, continue with others
                    solutions.append({
                        'error': str(e),
                        'trial': trial,
                        'success': False
                    })
            
            # Error correction: select best solution
            successful_solutions = [s for s in solutions if s.get('success', False)]
            
            if not successful_solutions:
                return {
                    'success': False,
                    'error': 'All quantum error correction trials failed',
                    'strategy': 'quantum_error_correction',
                    'trials_attempted': len(solutions)
                }
            
            # Select solution with highest fidelity (if available)
            best_solution = max(
                successful_solutions,
                key=lambda s: s.get('result', {}).get('quantum_fidelity', 0)
            )
            
            return {
                'success': True,
                'result': best_solution['result'],
                'strategy': 'quantum_error_correction',
                'successful_trials': len(successful_solutions),
                'total_trials': len(solutions),
                'best_trial': best_solution['trial']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Quantum error correction failed: {str(e)}",
                'strategy': 'quantum_error_correction'
            }
    
    def _adaptive_reconfiguration(self, context: RecoveryContext, planner_instance,
                                failed_operation: Callable = None) -> Dict[str, Any]:
        """Adaptively reconfigure planning parameters"""
        try:
            # Adaptive reconfiguration based on fault history
            original_config = planner_instance.config
            adapted_config = self._create_adaptive_config(context, original_config)
            
            # Apply adaptive configuration
            planner_instance.config = adapted_config
            
            # Try operation with new configuration
            if failed_operation:
                try:
                    result = failed_operation()
                    
                    # Success - keep adaptive configuration and return
                    return {
                        'success': True,
                        'result': result,
                        'strategy': 'adaptive_reconfiguration',
                        'config_changes': self._get_config_diff(original_config, adapted_config)
                    }
                    
                except Exception as e:
                    # Restore original configuration
                    planner_instance.config = original_config
                    return {
                        'success': False,
                        'error': f"Adaptive reconfiguration failed: {str(e)}",
                        'strategy': 'adaptive_reconfiguration'
                    }
            else:
                return {
                    'success': True,
                    'result': {'configuration_adapted': True},
                    'strategy': 'adaptive_reconfiguration',
                    'config_changes': self._get_config_diff(original_config, adapted_config)
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Adaptive reconfiguration failed: {str(e)}",
                'strategy': 'adaptive_reconfiguration'
            }
    
    def _create_perturbed_config(self, original_config: PlanningConfig, trial: int) -> PlanningConfig:
        """Create perturbed configuration for quantum error correction"""
        # Create a copy of the original config
        import copy
        perturbed = copy.deepcopy(original_config)
        
        # Apply small perturbations based on trial number
        perturbation_factor = 0.1 + (trial * 0.05)  # 10%, 15%, 20%
        
        # Perturb quantum parameters
        perturbed.superposition_width *= (1 + perturbation_factor * np.random.uniform(-0.5, 0.5))
        perturbed.entanglement_strength *= (1 + perturbation_factor * np.random.uniform(-0.5, 0.5))
        perturbed.interference_factor *= (1 + perturbation_factor * np.random.uniform(-0.5, 0.5))
        
        # Clamp to valid ranges
        perturbed.superposition_width = max(0.01, min(1.0, perturbed.superposition_width))
        perturbed.entanglement_strength = max(0.01, min(1.0, perturbed.entanglement_strength))
        perturbed.interference_factor = max(0.01, min(1.0, perturbed.interference_factor))
        
        return perturbed
    
    def _create_adaptive_config(self, context: RecoveryContext, 
                              original_config: PlanningConfig) -> PlanningConfig:
        """Create adaptively configured planning parameters"""
        import copy
        adapted = copy.deepcopy(original_config)
        
        # Adapt based on fault type and history
        if context.fault_type == FaultType.CONVERGENCE_FAILURE:
            # Increase iterations and reduce convergence threshold
            adapted.max_iterations = min(adapted.max_iterations * 2, 5000)
            adapted.convergence_threshold *= 0.5
            # Increase exploration
            adapted.superposition_width = min(adapted.superposition_width * 1.5, 0.5)
            
        elif context.fault_type == FaultType.QUANTUM_DECOHERENCE:
            # Reduce quantum parameters to improve stability
            adapted.superposition_width *= 0.7
            adapted.entanglement_strength *= 0.8
            adapted.interference_factor *= 0.9
            
        elif context.fault_type == FaultType.SYSTEM_OVERLOAD:
            # Reduce computational intensity
            adapted.max_iterations = max(adapted.max_iterations // 2, 100)
            adapted.superposition_width *= 0.6
            
        # Adapt based on fault frequency
        fault_count = self.fault_counters.get(context.fault_type, 0)
        if fault_count > 3:
            # Frequent faults - be more conservative
            adapted.convergence_threshold *= 2.0  # Easier convergence
            adapted.max_iterations = min(adapted.max_iterations, 1000)  # Limit computation
        
        return adapted
    
    def _get_config_diff(self, original: PlanningConfig, adapted: PlanningConfig) -> Dict[str, Any]:
        """Get differences between configurations"""
        changes = {}
        
        if original.max_iterations != adapted.max_iterations:
            changes['max_iterations'] = {
                'original': original.max_iterations,
                'adapted': adapted.max_iterations
            }
        
        if original.convergence_threshold != adapted.convergence_threshold:
            changes['convergence_threshold'] = {
                'original': original.convergence_threshold,
                'adapted': adapted.convergence_threshold
            }
        
        if original.superposition_width != adapted.superposition_width:
            changes['superposition_width'] = {
                'original': original.superposition_width,
                'adapted': adapted.superposition_width
            }
        
        return changes
    
    def _serialize_task(self, task: Task) -> Dict[str, Any]:
        """Serialize task for checkpointing"""
        return {
            'id': task.id,
            'name': task.name,
            'complexity': task.complexity,
            'dependencies': task.dependencies,
            'resources': task.resources,
            'duration_estimate': task.duration_estimate,
            'priority': task.priority,
            'state': task.state.value if hasattr(task.state, 'value') else str(task.state),
            'metadata': task.metadata
        }
    
    def _serialize_resource_state(self, resource_state) -> Dict[str, Any]:
        """Serialize resource state for checkpointing"""
        if hasattr(resource_state, '__dict__'):
            return resource_state.__dict__.copy()
        else:
            return {'state': str(resource_state)}
    
    def _serialize_quantum_state(self, planner_instance) -> Optional[bytes]:
        """Serialize quantum state data for checkpointing"""
        try:
            if hasattr(planner_instance, 'quantum_state'):
                return pickle.dumps(planner_instance.quantum_state)
        except Exception:
            pass
        return None
    
    def _restore_planning_state(self, planner_instance, checkpoint: CheckpointData) -> None:
        """Restore planner state from checkpoint"""
        # Restore tasks
        restored_tasks = {}
        for task_id, task_data in checkpoint.planning_state['tasks'].items():
            restored_task = self._deserialize_task(task_data)
            restored_tasks[task_id] = restored_task
        
        planner_instance.tasks = restored_tasks
        
        # Restore caches
        if hasattr(planner_instance, '_quantum_state_cache'):
            planner_instance._quantum_state_cache = checkpoint.planning_state.get('quantum_state_cache', {})
        
        if hasattr(planner_instance, '_solution_cache'):
            planner_instance._solution_cache = checkpoint.planning_state.get('solution_cache', {})
        
        # Restore quantum state if available
        if checkpoint.quantum_state_data:
            try:
                planner_instance.quantum_state = pickle.loads(checkpoint.quantum_state_data)
            except Exception:
                pass  # Skip if restoration fails
    
    def _deserialize_task(self, task_data: Dict[str, Any]) -> Task:
        """Deserialize task from checkpoint data"""
        task = Task(
            id=task_data['id'],
            name=task_data['name'],
            complexity=task_data['complexity'],
            dependencies=task_data['dependencies'],
            resources=task_data['resources'],
            duration_estimate=task_data['duration_estimate'],
            priority=task_data['priority'],
            metadata=task_data.get('metadata', {})
        )
        
        # Restore state
        state_str = task_data.get('state', 'SUPERPOSITION')
        if hasattr(TaskState, state_str):
            task.state = getattr(TaskState, state_str)
        
        return task
    
    def start_self_healing(self, planner_instance) -> None:
        """Start self-healing background process"""
        if self._healing_thread and self._healing_thread.is_alive():
            return
        
        self._stop_healing.clear()
        self._healing_thread = threading.Thread(
            target=self._self_healing_loop,
            args=(planner_instance,),
            daemon=True
        )
        self._healing_thread.start()
        
        if self.metrics:
            self.metrics.record_event("self_healing_started", {})
    
    def stop_self_healing(self) -> None:
        """Stop self-healing background process"""
        self._stop_healing.set()
        if self._healing_thread:
            self._healing_thread.join(timeout=5.0)
        
        if self.metrics:
            self.metrics.record_event("self_healing_stopped", {})
    
    def _self_healing_loop(self, planner_instance) -> None:
        """Self-healing background loop"""
        while not self._stop_healing.is_set():
            try:
                # Check if checkpoint is needed
                time_since_checkpoint = datetime.now() - self._last_checkpoint
                if time_since_checkpoint > self.checkpoint_interval:
                    self.create_checkpoint(planner_instance)
                
                # Check for signs of degradation
                self._check_system_health(planner_instance)
                
                # Sleep between health checks
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                if self.metrics:
                    self.metrics.record_event("self_healing_error", {"error": str(e)})
                time.sleep(60)  # Longer sleep on error
    
    def _check_system_health(self, planner_instance) -> None:
        """Check system health and trigger preventive recovery if needed"""
        # Check for memory usage
        if hasattr(planner_instance, '_solution_cache'):
            cache_size = len(planner_instance._solution_cache)
            if cache_size > 1000:  # Cache growing too large
                # Clear old cache entries
                planner_instance._solution_cache.clear()
                if self.metrics:
                    self.metrics.record_event("preventive_cache_clear", {"cache_size": cache_size})
        
        # Check for frequent faults
        total_faults = sum(self.fault_counters.values())
        if total_faults > 50:  # Many faults occurred
            # Reset fault counters and create preventive checkpoint
            self.fault_counters.clear()
            self.create_checkpoint(planner_instance)
            if self.metrics:
                self.metrics.record_event("preventive_fault_reset", {"total_faults": total_faults})
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics"""
        total_recoveries = len(self.recovery_history)
        successful_recoveries = sum(1 for r in self.recovery_history 
                                  if any(attempt.get('success', False) for attempt in [r]))
        
        # Strategy effectiveness
        strategy_stats = {}
        for recovery in self.recovery_history:
            # This is a simplified version - in reality you'd track strategy success
            pass
        
        return {
            'total_checkpoints': len(self.checkpoints),
            'total_recoveries_attempted': total_recoveries,
            'successful_recoveries': successful_recoveries,
            'recovery_success_rate': successful_recoveries / total_recoveries if total_recoveries > 0 else 1.0,
            'fault_distribution': dict(self.fault_counters),
            'most_common_fault': max(self.fault_counters, key=self.fault_counters.get) if self.fault_counters else None,
            'self_healing_active': self._healing_thread and self._healing_thread.is_alive(),
            'last_checkpoint_age': (datetime.now() - self._last_checkpoint).total_seconds()
        }