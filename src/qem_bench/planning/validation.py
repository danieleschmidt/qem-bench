"""
Planning Validation and Error Handling

Comprehensive validation, error handling, and resilience mechanisms
for quantum-inspired task planning systems.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import logging

from .core import Task, TaskState, PlanningConfig
from .optimizer import OptimizationResult
from ..security import InputSanitizer, SecurityPolicy


class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class ErrorSeverity(Enum):
    """Error severity classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationError:
    """Validation error representation"""
    error_id: str
    severity: ErrorSeverity
    category: str
    message: str
    task_id: Optional[str] = None
    suggestion: Optional[str] = None
    recoverable: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationResult:
    """Comprehensive validation result"""
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    performance_score: float = 1.0
    security_score: float = 1.0
    reliability_score: float = 1.0
    validation_time: float = 0.0


class PlanningValidator:
    """
    Comprehensive planning validation and error handling
    
    Features:
    - Multi-level validation (basic to paranoid)
    - Task dependency validation
    - Resource constraint validation
    - Security and safety validation
    - Performance impact assessment
    - Error recovery suggestions
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD,
                 security_policy: SecurityPolicy = None):
        self.validation_level = validation_level
        self.security_policy = security_policy or SecurityPolicy()
        self.input_sanitizer = InputSanitizer()
        
        # Validation state
        self._validation_cache: Dict[str, ValidationResult] = {}
        self._error_history: List[ValidationError] = []
        self._validation_stats: Dict[str, Any] = {}
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def validate_planning_config(self, config: PlanningConfig) -> ValidationResult:
        """Validate planning configuration parameters"""
        start_time = datetime.now()
        result = ValidationResult(valid=True)
        
        try:
            # Basic parameter validation
            if config.max_iterations <= 0:
                result.errors.append(ValidationError(
                    error_id="config_invalid_iterations",
                    severity=ErrorSeverity.HIGH,
                    category="configuration",
                    message="max_iterations must be positive",
                    suggestion="Set max_iterations to at least 100"
                ))
            
            if not (0 < config.convergence_threshold < 1):
                result.errors.append(ValidationError(
                    error_id="config_invalid_threshold",
                    severity=ErrorSeverity.MEDIUM,
                    category="configuration", 
                    message="convergence_threshold must be between 0 and 1",
                    suggestion="Use typical value between 1e-6 and 1e-3"
                ))
            
            # Quantum parameters validation
            if not (0 <= config.superposition_width <= 1):
                result.errors.append(ValidationError(
                    error_id="config_invalid_superposition",
                    severity=ErrorSeverity.MEDIUM,
                    category="quantum_parameters",
                    message="superposition_width must be between 0 and 1",
                    suggestion="Use value between 0.05 and 0.2"
                ))
            
            if not (0 <= config.entanglement_strength <= 1):
                result.errors.append(ValidationError(
                    error_id="config_invalid_entanglement",
                    severity=ErrorSeverity.MEDIUM,
                    category="quantum_parameters",
                    message="entanglement_strength must be between 0 and 1",
                    suggestion="Use value between 0.3 and 0.8"
                ))
            
            # Performance warnings
            if config.max_iterations > 10000:
                result.warnings.append(ValidationError(
                    error_id="config_performance_warning",
                    severity=ErrorSeverity.LOW,
                    category="performance",
                    message="High max_iterations may impact performance",
                    suggestion="Consider using adaptive convergence"
                ))
            
            # Security validation
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                self._validate_config_security(config, result)
            
            # Update result validity
            result.valid = len(result.errors) == 0
            result.validation_time = (datetime.now() - start_time).total_seconds()
            
            return result
            
        except Exception as e:
            result.errors.append(ValidationError(
                error_id="config_validation_exception",
                severity=ErrorSeverity.CRITICAL,
                category="system",
                message=f"Configuration validation failed: {str(e)}",
                recoverable=False
            ))
            result.valid = False
            return result
    
    def validate_task_set(self, tasks: Dict[str, Task]) -> ValidationResult:
        """Validate set of tasks for consistency and correctness"""
        start_time = datetime.now()
        result = ValidationResult(valid=True)
        
        if not tasks:
            result.errors.append(ValidationError(
                error_id="task_set_empty",
                severity=ErrorSeverity.HIGH,
                category="task_validation",
                message="Task set is empty",
                suggestion="Add at least one task to plan"
            ))
            result.valid = False
            return result
        
        try:
            # Individual task validation
            for task_id, task in tasks.items():
                task_validation = self.validate_task(task)
                result.errors.extend(task_validation.errors)
                result.warnings.extend(task_validation.warnings)
            
            # Cross-task validation
            self._validate_task_dependencies(tasks, result)
            self._validate_resource_constraints(tasks, result)
            self._validate_task_priorities(tasks, result)
            
            # Complexity analysis
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                self._validate_task_complexity(tasks, result)
                self._validate_scheduling_feasibility(tasks, result)
            
            # Performance impact
            result.performance_score = self._calculate_performance_score(tasks, result)
            result.reliability_score = self._calculate_reliability_score(tasks, result)
            result.security_score = self._calculate_security_score(tasks, result)
            
            result.valid = len(result.errors) == 0
            result.validation_time = (datetime.now() - start_time).total_seconds()
            
            return result
            
        except Exception as e:
            result.errors.append(ValidationError(
                error_id="task_set_validation_exception",
                severity=ErrorSeverity.CRITICAL,
                category="system",
                message=f"Task set validation failed: {str(e)}",
                recoverable=False
            ))
            result.valid = False
            return result
    
    def validate_task(self, task: Task) -> ValidationResult:
        """Validate individual task"""
        result = ValidationResult(valid=True)
        
        try:
            # Basic validation
            if not task.id or not isinstance(task.id, str):
                result.errors.append(ValidationError(
                    error_id="task_invalid_id",
                    severity=ErrorSeverity.HIGH,
                    category="task_validation",
                    message="Task ID must be non-empty string",
                    task_id=getattr(task, 'id', None)
                ))
            
            if not task.name or not isinstance(task.name, str):
                result.errors.append(ValidationError(
                    error_id="task_invalid_name", 
                    severity=ErrorSeverity.MEDIUM,
                    category="task_validation",
                    message="Task name must be non-empty string",
                    task_id=task.id
                ))
            
            # Numeric validation
            if task.complexity < 0:
                result.errors.append(ValidationError(
                    error_id="task_negative_complexity",
                    severity=ErrorSeverity.MEDIUM,
                    category="task_validation",
                    message="Task complexity cannot be negative",
                    task_id=task.id,
                    suggestion="Set complexity to positive value"
                ))
            
            if task.priority <= 0:
                result.warnings.append(ValidationError(
                    error_id="task_zero_priority",
                    severity=ErrorSeverity.LOW,
                    category="task_validation",
                    message="Task priority should be positive",
                    task_id=task.id,
                    suggestion="Set priority > 0"
                ))
            
            if task.duration_estimate < 0:
                result.errors.append(ValidationError(
                    error_id="task_negative_duration",
                    severity=ErrorSeverity.HIGH,
                    category="task_validation",
                    message="Task duration cannot be negative",
                    task_id=task.id
                ))
            
            # Resource validation
            if task.resources:
                for resource_type, amount in task.resources.items():
                    if amount < 0:
                        result.errors.append(ValidationError(
                            error_id="task_negative_resource",
                            severity=ErrorSeverity.HIGH,
                            category="resource_validation",
                            message=f"Resource {resource_type} amount cannot be negative",
                            task_id=task.id
                        ))
            
            # Security validation
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                sanitized_id = self.input_sanitizer.sanitize_string(task.id)
                if sanitized_id != task.id:
                    result.warnings.append(ValidationError(
                        error_id="task_suspicious_id",
                        severity=ErrorSeverity.MEDIUM,
                        category="security",
                        message="Task ID contains potentially unsafe characters",
                        task_id=task.id,
                        suggestion="Use alphanumeric characters and underscores only"
                    ))
            
            result.valid = len(result.errors) == 0
            return result
            
        except Exception as e:
            result.errors.append(ValidationError(
                error_id="task_validation_exception",
                severity=ErrorSeverity.CRITICAL,
                category="system",
                message=f"Task validation failed: {str(e)}",
                task_id=getattr(task, 'id', None),
                recoverable=False
            ))
            result.valid = False
            return result
    
    def validate_planning_result(self, result: Dict[str, Any], 
                               original_tasks: Dict[str, Task]) -> ValidationResult:
        """Validate planning algorithm result"""
        validation_result = ValidationResult(valid=True)
        
        try:
            # Basic structure validation
            if 'schedule' not in result:
                validation_result.errors.append(ValidationError(
                    error_id="result_missing_schedule",
                    severity=ErrorSeverity.CRITICAL,
                    category="result_validation",
                    message="Planning result missing schedule"
                ))
            
            schedule = result.get('schedule', [])
            
            # Schedule validation
            if not schedule and original_tasks:
                validation_result.errors.append(ValidationError(
                    error_id="result_empty_schedule",
                    severity=ErrorSeverity.HIGH,
                    category="result_validation",
                    message="Schedule is empty but tasks were provided"
                ))
            
            # Task coverage validation
            scheduled_tasks = {event['task_id'] for event in schedule}
            missing_tasks = set(original_tasks.keys()) - scheduled_tasks
            
            if missing_tasks:
                validation_result.errors.append(ValidationError(
                    error_id="result_missing_tasks",
                    severity=ErrorSeverity.HIGH,
                    category="result_validation",
                    message=f"Tasks not scheduled: {list(missing_tasks)}",
                    suggestion="Check task dependencies and constraints"
                ))
            
            # Dependency order validation
            self._validate_schedule_dependencies(schedule, original_tasks, validation_result)
            
            # Time consistency validation
            self._validate_schedule_timing(schedule, validation_result)
            
            # Resource consistency validation
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                self._validate_schedule_resources(schedule, validation_result)
            
            # Quality metrics validation
            if 'quantum_fidelity' in result:
                fidelity = result['quantum_fidelity']
                if not (0 <= fidelity <= 1):
                    validation_result.warnings.append(ValidationError(
                        error_id="result_invalid_fidelity",
                        severity=ErrorSeverity.MEDIUM,
                        category="quality_validation",
                        message=f"Quantum fidelity {fidelity} outside valid range [0,1]"
                    ))
            
            validation_result.valid = len(validation_result.errors) == 0
            return validation_result
            
        except Exception as e:
            validation_result.errors.append(ValidationError(
                error_id="result_validation_exception",
                severity=ErrorSeverity.CRITICAL,
                category="system",
                message=f"Result validation failed: {str(e)}",
                recoverable=False
            ))
            validation_result.valid = False
            return validation_result
    
    def _validate_task_dependencies(self, tasks: Dict[str, Task], 
                                  result: ValidationResult) -> None:
        """Validate task dependency constraints"""
        # Check for circular dependencies
        def has_cycle(task_id: str, visited: Set[str], rec_stack: Set[str]) -> bool:
            if task_id not in tasks:
                return False
            
            visited.add(task_id)
            rec_stack.add(task_id)
            
            for dep_id in tasks[task_id].dependencies:
                if dep_id not in visited:
                    if has_cycle(dep_id, visited, rec_stack):
                        return True
                elif dep_id in rec_stack:
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        visited = set()
        for task_id in tasks:
            if task_id not in visited:
                if has_cycle(task_id, visited, set()):
                    result.errors.append(ValidationError(
                        error_id="dependency_cycle_detected",
                        severity=ErrorSeverity.HIGH,
                        category="dependency_validation",
                        message="Circular dependency detected in task graph",
                        suggestion="Remove circular dependencies"
                    ))
                    break
        
        # Check for missing dependencies
        for task_id, task in tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in tasks:
                    result.errors.append(ValidationError(
                        error_id="dependency_missing_task",
                        severity=ErrorSeverity.HIGH,
                        category="dependency_validation",
                        message=f"Task {task_id} depends on non-existent task {dep_id}",
                        task_id=task_id,
                        suggestion=f"Add task {dep_id} or remove dependency"
                    ))
    
    def _validate_resource_constraints(self, tasks: Dict[str, Task],
                                     result: ValidationResult) -> None:
        """Validate resource constraint feasibility"""
        # Aggregate resource requirements
        resource_totals = {}
        for task in tasks.values():
            for resource_type, amount in task.resources.items():
                if resource_type not in resource_totals:
                    resource_totals[resource_type] = 0
                resource_totals[resource_type] += amount
        
        # Check for unrealistic resource requirements
        for resource_type, total_amount in resource_totals.items():
            if total_amount > 1000:  # Configurable threshold
                result.warnings.append(ValidationError(
                    error_id="resource_high_demand",
                    severity=ErrorSeverity.LOW,
                    category="resource_validation",
                    message=f"High total demand for resource {resource_type}: {total_amount}",
                    suggestion="Consider resource optimization or scaling"
                ))
    
    def _validate_task_priorities(self, tasks: Dict[str, Task],
                                result: ValidationResult) -> None:
        """Validate task priority distribution"""
        priorities = [task.priority for task in tasks.values()]
        
        if not priorities:
            return
        
        # Check for priority range
        min_priority = min(priorities)
        max_priority = max(priorities)
        
        if max_priority / min_priority > 1000:  # Large priority spread
            result.warnings.append(ValidationError(
                error_id="priority_large_spread",
                severity=ErrorSeverity.LOW,
                category="priority_validation",
                message="Large spread in task priorities may cause scheduling issues",
                suggestion="Normalize priority values to reasonable range"
            ))
    
    def _validate_task_complexity(self, tasks: Dict[str, Task],
                                result: ValidationResult) -> None:
        """Validate task complexity distribution"""
        complexities = [task.complexity for task in tasks.values()]
        
        if not complexities:
            return
        
        mean_complexity = np.mean(complexities)
        std_complexity = np.std(complexities)
        
        # Check for outliers (>3 standard deviations)
        for task in tasks.values():
            if abs(task.complexity - mean_complexity) > 3 * std_complexity:
                result.warnings.append(ValidationError(
                    error_id="complexity_outlier",
                    severity=ErrorSeverity.LOW,
                    category="complexity_validation",
                    message=f"Task {task.id} has outlier complexity: {task.complexity}",
                    task_id=task.id,
                    suggestion="Review task complexity calculation"
                ))
    
    def _validate_scheduling_feasibility(self, tasks: Dict[str, Task],
                                       result: ValidationResult) -> None:
        """Validate overall scheduling feasibility"""
        # Calculate critical path length
        def calculate_critical_path(tasks_dict):
            # Simplified critical path calculation
            task_depths = {}
            
            def get_depth(task_id):
                if task_id in task_depths:
                    return task_depths[task_id]
                
                task = tasks_dict.get(task_id)
                if not task:
                    return 0
                
                max_dep_depth = 0
                for dep_id in task.dependencies:
                    max_dep_depth = max(max_dep_depth, get_depth(dep_id))
                
                depth = max_dep_depth + (task.duration_estimate or 1.0)
                task_depths[task_id] = depth
                return depth
            
            return max(get_depth(task_id) for task_id in tasks_dict) if tasks_dict else 0
        
        critical_path_length = calculate_critical_path(tasks)
        
        # Warn if critical path is very long
        if critical_path_length > 10000:  # Configurable threshold
            result.warnings.append(ValidationError(
                error_id="scheduling_long_critical_path",
                severity=ErrorSeverity.MEDIUM,
                category="scheduling_validation",
                message=f"Critical path length {critical_path_length} may cause long execution times",
                suggestion="Consider task parallelization or complexity reduction"
            ))
    
    def _validate_schedule_dependencies(self, schedule: List[Dict[str, Any]], 
                                      original_tasks: Dict[str, Task],
                                      result: ValidationResult) -> None:
        """Validate dependency order in schedule"""
        task_positions = {event['task_id']: i for i, event in enumerate(schedule)}
        
        for event in schedule:
            task_id = event['task_id']
            task = original_tasks.get(task_id)
            
            if not task:
                continue
            
            task_pos = task_positions.get(task_id, -1)
            
            for dep_id in task.dependencies:
                dep_pos = task_positions.get(dep_id, -1)
                
                if dep_pos >= task_pos or dep_pos == -1:
                    result.errors.append(ValidationError(
                        error_id="schedule_dependency_violation",
                        severity=ErrorSeverity.HIGH,
                        category="schedule_validation",
                        message=f"Task {task_id} scheduled before dependency {dep_id}",
                        task_id=task_id,
                        suggestion="Fix dependency ordering in schedule"
                    ))
    
    def _validate_schedule_timing(self, schedule: List[Dict[str, Any]],
                                result: ValidationResult) -> None:
        """Validate timing consistency in schedule"""
        for event in schedule:
            start_time = event.get('start_time', 0)
            end_time = event.get('end_time', 0)
            duration = event.get('duration', 0)
            
            if start_time < 0:
                result.errors.append(ValidationError(
                    error_id="schedule_negative_start_time",
                    severity=ErrorSeverity.HIGH,
                    category="schedule_validation",
                    message=f"Task {event.get('task_id')} has negative start time",
                    task_id=event.get('task_id')
                ))
            
            if end_time < start_time:
                result.errors.append(ValidationError(
                    error_id="schedule_invalid_time_order",
                    severity=ErrorSeverity.HIGH,
                    category="schedule_validation", 
                    message=f"Task {event.get('task_id')} end time before start time",
                    task_id=event.get('task_id')
                ))
            
            if duration != (end_time - start_time):
                result.warnings.append(ValidationError(
                    error_id="schedule_duration_mismatch",
                    severity=ErrorSeverity.LOW,
                    category="schedule_validation",
                    message=f"Task {event.get('task_id')} duration doesn't match time difference",
                    task_id=event.get('task_id')
                ))
    
    def _validate_schedule_resources(self, schedule: List[Dict[str, Any]],
                                   result: ValidationResult) -> None:
        """Validate resource allocation in schedule"""
        # Check for resource over-allocation at any time point
        time_points = []
        for event in schedule:
            time_points.extend([event.get('start_time', 0), event.get('end_time', 0)])
        
        time_points = sorted(set(time_points))
        
        for t in time_points[:-1]:  # Check each time interval
            active_resources = {}
            
            # Find all active tasks at time t
            for event in schedule:
                start = event.get('start_time', 0)
                end = event.get('end_time', 0)
                
                if start <= t < end:
                    # Task is active, add its resources
                    for resource_type, amount in event.get('resources', {}).items():
                        if resource_type not in active_resources:
                            active_resources[resource_type] = 0
                        active_resources[resource_type] += amount
            
            # Check if any resource is over-allocated
            # (This would require knowing resource limits, which aren't provided)
            # For now, just warn about high resource usage
            for resource_type, total_usage in active_resources.items():
                if total_usage > 100:  # Configurable threshold
                    result.warnings.append(ValidationError(
                        error_id="schedule_high_resource_usage",
                        severity=ErrorSeverity.LOW,
                        category="resource_validation",
                        message=f"High {resource_type} usage ({total_usage}) at time {t}",
                        suggestion="Consider resource optimization or load balancing"
                    ))
    
    def _validate_config_security(self, config: PlanningConfig, result: ValidationResult) -> None:
        """Validate configuration security aspects"""
        # Check for potentially unsafe configurations
        if config.enable_monitoring and not hasattr(config, 'monitoring_secure'):
            result.warnings.append(ValidationError(
                error_id="config_monitoring_security",
                severity=ErrorSeverity.MEDIUM,
                category="security",
                message="Monitoring enabled without explicit security configuration",
                suggestion="Configure secure monitoring settings"
            ))
    
    def _calculate_performance_score(self, tasks: Dict[str, Task], 
                                   result: ValidationResult) -> float:
        """Calculate performance impact score"""
        base_score = 1.0
        
        # Penalize based on errors
        for error in result.errors:
            if error.category == "performance":
                base_score *= 0.8
        
        for warning in result.warnings:
            if warning.category == "performance":
                base_score *= 0.95
        
        return max(0.0, base_score)
    
    def _calculate_reliability_score(self, tasks: Dict[str, Task],
                                   result: ValidationResult) -> float:
        """Calculate reliability score"""
        base_score = 1.0
        
        # Factor in error severity
        for error in result.errors:
            if error.severity == ErrorSeverity.CRITICAL:
                base_score *= 0.5
            elif error.severity == ErrorSeverity.HIGH:
                base_score *= 0.7
            elif error.severity == ErrorSeverity.MEDIUM:
                base_score *= 0.9
        
        return max(0.0, base_score)
    
    def _calculate_security_score(self, tasks: Dict[str, Task],
                                result: ValidationResult) -> float:
        """Calculate security score"""
        base_score = 1.0
        
        # Penalize security issues
        for error in result.errors + result.warnings:
            if error.category == "security":
                if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                    base_score *= 0.6
                else:
                    base_score *= 0.9
        
        return max(0.0, base_score)
    
    def generate_validation_report(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        if not validation_results:
            return {"error": "No validation results to report"}
        
        # Aggregate statistics
        total_errors = sum(len(r.errors) for r in validation_results)
        total_warnings = sum(len(r.warnings) for r in validation_results)
        
        # Error categorization
        error_categories = {}
        severity_counts = {}
        
        for result in validation_results:
            for error in result.errors + result.warnings:
                # Category counting
                if error.category not in error_categories:
                    error_categories[error.category] = 0
                error_categories[error.category] += 1
                
                # Severity counting
                if error.severity.value not in severity_counts:
                    severity_counts[error.severity.value] = 0
                severity_counts[error.severity.value] += 1
        
        # Performance metrics
        avg_performance_score = np.mean([r.performance_score for r in validation_results])
        avg_reliability_score = np.mean([r.reliability_score for r in validation_results])
        avg_security_score = np.mean([r.security_score for r in validation_results])
        
        return {
            "validation_summary": {
                "total_validations": len(validation_results),
                "successful_validations": sum(r.valid for r in validation_results),
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "validation_level": self.validation_level.value
            },
            "error_analysis": {
                "by_category": error_categories,
                "by_severity": severity_counts,
                "most_common_errors": self._get_most_common_errors(validation_results)
            },
            "quality_scores": {
                "average_performance": avg_performance_score,
                "average_reliability": avg_reliability_score,
                "average_security": avg_security_score,
                "overall_health": (avg_performance_score + avg_reliability_score + avg_security_score) / 3
            },
            "recommendations": self._generate_recommendations(validation_results)
        }
    
    def _get_most_common_errors(self, results: List[ValidationResult]) -> List[Dict[str, Any]]:
        """Get most common error patterns"""
        error_counts = {}
        
        for result in results:
            for error in result.errors + result.warnings:
                error_key = f"{error.category}:{error.error_id}"
                if error_key not in error_counts:
                    error_counts[error_key] = {
                        "error_id": error.error_id,
                        "category": error.category,
                        "severity": error.severity.value,
                        "count": 0,
                        "sample_message": error.message
                    }
                error_counts[error_key]["count"] += 1
        
        # Sort by count and return top 5
        sorted_errors = sorted(error_counts.values(), key=lambda x: x["count"], reverse=True)
        return sorted_errors[:5]
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Analyze error patterns and generate recommendations
        error_categories = {}
        for result in results:
            for error in result.errors:
                if error.category not in error_categories:
                    error_categories[error.category] = 0
                error_categories[error.category] += 1
        
        # Generate category-specific recommendations
        if error_categories.get("dependency_validation", 0) > 0:
            recommendations.append("Review task dependency graphs for cycles and missing dependencies")
        
        if error_categories.get("resource_validation", 0) > 0:
            recommendations.append("Optimize resource allocation and consider scaling strategies")
        
        if error_categories.get("security", 0) > 0:
            recommendations.append("Strengthen security validation and input sanitization")
        
        if error_categories.get("configuration", 0) > 0:
            recommendations.append("Review and validate configuration parameters")
        
        if not recommendations:
            recommendations.append("No specific recommendations - validation results look good!")
        
        return recommendations


class ResilientPlanningWrapper:
    """
    Resilient wrapper for planning operations with automatic error recovery
    
    Provides automatic retry, fallback strategies, and error recovery
    for robust planning operations.
    """
    
    def __init__(self, planner_class, max_retries: int = 3, 
                 fallback_strategies: List[str] = None):
        self.planner_class = planner_class
        self.max_retries = max_retries
        self.fallback_strategies = fallback_strategies or ["simple", "greedy"]
        self.validator = PlanningValidator(ValidationLevel.STANDARD)
        
        # Error tracking
        self._error_count = 0
        self._recovery_attempts = 0
        self._last_error = None
    
    def resilient_plan(self, planner_instance, *args, **kwargs) -> Dict[str, Any]:
        """Execute planning with resilience and error recovery"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Attempt planning
                result = planner_instance.plan(*args, **kwargs)
                
                # Validate result
                if hasattr(planner_instance, 'tasks'):
                    validation = self.validator.validate_planning_result(result, planner_instance.tasks)
                    
                    if not validation.valid:
                        # Try to recover from validation errors
                        recovered_result = self._attempt_error_recovery(result, validation, planner_instance)
                        if recovered_result:
                            result = recovered_result
                
                # Success
                if attempt > 0:
                    result['recovery_info'] = {
                        'attempts': attempt + 1,
                        'recovered': True,
                        'strategy_used': 'retry'
                    }
                
                return result
                
            except Exception as e:
                last_exception = e
                self._error_count += 1
                
                if attempt < self.max_retries:
                    # Try fallback strategy
                    try:
                        fallback_result = self._try_fallback_strategy(planner_instance, attempt, *args, **kwargs)
                        if fallback_result:
                            fallback_result['recovery_info'] = {
                                'attempts': attempt + 1,
                                'recovered': True,
                                'strategy_used': self.fallback_strategies[attempt % len(self.fallback_strategies)]
                            }
                            return fallback_result
                    except Exception:
                        pass  # Continue to retry
        
        # All attempts failed
        raise Exception(f"Planning failed after {self.max_retries + 1} attempts. Last error: {last_exception}")
    
    def _attempt_error_recovery(self, result: Dict[str, Any], validation: ValidationResult,
                              planner_instance) -> Optional[Dict[str, Any]]:
        """Attempt to recover from validation errors"""
        self._recovery_attempts += 1
        
        # Simple recovery strategies
        recoverable_errors = [e for e in validation.errors if e.recoverable]
        
        if not recoverable_errors:
            return None
        
        # Try to fix common issues
        fixed_result = result.copy()
        
        for error in recoverable_errors:
            if error.error_id == "result_missing_tasks":
                # Try to add missing tasks with simple scheduling
                pass  # Implementation would depend on specific error
            
            elif error.error_id == "schedule_dependency_violation":
                # Try to reorder schedule to fix dependencies
                pass  # Implementation would require dependency graph analysis
        
        return fixed_result if fixed_result != result else None
    
    def _try_fallback_strategy(self, planner_instance, attempt: int, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Try fallback planning strategy"""
        strategy = self.fallback_strategies[attempt % len(self.fallback_strategies)]
        
        if strategy == "simple":
            # Simple greedy strategy
            return self._simple_greedy_planning(planner_instance)
        elif strategy == "greedy":
            # Priority-based greedy strategy
            return self._priority_greedy_planning(planner_instance)
        
        return None
    
    def _simple_greedy_planning(self, planner_instance) -> Dict[str, Any]:
        """Simple greedy fallback planning"""
        if not hasattr(planner_instance, 'tasks'):
            return None
        
        tasks = list(planner_instance.tasks.values())
        # Sort by complexity (ascending)
        tasks.sort(key=lambda t: t.complexity)
        
        schedule = []
        current_time = 0
        
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
        
        return {
            'schedule': schedule,
            'total_time': current_time,
            'quantum_fidelity': 0.5,  # Default for fallback
            'convergence_achieved': True,
            'fallback_strategy': 'simple_greedy'
        }
    
    def _priority_greedy_planning(self, planner_instance) -> Dict[str, Any]:
        """Priority-based greedy fallback planning"""
        if not hasattr(planner_instance, 'tasks'):
            return None
        
        tasks = list(planner_instance.tasks.values())
        # Sort by priority (descending)
        tasks.sort(key=lambda t: -t.priority)
        
        schedule = []
        current_time = 0
        
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
        
        return {
            'schedule': schedule,
            'total_time': current_time,
            'quantum_fidelity': 0.6,  # Slightly better for priority-based
            'convergence_achieved': True,
            'fallback_strategy': 'priority_greedy'
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error and recovery statistics"""
        return {
            'total_errors': self._error_count,
            'recovery_attempts': self._recovery_attempts,
            'last_error': str(self._last_error) if self._last_error else None,
            'fallback_strategies': self.fallback_strategies,
            'max_retries': self.max_retries
        }