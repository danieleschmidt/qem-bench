"""
Distributed computing system for quantum error mitigation workloads.

This module provides sophisticated distributed execution capabilities including
task scheduling, result aggregation, fault tolerance, and coordinated execution
across multiple nodes and quantum backends.
"""

import time
import logging
import asyncio
import hashlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures

from ..security import SecureConfig, AccessControl
from ..monitoring import MetricsCollector


logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class NodeStatus(Enum):
    """Distributed node status."""
    AVAILABLE = "available"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class FaultHandlingStrategy(Enum):
    """Fault handling strategies."""
    RETRY = "retry"
    REDIRECT = "redirect"
    REPLICATE = "replicate"
    ABORT = "abort"


@dataclass
class DistributedTask:
    """Represents a task for distributed execution."""
    id: str
    function_name: str
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Task metadata
    priority: TaskPriority = TaskPriority.NORMAL
    created_time: float = field(default_factory=time.time)
    timeout: Optional[float] = None
    max_retries: int = 3
    
    # Resource requirements
    min_memory_gb: float = 1.0
    min_cpu_cores: int = 1
    required_backends: List[str] = field(default_factory=list)
    gpu_required: bool = False
    
    # Dependencies and constraints
    dependencies: List[str] = field(default_factory=list)
    node_affinity: Optional[str] = None
    node_anti_affinity: List[str] = field(default_factory=list)
    
    # Execution state
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    started_time: Optional[float] = None
    completed_time: Optional[float] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    result: Optional[Any] = None
    
    def duration(self) -> Optional[float]:
        """Get task execution duration."""
        if self.started_time and self.completed_time:
            return self.completed_time - self.started_time
        return None
    
    def age(self) -> float:
        """Get task age in seconds."""
        return time.time() - self.created_time
    
    def is_expired(self) -> bool:
        """Check if task has exceeded timeout."""
        if self.timeout is None:
            return False
        return self.age() > self.timeout
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries
    
    def priority_score(self) -> float:
        """Calculate priority score for scheduling."""
        base_score = self.priority.value * 1000
        age_bonus = min(self.age() / 60.0, 100)  # Up to 100 points for age
        return base_score + age_bonus
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "function_name": self.function_name,
            "priority": self.priority.value,
            "created_time": self.created_time,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "min_memory_gb": self.min_memory_gb,
            "min_cpu_cores": self.min_cpu_cores,
            "required_backends": self.required_backends,
            "gpu_required": self.gpu_required,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "assigned_node": self.assigned_node,
            "retry_count": self.retry_count,
            "duration": self.duration(),
            "age": self.age()
        }


@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""
    id: str
    hostname: str
    port: int
    
    # Node capabilities
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    available_backends: List[str] = field(default_factory=list)
    
    # Node state
    status: NodeStatus = NodeStatus.AVAILABLE
    current_tasks: int = 0
    max_concurrent_tasks: int = 4
    
    # Performance metrics
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    network_latency: float = 0.0
    reliability_score: float = 1.0
    
    # Health tracking
    last_heartbeat: float = field(default_factory=time.time)
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_task_duration: float = 60.0
    
    def is_available(self) -> bool:
        """Check if node is available for new tasks."""
        return (self.status == NodeStatus.AVAILABLE and 
                self.current_tasks < self.max_concurrent_tasks)
    
    def utilization(self) -> float:
        """Calculate node utilization."""
        return self.current_tasks / max(self.max_concurrent_tasks, 1)
    
    def is_healthy(self, timeout: float = 300.0) -> bool:
        """Check if node is healthy based on heartbeat."""
        return time.time() - self.last_heartbeat < timeout
    
    def success_rate(self) -> float:
        """Calculate task success rate."""
        total_tasks = self.total_tasks_completed + self.total_tasks_failed
        if total_tasks == 0:
            return 1.0
        return self.total_tasks_completed / total_tasks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            "id": self.id,
            "hostname": self.hostname,
            "port": self.port,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "gpu_count": self.gpu_count,
            "available_backends": self.available_backends,
            "status": self.status.value,
            "current_tasks": self.current_tasks,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "network_latency": self.network_latency,
            "reliability_score": self.reliability_score,
            "utilization": self.utilization(),
            "is_healthy": self.is_healthy(),
            "success_rate": self.success_rate()
        }


class TaskScheduler:
    """
    Intelligent task scheduler for distributed quantum computing workloads.
    
    Features:
    - Priority-based scheduling with dependency resolution
    - Resource-aware node selection
    - Load balancing across compute nodes
    - Fault-tolerant scheduling with automatic retry
    - Performance-based node scoring
    """
    
    def __init__(self, config: Optional[SecureConfig] = None):
        self.config = config or SecureConfig()
        
        # Scheduling state
        self.pending_tasks: List[DistributedTask] = []
        self.running_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: List[DistributedTask] = []
        self.nodes: Dict[str, ComputeNode] = {}
        
        # Scheduling policies
        self.max_queue_size = 1000
        self.scheduling_interval = 5.0  # seconds
        self.node_health_check_interval = 30.0  # seconds
        
        # Statistics
        self.total_tasks_scheduled = 0
        self.scheduling_decisions: List[Dict[str, Any]] = []
        
        logger.info("TaskScheduler initialized")
    
    def add_node(self, node: ComputeNode) -> None:
        """Add a compute node to the scheduler."""
        self.nodes[node.id] = node
        logger.info(f"Added compute node: {node.id}")
    
    def remove_node(self, node_id: str) -> None:
        """Remove a compute node from the scheduler."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Removed compute node: {node_id}")
    
    async def submit_task(self, task: DistributedTask) -> bool:
        """Submit a task for scheduling."""
        if len(self.pending_tasks) >= self.max_queue_size:
            logger.warning(f"Task queue is full, rejecting task {task.id}")
            return False
        
        # Check dependencies
        if not self._validate_dependencies(task):
            logger.error(f"Task {task.id} has invalid dependencies")
            return False
        
        self.pending_tasks.append(task)
        logger.info(f"Task {task.id} submitted for scheduling")
        return True
    
    def _validate_dependencies(self, task: DistributedTask) -> bool:
        """Validate task dependencies."""
        for dep_id in task.dependencies:
            # Check if dependency exists and is completed
            if not any(t.id == dep_id and t.status == TaskStatus.COMPLETED 
                      for t in self.completed_tasks):
                # Check if dependency is still running
                if dep_id not in self.running_tasks:
                    return False
        return True
    
    async def schedule_tasks(self) -> None:
        """Schedule pending tasks to available nodes."""
        if not self.pending_tasks or not self.nodes:
            return
        
        # Sort tasks by priority
        self.pending_tasks.sort(key=lambda t: t.priority_score(), reverse=True)
        
        scheduled_tasks = []
        
        for task in self.pending_tasks:
            # Check if task is expired
            if task.is_expired():
                task.status = TaskStatus.CANCELLED
                task.error_message = "Task expired"
                self.completed_tasks.append(task)
                scheduled_tasks.append(task)
                continue
            
            # Check if dependencies are satisfied
            if not self._dependencies_satisfied(task):
                continue
            
            # Find suitable node
            selected_node = self._select_node_for_task(task)
            
            if selected_node:
                # Schedule task
                await self._schedule_task_to_node(task, selected_node)
                scheduled_tasks.append(task)
                self.total_tasks_scheduled += 1
        
        # Remove scheduled tasks from pending queue
        for task in scheduled_tasks:
            if task in self.pending_tasks:
                self.pending_tasks.remove(task)
    
    def _dependencies_satisfied(self, task: DistributedTask) -> bool:
        """Check if all task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if not any(t.id == dep_id and t.status == TaskStatus.COMPLETED 
                      for t in self.completed_tasks):
                return False
        return True
    
    def _select_node_for_task(self, task: DistributedTask) -> Optional[ComputeNode]:
        """Select the best node for a task."""
        suitable_nodes = []
        
        for node in self.nodes.values():
            if self._node_meets_requirements(node, task):
                suitable_nodes.append(node)
        
        if not suitable_nodes:
            return None
        
        # Score nodes and select the best one
        scored_nodes = []
        for node in suitable_nodes:
            score = self._calculate_node_score(node, task)
            scored_nodes.append((score, node))
        
        # Sort by score (highest first)
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        return scored_nodes[0][1]
    
    def _node_meets_requirements(self, node: ComputeNode, task: DistributedTask) -> bool:
        """Check if node meets task requirements."""
        # Check availability
        if not node.is_available() or not node.is_healthy():
            return False
        
        # Check resource requirements
        if (node.cpu_cores < task.min_cpu_cores or 
            node.memory_gb < task.min_memory_gb):
            return False
        
        # Check GPU requirements
        if task.gpu_required and node.gpu_count == 0:
            return False
        
        # Check backend requirements
        if (task.required_backends and 
            not all(backend in node.available_backends 
                   for backend in task.required_backends)):
            return False
        
        # Check affinity constraints
        if (task.node_affinity and task.node_affinity != node.id):
            return False
        
        if task.node_anti_affinity and node.id in task.node_anti_affinity:
            return False
        
        return True
    
    def _calculate_node_score(self, node: ComputeNode, task: DistributedTask) -> float:
        """Calculate score for node-task pair."""
        score = 0.0
        
        # Base score from node performance
        score += node.reliability_score * 100
        
        # Penalize high utilization
        score -= node.utilization() * 50
        
        # Bonus for lower latency
        score += max(0, 50 - node.network_latency)
        
        # Bonus for matching resources (avoid waste)
        cpu_efficiency = 1.0 - max(0, (node.cpu_cores - task.min_cpu_cores) / node.cpu_cores)
        memory_efficiency = 1.0 - max(0, (node.memory_gb - task.min_memory_gb) / node.memory_gb)
        score += (cpu_efficiency + memory_efficiency) * 25
        
        # Bonus for required backends
        if task.required_backends:
            backend_match_ratio = len(set(task.required_backends) & 
                                    set(node.available_backends)) / len(task.required_backends)
            score += backend_match_ratio * 30
        
        return score
    
    async def _schedule_task_to_node(self, task: DistributedTask, node: ComputeNode) -> None:
        """Schedule a task to a specific node."""
        task.status = TaskStatus.SCHEDULED
        task.assigned_node = node.id
        
        # Update node state
        node.current_tasks += 1
        if node.current_tasks >= node.max_concurrent_tasks:
            node.status = NodeStatus.BUSY
        
        # Move to running tasks
        self.running_tasks[task.id] = task
        
        # Record scheduling decision
        decision = {
            "timestamp": time.time(),
            "task_id": task.id,
            "node_id": node.id,
            "node_score": self._calculate_node_score(node, task),
            "node_utilization": node.utilization()
        }
        self.scheduling_decisions.append(decision)
        
        logger.info(f"Scheduled task {task.id} to node {node.id}")
    
    async def complete_task(
        self, 
        task_id: str, 
        result: Any = None, 
        error: Optional[str] = None
    ) -> None:
        """Mark a task as completed or failed."""
        if task_id not in self.running_tasks:
            logger.warning(f"Attempted to complete unknown task: {task_id}")
            return
        
        task = self.running_tasks[task_id]
        task.completed_time = time.time()
        
        if error:
            task.status = TaskStatus.FAILED
            task.error_message = error
            
            # Handle task failure
            await self._handle_task_failure(task)
        else:
            task.status = TaskStatus.COMPLETED
            task.result = result
        
        # Update node state
        if task.assigned_node and task.assigned_node in self.nodes:
            node = self.nodes[task.assigned_node]
            node.current_tasks = max(0, node.current_tasks - 1)
            
            if task.status == TaskStatus.COMPLETED:
                node.total_tasks_completed += 1
                if task.duration():
                    # Update average duration
                    total_duration = (node.average_task_duration * 
                                    (node.total_tasks_completed - 1) + task.duration())
                    node.average_task_duration = total_duration / node.total_tasks_completed
            else:
                node.total_tasks_failed += 1
            
            # Update node status
            if node.status == NodeStatus.BUSY and node.current_tasks < node.max_concurrent_tasks:
                node.status = NodeStatus.AVAILABLE
        
        # Move to completed tasks
        self.completed_tasks.append(task)
        del self.running_tasks[task_id]
        
        logger.info(f"Task {task_id} completed with status {task.status.value}")
    
    async def _handle_task_failure(self, task: DistributedTask) -> None:
        """Handle task failure with retry logic."""
        if task.can_retry():
            task.retry_count += 1
            task.status = TaskStatus.RETRYING
            task.assigned_node = None
            
            # Add back to pending queue for retry
            self.pending_tasks.append(task)
            logger.info(f"Task {task.id} queued for retry ({task.retry_count}/{task.max_retries})")
        else:
            logger.error(f"Task {task.id} failed permanently after {task.retry_count} retries")
    
    async def update_node_health(self, node_id: str, health_data: Dict[str, Any]) -> None:
        """Update node health metrics."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.last_heartbeat = time.time()
        node.cpu_utilization = health_data.get("cpu_utilization", node.cpu_utilization)
        node.memory_utilization = health_data.get("memory_utilization", node.memory_utilization)
        node.network_latency = health_data.get("network_latency", node.network_latency)
        
        # Update node status based on health
        if node.cpu_utilization > 90 or node.memory_utilization > 90:
            node.status = NodeStatus.OVERLOADED
        elif node.utilization() >= 1.0:
            node.status = NodeStatus.BUSY
        else:
            node.status = NodeStatus.AVAILABLE
    
    def get_scheduling_stats(self) -> Dict[str, Any]:
        """Get scheduling statistics."""
        return {
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "total_scheduled": self.total_tasks_scheduled,
            "active_nodes": len([n for n in self.nodes.values() if n.is_healthy()]),
            "total_nodes": len(self.nodes),
            "average_queue_time": self._calculate_average_queue_time(),
            "node_utilization": self._calculate_average_node_utilization()
        }
    
    def _calculate_average_queue_time(self) -> float:
        """Calculate average time tasks spend in queue."""
        if not self.completed_tasks:
            return 0.0
        
        queue_times = []
        for task in self.completed_tasks[-100:]:  # Last 100 tasks
            if task.started_time:
                queue_time = task.started_time - task.created_time
                queue_times.append(queue_time)
        
        return np.mean(queue_times) if queue_times else 0.0
    
    def _calculate_average_node_utilization(self) -> float:
        """Calculate average node utilization."""
        if not self.nodes:
            return 0.0
        return np.mean([node.utilization() for node in self.nodes.values()])


class ResultAggregator:
    """
    Aggregates and combines results from distributed quantum computing tasks.
    
    Features:
    - Statistical aggregation of quantum measurement results
    - Error mitigation result combination
    - Confidence interval calculation
    - Result validation and consistency checking
    """
    
    def __init__(self):
        self.aggregation_methods = {
            "mean": self._aggregate_mean,
            "weighted_mean": self._aggregate_weighted_mean,
            "median": self._aggregate_median,
            "majority_vote": self._aggregate_majority_vote,
            "confidence_weighted": self._aggregate_confidence_weighted
        }
        
        logger.info("ResultAggregator initialized")
    
    def aggregate_results(
        self,
        results: List[Dict[str, Any]],
        method: str = "confidence_weighted",
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple distributed tasks.
        
        Args:
            results: List of result dictionaries from distributed tasks
            method: Aggregation method to use
            confidence_level: Confidence level for interval estimation
            
        Returns:
            Aggregated result with confidence intervals
        """
        if not results:
            return {"error": "No results to aggregate"}
        
        if method not in self.aggregation_methods:
            logger.error(f"Unknown aggregation method: {method}")
            method = "mean"
        
        try:
            # Filter valid results
            valid_results = [r for r in results if self._is_valid_result(r)]
            
            if not valid_results:
                return {"error": "No valid results to aggregate"}
            
            # Apply aggregation method
            aggregated = self.aggregation_methods[method](valid_results)
            
            # Add metadata
            aggregated.update({
                "aggregation_method": method,
                "num_results": len(valid_results),
                "num_total_inputs": len(results),
                "confidence_level": confidence_level,
                "aggregation_timestamp": time.time()
            })
            
            # Calculate confidence intervals if possible
            if len(valid_results) > 1:
                confidence_intervals = self._calculate_confidence_intervals(
                    valid_results, confidence_level
                )
                aggregated["confidence_intervals"] = confidence_intervals
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating results: {e}")
            return {"error": f"Aggregation failed: {str(e)}"}
    
    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        """Check if a result is valid for aggregation."""
        if not isinstance(result, dict):
            return False
        
        # Check for required fields
        required_fields = ["value", "error_estimate"]
        return all(field in result for field in required_fields)
    
    def _aggregate_mean(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple mean aggregation."""
        values = [r["value"] for r in results]
        errors = [r.get("error_estimate", 0.0) for r in results]
        
        aggregated_value = np.mean(values)
        aggregated_error = np.sqrt(np.sum(np.array(errors)**2)) / len(errors)
        
        return {
            "value": float(aggregated_value),
            "error_estimate": float(aggregated_error),
            "std_dev": float(np.std(values))
        }
    
    def _aggregate_weighted_mean(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Weighted mean aggregation based on inverse error."""
        values = np.array([r["value"] for r in results])
        errors = np.array([max(r.get("error_estimate", 1.0), 1e-10) for r in results])
        
        # Weights are inverse of squared errors
        weights = 1.0 / (errors ** 2)
        weights = weights / np.sum(weights)  # Normalize
        
        aggregated_value = np.sum(values * weights)
        aggregated_error = 1.0 / np.sqrt(np.sum(1.0 / (errors ** 2)))
        
        return {
            "value": float(aggregated_value),
            "error_estimate": float(aggregated_error),
            "effective_samples": float(np.sum(weights) ** 2 / np.sum(weights ** 2))
        }
    
    def _aggregate_median(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Median aggregation for robust estimation."""
        values = [r["value"] for r in results]
        
        aggregated_value = np.median(values)
        mad = np.median(np.abs(values - aggregated_value))  # Median Absolute Deviation
        error_estimate = 1.4826 * mad  # Convert MAD to std estimate
        
        return {
            "value": float(aggregated_value),
            "error_estimate": float(error_estimate),
            "median_absolute_deviation": float(mad)
        }
    
    def _aggregate_majority_vote(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Majority vote aggregation for discrete results."""
        values = [r["value"] for r in results]
        
        # Find most common value
        unique_values, counts = np.unique(values, return_counts=True)
        majority_idx = np.argmax(counts)
        majority_value = unique_values[majority_idx]
        confidence = counts[majority_idx] / len(values)
        
        return {
            "value": float(majority_value),
            "confidence": float(confidence),
            "vote_distribution": dict(zip(unique_values.tolist(), counts.tolist()))
        }
    
    def _aggregate_confidence_weighted(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregation weighted by result confidence scores."""
        values = np.array([r["value"] for r in results])
        confidences = np.array([r.get("confidence", 1.0) for r in results])
        errors = np.array([r.get("error_estimate", 1.0) for r in results])
        
        # Combine confidence and error information for weights
        weights = confidences / (errors + 1e-10)
        weights = weights / np.sum(weights)  # Normalize
        
        aggregated_value = np.sum(values * weights)
        aggregated_error = np.sqrt(np.sum((errors * weights) ** 2))
        
        return {
            "value": float(aggregated_value),
            "error_estimate": float(aggregated_error),
            "average_confidence": float(np.mean(confidences)),
            "weight_distribution": weights.tolist()
        }
    
    def _calculate_confidence_intervals(
        self,
        results: List[Dict[str, Any]],
        confidence_level: float
    ) -> Dict[str, float]:
        """Calculate confidence intervals for aggregated results."""
        values = np.array([r["value"] for r in results])
        
        if len(values) < 2:
            return {}
        
        # Calculate percentiles for confidence interval
        alpha = 1.0 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        return {
            "lower": float(np.percentile(values, lower_percentile)),
            "upper": float(np.percentile(values, upper_percentile)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }
    
    def validate_result_consistency(
        self,
        results: List[Dict[str, Any]],
        tolerance: float = 0.1
    ) -> Dict[str, Any]:
        """Validate consistency across distributed results."""
        if len(results) < 2:
            return {"consistent": True, "message": "Insufficient results for consistency check"}
        
        values = np.array([r["value"] for r in results if self._is_valid_result(r)])
        
        if len(values) == 0:
            return {"consistent": False, "message": "No valid results found"}
        
        # Check relative standard deviation
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        if mean_value != 0:
            relative_std = std_value / abs(mean_value)
        else:
            relative_std = std_value
        
        is_consistent = relative_std <= tolerance
        
        # Identify outliers
        outliers = []
        for i, value in enumerate(values):
            if abs(value - mean_value) > 2 * std_value:
                outliers.append(i)
        
        return {
            "consistent": is_consistent,
            "relative_std": float(relative_std),
            "tolerance": tolerance,
            "num_outliers": len(outliers),
            "outlier_indices": outliers,
            "mean": float(mean_value),
            "std": float(std_value)
        }


class FaultTolerantExecutor:
    """
    Fault-tolerant executor for distributed quantum computing tasks.
    
    Features:
    - Automatic failure detection and recovery
    - Node failure handling with task redistribution
    - Circuit checkpointing and restart capabilities
    - Adaptive retry strategies
    - Health monitoring and alerting
    """
    
    def __init__(
        self,
        task_scheduler: TaskScheduler,
        result_aggregator: ResultAggregator,
        config: Optional[SecureConfig] = None
    ):
        self.task_scheduler = task_scheduler
        self.result_aggregator = result_aggregator
        self.config = config or SecureConfig()
        
        # Fault tolerance settings
        self.max_node_failures = 3
        self.failure_detection_interval = 30.0  # seconds
        self.recovery_strategies = {
            FaultHandlingStrategy.RETRY: self._handle_retry,
            FaultHandlingStrategy.REDIRECT: self._handle_redirect,
            FaultHandlingStrategy.REPLICATE: self._handle_replicate,
            FaultHandlingStrategy.ABORT: self._handle_abort
        }
        
        # Failure tracking
        self.node_failures: Dict[str, List[float]] = {}
        self.task_failures: Dict[str, List[float]] = {}
        self.recovery_actions: List[Dict[str, Any]] = []
        
        # Health monitoring
        self.is_monitoring = False
        
        logger.info("FaultTolerantExecutor initialized")
    
    async def start_monitoring(self) -> None:
        """Start fault detection and recovery monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        logger.info("Starting fault tolerance monitoring")
        
        # Start monitoring loops
        asyncio.create_task(self._node_health_monitoring_loop())
        asyncio.create_task(self._task_failure_monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop fault tolerance monitoring."""
        self.is_monitoring = False
        logger.info("Stopped fault tolerance monitoring")
    
    async def _node_health_monitoring_loop(self) -> None:
        """Monitor node health and detect failures."""
        while self.is_monitoring:
            try:
                await self._check_node_health()
                await asyncio.sleep(self.failure_detection_interval)
            except Exception as e:
                logger.error(f"Error in node health monitoring: {e}")
                await asyncio.sleep(self.failure_detection_interval)
    
    async def _task_failure_monitoring_loop(self) -> None:
        """Monitor task failures and trigger recovery."""
        while self.is_monitoring:
            try:
                await self._check_task_failures()
                await asyncio.sleep(self.failure_detection_interval)
            except Exception as e:
                logger.error(f"Error in task failure monitoring: {e}")
                await asyncio.sleep(self.failure_detection_interval)
    
    async def _check_node_health(self) -> None:
        """Check health of all nodes and detect failures."""
        current_time = time.time()
        
        for node_id, node in self.task_scheduler.nodes.items():
            if not node.is_healthy():
                # Node failure detected
                await self._handle_node_failure(node_id, current_time)
    
    async def _check_task_failures(self) -> None:
        """Check for task failures and trigger recovery."""
        current_time = time.time()
        
        # Check for tasks that have been running too long
        for task_id, task in self.task_scheduler.running_tasks.items():
            if (task.timeout and 
                task.started_time and 
                current_time - task.started_time > task.timeout):
                
                logger.warning(f"Task {task_id} timed out")
                await self._handle_task_timeout(task_id)
    
    async def _handle_node_failure(self, node_id: str, failure_time: float) -> None:
        """Handle node failure with appropriate recovery strategy."""
        logger.error(f"Node failure detected: {node_id}")
        
        # Record failure
        if node_id not in self.node_failures:
            self.node_failures[node_id] = []
        self.node_failures[node_id].append(failure_time)
        
        # Update node status
        if node_id in self.task_scheduler.nodes:
            self.task_scheduler.nodes[node_id].status = NodeStatus.FAILED
        
        # Find tasks running on failed node
        affected_tasks = [
            task for task in self.task_scheduler.running_tasks.values()
            if task.assigned_node == node_id
        ]
        
        # Recover affected tasks
        for task in affected_tasks:
            await self._recover_task(task, FaultHandlingStrategy.REDIRECT)
        
        # Record recovery action
        recovery_action = {
            "timestamp": failure_time,
            "type": "node_failure",
            "node_id": node_id,
            "affected_tasks": len(affected_tasks),
            "recovery_strategy": FaultHandlingStrategy.REDIRECT.value
        }
        self.recovery_actions.append(recovery_action)
    
    async def _handle_task_timeout(self, task_id: str) -> None:
        """Handle task timeout."""
        if task_id not in self.task_scheduler.running_tasks:
            return
        
        task = self.task_scheduler.running_tasks[task_id]
        logger.warning(f"Task {task_id} timed out after {task.timeout} seconds")
        
        # Mark task as failed and trigger recovery
        await self.task_scheduler.complete_task(
            task_id, 
            error="Task timeout"
        )
        
        # The task scheduler will handle retry automatically
    
    async def _recover_task(
        self, 
        task: DistributedTask, 
        strategy: FaultHandlingStrategy
    ) -> None:
        """Recover a failed task using specified strategy."""
        logger.info(f"Recovering task {task.id} using strategy {strategy.value}")
        
        if strategy in self.recovery_strategies:
            await self.recovery_strategies[strategy](task)
        else:
            logger.error(f"Unknown recovery strategy: {strategy}")
            await self._handle_abort(task)
    
    async def _handle_retry(self, task: DistributedTask) -> None:
        """Handle task recovery with retry strategy."""
        if task.can_retry():
            # Reset task state for retry
            task.status = TaskStatus.PENDING
            task.assigned_node = None
            task.started_time = None
            task.retry_count += 1
            
            # Add back to pending queue
            self.task_scheduler.pending_tasks.append(task)
            
            # Remove from running tasks
            if task.id in self.task_scheduler.running_tasks:
                del self.task_scheduler.running_tasks[task.id]
            
            logger.info(f"Task {task.id} queued for retry ({task.retry_count}/{task.max_retries})")
        else:
            await self._handle_abort(task)
    
    async def _handle_redirect(self, task: DistributedTask) -> None:
        """Handle task recovery with redirect strategy."""
        # Find alternative node
        available_nodes = [
            node for node in self.task_scheduler.nodes.values()
            if (node.is_available() and node.is_healthy() and 
                node.id != task.assigned_node)
        ]
        
        if available_nodes:
            # Reset task for rescheduling
            task.status = TaskStatus.PENDING
            task.assigned_node = None
            task.started_time = None
            
            # Add back to pending queue
            self.task_scheduler.pending_tasks.append(task)
            
            # Remove from running tasks
            if task.id in self.task_scheduler.running_tasks:
                del self.task_scheduler.running_tasks[task.id]
            
            logger.info(f"Task {task.id} redirected to alternative node")
        else:
            logger.warning(f"No alternative nodes available for task {task.id}")
            await self._handle_retry(task)
    
    async def _handle_replicate(self, task: DistributedTask) -> None:
        """Handle task recovery with replication strategy."""
        # Create multiple copies of the task
        replication_factor = min(3, len([n for n in self.task_scheduler.nodes.values() 
                                       if n.is_available() and n.is_healthy()]))
        
        if replication_factor > 1:
            for i in range(replication_factor - 1):
                replica_task = DistributedTask(
                    id=f"{task.id}_replica_{i}",
                    function_name=task.function_name,
                    args=task.args,
                    kwargs=task.kwargs,
                    priority=task.priority,
                    timeout=task.timeout,
                    max_retries=task.max_retries
                )
                
                await self.task_scheduler.submit_task(replica_task)
            
            logger.info(f"Task {task.id} replicated {replication_factor} times")
        
        # Original task continues as redirect
        await self._handle_redirect(task)
    
    async def _handle_abort(self, task: DistributedTask) -> None:
        """Handle task recovery with abort strategy."""
        task.status = TaskStatus.FAILED
        task.error_message = "Task aborted due to unrecoverable failure"
        task.completed_time = time.time()
        
        # Move to completed tasks
        self.task_scheduler.completed_tasks.append(task)
        
        # Remove from running tasks
        if task.id in self.task_scheduler.running_tasks:
            del self.task_scheduler.running_tasks[task.id]
        
        logger.error(f"Task {task.id} aborted due to unrecoverable failure")
    
    def get_fault_tolerance_stats(self) -> Dict[str, Any]:
        """Get fault tolerance statistics."""
        current_time = time.time()
        
        # Calculate failure rates
        recent_window = 3600.0  # 1 hour
        recent_node_failures = sum(
            len([f for f in failures if current_time - f < recent_window])
            for failures in self.node_failures.values()
        )
        
        recent_task_failures = sum(
            len([f for f in failures if current_time - f < recent_window])
            for failures in self.task_failures.values()
        )
        
        # Calculate recovery success rate
        total_recoveries = len(self.recovery_actions)
        successful_recoveries = len([
            action for action in self.recovery_actions
            if action.get("success", True)
        ])
        
        recovery_rate = (successful_recoveries / max(total_recoveries, 1)) * 100
        
        return {
            "recent_node_failures": recent_node_failures,
            "recent_task_failures": recent_task_failures,
            "total_recovery_actions": total_recoveries,
            "recovery_success_rate": recovery_rate,
            "failed_nodes": len(self.node_failures),
            "monitoring_active": self.is_monitoring,
            "fault_detection_interval": self.failure_detection_interval
        }


class DistributedExecutor:
    """
    Main distributed execution coordinator for quantum error mitigation workloads.
    
    This class orchestrates the entire distributed computing pipeline including
    task submission, scheduling, execution, result aggregation, and fault tolerance.
    
    Example:
        >>> executor = DistributedExecutor()
        >>> await executor.start()
        >>> 
        >>> # Add compute nodes
        >>> node = ComputeNode("node1", "192.168.1.100", 8080, cpu_cores=8, memory_gb=32)
        >>> executor.add_node(node)
        >>> 
        >>> # Submit distributed task
        >>> task = DistributedTask("task1", "run_zne_experiment", args=(circuit, backend))
        >>> result = await executor.execute_task(task)
    """
    
    def __init__(self, config: Optional[SecureConfig] = None):
        self.config = config or SecureConfig()
        
        # Core components
        self.task_scheduler = TaskScheduler(config)
        self.result_aggregator = ResultAggregator()
        self.fault_tolerant_executor = FaultTolerantExecutor(
            self.task_scheduler, 
            self.result_aggregator, 
            config
        )
        
        # Execution state
        self.is_running = False
        
        # Statistics
        self.execution_stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0
        }
        
        logger.info("DistributedExecutor initialized")
    
    async def start(self) -> None:
        """Start the distributed execution system."""
        if self.is_running:
            logger.warning("DistributedExecutor is already running")
            return
        
        self.is_running = True
        logger.info("Starting DistributedExecutor")
        
        # Start fault tolerance monitoring
        await self.fault_tolerant_executor.start_monitoring()
        
        # Start scheduling loop
        asyncio.create_task(self._scheduling_loop())
    
    async def stop(self) -> None:
        """Stop the distributed execution system."""
        self.is_running = False
        
        # Stop fault tolerance monitoring
        await self.fault_tolerant_executor.stop_monitoring()
        
        logger.info("DistributedExecutor stopped")
    
    def add_node(self, node: ComputeNode) -> None:
        """Add a compute node to the distributed system."""
        self.task_scheduler.add_node(node)
    
    def remove_node(self, node_id: str) -> None:
        """Remove a compute node from the distributed system."""
        self.task_scheduler.remove_node(node_id)
    
    async def execute_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute a single distributed task."""
        self.execution_stats["tasks_submitted"] += 1
        
        # Submit task to scheduler
        success = await self.task_scheduler.submit_task(task)
        if not success:
            self.execution_stats["tasks_failed"] += 1
            return {"error": "Failed to submit task to scheduler"}
        
        # Wait for task completion
        return await self._wait_for_task_completion(task.id)
    
    async def execute_batch(
        self, 
        tasks: List[DistributedTask],
        aggregation_method: str = "confidence_weighted"
    ) -> Dict[str, Any]:
        """Execute a batch of related tasks and aggregate results."""
        # Submit all tasks
        submitted_tasks = []
        for task in tasks:
            success = await self.task_scheduler.submit_task(task)
            if success:
                submitted_tasks.append(task)
                self.execution_stats["tasks_submitted"] += 1
        
        if not submitted_tasks:
            return {"error": "No tasks could be submitted"}
        
        # Wait for all tasks to complete
        results = []
        for task in submitted_tasks:
            result = await self._wait_for_task_completion(task.id)
            if "error" not in result:
                results.append(result)
        
        # Aggregate results
        if results:
            aggregated = self.result_aggregator.aggregate_results(
                results, method=aggregation_method
            )
            return {
                "aggregated_result": aggregated,
                "individual_results": results,
                "batch_stats": {
                    "submitted_tasks": len(submitted_tasks),
                    "completed_tasks": len(results),
                    "success_rate": len(results) / len(submitted_tasks)
                }
            }
        else:
            return {"error": "No tasks completed successfully"}
    
    async def _scheduling_loop(self) -> None:
        """Main scheduling loop."""
        while self.is_running:
            try:
                await self.task_scheduler.schedule_tasks()
                await asyncio.sleep(self.task_scheduler.scheduling_interval)
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                await asyncio.sleep(self.task_scheduler.scheduling_interval)
    
    async def _wait_for_task_completion(
        self, 
        task_id: str, 
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Wait for a task to complete and return its result."""
        start_time = time.time()
        
        while self.is_running:
            # Check if task is completed
            completed_task = None
            for task in self.task_scheduler.completed_tasks:
                if task.id == task_id:
                    completed_task = task
                    break
            
            if completed_task:
                if completed_task.status == TaskStatus.COMPLETED:
                    self.execution_stats["tasks_completed"] += 1
                    execution_time = completed_task.duration() or 0.0
                    self.execution_stats["total_execution_time"] += execution_time
                    
                    return {
                        "task_id": task_id,
                        "status": "completed",
                        "result": completed_task.result,
                        "execution_time": execution_time,
                        "node_id": completed_task.assigned_node
                    }
                else:
                    self.execution_stats["tasks_failed"] += 1
                    return {
                        "task_id": task_id,
                        "status": "failed",
                        "error": completed_task.error_message,
                        "retry_count": completed_task.retry_count
                    }
            
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                return {"error": f"Task {task_id} timed out"}
            
            # Wait before checking again
            await asyncio.sleep(1.0)
        
        return {"error": "Executor stopped while waiting for task"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        scheduling_stats = self.task_scheduler.get_scheduling_stats()
        fault_tolerance_stats = self.fault_tolerant_executor.get_fault_tolerance_stats()
        
        return {
            "executor_running": self.is_running,
            "execution_stats": self.execution_stats.copy(),
            "scheduling_stats": scheduling_stats,
            "fault_tolerance_stats": fault_tolerance_stats,
            "nodes": [node.to_dict() for node in self.task_scheduler.nodes.values()],
            "timestamp": time.time()
        }