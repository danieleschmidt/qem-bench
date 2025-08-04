"""
Resource scheduling and allocation system for optimal quantum computing workload distribution.

This module provides sophisticated scheduling algorithms that optimize resource
allocation across quantum backends, classical compute nodes, and cloud instances
based on workload characteristics, cost constraints, and performance requirements.
"""

import time
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum
import heapq
import numpy as np
from datetime import datetime, timedelta

from ..security import SecureConfig, AccessControl
from ..monitoring import MetricsCollector


logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of computational resources."""
    QUANTUM_BACKEND = "quantum_backend"
    CLASSICAL_CPU = "classical_cpu"
    CLASSICAL_GPU = "classical_gpu"
    CLOUD_INSTANCE = "cloud_instance"
    SIMULATOR = "simulator"


class ResourceState(Enum):
    """Resource availability states."""
    AVAILABLE = "available"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    FAILED = "failed"
    WARMING_UP = "warming_up"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class SchedulingStrategy(Enum):
    """Resource scheduling strategies."""
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    COST_OPTIMIZED = "cost_optimized"
    LATENCY_OPTIMIZED = "latency_optimized"


@dataclass
class Resource:
    """Represents a computational resource."""
    id: str
    type: ResourceType
    state: ResourceState = ResourceState.AVAILABLE
    
    # Capacity metrics
    max_concurrent_jobs: int = 1
    current_jobs: int = 0
    cpu_cores: int = 1
    memory_gb: float = 1.0
    
    # Performance characteristics
    performance_score: float = 1.0
    reliability_score: float = 1.0
    average_job_duration: float = 60.0  # seconds
    
    # Cost information
    cost_per_second: float = 0.001
    cost_per_job: float = 0.0
    
    # Quantum-specific attributes
    num_qubits: Optional[int] = None
    gate_fidelity: Optional[float] = None
    readout_fidelity: Optional[float] = None
    connectivity_map: Optional[Dict] = None
    
    # Scheduling metadata
    last_assigned_time: float = 0.0
    total_jobs_completed: int = 0
    total_execution_time: float = 0.0
    
    def is_available(self) -> bool:
        """Check if resource is available for new jobs."""
        return (self.state == ResourceState.AVAILABLE and 
                self.current_jobs < self.max_concurrent_jobs)
    
    def utilization(self) -> float:
        """Calculate current resource utilization."""
        return self.current_jobs / max(self.max_concurrent_jobs, 1)
    
    def estimated_wait_time(self) -> float:
        """Estimate wait time for new jobs."""
        if self.is_available():
            return 0.0
        else:
            # Simple estimation based on current load
            return self.average_job_duration * self.current_jobs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert resource to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "state": self.state.value,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "current_jobs": self.current_jobs,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "performance_score": self.performance_score,
            "reliability_score": self.reliability_score,
            "average_job_duration": self.average_job_duration,
            "cost_per_second": self.cost_per_second,
            "cost_per_job": self.cost_per_job,
            "num_qubits": self.num_qubits,
            "gate_fidelity": self.gate_fidelity,
            "readout_fidelity": self.readout_fidelity,
            "utilization": self.utilization(),
            "estimated_wait_time": self.estimated_wait_time()
        }


@dataclass
class Job:
    """Represents a computational job to be scheduled."""
    id: str
    priority: JobPriority
    submitted_time: float = field(default_factory=time.time)
    
    # Resource requirements
    required_resource_types: List[ResourceType] = field(default_factory=list)
    min_qubits: Optional[int] = None
    estimated_duration: float = 60.0  # seconds
    cpu_requirements: int = 1
    memory_requirements: float = 1.0  # GB
    
    # Scheduling constraints
    deadline: Optional[float] = None
    max_cost: Optional[float] = None
    preferred_backends: List[str] = field(default_factory=list)
    excluded_backends: List[str] = field(default_factory=list)
    
    # Job metadata
    user_id: Optional[str] = None
    experiment_id: Optional[str] = None
    circuit_depth: Optional[int] = None
    shots: int = 1024
    
    # Scheduling state
    assigned_resource: Optional[str] = None
    scheduled_time: Optional[float] = None
    started_time: Optional[float] = None
    completed_time: Optional[float] = None
    
    def age(self) -> float:
        """Get job age in seconds."""
        return time.time() - self.submitted_time
    
    def is_expired(self) -> bool:
        """Check if job has exceeded its deadline."""
        return self.deadline is not None and time.time() > self.deadline
    
    def priority_score(self) -> float:
        """Calculate priority score for scheduling."""
        base_score = self.priority.value * 1000
        
        # Add age bonus (older jobs get higher priority)
        age_bonus = min(self.age() / 60.0, 100)  # Up to 100 points for age
        
        # Add deadline urgency
        urgency_bonus = 0
        if self.deadline:
            time_to_deadline = self.deadline - time.time()
            if time_to_deadline > 0:
                urgency_bonus = max(0, 200 - time_to_deadline / 60.0)
        
        return base_score + age_bonus + urgency_bonus
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "id": self.id,
            "priority": self.priority.value,
            "submitted_time": self.submitted_time,
            "required_resource_types": [rt.value for rt in self.required_resource_types],
            "min_qubits": self.min_qubits,
            "estimated_duration": self.estimated_duration,
            "cpu_requirements": self.cpu_requirements,
            "memory_requirements": self.memory_requirements,
            "deadline": self.deadline,
            "max_cost": self.max_cost,
            "preferred_backends": self.preferred_backends,
            "excluded_backends": self.excluded_backends,
            "assigned_resource": self.assigned_resource,
            "age": self.age(),
            "priority_score": self.priority_score()
        }


@dataclass
class ResourceAllocation:
    """Represents an allocation of a job to a resource."""
    job_id: str
    resource_id: str
    allocation_time: float = field(default_factory=time.time)
    estimated_start_time: float = 0.0
    estimated_completion_time: float = 0.0
    estimated_cost: float = 0.0
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert allocation to dictionary."""
        return {
            "job_id": self.job_id,
            "resource_id": self.resource_id,
            "allocation_time": self.allocation_time,
            "estimated_start_time": self.estimated_start_time,
            "estimated_completion_time": self.estimated_completion_time,
            "estimated_cost": self.estimated_cost,
            "confidence": self.confidence
        }


@dataclass
class SchedulingPolicy:
    """Policy configuration for resource scheduling."""
    strategy: SchedulingStrategy = SchedulingStrategy.LOAD_BALANCED
    
    # Performance thresholds
    max_queue_length: int = 100
    max_wait_time_seconds: float = 300.0  # 5 minutes
    target_utilization: float = 0.8
    
    # Cost constraints
    enable_cost_optimization: bool = True
    cost_weight: float = 0.3  # Weight for cost in scoring
    performance_weight: float = 0.7  # Weight for performance in scoring
    
    # Failure handling
    max_retries: int = 3
    retry_delay_seconds: float = 60.0
    
    # Resource management
    enable_preemption: bool = False
    preemption_threshold: float = 2.0  # Priority difference for preemption
    resource_warmup_time: float = 30.0
    
    # Load balancing
    load_balance_threshold: float = 0.2  # Max utilization difference
    round_robin_enabled: bool = True
    
    def __post_init__(self):
        """Validate policy configuration."""
        if self.cost_weight + self.performance_weight != 1.0:
            raise ValueError("Cost and performance weights must sum to 1.0")
        if self.target_utilization <= 0 or self.target_utilization > 1.0:
            raise ValueError("Target utilization must be between 0 and 1")


class SchedulingAlgorithm(ABC):
    """Abstract base class for scheduling algorithms."""
    
    @abstractmethod
    def select_resource(
        self,
        job: Job,
        available_resources: List[Resource],
        policy: SchedulingPolicy
    ) -> Optional[Resource]:
        """Select the best resource for a given job."""
        pass


class FirstFitScheduler(SchedulingAlgorithm):
    """First-fit scheduling algorithm."""
    
    def select_resource(
        self,
        job: Job,
        available_resources: List[Resource],
        policy: SchedulingPolicy
    ) -> Optional[Resource]:
        """Select first available resource that meets requirements."""
        for resource in available_resources:
            if self._resource_meets_requirements(job, resource):
                return resource
        return None
    
    def _resource_meets_requirements(self, job: Job, resource: Resource) -> bool:
        """Check if resource meets job requirements."""
        # Check resource type
        if (job.required_resource_types and 
            resource.type not in job.required_resource_types):
            return False
        
        # Check quantum requirements
        if (job.min_qubits and 
            (resource.num_qubits is None or resource.num_qubits < job.min_qubits)):
            return False
        
        # Check exclusions
        if resource.id in job.excluded_backends:
            return False
        
        # Check basic capacity
        if not resource.is_available():
            return False
        
        return True


class BestFitScheduler(SchedulingAlgorithm):
    """Best-fit scheduling algorithm with multi-criteria optimization."""
    
    def select_resource(
        self,
        job: Job,
        available_resources: List[Resource],
        policy: SchedulingPolicy
    ) -> Optional[Resource]:
        """Select best resource based on multiple criteria."""
        suitable_resources = [
            r for r in available_resources 
            if self._resource_meets_requirements(job, r)
        ]
        
        if not suitable_resources:
            return None
        
        # Score resources based on multiple criteria
        scored_resources = []
        for resource in suitable_resources:
            score = self._calculate_resource_score(job, resource, policy)
            scored_resources.append((score, resource))
        
        # Return highest scoring resource
        scored_resources.sort(key=lambda x: x[0], reverse=True)
        return scored_resources[0][1]
    
    def _resource_meets_requirements(self, job: Job, resource: Resource) -> bool:
        """Check if resource meets job requirements."""
        # Check resource type
        if (job.required_resource_types and 
            resource.type not in job.required_resource_types):
            return False
        
        # Check quantum requirements
        if (job.min_qubits and 
            (resource.num_qubits is None or resource.num_qubits < job.min_qubits)):
            return False
        
        # Check exclusions
        if resource.id in job.excluded_backends:
            return False
        
        # Check basic capacity
        if not resource.is_available():
            return False
        
        # Check cost constraints
        if job.max_cost:
            estimated_cost = resource.cost_per_second * job.estimated_duration
            if estimated_cost > job.max_cost:
                return False
        
        return True
    
    def _calculate_resource_score(
        self,
        job: Job,
        resource: Resource,
        policy: SchedulingPolicy
    ) -> float:
        """Calculate score for resource-job pair."""
        score = 0.0
        
        # Performance score component
        performance_score = (
            resource.performance_score * 0.4 +
            resource.reliability_score * 0.3 +
            (1.0 - resource.utilization()) * 0.3
        )
        
        # Cost score component (lower cost = higher score)
        if policy.enable_cost_optimization:
            estimated_cost = resource.cost_per_second * job.estimated_duration
            max_reasonable_cost = 1.0  # Normalize against reasonable maximum
            cost_score = max(0, 1.0 - estimated_cost / max_reasonable_cost)
        else:
            cost_score = 0.5  # Neutral cost score
        
        # Wait time penalty
        wait_time_penalty = min(resource.estimated_wait_time() / 300.0, 1.0)
        
        # Preference bonus
        preference_bonus = 0.0
        if resource.id in job.preferred_backends:
            preference_bonus = 0.2
        
        # Quantum-specific bonuses
        quantum_bonus = 0.0
        if (resource.type == ResourceType.QUANTUM_BACKEND and 
            job.min_qubits and resource.num_qubits):
            # Prefer resources with just enough qubits (avoid waste)
            qubit_efficiency = 1.0 - max(0, (resource.num_qubits - job.min_qubits) / resource.num_qubits)
            quantum_bonus = qubit_efficiency * 0.1
        
        # Combine scores
        score = (
            performance_score * policy.performance_weight +
            cost_score * policy.cost_weight -
            wait_time_penalty * 0.2 +
            preference_bonus +
            quantum_bonus
        )
        
        return score


class LoadBalancedScheduler(SchedulingAlgorithm):
    """Load-balanced scheduling algorithm."""
    
    def select_resource(
        self,
        job: Job,
        available_resources: List[Resource],
        policy: SchedulingPolicy
    ) -> Optional[Resource]:
        """Select resource to balance load across resources."""
        suitable_resources = [
            r for r in available_resources 
            if self._resource_meets_requirements(job, r)
        ]
        
        if not suitable_resources:
            return None
        
        # Sort by utilization (lowest first)
        suitable_resources.sort(key=lambda r: r.utilization())
        
        # Check if load balancing is needed
        if len(suitable_resources) > 1:
            min_util = suitable_resources[0].utilization()
            max_util = suitable_resources[-1].utilization()
            
            if max_util - min_util > policy.load_balance_threshold:
                # Return least utilized resource
                return suitable_resources[0]
        
        # Fall back to best-fit
        best_fit = BestFitScheduler()
        return best_fit.select_resource(job, suitable_resources, policy)
    
    def _resource_meets_requirements(self, job: Job, resource: Resource) -> bool:
        """Check if resource meets job requirements."""
        return BestFitScheduler()._resource_meets_requirements(job, resource)


class ResourceScheduler:
    """
    Advanced resource scheduler for quantum computing workloads.
    
    Features:
    - Multiple scheduling algorithms (first-fit, best-fit, load-balanced)
    - Multi-criteria optimization (cost, performance, latency)
    - Queue management with priority handling
    - Resource health monitoring and failure recovery
    - Cost-aware scheduling with budget constraints
    - Quantum-specific scheduling optimizations
    
    Example:
        >>> policy = SchedulingPolicy(
        ...     strategy=SchedulingStrategy.BEST_FIT,
        ...     enable_cost_optimization=True
        ... )
        >>> scheduler = ResourceScheduler(policy=policy)
        >>> await scheduler.start()
        >>> job = Job(id="job1", priority=JobPriority.HIGH)
        >>> allocation = await scheduler.schedule_job(job)
    """
    
    def __init__(
        self,
        policy: Optional[SchedulingPolicy] = None,
        config: Optional[SecureConfig] = None
    ):
        self.policy = policy or SchedulingPolicy()
        self.config = config or SecureConfig()
        
        # Resource and job tracking
        self.resources: Dict[str, Resource] = {}
        self.job_queue: List[Job] = []
        self.active_jobs: Dict[str, Job] = {}
        self.completed_jobs: List[Job] = []
        self.allocations: Dict[str, ResourceAllocation] = {}
        
        # Scheduling algorithms
        self.algorithms = {
            SchedulingStrategy.FIRST_FIT: FirstFitScheduler(),
            SchedulingStrategy.BEST_FIT: BestFitScheduler(),
            SchedulingStrategy.LOAD_BALANCED: LoadBalancedScheduler(),
        }
        
        # Scheduler state
        self.is_running = False
        self.scheduling_interval = 5.0  # seconds
        
        # Statistics
        self.total_jobs_scheduled = 0
        self.total_scheduling_time = 0.0
        self.failed_scheduling_attempts = 0
        
        logger.info(f"ResourceScheduler initialized with policy: {policy}")
    
    async def start(self) -> None:
        """Start the resource scheduler."""
        if self.is_running:
            logger.warning("ResourceScheduler is already running")
            return
        
        self.is_running = True
        logger.info("Starting ResourceScheduler")
        
        # Start scheduling loop
        asyncio.create_task(self._scheduling_loop())
        
        # Start resource monitoring
        asyncio.create_task(self._resource_monitoring_loop())
    
    async def stop(self) -> None:
        """Stop the resource scheduler."""
        self.is_running = False
        logger.info("ResourceScheduler stopped")
    
    def add_resource(self, resource: Resource) -> None:
        """Add a resource to the scheduler."""
        self.resources[resource.id] = resource
        logger.info(f"Added resource: {resource.id} ({resource.type.value})")
    
    def remove_resource(self, resource_id: str) -> None:
        """Remove a resource from the scheduler."""
        if resource_id in self.resources:
            del self.resources[resource_id]
            logger.info(f"Removed resource: {resource_id}")
    
    async def submit_job(self, job: Job) -> bool:
        """Submit a job to the scheduler."""
        # Validate job
        if job.is_expired():
            logger.warning(f"Job {job.id} is already expired, rejecting")
            return False
        
        # Check queue capacity
        if len(self.job_queue) >= self.policy.max_queue_length:
            logger.warning(f"Job queue is full, rejecting job {job.id}")
            return False
        
        # Add to queue
        self.job_queue.append(job)
        logger.info(f"Job {job.id} submitted with priority {job.priority.value}")
        
        return True
    
    async def _scheduling_loop(self) -> None:
        """Main scheduling loop."""
        while self.is_running:
            try:
                if self.job_queue:
                    await self._process_job_queue()
                
                await asyncio.sleep(self.scheduling_interval)
                
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                await asyncio.sleep(self.scheduling_interval)
    
    async def _process_job_queue(self) -> None:
        """Process jobs in the queue."""
        # Sort queue by priority
        self.job_queue.sort(key=lambda j: j.priority_score(), reverse=True)
        
        # Try to schedule jobs
        scheduled_jobs = []
        
        for job in self.job_queue:
            # Check if job is expired
            if job.is_expired():
                logger.warning(f"Job {job.id} expired, removing from queue")
                scheduled_jobs.append(job)
                continue
            
            # Try to schedule job
            allocation = await self._schedule_single_job(job)
            
            if allocation:
                # Job successfully scheduled
                self.allocations[job.id] = allocation
                self.active_jobs[job.id] = job
                scheduled_jobs.append(job)
                self.total_jobs_scheduled += 1
                
                logger.info(f"Scheduled job {job.id} on resource {allocation.resource_id}")
            
            # Break if we've processed enough jobs for this cycle
            if len(scheduled_jobs) >= 10:  # Process up to 10 jobs per cycle
                break
        
        # Remove scheduled jobs from queue
        for job in scheduled_jobs:
            self.job_queue.remove(job)
    
    async def _schedule_single_job(self, job: Job) -> Optional[ResourceAllocation]:
        """Schedule a single job."""
        start_time = time.time()
        
        try:
            # Get available resources
            available_resources = [
                r for r in self.resources.values()
                if r.is_available()
            ]
            
            if not available_resources:
                return None
            
            # Select scheduling algorithm
            algorithm = self.algorithms.get(
                self.policy.strategy,
                self.algorithms[SchedulingStrategy.BEST_FIT]
            )
            
            # Select resource
            selected_resource = algorithm.select_resource(
                job, available_resources, self.policy
            )
            
            if selected_resource:
                # Create allocation
                allocation = self._create_allocation(job, selected_resource)
                
                # Update resource state
                selected_resource.current_jobs += 1
                selected_resource.last_assigned_time = time.time()
                
                return allocation
            else:
                self.failed_scheduling_attempts += 1
                return None
                
        except Exception as e:
            logger.error(f"Error scheduling job {job.id}: {e}")
            self.failed_scheduling_attempts += 1
            return None
        
        finally:
            self.total_scheduling_time += time.time() - start_time
    
    def _create_allocation(self, job: Job, resource: Resource) -> ResourceAllocation:
        """Create a resource allocation for a job."""
        current_time = time.time()
        
        # Calculate timing estimates
        estimated_start_time = current_time + resource.estimated_wait_time()
        estimated_completion_time = estimated_start_time + job.estimated_duration
        
        # Calculate cost estimate
        estimated_cost = (
            resource.cost_per_second * job.estimated_duration +
            resource.cost_per_job
        )
        
        # Calculate confidence based on resource reliability and current load
        confidence = resource.reliability_score * (1.0 - resource.utilization() * 0.5)
        
        return ResourceAllocation(
            job_id=job.id,
            resource_id=resource.id,
            allocation_time=current_time,
            estimated_start_time=estimated_start_time,
            estimated_completion_time=estimated_completion_time,
            estimated_cost=estimated_cost,
            confidence=confidence
        )
    
    async def _resource_monitoring_loop(self) -> None:
        """Monitor resource health and availability."""
        while self.is_running:
            try:
                for resource in self.resources.values():
                    await self._update_resource_metrics(resource)
                
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(30.0)
    
    async def _update_resource_metrics(self, resource: Resource) -> None:
        """Update resource performance metrics."""
        # This would integrate with actual resource monitoring
        # For now, simulate basic updates
        
        # Update average job duration based on completed jobs
        if resource.total_jobs_completed > 0:
            resource.average_job_duration = (
                resource.total_execution_time / resource.total_jobs_completed
            )
        
        # Check resource health (simulated)
        # In practice, this would ping the resource or check metrics
        if resource.state == ResourceState.FAILED:
            # Try to recover failed resources
            if time.time() - resource.last_assigned_time > 300:  # 5 minutes
                resource.state = ResourceState.AVAILABLE
                logger.info(f"Resource {resource.id} recovered from failure")
    
    async def complete_job(self, job_id: str, success: bool = True) -> None:
        """Mark a job as completed."""
        if job_id not in self.active_jobs:
            logger.warning(f"Attempted to complete unknown job: {job_id}")
            return
        
        job = self.active_jobs[job_id]
        job.completed_time = time.time()
        
        # Update resource state
        if job.assigned_resource and job.assigned_resource in self.resources:
            resource = self.resources[job.assigned_resource]
            resource.current_jobs = max(0, resource.current_jobs - 1)
            
            if success:
                resource.total_jobs_completed += 1
                execution_time = job.completed_time - (job.started_time or job.completed_time)
                resource.total_execution_time += execution_time
        
        # Move job to completed list
        self.completed_jobs.append(job)
        del self.active_jobs[job_id]
        
        if job_id in self.allocations:
            del self.allocations[job_id]
        
        logger.info(f"Job {job_id} completed {'successfully' if success else 'with failure'}")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            "queued_jobs": len(self.job_queue),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "total_scheduled": self.total_jobs_scheduled,
            "failed_attempts": self.failed_scheduling_attempts,
            "average_scheduling_time": (
                self.total_scheduling_time / max(self.total_jobs_scheduled, 1)
            ),
            "queue_by_priority": {
                priority.name: len([j for j in self.job_queue if j.priority == priority])
                for priority in JobPriority
            }
        }
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        total_resources = len(self.resources)
        available_resources = len([r for r in self.resources.values() if r.is_available()])
        
        utilization_by_type = {}
        for resource_type in ResourceType:
            type_resources = [r for r in self.resources.values() if r.type == resource_type]
            if type_resources:
                avg_util = np.mean([r.utilization() for r in type_resources])
                utilization_by_type[resource_type.value] = avg_util
        
        return {
            "total_resources": total_resources,
            "available_resources": available_resources,
            "busy_resources": total_resources - available_resources,
            "overall_utilization": np.mean([r.utilization() for r in self.resources.values()]),
            "utilization_by_type": utilization_by_type,
            "resources": [r.to_dict() for r in self.resources.values()]
        }
    
    async def optimize_schedule(self) -> Dict[str, Any]:
        """Analyze and optimize current scheduling performance."""
        # Analyze queue patterns
        queue_analysis = self._analyze_queue_patterns()
        
        # Analyze resource utilization
        utilization_analysis = self._analyze_resource_utilization()
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            queue_analysis, utilization_analysis
        )
        
        return {
            "queue_analysis": queue_analysis,
            "utilization_analysis": utilization_analysis,
            "recommendations": recommendations
        }
    
    def _analyze_queue_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in job queue."""
        if not self.job_queue:
            return {"message": "No jobs in queue"}
        
        # Analyze wait times by priority
        wait_times_by_priority = {}
        for priority in JobPriority:
            jobs_with_priority = [j for j in self.job_queue if j.priority == priority]
            if jobs_with_priority:
                avg_wait = np.mean([j.age() for j in jobs_with_priority])
                wait_times_by_priority[priority.name] = avg_wait
        
        # Analyze resource requirements
        resource_demand = {}
        for resource_type in ResourceType:
            demand = len([
                j for j in self.job_queue 
                if resource_type in j.required_resource_types
            ])
            resource_demand[resource_type.value] = demand
        
        return {
            "queue_length": len(self.job_queue),
            "average_wait_time": np.mean([j.age() for j in self.job_queue]),
            "wait_times_by_priority": wait_times_by_priority,
            "resource_demand": resource_demand,
            "longest_waiting_job": max(self.job_queue, key=lambda j: j.age()).id
        }
    
    def _analyze_resource_utilization(self) -> Dict[str, Any]:
        """Analyze resource utilization patterns."""
        if not self.resources:
            return {"message": "No resources available"}
        
        utilizations = [r.utilization() for r in self.resources.values()]
        
        # Find bottlenecks
        bottlenecks = [
            r.id for r in self.resources.values() 
            if r.utilization() > 0.9
        ]
        
        # Find underutilized resources
        underutilized = [
            r.id for r in self.resources.values()
            if r.utilization() < 0.2 and r.is_available()
        ]
        
        return {
            "average_utilization": np.mean(utilizations),
            "max_utilization": np.max(utilizations),
            "min_utilization": np.min(utilizations),
            "utilization_std": np.std(utilizations),
            "bottlenecks": bottlenecks,
            "underutilized": underutilized,
            "load_balance_score": 1.0 - np.std(utilizations)  # Higher is better
        }
    
    def _generate_optimization_recommendations(
        self,
        queue_analysis: Dict[str, Any],
        utilization_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for scheduling optimization."""
        recommendations = []
        
        # Queue-based recommendations
        if queue_analysis.get("queue_length", 0) > self.policy.max_queue_length * 0.8:
            recommendations.append(
                "Queue is getting full - consider adding more resources or "
                "optimizing job priorities"
            )
        
        avg_wait = queue_analysis.get("average_wait_time", 0)
        if avg_wait > self.policy.max_wait_time_seconds:
            recommendations.append(
                f"Average wait time ({avg_wait:.1f}s) exceeds threshold - "
                "consider scaling up resources"
            )
        
        # Utilization-based recommendations
        avg_util = utilization_analysis.get("average_utilization", 0)
        if avg_util > 0.9:
            recommendations.append(
                "Resources are highly utilized - consider adding capacity"
            )
        elif avg_util < 0.3:
            recommendations.append(
                "Resources are underutilized - consider scaling down to save costs"
            )
        
        load_balance_score = utilization_analysis.get("load_balance_score", 1.0)
        if load_balance_score < 0.7:
            recommendations.append(
                "Load is imbalanced across resources - consider using "
                "load-balanced scheduling strategy"
            )
        
        # Bottleneck recommendations
        bottlenecks = utilization_analysis.get("bottlenecks", [])
        if bottlenecks:
            recommendations.append(
                f"Bottleneck resources detected: {', '.join(bottlenecks)} - "
                "consider rebalancing workload or adding similar resources"
            )
        
        return recommendations or ["No optimization recommendations at this time"]