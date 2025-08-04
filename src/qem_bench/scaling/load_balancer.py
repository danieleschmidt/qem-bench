"""
Load balancing system for quantum backends and compute resources.

This module provides intelligent load balancing across quantum devices,
queue management, priority scheduling, and capacity monitoring to ensure
optimal resource utilization and performance.
"""

import time
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import heapq

from ..security import SecureConfig
from ..monitoring import MetricsCollector


logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    QUANTUM_AWARE = "quantum_aware"


class BackendStatus(Enum):
    """Quantum backend status."""
    AVAILABLE = "available"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    CALIBRATING = "calibrating"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class QuantumBackend:
    """Represents a quantum backend with its characteristics."""
    id: str
    name: str
    provider: str
    
    # Hardware characteristics
    num_qubits: int
    topology: str  # "linear", "grid", "all_to_all", etc.
    gate_set: List[str] = field(default_factory=list)
    max_shots: int = 8192
    max_circuits: int = 1
    
    # Performance metrics
    gate_fidelity: Dict[str, float] = field(default_factory=dict)
    readout_fidelity: float = 0.95
    t1_times: List[float] = field(default_factory=list)  # T1 for each qubit
    t2_times: List[float] = field(default_factory=list)  # T2 for each qubit
    
    # Current state
    status: BackendStatus = BackendStatus.AVAILABLE
    current_queue_length: int = 0
    estimated_queue_time: float = 0.0  # seconds
    
    # Load balancing metrics
    current_load: float = 0.0  # 0.0 to 1.0
    weight: float = 1.0  # Load balancing weight
    success_rate: float = 1.0
    average_execution_time: float = 60.0  # seconds
    
    # Capacity tracking
    jobs_completed_today: int = 0
    total_shots_executed: int = 0
    last_calibration_time: float = 0.0
    
    def is_available(self) -> bool:
        """Check if backend is available for new jobs."""
        return self.status in [BackendStatus.AVAILABLE, BackendStatus.BUSY]
    
    def utilization(self) -> float:
        """Calculate backend utilization."""
        return self.current_load
    
    def quality_score(self) -> float:
        """Calculate overall quality score."""
        # Combine various quality metrics
        gate_quality = np.mean(list(self.gate_fidelity.values())) if self.gate_fidelity else 0.8
        readout_quality = self.readout_fidelity
        coherence_quality = 1.0  # Simplified - would use T1/T2 times
        
        return (gate_quality + readout_quality + coherence_quality) / 3.0
    
    def efficiency_score(self) -> float:
        """Calculate efficiency score based on load and performance."""
        load_penalty = self.current_load * 0.5
        success_bonus = self.success_rate * 0.3
        speed_bonus = max(0, (120 - self.average_execution_time) / 120) * 0.2
        
        return max(0, 1.0 - load_penalty + success_bonus + speed_bonus)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert backend to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "num_qubits": self.num_qubits,
            "topology": self.topology,
            "gate_set": self.gate_set,
            "max_shots": self.max_shots,
            "max_circuits": self.max_circuits,
            "status": self.status.value,
            "current_queue_length": self.current_queue_length,
            "estimated_queue_time": self.estimated_queue_time,
            "current_load": self.current_load,
            "weight": self.weight,
            "success_rate": self.success_rate,
            "average_execution_time": self.average_execution_time,
            "utilization": self.utilization(),
            "quality_score": self.quality_score(),
            "efficiency_score": self.efficiency_score(),
            "is_available": self.is_available()
        }


@dataclass
class QueuedJob:
    """Represents a job in the queue."""
    id: str
    circuit_id: str
    user_id: str
    priority: JobPriority
    submitted_time: float = field(default_factory=time.time)
    
    # Job requirements
    required_qubits: int = 1
    shots: int = 1024
    estimated_execution_time: float = 60.0
    max_execution_time: Optional[float] = None
    
    # Backend preferences
    preferred_backends: List[str] = field(default_factory=list)
    excluded_backends: List[str] = field(default_factory=list)
    min_fidelity: float = 0.8
    
    # Queue metadata
    assigned_backend: Optional[str] = None
    queue_position: int = 0
    estimated_start_time: Optional[float] = None
    
    def age(self) -> float:
        """Get job age in seconds."""
        return time.time() - self.submitted_time
    
    def priority_score(self) -> float:
        """Calculate priority score for queue ordering."""
        base_score = self.priority.value * 1000
        age_bonus = min(self.age() / 60.0, 500)  # Up to 500 points for age
        return base_score + age_bonus
    
    def meets_backend_requirements(self, backend: QuantumBackend) -> bool:
        """Check if backend meets job requirements."""
        # Check qubit requirements
        if backend.num_qubits < self.required_qubits:
            return False
        
        # Check exclusions
        if backend.id in self.excluded_backends:
            return False
        
        # Check minimum fidelity
        if backend.quality_score() < self.min_fidelity:
            return False
        
        # Check shot limits
        if self.shots > backend.max_shots:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "id": self.id,
            "circuit_id": self.circuit_id,
            "user_id": self.user_id,
            "priority": self.priority.value,
            "submitted_time": self.submitted_time,
            "required_qubits": self.required_qubits,
            "shots": self.shots,
            "estimated_execution_time": self.estimated_execution_time,
            "preferred_backends": self.preferred_backends,
            "excluded_backends": self.excluded_backends,
            "assigned_backend": self.assigned_backend,
            "age": self.age(),
            "priority_score": self.priority_score()
        }


class LoadBalancingAlgorithm(ABC):
    """Abstract base class for load balancing algorithms."""
    
    @abstractmethod
    def select_backend(
        self,
        job: QueuedJob,
        available_backends: List[QuantumBackend]
    ) -> Optional[QuantumBackend]:
        """Select the best backend for a job."""
        pass


class RoundRobinBalancer(LoadBalancingAlgorithm):
    """Round-robin load balancing algorithm."""
    
    def __init__(self):
        self.last_selected_index = 0
    
    def select_backend(
        self,
        job: QueuedJob,
        available_backends: List[QuantumBackend]
    ) -> Optional[QuantumBackend]:
        """Select backend using round-robin."""
        if not available_backends:
            return None
        
        # Filter suitable backends
        suitable_backends = [
            backend for backend in available_backends
            if job.meets_backend_requirements(backend) and backend.is_available()
        ]
        
        if not suitable_backends:
            return None
        
        # Select next backend in round-robin order
        selected_backend = suitable_backends[
            self.last_selected_index % len(suitable_backends)
        ]
        self.last_selected_index += 1
        
        return selected_backend


class LeastConnectionsBalancer(LoadBalancingAlgorithm):
    """Least connections load balancing algorithm."""
    
    def select_backend(
        self,
        job: QueuedJob,
        available_backends: List[QuantumBackend]
    ) -> Optional[QuantumBackend]:
        """Select backend with least current load."""
        suitable_backends = [
            backend for backend in available_backends
            if job.meets_backend_requirements(backend) and backend.is_available()
        ]
        
        if not suitable_backends:
            return None
        
        # Sort by current load (least loaded first)
        suitable_backends.sort(key=lambda b: b.current_load)
        
        return suitable_backends[0]


class WeightedRoundRobinBalancer(LoadBalancingAlgorithm):
    """Weighted round-robin load balancing algorithm."""
    
    def __init__(self):
        self.current_weights: Dict[str, float] = {}
    
    def select_backend(
        self,
        job: QueuedJob,
        available_backends: List[QuantumBackend]
    ) -> Optional[QuantumBackend]:
        """Select backend using weighted round-robin."""
        suitable_backends = [
            backend for backend in available_backends
            if job.meets_backend_requirements(backend) and backend.is_available()
        ]
        
        if not suitable_backends:
            return None
        
        # Initialize current weights if needed
        for backend in suitable_backends:
            if backend.id not in self.current_weights:
                self.current_weights[backend.id] = 0.0
        
        # Find backend with highest current weight
        best_backend = None
        best_weight = -1.0
        
        for backend in suitable_backends:
            current_weight = self.current_weights[backend.id] + backend.weight
            if current_weight > best_weight:
                best_weight = current_weight
                best_backend = backend
        
        if best_backend:
            # Decrease selected backend's current weight
            self.current_weights[best_backend.id] -= 1.0
            
            # Increase all other backend weights
            for backend in suitable_backends:
                if backend.id != best_backend.id:
                    self.current_weights[backend.id] += backend.weight / len(suitable_backends)
        
        return best_backend


class QuantumAwareBalancer(LoadBalancingAlgorithm):
    """Quantum-aware load balancing with fidelity and topology consideration."""
    
    def select_backend(
        self,
        job: QueuedJob,
        available_backends: List[QuantumBackend]
    ) -> Optional[QuantumBackend]:
        """Select backend optimized for quantum characteristics."""
        suitable_backends = [
            backend for backend in available_backends
            if job.meets_backend_requirements(backend) and backend.is_available()
        ]
        
        if not suitable_backends:
            return None
        
        # Score backends based on quantum-specific criteria
        scored_backends = []
        for backend in suitable_backends:
            score = self._calculate_quantum_score(backend, job)
            scored_backends.append((score, backend))
        
        # Sort by score (highest first)
        scored_backends.sort(key=lambda x: x[0], reverse=True)
        
        return scored_backends[0][1]
    
    def _calculate_quantum_score(self, backend: QuantumBackend, job: QueuedJob) -> float:
        """Calculate quantum-specific score for backend-job pair."""
        score = 0.0
        
        # Quality score (40% weight)
        score += backend.quality_score() * 40
        
        # Efficiency score (30% weight)
        score += backend.efficiency_score() * 30
        
        # Queue time penalty (20% weight)
        queue_penalty = min(backend.estimated_queue_time / 300.0, 1.0)  # Max 5 min penalty
        score += (1.0 - queue_penalty) * 20
        
        # Preference bonus (10% weight)
        if backend.id in job.preferred_backends:
            score += 10
        
        return score


class QueueManager:
    """
    Advanced queue management system for quantum computing jobs.
    
    Features:
    - Priority-based job queuing
    - Multi-backend queue optimization
    - Dynamic queue reordering
    - Fair scheduling across users
    - Queue analytics and monitoring
    """
    
    def __init__(self, config: Optional[SecureConfig] = None):
        self.config = config or SecureConfig()
        
        # Queue storage
        self.job_queues: Dict[str, List[QueuedJob]] = {}  # backend_id -> jobs
        self.global_queue: List[QueuedJob] = []
        self.completed_jobs: List[QueuedJob] = []
        
        # Queue management settings
        self.max_queue_length = 1000
        self.max_user_concurrent_jobs = 10
        self.queue_timeout_hours = 24
        
        # Fair scheduling
        self.user_job_counts: Dict[str, int] = {}
        self.user_priority_adjustments: Dict[str, float] = {}
        
        # Statistics
        self.queue_stats = {
            "jobs_queued": 0,
            "jobs_completed": 0,
            "average_queue_time": 0.0,
            "max_queue_length": 0
        }
        
        logger.info("QueueManager initialized")
    
    def add_job(self, job: QueuedJob, backend_id: Optional[str] = None) -> bool:
        """Add a job to the queue."""
        # Check queue capacity
        if len(self.global_queue) >= self.max_queue_length:
            logger.warning(f"Queue is full, rejecting job {job.id}")
            return False
        
        # Check user limits
        user_jobs = self.user_job_counts.get(job.user_id, 0)
        if user_jobs >= self.max_user_concurrent_jobs:
            logger.warning(f"User {job.user_id} has too many concurrent jobs")
            return False
        
        # Add to appropriate queue
        if backend_id:
            # Add to specific backend queue
            if backend_id not in self.job_queues:
                self.job_queues[backend_id] = []
            self.job_queues[backend_id].append(job)
            job.assigned_backend = backend_id
        else:
            # Add to global queue for backend selection
            self.global_queue.append(job)
        
        # Update statistics
        self.user_job_counts[job.user_id] = user_jobs + 1
        self.queue_stats["jobs_queued"] += 1
        self.queue_stats["max_queue_length"] = max(
            self.queue_stats["max_queue_length"],
            len(self.global_queue)
        )
        
        logger.info(f"Job {job.id} added to queue")
        return True
    
    def remove_job(self, job_id: str) -> Optional[QueuedJob]:
        """Remove a job from the queue."""
        # Search in global queue
        for i, job in enumerate(self.global_queue):
            if job.id == job_id:
                removed_job = self.global_queue.pop(i)
                self._update_user_count(removed_job.user_id, -1)
                return removed_job
        
        # Search in backend-specific queues
        for backend_id, queue in self.job_queues.items():
            for i, job in enumerate(queue):
                if job.id == job_id:
                    removed_job = queue.pop(i)
                    self._update_user_count(removed_job.user_id, -1)
                    return removed_job
        
        return None
    
    def get_next_job(self, backend_id: str) -> Optional[QueuedJob]:
        """Get the next job for a specific backend."""
        # First check backend-specific queue
        if backend_id in self.job_queues and self.job_queues[backend_id]:
            backend_queue = self.job_queues[backend_id]
            backend_queue.sort(key=lambda j: j.priority_score(), reverse=True)
            return backend_queue.pop(0)
        
        return None
    
    def get_global_queue_jobs(self, limit: int = 10) -> List[QueuedJob]:
        """Get jobs from global queue sorted by priority."""
        # Sort by priority score
        self.global_queue.sort(key=lambda j: j.priority_score(), reverse=True)
        
        # Apply fair scheduling adjustments
        adjusted_jobs = []
        for job in self.global_queue[:limit]:
            adjusted_score = self._calculate_fair_priority_score(job)
            adjusted_jobs.append((adjusted_score, job))
        
        # Re-sort with fair scheduling
        adjusted_jobs.sort(key=lambda x: x[0], reverse=True)
        
        return [job for _, job in adjusted_jobs]
    
    def _calculate_fair_priority_score(self, job: QueuedJob) -> float:
        """Calculate priority score with fair scheduling adjustments."""
        base_score = job.priority_score()
        
        # Apply user-specific adjustments for fairness
        user_adjustment = self.user_priority_adjustments.get(job.user_id, 0.0)
        
        # Penalize users with many active jobs
        user_jobs = self.user_job_counts.get(job.user_id, 0)
        if user_jobs > 1:
            fairness_penalty = (user_jobs - 1) * 50  # 50 points per extra job
            base_score -= fairness_penalty
        
        return base_score + user_adjustment
    
    def complete_job(self, job_id: str) -> None:
        """Mark a job as completed."""
        # Find and remove job from active queues
        completed_job = self.remove_job(job_id)
        
        if completed_job:
            completed_job.estimated_start_time = time.time()  # Mark completion time
            self.completed_jobs.append(completed_job)
            
            # Update statistics
            self.queue_stats["jobs_completed"] += 1
            
            # Calculate average queue time
            queue_time = completed_job.age()
            current_avg = self.queue_stats["average_queue_time"]
            completed_count = self.queue_stats["jobs_completed"]
            
            new_avg = ((current_avg * (completed_count - 1)) + queue_time) / completed_count
            self.queue_stats["average_queue_time"] = new_avg
            
            logger.info(f"Job {job_id} completed after {queue_time:.1f}s in queue")
    
    def _update_user_count(self, user_id: str, delta: int) -> None:
        """Update user job count."""
        current_count = self.user_job_counts.get(user_id, 0)
        new_count = max(0, current_count + delta)
        
        if new_count == 0:
            self.user_job_counts.pop(user_id, None)
        else:
            self.user_job_counts[user_id] = new_count
    
    def cleanup_expired_jobs(self) -> int:
        """Remove expired jobs from queues."""
        current_time = time.time()
        timeout_seconds = self.queue_timeout_hours * 3600
        
        expired_count = 0
        
        # Clean global queue
        self.global_queue = [
            job for job in self.global_queue
            if current_time - job.submitted_time < timeout_seconds
        ]
        
        # Clean backend-specific queues
        for backend_id in self.job_queues:
            original_length = len(self.job_queues[backend_id])
            self.job_queues[backend_id] = [
                job for job in self.job_queues[backend_id]
                if current_time - job.submitted_time < timeout_seconds
            ]
            expired_count += original_length - len(self.job_queues[backend_id])
        
        if expired_count > 0:
            logger.info(f"Removed {expired_count} expired jobs from queues")
        
        return expired_count
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get comprehensive queue status."""
        total_queued = len(self.global_queue) + sum(
            len(queue) for queue in self.job_queues.values()
        )
        
        # Priority distribution
        priority_distribution = {}
        all_jobs = self.global_queue + [
            job for queue in self.job_queues.values() for job in queue
        ]
        
        for priority in JobPriority:
            priority_distribution[priority.name] = len([
                job for job in all_jobs if job.priority == priority
            ])
        
        # User distribution
        user_distribution = {}
        for job in all_jobs:
            user_distribution[job.user_id] = user_distribution.get(job.user_id, 0) + 1
        
        return {
            "total_queued_jobs": total_queued,
            "global_queue_length": len(self.global_queue),
            "backend_queue_lengths": {
                backend_id: len(queue) 
                for backend_id, queue in self.job_queues.items()
            },
            "priority_distribution": priority_distribution,
            "user_distribution": user_distribution,
            "queue_stats": self.queue_stats.copy(),
            "oldest_job_age": max([job.age() for job in all_jobs]) if all_jobs else 0
        }


class PriorityScheduler:
    """
    Priority-based job scheduler with advanced features.
    
    Features:
    - Multi-level priority queues
    - Dynamic priority adjustment
    - Resource-aware scheduling
    - Deadline scheduling
    - Fair share scheduling
    """
    
    def __init__(self, config: Optional[SecureConfig] = None):
        self.config = config or SecureConfig()
        
        # Priority queues for each priority level
        self.priority_queues: Dict[JobPriority, List[QueuedJob]] = {
            priority: [] for priority in JobPriority
        }
        
        # Scheduling policies
        self.time_slice_ms = 1000  # Time slice for round-robin within priority
        self.priority_aging_enabled = True
        self.aging_threshold_minutes = 30
        
        # Fair share tracking
        self.user_share_allocations: Dict[str, float] = {}  # user_id -> share (0-1)
        self.user_usage_tracking: Dict[str, float] = {}    # user_id -> recent usage
        
        logger.info("PriorityScheduler initialized")
    
    def schedule_job(self, job: QueuedJob) -> None:
        """Add a job to the appropriate priority queue."""
        # Apply priority aging if enabled
        if self.priority_aging_enabled:
            job = self._apply_priority_aging(job)
        
        # Add to priority queue
        self.priority_queues[job.priority].append(job)
        
        # Sort queue by priority score
        self.priority_queues[job.priority].sort(
            key=lambda j: j.priority_score(), 
            reverse=True
        )
        
        logger.debug(f"Job {job.id} scheduled with priority {job.priority.name}")
    
    def get_next_job(self, backend: QuantumBackend) -> Optional[QueuedJob]:
        """Get the next job to execute based on priority and backend suitability."""
        # Check each priority level from highest to lowest
        for priority in sorted(JobPriority, key=lambda p: p.value, reverse=True):
            queue = self.priority_queues[priority]
            
            # Find first suitable job in this priority level
            for i, job in enumerate(queue):
                if job.meets_backend_requirements(backend):
                    # Apply fair share considerations
                    if self._check_fair_share(job):
                        # Remove job from queue
                        selected_job = queue.pop(i)
                        self._update_user_usage(selected_job.user_id)
                        return selected_job
        
        return None
    
    def _apply_priority_aging(self, job: QueuedJob) -> QueuedJob:
        """Apply priority aging to prevent starvation."""
        job_age_minutes = job.age() / 60.0
        
        if (job_age_minutes > self.aging_threshold_minutes and 
            job.priority != JobPriority.CRITICAL):
            
            # Increase priority for old jobs
            if job.priority == JobPriority.LOW:
                job.priority = JobPriority.NORMAL
            elif job.priority == JobPriority.NORMAL:
                job.priority = JobPriority.HIGH
            elif job.priority == JobPriority.HIGH:
                job.priority = JobPriority.URGENT
            
            logger.info(f"Job {job.id} priority increased due to aging")
        
        return job
    
    def _check_fair_share(self, job: QueuedJob) -> bool:
        """Check if job execution respects fair share allocations."""
        user_id = job.user_id
        
        # If no fair share configured, allow execution
        if user_id not in self.user_share_allocations:
            return True
        
        allocated_share = self.user_share_allocations[user_id]
        current_usage = self.user_usage_tracking.get(user_id, 0.0)
        
        # Allow execution if user hasn't exceeded their fair share
        return current_usage <= allocated_share
    
    def _update_user_usage(self, user_id: str) -> None:
        """Update user usage tracking."""
        current_usage = self.user_usage_tracking.get(user_id, 0.0)
        # Simple increment - in practice would be time-weighted
        self.user_usage_tracking[user_id] = current_usage + 1
    
    def set_user_share(self, user_id: str, share: float) -> None:
        """Set fair share allocation for a user."""
        if 0 <= share <= 1:
            self.user_share_allocations[user_id] = share
            logger.info(f"Set fair share for user {user_id}: {share:.2%}")
        else:
            raise ValueError("Share must be between 0 and 1")
    
    def get_scheduling_stats(self) -> Dict[str, Any]:
        """Get scheduling statistics."""
        total_jobs = sum(len(queue) for queue in self.priority_queues.values())
        
        priority_counts = {
            priority.name: len(queue) 
            for priority, queue in self.priority_queues.items()
        }
        
        return {
            "total_queued_jobs": total_jobs,
            "priority_distribution": priority_counts,
            "fair_share_users": len(self.user_share_allocations),
            "active_users": len(self.user_usage_tracking),
            "priority_aging_enabled": self.priority_aging_enabled
        }


class CapacityMonitor:
    """
    Monitor and track backend capacity and performance metrics.
    
    Features:
    - Real-time capacity monitoring
    - Performance trend analysis
    - Predictive capacity planning
    - Alert generation for capacity issues
    - Historical capacity analytics
    """
    
    def __init__(self, config: Optional[SecureConfig] = None):
        self.config = config or SecureConfig()
        
        # Monitoring data
        self.capacity_history: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        
        # Monitoring configuration
        self.monitoring_interval = 60.0  # seconds
        self.history_retention_hours = 72  # 3 days
        self.is_monitoring = False
        
        # Alert thresholds
        self.capacity_warning_threshold = 0.8  # 80%
        self.capacity_critical_threshold = 0.95  # 95%
        self.performance_degradation_threshold = 0.2  # 20% degradation
        
        logger.info("CapacityMonitor initialized")
    
    async def start_monitoring(self, backends: Dict[str, QuantumBackend]) -> None:
        """Start capacity monitoring for given backends."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.backends = backends
        
        logger.info("Starting capacity monitoring")
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop capacity monitoring."""
        self.is_monitoring = False
        logger.info("Capacity monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                for backend_id, backend in self.backends.items():
                    await self._collect_backend_metrics(backend_id, backend)
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Check for alerts
                await self._check_capacity_alerts()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in capacity monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_backend_metrics(
        self, 
        backend_id: str, 
        backend: QuantumBackend
    ) -> None:
        """Collect metrics for a specific backend."""
        current_time = time.time()
        
        # Collect capacity metrics
        metrics = {
            "utilization": backend.utilization(),
            "queue_length": backend.current_queue_length,
            "estimated_queue_time": backend.estimated_queue_time,
            "success_rate": backend.success_rate,
            "average_execution_time": backend.average_execution_time,
            "quality_score": backend.quality_score(),
            "efficiency_score": backend.efficiency_score(),
            "jobs_completed_today": backend.jobs_completed_today,
            "total_shots_executed": backend.total_shots_executed
        }
        
        # Store metrics
        if backend_id not in self.capacity_history:
            self.capacity_history[backend_id] = []
        
        self.capacity_history[backend_id].append((current_time, metrics))
        
        # Update performance metrics
        self.performance_metrics[backend_id] = metrics
    
    def _cleanup_old_data(self) -> None:
        """Remove old monitoring data."""
        cutoff_time = time.time() - self.history_retention_hours * 3600
        
        for backend_id in self.capacity_history:
            self.capacity_history[backend_id] = [
                (timestamp, metrics) 
                for timestamp, metrics in self.capacity_history[backend_id]
                if timestamp > cutoff_time
            ]
    
    async def _check_capacity_alerts(self) -> None:
        """Check for capacity-related alerts."""
        for backend_id, metrics in self.performance_metrics.items():
            utilization = metrics.get("utilization", 0.0)
            
            if utilization >= self.capacity_critical_threshold:
                await self._send_alert(
                    backend_id, 
                    "critical", 
                    f"Backend utilization critical: {utilization:.1%}"
                )
            elif utilization >= self.capacity_warning_threshold:
                await self._send_alert(
                    backend_id,
                    "warning",
                    f"Backend utilization high: {utilization:.1%}"
                )
            
            # Check for performance degradation
            degradation = self._calculate_performance_degradation(backend_id)
            if degradation > self.performance_degradation_threshold:
                await self._send_alert(
                    backend_id,
                    "warning",
                    f"Performance degraded by {degradation:.1%}"
                )
    
    def _calculate_performance_degradation(self, backend_id: str) -> float:
        """Calculate performance degradation over time."""
        if backend_id not in self.capacity_history:
            return 0.0
        
        history = self.capacity_history[backend_id]
        if len(history) < 10:  # Need enough data
            return 0.0
        
        # Compare recent performance to baseline
        recent_metrics = [metrics for _, metrics in history[-5:]]  # Last 5 samples
        baseline_metrics = [metrics for _, metrics in history[-20:-10]]  # Earlier samples
        
        if not baseline_metrics:
            return 0.0
        
        # Calculate average success rates
        recent_success_rate = np.mean([m.get("success_rate", 1.0) for m in recent_metrics])
        baseline_success_rate = np.mean([m.get("success_rate", 1.0) for m in baseline_metrics])
        
        if baseline_success_rate > 0:
            degradation = (baseline_success_rate - recent_success_rate) / baseline_success_rate
            return max(0, degradation)
        
        return 0.0
    
    async def _send_alert(self, backend_id: str, severity: str, message: str) -> None:
        """Send capacity alert."""
        alert = {
            "timestamp": time.time(),
            "backend_id": backend_id,
            "severity": severity,
            "message": message
        }
        
        logger.log(
            logging.CRITICAL if severity == "critical" else logging.WARNING,
            f"Capacity alert for {backend_id}: {message}"
        )
        
        # In practice, this would integrate with alerting systems
    
    def get_capacity_report(self, backend_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get capacity report for a backend."""
        if backend_id not in self.capacity_history:
            return {"error": "No data available for backend"}
        
        cutoff_time = time.time() - hours * 3600
        recent_data = [
            (timestamp, metrics) 
            for timestamp, metrics in self.capacity_history[backend_id]
            if timestamp > cutoff_time
        ]
        
        if not recent_data:
            return {"error": "No recent data available"}
        
        # Calculate statistics
        utilizations = [metrics["utilization"] for _, metrics in recent_data]
        queue_lengths = [metrics["queue_length"] for _, metrics in recent_data]
        success_rates = [metrics["success_rate"] for _, metrics in recent_data]
        
        return {
            "backend_id": backend_id,
            "report_period_hours": hours,
            "data_points": len(recent_data),
            "utilization": {
                "average": np.mean(utilizations),
                "max": np.max(utilizations),
                "min": np.min(utilizations),
                "std": np.std(utilizations)
            },
            "queue_performance": {
                "average_length": np.mean(queue_lengths),
                "max_length": np.max(queue_lengths),
                "peak_times": self._find_peak_times(recent_data)
            },
            "success_rate": {
                "average": np.mean(success_rates),
                "trend": self._calculate_trend(success_rates)
            },
            "current_metrics": self.performance_metrics.get(backend_id, {})
        }
    
    def _find_peak_times(self, data: List[Tuple[float, Dict[str, Any]]]) -> List[str]:
        """Find peak usage times."""
        # Simple implementation - find hours with highest average utilization
        hourly_utilization = {}
        
        for timestamp, metrics in data:
            hour = datetime.fromtimestamp(timestamp).hour
            if hour not in hourly_utilization:
                hourly_utilization[hour] = []
            hourly_utilization[hour].append(metrics["utilization"])
        
        # Calculate average utilization per hour
        avg_hourly = {
            hour: np.mean(utilizations) 
            for hour, utilizations in hourly_utilization.items()
        }
        
        # Find top 3 peak hours
        peak_hours = sorted(avg_hourly.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return [f"{hour:02d}:00" for hour, _ in peak_hours]
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        else:
            return "stable"


class BackendBalancer:
    """
    Main backend load balancer that coordinates all balancing components.
    
    This class orchestrates queue management, priority scheduling, capacity
    monitoring, and load balancing algorithms to optimize quantum backend
    utilization and job execution performance.
    
    Example:
        >>> balancer = BackendBalancer(
        ...     strategy=LoadBalancingStrategy.QUANTUM_AWARE
        ... )
        >>> await balancer.start()
        >>> 
        >>> # Add backends
        >>> backend = QuantumBackend("ibm_quantum", "IBM Quantum", "IBM", num_qubits=5)
        >>> balancer.add_backend(backend)
        >>> 
        >>> # Submit job
        >>> job = QueuedJob("job1", "circuit1", "user1", JobPriority.HIGH)
        >>> result = await balancer.submit_job(job)
    """
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.QUANTUM_AWARE,
        config: Optional[SecureConfig] = None
    ):
        self.config = config or SecureConfig()
        self.strategy = strategy
        
        # Core components
        self.queue_manager = QueueManager(config)
        self.priority_scheduler = PriorityScheduler(config)
        self.capacity_monitor = CapacityMonitor(config)
        
        # Load balancing algorithms
        self.balancing_algorithms = {
            LoadBalancingStrategy.ROUND_ROBIN: RoundRobinBalancer(),
            LoadBalancingStrategy.LEAST_CONNECTIONS: LeastConnectionsBalancer(),
            LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN: WeightedRoundRobinBalancer(),
            LoadBalancingStrategy.QUANTUM_AWARE: QuantumAwareBalancer()
        }
        
        # Backend management
        self.backends: Dict[str, QuantumBackend] = {}
        
        # Balancer state
        self.is_running = False
        self.balancing_interval = 10.0  # seconds
        
        # Statistics
        self.balancing_stats = {
            "jobs_balanced": 0,
            "backend_assignments": {},
            "strategy_switches": 0
        }
        
        logger.info(f"BackendBalancer initialized with strategy: {strategy.value}")
    
    async def start(self) -> None:
        """Start the backend balancer."""
        if self.is_running:
            logger.warning("BackendBalancer is already running")
            return
        
        self.is_running = True
        logger.info("Starting BackendBalancer")
        
        # Start capacity monitoring
        await self.capacity_monitor.start_monitoring(self.backends)
        
        # Start balancing loop
        asyncio.create_task(self._balancing_loop())
    
    async def stop(self) -> None:
        """Stop the backend balancer."""
        self.is_running = False
        
        # Stop capacity monitoring
        await self.capacity_monitor.stop_monitoring()
        
        logger.info("BackendBalancer stopped")
    
    def add_backend(self, backend: QuantumBackend) -> None:
        """Add a quantum backend to the load balancer."""
        self.backends[backend.id] = backend
        self.balancing_stats["backend_assignments"][backend.id] = 0
        logger.info(f"Added backend: {backend.name} ({backend.id})")
    
    def remove_backend(self, backend_id: str) -> None:
        """Remove a quantum backend from the load balancer."""
        if backend_id in self.backends:
            del self.backends[backend_id]
            logger.info(f"Removed backend: {backend_id}")
    
    async def submit_job(self, job: QueuedJob) -> Dict[str, Any]:
        """Submit a job for load-balanced execution."""
        # Add job to queue manager
        success = self.queue_manager.add_job(job)
        if not success:
            return {"error": "Failed to add job to queue"}
        
        # Schedule job with priority scheduler
        self.priority_scheduler.schedule_job(job)
        
        logger.info(f"Job {job.id} submitted for load balancing")
        
        return {
            "job_id": job.id,
            "status": "queued",
            "estimated_position": self._estimate_queue_position(job),
            "estimated_wait_time": self._estimate_wait_time(job)
        }
    
    async def _balancing_loop(self) -> None:
        """Main load balancing loop."""
        while self.is_running:
            try:
                await self._process_job_assignments()
                await self._update_backend_metrics()
                await self._optimize_load_distribution()
                
                await asyncio.sleep(self.balancing_interval)
                
            except Exception as e:
                logger.error(f"Error in load balancing loop: {e}")
                await asyncio.sleep(self.balancing_interval)
    
    async def _process_job_assignments(self) -> None:
        """Process job assignments to backends."""
        # Get available backends
        available_backends = [
            backend for backend in self.backends.values()
            if backend.is_available()
        ]
        
        if not available_backends:
            return
        
        # Process jobs from priority scheduler
        balancer = self.balancing_algorithms[self.strategy]
        
        for backend in available_backends:
            if backend.current_load >= 1.0:
                continue
            
            # Get next job for this backend
            job = self.priority_scheduler.get_next_job(backend)
            
            if job:
                # Assign job to backend
                await self._assign_job_to_backend(job, backend)
    
    async def _assign_job_to_backend(self, job: QueuedJob, backend: QuantumBackend) -> None:
        """Assign a job to a specific backend."""
        job.assigned_backend = backend.id
        job.estimated_start_time = time.time()
        
        # Update backend load
        backend.current_load = min(1.0, backend.current_load + 0.1)
        backend.current_queue_length += 1
        
        # Update statistics
        self.balancing_stats["jobs_balanced"] += 1
        self.balancing_stats["backend_assignments"][backend.id] += 1
        
        logger.info(f"Assigned job {job.id} to backend {backend.id}")
        
        # In practice, this would trigger actual job execution
    
    async def _update_backend_metrics(self) -> None:
        """Update backend performance metrics."""
        for backend in self.backends.values():
            # Simulate metric updates (in practice, would query real backends)
            # Update queue times, success rates, etc.
            
            # Simple simulation of load decay
            backend.current_load = max(0.0, backend.current_load - 0.05)
            
            if backend.current_load < 0.5:
                backend.status = BackendStatus.AVAILABLE
            elif backend.current_load < 0.9:
                backend.status = BackendStatus.BUSY
            else:
                backend.status = BackendStatus.OVERLOADED
    
    async def _optimize_load_distribution(self) -> None:
        """Optimize load distribution across backends."""
        if len(self.backends) < 2:
            return
        
        # Calculate load imbalance
        loads = [backend.current_load for backend in self.backends.values()]
        load_std = np.std(loads)
        
        # If load is imbalanced, consider strategy adjustment
        if load_std > 0.3:  # High imbalance
            await self._consider_strategy_change()
    
    async def _consider_strategy_change(self) -> None:
        """Consider changing load balancing strategy based on performance."""
        # Simple strategy adaptation logic
        current_performance = self._calculate_system_performance()
        
        if current_performance < 0.7:  # Poor performance
            # Try a different strategy
            strategies = list(LoadBalancingStrategy)
            current_index = strategies.index(self.strategy)
            next_strategy = strategies[(current_index + 1) % len(strategies)]
            
            logger.info(f"Switching load balancing strategy to {next_strategy.value}")
            self.strategy = next_strategy
            self.balancing_stats["strategy_switches"] += 1
    
    def _calculate_system_performance(self) -> float:
        """Calculate overall system performance score."""
        if not self.backends:
            return 0.0
        
        # Simple performance calculation based on backend efficiency
        total_efficiency = sum(
            backend.efficiency_score() for backend in self.backends.values()
        )
        
        return total_efficiency / len(self.backends)
    
    def _estimate_queue_position(self, job: QueuedJob) -> int:
        """Estimate queue position for a job."""
        # Count jobs with higher priority
        higher_priority_jobs = 0
        
        for priority in JobPriority:
            if priority.value > job.priority.value:
                higher_priority_jobs += len(self.priority_scheduler.priority_queues[priority])
        
        return higher_priority_jobs + 1
    
    def _estimate_wait_time(self, job: QueuedJob) -> float:
        """Estimate wait time for a job."""
        queue_position = self._estimate_queue_position(job)
        
        # Estimate based on average job execution time and available backends
        avg_execution_time = np.mean([
            backend.average_execution_time for backend in self.backends.values()
        ]) if self.backends else 60.0
        
        available_backends = len([
            backend for backend in self.backends.values()
            if backend.is_available()
        ])
        
        if available_backends > 0:
            estimated_wait = (queue_position * avg_execution_time) / available_backends
        else:
            estimated_wait = queue_position * avg_execution_time
        
        return estimated_wait
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        queue_status = self.queue_manager.get_queue_status()
        scheduler_stats = self.priority_scheduler.get_scheduling_stats()
        
        # Backend status summary
        backend_summary = {
            "total_backends": len(self.backends),
            "available_backends": len([
                b for b in self.backends.values() if b.is_available()
            ]),
            "average_load": np.mean([
                b.current_load for b in self.backends.values()
            ]) if self.backends else 0.0,
            "backends": [backend.to_dict() for backend in self.backends.values()]
        }
        
        return {
            "balancer_running": self.is_running,
            "current_strategy": self.strategy.value,
            "balancing_stats": self.balancing_stats.copy(),
            "queue_status": queue_status,
            "scheduler_stats": scheduler_stats,
            "backend_summary": backend_summary,
            "system_performance": self._calculate_system_performance(),
            "timestamp": time.time()
        }
    
    async def rebalance_jobs(self) -> Dict[str, Any]:
        """Manually trigger job rebalancing."""
        logger.info("Manual job rebalancing triggered")
        
        rebalanced_jobs = 0
        
        # Get jobs from global queue
        jobs = self.queue_manager.get_global_queue_jobs(limit=50)
        
        # Reassign jobs using current strategy
        balancer = self.balancing_algorithms[self.strategy]
        available_backends = [
            backend for backend in self.backends.values()
            if backend.is_available()
        ]
        
        for job in jobs:
            if job.assigned_backend:
                continue  # Already assigned
            
            # Select best backend
            selected_backend = balancer.select_backend(job, available_backends)
            
            if selected_backend:
                await self._assign_job_to_backend(job, selected_backend)
                rebalanced_jobs += 1
        
        return {
            "rebalanced_jobs": rebalanced_jobs,
            "total_jobs_processed": len(jobs),
            "timestamp": time.time()
        }