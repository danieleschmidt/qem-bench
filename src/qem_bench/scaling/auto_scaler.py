"""
Auto-scaling framework for dynamic resource management in quantum error mitigation.

This module provides intelligent auto-scaling capabilities that automatically
adapt compute resources based on workload demands, cost constraints, and
performance requirements.
"""

import time
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import numpy as np

from ..monitoring import SystemMonitor, PerformanceMonitor
from ..security import SecureConfig, AccessControl


logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Direction of scaling operations."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingTrigger(Enum):
    """Triggers for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_LENGTH = "queue_length"
    LATENCY = "latency"
    COST_THRESHOLD = "cost_threshold"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    queue_length: int = 0
    average_latency: float = 0.0
    error_rate: float = 0.0
    cost_per_hour: float = 0.0
    throughput: float = 0.0
    active_backends: int = 0
    pending_jobs: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "queue_length": self.queue_length,
            "average_latency": self.average_latency,
            "error_rate": self.error_rate,
            "cost_per_hour": self.cost_per_hour,
            "throughput": self.throughput,
            "active_backends": self.active_backends,
            "pending_jobs": self.pending_jobs,
            "timestamp": self.timestamp
        }


@dataclass
class ScalingPolicy:
    """Policy configuration for auto-scaling behavior."""
    # Scale-up thresholds
    cpu_scale_up_threshold: float = 80.0
    memory_scale_up_threshold: float = 80.0
    queue_scale_up_threshold: int = 10
    latency_scale_up_threshold: float = 5.0  # seconds
    
    # Scale-down thresholds
    cpu_scale_down_threshold: float = 20.0
    memory_scale_down_threshold: float = 20.0
    queue_scale_down_threshold: int = 2
    latency_scale_down_threshold: float = 1.0  # seconds
    
    # Scaling constraints
    min_instances: int = 1
    max_instances: int = 50
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes
    
    # Cost constraints
    max_cost_per_hour: Optional[float] = None
    cost_optimization_enabled: bool = True
    
    # Advanced settings
    prediction_window: float = 900.0  # 15 minutes
    evaluation_period: float = 60.0  # 1 minute
    warmup_time: float = 180.0  # 3 minutes
    
    def __post_init__(self):
        """Validate policy configuration."""
        if self.min_instances < 1:
            raise ValueError("min_instances must be at least 1")
        if self.max_instances < self.min_instances:
            raise ValueError("max_instances must be >= min_instances")
        if self.scale_up_cooldown < 0 or self.scale_down_cooldown < 0:
            raise ValueError("Cooldown periods must be non-negative")


class ScalingDecision:
    """Represents a scaling decision with rationale."""
    
    def __init__(
        self,
        direction: ScalingDirection,
        target_instances: int,
        current_instances: int,
        triggers: List[ScalingTrigger],
        confidence: float,
        estimated_cost_impact: float,
        rationale: str
    ):
        self.direction = direction
        self.target_instances = target_instances
        self.current_instances = current_instances
        self.triggers = triggers
        self.confidence = confidence
        self.estimated_cost_impact = estimated_cost_impact
        self.rationale = rationale
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary."""
        return {
            "direction": self.direction.value,
            "target_instances": self.target_instances,
            "current_instances": self.current_instances,
            "triggers": [t.value for t in self.triggers],
            "confidence": self.confidence,
            "estimated_cost_impact": self.estimated_cost_impact,
            "rationale": self.rationale,
            "timestamp": self.timestamp
        }


class MetricsCollector(ABC):
    """Abstract base class for collecting scaling metrics."""
    
    @abstractmethod
    async def collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        pass


class DefaultMetricsCollector(MetricsCollector):
    """Default implementation using system monitoring."""
    
    def __init__(
        self,
        system_monitor: Optional[SystemMonitor] = None,
        performance_monitor: Optional[PerformanceMonitor] = None
    ):
        self.system_monitor = system_monitor or SystemMonitor()
        self.performance_monitor = performance_monitor or PerformanceMonitor()
    
    async def collect_metrics(self) -> ScalingMetrics:
        """Collect metrics from system monitors."""
        try:
            # Get system metrics
            system_stats = self.system_monitor.get_system_stats()
            perf_stats = self.performance_monitor.get_performance_stats()
            
            return ScalingMetrics(
                cpu_utilization=system_stats.get("cpu_percent", 0.0),
                memory_utilization=system_stats.get("memory_percent", 0.0),
                queue_length=perf_stats.get("queue_length", 0),
                average_latency=perf_stats.get("average_latency", 0.0),
                error_rate=perf_stats.get("error_rate", 0.0),
                throughput=perf_stats.get("throughput", 0.0),
                active_backends=perf_stats.get("active_backends", 0),
                pending_jobs=perf_stats.get("pending_jobs", 0)
            )
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return ScalingMetrics()


class ResourceProvider(ABC):
    """Abstract interface for resource provisioning."""
    
    @abstractmethod
    async def scale_to(self, target_instances: int) -> bool:
        """Scale resources to target number of instances."""
        pass
    
    @abstractmethod
    async def get_current_instances(self) -> int:
        """Get current number of active instances."""
        pass
    
    @abstractmethod
    async def get_scaling_cost(self, from_instances: int, to_instances: int) -> float:
        """Estimate cost impact of scaling operation."""
        pass


class AutoScaler:
    """
    Intelligent auto-scaling system for quantum error mitigation workloads.
    
    Features:
    - Predictive scaling based on workload analysis
    - Cost-aware scaling decisions
    - Multiple scaling triggers and policies
    - Fault-tolerant operation with rollback capabilities
    - Integration with cloud providers and resource managers
    
    Example:
        >>> policy = ScalingPolicy(
        ...     min_instances=2,
        ...     max_instances=20,
        ...     cpu_scale_up_threshold=75.0
        ... )
        >>> scaler = AutoScaler(policy=policy)
        >>> await scaler.start()
    """
    
    def __init__(
        self,
        policy: ScalingPolicy,
        metrics_collector: Optional[MetricsCollector] = None,
        resource_provider: Optional[ResourceProvider] = None,
        config: Optional[SecureConfig] = None
    ):
        self.policy = policy
        self.metrics_collector = metrics_collector or DefaultMetricsCollector()
        self.resource_provider = resource_provider
        self.config = config or SecureConfig()
        
        # Scaling state
        self.current_instances = policy.min_instances
        self.last_scale_up_time = 0.0
        self.last_scale_down_time = 0.0
        self.is_running = False
        self.metrics_history: List[ScalingMetrics] = []
        self.scaling_history: List[ScalingDecision] = []
        
        # Callbacks for scaling events
        self.on_scale_up: List[Callable] = []
        self.on_scale_down: List[Callable] = []
        self.on_scaling_decision: List[Callable] = []
        
        logger.info(f"AutoScaler initialized with policy: {policy}")
    
    async def start(self) -> None:
        """Start the auto-scaling loop."""
        if self.is_running:
            logger.warning("AutoScaler is already running")
            return
        
        self.is_running = True
        logger.info("Starting AutoScaler")
        
        # Start metrics collection loop
        asyncio.create_task(self._metrics_collection_loop())
        
        # Start scaling decision loop
        asyncio.create_task(self._scaling_decision_loop())
    
    async def stop(self) -> None:
        """Stop the auto-scaling system."""
        self.is_running = False
        logger.info("AutoScaler stopped")
    
    async def _metrics_collection_loop(self) -> None:
        """Continuously collect system metrics."""
        while self.is_running:
            try:
                metrics = await self.metrics_collector.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics (last hour)
                cutoff_time = time.time() - 3600
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
                await asyncio.sleep(self.policy.evaluation_period)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.policy.evaluation_period)
    
    async def _scaling_decision_loop(self) -> None:
        """Main scaling decision loop."""
        while self.is_running:
            try:
                if len(self.metrics_history) < 2:
                    await asyncio.sleep(self.policy.evaluation_period)
                    continue
                
                # Make scaling decision
                decision = await self._make_scaling_decision()
                
                if decision.direction != ScalingDirection.STABLE:
                    logger.info(f"Scaling decision: {decision.rationale}")
                    
                    # Execute scaling if needed
                    await self._execute_scaling_decision(decision)
                    
                    # Record decision
                    self.scaling_history.append(decision)
                    
                    # Notify callbacks
                    for callback in self.on_scaling_decision:
                        try:
                            await callback(decision)
                        except Exception as e:
                            logger.error(f"Error in scaling callback: {e}")
                
                await asyncio.sleep(self.policy.evaluation_period)
                
            except Exception as e:
                logger.error(f"Error in scaling decision loop: {e}")
                await asyncio.sleep(self.policy.evaluation_period)
    
    async def _make_scaling_decision(self) -> ScalingDecision:
        """Analyze metrics and make scaling decision."""
        current_metrics = self.metrics_history[-1]
        current_instances = await self._get_current_instances()
        
        # Check cooldown periods
        now = time.time()
        scale_up_allowed = (now - self.last_scale_up_time) > self.policy.scale_up_cooldown
        scale_down_allowed = (now - self.last_scale_down_time) > self.policy.scale_down_cooldown
        
        # Evaluate scale-up triggers
        scale_up_triggers = []
        if (current_metrics.cpu_utilization > self.policy.cpu_scale_up_threshold and
            scale_up_allowed):
            scale_up_triggers.append(ScalingTrigger.CPU_UTILIZATION)
        
        if (current_metrics.memory_utilization > self.policy.memory_scale_up_threshold and
            scale_up_allowed):
            scale_up_triggers.append(ScalingTrigger.MEMORY_UTILIZATION)
        
        if (current_metrics.queue_length > self.policy.queue_scale_up_threshold and
            scale_up_allowed):
            scale_up_triggers.append(ScalingTrigger.QUEUE_LENGTH)
        
        if (current_metrics.average_latency > self.policy.latency_scale_up_threshold and
            scale_up_allowed):
            scale_up_triggers.append(ScalingTrigger.LATENCY)
        
        # Evaluate scale-down triggers
        scale_down_triggers = []
        if (current_metrics.cpu_utilization < self.policy.cpu_scale_down_threshold and
            current_metrics.memory_utilization < self.policy.memory_scale_down_threshold and
            current_metrics.queue_length < self.policy.queue_scale_down_threshold and
            scale_down_allowed):
            scale_down_triggers.append(ScalingTrigger.CPU_UTILIZATION)
        
        # Determine scaling action
        if scale_up_triggers and current_instances < self.policy.max_instances:
            target_instances = min(
                current_instances + self._calculate_scale_up_amount(current_metrics),
                self.policy.max_instances
            )
            
            # Check cost constraints
            if self.policy.max_cost_per_hour:
                estimated_cost = await self._estimate_scaling_cost(
                    current_instances, target_instances
                )
                if estimated_cost > self.policy.max_cost_per_hour:
                    # Reduce scaling to stay within budget
                    target_instances = await self._find_max_instances_within_budget(
                        current_instances
                    )
            
            confidence = self._calculate_confidence(scale_up_triggers, current_metrics)
            estimated_cost = await self._estimate_scaling_cost(
                current_instances, target_instances
            )
            
            return ScalingDecision(
                direction=ScalingDirection.UP,
                target_instances=target_instances,
                current_instances=current_instances,
                triggers=scale_up_triggers,
                confidence=confidence,
                estimated_cost_impact=estimated_cost,
                rationale=f"Scale up: {', '.join([t.value for t in scale_up_triggers])}"
            )
        
        elif scale_down_triggers and current_instances > self.policy.min_instances:
            target_instances = max(
                current_instances - self._calculate_scale_down_amount(current_metrics),
                self.policy.min_instances
            )
            
            confidence = self._calculate_confidence(scale_down_triggers, current_metrics)
            estimated_cost = await self._estimate_scaling_cost(
                current_instances, target_instances
            )
            
            return ScalingDecision(
                direction=ScalingDirection.DOWN,
                target_instances=target_instances,
                current_instances=current_instances,
                triggers=scale_down_triggers,
                confidence=confidence,
                estimated_cost_impact=estimated_cost,
                rationale=f"Scale down: low resource utilization"
            )
        
        else:
            return ScalingDecision(
                direction=ScalingDirection.STABLE,
                target_instances=current_instances,
                current_instances=current_instances,
                triggers=[],
                confidence=1.0,
                estimated_cost_impact=0.0,
                rationale="No scaling needed"
            )
    
    def _calculate_scale_up_amount(self, metrics: ScalingMetrics) -> int:
        """Calculate number of instances to add during scale-up."""
        # Simple algorithm: scale based on highest utilization
        max_utilization = max(
            metrics.cpu_utilization,
            metrics.memory_utilization,
            min(metrics.queue_length * 10, 100)  # Queue length as percentage
        )
        
        if max_utilization > 90:
            return max(2, int(self.current_instances * 0.5))  # 50% increase
        elif max_utilization > 80:
            return max(1, int(self.current_instances * 0.3))  # 30% increase
        else:
            return 1  # Add one instance
    
    def _calculate_scale_down_amount(self, metrics: ScalingMetrics) -> int:
        """Calculate number of instances to remove during scale-down."""
        # Conservative scale-down: remove 1-2 instances at a time
        avg_utilization = (metrics.cpu_utilization + metrics.memory_utilization) / 2
        
        if avg_utilization < 10:
            return max(1, int(self.current_instances * 0.2))  # Remove up to 20%
        else:
            return 1  # Remove one instance
    
    def _calculate_confidence(
        self, 
        triggers: List[ScalingTrigger], 
        metrics: ScalingMetrics
    ) -> float:
        """Calculate confidence score for scaling decision."""
        if not triggers:
            return 1.0
        
        # Base confidence on number of triggers and historical data
        base_confidence = min(len(triggers) * 0.3, 1.0)
        
        # Adjust based on metric consistency over time
        if len(self.metrics_history) >= 3:
            recent_metrics = self.metrics_history[-3:]
            consistency_bonus = self._calculate_metric_consistency(recent_metrics)
            base_confidence = min(base_confidence + consistency_bonus, 1.0)
        
        return base_confidence
    
    def _calculate_metric_consistency(self, metrics: List[ScalingMetrics]) -> float:
        """Calculate consistency bonus based on metric trends."""
        if len(metrics) < 2:
            return 0.0
        
        # Check CPU utilization trend
        cpu_values = [m.cpu_utilization for m in metrics]
        cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]
        
        # Higher consistency bonus for consistent trends
        if abs(cpu_trend) > 5:  # Strong trend
            return 0.2
        elif abs(cpu_trend) > 2:  # Moderate trend
            return 0.1
        else:
            return 0.0
    
    async def _get_current_instances(self) -> int:
        """Get current number of instances."""
        if self.resource_provider:
            return await self.resource_provider.get_current_instances()
        return self.current_instances
    
    async def _estimate_scaling_cost(self, from_instances: int, to_instances: int) -> float:
        """Estimate cost impact of scaling operation."""
        if self.resource_provider:
            return await self.resource_provider.get_scaling_cost(from_instances, to_instances)
        
        # Simple cost estimation
        instance_cost_per_hour = 0.10  # $0.10 per instance per hour
        instance_diff = to_instances - from_instances
        return instance_diff * instance_cost_per_hour
    
    async def _find_max_instances_within_budget(self, current_instances: int) -> int:
        """Find maximum instances within cost budget."""
        if not self.policy.max_cost_per_hour:
            return self.policy.max_instances
        
        # Binary search for max instances within budget
        low, high = current_instances, self.policy.max_instances
        
        while low < high:
            mid = (low + high + 1) // 2
            cost = await self._estimate_scaling_cost(current_instances, mid)
            
            if cost <= self.policy.max_cost_per_hour:
                low = mid
            else:
                high = mid - 1
        
        return low
    
    async def _execute_scaling_decision(self, decision: ScalingDecision) -> None:
        """Execute the scaling decision."""
        if not self.resource_provider:
            logger.warning("No resource provider configured, cannot execute scaling")
            return
        
        try:
            success = await self.resource_provider.scale_to(decision.target_instances)
            
            if success:
                self.current_instances = decision.target_instances
                
                if decision.direction == ScalingDirection.UP:
                    self.last_scale_up_time = time.time()
                    for callback in self.on_scale_up:
                        await callback(decision)
                elif decision.direction == ScalingDirection.DOWN:
                    self.last_scale_down_time = time.time()
                    for callback in self.on_scale_down:
                        await callback(decision)
                
                logger.info(f"Successfully scaled to {decision.target_instances} instances")
            else:
                logger.error("Failed to execute scaling decision")
                
        except Exception as e:
            logger.error(f"Error executing scaling decision: {e}")
    
    def add_scale_up_callback(self, callback: Callable) -> None:
        """Add callback for scale-up events."""
        self.on_scale_up.append(callback)
    
    def add_scale_down_callback(self, callback: Callable) -> None:
        """Add callback for scale-down events."""
        self.on_scale_down.append(callback)
    
    def add_scaling_decision_callback(self, callback: Callable) -> None:
        """Add callback for all scaling decisions."""
        self.on_scaling_decision.append(callback)
    
    def get_metrics_history(self, duration_seconds: float = 3600) -> List[ScalingMetrics]:
        """Get metrics history for specified duration."""
        cutoff_time = time.time() - duration_seconds
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def get_scaling_history(self, duration_seconds: float = 3600) -> List[ScalingDecision]:
        """Get scaling decision history for specified duration."""
        cutoff_time = time.time() - duration_seconds
        return [d for d in self.scaling_history if d.timestamp > cutoff_time]
    
    async def force_scale_to(self, target_instances: int, reason: str = "Manual override") -> bool:
        """Force scaling to specific number of instances."""
        if target_instances < self.policy.min_instances:
            raise ValueError(f"Target instances below minimum: {self.policy.min_instances}")
        if target_instances > self.policy.max_instances:
            raise ValueError(f"Target instances above maximum: {self.policy.max_instances}")
        
        current_instances = await self._get_current_instances()
        
        decision = ScalingDecision(
            direction=(ScalingDirection.UP if target_instances > current_instances 
                      else ScalingDirection.DOWN if target_instances < current_instances
                      else ScalingDirection.STABLE),
            target_instances=target_instances,
            current_instances=current_instances,
            triggers=[ScalingTrigger.CUSTOM_METRIC],
            confidence=1.0,
            estimated_cost_impact=await self._estimate_scaling_cost(current_instances, target_instances),
            rationale=f"Manual scaling: {reason}"
        )
        
        await self._execute_scaling_decision(decision)
        self.scaling_history.append(decision)
        
        return True