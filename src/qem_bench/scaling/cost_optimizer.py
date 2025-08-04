"""
Cost optimization system for quantum error mitigation workloads.

This module provides intelligent cost optimization strategies that minimize
quantum computing expenses while maintaining performance requirements through
spot instance management, resource scheduling, and budget-aware allocation.
"""

import time
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

from ..security import SecureConfig


logger = logging.getLogger(__name__)


class CostModel(Enum):
    """Cost calculation models."""
    PAY_PER_USE = "pay_per_use"
    RESERVED_CAPACITY = "reserved_capacity"
    SPOT_PRICING = "spot_pricing"
    HYBRID = "hybrid"


class OptimizationStrategy(Enum):
    """Cost optimization strategies."""
    MINIMIZE_COST = "minimize_cost"
    COST_PERFORMANCE_BALANCED = "cost_performance_balanced"
    BUDGET_CONSTRAINED = "budget_constrained"
    DEADLINE_AWARE = "deadline_aware"


class ResourceTier(Enum):
    """Resource performance/cost tiers."""
    ECONOMY = "economy"      # Lowest cost, basic performance
    STANDARD = "standard"    # Balanced cost/performance
    PREMIUM = "premium"      # High performance, higher cost
    SPOT = "spot"           # Variable pricing, can be interrupted


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a resource or workload."""
    compute_cost: float = 0.0
    storage_cost: float = 0.0
    network_cost: float = 0.0
    quantum_backend_cost: float = 0.0
    overhead_cost: float = 0.0
    
    # Time-based breakdown
    hourly_cost: float = 0.0
    daily_cost: float = 0.0
    monthly_cost: float = 0.0
    
    # Usage-based breakdown
    per_job_cost: float = 0.0
    per_shot_cost: float = 0.0
    per_circuit_cost: float = 0.0
    
    def total_cost(self) -> float:
        """Calculate total cost."""
        return (
            self.compute_cost + 
            self.storage_cost + 
            self.network_cost + 
            self.quantum_backend_cost + 
            self.overhead_cost
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "compute_cost": self.compute_cost,
            "storage_cost": self.storage_cost,
            "network_cost": self.network_cost,
            "quantum_backend_cost": self.quantum_backend_cost,
            "overhead_cost": self.overhead_cost,
            "total_cost": self.total_cost(),
            "hourly_cost": self.hourly_cost,
            "daily_cost": self.daily_cost,
            "monthly_cost": self.monthly_cost,
            "per_job_cost": self.per_job_cost,
            "per_shot_cost": self.per_shot_cost,
            "per_circuit_cost": self.per_circuit_cost
        }


@dataclass
class Budget:
    """Budget configuration and tracking."""
    total_budget: float
    period_days: int = 30  # Budget period in days
    
    # Spending tracking
    current_spending: float = 0.0
    period_start_time: float = field(default_factory=time.time)
    
    # Alerts and limits
    warning_threshold: float = 0.8  # 80% of budget
    critical_threshold: float = 0.95  # 95% of budget
    hard_limit: bool = True  # Stop spending at budget limit
    
    # Category-specific budgets
    compute_budget: Optional[float] = None
    quantum_budget: Optional[float] = None
    storage_budget: Optional[float] = None
    
    def remaining_budget(self) -> float:
        """Calculate remaining budget."""
        return max(0, self.total_budget - self.current_spending)
    
    def utilization(self) -> float:
        """Calculate budget utilization percentage."""
        return self.current_spending / self.total_budget
    
    def daily_burn_rate(self) -> float:
        """Calculate current daily spending rate."""
        elapsed_days = (time.time() - self.period_start_time) / (24 * 3600)
        if elapsed_days <= 0:
            return 0.0
        return self.current_spending / elapsed_days
    
    def projected_spending(self) -> float:
        """Project spending for full period based on current rate."""
        daily_rate = self.daily_burn_rate()
        return daily_rate * self.period_days
    
    def is_over_budget(self) -> bool:
        """Check if over budget."""
        return self.current_spending > self.total_budget
    
    def is_at_warning(self) -> bool:
        """Check if at warning threshold."""
        return self.utilization() >= self.warning_threshold
    
    def is_at_critical(self) -> bool:
        """Check if at critical threshold."""
        return self.utilization() >= self.critical_threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_budget": self.total_budget,
            "period_days": self.period_days,
            "current_spending": self.current_spending,
            "remaining_budget": self.remaining_budget(),
            "utilization": self.utilization(),
            "daily_burn_rate": self.daily_burn_rate(),
            "projected_spending": self.projected_spending(),
            "is_over_budget": self.is_over_budget(),
            "is_at_warning": self.is_at_warning(),
            "is_at_critical": self.is_at_critical()
        }


@dataclass
class CostOptimizationPolicy:
    """Policy for cost optimization behavior."""
    strategy: OptimizationStrategy = OptimizationStrategy.COST_PERFORMANCE_BALANCED
    
    # Performance trade-offs
    max_performance_degradation: float = 0.2  # 20% max degradation
    acceptable_latency_increase: float = 2.0  # 2x latency increase acceptable
    
    # Spot instance configuration
    enable_spot_instances: bool = True
    max_spot_interruption_rate: float = 0.05  # 5% max interruption rate
    spot_savings_threshold: float = 0.3  # 30% minimum savings for spot
    
    # Resource scheduling
    enable_delayed_execution: bool = True
    max_delay_hours: float = 24.0  # Maximum delay for cost savings
    off_peak_hours: List[int] = field(default_factory=lambda: list(range(20, 6)))  # 8 PM to 6 AM
    
    # Budget management
    emergency_stop_enabled: bool = True
    cost_per_job_limit: Optional[float] = None
    hourly_spending_limit: Optional[float] = None
    
    # Optimization preferences
    prefer_reserved_instances: bool = False
    consolidation_enabled: bool = True
    auto_scaling_cost_factor: float = 0.3  # Weight of cost in scaling decisions


@dataclass
class SpotInstanceInfo:
    """Information about spot instance pricing and availability."""
    instance_type: str
    availability_zone: str
    current_price: float
    price_history: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, price)
    interruption_rate: float = 0.0
    savings_vs_on_demand: float = 0.0
    
    def average_price(self, hours: int = 24) -> float:
        """Calculate average price over recent hours."""
        if not self.price_history:
            return self.current_price
        
        cutoff_time = time.time() - hours * 3600
        recent_prices = [price for timestamp, price in self.price_history if timestamp > cutoff_time]
        
        return np.mean(recent_prices) if recent_prices else self.current_price
    
    def price_volatility(self, hours: int = 24) -> float:
        """Calculate price volatility over recent hours."""
        if not self.price_history:
            return 0.0
        
        cutoff_time = time.time() - hours * 3600
        recent_prices = [price for timestamp, price in self.price_history if timestamp > cutoff_time]
        
        return np.std(recent_prices) if len(recent_prices) > 1 else 0.0


class CostCalculator(ABC):
    """Abstract base class for cost calculation."""
    
    @abstractmethod
    def calculate_cost(
        self,
        resource_usage: Dict[str, Any],
        duration_hours: float
    ) -> CostBreakdown:
        """Calculate cost for given resource usage."""
        pass


class SimpleCostCalculator(CostCalculator):
    """Simple cost calculator with fixed rates."""
    
    def __init__(
        self,
        compute_rate_per_hour: float = 0.10,
        storage_rate_per_gb_hour: float = 0.001,
        quantum_rate_per_shot: float = 0.0001,
        network_rate_per_gb: float = 0.01
    ):
        self.compute_rate_per_hour = compute_rate_per_hour
        self.storage_rate_per_gb_hour = storage_rate_per_gb_hour
        self.quantum_rate_per_shot = quantum_rate_per_shot
        self.network_rate_per_gb = network_rate_per_gb
    
    def calculate_cost(
        self,
        resource_usage: Dict[str, Any],
        duration_hours: float
    ) -> CostBreakdown:
        """Calculate cost with simple fixed rates."""
        breakdown = CostBreakdown()
        
        # Compute cost
        cpu_hours = resource_usage.get("cpu_hours", 0)
        breakdown.compute_cost = cpu_hours * self.compute_rate_per_hour
        
        # Storage cost
        storage_gb_hours = resource_usage.get("storage_gb", 0) * duration_hours
        breakdown.storage_cost = storage_gb_hours * self.storage_rate_per_gb_hour
        
        # Network cost
        network_gb = resource_usage.get("network_gb", 0)
        breakdown.network_cost = network_gb * self.network_rate_per_gb
        
        # Quantum backend cost
        shots = resource_usage.get("shots", 0)
        breakdown.quantum_backend_cost = shots * self.quantum_rate_per_shot
        
        # Time-based costs
        breakdown.hourly_cost = breakdown.total_cost() / max(duration_hours, 1)
        breakdown.daily_cost = breakdown.hourly_cost * 24
        breakdown.monthly_cost = breakdown.daily_cost * 30
        
        # Usage-based costs
        jobs = resource_usage.get("jobs", 1)
        circuits = resource_usage.get("circuits", 1)
        breakdown.per_job_cost = breakdown.total_cost() / max(jobs, 1)
        breakdown.per_circuit_cost = breakdown.total_cost() / max(circuits, 1)
        breakdown.per_shot_cost = breakdown.quantum_backend_cost / max(shots, 1)
        
        return breakdown


class SpotInstanceManager:
    """Manages spot instance pricing and availability."""
    
    def __init__(self):
        self.spot_info: Dict[str, SpotInstanceInfo] = {}
        self.price_update_interval = 300.0  # 5 minutes
        self.is_monitoring = False
    
    async def start_monitoring(self) -> None:
        """Start monitoring spot instance prices."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        logger.info("Starting spot instance price monitoring")
        
        # Start price monitoring loop
        asyncio.create_task(self._price_monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring spot instance prices."""
        self.is_monitoring = False
        logger.info("Stopped spot instance price monitoring")
    
    async def _price_monitoring_loop(self) -> None:
        """Monitor spot instance prices."""
        while self.is_monitoring:
            try:
                await self._update_spot_prices()
                await asyncio.sleep(self.price_update_interval)
            except Exception as e:
                logger.error(f"Error updating spot prices: {e}")
                await asyncio.sleep(self.price_update_interval)
    
    async def _update_spot_prices(self) -> None:
        """Update spot instance price information."""
        # In practice, this would call cloud provider APIs
        # For now, simulate price updates
        
        for instance_type in ["t3.medium", "t3.large", "c5.xlarge"]:
            if instance_type not in self.spot_info:
                self.spot_info[instance_type] = SpotInstanceInfo(
                    instance_type=instance_type,
                    availability_zone="us-east-1a",
                    current_price=0.05,
                    interruption_rate=0.02,
                    savings_vs_on_demand=0.7
                )
            
            # Simulate price fluctuation
            info = self.spot_info[instance_type]
            base_price = 0.05
            fluctuation = np.random.normal(0, 0.01)
            new_price = max(0.01, base_price + fluctuation)
            
            info.price_history.append((time.time(), new_price))
            info.current_price = new_price
            
            # Keep only recent history
            cutoff_time = time.time() - 24 * 3600
            info.price_history = [
                (timestamp, price) for timestamp, price in info.price_history
                if timestamp > cutoff_time
            ]
    
    def get_best_spot_instances(
        self,
        requirements: Dict[str, Any],
        max_interruption_rate: float = 0.05
    ) -> List[SpotInstanceInfo]:
        """Get best spot instances based on requirements."""
        suitable_instances = []
        
        for info in self.spot_info.values():
            if info.interruption_rate <= max_interruption_rate:
                suitable_instances.append(info)
        
        # Sort by current price
        suitable_instances.sort(key=lambda x: x.current_price)
        
        return suitable_instances
    
    def predict_spot_price_stability(
        self,
        instance_type: str,
        hours_ahead: int = 1
    ) -> float:
        """Predict spot price stability (0-1, higher is more stable)."""
        if instance_type not in self.spot_info:
            return 0.5  # Unknown stability
        
        info = self.spot_info[instance_type]
        volatility = info.price_volatility(hours=24)
        
        # Higher volatility = lower stability
        stability = max(0, 1.0 - volatility / info.current_price)
        
        return stability


class CostOptimizer:
    """
    Intelligent cost optimization system for quantum error mitigation workloads.
    
    Features:
    - Multi-strategy cost optimization (minimize cost, balanced, deadline-aware)
    - Spot instance management with interruption handling
    - Budget tracking and alerts
    - Resource consolidation and right-sizing
    - Off-peak scheduling for cost savings
    - Cost-performance trade-off analysis
    
    Example:
        >>> budget = Budget(total_budget=1000.0, period_days=30)
        >>> policy = CostOptimizationPolicy(
        ...     strategy=OptimizationStrategy.COST_PERFORMANCE_BALANCED,
        ...     enable_spot_instances=True
        ... )
        >>> optimizer = CostOptimizer(budget=budget, policy=policy)
        >>> await optimizer.start()
        >>> recommendations = await optimizer.get_cost_recommendations()
    """
    
    def __init__(
        self,
        budget: Optional[Budget] = None,
        policy: Optional[CostOptimizationPolicy] = None,
        cost_calculator: Optional[CostCalculator] = None,
        config: Optional[SecureConfig] = None
    ):
        self.budget = budget
        self.policy = policy or CostOptimizationPolicy()
        self.cost_calculator = cost_calculator or SimpleCostCalculator()
        self.config = config or SecureConfig()
        
        # Spot instance management
        self.spot_manager = SpotInstanceManager()
        
        # Cost tracking
        self.cost_history: List[Tuple[float, CostBreakdown]] = []
        self.cost_alerts: List[Dict[str, Any]] = []
        
        # Optimization state
        self.is_running = False
        self.optimization_interval = 300.0  # 5 minutes
        
        # Statistics
        self.total_cost_saved: float = 0.0
        self.optimization_decisions: List[Dict[str, Any]] = []
        
        logger.info("CostOptimizer initialized")
    
    async def start(self) -> None:
        """Start the cost optimization system."""
        if self.is_running:
            logger.warning("CostOptimizer is already running")
            return
        
        self.is_running = True
        logger.info("Starting CostOptimizer")
        
        # Start spot instance monitoring
        if self.policy.enable_spot_instances:
            await self.spot_manager.start_monitoring()
        
        # Start optimization loop
        asyncio.create_task(self._optimization_loop())
        
        # Start budget monitoring
        if self.budget:
            asyncio.create_task(self._budget_monitoring_loop())
    
    async def stop(self) -> None:
        """Stop the cost optimization system."""
        self.is_running = False
        
        if self.policy.enable_spot_instances:
            await self.spot_manager.stop_monitoring()
        
        logger.info("CostOptimizer stopped")
    
    async def _optimization_loop(self) -> None:
        """Main cost optimization loop."""
        while self.is_running:
            try:
                # Analyze current costs
                await self._analyze_current_costs()
                
                # Generate optimization recommendations
                recommendations = await self._generate_cost_recommendations()
                
                # Apply automatic optimizations
                await self._apply_automatic_optimizations(recommendations)
                
                await asyncio.sleep(self.optimization_interval)
                
            except Exception as e:
                logger.error(f"Error in cost optimization loop: {e}")
                await asyncio.sleep(self.optimization_interval)
    
    async def _budget_monitoring_loop(self) -> None:
        """Monitor budget usage and generate alerts."""
        if not self.budget:
            return
        
        while self.is_running:
            try:
                # Check budget thresholds
                if self.budget.is_at_critical():
                    await self._handle_critical_budget_alert()
                elif self.budget.is_at_warning():
                    await self._handle_warning_budget_alert()
                
                # Check if budget period has reset
                if self._should_reset_budget():
                    await self._reset_budget_period()
                
                await asyncio.sleep(3600.0)  # Check hourly
                
            except Exception as e:
                logger.error(f"Error in budget monitoring: {e}")
                await asyncio.sleep(3600.0)
    
    async def _analyze_current_costs(self) -> None:
        """Analyze current cost patterns."""
        current_time = time.time()
        
        # Calculate recent costs (placeholder - would integrate with actual billing)
        recent_usage = {
            "cpu_hours": 10.0,
            "storage_gb": 100.0,
            "shots": 50000,
            "jobs": 25,
            "circuits": 100,
            "network_gb": 5.0
        }
        
        cost_breakdown = self.cost_calculator.calculate_cost(recent_usage, 1.0)
        self.cost_history.append((current_time, cost_breakdown))
        
        # Update budget if configured
        if self.budget:
            self.budget.current_spending += cost_breakdown.total_cost()
        
        # Keep only recent history
        cutoff_time = current_time - 7 * 24 * 3600  # 7 days
        self.cost_history = [
            (timestamp, breakdown) for timestamp, breakdown in self.cost_history
            if timestamp > cutoff_time
        ]
    
    async def _generate_cost_recommendations(self) -> List[Dict[str, Any]]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        # Analyze spending patterns
        if len(self.cost_history) >= 24:  # Need at least 24 hours of data
            recent_costs = [breakdown.total_cost() for _, breakdown in self.cost_history[-24:]]
            cost_trend = np.polyfit(range(len(recent_costs)), recent_costs, 1)[0]
            
            if cost_trend > 0.01:  # Increasing costs
                recommendations.append({
                    "type": "cost_trend",
                    "severity": "medium",
                    "message": "Costs are trending upward",
                    "action": "Consider resource consolidation or spot instances",
                    "potential_savings": cost_trend * 24 * 30  # Monthly projection
                })
        
        # Spot instance recommendations
        if self.policy.enable_spot_instances:
            spot_recommendations = await self._generate_spot_recommendations()
            recommendations.extend(spot_recommendations)
        
        # Resource utilization recommendations
        utilization_recommendations = await self._generate_utilization_recommendations()
        recommendations.extend(utilization_recommendations)
        
        # Scheduling recommendations
        scheduling_recommendations = await self._generate_scheduling_recommendations()
        recommendations.extend(scheduling_recommendations)
        
        return recommendations
    
    async def _generate_spot_recommendations(self) -> List[Dict[str, Any]]:
        """Generate spot instance recommendations."""
        recommendations = []
        
        # Get best spot instances
        spot_instances = self.spot_manager.get_best_spot_instances(
            requirements={},
            max_interruption_rate=self.policy.max_spot_interruption_rate
        )
        
        for instance in spot_instances[:3]:  # Top 3 recommendations
            if instance.savings_vs_on_demand >= self.policy.spot_savings_threshold:
                stability = self.spot_manager.predict_spot_price_stability(
                    instance.instance_type
                )
                
                recommendations.append({
                    "type": "spot_instance",
                    "severity": "low",
                    "message": f"Consider using {instance.instance_type} spot instances",
                    "action": f"Switch to spot instances for {instance.savings_vs_on_demand*100:.1f}% savings",
                    "potential_savings": instance.savings_vs_on_demand * 100,  # Estimated monthly savings
                    "stability_score": stability,
                    "current_price": instance.current_price
                })
        
        return recommendations
    
    async def _generate_utilization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate resource utilization recommendations."""
        recommendations = []
        
        # This would analyze actual resource utilization
        # For now, provide generic recommendations
        
        recommendations.append({
            "type": "resource_sizing",
            "severity": "medium",
            "message": "Some resources appear underutilized",
            "action": "Consider downsizing or consolidating workloads",
            "potential_savings": 50.0  # Estimated savings
        })
        
        return recommendations
    
    async def _generate_scheduling_recommendations(self) -> List[Dict[str, Any]]:
        """Generate job scheduling recommendations."""
        recommendations = []
        
        if self.policy.enable_delayed_execution:
            current_hour = datetime.now().hour
            
            if current_hour not in self.policy.off_peak_hours:
                recommendations.append({
                    "type": "scheduling",
                    "severity": "low",
                    "message": "Consider scheduling non-urgent jobs during off-peak hours",
                    "action": f"Delay execution to off-peak hours ({self.policy.off_peak_hours})",
                    "potential_savings": 20.0  # Estimated savings from off-peak pricing
                })
        
        return recommendations
    
    async def _apply_automatic_optimizations(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> None:
        """Apply automatic cost optimizations based on policy."""
        for recommendation in recommendations:
            if recommendation["severity"] == "low":
                # Apply low-impact optimizations automatically
                await self._apply_optimization(recommendation)
    
    async def _apply_optimization(self, recommendation: Dict[str, Any]) -> None:
        """Apply a specific optimization."""
        optimization_type = recommendation["type"]
        
        try:
            if optimization_type == "spot_instance":
                # In practice, this would trigger spot instance launch
                logger.info(f"Applied spot instance optimization: {recommendation['message']}")
                self.total_cost_saved += recommendation.get("potential_savings", 0)
            
            elif optimization_type == "scheduling":
                # In practice, this would reschedule jobs
                logger.info(f"Applied scheduling optimization: {recommendation['message']}")
                self.total_cost_saved += recommendation.get("potential_savings", 0)
            
            # Record the decision
            self.optimization_decisions.append({
                "timestamp": time.time(),
                "type": optimization_type,
                "action": recommendation["action"],
                "estimated_savings": recommendation.get("potential_savings", 0)
            })
            
        except Exception as e:
            logger.error(f"Failed to apply optimization {optimization_type}: {e}")
    
    async def _handle_warning_budget_alert(self) -> None:
        """Handle budget warning threshold reached."""
        alert = {
            "timestamp": time.time(),
            "type": "budget_warning",
            "message": f"Budget utilization at {self.budget.utilization()*100:.1f}%",
            "remaining_budget": self.budget.remaining_budget(),
            "projected_overage": max(0, self.budget.projected_spending() - self.budget.total_budget)
        }
        
        self.cost_alerts.append(alert)
        logger.warning(f"Budget warning: {alert['message']}")
    
    async def _handle_critical_budget_alert(self) -> None:
        """Handle critical budget threshold reached."""
        alert = {
            "timestamp": time.time(),
            "type": "budget_critical",
            "message": f"Budget utilization at {self.budget.utilization()*100:.1f}%",
            "remaining_budget": self.budget.remaining_budget(),
            "action_taken": "Enhanced cost monitoring enabled"
        }
        
        self.cost_alerts.append(alert)
        logger.critical(f"Budget critical: {alert['message']}")
        
        if self.budget.hard_limit and self.budget.is_over_budget():
            # Emergency stop if over budget with hard limit
            alert["action_taken"] = "Emergency cost controls activated"
            await self._activate_emergency_cost_controls()
    
    async def _activate_emergency_cost_controls(self) -> None:
        """Activate emergency cost control measures."""
        logger.critical("Activating emergency cost controls")
        
        # In practice, this would:
        # - Stop non-critical jobs
        # - Scale down resources
        # - Switch to cheapest available options
        # - Send alerts to administrators
        
    def _should_reset_budget(self) -> bool:
        """Check if budget period should reset."""
        if not self.budget:
            return False
        
        elapsed_days = (time.time() - self.budget.period_start_time) / (24 * 3600)
        return elapsed_days >= self.budget.period_days
    
    async def _reset_budget_period(self) -> None:
        """Reset budget for new period."""
        logger.info("Resetting budget period")
        
        self.budget.current_spending = 0.0
        self.budget.period_start_time = time.time()
    
    def calculate_job_cost(
        self,
        job_requirements: Dict[str, Any],
        resource_options: List[Dict[str, Any]]
    ) -> Dict[str, CostBreakdown]:
        """Calculate cost for a job across different resource options."""
        cost_estimates = {}
        
        for resource in resource_options:
            resource_id = resource["id"]
            
            # Estimate resource usage for this job
            estimated_usage = self._estimate_job_resource_usage(
                job_requirements, resource
            )
            
            # Calculate cost
            duration_hours = job_requirements.get("estimated_duration", 3600) / 3600
            cost_breakdown = self.cost_calculator.calculate_cost(
                estimated_usage, duration_hours
            )
            
            cost_estimates[resource_id] = cost_breakdown
        
        return cost_estimates
    
    def _estimate_job_resource_usage(
        self,
        job_requirements: Dict[str, Any],
        resource: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate resource usage for a job."""
        # This would use historical data and job characteristics
        # For now, provide basic estimates
        
        return {
            "cpu_hours": job_requirements.get("estimated_duration", 3600) / 3600,
            "storage_gb": 1.0,  # Minimal storage for results
            "shots": job_requirements.get("shots", 1024),
            "jobs": 1,
            "circuits": job_requirements.get("circuits", 1),
            "network_gb": 0.1  # Minimal network usage
        }
    
    def recommend_cost_optimal_resources(
        self,
        job_requirements: Dict[str, Any],
        available_resources: List[Dict[str, Any]],
        performance_threshold: float = 0.8
    ) -> List[Tuple[Dict[str, Any], CostBreakdown, float]]:
        """Recommend cost-optimal resources for a job."""
        cost_estimates = self.calculate_job_cost(job_requirements, available_resources)
        
        recommendations = []
        
        for resource in available_resources:
            resource_id = resource["id"]
            cost_breakdown = cost_estimates[resource_id]
            
            # Calculate performance score (simplified)
            performance_score = resource.get("performance_score", 1.0)
            
            # Only consider resources that meet performance threshold
            if performance_score >= performance_threshold:
                recommendations.append((resource, cost_breakdown, performance_score))
        
        # Sort by cost (lowest first)
        recommendations.sort(key=lambda x: x[1].total_cost())
        
        return recommendations
    
    def get_cost_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive cost analytics."""
        cutoff_time = time.time() - days * 24 * 3600
        recent_costs = [
            breakdown for timestamp, breakdown in self.cost_history
            if timestamp > cutoff_time
        ]
        
        if not recent_costs:
            return {"message": "No cost data available"}
        
        # Calculate analytics
        total_costs = [cost.total_cost() for cost in recent_costs]
        
        analytics = {
            "period_days": days,
            "total_cost": sum(total_costs),
            "average_daily_cost": sum(total_costs) / max(days, 1),
            "cost_trend": self._calculate_cost_trend(recent_costs),
            "cost_breakdown": self._aggregate_cost_breakdown(recent_costs),
            "cost_savings": self.total_cost_saved,
            "optimization_decisions": len(self.optimization_decisions),
            "recent_alerts": len([a for a in self.cost_alerts if a["timestamp"] > cutoff_time])
        }
        
        if self.budget:
            analytics["budget_status"] = self.budget.to_dict()
        
        return analytics
    
    def _calculate_cost_trend(self, cost_history: List[CostBreakdown]) -> Dict[str, float]:
        """Calculate cost trend over time."""
        if len(cost_history) < 2:
            return {"slope": 0.0, "r_squared": 0.0}
        
        costs = [cost.total_cost() for cost in cost_history]
        x = np.arange(len(costs))
        
        # Linear regression
        coeffs = np.polyfit(x, costs, 1)
        slope = coeffs[0]
        
        # Calculate R-squared
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((costs - y_pred) ** 2)
        ss_tot = np.sum((costs - np.mean(costs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {"slope": float(slope), "r_squared": float(r_squared)}
    
    def _aggregate_cost_breakdown(self, cost_history: List[CostBreakdown]) -> Dict[str, float]:
        """Aggregate cost breakdown across time period."""
        if not cost_history:
            return {}
        
        return {
            "compute_cost": sum(cost.compute_cost for cost in cost_history),
            "storage_cost": sum(cost.storage_cost for cost in cost_history),
            "network_cost": sum(cost.network_cost for cost in cost_history),
            "quantum_backend_cost": sum(cost.quantum_backend_cost for cost in cost_history),
            "overhead_cost": sum(cost.overhead_cost for cost in cost_history)
        }
    
    async def get_cost_recommendations(self) -> List[Dict[str, Any]]:
        """Get current cost optimization recommendations."""
        return await self._generate_cost_recommendations()
    
    def set_budget(self, budget: Budget) -> None:
        """Set or update budget configuration."""
        self.budget = budget
        logger.info(f"Budget set: ${budget.total_budget} for {budget.period_days} days")
    
    def add_cost_alert_callback(self, callback) -> None:
        """Add callback for cost alerts."""
        # In practice, this would register callback functions
        pass