"""
Auto-scaling and load balancing for quantum error mitigation computations.

This module provides dynamic resource allocation, load balancing across
multiple backends, and cost optimization for cloud quantum services.
"""

import time
import threading
import queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict, deque
import statistics

from ..logging import get_logger


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"      # Scale based on current load
    PREDICTIVE = "predictive"  # Scale based on predicted load
    PROACTIVE = "proactive"    # Pre-scale for expected workload
    HYBRID = "hybrid"          # Combination of strategies


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"        # Simple round-robin
    LEAST_CONNECTIONS = "least_connections"  # Least active connections
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"  # Based on response time
    RESOURCE_BASED = "resource_based"  # Based on resource utilization
    QUEUE_LENGTH = "queue_length"      # Based on queue length
    ADAPTIVE = "adaptive"              # Adaptive based on performance


class BackendStatus(Enum):
    """Backend status states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling."""
    
    # Scaling parameters
    min_backends: int = 1
    max_backends: int = 10
    target_utilization: float = 0.7
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    
    # Timing parameters
    scale_up_cooldown: float = 300.0    # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes
    health_check_interval: float = 30.0
    
    # Predictive scaling
    enable_predictive: bool = True
    prediction_window: int = 10  # Number of samples for prediction
    prediction_accuracy_threshold: float = 0.8
    
    # Cost optimization
    enable_cost_optimization: bool = True
    cost_per_hour: Dict[str, float] = field(default_factory=dict)
    max_cost_per_hour: float = 100.0
    
    # Performance thresholds
    max_response_time: float = 30.0
    max_queue_length: int = 100
    max_error_rate: float = 0.1


@dataclass
class BackendMetrics:
    """Metrics for a quantum backend."""
    
    backend_id: str
    status: BackendStatus = BackendStatus.HEALTHY
    
    # Performance metrics
    response_time: float = 0.0
    queue_length: int = 0
    active_jobs: int = 0
    throughput: float = 0.0
    error_rate: float = 0.0
    
    # Resource metrics
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    cost_per_hour: float = 0.0
    
    # History
    response_time_history: deque = field(default_factory=lambda: deque(maxlen=100))
    throughput_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    last_updated: float = field(default_factory=time.time)
    
    def update_metrics(
        self,
        response_time: Optional[float] = None,
        queue_length: Optional[int] = None,
        active_jobs: Optional[int] = None,
        error_rate: Optional[float] = None
    ) -> None:
        """Update backend metrics."""
        current_time = time.time()
        
        if response_time is not None:
            self.response_time = response_time
            self.response_time_history.append((current_time, response_time))
        
        if queue_length is not None:
            self.queue_length = queue_length
        
        if active_jobs is not None:
            self.active_jobs = active_jobs
        
        if error_rate is not None:
            self.error_rate = error_rate
        
        # Calculate throughput based on completed jobs
        if len(self.throughput_history) >= 2:
            time_diff = current_time - self.throughput_history[-1][0]
            if time_diff > 0:
                job_diff = max(0, self.active_jobs - self.throughput_history[-1][1])
                self.throughput = job_diff / time_diff
        
        self.throughput_history.append((current_time, self.active_jobs))
        self.last_updated = current_time
    
    def get_health_score(self) -> float:
        """Calculate health score (0-1, higher is better)."""
        score = 1.0
        
        # Response time penalty
        if self.response_time > 10.0:
            score *= 0.8
        elif self.response_time > 5.0:
            score *= 0.9
        
        # Queue length penalty
        if self.queue_length > 50:
            score *= 0.7
        elif self.queue_length > 20:
            score *= 0.85
        
        # Error rate penalty
        if self.error_rate > 0.05:
            score *= 0.6
        elif self.error_rate > 0.02:
            score *= 0.8
        
        return max(0.0, score)
    
    def get_load_score(self) -> float:
        """Calculate load score (0-1, higher means more loaded)."""
        # Weighted combination of metrics
        queue_score = min(1.0, self.queue_length / 100.0)
        response_score = min(1.0, self.response_time / 30.0)
        active_jobs_score = min(1.0, self.active_jobs / 50.0)
        
        return (queue_score * 0.4 + response_score * 0.3 + active_jobs_score * 0.3)


class LoadBalancer:
    """
    Load balancer for distributing quantum computations across backends.
    
    This class implements various load balancing strategies to optimize
    resource utilization and minimize response times across multiple
    quantum backends.
    
    Features:
    - Multiple load balancing algorithms
    - Real-time backend health monitoring
    - Automatic failover and recovery
    - Performance-based routing
    - Queue management
    - Cost-aware routing
    
    Example:
        >>> balancer = LoadBalancer(strategy=LoadBalancingStrategy.ADAPTIVE)
        >>> balancer.add_backend("ibm_quantum", backend_instance)
        >>> balancer.add_backend("rigetti", backend_instance)
        >>> 
        >>> # Automatically route to best backend
        >>> result = balancer.execute(circuit, shots=1024)
    """
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
        config: Optional[ScalingConfig] = None
    ):
        """
        Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
            config: Scaling configuration
        """
        self.strategy = strategy
        self.config = config or ScalingConfig()
        self.logger = get_logger()
        
        # Backend management
        self.backends: Dict[str, Any] = {}
        self.backend_metrics: Dict[str, BackendMetrics] = {}
        self.backend_weights: Dict[str, float] = {}
        
        # Load balancing state
        self.current_backend_index = 0
        self.request_counts: Dict[str, int] = defaultdict(int)
        
        # Monitoring and health checking
        self._monitoring_active = False
        self._monitor_thread = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info(f"LoadBalancer initialized with strategy: {strategy.value}")
    
    def add_backend(
        self,
        backend_id: str,
        backend: Any,
        weight: float = 1.0,
        cost_per_hour: float = 0.0
    ) -> None:
        """
        Add a backend to the load balancer.
        
        Args:
            backend_id: Unique identifier for the backend
            backend: Backend instance
            weight: Weight for weighted load balancing
            cost_per_hour: Cost per hour for this backend
        """
        with self._lock:
            self.backends[backend_id] = backend
            self.backend_weights[backend_id] = weight
            self.backend_metrics[backend_id] = BackendMetrics(
                backend_id=backend_id,
                cost_per_hour=cost_per_hour
            )
            
            self.logger.info(f"Added backend {backend_id} with weight {weight}")
            
            # Start monitoring if this is the first backend
            if len(self.backends) == 1 and not self._monitoring_active:
                self._start_monitoring()
    
    def remove_backend(self, backend_id: str) -> bool:
        """
        Remove a backend from the load balancer.
        
        Args:
            backend_id: Backend identifier to remove
            
        Returns:
            True if backend was removed
        """
        with self._lock:
            if backend_id in self.backends:
                del self.backends[backend_id]
                del self.backend_metrics[backend_id]
                del self.backend_weights[backend_id]
                self.request_counts.pop(backend_id, None)
                
                self.logger.info(f"Removed backend {backend_id}")
                return True
            
            return False
    
    def execute(
        self,
        task: Callable,
        *args,
        timeout: Optional[float] = None,
        retry_count: int = 3,
        **kwargs
    ) -> Any:
        """
        Execute task on the best available backend.
        
        Args:
            task: Task to execute
            *args: Task arguments
            timeout: Execution timeout
            retry_count: Number of retries on failure
            **kwargs: Task keyword arguments
            
        Returns:
            Task result
        """
        for attempt in range(retry_count + 1):
            try:
                # Select best backend
                backend_id = self.select_backend()
                if backend_id is None:
                    raise RuntimeError("No healthy backends available")
                
                backend = self.backends[backend_id]
                metrics = self.backend_metrics[backend_id]
                
                # Execute task
                start_time = time.time()
                result = task(backend, *args, **kwargs)
                execution_time = time.time() - start_time
                
                # Update metrics
                self._update_backend_metrics(backend_id, execution_time, success=True)
                
                return result
                
            except Exception as e:
                self.logger.warning(f"Task execution failed on attempt {attempt + 1}: {e}")
                
                # Update metrics for failure
                if 'backend_id' in locals():
                    self._update_backend_metrics(backend_id, 0, success=False)
                
                if attempt == retry_count:
                    raise
                
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
    
    def select_backend(self) -> Optional[str]:
        """
        Select the best backend based on the current strategy.
        
        Returns:
            Selected backend ID or None if no backends available
        """
        with self._lock:
            healthy_backends = [
                backend_id for backend_id, metrics in self.backend_metrics.items()
                if metrics.status in [BackendStatus.HEALTHY, BackendStatus.DEGRADED]
            ]
            
            if not healthy_backends:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._select_round_robin(healthy_backends)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._select_least_connections(healthy_backends)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
                return self._select_weighted_response_time(healthy_backends)
            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                return self._select_resource_based(healthy_backends)
            elif self.strategy == LoadBalancingStrategy.QUEUE_LENGTH:
                return self._select_queue_length(healthy_backends)
            else:  # ADAPTIVE
                return self._select_adaptive(healthy_backends)
    
    def get_backend_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all backends.
        
        Returns:
            Dictionary with backend status information
        """
        with self._lock:
            status = {}
            for backend_id, metrics in self.backend_metrics.items():
                status[backend_id] = {
                    'status': metrics.status.value,
                    'response_time': metrics.response_time,
                    'queue_length': metrics.queue_length,
                    'active_jobs': metrics.active_jobs,
                    'throughput': metrics.throughput,
                    'error_rate': metrics.error_rate,
                    'health_score': metrics.get_health_score(),
                    'load_score': metrics.get_load_score(),
                    'request_count': self.request_counts[backend_id],
                    'cost_per_hour': metrics.cost_per_hour,
                }
            
            return status
    
    def _select_round_robin(self, backends: List[str]) -> str:
        """Round-robin backend selection."""
        backend = backends[self.current_backend_index % len(backends)]
        self.current_backend_index += 1
        return backend
    
    def _select_least_connections(self, backends: List[str]) -> str:
        """Select backend with least active connections."""
        return min(backends, key=lambda b: self.backend_metrics[b].active_jobs)
    
    def _select_weighted_response_time(self, backends: List[str]) -> str:
        """Select backend based on weighted response time."""
        def score_function(backend_id):
            metrics = self.backend_metrics[backend_id]
            weight = self.backend_weights[backend_id]
            # Lower response time and higher weight = better score
            return metrics.response_time / weight if weight > 0 else float('inf')
        
        return min(backends, key=score_function)
    
    def _select_resource_based(self, backends: List[str]) -> str:
        """Select backend based on resource utilization."""
        def score_function(backend_id):
            metrics = self.backend_metrics[backend_id]
            # Combine CPU and memory utilization
            return (metrics.cpu_utilization + metrics.memory_utilization) / 2
        
        return min(backends, key=score_function)
    
    def _select_queue_length(self, backends: List[str]) -> str:
        """Select backend with shortest queue."""
        return min(backends, key=lambda b: self.backend_metrics[b].queue_length)
    
    def _select_adaptive(self, backends: List[str]) -> str:
        """Adaptive backend selection based on multiple factors."""
        def score_function(backend_id):
            metrics = self.backend_metrics[backend_id]
            weight = self.backend_weights[backend_id]
            
            # Multi-factor scoring
            health_score = metrics.get_health_score()
            load_score = 1.0 - metrics.get_load_score()  # Invert so lower load = higher score
            weight_score = weight
            
            # Cost factor (if cost optimization enabled)
            cost_score = 1.0
            if self.config.enable_cost_optimization and metrics.cost_per_hour > 0:
                max_cost = max(m.cost_per_hour for m in self.backend_metrics.values())
                cost_score = 1.0 - (metrics.cost_per_hour / max_cost) if max_cost > 0 else 1.0
            
            # Weighted combination
            total_score = (
                health_score * 0.3 +
                load_score * 0.3 +
                weight_score * 0.2 +
                cost_score * 0.2
            )
            
            return -total_score  # Negative because min() selects lowest
        
        return min(backends, key=score_function)
    
    def _update_backend_metrics(
        self,
        backend_id: str,
        execution_time: float,
        success: bool
    ) -> None:
        """Update backend metrics after task execution."""
        if backend_id not in self.backend_metrics:
            return
        
        metrics = self.backend_metrics[backend_id]
        
        # Update request count
        self.request_counts[backend_id] += 1
        
        # Update metrics
        if success:
            metrics.update_metrics(response_time=execution_time)
        else:
            # Update error rate
            total_requests = self.request_counts[backend_id]
            current_errors = metrics.error_rate * (total_requests - 1)
            new_error_rate = (current_errors + 1) / total_requests
            metrics.update_metrics(error_rate=new_error_rate)
        
        # Update backend status based on metrics
        self._update_backend_status(backend_id)
    
    def _update_backend_status(self, backend_id: str) -> None:
        """Update backend status based on current metrics."""
        metrics = self.backend_metrics[backend_id]
        
        # Determine status based on thresholds
        if metrics.error_rate > self.config.max_error_rate:
            metrics.status = BackendStatus.FAILED
        elif (metrics.response_time > self.config.max_response_time or
              metrics.queue_length > self.config.max_queue_length):
            metrics.status = BackendStatus.OVERLOADED
        elif metrics.get_health_score() < 0.5:
            metrics.status = BackendStatus.DEGRADED
        else:
            metrics.status = BackendStatus.HEALTHY
    
    def _start_monitoring(self) -> None:
        """Start backend monitoring thread."""
        def monitor_loop():
            while self._monitoring_active:
                try:
                    self._health_check_all_backends()
                    time.sleep(self.config.health_check_interval)
                except Exception as e:
                    self.logger.error(f"Backend monitoring error: {e}")
                    time.sleep(self.config.health_check_interval)
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("Backend monitoring started")
    
    def _health_check_all_backends(self) -> None:
        """Perform health checks on all backends."""
        with self._lock:
            for backend_id, backend in self.backends.items():
                try:
                    # Simple health check - try to get backend info
                    if hasattr(backend, 'status'):
                        status = backend.status()
                        # Update metrics based on backend status
                        metrics = self.backend_metrics[backend_id]
                        if hasattr(status, 'pending_jobs'):
                            metrics.update_metrics(queue_length=status.pending_jobs)
                        
                except Exception as e:
                    self.logger.warning(f"Health check failed for {backend_id}: {e}")
                    self.backend_metrics[backend_id].status = BackendStatus.FAILED


class AutoScaler:
    """
    Auto-scaler for dynamic resource allocation.
    
    This class provides automatic scaling of quantum computing resources
    based on workload demands, cost constraints, and performance targets.
    
    Features:
    - Reactive and predictive scaling
    - Cost-aware scaling decisions
    - Integration with load balancer
    - Performance-based scaling triggers
    - Cooldown periods to prevent oscillation
    
    Example:
        >>> auto_scaler = AutoScaler(strategy=ScalingStrategy.HYBRID)
        >>> auto_scaler.set_load_balancer(load_balancer)
        >>> 
        >>> # Auto-scaler will monitor and scale resources automatically
        >>> auto_scaler.start()
    """
    
    def __init__(
        self,
        strategy: ScalingStrategy = ScalingStrategy.REACTIVE,
        config: Optional[ScalingConfig] = None
    ):
        """
        Initialize auto-scaler.
        
        Args:
            strategy: Scaling strategy
            config: Scaling configuration
        """
        self.strategy = strategy
        self.config = config or ScalingConfig()
        self.logger = get_logger()
        
        # Load balancer integration
        self.load_balancer: Optional[LoadBalancer] = None
        
        # Scaling history
        self.scale_events: List[Dict[str, Any]] = []
        self.utilization_history: deque = deque(maxlen=self.config.prediction_window)
        
        # Scaling state
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        self.current_backend_count = 0
        
        # Auto-scaling control
        self._scaling_active = False
        self._scaling_thread = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info(f"AutoScaler initialized with strategy: {strategy.value}")
    
    def set_load_balancer(self, load_balancer: LoadBalancer) -> None:
        """
        Set the load balancer to manage.
        
        Args:
            load_balancer: LoadBalancer instance to manage
        """
        self.load_balancer = load_balancer
        self.current_backend_count = len(load_balancer.backends)
        self.logger.info("Load balancer attached to auto-scaler")
    
    def start(self) -> None:
        """Start auto-scaling."""
        if self.load_balancer is None:
            raise RuntimeError("Load balancer must be set before starting auto-scaler")
        
        self._scaling_active = True
        self._scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self._scaling_thread.start()
        
        self.logger.info("Auto-scaling started")
    
    def stop(self) -> None:
        """Stop auto-scaling."""
        self._scaling_active = False
        if self._scaling_thread:
            self._scaling_thread.join(timeout=5.0)
        
        self.logger.info("Auto-scaling stopped")
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """
        Get current scaling metrics.
        
        Returns:
            Dictionary with scaling metrics
        """
        with self._lock:
            if not self.load_balancer:
                return {}
            
            # Calculate current utilization
            backend_status = self.load_balancer.get_backend_status()
            total_utilization = sum(
                status['load_score'] for status in backend_status.values()
            ) / len(backend_status) if backend_status else 0
            
            return {
                'current_backends': len(backend_status),
                'target_backends': self._calculate_target_backend_count(),
                'utilization': total_utilization,
                'target_utilization': self.config.target_utilization,
                'scale_up_threshold': self.config.scale_up_threshold,
                'scale_down_threshold': self.config.scale_down_threshold,
                'last_scale_up': self.last_scale_up,
                'last_scale_down': self.last_scale_down,
                'scale_events': len(self.scale_events),
                'cost_per_hour': self._calculate_current_cost(),
            }
    
    def _scaling_loop(self) -> None:
        """Main auto-scaling loop."""
        while self._scaling_active:
            try:
                self._evaluate_scaling_decision()
                time.sleep(30.0)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Auto-scaling error: {e}")
                time.sleep(30.0)
    
    def _evaluate_scaling_decision(self) -> None:
        """Evaluate and execute scaling decisions."""
        if not self.load_balancer:
            return
        
        current_time = time.time()
        backend_status = self.load_balancer.get_backend_status()
        
        if not backend_status:
            return
        
        # Calculate current utilization
        total_utilization = sum(
            status['load_score'] for status in backend_status.values()
        ) / len(backend_status)
        
        # Record utilization history
        self.utilization_history.append((current_time, total_utilization))
        
        # Determine scaling action
        scale_action = self._determine_scaling_action(total_utilization, current_time)
        
        if scale_action == "scale_up":
            self._scale_up()
        elif scale_action == "scale_down":
            self._scale_down()
    
    def _determine_scaling_action(self, utilization: float, current_time: float) -> Optional[str]:
        """Determine what scaling action to take."""
        current_backends = len(self.load_balancer.backends)
        
        # Check cooldown periods
        if (current_time - self.last_scale_up) < self.config.scale_up_cooldown:
            if utilization > self.config.scale_up_threshold:
                return None  # Still in cooldown
        
        if (current_time - self.last_scale_down) < self.config.scale_down_cooldown:
            if utilization < self.config.scale_down_threshold:
                return None  # Still in cooldown
        
        # Scaling decisions based on strategy
        if self.strategy == ScalingStrategy.REACTIVE:
            return self._reactive_scaling_decision(utilization, current_backends)
        elif self.strategy == ScalingStrategy.PREDICTIVE:
            return self._predictive_scaling_decision(utilization, current_backends)
        elif self.strategy == ScalingStrategy.PROACTIVE:
            return self._proactive_scaling_decision(utilization, current_backends)
        else:  # HYBRID
            return self._hybrid_scaling_decision(utilization, current_backends)
    
    def _reactive_scaling_decision(self, utilization: float, current_backends: int) -> Optional[str]:
        """Reactive scaling based on current utilization."""
        if utilization > self.config.scale_up_threshold and current_backends < self.config.max_backends:
            return "scale_up"
        elif utilization < self.config.scale_down_threshold and current_backends > self.config.min_backends:
            return "scale_down"
        return None
    
    def _predictive_scaling_decision(self, utilization: float, current_backends: int) -> Optional[str]:
        """Predictive scaling based on utilization trends."""
        if len(self.utilization_history) < 3:
            return self._reactive_scaling_decision(utilization, current_backends)
        
        # Calculate trend
        recent_utilizations = [u for _, u in list(self.utilization_history)[-5:]]
        trend = statistics.linear_regression([i for i in range(len(recent_utilizations))], recent_utilizations).slope
        
        # Predict future utilization
        predicted_utilization = utilization + trend * 3  # 3 time steps ahead
        
        if predicted_utilization > self.config.scale_up_threshold and current_backends < self.config.max_backends:
            return "scale_up"
        elif predicted_utilization < self.config.scale_down_threshold and current_backends > self.config.min_backends:
            return "scale_down"
        
        return None
    
    def _proactive_scaling_decision(self, utilization: float, current_backends: int) -> Optional[str]:
        """Proactive scaling based on expected workload patterns."""
        # This would typically integrate with workload scheduling systems
        # For now, implement conservative proactive scaling
        target_backends = self._calculate_target_backend_count()
        
        if target_backends > current_backends and current_backends < self.config.max_backends:
            return "scale_up"
        elif target_backends < current_backends and current_backends > self.config.min_backends:
            return "scale_down"
        
        return None
    
    def _hybrid_scaling_decision(self, utilization: float, current_backends: int) -> Optional[str]:
        """Hybrid scaling combining multiple strategies."""
        # Weight different strategies
        reactive_decision = self._reactive_scaling_decision(utilization, current_backends)
        predictive_decision = self._predictive_scaling_decision(utilization, current_backends)
        
        # Use reactive for urgent scaling, predictive for planned scaling
        if utilization > 0.9:  # Very high utilization - use reactive
            return reactive_decision
        else:
            return predictive_decision
    
    def _scale_up(self) -> None:
        """Scale up by adding backends."""
        with self._lock:
            current_time = time.time()
            current_backends = len(self.load_balancer.backends)
            
            if current_backends >= self.config.max_backends:
                return
            
            # Check cost constraints
            if self.config.enable_cost_optimization:
                projected_cost = self._calculate_projected_cost(current_backends + 1)
                if projected_cost > self.config.max_cost_per_hour:
                    self.logger.warning(f"Scale up blocked by cost constraint: ${projected_cost}/hour")
                    return
            
            # Add a new backend (this would typically involve creating/starting a new backend)
            new_backend_id = f"auto_backend_{current_backends + 1}"
            
            # In a real implementation, you would create/start a new backend here
            # For now, we'll log the scaling event
            self.logger.info(f"Scaling up: adding backend {new_backend_id}")
            
            # Record scaling event
            self.scale_events.append({
                'timestamp': current_time,
                'action': 'scale_up',
                'backend_count_before': current_backends,
                'backend_count_after': current_backends + 1,
                'trigger_utilization': self._get_current_utilization(),
            })
            
            self.last_scale_up = current_time
    
    def _scale_down(self) -> None:
        """Scale down by removing backends."""
        with self._lock:
            current_time = time.time()
            current_backends = len(self.load_balancer.backends)
            
            if current_backends <= self.config.min_backends:
                return
            
            # Find the least utilized backend to remove
            backend_status = self.load_balancer.get_backend_status()
            least_utilized_backend = min(
                backend_status.items(),
                key=lambda x: x[1]['load_score']
            )[0]
            
            # Remove the backend
            self.logger.info(f"Scaling down: removing backend {least_utilized_backend}")
            
            # In a real implementation, you would gracefully shutdown the backend here
            # self.load_balancer.remove_backend(least_utilized_backend)
            
            # Record scaling event
            self.scale_events.append({
                'timestamp': current_time,
                'action': 'scale_down',
                'backend_count_before': current_backends,
                'backend_count_after': current_backends - 1,
                'trigger_utilization': self._get_current_utilization(),
                'removed_backend': least_utilized_backend,
            })
            
            self.last_scale_down = current_time
    
    def _calculate_target_backend_count(self) -> int:
        """Calculate target number of backends."""
        if not self.load_balancer:
            return self.config.min_backends
        
        backend_status = self.load_balancer.get_backend_status()
        if not backend_status:
            return self.config.min_backends
        
        # Calculate based on current utilization and target
        current_utilization = sum(
            status['load_score'] for status in backend_status.values()
        ) / len(backend_status)
        
        if current_utilization > 0:
            target_backends = int(
                len(backend_status) * current_utilization / self.config.target_utilization
            )
        else:
            target_backends = self.config.min_backends
        
        return max(self.config.min_backends, min(self.config.max_backends, target_backends))
    
    def _calculate_current_cost(self) -> float:
        """Calculate current cost per hour."""
        if not self.load_balancer:
            return 0.0
        
        backend_status = self.load_balancer.get_backend_status()
        total_cost = sum(status['cost_per_hour'] for status in backend_status.values())
        
        return total_cost
    
    def _calculate_projected_cost(self, backend_count: int) -> float:
        """Calculate projected cost for given backend count."""
        if not self.load_balancer:
            return 0.0
        
        backend_status = self.load_balancer.get_backend_status()
        if not backend_status:
            return 0.0
        
        avg_cost_per_backend = sum(
            status['cost_per_hour'] for status in backend_status.values()
        ) / len(backend_status)
        
        return backend_count * avg_cost_per_backend
    
    def _get_current_utilization(self) -> float:
        """Get current system utilization."""
        if not self.load_balancer:
            return 0.0
        
        backend_status = self.load_balancer.get_backend_status()
        if not backend_status:
            return 0.0
        
        return sum(status['load_score'] for status in backend_status.values()) / len(backend_status)


def create_load_balancer(
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
    **config_kwargs
) -> LoadBalancer:
    """
    Create a load balancer with specified configuration.
    
    Args:
        strategy: Load balancing strategy
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured LoadBalancer instance
    """
    config = ScalingConfig(**config_kwargs)
    return LoadBalancer(strategy=strategy, config=config)


def create_auto_scaler(
    strategy: ScalingStrategy = ScalingStrategy.REACTIVE,
    **config_kwargs
) -> AutoScaler:
    """
    Create an auto-scaler with specified configuration.
    
    Args:
        strategy: Scaling strategy
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured AutoScaler instance
    """
    config = ScalingConfig(**config_kwargs)
    return AutoScaler(strategy=strategy, config=config)