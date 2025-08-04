"""
Multi-backend orchestration system for quantum error mitigation.

This module provides comprehensive orchestration across multiple quantum backends
including calibration-aware scheduling, cross-backend benchmarking, fallback
strategies, and performance optimization.
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

from ..security import SecureConfig
from ..monitoring import MetricsCollector


logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Types of quantum backends."""
    SIMULATOR = "simulator"
    HARDWARE = "hardware"
    HYBRID = "hybrid"
    CLOUD = "cloud"


class CalibrationStatus(Enum):
    """Backend calibration status."""
    FRESH = "fresh"          # Recently calibrated
    GOOD = "good"            # Good calibration
    STALE = "stale"          # Needs recalibration
    UNKNOWN = "unknown"      # Status unknown
    CALIBRATING = "calibrating"  # Currently calibrating


class BackendHealth(Enum):
    """Backend health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


@dataclass
class QuantumBackendInfo:
    """Comprehensive quantum backend information."""
    id: str
    name: str
    provider: str
    backend_type: BackendType
    
    # Hardware specifications
    num_qubits: int
    coupling_map: List[Tuple[int, int]] = field(default_factory=list)
    gate_set: List[str] = field(default_factory=list)
    instruction_durations: Dict[str, float] = field(default_factory=dict)
    
    # Performance characteristics
    gate_fidelities: Dict[str, float] = field(default_factory=dict)
    readout_fidelities: List[float] = field(default_factory=list)
    t1_times: List[float] = field(default_factory=list)  # Relaxation times
    t2_times: List[float] = field(default_factory=list)  # Dephasing times
    
    # Calibration information
    calibration_status: CalibrationStatus = CalibrationStatus.UNKNOWN
    last_calibration_time: Optional[float] = None
    calibration_data: Dict[str, Any] = field(default_factory=dict)
    
    # Current status
    health_status: BackendHealth = BackendHealth.HEALTHY
    queue_length: int = 0
    estimated_wait_time: float = 0.0
    
    # Performance metrics
    success_rate: float = 1.0
    average_execution_time: float = 60.0
    cost_per_shot: float = 0.0
    availability_score: float = 1.0
    
    # Benchmarking results
    quantum_volume: Optional[int] = None
    randomized_benchmarking_fidelity: Optional[float] = None
    cross_talk_matrix: Optional[List[List[float]]] = None
    
    def is_available(self) -> bool:
        """Check if backend is available."""
        return self.health_status in [BackendHealth.HEALTHY, BackendHealth.DEGRADED]
    
    def overall_fidelity(self) -> float:
        """Calculate overall backend fidelity."""
        if not self.gate_fidelities or not self.readout_fidelities:
            return 0.8  # Default estimate
        
        avg_gate_fidelity = np.mean(list(self.gate_fidelities.values()))
        avg_readout_fidelity = np.mean(self.readout_fidelities)
        
        return (avg_gate_fidelity + avg_readout_fidelity) / 2
    
    def calibration_freshness(self) -> float:
        """Calculate calibration freshness score (0-1)."""
        if not self.last_calibration_time:
            return 0.0
        
        hours_since_calibration = (time.time() - self.last_calibration_time) / 3600
        
        # Calibration is fresh for first 8 hours, degrades linearly to 0 at 48 hours
        if hours_since_calibration <= 8:
            return 1.0
        elif hours_since_calibration <= 48:
            return 1.0 - (hours_since_calibration - 8) / 40
        else:
            return 0.0
    
    def performance_score(self) -> float:
        """Calculate overall performance score."""
        fidelity_score = self.overall_fidelity() * 0.4
        calibration_score = self.calibration_freshness() * 0.3
        availability_score = self.availability_score * 0.2
        success_score = self.success_rate * 0.1
        
        return fidelity_score + calibration_score + availability_score + success_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "backend_type": self.backend_type.value,
            "num_qubits": self.num_qubits,
            "coupling_map": self.coupling_map,
            "gate_set": self.gate_set,
            "calibration_status": self.calibration_status.value,
            "health_status": self.health_status.value,
            "queue_length": self.queue_length,
            "estimated_wait_time": self.estimated_wait_time,
            "success_rate": self.success_rate,
            "cost_per_shot": self.cost_per_shot,
            "overall_fidelity": self.overall_fidelity(),
            "calibration_freshness": self.calibration_freshness(),
            "performance_score": self.performance_score(),
            "is_available": self.is_available(),
            "quantum_volume": self.quantum_volume
        }


@dataclass
class OrchestrationJob:
    """Job for multi-backend orchestration."""
    id: str
    circuit_description: Dict[str, Any]
    requirements: Dict[str, Any]
    
    # Job configuration
    shots: int = 1024
    priority: int = 1
    deadline: Optional[float] = None
    
    # Backend preferences
    preferred_backends: List[str] = field(default_factory=list)
    excluded_backends: List[str] = field(default_factory=list)
    min_fidelity: float = 0.8
    max_cost_per_shot: Optional[float] = None
    
    # Orchestration strategy
    enable_fallback: bool = True
    enable_cross_validation: bool = False
    redundancy_level: int = 1  # Number of backends to run on
    
    # Execution state
    assigned_backends: List[str] = field(default_factory=list)
    submitted_time: float = field(default_factory=time.time)
    results: Dict[str, Any] = field(default_factory=dict)
    
    def age(self) -> float:
        """Get job age in seconds."""
        return time.time() - self.submitted_time
    
    def is_expired(self) -> bool:
        """Check if job has passed deadline."""
        return self.deadline is not None and time.time() > self.deadline
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "shots": self.shots,
            "priority": self.priority,
            "deadline": self.deadline,
            "preferred_backends": self.preferred_backends,
            "excluded_backends": self.excluded_backends,
            "min_fidelity": self.min_fidelity,
            "max_cost_per_shot": self.max_cost_per_shot,
            "enable_fallback": self.enable_fallback,
            "enable_cross_validation": self.enable_cross_validation,
            "redundancy_level": self.redundancy_level,
            "assigned_backends": self.assigned_backends,
            "age": self.age(),
            "is_expired": self.is_expired()
        }


class CalibrationAwareScheduler:
    """
    Scheduler that considers backend calibration status in decisions.
    
    Features:
    - Calibration freshness scoring
    - Automatic recalibration scheduling
    - Quality-based backend selection
    - Calibration drift prediction
    """
    
    def __init__(self, config: Optional[SecureConfig] = None):
        self.config = config or SecureConfig()
        
        # Calibration thresholds
        self.fresh_calibration_hours = 8
        self.stale_calibration_hours = 48
        self.min_calibration_score = 0.3
        
        # Scheduling policies
        self.prefer_fresh_calibration = True
        self.calibration_weight = 0.4  # Weight in selection algorithm
        
        logger.info("CalibrationAwareScheduler initialized")
    
    def select_backend(
        self,
        job: OrchestrationJob,
        available_backends: List[QuantumBackendInfo]
    ) -> Optional[QuantumBackendInfo]:
        """Select best backend considering calibration status."""
        
        # Filter suitable backends
        suitable_backends = []
        for backend in available_backends:
            if self._backend_meets_requirements(backend, job):
                suitable_backends.append(backend)
        
        if not suitable_backends:
            return None
        
        # Score backends
        scored_backends = []
        for backend in suitable_backends:
            score = self._calculate_calibration_aware_score(backend, job)
            scored_backends.append((score, backend))
        
        # Sort by score and return best
        scored_backends.sort(key=lambda x: x[0], reverse=True)
        return scored_backends[0][1]
    
    def _backend_meets_requirements(
        self,
        backend: QuantumBackendInfo,
        job: OrchestrationJob
    ) -> bool:
        """Check if backend meets job requirements."""
        # Basic availability check
        if not backend.is_available():
            return False
        
        # Check exclusions
        if backend.id in job.excluded_backends:
            return False
        
        # Check minimum fidelity
        if backend.overall_fidelity() < job.min_fidelity:
            return False
        
        # Check cost constraints
        if (job.max_cost_per_shot and 
            backend.cost_per_shot > job.max_cost_per_shot):
            return False
        
        # Check calibration minimum
        if backend.calibration_freshness() < self.min_calibration_score:
            return False
        
        return True
    
    def _calculate_calibration_aware_score(
        self,
        backend: QuantumBackendInfo,
        job: OrchestrationJob
    ) -> float:
        """Calculate score with calibration awareness."""
        score = 0.0
        
        # Calibration score (40% weight)
        calibration_score = backend.calibration_freshness() * 40
        score += calibration_score
        
        # Performance score (30% weight)
        performance_score = backend.performance_score() * 30
        score += performance_score
        
        # Queue/availability score (20% weight)
        queue_penalty = min(backend.estimated_wait_time / 300.0, 1.0)  # Max 5 min penalty
        availability_score = (1.0 - queue_penalty) * 20
        score += availability_score
        
        # Preference bonus (10% weight)
        if backend.id in job.preferred_backends:
            score += 10
        
        return score
    
    def schedule_calibration(
        self,
        backends: List[QuantumBackendInfo]
    ) -> List[str]:
        """Schedule backends that need calibration."""
        calibration_needed = []
        
        for backend in backends:
            if backend.calibration_status == CalibrationStatus.STALE:
                calibration_needed.append(backend.id)
            elif backend.calibration_freshness() < 0.3:
                calibration_needed.append(backend.id)
        
        # Sort by urgency (lowest freshness first)
        calibration_needed.sort(
            key=lambda bid: next(
                b.calibration_freshness() for b in backends if b.id == bid
            )
        )
        
        return calibration_needed
    
    def predict_calibration_drift(
        self,
        backend: QuantumBackendInfo,
        hours_ahead: int = 24
    ) -> float:
        """Predict calibration quality after specified hours."""
        current_freshness = backend.calibration_freshness()
        
        if not backend.last_calibration_time:
            return 0.1  # Very low if never calibrated
        
        # Simple linear degradation model
        hours_since_calibration = (time.time() - backend.last_calibration_time) / 3600
        total_hours = hours_since_calibration + hours_ahead
        
        if total_hours <= 8:
            return 1.0
        elif total_hours <= 48:
            return 1.0 - (total_hours - 8) / 40
        else:
            return 0.0


class CrossBackendBenchmarker:
    """
    Cross-backend benchmarking and performance comparison system.
    
    Features:
    - Standardized benchmark circuits
    - Performance comparison across backends
    - Quality drift detection
    - Benchmark result analytics
    """
    
    def __init__(self, config: Optional[SecureConfig] = None):
        self.config = config or SecureConfig()
        
        # Benchmark configuration
        self.benchmark_circuits = [
            "quantum_volume",
            "randomized_benchmarking", 
            "process_tomography",
            "gate_set_tomography",
            "cross_talk_characterization"
        ]
        
        # Benchmarking results storage
        self.benchmark_results: Dict[str, List[Dict[str, Any]]] = {}
        self.comparison_reports: List[Dict[str, Any]] = []
        
        # Benchmarking schedule
        self.benchmark_interval_hours = 24
        self.last_benchmark_time: Dict[str, float] = {}
        
        logger.info("CrossBackendBenchmarker initialized")
    
    async def run_benchmark_suite(
        self,
        backends: List[QuantumBackendInfo],
        circuits: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark suite across backends."""
        circuits = circuits or self.benchmark_circuits
        
        benchmark_results = {}
        
        for circuit_type in circuits:
            logger.info(f"Running {circuit_type} benchmark")
            
            circuit_results = {}
            for backend in backends:
                if backend.is_available():
                    result = await self._run_single_benchmark(backend, circuit_type)
                    circuit_results[backend.id] = result
            
            benchmark_results[circuit_type] = circuit_results
        
        # Store results
        timestamp = time.time()
        for backend_id in [b.id for b in backends]:
            if backend_id not in self.benchmark_results:
                self.benchmark_results[backend_id] = []
            
            self.benchmark_results[backend_id].append({
                "timestamp": timestamp,
                "results": {
                    circuit: benchmark_results[circuit].get(backend_id, {})
                    for circuit in circuits
                }
            })
        
        # Generate comparison report
        comparison_report = self._generate_comparison_report(benchmark_results)
        self.comparison_reports.append(comparison_report)
        
        return {
            "benchmark_results": benchmark_results,
            "comparison_report": comparison_report,
            "timestamp": timestamp
        }
    
    async def _run_single_benchmark(
        self,
        backend: QuantumBackendInfo,
        circuit_type: str
    ) -> Dict[str, Any]:
        """Run a single benchmark on a backend."""
        # Simulate benchmark execution
        await asyncio.sleep(0.1)  # Simulate execution time
        
        if circuit_type == "quantum_volume":
            return await self._run_quantum_volume_benchmark(backend)
        elif circuit_type == "randomized_benchmarking":
            return await self._run_randomized_benchmarking(backend)
        elif circuit_type == "process_tomography":
            return await self._run_process_tomography(backend)
        elif circuit_type == "gate_set_tomography":
            return await self._run_gate_set_tomography(backend)
        elif circuit_type == "cross_talk_characterization":
            return await self._run_crosstalk_characterization(backend)
        else:
            return {"error": f"Unknown benchmark type: {circuit_type}"}
    
    async def _run_quantum_volume_benchmark(
        self,
        backend: QuantumBackendInfo
    ) -> Dict[str, Any]:
        """Run quantum volume benchmark."""
        # Simulate quantum volume calculation
        max_qubits = min(backend.num_qubits, 8)  # Practical limit
        
        # Estimate quantum volume based on backend characteristics
        fidelity_factor = backend.overall_fidelity()
        connectivity_factor = len(backend.coupling_map) / (backend.num_qubits * (backend.num_qubits - 1) / 2)
        
        base_qv = max_qubits ** 2
        adjusted_qv = int(base_qv * fidelity_factor * connectivity_factor)
        
        return {
            "quantum_volume": adjusted_qv,
            "qubits_used": max_qubits,
            "success_rate": fidelity_factor * 0.9,  # Slight penalty
            "execution_time": np.random.uniform(30, 180),  # seconds
            "confidence": 0.95
        }
    
    async def _run_randomized_benchmarking(
        self,
        backend: QuantumBackendInfo
    ) -> Dict[str, Any]:
        """Run randomized benchmarking."""
        # Simulate RB results
        base_fidelity = backend.overall_fidelity()
        
        # Add some noise to simulate real measurement
        measured_fidelity = base_fidelity * np.random.uniform(0.95, 1.05)
        measured_fidelity = max(0, min(1, measured_fidelity))
        
        return {
            "gate_fidelity": measured_fidelity,
            "error_per_gate": 1 - measured_fidelity,
            "clifford_fidelity": measured_fidelity * 0.98,
            "sequence_lengths": [1, 2, 4, 8, 16, 32, 64, 128],
            "decay_constant": -np.log(measured_fidelity),
            "r_squared": np.random.uniform(0.95, 0.99)
        }
    
    async def _run_process_tomography(
        self,
        backend: QuantumBackendInfo
    ) -> Dict[str, Any]:
        """Run process tomography benchmark."""
        # Simulate process tomography results
        process_fidelity = backend.overall_fidelity() * np.random.uniform(0.9, 1.0)
        
        return {
            "process_fidelity": process_fidelity,
            "gate_fidelities": {
                gate: backend.gate_fidelities.get(gate, 0.95) * np.random.uniform(0.95, 1.05)
                for gate in backend.gate_set[:5]  # Test first 5 gates
            },
            "coherence_time_t1": np.mean(backend.t1_times) if backend.t1_times else 50.0,
            "coherence_time_t2": np.mean(backend.t2_times) if backend.t2_times else 25.0
        }
    
    async def _run_gate_set_tomography(
        self,
        backend: QuantumBackendInfo
    ) -> Dict[str, Any]:
        """Run gate set tomography benchmark."""
        # Simulate GST results
        return {
            "gate_set_fidelity": backend.overall_fidelity() * np.random.uniform(0.95, 1.0),
            "spam_error": np.random.uniform(0.01, 0.05),
            "gate_errors": {
                gate: np.random.uniform(0.001, 0.01)
                for gate in backend.gate_set
            },
            "crosstalk_strength": np.random.uniform(0.01, 0.1)
        }
    
    async def _run_crosstalk_characterization(
        self,
        backend: QuantumBackendInfo
    ) -> Dict[str, Any]:
        """Run crosstalk characterization."""
        # Simulate crosstalk measurement
        n_qubits = min(backend.num_qubits, 5)  # Limit for practical measurement
        
        # Generate random crosstalk matrix
        crosstalk_matrix = np.random.uniform(0, 0.1, (n_qubits, n_qubits))
        np.fill_diagonal(crosstalk_matrix, 0)  # No self-crosstalk
        
        return {
            "crosstalk_matrix": crosstalk_matrix.tolist(),
            "max_crosstalk": float(np.max(crosstalk_matrix)),
            "avg_crosstalk": float(np.mean(crosstalk_matrix)),
            "qubits_measured": n_qubits,
            "measurement_fidelity": backend.overall_fidelity()
        }
    
    def _generate_comparison_report(
        self,
        benchmark_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comparison report across backends."""
        report = {
            "timestamp": time.time(),
            "backends_compared": list(next(iter(benchmark_results.values())).keys()),
            "benchmarks_run": list(benchmark_results.keys()),
            "rankings": {},
            "insights": []
        }
        
        # Rank backends for each benchmark
        for benchmark_type, backend_results in benchmark_results.items():
            rankings = self._rank_backends_for_benchmark(benchmark_type, backend_results)
            report["rankings"][benchmark_type] = rankings
        
        # Generate insights
        report["insights"] = self._generate_benchmark_insights(benchmark_results)
        
        return report
    
    def _rank_backends_for_benchmark(
        self,
        benchmark_type: str,
        backend_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rank backends for a specific benchmark."""
        rankings = []
        
        for backend_id, result in backend_results.items():
            score = self._calculate_benchmark_score(benchmark_type, result)
            rankings.append({
                "backend_id": backend_id,
                "score": score,
                "key_metrics": self._extract_key_metrics(benchmark_type, result)
            })
        
        # Sort by score (highest first)
        rankings.sort(key=lambda x: x["score"], reverse=True)
        
        # Add ranks
        for i, ranking in enumerate(rankings):
            ranking["rank"] = i + 1
        
        return rankings
    
    def _calculate_benchmark_score(
        self,
        benchmark_type: str,
        result: Dict[str, Any]
    ) -> float:
        """Calculate overall score for benchmark result."""
        if "error" in result:
            return 0.0
        
        if benchmark_type == "quantum_volume":
            return result.get("quantum_volume", 0) * result.get("success_rate", 0)
        
        elif benchmark_type == "randomized_benchmarking":
            return result.get("gate_fidelity", 0) * 100
        
        elif benchmark_type == "process_tomography":
            return result.get("process_fidelity", 0) * 100
        
        elif benchmark_type == "gate_set_tomography":
            return result.get("gate_set_fidelity", 0) * 100
        
        elif benchmark_type == "cross_talk_characterization":
            max_crosstalk = result.get("max_crosstalk", 1.0)
            return max(0, 100 * (1 - max_crosstalk))  # Lower crosstalk = higher score
        
        return 0.0
    
    def _extract_key_metrics(
        self,
        benchmark_type: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract key metrics for benchmark type."""
        if benchmark_type == "quantum_volume":
            return {
                "quantum_volume": result.get("quantum_volume"),
                "success_rate": result.get("success_rate")
            }
        elif benchmark_type == "randomized_benchmarking":
            return {
                "gate_fidelity": result.get("gate_fidelity"),
                "error_per_gate": result.get("error_per_gate")
            }
        else:
            return {}
    
    def _generate_benchmark_insights(
        self,
        benchmark_results: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate insights from benchmark results."""
        insights = []
        
        # Find best performing backend overall
        backend_scores = {}
        for benchmark_type, backend_results in benchmark_results.items():
            for backend_id, result in backend_results.items():
                score = self._calculate_benchmark_score(benchmark_type, result)
                if backend_id not in backend_scores:
                    backend_scores[backend_id] = []
                backend_scores[backend_id].append(score)
        
        # Calculate average scores
        avg_scores = {
            backend_id: np.mean(scores)
            for backend_id, scores in backend_scores.items()
        }
        
        if avg_scores:
            best_backend = max(avg_scores.keys(), key=lambda k: avg_scores[k])
            insights.append(f"Overall best performing backend: {best_backend}")
            
            worst_backend = min(avg_scores.keys(), key=lambda k: avg_scores[k])
            insights.append(f"Lowest performing backend: {worst_backend}")
        
        return insights
    
    def get_backend_performance_history(
        self,
        backend_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get performance history for a backend."""
        if backend_id not in self.benchmark_results:
            return {"error": "No benchmark data for backend"}
        
        cutoff_time = time.time() - days * 24 * 3600
        recent_results = [
            result for result in self.benchmark_results[backend_id]
            if result["timestamp"] > cutoff_time
        ]
        
        if not recent_results:
            return {"error": "No recent benchmark data"}
        
        # Analyze trends
        trends = {}
        for benchmark_type in self.benchmark_circuits:
            scores = []
            timestamps = []
            
            for result in recent_results:
                if benchmark_type in result["results"]:
                    score = self._calculate_benchmark_score(
                        benchmark_type, 
                        result["results"][benchmark_type]
                    )
                    scores.append(score)
                    timestamps.append(result["timestamp"])
            
            if len(scores) >= 2:
                # Calculate trend
                trend_slope = np.polyfit(timestamps, scores, 1)[0]
                trends[benchmark_type] = {
                    "scores": scores,
                    "trend": "improving" if trend_slope > 0 else "degrading",
                    "slope": trend_slope
                }
        
        return {
            "backend_id": backend_id,
            "analysis_period_days": days,
            "total_benchmarks": len(recent_results),
            "trends": trends,
            "latest_scores": {
                benchmark_type: self._calculate_benchmark_score(
                    benchmark_type, 
                    recent_results[-1]["results"].get(benchmark_type, {})
                )
                for benchmark_type in self.benchmark_circuits
                if benchmark_type in recent_results[-1]["results"]
            } if recent_results else {}
        }


class FallbackStrategy:
    """
    Fallback strategy implementation for backend failures.
    
    Features:
    - Automatic backend selection for failed jobs
    - Quality-based fallback ordering
    - Circuit adaptation for different backends
    - Performance impact minimization
    """
    
    def __init__(self, config: Optional[SecureConfig] = None):
        self.config = config or SecureConfig()
        
        # Fallback configuration
        self.max_fallback_attempts = 3
        self.fallback_timeout_seconds = 300  # 5 minutes
        self.quality_degradation_threshold = 0.2  # 20% max quality loss
        
        # Fallback statistics
        self.fallback_events: List[Dict[str, Any]] = []
        self.success_rates: Dict[str, float] = {}
        
        logger.info("FallbackStrategy initialized")
    
    async def execute_with_fallback(
        self,
        job: OrchestrationJob,
        primary_backend: QuantumBackendInfo,
        fallback_backends: List[QuantumBackendInfo]
    ) -> Dict[str, Any]:
        """Execute job with fallback strategy."""
        execution_attempts = []
        
        # Try primary backend first
        logger.info(f"Attempting execution on primary backend: {primary_backend.id}")
        
        try:
            result = await self._execute_on_backend(job, primary_backend)
            
            if self._is_successful_result(result):
                return {
                    "success": True,
                    "result": result,
                    "backend_used": primary_backend.id,
                    "fallback_used": False,
                    "attempts": 1
                }
            else:
                execution_attempts.append({
                    "backend_id": primary_backend.id,
                    "success": False,
                    "error": result.get("error", "Unknown error")
                })
        
        except Exception as e:
            execution_attempts.append({
                "backend_id": primary_backend.id,  
                "success": False,
                "error": str(e)
            })
        
        # Try fallback backends
        for i, fallback_backend in enumerate(fallback_backends[:self.max_fallback_attempts]):
            logger.info(f"Attempting fallback {i+1}: {fallback_backend.id}")
            
            try:
                # Check if quality degradation is acceptable
                if not self._is_acceptable_fallback(primary_backend, fallback_backend):
                    logger.warning(f"Fallback {fallback_backend.id} quality too low")
                    continue
                
                # Adapt job for fallback backend if needed
                adapted_job = await self._adapt_job_for_backend(job, fallback_backend)
                
                result = await self._execute_on_backend(adapted_job, fallback_backend)
                
                if self._is_successful_result(result):
                    # Record successful fallback
                    self._record_fallback_event(job.id, primary_backend.id, fallback_backend.id, True)
                    
                    return {
                        "success": True,
                        "result": result,
                        "backend_used": fallback_backend.id,
                        "fallback_used": True,
                        "primary_backend": primary_backend.id,
                        "attempts": len(execution_attempts) + 1,
                        "execution_attempts": execution_attempts
                    }
                else:
                    execution_attempts.append({
                        "backend_id": fallback_backend.id,
                        "success": False,
                        "error": result.get("error", "Unknown error")
                    })
            
            except Exception as e:
                execution_attempts.append({
                    "backend_id": fallback_backend.id,
                    "success": False,
                    "error": str(e)
                })
        
        # All attempts failed
        self._record_fallback_event(job.id, primary_backend.id, None, False)
        
        return {
            "success": False,
            "error": "All execution attempts failed",
            "backend_used": None,
            "fallback_used": True,
            "attempts": len(execution_attempts),
            "execution_attempts": execution_attempts
        }
    
    async def _execute_on_backend(
        self,
        job: OrchestrationJob,
        backend: QuantumBackendInfo
    ) -> Dict[str, Any]:
        """Execute job on specific backend (simulated)."""
        # Simulate execution
        await asyncio.sleep(np.random.uniform(0.1, 1.0))
        
        # Simulate success/failure based on backend reliability
        if np.random.random() < backend.success_rate:
            return {
                "counts": {"00": 500, "11": 524},  # Simulated results
                "execution_time": np.random.uniform(30, 120),
                "backend_id": backend.id,
                "shots": job.shots
            }
        else:
            return {
                "error": f"Execution failed on {backend.id}",
                "error_code": "BACKEND_ERROR"
            }
    
    def _is_successful_result(self, result: Dict[str, Any]) -> bool:
        """Check if execution result is successful."""
        return "error" not in result and "counts" in result
    
    def _is_acceptable_fallback(
        self,
        primary_backend: QuantumBackendInfo,
        fallback_backend: QuantumBackendInfo
    ) -> bool:
        """Check if fallback backend is acceptable."""
        primary_quality = primary_backend.performance_score()
        fallback_quality = fallback_backend.performance_score()
        
        quality_degradation = (primary_quality - fallback_quality) / primary_quality
        
        return quality_degradation <= self.quality_degradation_threshold
    
    async def _adapt_job_for_backend(
        self,
        job: OrchestrationJob,
        backend: QuantumBackendInfo
    ) -> OrchestrationJob:
        """Adapt job configuration for different backend."""
        # Create adapted job copy
        adapted_job = OrchestrationJob(
            id=f"{job.id}_adapted_{backend.id}",
            circuit_description=job.circuit_description.copy(),
            requirements=job.requirements.copy(),
            shots=job.shots,
            priority=job.priority,
            deadline=job.deadline
        )
        
        # Adapt shots if backend has limitations
        if hasattr(backend, 'max_shots'):
            adapted_job.shots = min(job.shots, getattr(backend, 'max_shots', job.shots))
        
        # Could add circuit transpilation/adaptation here
        
        return adapted_job
    
    def _record_fallback_event(
        self,
        job_id: str,
        primary_backend_id: str,
        fallback_backend_id: Optional[str],
        success: bool
    ) -> None:
        """Record fallback event for statistics."""
        event = {
            "timestamp": time.time(),
            "job_id": job_id,
            "primary_backend": primary_backend_id,
            "fallback_backend": fallback_backend_id,
            "success": success
        }
        
        self.fallback_events.append(event)
        
        # Update success rates
        if fallback_backend_id:
            if fallback_backend_id not in self.success_rates:
                self.success_rates[fallback_backend_id] = []
            self.success_rates[fallback_backend_id].append(1.0 if success else 0.0)
    
    def get_fallback_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get fallback usage statistics."""
        cutoff_time = time.time() - days * 24 * 3600
        recent_events = [
            event for event in self.fallback_events
            if event["timestamp"] > cutoff_time
        ]
        
        if not recent_events:
            return {"message": "No recent fallback events"}
        
        # Calculate statistics
        total_events = len(recent_events)
        successful_fallbacks = len([e for e in recent_events if e["success"]])
        
        # Backend usage statistics
        backend_usage = {}
        for event in recent_events:
            primary = event["primary_backend"]
            fallback = event["fallback_backend"]
            
            if primary not in backend_usage:
                backend_usage[primary] = {"as_primary": 0, "as_fallback": 0, "failures": 0}
            backend_usage[primary]["as_primary"] += 1
            
            if not event["success"]:
                backend_usage[primary]["failures"] += 1
            
            if fallback:
                if fallback not in backend_usage:
                    backend_usage[fallback] = {"as_primary": 0, "as_fallback": 0, "failures": 0}
                backend_usage[fallback]["as_fallback"] += 1
        
        return {
            "analysis_period_days": days,
            "total_fallback_events": total_events,
            "successful_fallbacks": successful_fallbacks,
            "fallback_success_rate": successful_fallbacks / total_events if total_events > 0 else 0,
            "backend_usage": backend_usage,
            "most_failed_primary": max(
                backend_usage.keys(),
                key=lambda k: backend_usage[k]["failures"]
            ) if backend_usage else None,
            "most_used_fallback": max(
                backend_usage.keys(),
                key=lambda k: backend_usage[k]["as_fallback"]  
            ) if backend_usage else None
        }


class BackendOrchestrator:
    """
    Main orchestration system coordinating multiple quantum backends.
    
    This class provides comprehensive orchestration capabilities including
    intelligent backend selection, calibration-aware scheduling, cross-backend
    benchmarking, and robust fallback strategies.
    
    Example:
        >>> orchestrator = BackendOrchestrator()
        >>> await orchestrator.start()
        >>> 
        >>> # Add backends
        >>> backend = QuantumBackendInfo("ibm_quantum", "IBM Quantum", "IBM", BackendType.HARDWARE, 5)
        >>> orchestrator.add_backend(backend)
        >>> 
        >>> # Execute job with orchestration
        >>> job = OrchestrationJob("job1", circuit_description, requirements)
        >>> result = await orchestrator.execute_job(job)
    """
    
    def __init__(self, config: Optional[SecureConfig] = None):
        self.config = config or SecureConfig()
        
        # Core components
        self.calibration_scheduler = CalibrationAwareScheduler(config)
        self.benchmarker = CrossBackendBenchmarker(config)
        self.fallback_strategy = FallbackStrategy(config)
        
        # Backend management
        self.backends: Dict[str, QuantumBackendInfo] = {}
        self.job_queue: List[OrchestrationJob] = []
        self.active_jobs: Dict[str, OrchestrationJob] = {}
        self.completed_jobs: List[OrchestrationJob] = []
        
        # Orchestration state
        self.is_running = False
        self.orchestration_interval = 10.0  # seconds
        
        # Statistics
        self.orchestration_stats = {
            "jobs_orchestrated": 0,
            "successful_executions": 0,
            "fallback_usage": 0,
            "backend_utilization": {}
        }
        
        logger.info("BackendOrchestrator initialized")
    
    async def start(self) -> None:
        """Start the backend orchestrator."""
        if self.is_running:
            logger.warning("BackendOrchestrator is already running")
            return
        
        self.is_running = True
        logger.info("Starting BackendOrchestrator")
        
        # Start orchestration loops
        asyncio.create_task(self._orchestration_loop())
        asyncio.create_task(self._backend_monitoring_loop())
        asyncio.create_task(self._calibration_management_loop())
    
    async def stop(self) -> None:
        """Stop the backend orchestrator."""
        self.is_running = False
        logger.info("BackendOrchestrator stopped")
    
    def add_backend(self, backend: QuantumBackendInfo) -> None:
        """Add a backend to the orchestrator."""
        self.backends[backend.id] = backend
        self.orchestration_stats["backend_utilization"][backend.id] = 0
        logger.info(f"Added backend: {backend.name} ({backend.id})")
    
    def remove_backend(self, backend_id: str) -> None:
        """Remove a backend from the orchestrator."""
        if backend_id in self.backends:
            del self.backends[backend_id]
            logger.info(f"Removed backend: {backend_id}")
    
    async def submit_job(self, job: OrchestrationJob) -> Dict[str, Any]:
        """Submit a job for orchestrated execution."""
        if job.is_expired():
            return {"error": "Job has already expired"}
        
        self.job_queue.append(job)
        self.orchestration_stats["jobs_orchestrated"] += 1
        
        logger.info(f"Job {job.id} submitted for orchestration")
        
        return {
            "job_id": job.id,
            "status": "queued",
            "queue_position": len(self.job_queue),
            "estimated_backends": self._estimate_suitable_backends(job)
        }
    
    async def execute_job(self, job: OrchestrationJob) -> Dict[str, Any]:
        """Execute a job with full orchestration."""
        logger.info(f"Executing job {job.id} with orchestration")
        
        # Select primary backend
        primary_backend = self.calibration_scheduler.select_backend(
            job, list(self.backends.values())
        )
        
        if not primary_backend:
            return {"error": "No suitable backend available"}
        
        # Select fallback backends
        fallback_backends = self._select_fallback_backends(job, primary_backend)
        
        # Execute with fallback strategy
        result = await self.fallback_strategy.execute_with_fallback(
            job, primary_backend, fallback_backends
        )
        
        # Update statistics
        if result["success"]:
            self.orchestration_stats["successful_executions"] += 1
            
            backend_used = result["backend_used"]
            if backend_used in self.orchestration_stats["backend_utilization"]:
                self.orchestration_stats["backend_utilization"][backend_used] += 1
        
        if result["fallback_used"]:
            self.orchestration_stats["fallback_usage"] += 1
        
        # Move job to completed
        job.results = result
        self.completed_jobs.append(job)
        
        return result
    
    async def _orchestration_loop(self) -> None:
        """Main orchestration loop."""
        while self.is_running:
            try:
                # Process job queue
                if self.job_queue:
                    await self._process_job_queue()
                
                await asyncio.sleep(self.orchestration_interval)
                
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                await asyncio.sleep(self.orchestration_interval)
    
    async def _backend_monitoring_loop(self) -> None:
        """Monitor backend health and performance."""
        while self.is_running:
            try:
                for backend in self.backends.values():
                    await self._update_backend_status(backend)
                
                await asyncio.sleep(60.0)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in backend monitoring: {e}")
                await asyncio.sleep(60.0)
    
    async def _calibration_management_loop(self) -> None:
        """Manage backend calibration scheduling."""
        while self.is_running:
            try:
                # Check which backends need calibration
                backends_to_calibrate = self.calibration_scheduler.schedule_calibration(
                    list(self.backends.values())
                )
                
                # Trigger calibration for backends that need it
                for backend_id in backends_to_calibrate:
                    await self._trigger_backend_calibration(backend_id)
                
                await asyncio.sleep(3600.0)  # Check hourly
                
            except Exception as e:
                logger.error(f"Error in calibration management: {e}")
                await asyncio.sleep(3600.0)
    
    async def _process_job_queue(self) -> None:
        """Process jobs in the queue."""
        # Sort by priority and deadline
        self.job_queue.sort(key=lambda j: (j.priority, j.age()), reverse=True)
        
        processed_jobs = []
        
        for job in self.job_queue[:5]:  # Process up to 5 jobs at once
            if job.is_expired():
                processed_jobs.append(job)
                continue
            
            # Move to active jobs
            self.active_jobs[job.id] = job
            processed_jobs.append(job)
            
            # Execute job asynchronously
            asyncio.create_task(self._execute_job_async(job))
        
        # Remove processed jobs from queue
        for job in processed_jobs:
            if job in self.job_queue:
                self.job_queue.remove(job)
    
    async def _execute_job_async(self, job: OrchestrationJob) -> None:
        """Execute job asynchronously."""
        try:
            result = await self.execute_job(job)
            logger.info(f"Job {job.id} completed: {result.get('success', False)}")
        except Exception as e:
            logger.error(f"Error executing job {job.id}: {e}")
        finally:
            # Remove from active jobs
            self.active_jobs.pop(job.id, None)
    
    async def _update_backend_status(self, backend: QuantumBackendInfo) -> None:
        """Update backend status and metrics."""
        # Simulate status updates (in practice, would query real backends)
        
        # Simulate queue length changes
        backend.queue_length = max(0, backend.queue_length + np.random.randint(-2, 3))
        backend.estimated_wait_time = backend.queue_length * 30  # 30 seconds per job
        
        # Simulate calibration drift
        if backend.last_calibration_time:
            hours_since_cal = (time.time() - backend.last_calibration_time) / 3600
            if hours_since_cal > 48:
                backend.calibration_status = CalibrationStatus.STALE
            elif hours_since_cal > 24:
                backend.calibration_status = CalibrationStatus.GOOD
            else:
                backend.calibration_status = CalibrationStatus.FRESH
    
    async def _trigger_backend_calibration(self, backend_id: str) -> None:
        """Trigger calibration for a backend."""
        if backend_id not in self.backends:
            return
        
        backend = self.backends[backend_id]
        
        logger.info(f"Triggering calibration for backend {backend_id}")
        
        # Simulate calibration process
        backend.calibration_status = CalibrationStatus.CALIBRATING
        
        # Simulate calibration time
        await asyncio.sleep(0.1)
        
        # Update calibration results
        backend.calibration_status = CalibrationStatus.FRESH
        backend.last_calibration_time = time.time()
        
        # Slightly improve fidelities after calibration
        for gate in backend.gate_fidelities:
            backend.gate_fidelities[gate] = min(1.0, backend.gate_fidelities[gate] * 1.02)
        
        logger.info(f"Calibration completed for backend {backend_id}")
    
    def _estimate_suitable_backends(self, job: OrchestrationJob) -> List[str]:
        """Estimate which backends could handle the job."""
        suitable_backends = []
        
        for backend in self.backends.values():
            if (backend.is_available() and
                backend.id not in job.excluded_backends and
                backend.overall_fidelity() >= job.min_fidelity):
                suitable_backends.append(backend.id)
        
        return suitable_backends
    
    def _select_fallback_backends(
        self,
        job: OrchestrationJob,
        primary_backend: QuantumBackendInfo
    ) -> List[QuantumBackendInfo]:
        """Select fallback backends for a job."""
        fallback_backends = []
        
        for backend in self.backends.values():
            if (backend.id != primary_backend.id and
                backend.is_available() and
                backend.id not in job.excluded_backends):
                fallback_backends.append(backend)
        
        # Sort by performance score
        fallback_backends.sort(key=lambda b: b.performance_score(), reverse=True)
        
        return fallback_backends[:3]  # Top 3 fallback options
    
    async def run_cross_backend_benchmark(self, circuits: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run benchmarks across all backends."""
        available_backends = [b for b in self.backends.values() if b.is_available()]
        
        if not available_backends:
            return {"error": "No available backends for benchmarking"}
        
        return await self.benchmarker.run_benchmark_suite(available_backends, circuits)
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration status."""
        return {
            "orchestrator_running": self.is_running,
            "total_backends": len(self.backends),
            "available_backends": len([b for b in self.backends.values() if b.is_available()]),
            "queued_jobs": len(self.job_queue),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "orchestration_stats": self.orchestration_stats.copy(),
            "backend_health": {
                backend_id: {
                    "health": backend.health_status.value,
                    "calibration": backend.calibration_status.value,
                    "performance_score": backend.performance_score()
                }
                for backend_id, backend in self.backends.items()
            },
            "timestamp": time.time()
        }
    
    async def optimize_backend_selection(self) -> Dict[str, Any]:
        """Optimize backend selection based on historical performance."""
        # Analyze completed jobs to optimize selection
        if len(self.completed_jobs) < 10:
            return {"message": "Insufficient data for optimization"}
        
        # Calculate success rates by backend
        backend_performance = {}
        
        for job in self.completed_jobs[-100:]:  # Last 100 jobs
            if "backend_used" in job.results:
                backend_id = job.results["backend_used"]
                success = job.results.get("success", False)
                
                if backend_id not in backend_performance:
                    backend_performance[backend_id] = {"successes": 0, "total": 0}
                
                backend_performance[backend_id]["total"] += 1
                if success:
                    backend_performance[backend_id]["successes"] += 1
        
        # Calculate success rates
        success_rates = {}
        for backend_id, stats in backend_performance.items():
            success_rates[backend_id] = stats["successes"] / stats["total"]
        
        # Update backend reliability scores
        for backend_id, success_rate in success_rates.items():
            if backend_id in self.backends:
                self.backends[backend_id].success_rate = success_rate
        
        return {
            "optimization_completed": True,
            "analyzed_jobs": len(self.completed_jobs[-100:]),
            "backend_success_rates": success_rates,
            "recommendations": self._generate_optimization_recommendations(success_rates)
        }
    
    def _generate_optimization_recommendations(
        self,
        success_rates: Dict[str, float]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if not success_rates:
            return recommendations
        
        # Find best and worst performing backends
        best_backend = max(success_rates.keys(), key=lambda k: success_rates[k])
        worst_backend = min(success_rates.keys(), key=lambda k: success_rates[k])
        
        best_rate = success_rates[best_backend]
        worst_rate = success_rates[worst_backend]
        
        recommendations.append(f"Best performing backend: {best_backend} ({best_rate:.1%} success rate)")
        
        if worst_rate < 0.8:
            recommendations.append(f"Consider investigating {worst_backend} (low success rate: {worst_rate:.1%})")
        
        if best_rate - worst_rate > 0.3:
            recommendations.append("Large performance gap detected - consider rebalancing workloads")
        
        return recommendations