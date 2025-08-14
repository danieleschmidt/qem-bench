"""
Core Progressive Quality Gates Infrastructure

Implements the foundation for autonomous quality validation with progressive
enhancement capabilities across all development generations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class QualityGateStatus(Enum):
    """Quality gate execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


class GenerationType(Enum):
    """Development generation types"""
    GENERATION_1_SIMPLE = "gen1_simple"
    GENERATION_2_ROBUST = "gen2_robust"
    GENERATION_3_OPTIMIZED = "gen3_optimized"
    RESEARCH_VALIDATION = "research"
    CONTINUOUS = "continuous"


@dataclass
class QualityGateResult:
    """Result of quality gate execution"""
    gate_name: str
    status: QualityGateStatus
    score: float = 0.0
    max_score: float = 100.0
    execution_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        return (self.score / self.max_score) * 100 if self.max_score > 0 else 0.0
    
    @property
    def is_passing(self) -> bool:
        """Check if gate is passing (85% threshold)"""
        return self.status == QualityGateStatus.PASSED and self.success_rate >= 85.0


@dataclass
class QualityGateConfig:
    """Configuration for quality gate execution"""
    enabled: bool = True
    generation: GenerationType = GenerationType.CONTINUOUS
    required_score: float = 85.0
    timeout_seconds: float = 300.0
    retry_attempts: int = 3
    auto_fix: bool = True
    parallel_execution: bool = True
    
    # Progressive enhancement settings
    simple_mode: bool = False
    robust_validation: bool = True
    optimization_checks: bool = True
    research_mode: bool = False


class BaseQualityGate(ABC):
    """Base class for all quality gates"""
    
    def __init__(self, name: str, config: QualityGateConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"quality_gates.{name}")
    
    @abstractmethod
    async def execute(self, project_path: Path) -> QualityGateResult:
        """Execute the quality gate validation"""
        pass
    
    @abstractmethod
    def get_generation_requirements(self, generation: GenerationType) -> Dict[str, Any]:
        """Get requirements for specific generation"""
        pass
    
    async def auto_fix(self, project_path: Path, result: QualityGateResult) -> bool:
        """Attempt to automatically fix issues"""
        if not self.config.auto_fix:
            return False
        
        self.logger.info(f"Attempting auto-fix for {self.name}")
        # Base implementation - override in subclasses
        return False
    
    def _create_result(
        self, 
        status: QualityGateStatus, 
        score: float = 0.0,
        execution_time: float = 0.0,
        **kwargs
    ) -> QualityGateResult:
        """Create a quality gate result"""
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=score,
            execution_time=execution_time,
            **kwargs
        )


class QualityGateRunner:
    """Orchestrates quality gate execution"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.gates: List[BaseQualityGate] = []
        self.logger = logging.getLogger("quality_gates.runner")
    
    def register_gate(self, gate: BaseQualityGate) -> None:
        """Register a quality gate"""
        self.gates.append(gate)
        self.logger.info(f"Registered quality gate: {gate.name}")
    
    async def run_all_gates(
        self, 
        generation: GenerationType = GenerationType.CONTINUOUS,
        parallel: bool = True
    ) -> List[QualityGateResult]:
        """Run all registered quality gates"""
        self.logger.info(f"Running {len(self.gates)} quality gates for {generation.value}")
        
        if parallel and len(self.gates) > 1:
            return await self._run_parallel(generation)
        else:
            return await self._run_sequential(generation)
    
    async def _run_parallel(self, generation: GenerationType) -> List[QualityGateResult]:
        """Run gates in parallel"""
        tasks = []
        for gate in self.gates:
            if self._should_run_gate(gate, generation):
                task = asyncio.create_task(self._execute_gate_with_retry(gate))
                tasks.append(task)
        
        if not tasks:
            return []
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                gate_name = tasks[i].get_name() if hasattr(tasks[i], 'get_name') else f"gate_{i}"
                self.logger.error(f"Gate {gate_name} failed with exception: {result}")
                processed_results.append(QualityGateResult(
                    gate_name=gate_name,
                    status=QualityGateStatus.FAILED,
                    errors=[str(result)]
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _run_sequential(self, generation: GenerationType) -> List[QualityGateResult]:
        """Run gates sequentially"""
        results = []
        for gate in self.gates:
            if self._should_run_gate(gate, generation):
                result = await self._execute_gate_with_retry(gate)
                results.append(result)
        return results
    
    def _should_run_gate(self, gate: BaseQualityGate, generation: GenerationType) -> bool:
        """Check if gate should run for given generation"""
        if not gate.config.enabled:
            return False
        
        # Check generation compatibility
        requirements = gate.get_generation_requirements(generation)
        return requirements.get("enabled", True)
    
    async def _execute_gate_with_retry(self, gate: BaseQualityGate) -> QualityGateResult:
        """Execute gate with retry logic"""
        last_result = None
        
        for attempt in range(gate.config.retry_attempts):
            try:
                start_time = time.time()
                result = await asyncio.wait_for(
                    gate.execute(self.project_path),
                    timeout=gate.config.timeout_seconds
                )
                result.execution_time = time.time() - start_time
                
                if result.is_passing:
                    self.logger.info(f"Gate {gate.name} passed on attempt {attempt + 1}")
                    return result
                
                # Attempt auto-fix if enabled
                if gate.config.auto_fix and attempt < gate.config.retry_attempts - 1:
                    self.logger.info(f"Attempting auto-fix for {gate.name}")
                    fixed = await gate.auto_fix(self.project_path, result)
                    if fixed:
                        continue
                
                last_result = result
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Gate {gate.name} timed out on attempt {attempt + 1}")
                last_result = QualityGateResult(
                    gate_name=gate.name,
                    status=QualityGateStatus.FAILED,
                    errors=[f"Timeout after {gate.config.timeout_seconds}s"]
                )
            except Exception as e:
                self.logger.error(f"Gate {gate.name} failed on attempt {attempt + 1}: {e}")
                last_result = QualityGateResult(
                    gate_name=gate.name,
                    status=QualityGateStatus.FAILED,
                    errors=[str(e)]
                )
        
        return last_result or QualityGateResult(
            gate_name=gate.name,
            status=QualityGateStatus.FAILED,
            errors=["Failed after all retry attempts"]
        )
    
    def get_overall_status(self, results: List[QualityGateResult]) -> Tuple[bool, Dict[str, Any]]:
        """Get overall quality gate status"""
        total_gates = len(results)
        passed_gates = sum(1 for r in results if r.is_passing)
        failed_gates = sum(1 for r in results if r.status == QualityGateStatus.FAILED)
        
        overall_score = sum(r.success_rate for r in results) / total_gates if total_gates > 0 else 0
        
        # Require 85% of gates to pass
        success_threshold = 0.85
        is_passing = (passed_gates / total_gates) >= success_threshold if total_gates > 0 else False
        
        summary = {
            "total_gates": total_gates,
            "passed_gates": passed_gates,
            "failed_gates": failed_gates,
            "overall_score": overall_score,
            "success_rate": (passed_gates / total_gates * 100) if total_gates > 0 else 0,
            "is_passing": is_passing,
            "details": {r.gate_name: r.success_rate for r in results}
        }
        
        return is_passing, summary