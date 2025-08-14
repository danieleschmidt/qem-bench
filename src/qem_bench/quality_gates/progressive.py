"""
Progressive Quality Orchestrator

Manages quality gates across different development generations with
intelligent progression and autonomous enhancement capabilities.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime

from .core import (
    QualityGateRunner,
    QualityGateResult,
    QualityGateConfig,
    GenerationType,
    QualityGateStatus
)
from .gates import (
    CodeQualityGate,
    SecurityGate,
    PerformanceGate,
    TestingGate,
    DocumentationGate,
    ResearchValidationGate
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationReport:
    """Report for a single generation execution"""
    generation: GenerationType
    timestamp: datetime
    overall_passing: bool
    overall_score: float
    gate_results: List[QualityGateResult]
    execution_time: float
    recommendations: List[str]


class ProgressiveQualityOrchestrator:
    """
    Orchestrates quality gates across development generations with progressive enhancement
    """
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.logger = logging.getLogger("progressive_quality")
        self.generation_history: List[GenerationReport] = []
    
    async def run_generation_quality_gates(
        self, 
        generation: GenerationType,
        auto_progression: bool = True
    ) -> GenerationReport:
        """Run quality gates for a specific generation"""
        self.logger.info(f"Running quality gates for {generation.value}")
        
        start_time = asyncio.get_event_loop().time()
        
        # Configure gates based on generation
        config = self._get_generation_config(generation)
        runner = self._setup_runner(config, generation)
        
        # Execute all gates
        results = await runner.run_all_gates(generation, parallel=True)
        
        # Analyze results
        is_passing, summary = runner.get_overall_status(results)
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, generation)
        
        # Create report
        report = GenerationReport(
            generation=generation,
            timestamp=datetime.now(),
            overall_passing=is_passing,
            overall_score=summary["overall_score"],
            gate_results=results,
            execution_time=execution_time,
            recommendations=recommendations
        )
        
        self.generation_history.append(report)
        
        # Log results
        self._log_generation_results(report)
        
        # Auto-progression logic
        if auto_progression and is_passing:
            next_generation = self._get_next_generation(generation)
            if next_generation and self._should_progress_to_next(generation, next_generation):
                self.logger.info(f"Auto-progressing to {next_generation.value}")
                # Note: Actual progression would be handled by the calling code
        
        return report
    
    async def run_complete_sdlc_cycle(self, include_research: bool = False) -> List[GenerationReport]:
        """Run complete SDLC cycle with progressive quality gates"""
        self.logger.info("Starting complete SDLC quality validation cycle")
        
        generations = [
            GenerationType.GENERATION_1_SIMPLE,
            GenerationType.GENERATION_2_ROBUST,
            GenerationType.GENERATION_3_OPTIMIZED
        ]
        
        if include_research:
            generations.append(GenerationType.RESEARCH_VALIDATION)
        
        reports = []
        
        for generation in generations:
            try:
                report = await self.run_generation_quality_gates(generation, auto_progression=True)
                reports.append(report)
                
                # Stop if generation fails (unless in continuous mode)
                if not report.overall_passing:
                    self.logger.warning(f"Generation {generation.value} failed quality gates")
                    # Attempt auto-fixes
                    if await self._attempt_generation_fixes(report):
                        # Retry the generation
                        retry_report = await self.run_generation_quality_gates(generation)
                        reports[-1] = retry_report  # Replace the failed report
                        
                        if not retry_report.overall_passing:
                            self.logger.error(f"Generation {generation.value} failed after auto-fixes")
                            break
                    else:
                        break
                
            except Exception as e:
                self.logger.error(f"Failed to run generation {generation.value}: {e}")
                break
        
        # Generate overall SDLC report
        self._generate_sdlc_report(reports)
        
        return reports
    
    def _get_generation_config(self, generation: GenerationType) -> QualityGateConfig:
        """Get configuration for specific generation"""
        configs = {
            GenerationType.GENERATION_1_SIMPLE: QualityGateConfig(
                required_score=75.0,
                simple_mode=True,
                robust_validation=False,
                optimization_checks=False,
                research_mode=False,
                auto_fix=True
            ),
            GenerationType.GENERATION_2_ROBUST: QualityGateConfig(
                required_score=85.0,
                simple_mode=False,
                robust_validation=True,
                optimization_checks=False,
                research_mode=False,
                auto_fix=True
            ),
            GenerationType.GENERATION_3_OPTIMIZED: QualityGateConfig(
                required_score=90.0,
                simple_mode=False,
                robust_validation=True,
                optimization_checks=True,
                research_mode=False,
                auto_fix=True
            ),
            GenerationType.RESEARCH_VALIDATION: QualityGateConfig(
                required_score=95.0,
                simple_mode=False,
                robust_validation=True,
                optimization_checks=True,
                research_mode=True,
                auto_fix=False  # Research validation shouldn't auto-fix
            )
        }
        
        return configs.get(generation, QualityGateConfig())
    
    def _setup_runner(self, config: QualityGateConfig, generation: GenerationType) -> QualityGateRunner:
        """Setup quality gate runner for generation"""
        runner = QualityGateRunner(self.project_path)
        
        # Always include these gates
        runner.register_gate(CodeQualityGate(config))
        runner.register_gate(SecurityGate(config))
        runner.register_gate(TestingGate(config))
        runner.register_gate(DocumentationGate(config))
        
        # Conditional gates based on generation
        if generation in [GenerationType.GENERATION_2_ROBUST, GenerationType.GENERATION_3_OPTIMIZED]:
            runner.register_gate(PerformanceGate(config))
        
        if generation == GenerationType.RESEARCH_VALIDATION:
            runner.register_gate(ResearchValidationGate(config))
        
        return runner
    
    def _generate_recommendations(
        self, 
        results: List[QualityGateResult], 
        generation: GenerationType
    ) -> List[str]:
        """Generate improvement recommendations based on results"""
        recommendations = []
        
        for result in results:
            if result.status == QualityGateStatus.FAILED:
                if result.gate_name == "code_quality":
                    recommendations.append("Fix code quality issues with ruff and black formatting")
                elif result.gate_name == "security":
                    recommendations.append("Address security vulnerabilities and remove exposed secrets")
                elif result.gate_name == "testing":
                    recommendations.append("Improve test coverage and fix failing tests")
                elif result.gate_name == "performance":
                    recommendations.append("Optimize performance bottlenecks and add benchmarks")
                elif result.gate_name == "documentation":
                    recommendations.append("Complete documentation for all public APIs")
                elif result.gate_name == "research_validation":
                    recommendations.append("Improve research methodology and reproducibility")
            
            # Add specific recommendations from gate results
            recommendations.extend(result.recommendations)
        
        # Generation-specific recommendations
        if generation == GenerationType.GENERATION_1_SIMPLE:
            recommendations.append("Focus on core functionality and basic error handling")
        elif generation == GenerationType.GENERATION_2_ROBUST:
            recommendations.append("Add comprehensive error handling and logging")
        elif generation == GenerationType.GENERATION_3_OPTIMIZED:
            recommendations.append("Implement performance optimizations and scaling features")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _get_next_generation(self, current: GenerationType) -> Optional[GenerationType]:
        """Get next generation in progression"""
        progression = {
            GenerationType.GENERATION_1_SIMPLE: GenerationType.GENERATION_2_ROBUST,
            GenerationType.GENERATION_2_ROBUST: GenerationType.GENERATION_3_OPTIMIZED,
            GenerationType.GENERATION_3_OPTIMIZED: GenerationType.RESEARCH_VALIDATION
        }
        return progression.get(current)
    
    def _should_progress_to_next(self, current: GenerationType, next_gen: GenerationType) -> bool:
        """Determine if should auto-progress to next generation"""
        # Only progress if current generation has high quality scores
        if not self.generation_history:
            return False
        
        current_report = self.generation_history[-1]
        
        # Require >90% score for auto-progression
        return current_report.overall_score >= 90.0
    
    async def _attempt_generation_fixes(self, report: GenerationReport) -> bool:
        """Attempt to auto-fix issues for failed generation"""
        self.logger.info(f"Attempting auto-fixes for {report.generation.value}")
        
        fixes_applied = 0
        
        for result in report.gate_results:
            if result.status == QualityGateStatus.FAILED:
                # Create a temporary config for the specific gate
                config = self._get_generation_config(report.generation)
                
                # Try to apply fixes based on gate type
                if result.gate_name == "code_quality":
                    gate = CodeQualityGate(config)
                    if await gate.auto_fix(self.project_path, result):
                        fixes_applied += 1
                # Add more auto-fix logic for other gates as needed
        
        self.logger.info(f"Applied {fixes_applied} auto-fixes")
        return fixes_applied > 0
    
    def _log_generation_results(self, report: GenerationReport) -> None:
        """Log generation results"""
        status = "PASSED" if report.overall_passing else "FAILED"
        self.logger.info(
            f"Generation {report.generation.value} {status} "
            f"(Score: {report.overall_score:.1f}%, Time: {report.execution_time:.2f}s)"
        )
        
        for result in report.gate_results:
            gate_status = result.status.value.upper()
            self.logger.info(
                f"  {result.gate_name}: {gate_status} ({result.success_rate:.1f}%)"
            )
    
    def _generate_sdlc_report(self, reports: List[GenerationReport]) -> None:
        """Generate overall SDLC quality report"""
        self.logger.info("=== SDLC Quality Report ===")
        
        total_time = sum(r.execution_time for r in reports)
        passed_generations = sum(1 for r in reports if r.overall_passing)
        avg_score = sum(r.overall_score for r in reports) / len(reports) if reports else 0
        
        self.logger.info(f"Total Execution Time: {total_time:.2f}s")
        self.logger.info(f"Generations Passed: {passed_generations}/{len(reports)}")
        self.logger.info(f"Average Score: {avg_score:.1f}%")
        
        if passed_generations == len(reports):
            self.logger.info("🎉 SDLC QUALITY VALIDATION SUCCESSFUL!")
        else:
            self.logger.warning("⚠️  SDLC quality validation incomplete")
        
        # Log all recommendations
        all_recommendations = []
        for report in reports:
            all_recommendations.extend(report.recommendations)
        
        unique_recommendations = list(set(all_recommendations))
        if unique_recommendations:
            self.logger.info("Recommendations for improvement:")
            for rec in unique_recommendations[:5]:  # Top 5 recommendations
                self.logger.info(f"  - {rec}")
    
    def get_quality_trend(self) -> Dict[str, float]:
        """Get quality trend across generations"""
        if len(self.generation_history) < 2:
            return {}
        
        trend = {}
        for i in range(1, len(self.generation_history)):
            current = self.generation_history[i]
            previous = self.generation_history[i-1]
            
            improvement = current.overall_score - previous.overall_score
            trend[f"{previous.generation.value} -> {current.generation.value}"] = improvement
        
        return trend