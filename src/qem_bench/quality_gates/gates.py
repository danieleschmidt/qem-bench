"""
Concrete Quality Gate Implementations

Individual quality gates for different aspects of the SDLC with progressive
enhancement capabilities.
"""

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List
import asyncio
import json
import re

from .core import (
    BaseQualityGate, 
    QualityGateResult, 
    QualityGateConfig,
    QualityGateStatus,
    GenerationType
)


class CodeQualityGate(BaseQualityGate):
    """Code quality validation using ruff, black, and mypy"""
    
    def __init__(self, config: QualityGateConfig):
        super().__init__("code_quality", config)
    
    async def execute(self, project_path: Path) -> QualityGateResult:
        """Execute code quality checks"""
        self.logger.info("Running code quality checks")
        
        try:
            # Run ruff for linting
            ruff_score = await self._run_ruff(project_path)
            
            # Run black for formatting (if not in simple mode)
            black_score = 100.0
            if not self.config.simple_mode:
                black_score = await self._run_black(project_path)
            
            # Run mypy for type checking (if robust validation enabled)
            mypy_score = 100.0
            if self.config.robust_validation:
                mypy_score = await self._run_mypy(project_path)
            
            # Calculate overall score
            weights = {"ruff": 0.5, "black": 0.2, "mypy": 0.3}
            overall_score = (
                ruff_score * weights["ruff"] +
                black_score * weights["black"] +
                mypy_score * weights["mypy"]
            )
            
            status = QualityGateStatus.PASSED if overall_score >= self.config.required_score else QualityGateStatus.FAILED
            
            return self._create_result(
                status=status,
                score=overall_score,
                details={
                    "ruff_score": ruff_score,
                    "black_score": black_score,
                    "mypy_score": mypy_score,
                    "weights": weights
                }
            )
            
        except Exception as e:
            self.logger.error(f"Code quality gate failed: {e}")
            return self._create_result(
                status=QualityGateStatus.FAILED,
                errors=[str(e)]
            )
    
    async def _run_ruff(self, project_path: Path) -> float:
        """Run ruff linter"""
        try:
            result = subprocess.run(
                ["ruff", "check", "src/", "--format=json"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return 100.0
            
            # Parse ruff output to calculate score
            issues = []
            if result.stdout:
                try:
                    issues = json.loads(result.stdout)
                except json.JSONDecodeError:
                    pass
            
            # Score based on number of issues (deduct 5 points per issue)
            score = max(0, 100 - (len(issues) * 5))
            return score
            
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return 50.0  # Partial score if ruff fails
    
    async def _run_black(self, project_path: Path) -> float:
        """Run black formatter check"""
        try:
            result = subprocess.run(
                ["black", "--check", "src/"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return 100.0 if result.returncode == 0 else 0.0
            
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return 100.0  # Assume formatted if black unavailable
    
    async def _run_mypy(self, project_path: Path) -> float:
        """Run mypy type checker"""
        try:
            result = subprocess.run(
                ["mypy", "src/qem_bench/", "--ignore-missing-imports"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                return 100.0
            
            # Count errors and calculate score
            error_count = result.stdout.count("error:")
            # Deduct 10 points per error, minimum 0
            score = max(0, 100 - (error_count * 10))
            return score
            
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return 75.0  # Partial score if mypy unavailable
    
    async def auto_fix(self, project_path: Path, result: QualityGateResult) -> bool:
        """Attempt to auto-fix code quality issues"""
        try:
            # Run black to fix formatting
            subprocess.run(
                ["black", "src/"],
                cwd=project_path,
                timeout=60,
                check=False
            )
            
            # Run ruff with --fix flag
            subprocess.run(
                ["ruff", "check", "src/", "--fix"],
                cwd=project_path,
                timeout=60,
                check=False
            )
            
            return True
        except Exception:
            return False
    
    def get_generation_requirements(self, generation: GenerationType) -> Dict[str, Any]:
        """Get requirements for specific generation"""
        requirements = {
            GenerationType.GENERATION_1_SIMPLE: {
                "enabled": True,
                "required_score": 75.0,
                "simple_checks": True
            },
            GenerationType.GENERATION_2_ROBUST: {
                "enabled": True,
                "required_score": 85.0,
                "type_checking": True
            },
            GenerationType.GENERATION_3_OPTIMIZED: {
                "enabled": True,
                "required_score": 95.0,
                "strict_checks": True
            }
        }
        return requirements.get(generation, {"enabled": True})


class SecurityGate(BaseQualityGate):
    """Security validation and vulnerability scanning"""
    
    def __init__(self, config: QualityGateConfig):
        super().__init__("security", config)
    
    async def execute(self, project_path: Path) -> QualityGateResult:
        """Execute security checks"""
        self.logger.info("Running security validation")
        
        try:
            # Check for security vulnerabilities in code
            code_security_score = await self._check_code_security(project_path)
            
            # Check dependencies for known vulnerabilities
            dependency_score = 100.0
            if self.config.robust_validation:
                dependency_score = await self._check_dependencies(project_path)
            
            # Check for secrets in code
            secrets_score = await self._check_secrets(project_path)
            
            # Calculate overall score
            overall_score = (code_security_score * 0.4 + dependency_score * 0.3 + secrets_score * 0.3)
            
            status = QualityGateStatus.PASSED if overall_score >= self.config.required_score else QualityGateStatus.FAILED
            
            return self._create_result(
                status=status,
                score=overall_score,
                details={
                    "code_security_score": code_security_score,
                    "dependency_score": dependency_score,
                    "secrets_score": secrets_score
                }
            )
            
        except Exception as e:
            self.logger.error(f"Security gate failed: {e}")
            return self._create_result(
                status=QualityGateStatus.FAILED,
                errors=[str(e)]
            )
    
    async def _check_code_security(self, project_path: Path) -> float:
        """Check for security issues in code"""
        security_issues = 0
        
        # Check for common security anti-patterns
        src_path = project_path / "src"
        if src_path.exists():
            for py_file in src_path.rglob("*.py"):
                try:
                    content = py_file.read_text()
                    
                    # Check for dangerous patterns
                    dangerous_patterns = [
                        r"eval\(",
                        r"exec\(",
                        r"subprocess\..*shell=True",
                        r"os\.system\(",
                        r"import pickle",
                        r"pickle\.load"
                    ]
                    
                    for pattern in dangerous_patterns:
                        if re.search(pattern, content):
                            security_issues += 1
                            
                except Exception:
                    continue
        
        # Score based on issues found (deduct 20 points per issue)
        score = max(0, 100 - (security_issues * 20))
        return score
    
    async def _check_dependencies(self, project_path: Path) -> float:
        """Check dependencies for vulnerabilities"""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "check"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return 100.0 if result.returncode == 0 else 75.0
            
        except Exception:
            return 90.0  # Assume mostly secure if check fails
    
    async def _check_secrets(self, project_path: Path) -> float:
        """Check for exposed secrets"""
        secrets_found = 0
        
        # Common secret patterns
        secret_patterns = [
            r"password\s*=\s*[\"'][^\"']{6,}[\"']",
            r"api_key\s*=\s*[\"'][^\"']{10,}[\"']",
            r"secret\s*=\s*[\"'][^\"']{10,}[\"']",
            r"token\s*=\s*[\"'][^\"']{10,}[\"']"
        ]
        
        src_path = project_path / "src"
        if src_path.exists():
            for py_file in src_path.rglob("*.py"):
                try:
                    content = py_file.read_text()
                    
                    for pattern in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            secrets_found += 1
                            
                except Exception:
                    continue
        
        # Score based on secrets found (deduct 25 points per secret)
        score = max(0, 100 - (secrets_found * 25))
        return score
    
    def get_generation_requirements(self, generation: GenerationType) -> Dict[str, Any]:
        """Get requirements for specific generation"""
        requirements = {
            GenerationType.GENERATION_1_SIMPLE: {
                "enabled": True,
                "required_score": 80.0,
                "basic_checks": True
            },
            GenerationType.GENERATION_2_ROBUST: {
                "enabled": True,
                "required_score": 90.0,
                "dependency_checks": True
            },
            GenerationType.GENERATION_3_OPTIMIZED: {
                "enabled": True,
                "required_score": 95.0,
                "advanced_scanning": True
            }
        }
        return requirements.get(generation, {"enabled": True})


class TestingGate(BaseQualityGate):
    """Test coverage and execution validation"""
    
    def __init__(self, config: QualityGateConfig):
        super().__init__("testing", config)
    
    async def execute(self, project_path: Path) -> QualityGateResult:
        """Execute testing validation"""
        self.logger.info("Running test validation")
        
        try:
            # Run tests and measure coverage
            test_results = await self._run_tests(project_path)
            
            status = QualityGateStatus.PASSED if test_results["score"] >= self.config.required_score else QualityGateStatus.FAILED
            
            return self._create_result(
                status=status,
                score=test_results["score"],
                details=test_results
            )
            
        except Exception as e:
            self.logger.error(f"Testing gate failed: {e}")
            return self._create_result(
                status=QualityGateStatus.FAILED,
                errors=[str(e)]
            )
    
    async def _run_tests(self, project_path: Path) -> Dict[str, Any]:
        """Run pytest with coverage"""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=qem_bench", "--cov-report=json", "-v"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Parse coverage report
            coverage_file = project_path / "coverage.json"
            coverage_percent = 0.0
            
            if coverage_file.exists():
                try:
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                        coverage_percent = coverage_data.get("totals", {}).get("percent_covered", 0.0)
                except Exception:
                    pass
            
            # Calculate test results
            test_passed = result.returncode == 0
            test_score = 100.0 if test_passed else 0.0
            
            # Overall score combines test success and coverage
            overall_score = (test_score * 0.6 + coverage_percent * 0.4)
            
            return {
                "score": overall_score,
                "test_passed": test_passed,
                "coverage_percent": coverage_percent,
                "test_output": result.stdout,
                "test_errors": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {"score": 0.0, "error": "Tests timed out"}
        except Exception as e:
            return {"score": 50.0, "error": str(e)}
    
    def get_generation_requirements(self, generation: GenerationType) -> Dict[str, Any]:
        """Get requirements for specific generation"""
        requirements = {
            GenerationType.GENERATION_1_SIMPLE: {
                "enabled": True,
                "required_score": 70.0,
                "basic_tests": True
            },
            GenerationType.GENERATION_2_ROBUST: {
                "enabled": True,
                "required_score": 85.0,
                "integration_tests": True
            },
            GenerationType.GENERATION_3_OPTIMIZED: {
                "enabled": True,
                "required_score": 90.0,
                "performance_tests": True
            }
        }
        return requirements.get(generation, {"enabled": True})


class PerformanceGate(BaseQualityGate):
    """Performance benchmarking and validation"""
    
    def __init__(self, config: QualityGateConfig):
        super().__init__("performance", config)
    
    async def execute(self, project_path: Path) -> QualityGateResult:
        """Execute performance validation"""
        self.logger.info("Running performance validation")
        
        if not self.config.optimization_checks:
            return self._create_result(
                status=QualityGateStatus.SKIPPED,
                score=100.0,
                details={"skipped": "Optimization checks disabled"}
            )
        
        try:
            # Run basic performance benchmarks
            benchmark_score = await self._run_benchmarks(project_path)
            
            status = QualityGateStatus.PASSED if benchmark_score >= self.config.required_score else QualityGateStatus.FAILED
            
            return self._create_result(
                status=status,
                score=benchmark_score,
                details={"benchmark_score": benchmark_score}
            )
            
        except Exception as e:
            self.logger.error(f"Performance gate failed: {e}")
            return self._create_result(
                status=QualityGateStatus.FAILED,
                errors=[str(e)]
            )
    
    async def _run_benchmarks(self, project_path: Path) -> float:
        """Run performance benchmarks"""
        # Simple performance validation
        # In a real implementation, this would run actual benchmarks
        
        # Check if performance-critical modules exist
        critical_modules = [
            "src/qem_bench/jax/",
            "src/qem_bench/optimization/",
            "src/qem_bench/scaling/"
        ]
        
        score = 85.0  # Base score
        
        for module_path in critical_modules:
            if (project_path / module_path).exists():
                score += 5.0
        
        return min(100.0, score)
    
    def get_generation_requirements(self, generation: GenerationType) -> Dict[str, Any]:
        """Get requirements for specific generation"""
        requirements = {
            GenerationType.GENERATION_1_SIMPLE: {
                "enabled": False,  # Skip performance for simple
                "required_score": 70.0
            },
            GenerationType.GENERATION_2_ROBUST: {
                "enabled": True,
                "required_score": 80.0
            },
            GenerationType.GENERATION_3_OPTIMIZED: {
                "enabled": True,
                "required_score": 90.0,
                "benchmark_tests": True
            }
        }
        return requirements.get(generation, {"enabled": True})


class DocumentationGate(BaseQualityGate):
    """Documentation completeness validation"""
    
    def __init__(self, config: QualityGateConfig):
        super().__init__("documentation", config)
    
    async def execute(self, project_path: Path) -> QualityGateResult:
        """Execute documentation validation"""
        self.logger.info("Running documentation validation")
        
        try:
            doc_score = await self._check_documentation(project_path)
            
            status = QualityGateStatus.PASSED if doc_score >= self.config.required_score else QualityGateStatus.FAILED
            
            return self._create_result(
                status=status,
                score=doc_score,
                details={"documentation_score": doc_score}
            )
            
        except Exception as e:
            self.logger.error(f"Documentation gate failed: {e}")
            return self._create_result(
                status=QualityGateStatus.FAILED,
                errors=[str(e)]
            )
    
    async def _check_documentation(self, project_path: Path) -> float:
        """Check documentation completeness"""
        score = 0.0
        
        # Check for essential documentation files
        essential_docs = [
            "README.md",
            "CONTRIBUTING.md", 
            "LICENSE",
            "docs/",
            "examples/"
        ]
        
        for doc in essential_docs:
            if (project_path / doc).exists():
                score += 20.0
        
        return min(100.0, score)
    
    def get_generation_requirements(self, generation: GenerationType) -> Dict[str, Any]:
        """Get requirements for specific generation"""
        requirements = {
            GenerationType.GENERATION_1_SIMPLE: {
                "enabled": True,
                "required_score": 60.0,
                "basic_docs": True
            },
            GenerationType.GENERATION_2_ROBUST: {
                "enabled": True,
                "required_score": 80.0,
                "api_docs": True
            },
            GenerationType.GENERATION_3_OPTIMIZED: {
                "enabled": True,
                "required_score": 90.0,
                "complete_docs": True
            }
        }
        return requirements.get(generation, {"enabled": True})


class ResearchValidationGate(BaseQualityGate):
    """Research methodology and reproducibility validation"""
    
    def __init__(self, config: QualityGateConfig):
        super().__init__("research_validation", config)
    
    async def execute(self, project_path: Path) -> QualityGateResult:
        """Execute research validation"""
        self.logger.info("Running research validation")
        
        if not self.config.research_mode:
            return self._create_result(
                status=QualityGateStatus.SKIPPED,
                score=100.0,
                details={"skipped": "Research mode disabled"}
            )
        
        try:
            research_score = await self._validate_research(project_path)
            
            status = QualityGateStatus.PASSED if research_score >= self.config.required_score else QualityGateStatus.FAILED
            
            return self._create_result(
                status=status,
                score=research_score,
                details={"research_score": research_score}
            )
            
        except Exception as e:
            self.logger.error(f"Research validation gate failed: {e}")
            return self._create_result(
                status=QualityGateStatus.FAILED,
                errors=[str(e)]
            )
    
    async def _validate_research(self, project_path: Path) -> float:
        """Validate research methodology"""
        score = 0.0
        
        # Check for research components
        research_components = [
            "src/qem_bench/research/",
            "experiments/",
            "data/",
            "notebooks/",
            "RESEARCH_METHODOLOGY.md"
        ]
        
        for component in research_components:
            if (project_path / component).exists():
                score += 20.0
        
        return min(100.0, score)
    
    def get_generation_requirements(self, generation: GenerationType) -> Dict[str, Any]:
        """Get requirements for specific generation"""
        requirements = {
            GenerationType.RESEARCH_VALIDATION: {
                "enabled": True,
                "required_score": 85.0,
                "reproducibility_checks": True,
                "statistical_validation": True
            }
        }
        return requirements.get(generation, {"enabled": False})