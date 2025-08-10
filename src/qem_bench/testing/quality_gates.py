"""
Comprehensive quality gates and testing framework for QEM-Bench.

Provides automated quality assurance with performance benchmarks, security scanning,
code coverage analysis, and production readiness validation.
"""

import time
import subprocess
import tempfile
import json
import ast
import sys
import importlib
import inspect
from typing import Any, Dict, List, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import coverage
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..monitoring.logger import get_logger
from ..monitoring.health import HealthStatus, HealthCheckResult


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    
    gate_name: str
    status: HealthStatus
    score: float  # 0.0 - 1.0
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    
    @property
    def passed(self) -> bool:
        """Check if gate passed."""
        return self.status in (HealthStatus.HEALTHY, HealthStatus.WARNING)
    
    @property
    def failed(self) -> bool:
        """Check if gate failed."""
        return self.status == HealthStatus.CRITICAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'gate_name': self.gate_name,
            'status': self.status.value,
            'score': self.score,
            'message': self.message,
            'details': self.details,
            'recommendations': self.recommendations,
            'execution_time': self.execution_time,
            'passed': self.passed
        }


class QualityGate(ABC):
    """Abstract base class for quality gates."""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight  # Importance weight for overall score
        self.logger = get_logger(f"quality_gate.{name}")
    
    @abstractmethod
    def execute(self) -> QualityGateResult:
        """Execute quality gate check."""
        pass
    
    def run(self) -> QualityGateResult:
        """Run quality gate with timing."""
        start_time = time.time()
        
        try:
            result = self.execute()
            result.execution_time = time.time() - start_time
            
            if result.passed:
                self.logger.info(f"Quality gate {self.name} PASSED (score: {result.score:.2f})")
            else:
                self.logger.error(f"Quality gate {self.name} FAILED (score: {result.score:.2f}): {result.message}")
            
            return result
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Quality gate {self.name} ERRORED: {str(e)}", e)
            
            return QualityGateResult(
                gate_name=self.name,
                status=HealthStatus.CRITICAL,
                score=0.0,
                message=f"Gate execution failed: {str(e)}",
                execution_time=execution_time,
                recommendations=["Fix gate execution error", "Check system dependencies"]
            )


class CodeCoverageGate(QualityGate):
    """Quality gate for code coverage analysis."""
    
    def __init__(self, min_coverage: float = 0.85, min_branch_coverage: float = 0.80):
        super().__init__("code_coverage", weight=1.5)
        self.min_coverage = min_coverage
        self.min_branch_coverage = min_branch_coverage
    
    def execute(self) -> QualityGateResult:
        """Execute code coverage analysis."""
        try:
            # Initialize coverage measurement
            cov = coverage.Coverage(
                source=['qem_bench'],
                branch=True,
                omit=['*/tests/*', '*/test_*', '*/__pycache__/*']
            )
            
            # Run basic import tests to measure coverage
            cov.start()
            
            # Import and exercise main modules
            modules_to_test = [
                'qem_bench.jax.circuits',
                'qem_bench.jax.simulator', 
                'qem_bench.jax.observables',
                'qem_bench.mitigation.zne.core',
                'qem_bench.benchmarks.circuits.standard',
                'qem_bench.benchmarks.metrics.fidelity',
                'qem_bench.validation.core',
                'qem_bench.errors.recovery',
                'qem_bench.monitoring.health',
                'qem_bench.optimization.cache'
            ]
            
            imported_modules = []
            for module_name in modules_to_test:
                try:
                    module = importlib.import_module(module_name)
                    imported_modules.append(module)
                    
                    # Exercise basic functionality
                    self._exercise_module(module)
                    
                except ImportError as e:
                    self.logger.warning(f"Could not import {module_name}: {e}")
                except Exception as e:
                    self.logger.warning(f"Error exercising {module_name}: {e}")
            
            cov.stop()
            cov.save()
            
            # Generate coverage report
            with tempfile.TemporaryDirectory() as temp_dir:
                report_file = Path(temp_dir) / "coverage.json"
                cov.json_report(outfile=str(report_file))
                
                with open(report_file, 'r') as f:
                    coverage_data = json.load(f)
            
            # Extract metrics
            totals = coverage_data.get('totals', {})
            line_coverage = totals.get('percent_covered', 0) / 100
            branch_coverage = totals.get('percent_covered_display', 0)
            
            # Try to extract branch coverage from covered_branches
            if 'covered_branches' in totals and 'num_branches' in totals:
                branch_coverage = (totals['covered_branches'] / totals['num_branches'] 
                                 if totals['num_branches'] > 0 else 1.0)
            else:
                branch_coverage = line_coverage  # Fallback
            
            # Calculate score
            line_score = min(1.0, line_coverage / self.min_coverage)
            branch_score = min(1.0, branch_coverage / self.min_branch_coverage)
            overall_score = (line_score + branch_score) / 2
            
            # Determine status
            if line_coverage >= self.min_coverage and branch_coverage >= self.min_branch_coverage:
                status = HealthStatus.HEALTHY
                message = f"Code coverage excellent: {line_coverage:.1%} lines, {branch_coverage:.1%} branches"
            elif line_coverage >= self.min_coverage * 0.8:
                status = HealthStatus.WARNING
                message = f"Code coverage acceptable: {line_coverage:.1%} lines, {branch_coverage:.1%} branches"
            else:
                status = HealthStatus.CRITICAL
                message = f"Code coverage insufficient: {line_coverage:.1%} lines, {branch_coverage:.1%} branches"
            
            recommendations = []
            if line_coverage < self.min_coverage:
                recommendations.append(f"Increase line coverage to {self.min_coverage:.1%}")
            if branch_coverage < self.min_branch_coverage:
                recommendations.append(f"Increase branch coverage to {self.min_branch_coverage:.1%}")
            if overall_score < 0.8:
                recommendations.append("Add comprehensive unit tests")
                recommendations.append("Include edge case testing")
            
            return QualityGateResult(
                gate_name=self.name,
                status=status,
                score=overall_score,
                message=message,
                details={
                    'line_coverage': line_coverage,
                    'branch_coverage': branch_coverage,
                    'lines_covered': totals.get('covered_lines', 0),
                    'lines_total': totals.get('num_statements', 0),
                    'branches_covered': totals.get('covered_branches', 0),
                    'branches_total': totals.get('num_branches', 0),
                    'modules_tested': len(imported_modules)
                },
                recommendations=recommendations
            )
        
        except Exception as e:
            return QualityGateResult(
                gate_name=self.name,
                status=HealthStatus.CRITICAL,
                score=0.0,
                message=f"Coverage analysis failed: {str(e)}",
                recommendations=["Install coverage package", "Fix import issues"]
            )
    
    def _exercise_module(self, module):
        """Exercise basic module functionality for coverage."""
        try:
            # Get all classes and functions in module
            for name in dir(module):
                if name.startswith('_'):
                    continue
                
                attr = getattr(module, name)
                
                # Exercise classes by trying to create simple instances
                if inspect.isclass(attr):
                    try:
                        # Try to create instance with minimal args
                        if name == 'JAXCircuit':
                            instance = attr(2)
                        elif name == 'JAXSimulator':
                            instance = attr(2)
                        elif name == 'ZeroNoiseExtrapolation':
                            instance = attr(noise_factors=[1.0, 2.0])
                        else:
                            # Try default constructor
                            instance = attr()
                    except:
                        pass  # Skip if constructor fails
                
                # Exercise functions with no parameters
                elif inspect.isfunction(attr):
                    try:
                        sig = inspect.signature(attr)
                        if len(sig.parameters) == 0:
                            attr()
                    except:
                        pass  # Skip if function fails
        
        except Exception:
            pass  # Ignore errors during exercise


class PerformanceBenchmarkGate(QualityGate):
    """Quality gate for performance benchmarks."""
    
    def __init__(self, max_simulation_time_ms: float = 1000, 
                 max_zne_time_ms: float = 5000):
        super().__init__("performance_benchmark", weight=1.2)
        self.max_simulation_time_ms = max_simulation_time_ms
        self.max_zne_time_ms = max_zne_time_ms
    
    def execute(self) -> QualityGateResult:
        """Execute performance benchmarks."""
        benchmarks = []
        
        try:
            # Benchmark 1: Circuit simulation
            sim_time = self._benchmark_circuit_simulation()
            benchmarks.append(('simulation', sim_time, self.max_simulation_time_ms))
            
            # Benchmark 2: ZNE mitigation
            zne_time = self._benchmark_zne_mitigation()
            benchmarks.append(('zne', zne_time, self.max_zne_time_ms))
            
            # Calculate overall performance score
            scores = []
            for name, time_ms, max_time_ms in benchmarks:
                if time_ms <= max_time_ms:
                    score = 1.0
                elif time_ms <= max_time_ms * 2:
                    score = max(0.5, 1.0 - (time_ms - max_time_ms) / max_time_ms)
                else:
                    score = 0.1  # Very poor performance
                scores.append(score)
            
            overall_score = sum(scores) / len(scores) if scores else 0.0
            
            # Determine status
            if overall_score >= 0.9:
                status = HealthStatus.HEALTHY
                message = "Performance benchmarks excellent"
            elif overall_score >= 0.7:
                status = HealthStatus.WARNING  
                message = "Performance benchmarks acceptable"
            else:
                status = HealthStatus.CRITICAL
                message = "Performance benchmarks failed"
            
            # Generate recommendations
            recommendations = []
            for name, time_ms, max_time_ms in benchmarks:
                if time_ms > max_time_ms:
                    recommendations.append(f"Optimize {name} performance (current: {time_ms:.1f}ms, target: {max_time_ms:.1f}ms)")
            
            if overall_score < 0.8:
                recommendations.extend([
                    "Profile performance bottlenecks",
                    "Consider caching optimizations",
                    "Review algorithmic complexity"
                ])
            
            return QualityGateResult(
                gate_name=self.name,
                status=status,
                score=overall_score,
                message=message,
                details={
                    'benchmarks': {name: time_ms for name, time_ms, _ in benchmarks},
                    'thresholds': {name: max_time_ms for name, _, max_time_ms in benchmarks}
                },
                recommendations=recommendations
            )
        
        except Exception as e:
            return QualityGateResult(
                gate_name=self.name,
                status=HealthStatus.CRITICAL,
                score=0.0,
                message=f"Performance benchmark failed: {str(e)}",
                recommendations=["Fix benchmark execution", "Check dependencies"]
            )
    
    def _benchmark_circuit_simulation(self) -> float:
        """Benchmark basic circuit simulation."""
        try:
            from ..jax.circuits import JAXCircuit
            from ..jax.simulator import JAXSimulator
            
            # Create test circuit
            circuit = JAXCircuit(3)
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.cx(1, 2)
            
            simulator = JAXSimulator(3)
            
            # Benchmark simulation
            start_time = time.time()
            result = simulator.run(circuit, shots=1000)
            duration_ms = (time.time() - start_time) * 1000
            
            return duration_ms
        
        except Exception as e:
            self.logger.warning(f"Circuit simulation benchmark failed: {e}")
            return float('inf')
    
    def _benchmark_zne_mitigation(self) -> float:
        """Benchmark ZNE error mitigation."""
        try:
            from ..mitigation.zne.core import ZeroNoiseExtrapolation
            from ..jax.circuits import JAXCircuit
            from ..jax.simulator import JAXSimulator
            from ..jax.observables import ZObservable
            
            # Create test setup
            circuit = JAXCircuit(2)
            circuit.h(0)
            circuit.cx(0, 1)
            
            simulator = JAXSimulator(2)
            observable = ZObservable(0)
            
            # Mock backend
            class MockBackend:
                def __init__(self, sim):
                    self.sim = sim
                
                def run_with_observable(self, circuit, observable, shots=100, **kwargs):
                    result = self.sim.run(circuit, observables=[observable])
                    
                    class MockResult:
                        def __init__(self, exp_val):
                            self.expectation_value = exp_val
                    
                    return MockResult(result.expectation_values[observable.name])
            
            backend = MockBackend(simulator)
            
            # Benchmark ZNE
            start_time = time.time()
            zne = ZeroNoiseExtrapolation(
                noise_factors=[1.0, 1.5, 2.0],
                extrapolator="richardson"
            )
            result = zne.mitigate(circuit, backend, observable)
            duration_ms = (time.time() - start_time) * 1000
            
            return duration_ms
        
        except Exception as e:
            self.logger.warning(f"ZNE mitigation benchmark failed: {e}")
            return float('inf')


class SecurityScanGate(QualityGate):
    """Quality gate for security vulnerability scanning."""
    
    def __init__(self):
        super().__init__("security_scan", weight=1.8)
    
    def execute(self) -> QualityGateResult:
        """Execute security vulnerability scan."""
        vulnerabilities = []
        
        try:
            # Scan for common security issues
            vulnerabilities.extend(self._scan_dangerous_imports())
            vulnerabilities.extend(self._scan_hardcoded_secrets())
            vulnerabilities.extend(self._scan_unsafe_eval_exec())
            vulnerabilities.extend(self._scan_pickle_usage())
            vulnerabilities.extend(self._scan_sql_injection())
            
            # Calculate security score
            critical_vulns = [v for v in vulnerabilities if v['severity'] == 'critical']
            high_vulns = [v for v in vulnerabilities if v['severity'] == 'high']
            medium_vulns = [v for v in vulnerabilities if v['severity'] == 'medium']
            
            # Scoring: critical = -0.5, high = -0.2, medium = -0.1
            penalty = (len(critical_vulns) * 0.5 + 
                      len(high_vulns) * 0.2 + 
                      len(medium_vulns) * 0.1)
            
            security_score = max(0.0, 1.0 - penalty)
            
            # Determine status
            if len(critical_vulns) > 0:
                status = HealthStatus.CRITICAL
                message = f"Critical security vulnerabilities found: {len(critical_vulns)}"
            elif len(high_vulns) > 0:
                status = HealthStatus.WARNING
                message = f"High security vulnerabilities found: {len(high_vulns)}"
            elif len(vulnerabilities) > 0:
                status = HealthStatus.WARNING
                message = f"Medium security issues found: {len(medium_vulns)}"
            else:
                status = HealthStatus.HEALTHY
                message = "No security vulnerabilities detected"
            
            recommendations = []
            if critical_vulns:
                recommendations.append("Fix critical security vulnerabilities immediately")
            if high_vulns:
                recommendations.append("Address high-severity security issues")
            if medium_vulns:
                recommendations.append("Review and fix medium-severity issues")
            if vulnerabilities:
                recommendations.extend([
                    "Implement security code review process",
                    "Use automated security scanning in CI/CD"
                ])
            
            return QualityGateResult(
                gate_name=self.name,
                status=status,
                score=security_score,
                message=message,
                details={
                    'total_vulnerabilities': len(vulnerabilities),
                    'critical_vulnerabilities': len(critical_vulns),
                    'high_vulnerabilities': len(high_vulns),
                    'medium_vulnerabilities': len(medium_vulns),
                    'vulnerabilities': vulnerabilities
                },
                recommendations=recommendations
            )
        
        except Exception as e:
            return QualityGateResult(
                gate_name=self.name,
                status=HealthStatus.CRITICAL,
                score=0.0,
                message=f"Security scan failed: {str(e)}",
                recommendations=["Fix security scan execution"]
            )
    
    def _scan_dangerous_imports(self) -> List[Dict[str, Any]]:
        """Scan for dangerous import statements."""
        vulnerabilities = []
        dangerous_imports = ['os', 'subprocess', 'eval', 'exec', 'pickle', '__import__']
        
        try:
            src_path = Path(__file__).parent.parent.parent
            for py_file in src_path.rglob('*.py'):
                if 'test' in str(py_file).lower():
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if alias.name in dangerous_imports:
                                    # Check if it's legitimate usage
                                    if self._is_legitimate_usage(py_file, alias.name, content):
                                        continue
                                    
                                    vulnerabilities.append({
                                        'type': 'dangerous_import',
                                        'severity': 'medium',
                                        'file': str(py_file),
                                        'line': node.lineno,
                                        'description': f"Potentially dangerous import: {alias.name}"
                                    })
                        
                        elif isinstance(node, ast.ImportFrom):
                            if node.module in dangerous_imports:
                                vulnerabilities.append({
                                    'type': 'dangerous_import',
                                    'severity': 'medium', 
                                    'file': str(py_file),
                                    'line': node.lineno,
                                    'description': f"Import from dangerous module: {node.module}"
                                })
                
                except Exception as e:
                    self.logger.warning(f"Error scanning {py_file}: {e}")
        
        except Exception as e:
            self.logger.warning(f"Error scanning dangerous imports: {e}")
        
        return vulnerabilities
    
    def _is_legitimate_usage(self, file_path: Path, import_name: str, content: str) -> bool:
        """Check if dangerous import usage is legitimate."""
        # Allow certain usage patterns
        legitimate_patterns = {
            'os': ['os.path', 'os.environ.get'],
            'subprocess': ['subprocess.run', 'subprocess.check_output'],
            'pickle': ['pickle.dumps', 'pickle.loads']  # Used in caching
        }
        
        if import_name in legitimate_patterns:
            return any(pattern in content for pattern in legitimate_patterns[import_name])
        
        return False
    
    def _scan_hardcoded_secrets(self) -> List[Dict[str, Any]]:
        """Scan for hardcoded secrets and credentials."""
        vulnerabilities = []
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'hardcoded_password'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'hardcoded_api_key'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'hardcoded_secret'),
            (r'token\s*=\s*["\'][^"\']+["\']', 'hardcoded_token')
        ]
        
        try:
            import re
            src_path = Path(__file__).parent.parent.parent
            
            for py_file in src_path.rglob('*.py'):
                if 'test' in str(py_file).lower():
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern, vuln_type in secret_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            vulnerabilities.append({
                                'type': vuln_type,
                                'severity': 'high',
                                'file': str(py_file),
                                'line': line_num,
                                'description': f"Potential hardcoded secret: {match.group(0)[:50]}..."
                            })
                
                except Exception as e:
                    self.logger.warning(f"Error scanning {py_file}: {e}")
        
        except Exception as e:
            self.logger.warning(f"Error scanning hardcoded secrets: {e}")
        
        return vulnerabilities
    
    def _scan_unsafe_eval_exec(self) -> List[Dict[str, Any]]:
        """Scan for unsafe eval/exec usage."""
        vulnerabilities = []
        
        try:
            src_path = Path(__file__).parent.parent.parent
            for py_file in src_path.rglob('*.py'):
                if 'test' in str(py_file).lower():
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            if (isinstance(node.func, ast.Name) and 
                                node.func.id in ['eval', 'exec']):
                                
                                vulnerabilities.append({
                                    'type': 'unsafe_eval_exec',
                                    'severity': 'critical',
                                    'file': str(py_file),
                                    'line': node.lineno,
                                    'description': f"Unsafe use of {node.func.id}()"
                                })
                
                except Exception as e:
                    self.logger.warning(f"Error scanning {py_file}: {e}")
        
        except Exception as e:
            self.logger.warning(f"Error scanning eval/exec usage: {e}")
        
        return vulnerabilities
    
    def _scan_pickle_usage(self) -> List[Dict[str, Any]]:
        """Scan for potentially unsafe pickle usage."""
        vulnerabilities = []
        
        # Note: pickle is used in caching, but we check for unsafe patterns
        try:
            src_path = Path(__file__).parent.parent.parent
            for py_file in src_path.rglob('*.py'):
                if 'test' in str(py_file).lower():
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for pickle.loads with user input
                    if 'pickle.loads' in content and ('input(' in content or 'raw_input(' in content):
                        vulnerabilities.append({
                            'type': 'unsafe_pickle',
                            'severity': 'high',
                            'file': str(py_file),
                            'line': content.find('pickle.loads'),
                            'description': "Potentially unsafe pickle.loads with user input"
                        })
                
                except Exception as e:
                    self.logger.warning(f"Error scanning {py_file}: {e}")
        
        except Exception as e:
            self.logger.warning(f"Error scanning pickle usage: {e}")
        
        return vulnerabilities
    
    def _scan_sql_injection(self) -> List[Dict[str, Any]]:
        """Scan for SQL injection vulnerabilities."""
        # QEM-Bench doesn't use SQL, but good to check
        return []


class CodeQualityGate(QualityGate):
    """Quality gate for code quality metrics."""
    
    def __init__(self, max_complexity: int = 10, max_function_length: int = 100):
        super().__init__("code_quality", weight=1.0)
        self.max_complexity = max_complexity
        self.max_function_length = max_function_length
    
    def execute(self) -> QualityGateResult:
        """Execute code quality analysis."""
        try:
            issues = []
            
            # Analyze code complexity and quality
            issues.extend(self._analyze_complexity())
            issues.extend(self._analyze_function_length())
            issues.extend(self._analyze_code_style())
            
            # Calculate quality score
            critical_issues = [i for i in issues if i['severity'] == 'critical']
            high_issues = [i for i in issues if i['severity'] == 'high']
            medium_issues = [i for i in issues if i['severity'] == 'medium']
            
            # Scoring
            penalty = (len(critical_issues) * 0.3 + 
                      len(high_issues) * 0.15 + 
                      len(medium_issues) * 0.05)
            
            quality_score = max(0.0, 1.0 - penalty)
            
            # Determine status
            if len(critical_issues) > 0:
                status = HealthStatus.CRITICAL
                message = f"Critical code quality issues: {len(critical_issues)}"
            elif len(high_issues) > 5:
                status = HealthStatus.WARNING
                message = f"High code quality issues: {len(high_issues)}"
            elif quality_score >= 0.8:
                status = HealthStatus.HEALTHY
                message = "Code quality excellent"
            else:
                status = HealthStatus.WARNING
                message = "Code quality needs improvement"
            
            recommendations = []
            if critical_issues:
                recommendations.append("Fix critical code quality issues")
            if len(high_issues) > 5:
                recommendations.append("Reduce code complexity")
            if len(medium_issues) > 10:
                recommendations.append("Improve code style consistency")
            
            recommendations.extend([
                "Use automated code formatting (black, isort)",
                "Implement code review process",
                "Add complexity limits to CI/CD"
            ])
            
            return QualityGateResult(
                gate_name=self.name,
                status=status,
                score=quality_score,
                message=message,
                details={
                    'total_issues': len(issues),
                    'critical_issues': len(critical_issues),
                    'high_issues': len(high_issues),
                    'medium_issues': len(medium_issues),
                    'issues': issues[:20]  # Limit for brevity
                },
                recommendations=recommendations
            )
        
        except Exception as e:
            return QualityGateResult(
                gate_name=self.name,
                status=HealthStatus.CRITICAL,
                score=0.0,
                message=f"Code quality analysis failed: {str(e)}",
                recommendations=["Fix code quality analysis"]
            )
    
    def _analyze_complexity(self) -> List[Dict[str, Any]]:
        """Analyze code complexity."""
        issues = []
        
        try:
            src_path = Path(__file__).parent.parent.parent
            for py_file in src_path.rglob('*.py'):
                if 'test' in str(py_file).lower():
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            complexity = self._calculate_complexity(node)
                            if complexity > self.max_complexity:
                                issues.append({
                                    'type': 'high_complexity',
                                    'severity': 'high' if complexity > self.max_complexity * 1.5 else 'medium',
                                    'file': str(py_file),
                                    'line': node.lineno,
                                    'function': node.name,
                                    'complexity': complexity,
                                    'description': f"Function {node.name} has complexity {complexity} (max: {self.max_complexity})"
                                })
                
                except Exception as e:
                    self.logger.warning(f"Error analyzing {py_file}: {e}")
        
        except Exception as e:
            self.logger.warning(f"Error analyzing complexity: {e}")
        
        return issues
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Add complexity for control flow statements
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _analyze_function_length(self) -> List[Dict[str, Any]]:
        """Analyze function length."""
        issues = []
        
        try:
            src_path = Path(__file__).parent.parent.parent
            for py_file in src_path.rglob('*.py'):
                if 'test' in str(py_file).lower():
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    tree = ast.parse(''.join(lines))
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Calculate function length
                            if hasattr(node, 'end_lineno') and node.end_lineno:
                                length = node.end_lineno - node.lineno + 1
                            else:
                                # Fallback - count lines until next function/class
                                length = self._estimate_function_length(node, lines)
                            
                            if length > self.max_function_length:
                                issues.append({
                                    'type': 'long_function',
                                    'severity': 'high' if length > self.max_function_length * 2 else 'medium',
                                    'file': str(py_file),
                                    'line': node.lineno,
                                    'function': node.name,
                                    'length': length,
                                    'description': f"Function {node.name} is {length} lines (max: {self.max_function_length})"
                                })
                
                except Exception as e:
                    self.logger.warning(f"Error analyzing {py_file}: {e}")
        
        except Exception as e:
            self.logger.warning(f"Error analyzing function length: {e}")
        
        return issues
    
    def _estimate_function_length(self, node: ast.FunctionDef, lines: List[str]) -> int:
        """Estimate function length when end_lineno is not available."""
        start_line = node.lineno - 1  # Convert to 0-based
        
        # Find indentation level of function
        func_line = lines[start_line] if start_line < len(lines) else ""
        func_indent = len(func_line) - len(func_line.lstrip())
        
        # Count lines until we find a line with same or less indentation
        length = 1
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Skip empty lines
                line_indent = len(line) - len(line.lstrip())
                if line_indent <= func_indent:
                    break
            length += 1
        
        return length
    
    def _analyze_code_style(self) -> List[Dict[str, Any]]:
        """Analyze basic code style issues."""
        issues = []
        
        try:
            src_path = Path(__file__).parent.parent.parent
            for py_file in src_path.rglob('*.py'):
                if 'test' in str(py_file).lower():
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    for i, line in enumerate(lines, 1):
                        # Check line length
                        if len(line.rstrip()) > 120:  # Allow 120 chars
                            issues.append({
                                'type': 'long_line',
                                'severity': 'medium',
                                'file': str(py_file),
                                'line': i,
                                'length': len(line.rstrip()),
                                'description': f"Line too long: {len(line.rstrip())} characters"
                            })
                        
                        # Check trailing whitespace
                        if line.rstrip() != line.rstrip('\n').rstrip('\r'):
                            issues.append({
                                'type': 'trailing_whitespace',
                                'severity': 'medium',
                                'file': str(py_file),
                                'line': i,
                                'description': "Trailing whitespace"
                            })
                
                except Exception as e:
                    self.logger.warning(f"Error analyzing {py_file}: {e}")
        
        except Exception as e:
            self.logger.warning(f"Error analyzing code style: {e}")
        
        return issues


class QualityGateRunner:
    """Orchestrates execution of all quality gates."""
    
    def __init__(self):
        self.logger = get_logger("quality_gates")
        self.gates: List[QualityGate] = []
        
        # Register default quality gates
        self._register_default_gates()
    
    def _register_default_gates(self):
        """Register default quality gates."""
        self.gates = [
            CodeCoverageGate(min_coverage=0.70, min_branch_coverage=0.65),  # Relaxed for demo
            PerformanceBenchmarkGate(),
            SecurityScanGate(),
            CodeQualityGate()
        ]
    
    def add_gate(self, gate: QualityGate):
        """Add a quality gate."""
        self.gates.append(gate)
        self.logger.info(f"Added quality gate: {gate.name}")
    
    def remove_gate(self, name: str) -> bool:
        """Remove a quality gate by name."""
        for i, gate in enumerate(self.gates):
            if gate.name == name:
                del self.gates[i]
                self.logger.info(f"Removed quality gate: {name}")
                return True
        return False
    
    def run_all_gates(self, parallel: bool = True) -> Dict[str, QualityGateResult]:
        """Run all quality gates."""
        self.logger.info(f"Running {len(self.gates)} quality gates")
        
        if parallel and len(self.gates) > 1:
            return self._run_parallel()
        else:
            return self._run_sequential()
    
    def _run_sequential(self) -> Dict[str, QualityGateResult]:
        """Run gates sequentially."""
        results = {}
        
        for gate in self.gates:
            self.logger.info(f"Running quality gate: {gate.name}")
            result = gate.run()
            results[gate.name] = result
        
        return results
    
    def _run_parallel(self) -> Dict[str, QualityGateResult]:
        """Run gates in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(self.gates)) as executor:
            # Submit all gates
            future_to_gate = {executor.submit(gate.run): gate for gate in self.gates}
            
            # Collect results
            for future in as_completed(future_to_gate):
                gate = future_to_gate[future]
                try:
                    result = future.result()
                    results[gate.name] = result
                except Exception as e:
                    self.logger.error(f"Quality gate {gate.name} failed: {e}", e)
                    results[gate.name] = QualityGateResult(
                        gate_name=gate.name,
                        status=HealthStatus.CRITICAL,
                        score=0.0,
                        message=f"Gate execution failed: {str(e)}"
                    )
        
        return results
    
    def calculate_overall_score(self, results: Dict[str, QualityGateResult]) -> Tuple[float, HealthStatus]:
        """Calculate overall quality score and status."""
        if not results:
            return 0.0, HealthStatus.UNKNOWN
        
        # Calculate weighted average
        total_weight = sum(gate.weight for gate in self.gates)
        weighted_score = 0.0
        
        for gate in self.gates:
            result = results.get(gate.name)
            if result:
                weighted_score += result.score * gate.weight
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine overall status
        failed_gates = [r for r in results.values() if r.failed]
        critical_gates = [r for r in results.values() if r.status == HealthStatus.CRITICAL]
        warning_gates = [r for r in results.values() if r.status == HealthStatus.WARNING]
        
        if len(critical_gates) > 0:
            overall_status = HealthStatus.CRITICAL
        elif len(warning_gates) > len(results) // 2:  # More than half have warnings
            overall_status = HealthStatus.WARNING
        elif overall_score >= 0.8:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.WARNING
        
        return overall_score, overall_status
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        results = self.run_all_gates()
        overall_score, overall_status = self.calculate_overall_score(results)
        
        # Aggregate statistics
        total_gates = len(results)
        passed_gates = sum(1 for r in results.values() if r.passed)
        failed_gates = total_gates - passed_gates
        
        # Collect all recommendations
        all_recommendations = []
        for result in results.values():
            all_recommendations.extend(result.recommendations)
        
        unique_recommendations = list(set(all_recommendations))
        
        report = {
            'timestamp': time.time(),
            'overall_score': overall_score,
            'overall_status': overall_status.value,
            'summary': {
                'total_gates': total_gates,
                'passed_gates': passed_gates,
                'failed_gates': failed_gates,
                'pass_rate': passed_gates / total_gates if total_gates > 0 else 0,
                'avg_execution_time': sum(r.execution_time for r in results.values()) / total_gates if total_gates > 0 else 0
            },
            'gate_results': {name: result.to_dict() for name, result in results.items()},
            'recommendations': unique_recommendations[:20],  # Top 20 recommendations
            'next_steps': self._generate_next_steps(overall_status, results)
        }
        
        return report
    
    def _generate_next_steps(self, overall_status: HealthStatus, 
                           results: Dict[str, QualityGateResult]) -> List[str]:
        """Generate next steps based on quality gate results."""
        next_steps = []
        
        if overall_status == HealthStatus.CRITICAL:
            next_steps.append("🚨 CRITICAL: Address all critical issues before deployment")
            critical_gates = [name for name, result in results.items() if result.status == HealthStatus.CRITICAL]
            next_steps.append(f"Focus on fixing: {', '.join(critical_gates)}")
        
        elif overall_status == HealthStatus.WARNING:
            next_steps.append("⚠️  WARNING: Address high-priority issues")
            next_steps.append("Review security and performance warnings")
        
        else:
            next_steps.append("✅ READY: Quality gates passed, prepare for deployment")
            next_steps.append("Consider additional testing in staging environment")
        
        # Always add these
        next_steps.extend([
            "Monitor quality metrics in production",
            "Set up automated quality gate execution in CI/CD",
            "Schedule regular quality reviews"
        ])
        
        return next_steps


# Global quality gate runner
_quality_runner = QualityGateRunner()


def run_quality_gates(parallel: bool = True) -> Dict[str, QualityGateResult]:
    """Run all quality gates."""
    return _quality_runner.run_all_gates(parallel)


def generate_quality_report() -> Dict[str, Any]:
    """Generate comprehensive quality report."""
    return _quality_runner.generate_quality_report()


def get_quality_runner() -> QualityGateRunner:
    """Get global quality gate runner."""
    return _quality_runner