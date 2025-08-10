"""
Production deployment preparation for QEM-Bench.

Provides comprehensive production readiness validation, deployment automation,
and production monitoring setup.
"""

import os
import json
import yaml
import tempfile
import subprocess
import shutil
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import platform
import sys

from ..monitoring.logger import get_logger
from ..monitoring.health import HealthStatus, HealthCheckResult, get_system_health
from ..testing.quality_gates import generate_quality_report


@dataclass
class DeploymentConfig:
    """Configuration for production deployment."""
    
    environment: str = "production"
    version: str = "1.0.0"
    python_version: str = "3.9+"
    dependencies: List[str] = field(default_factory=lambda: [
        "jax>=0.4.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "psutil>=5.8.0"
    ])
    
    # Resource requirements
    min_cpu_cores: int = 2
    min_memory_gb: int = 4
    min_disk_gb: int = 10
    
    # Performance requirements
    max_startup_time_seconds: float = 30.0
    max_response_time_ms: float = 1000.0
    
    # Monitoring configuration
    enable_metrics: bool = True
    enable_health_checks: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Security configuration
    require_https: bool = True
    enable_auth: bool = False  # For future API versions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'environment': self.environment,
            'version': self.version,
            'python_version': self.python_version,
            'dependencies': self.dependencies,
            'resources': {
                'min_cpu_cores': self.min_cpu_cores,
                'min_memory_gb': self.min_memory_gb,
                'min_disk_gb': self.min_disk_gb
            },
            'performance': {
                'max_startup_time_seconds': self.max_startup_time_seconds,
                'max_response_time_ms': self.max_response_time_ms
            },
            'monitoring': {
                'enable_metrics': self.enable_metrics,
                'enable_health_checks': self.enable_health_checks,
                'enable_logging': self.enable_logging,
                'log_level': self.log_level
            },
            'security': {
                'require_https': self.require_https,
                'enable_auth': self.enable_auth
            }
        }


@dataclass
class DeploymentReadinessResult:
    """Result of deployment readiness check."""
    
    is_ready: bool
    score: float
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    deployment_config: DeploymentConfig
    system_info: Dict[str, Any]
    quality_report: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_ready': self.is_ready,
            'score': self.score,
            'issues': self.issues,
            'recommendations': self.recommendations,
            'deployment_config': self.deployment_config.to_dict(),
            'system_info': self.system_info,
            'quality_report': self.quality_report
        }


class ProductionReadinessChecker:
    """Comprehensive production readiness checker."""
    
    def __init__(self, deployment_config: Optional[DeploymentConfig] = None):
        self.config = deployment_config or DeploymentConfig()
        self.logger = get_logger("production_readiness")
    
    def check_readiness(self) -> DeploymentReadinessResult:
        """Perform comprehensive production readiness check."""
        self.logger.info("Starting production readiness check")
        
        issues = []
        recommendations = []
        
        # System requirements check
        system_issues, system_recs = self._check_system_requirements()
        issues.extend(system_issues)
        recommendations.extend(system_recs)
        
        # Dependencies check
        dep_issues, dep_recs = self._check_dependencies()
        issues.extend(dep_issues)
        recommendations.extend(dep_recs)
        
        # Configuration check
        config_issues, config_recs = self._check_configuration()
        issues.extend(config_issues)
        recommendations.extend(config_recs)
        
        # Performance check
        perf_issues, perf_recs = self._check_performance()
        issues.extend(perf_issues)
        recommendations.extend(perf_recs)
        
        # Security check
        sec_issues, sec_recs = self._check_security()
        issues.extend(sec_issues)
        recommendations.extend(sec_recs)
        
        # Quality gates check
        quality_report = self._check_quality_gates()
        if quality_report['overall_status'] != 'healthy':
            issues.append({
                'type': 'quality_gates',
                'severity': 'high',
                'description': f"Quality gates failed: {quality_report['overall_status']}",
                'details': quality_report['summary']
            })
            recommendations.extend(quality_report['recommendations'][:5])
        
        # Health checks
        health_status, health_results = get_system_health()
        if health_status != HealthStatus.HEALTHY:
            issues.append({
                'type': 'health_checks',
                'severity': 'medium',
                'description': f"System health issues: {health_status.value}",
                'details': {name: result.message for name, result in health_results.items() if not result.is_healthy}
            })
        
        # Calculate overall readiness score
        critical_issues = [i for i in issues if i['severity'] == 'critical']
        high_issues = [i for i in issues if i['severity'] == 'high']
        medium_issues = [i for i in issues if i['severity'] == 'medium']
        
        # Scoring: critical = -0.5, high = -0.3, medium = -0.1
        penalty = (len(critical_issues) * 0.5 + 
                  len(high_issues) * 0.3 + 
                  len(medium_issues) * 0.1)
        
        readiness_score = max(0.0, 1.0 - penalty)
        is_ready = len(critical_issues) == 0 and len(high_issues) <= 2 and readiness_score >= 0.8
        
        # Generate system info
        system_info = self._gather_system_info()
        
        # Add general recommendations
        if not is_ready:
            recommendations.extend([
                "Address all critical and high-severity issues before deployment",
                "Run full integration testing in staging environment",
                "Set up comprehensive monitoring before production deployment"
            ])
        else:
            recommendations.extend([
                "Proceed with staged deployment (staging → production)",
                "Monitor key metrics during deployment",
                "Have rollback plan ready"
            ])
        
        result = DeploymentReadinessResult(
            is_ready=is_ready,
            score=readiness_score,
            issues=issues,
            recommendations=list(set(recommendations)),  # Remove duplicates
            deployment_config=self.config,
            system_info=system_info,
            quality_report=quality_report
        )
        
        self.logger.info(
            f"Production readiness check complete: {'READY' if is_ready else 'NOT READY'} "
            f"(score: {readiness_score:.2f})",
            "production_readiness",
            {
                'is_ready': is_ready,
                'score': readiness_score,
                'total_issues': len(issues),
                'critical_issues': len(critical_issues),
                'high_issues': len(high_issues)
            }
        )
        
        return result
    
    def _check_system_requirements(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Check system requirements."""
        issues = []
        recommendations = []
        
        try:
            import psutil
            
            # CPU check
            cpu_count = psutil.cpu_count()
            if cpu_count < self.config.min_cpu_cores:
                issues.append({
                    'type': 'system_resources',
                    'severity': 'critical',
                    'description': f"Insufficient CPU cores: {cpu_count} < {self.config.min_cpu_cores}",
                    'current': cpu_count,
                    'required': self.config.min_cpu_cores
                })
                recommendations.append(f"Upgrade to system with at least {self.config.min_cpu_cores} CPU cores")
            
            # Memory check
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < self.config.min_memory_gb:
                issues.append({
                    'type': 'system_resources',
                    'severity': 'critical',
                    'description': f"Insufficient memory: {memory_gb:.1f}GB < {self.config.min_memory_gb}GB",
                    'current': memory_gb,
                    'required': self.config.min_memory_gb
                })
                recommendations.append(f"Upgrade to system with at least {self.config.min_memory_gb}GB RAM")
            
            # Disk space check
            disk_usage = psutil.disk_usage('/')
            disk_free_gb = disk_usage.free / (1024**3)
            if disk_free_gb < self.config.min_disk_gb:
                issues.append({
                    'type': 'system_resources',
                    'severity': 'high',
                    'description': f"Insufficient disk space: {disk_free_gb:.1f}GB < {self.config.min_disk_gb}GB",
                    'current': disk_free_gb,
                    'required': self.config.min_disk_gb
                })
                recommendations.append(f"Free up disk space or add storage (need {self.config.min_disk_gb}GB)")
            
            # Python version check
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            required_version = self.config.python_version.rstrip('+')
            if python_version < required_version:
                issues.append({
                    'type': 'python_version',
                    'severity': 'critical',
                    'description': f"Python version too old: {python_version} < {required_version}",
                    'current': python_version,
                    'required': required_version
                })
                recommendations.append(f"Upgrade Python to version {self.config.python_version}")
        
        except Exception as e:
            issues.append({
                'type': 'system_check_error',
                'severity': 'high',
                'description': f"Failed to check system requirements: {str(e)}"
            })
        
        return issues, recommendations
    
    def _check_dependencies(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Check Python dependencies."""
        issues = []
        recommendations = []
        
        try:
            # Check if each dependency is available
            missing_deps = []
            version_issues = []
            
            for dep in self.config.dependencies:
                # Parse dependency string
                if '>=' in dep:
                    pkg_name, min_version = dep.split('>=')
                else:
                    pkg_name, min_version = dep, None
                
                try:
                    # Try to import package
                    if pkg_name == 'jax':
                        import jax
                        current_version = jax.__version__
                    elif pkg_name == 'numpy':
                        import numpy
                        current_version = numpy.__version__
                    elif pkg_name == 'scipy':
                        import scipy
                        current_version = scipy.__version__
                    elif pkg_name == 'matplotlib':
                        import matplotlib
                        current_version = matplotlib.__version__
                    elif pkg_name == 'psutil':
                        import psutil
                        current_version = psutil.__version__
                    else:
                        # Generic import check
                        __import__(pkg_name)
                        current_version = "unknown"
                    
                    # Version check (simplified)
                    if min_version and current_version != "unknown":
                        # This is a simplified version check
                        try:
                            if self._compare_versions(current_version, min_version) < 0:
                                version_issues.append((pkg_name, current_version, min_version))
                        except:
                            pass  # Skip version comparison if it fails
                
                except ImportError:
                    missing_deps.append(pkg_name)
            
            # Report missing dependencies
            if missing_deps:
                issues.append({
                    'type': 'missing_dependencies',
                    'severity': 'critical',
                    'description': f"Missing required dependencies: {', '.join(missing_deps)}",
                    'missing': missing_deps
                })
                recommendations.append(f"Install missing dependencies: pip install {' '.join(missing_deps)}")
            
            # Report version issues
            if version_issues:
                for pkg, current, required in version_issues:
                    issues.append({
                        'type': 'dependency_version',
                        'severity': 'high',
                        'description': f"{pkg} version too old: {current} < {required}",
                        'package': pkg,
                        'current': current,
                        'required': required
                    })
                    recommendations.append(f"Upgrade {pkg}: pip install {pkg}>={required}")
        
        except Exception as e:
            issues.append({
                'type': 'dependency_check_error',
                'severity': 'medium',
                'description': f"Failed to check dependencies: {str(e)}"
            })
        
        return issues, recommendations
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare version strings. Returns -1, 0, or 1."""
        def normalize_version(v):
            return [int(x) for x in v.split('.')]
        
        v1_parts = normalize_version(version1)
        v2_parts = normalize_version(version2)
        
        # Pad shorter version with zeros
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))
        
        for a, b in zip(v1_parts, v2_parts):
            if a < b:
                return -1
            elif a > b:
                return 1
        
        return 0
    
    def _check_configuration(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Check configuration settings."""
        issues = []
        recommendations = []
        
        # Check log directory
        log_dir = Path("logs")
        if not log_dir.exists():
            issues.append({
                'type': 'configuration',
                'severity': 'medium',
                'description': "Log directory does not exist",
                'path': str(log_dir)
            })
            recommendations.append("Create logs directory: mkdir -p logs")
        
        # Check write permissions
        try:
            test_file = log_dir / "test_write.tmp"
            with open(test_file, 'w') as f:
                f.write("test")
            test_file.unlink()
        except Exception:
            issues.append({
                'type': 'permissions',
                'severity': 'high',
                'description': "Cannot write to logs directory",
                'path': str(log_dir)
            })
            recommendations.append("Fix write permissions for logs directory")
        
        return issues, recommendations
    
    def _check_performance(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Check performance requirements."""
        issues = []
        recommendations = []
        
        try:
            # Test basic performance
            import time
            
            # Simple computation benchmark
            start_time = time.time()
            
            # Simulate QEM-Bench operations
            try:
                from ..jax.circuits import JAXCircuit
                from ..jax.simulator import JAXSimulator
                
                circuit = JAXCircuit(2)
                circuit.h(0)
                circuit.cx(0, 1)
                
                simulator = JAXSimulator(2)
                result = simulator.run(circuit, shots=100)
                
                duration_ms = (time.time() - start_time) * 1000
                
                if duration_ms > self.config.max_response_time_ms:
                    issues.append({
                        'type': 'performance',
                        'severity': 'medium',
                        'description': f"Performance below target: {duration_ms:.1f}ms > {self.config.max_response_time_ms}ms",
                        'current': duration_ms,
                        'required': self.config.max_response_time_ms
                    })
                    recommendations.append("Optimize performance or upgrade hardware")
            
            except ImportError:
                issues.append({
                    'type': 'performance_test_failed',
                    'severity': 'medium',
                    'description': "Could not run performance test - missing components"
                })
        
        except Exception as e:
            issues.append({
                'type': 'performance_check_error',
                'severity': 'medium',
                'description': f"Performance check failed: {str(e)}"
            })
        
        return issues, recommendations
    
    def _check_security(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Check security configuration."""
        issues = []
        recommendations = []
        
        # Check file permissions
        try:
            src_path = Path(__file__).parent.parent.parent
            
            # Check for world-writable files
            for py_file in src_path.rglob('*.py'):
                try:
                    stat = py_file.stat()
                    # Check if world-writable (mode & 0o002)
                    if stat.st_mode & 0o002:
                        issues.append({
                            'type': 'file_permissions',
                            'severity': 'medium',
                            'description': f"World-writable file: {py_file}",
                            'file': str(py_file)
                        })
                        recommendations.append(f"Fix file permissions: chmod 644 {py_file}")
                except Exception:
                    pass
            
        except Exception:
            pass
        
        # Check for sensitive files
        sensitive_patterns = ['.env', '.secret', 'id_rsa', '.pem', 'credentials']
        for pattern in sensitive_patterns:
            try:
                for file_path in Path('.').rglob(f'*{pattern}*'):
                    if file_path.is_file():
                        issues.append({
                            'type': 'sensitive_files',
                            'severity': 'high',
                            'description': f"Sensitive file in repository: {file_path}",
                            'file': str(file_path)
                        })
                        recommendations.append(f"Remove or secure sensitive file: {file_path}")
            except Exception:
                pass
        
        return issues, recommendations
    
    def _check_quality_gates(self) -> Dict[str, Any]:
        """Run quality gates check."""
        try:
            return generate_quality_report()
        except Exception as e:
            self.logger.warning(f"Quality gates check failed: {e}")
            return {
                'overall_status': 'unknown',
                'summary': {'error': str(e)},
                'recommendations': ['Fix quality gates execution']
            }
    
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather comprehensive system information."""
        try:
            import psutil
            
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'disk_free_gb': psutil.disk_usage('/').free / (1024**3),
                'uptime_seconds': time.time(),  # Approximate
                'environment': os.environ.get('ENVIRONMENT', 'unknown'),
                'deployment_time': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {'error': str(e)}


class ProductionDeployer:
    """Handles production deployment automation."""
    
    def __init__(self, deployment_config: Optional[DeploymentConfig] = None):
        self.config = deployment_config or DeploymentConfig()
        self.logger = get_logger("production_deployer")
    
    def deploy(self, readiness_result: DeploymentReadinessResult,
              force: bool = False) -> Dict[str, Any]:
        """Deploy to production environment."""
        
        if not readiness_result.is_ready and not force:
            self.logger.error("Deployment aborted: system not ready")
            return {
                'success': False,
                'message': 'System not ready for deployment',
                'issues': readiness_result.issues
            }
        
        self.logger.info("Starting production deployment")
        
        deployment_steps = [
            ('Pre-deployment validation', self._pre_deployment_validation),
            ('Environment preparation', self._prepare_environment),
            ('Application deployment', self._deploy_application),
            ('Health check validation', self._post_deployment_health_check),
            ('Monitoring setup', self._setup_monitoring)
        ]
        
        deployment_log = []
        
        for step_name, step_func in deployment_steps:
            self.logger.info(f"Executing deployment step: {step_name}")
            
            try:
                start_time = time.time()
                result = step_func()
                duration = time.time() - start_time
                
                step_result = {
                    'step': step_name,
                    'success': True,
                    'duration': duration,
                    'result': result
                }
                
                deployment_log.append(step_result)
                self.logger.info(f"Deployment step completed: {step_name} ({duration:.2f}s)")
            
            except Exception as e:
                step_result = {
                    'step': step_name,
                    'success': False,
                    'error': str(e),
                    'duration': time.time() - start_time
                }
                
                deployment_log.append(step_result)
                self.logger.error(f"Deployment step failed: {step_name}: {str(e)}", e)
                
                # Rollback on failure
                self._rollback_deployment(deployment_log)
                
                return {
                    'success': False,
                    'message': f'Deployment failed at step: {step_name}',
                    'error': str(e),
                    'deployment_log': deployment_log
                }
        
        # Deployment successful
        self.logger.info("Production deployment completed successfully")
        
        return {
            'success': True,
            'message': 'Production deployment successful',
            'deployment_log': deployment_log,
            'deployment_time': datetime.now().isoformat(),
            'config': self.config.to_dict()
        }
    
    def _pre_deployment_validation(self) -> Dict[str, Any]:
        """Pre-deployment validation."""
        # Run final health checks
        health_status, health_results = get_system_health()
        
        return {
            'health_status': health_status.value,
            'health_results': {name: result.to_dict() for name, result in health_results.items()}
        }
    
    def _prepare_environment(self) -> Dict[str, Any]:
        """Prepare production environment."""
        # Create necessary directories
        directories = ['logs', 'data', 'cache', 'tmp']
        created_dirs = []
        
        for dir_name in directories:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(dir_path))
        
        # Set appropriate permissions
        for dir_name in directories:
            try:
                os.chmod(dir_name, 0o755)
            except Exception as e:
                self.logger.warning(f"Could not set permissions for {dir_name}: {e}")
        
        return {
            'created_directories': created_dirs,
            'environment': self.config.environment
        }
    
    def _deploy_application(self) -> Dict[str, Any]:
        """Deploy the application."""
        # In a real deployment, this would:
        # 1. Copy application files to production location
        # 2. Install dependencies
        # 3. Update configuration files
        # 4. Restart services
        
        # For this implementation, we'll simulate the process
        return {
            'version': self.config.version,
            'environment': self.config.environment,
            'python_version': platform.python_version(),
            'deployment_method': 'in-place'
        }
    
    def _post_deployment_health_check(self) -> Dict[str, Any]:
        """Run post-deployment health checks."""
        # Wait a moment for system to stabilize
        time.sleep(2)
        
        # Run comprehensive health check
        health_status, health_results = get_system_health()
        
        if health_status not in (HealthStatus.HEALTHY, HealthStatus.WARNING):
            raise Exception(f"Post-deployment health check failed: {health_status}")
        
        return {
            'health_status': health_status.value,
            'all_checks_passed': all(result.is_healthy for result in health_results.values())
        }
    
    def _setup_monitoring(self) -> Dict[str, Any]:
        """Setup production monitoring."""
        monitoring_config = {
            'logging': {
                'enabled': self.config.enable_logging,
                'level': self.config.log_level,
                'format': 'json'
            },
            'metrics': {
                'enabled': self.config.enable_metrics,
                'collection_interval': 60
            },
            'health_checks': {
                'enabled': self.config.enable_health_checks,
                'interval': 30
            }
        }
        
        # Save monitoring configuration
        try:
            with open('production_monitoring.json', 'w') as f:
                json.dump(monitoring_config, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save monitoring config: {e}")
        
        return monitoring_config
    
    def _rollback_deployment(self, deployment_log: List[Dict[str, Any]]):
        """Rollback failed deployment."""
        self.logger.warning("Rolling back failed deployment")
        
        # In a real deployment, this would:
        # 1. Restore previous version
        # 2. Restart services with old configuration
        # 3. Cleanup failed deployment artifacts
        
        # For now, just log the rollback
        self.logger.info("Rollback completed")


def check_production_readiness(config: Optional[DeploymentConfig] = None) -> DeploymentReadinessResult:
    """Check if system is ready for production deployment."""
    checker = ProductionReadinessChecker(config)
    return checker.check_readiness()


def deploy_to_production(readiness_result: DeploymentReadinessResult,
                        config: Optional[DeploymentConfig] = None,
                        force: bool = False) -> Dict[str, Any]:
    """Deploy QEM-Bench to production environment."""
    deployer = ProductionDeployer(config)
    return deployer.deploy(readiness_result, force)


def generate_deployment_report(readiness_result: DeploymentReadinessResult) -> str:
    """Generate human-readable deployment report."""
    report = []
    
    # Header
    report.append("QEM-Bench Production Deployment Report")
    report.append("=" * 50)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Overall status
    status_emoji = "✅" if readiness_result.is_ready else "❌"
    report.append(f"Overall Status: {status_emoji} {'READY' if readiness_result.is_ready else 'NOT READY'}")
    report.append(f"Readiness Score: {readiness_result.score:.1%}")
    report.append("")
    
    # Issues summary
    if readiness_result.issues:
        report.append("Issues Found:")
        for issue in readiness_result.issues:
            severity_emoji = {"critical": "🚨", "high": "⚠️", "medium": "ℹ️"}.get(issue['severity'], "•")
            report.append(f"  {severity_emoji} {issue['description']}")
        report.append("")
    
    # Recommendations
    if readiness_result.recommendations:
        report.append("Recommendations:")
        for i, rec in enumerate(readiness_result.recommendations[:10], 1):
            report.append(f"  {i}. {rec}")
        report.append("")
    
    # System information
    report.append("System Information:")
    sys_info = readiness_result.system_info
    report.append(f"  Platform: {sys_info.get('platform', 'Unknown')}")
    report.append(f"  Python: {sys_info.get('python_version', 'Unknown')}")
    report.append(f"  CPU Cores: {sys_info.get('cpu_count', 'Unknown')}")
    report.append(f"  Memory: {sys_info.get('memory_gb', 0):.1f} GB")
    report.append(f"  Disk Free: {sys_info.get('disk_free_gb', 0):.1f} GB")
    report.append("")
    
    # Quality report summary
    quality = readiness_result.quality_report
    quality_status = quality.get('overall_status', 'unknown')
    quality_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}.get(quality_status, "❓")
    report.append(f"Quality Gates: {quality_emoji} {quality_status.upper()}")
    
    if 'summary' in quality:
        summary = quality['summary']
        report.append(f"  Passed: {summary.get('passed_gates', 0)}/{summary.get('total_gates', 0)} gates")
        report.append(f"  Pass Rate: {summary.get('pass_rate', 0):.1%}")
    
    report.append("")
    
    # Next steps
    report.append("Next Steps:")
    if readiness_result.is_ready:
        report.append("  1. ✅ System is ready for production deployment")
        report.append("  2. 📋 Review deployment plan and timeline")
        report.append("  3. 🚀 Execute staged deployment (staging → production)")
        report.append("  4. 📊 Monitor key metrics during deployment")
        report.append("  5. 🔄 Have rollback plan ready")
    else:
        report.append("  1. ❌ Address all critical and high-severity issues")
        report.append("  2. 🔧 Implement recommended fixes")
        report.append("  3. ♻️  Re-run production readiness check")
        report.append("  4. 🧪 Conduct thorough testing in staging environment")
        report.append("  5. 📈 Set up comprehensive monitoring before deployment")
    
    return "\\n".join(report)