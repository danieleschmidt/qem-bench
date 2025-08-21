"""Dependency checking for QEM-Bench health monitoring."""

import importlib
import subprocess
import sys
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from packaging import version
import importlib.metadata as importlib_metadata

from .health_checker import HealthCheckProvider, HealthCheck, HealthStatus


logger = logging.getLogger(__name__)


class DependencyChecker(HealthCheckProvider):
    """
    Health check provider for Python dependencies and system requirements.
    
    Checks:
    - Required Python packages and versions
    - Optional packages for enhanced functionality
    - Python version compatibility
    - System-level dependencies
    """
    
    def __init__(self):
        # Define required dependencies with minimum versions
        self.required_deps = {
            'numpy': '1.20.0',
            'jax': '0.4.0',
            'jaxlib': '0.4.0',
            'scipy': '1.7.0',
            'matplotlib': '3.3.0',
            'psutil': '5.8.0'
        }
        
        # Optional dependencies for enhanced functionality
        self.optional_deps = {
            'qiskit': '0.40.0',
            'cirq': '1.0.0',
            'pennylane': '0.28.0',
            'GPUtil': '1.4.0',
            'cuda': None,  # System level
            'cupy': '10.0.0',
            'tensorboard': '2.8.0',
            'jupyter': '1.0.0',
            'ipython': '7.0.0'
        }
        
        # Python version requirements
        self.min_python_version = (3, 8)
        self.max_python_version = (3, 12)
    
    def get_name(self) -> str:
        """Get the name of this health check provider."""
        return "dependency_checker"
    
    def is_critical(self) -> bool:
        """Dependency checks are critical for system operation."""
        return True
    
    def check_health(self) -> HealthCheck:
        """Perform dependency health check."""
        start_time = time.time()
        issues = []
        warnings = []
        details = {}
        recommendations = []
        
        # Check Python version
        python_status = self._check_python_version()
        details['python'] = python_status
        if python_status['status'] == 'error':
            issues.append(python_status['message'])
            recommendations.extend(python_status.get('recommendations', []))
        elif python_status['status'] == 'warning':
            warnings.append(python_status['message'])
        
        # Check required dependencies
        required_status = self._check_required_dependencies()
        details['required_dependencies'] = required_status
        for dep_name, dep_info in required_status.items():
            if dep_info['status'] == 'error':
                issues.append(f"Required dependency {dep_name}: {dep_info['message']}")
                recommendations.extend(dep_info.get('recommendations', []))
            elif dep_info['status'] == 'warning':
                warnings.append(f"Required dependency {dep_name}: {dep_info['message']}")
        
        # Check optional dependencies
        optional_status = self._check_optional_dependencies()
        details['optional_dependencies'] = optional_status
        for dep_name, dep_info in optional_status.items():
            if dep_info['status'] == 'warning':
                warnings.append(f"Optional dependency {dep_name}: {dep_info['message']}")
        
        # Check JAX/GPU setup
        jax_status = self._check_jax_setup()
        details['jax_setup'] = jax_status
        if jax_status['status'] == 'warning':
            warnings.append(f"JAX setup: {jax_status['message']}")
            recommendations.extend(jax_status.get('recommendations', []))
        
        # Determine overall status
        if issues:
            status = HealthStatus.UNHEALTHY
            message = f"Dependency issues found: {len(issues)} critical, {len(warnings)} warnings"
        elif warnings:
            status = HealthStatus.WARNING
            message = f"Dependency warnings: {len(warnings)} items need attention"
        else:
            status = HealthStatus.HEALTHY
            message = "All dependencies are properly configured"
        
        # Add general recommendations
        if issues or warnings:
            recommendations.extend([
                "Run 'pip install --upgrade qem-bench[all]' to update dependencies",
                "Check the installation guide for system-specific requirements"
            ])
        
        return HealthCheck(
            name=self.get_name(),
            status=status,
            timestamp=start_time,
            duration=time.time() - start_time,
            message=message,
            details=details,
            recommendations=list(set(recommendations))  # Remove duplicates
        )
    
    def _check_python_version(self) -> Dict[str, Any]:
        """Check Python version compatibility."""
        current_version = sys.version_info[:2]
        
        if current_version < self.min_python_version:
            return {
                'status': 'error',
                'current_version': f"{current_version[0]}.{current_version[1]}",
                'required_version': f">={self.min_python_version[0]}.{self.min_python_version[1]}",
                'message': f"Python {current_version[0]}.{current_version[1]} is too old",
                'recommendations': [
                    f"Upgrade to Python {self.min_python_version[0]}.{self.min_python_version[1]} or later"
                ]
            }
        elif current_version > self.max_python_version:
            return {
                'status': 'warning',
                'current_version': f"{current_version[0]}.{current_version[1]}",
                'supported_version': f"<={self.max_python_version[0]}.{self.max_python_version[1]}",
                'message': f"Python {current_version[0]}.{current_version[1]} may have compatibility issues",
                'recommendations': [
                    "Test thoroughly or consider using a supported Python version"
                ]
            }
        else:
            return {
                'status': 'ok',
                'current_version': f"{current_version[0]}.{current_version[1]}",
                'message': "Python version is compatible"
            }
    
    def _check_required_dependencies(self) -> Dict[str, Dict[str, Any]]:
        """Check required dependencies."""
        results = {}
        
        for dep_name, min_version in self.required_deps.items():
            results[dep_name] = self._check_package(dep_name, min_version, required=True)
        
        return results
    
    def _check_optional_dependencies(self) -> Dict[str, Dict[str, Any]]:
        """Check optional dependencies."""
        results = {}
        
        for dep_name, min_version in self.optional_deps.items():
            results[dep_name] = self._check_package(dep_name, min_version, required=False)
        
        return results
    
    def _check_package(self, package_name: str, min_version: Optional[str] = None, 
                      required: bool = True) -> Dict[str, Any]:
        """Check a specific package."""
        try:
            # Try to import the package
            module = importlib.import_module(package_name)
            
            # Get version if available
            package_version = None
            if hasattr(module, '__version__'):
                package_version = module.__version__
            else:
                # Try to get from importlib.metadata
                try:
                    package_version = importlib_metadata.version(package_name)
                except (importlib_metadata.PackageNotFoundError, Exception):
                    pass
            
            # Check version if specified
            if min_version and package_version:
                try:
                    if version.parse(package_version) < version.parse(min_version):
                        return {
                            'status': 'warning' if not required else 'error',
                            'installed_version': package_version,
                            'required_version': f">={min_version}",
                            'message': f"Version {package_version} is older than required {min_version}",
                            'recommendations': [f"Upgrade {package_name} to version {min_version} or later"]
                        }
                except Exception as e:
                    logger.debug(f"Version comparison failed for {package_name}: {e}")
            
            return {
                'status': 'ok',
                'installed_version': package_version or 'unknown',
                'message': f"Package is installed and available"
            }
            
        except ImportError:
            if required:
                return {
                    'status': 'error',
                    'message': f"Required package {package_name} is not installed",
                    'recommendations': [f"Install {package_name} with: pip install {package_name}"]
                }
            else:
                return {
                    'status': 'warning',
                    'message': f"Optional package {package_name} is not installed",
                    'recommendations': [f"Install for enhanced functionality: pip install {package_name}"]
                }
        except Exception as e:
            return {
                'status': 'error' if required else 'warning',
                'message': f"Error checking {package_name}: {str(e)}",
                'recommendations': [f"Reinstall {package_name} or check installation"]
            }
    
    def _check_jax_setup(self) -> Dict[str, Any]:
        """Check JAX configuration and GPU availability."""
        try:
            import jax
            import jax.numpy as jnp
            
            # Test basic JAX functionality
            try:
                test_array = jnp.array([1, 2, 3])
                result = jnp.sum(test_array)
                float(result)  # Convert to Python float
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f"JAX basic functionality test failed: {str(e)}",
                    'recommendations': ["Reinstall JAX and JAXlib"]
                }
            
            # Check available devices
            devices = jax.devices()
            device_info = {
                'total_devices': len(devices),
                'device_types': {},
                'devices': []
            }
            
            for device in devices:
                device_type = device.device_kind
                device_info['device_types'][device_type] = device_info['device_types'].get(device_type, 0) + 1
                device_info['devices'].append({
                    'id': device.id,
                    'type': device_type,
                    'platform': device.platform
                })
            
            # Check GPU availability
            has_gpu = any(d.device_kind in ['gpu', 'tpu'] for d in devices)
            
            if has_gpu:
                message = f"JAX configured with {len(devices)} devices including GPU/TPU"
                status = 'ok'
                recommendations = []
            else:
                message = f"JAX configured with {len(devices)} CPU devices only"
                status = 'warning'
                recommendations = [
                    "Install CUDA-enabled JAX for GPU acceleration: pip install jax[cuda]",
                    "Ensure CUDA drivers are properly installed"
                ]
            
            return {
                'status': status,
                'message': message,
                'device_info': device_info,
                'recommendations': recommendations
            }
            
        except ImportError:
            return {
                'status': 'error',
                'message': "JAX is not installed",
                'recommendations': ["Install JAX: pip install jax jaxlib"]
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f"JAX setup check failed: {str(e)}",
                'recommendations': ["Check JAX installation and configuration"]
            }
    
    def get_dependency_report(self) -> Dict[str, Any]:
        """Get a detailed dependency report."""
        health_check = self.check_health()
        
        report = {
            'timestamp': health_check.timestamp,
            'overall_status': health_check.status.value,
            'python_info': {
                'version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'executable': sys.executable,
                'platform': sys.platform
            },
            'dependencies': health_check.details,
            'recommendations': health_check.recommendations
        }
        
        return report
    
    def install_missing_dependencies(self, optional: bool = False) -> List[str]:
        """
        Attempt to install missing dependencies.
        
        Args:
            optional: Whether to install optional dependencies
        
        Returns:
            List of installation results/errors
        """
        results = []
        deps_to_check = self.required_deps.copy()
        
        if optional:
            deps_to_check.update(self.optional_deps)
        
        for dep_name, min_version in deps_to_check.items():
            if dep_name == 'cuda':  # Skip system-level dependencies
                continue
                
            package_info = self._check_package(dep_name, min_version, required=(dep_name in self.required_deps))
            
            if package_info['status'] in ['error', 'warning'] and 'not installed' in package_info['message']:
                try:
                    install_cmd = [sys.executable, '-m', 'pip', 'install', dep_name]
                    if min_version:
                        install_cmd[-1] = f"{dep_name}>={min_version}"
                    
                    result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        results.append(f"Successfully installed {dep_name}")
                    else:
                        results.append(f"Failed to install {dep_name}: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    results.append(f"Installation of {dep_name} timed out")
                except Exception as e:
                    results.append(f"Error installing {dep_name}: {str(e)}")
        
        return results