"""Hardware capability detection for QEM-Bench health monitoring."""

import platform
import subprocess
import time
import logging
from typing import Dict, List, Optional, Any

from .health_checker import HealthCheckProvider, HealthCheck, HealthStatus


logger = logging.getLogger(__name__)


class HardwareDetector(HealthCheckProvider):
    """
    Health check provider for hardware capability detection.
    
    Detects:
    - CPU information and capabilities
    - GPU availability and specifications
    - Memory configuration
    - Storage information
    - Network capabilities
    """
    
    def get_name(self) -> str:
        """Get the name of this health check provider."""
        return "hardware_detector"
    
    def is_critical(self) -> bool:
        """Hardware detection is not critical but provides valuable info."""
        return False
    
    def check_health(self) -> HealthCheck:
        """Perform hardware capability detection."""
        start_time = time.time()
        warnings = []
        details = {}
        recommendations = []
        
        # CPU information
        cpu_info = self._detect_cpu()
        details['cpu'] = cpu_info
        if cpu_info.get('warnings'):
            warnings.extend(cpu_info['warnings'])
        
        # Memory information
        memory_info = self._detect_memory()
        details['memory'] = memory_info
        if memory_info.get('warnings'):
            warnings.extend(memory_info['warnings'])
        
        # GPU information
        gpu_info = self._detect_gpu()
        details['gpu'] = gpu_info
        if gpu_info.get('warnings'):
            warnings.extend(gpu_info['warnings'])
        if gpu_info.get('recommendations'):
            recommendations.extend(gpu_info['recommendations'])
        
        # Storage information
        storage_info = self._detect_storage()
        details['storage'] = storage_info
        if storage_info.get('warnings'):
            warnings.extend(storage_info['warnings'])
        
        # Network information
        network_info = self._detect_network()
        details['network'] = network_info
        
        # Overall assessment
        critical_issues = []
        
        # Check for performance concerns
        if memory_info.get('total_gb', 0) < 8:
            critical_issues.append("Low system memory (< 8GB) may impact performance")
            recommendations.append("Consider upgrading system memory for better performance")
        
        if cpu_info.get('core_count', 0) < 4:
            warnings.append("Low CPU core count may limit parallelization")
            recommendations.append("Consider using more CPU cores for parallel computations")
        
        # Determine status
        if critical_issues:
            status = HealthStatus.UNHEALTHY
            message = f"Hardware limitations detected: {len(critical_issues)} critical issues"
        elif warnings:
            status = HealthStatus.WARNING
            message = f"Hardware recommendations available: {len(warnings)} items"
        else:
            status = HealthStatus.HEALTHY
            message = "Hardware configuration is suitable for QEM-Bench"
        
        # Add capabilities summary
        capabilities = self._summarize_capabilities(details)
        details['capabilities'] = capabilities
        
        return HealthCheck(
            name=self.get_name(),
            status=status,
            timestamp=start_time,
            duration=time.time() - start_time,
            message=message,
            details=details,
            recommendations=list(set(recommendations))
        )
    
    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU information and capabilities."""
        cpu_info = {
            'architecture': platform.machine(),
            'platform': platform.system(),
            'processor': platform.processor()
        }
        
        try:
            import psutil
            
            # CPU count and frequency
            cpu_info['logical_cores'] = psutil.cpu_count(logical=True)
            cpu_info['physical_cores'] = psutil.cpu_count(logical=False)
            
            # CPU frequency
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_info['base_frequency_mhz'] = cpu_freq.current
                cpu_info['max_frequency_mhz'] = cpu_freq.max
            
            # CPU usage stats
            cpu_info['current_usage_percent'] = psutil.cpu_percent(interval=1)
            
        except ImportError:
            cpu_info['warning'] = "psutil not available for detailed CPU info"
        except Exception as e:
            cpu_info['error'] = f"Error getting CPU info: {str(e)}"
        
        # Platform-specific detection
        warnings = []
        if platform.system() == "Linux":
            try:
                # Try to get CPU model from /proc/cpuinfo
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            cpu_info['model_name'] = line.split(':')[1].strip()
                            break
            except:
                pass
        
        elif platform.system() == "Darwin":  # macOS
            try:
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    cpu_info['model_name'] = result.stdout.strip()
            except:
                pass
        
        elif platform.system() == "Windows":
            try:
                result = subprocess.run(['wmic', 'cpu', 'get', 'name'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        cpu_info['model_name'] = lines[1].strip()
            except:
                pass
        
        # Performance assessment
        if cpu_info.get('logical_cores', 0) < 4:
            warnings.append("Low CPU core count may limit parallel processing")
        
        if cpu_info.get('current_usage_percent', 0) > 80:
            warnings.append("High CPU usage detected during health check")
        
        if warnings:
            cpu_info['warnings'] = warnings
        
        return cpu_info
    
    def _detect_memory(self) -> Dict[str, Any]:
        """Detect memory information."""
        memory_info = {}
        warnings = []
        
        try:
            import psutil
            
            # Virtual memory
            vm = psutil.virtual_memory()
            memory_info['total_gb'] = vm.total / (1024**3)
            memory_info['available_gb'] = vm.available / (1024**3)
            memory_info['used_percent'] = vm.percent
            
            # Swap memory
            swap = psutil.swap_memory()
            memory_info['swap_total_gb'] = swap.total / (1024**3)
            memory_info['swap_used_percent'] = swap.percent
            
            # Performance assessment
            if memory_info['total_gb'] < 8:
                warnings.append("System has less than 8GB RAM")
            
            if memory_info['used_percent'] > 85:
                warnings.append("High memory usage detected")
            
            if memory_info['total_gb'] < 4:
                warnings.append("Very low memory may cause performance issues")
            
        except ImportError:
            memory_info['error'] = "psutil not available for memory info"
        except Exception as e:
            memory_info['error'] = f"Error getting memory info: {str(e)}"
        
        if warnings:
            memory_info['warnings'] = warnings
        
        return memory_info
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU information and capabilities."""
        gpu_info = {
            'available': False,
            'gpus': []
        }
        warnings = []
        recommendations = []
        
        # Try GPUtil first
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            if gpus:
                gpu_info['available'] = True
                gpu_info['count'] = len(gpus)
                
                for gpu in gpus:
                    gpu_data = {
                        'id': gpu.id,
                        'name': gpu.name,
                        'memory_total_gb': gpu.memoryTotal / 1024,
                        'memory_used_gb': gpu.memoryUsed / 1024,
                        'memory_usage_percent': (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                        'gpu_usage_percent': gpu.load * 100,
                        'temperature': gpu.temperature
                    }
                    gpu_info['gpus'].append(gpu_data)
                    
                    # Performance warnings
                    if gpu_data['memory_usage_percent'] > 90:
                        warnings.append(f"GPU {gpu.id} memory usage is very high")
                    
                    if gpu_data['gpu_usage_percent'] > 95:
                        warnings.append(f"GPU {gpu.id} utilization is very high")
            else:
                recommendations.append("No GPUs detected - consider GPU acceleration for faster computations")
                
        except ImportError:
            # Try alternative methods
            try:
                # NVIDIA-SMI check
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    gpu_info['available'] = True
                    gpus_data = []
                    
                    for i, line in enumerate(result.stdout.strip().split('\n')):
                        if line.strip():
                            parts = [part.strip() for part in line.split(',')]
                            if len(parts) >= 5:
                                gpu_data = {
                                    'id': i,
                                    'name': parts[0],
                                    'memory_total_gb': float(parts[1]) / 1024,
                                    'memory_used_gb': float(parts[2]) / 1024,
                                    'memory_usage_percent': (float(parts[2]) / float(parts[1]) * 100) if float(parts[1]) > 0 else 0,
                                    'gpu_usage_percent': float(parts[3]),
                                    'temperature': float(parts[4])
                                }
                                gpus_data.append(gpu_data)
                    
                    gpu_info['gpus'] = gpus_data
                    gpu_info['count'] = len(gpus_data)
                else:
                    recommendations.append("NVIDIA GPUs may be available but nvidia-smi failed")
                    
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                # Try checking for AMD GPUs
                try:
                    result = subprocess.run(['rocm-smi', '--showtemp', '--showmeminfo'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        gpu_info['available'] = True
                        gpu_info['type'] = 'AMD'
                        recommendations.append("AMD GPU detected - ensure ROCm support is configured")
                except:
                    pass
                
                if not gpu_info['available']:
                    recommendations.extend([
                        "No GPU acceleration detected",
                        "Install NVIDIA drivers and CUDA for GPU support",
                        "Consider using Google Colab or cloud instances with GPU acceleration"
                    ])
        
        except Exception as e:
            gpu_info['error'] = f"Error detecting GPU: {str(e)}"
        
        # JAX GPU compatibility check
        try:
            import jax
            jax_devices = jax.devices()
            gpu_devices = [d for d in jax_devices if d.device_kind in ['gpu', 'tpu']]
            
            if gpu_devices:
                gpu_info['jax_gpu_available'] = True
                gpu_info['jax_gpu_count'] = len(gpu_devices)
                gpu_info['jax_devices'] = [
                    {'type': d.device_kind, 'platform': d.platform, 'id': d.id}
                    for d in gpu_devices
                ]
            else:
                gpu_info['jax_gpu_available'] = False
                if gpu_info['available']:
                    recommendations.append("GPUs detected but not available to JAX - install jax[cuda] for GPU support")
        except:
            pass
        
        if warnings:
            gpu_info['warnings'] = warnings
        if recommendations:
            gpu_info['recommendations'] = recommendations
        
        return gpu_info
    
    def _detect_storage(self) -> Dict[str, Any]:
        """Detect storage information."""
        storage_info = {}
        warnings = []
        
        try:
            import psutil
            
            # Disk usage for current directory
            disk_usage = psutil.disk_usage('/')
            storage_info['total_gb'] = disk_usage.total / (1024**3)
            storage_info['free_gb'] = disk_usage.free / (1024**3)
            storage_info['used_percent'] = (disk_usage.used / disk_usage.total * 100)
            
            # Performance warnings
            if storage_info['free_gb'] < 10:
                warnings.append("Low disk space (< 10GB free)")
            
            if storage_info['used_percent'] > 90:
                warnings.append("Disk usage is very high (> 90%)")
            
            # Try to detect SSD vs HDD (Linux)
            if platform.system() == "Linux":
                try:
                    result = subprocess.run(['lsblk', '-d', '-o', 'name,rota'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        storage_info['drives'] = []
                        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                            parts = line.split()
                            if len(parts) >= 2:
                                is_ssd = parts[1] == '0'
                                storage_info['drives'].append({
                                    'name': parts[0],
                                    'type': 'SSD' if is_ssd else 'HDD'
                                })
                except:
                    pass
        
        except ImportError:
            storage_info['error'] = "psutil not available for storage info"
        except Exception as e:
            storage_info['error'] = f"Error getting storage info: {str(e)}"
        
        if warnings:
            storage_info['warnings'] = warnings
        
        return storage_info
    
    def _detect_network(self) -> Dict[str, Any]:
        """Detect network capabilities."""
        network_info = {}
        
        try:
            import psutil
            
            # Network interfaces
            interfaces = psutil.net_if_addrs()
            network_info['interfaces'] = list(interfaces.keys())
            
            # Network I/O stats
            net_io = psutil.net_io_counters()
            if net_io:
                network_info['bytes_sent'] = net_io.bytes_sent
                network_info['bytes_recv'] = net_io.bytes_recv
                network_info['packets_sent'] = net_io.packets_sent
                network_info['packets_recv'] = net_io.packets_recv
            
        except ImportError:
            network_info['error'] = "psutil not available for network info"
        except Exception as e:
            network_info['error'] = f"Error getting network info: {str(e)}"
        
        return network_info
    
    def _summarize_capabilities(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize hardware capabilities."""
        capabilities = {
            'quantum_simulation_ready': True,
            'gpu_acceleration': False,
            'parallel_processing': False,
            'memory_sufficient': False,
            'performance_tier': 'basic'
        }
        
        # Check GPU acceleration
        gpu_info = details.get('gpu', {})
        if gpu_info.get('jax_gpu_available') or gpu_info.get('available'):
            capabilities['gpu_acceleration'] = True
        
        # Check parallel processing
        cpu_info = details.get('cpu', {})
        if cpu_info.get('logical_cores', 0) >= 4:
            capabilities['parallel_processing'] = True
        
        # Check memory
        memory_info = details.get('memory', {})
        if memory_info.get('total_gb', 0) >= 8:
            capabilities['memory_sufficient'] = True
        
        # Determine performance tier
        score = 0
        if capabilities['gpu_acceleration']:
            score += 3
        if capabilities['parallel_processing']:
            score += 2
        if capabilities['memory_sufficient']:
            score += 2
        if cpu_info.get('logical_cores', 0) >= 8:
            score += 1
        if memory_info.get('total_gb', 0) >= 16:
            score += 1
        
        if score >= 7:
            capabilities['performance_tier'] = 'high'
        elif score >= 4:
            capabilities['performance_tier'] = 'medium'
        else:
            capabilities['performance_tier'] = 'basic'
        
        return capabilities