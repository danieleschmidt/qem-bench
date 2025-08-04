"""System resource monitoring for QEM-Bench experiments."""

import time
import threading
import psutil
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json


logger = logging.getLogger(__name__)


@dataclass
class SystemSnapshot:
    """Snapshot of system resource usage at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    gpu_info: Optional[Dict[str, Any]] = None
    disk_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    load_average: Optional[List[float]] = None
    process_count: int = 0


@dataclass  
class SystemMonitorConfig:
    """Configuration for system monitoring."""
    enabled: bool = True
    sampling_interval: float = 1.0  # seconds
    max_history_size: int = 1000
    monitor_gpu: bool = True
    monitor_network: bool = True
    monitor_disk: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_percent': 90.0,
        'memory_percent': 85.0,
        'disk_usage_percent': 90.0
    })


class SystemMonitor:
    """
    System resource monitor for tracking CPU, memory, GPU, and other resources.
    
    This monitor runs in a background thread and provides real-time system
    resource information for quantum error mitigation experiments.
    
    Example:
        >>> monitor = SystemMonitor()
        >>> monitor.start()
        >>> # Run some experiments
        >>> stats = monitor.get_current_stats()
        >>> print(f"CPU: {stats.cpu_percent:.1f}%, Memory: {stats.memory_percent:.1f}%")
        >>> monitor.stop()
    """
    
    def __init__(self, config: Optional[SystemMonitorConfig] = None):
        self.config = config or SystemMonitorConfig()
        self._history: deque = deque(maxlen=self.config.max_history_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[SystemSnapshot], None]] = []
        
        # Initialize GPU monitoring if available
        self._gpu_available = False
        if self.config.monitor_gpu:
            self._init_gpu_monitoring()
    
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring capabilities."""
        try:
            import GPUtil
            self._gpu_available = True
            logger.info("GPU monitoring enabled")
        except ImportError:
            logger.info("GPUtil not available, GPU monitoring disabled")
            self._gpu_available = False
    
    def start(self):
        """Start the system monitoring thread."""
        if not self.config.enabled:
            logger.info("System monitoring disabled")
            return
            
        if self._running:
            logger.warning("System monitor already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        logger.info(f"System monitor started with {self.config.sampling_interval}s interval")
    
    def stop(self):
        """Stop the system monitoring thread."""
        if not self._running:
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("System monitor stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in a background thread."""
        while self._running:
            try:
                snapshot = self._take_snapshot()
                
                with self._lock:
                    self._history.append(snapshot)
                
                # Execute callbacks
                for callback in self._callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        logger.error(f"Monitor callback failed: {e}")
                
                time.sleep(self.config.sampling_interval)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(self.config.sampling_interval)
    
    def _take_snapshot(self) -> SystemSnapshot:
        """Take a snapshot of current system resources."""
        # Basic system metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # Network stats
        network_bytes_sent = 0
        network_bytes_recv = 0
        if self.config.monitor_network:
            try:
                net_io = psutil.net_io_counters()
                network_bytes_sent = net_io.bytes_sent
                network_bytes_recv = net_io.bytes_recv
            except Exception:
                pass
        
        # Disk usage
        disk_usage_percent = 0.0
        if self.config.monitor_disk:
            try:
                disk_usage = psutil.disk_usage('/')
                disk_usage_percent = (disk_usage.used / disk_usage.total) * 100
            except Exception:
                pass
        
        # Load average (Unix-like systems)
        load_average = None
        try:
            load_average = list(psutil.getloadavg())
        except (AttributeError, OSError):
            pass
        
        # Process count
        process_count = len(psutil.pids())
        
        # GPU information
        gpu_info = None
        if self._gpu_available:
            gpu_info = self._get_gpu_info()
        
        return SystemSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            gpu_info=gpu_info,
            disk_usage_percent=disk_usage_percent,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            load_average=load_average,
            process_count=process_count
        )
    
    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU information if available."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            if not gpus:
                return None
            
            gpu_info = {
                'count': len(gpus),
                'gpus': []
            }
            
            for gpu in gpus:
                gpu_data = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                    'gpu_percent': gpu.load * 100,
                    'temperature': gpu.temperature
                }
                gpu_info['gpus'].append(gpu_data)
            
            return gpu_info
            
        except Exception as e:
            logger.debug(f"Failed to get GPU info: {e}")
            return None
    
    def get_current_stats(self) -> Optional[SystemSnapshot]:
        """Get the most recent system statistics."""
        with self._lock:
            if not self._history:
                return None
            return self._history[-1]
    
    def get_history(self, duration_seconds: Optional[float] = None) -> List[SystemSnapshot]:
        """
        Get historical system statistics.
        
        Args:
            duration_seconds: If specified, only return snapshots from this many
                            seconds ago. If None, return all history.
        
        Returns:
            List of SystemSnapshot objects
        """
        with self._lock:
            if duration_seconds is None:
                return list(self._history)
            
            cutoff_time = time.time() - duration_seconds
            return [
                snapshot for snapshot in self._history 
                if snapshot.timestamp >= cutoff_time
            ]
    
    def get_average_stats(self, duration_seconds: Optional[float] = None) -> Optional[Dict[str, float]]:
        """
        Get average system statistics over a time period.
        
        Args:
            duration_seconds: Time period to average over. If None, use all history.
        
        Returns:
            Dictionary with average statistics
        """
        history = self.get_history(duration_seconds)
        if not history:
            return None
        
        avg_stats = {
            'cpu_percent': sum(s.cpu_percent for s in history) / len(history),
            'memory_percent': sum(s.memory_percent for s in history) / len(history),
            'memory_used_gb': sum(s.memory_used_gb for s in history) / len(history),
            'disk_usage_percent': sum(s.disk_usage_percent for s in history) / len(history),
            'process_count': sum(s.process_count for s in history) / len(history)
        }
        
        # Add GPU averages if available
        gpu_histories = [s.gpu_info for s in history if s.gpu_info is not None]
        if gpu_histories:
            gpu_count = gpu_histories[0]['count']
            avg_stats['gpu_count'] = gpu_count
            
            for gpu_id in range(gpu_count):
                gpu_data = []
                for gpu_info in gpu_histories:
                    if gpu_id < len(gpu_info['gpus']):
                        gpu_data.append(gpu_info['gpus'][gpu_id])
                
                if gpu_data:
                    avg_stats[f'gpu_{gpu_id}_memory_percent'] = sum(
                        g['memory_percent'] for g in gpu_data
                    ) / len(gpu_data)
                    avg_stats[f'gpu_{gpu_id}_utilization'] = sum(
                        g['gpu_percent'] for g in gpu_data
                    ) / len(gpu_data)
        
        return avg_stats
    
    def check_alert_thresholds(self) -> List[Dict[str, Any]]:
        """
        Check if any system metrics exceed alert thresholds.
        
        Returns:
            List of alert dictionaries
        """
        current_stats = self.get_current_stats()
        if not current_stats:
            return []
        
        alerts = []
        thresholds = self.config.alert_thresholds
        
        # Check CPU threshold
        if current_stats.cpu_percent > thresholds.get('cpu_percent', 100):
            alerts.append({
                'metric': 'cpu_percent',
                'value': current_stats.cpu_percent,
                'threshold': thresholds['cpu_percent'],
                'message': f"CPU usage ({current_stats.cpu_percent:.1f}%) exceeds threshold ({thresholds['cpu_percent']}%)"
            })
        
        # Check memory threshold
        if current_stats.memory_percent > thresholds.get('memory_percent', 100):
            alerts.append({
                'metric': 'memory_percent', 
                'value': current_stats.memory_percent,
                'threshold': thresholds['memory_percent'],
                'message': f"Memory usage ({current_stats.memory_percent:.1f}%) exceeds threshold ({thresholds['memory_percent']}%)"
            })
        
        # Check disk threshold
        if current_stats.disk_usage_percent > thresholds.get('disk_usage_percent', 100):
            alerts.append({
                'metric': 'disk_usage_percent',
                'value': current_stats.disk_usage_percent, 
                'threshold': thresholds['disk_usage_percent'],
                'message': f"Disk usage ({current_stats.disk_usage_percent:.1f}%) exceeds threshold ({thresholds['disk_usage_percent']}%)"
            })
        
        return alerts
    
    def add_callback(self, callback: Callable[[SystemSnapshot], None]):
        """Add a callback function to be called on each monitoring snapshot."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[SystemSnapshot], None]):
        """Remove a callback function."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def export_stats(self, filepath: str, duration_seconds: Optional[float] = None):
        """
        Export system statistics to a file.
        
        Args:
            filepath: Path to export file (JSON format)
            duration_seconds: Time period to export. If None, export all history.
        """
        history = self.get_history(duration_seconds)
        
        # Convert to serializable format
        export_data = {
            'config': {
                'sampling_interval': self.config.sampling_interval,
                'alert_thresholds': self.config.alert_thresholds
            },
            'snapshots': []
        }
        
        for snapshot in history:
            snapshot_data = {
                'timestamp': snapshot.timestamp,
                'cpu_percent': snapshot.cpu_percent,
                'memory_percent': snapshot.memory_percent,
                'memory_used_gb': snapshot.memory_used_gb,
                'memory_available_gb': snapshot.memory_available_gb,
                'disk_usage_percent': snapshot.disk_usage_percent,
                'network_bytes_sent': snapshot.network_bytes_sent,
                'network_bytes_recv': snapshot.network_bytes_recv,
                'load_average': snapshot.load_average,
                'process_count': snapshot.process_count,
                'gpu_info': snapshot.gpu_info
            }
            export_data['snapshots'].append(snapshot_data)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(history)} system snapshots to {filepath}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()