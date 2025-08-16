"""
Real-Time Analytics Engine for QEM-Bench

Advanced real-time monitoring, analysis, and intelligent insights
for quantum error mitigation systems with streaming data processing.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
from collections import deque, defaultdict
import threading
import queue
import logging
from concurrent.futures import ThreadPoolExecutor
import json
import asyncio
import websockets
from datetime import datetime, timedelta
import sqlite3

from ..monitoring import SystemMonitor, PerformanceMonitor


@dataclass
class AnalyticsMetric:
    """Structured metric for real-time analytics."""
    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "tags": self.tags
        }


@dataclass
class StreamingDataPoint:
    """Data point in streaming analytics pipeline."""
    experiment_id: str
    metric_name: str
    value: float
    timestamp: datetime
    circuit_id: Optional[str] = None
    backend_id: Optional[str] = None
    mitigation_method: Optional[str] = None
    noise_level: Optional[float] = None
    
    def to_metric(self) -> AnalyticsMetric:
        """Convert to analytics metric."""
        metadata = {}
        if self.circuit_id:
            metadata["circuit_id"] = self.circuit_id
        if self.backend_id:
            metadata["backend_id"] = self.backend_id
        if self.mitigation_method:
            metadata["mitigation_method"] = self.mitigation_method
        if self.noise_level is not None:
            metadata["noise_level"] = self.noise_level
            
        tags = [self.experiment_id]
        if self.mitigation_method:
            tags.append(self.mitigation_method)
            
        return AnalyticsMetric(
            name=self.metric_name,
            value=self.value,
            timestamp=self.timestamp,
            metadata=metadata,
            tags=tags
        )


class MetricsProcessor:
    """High-performance metrics processing engine."""
    
    def __init__(self, buffer_size: int = 10000, processing_interval: float = 1.0):
        self.buffer_size = buffer_size
        self.processing_interval = processing_interval
        
        # Circular buffers for different metric types
        self.fidelity_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)
        self.efficiency_buffer = deque(maxlen=buffer_size)
        self.error_buffer = deque(maxlen=buffer_size)
        
        # Statistical accumulators
        self.running_stats = defaultdict(lambda: {
            "count": 0, "sum": 0.0, "sum_sq": 0.0, "min": float('inf'), "max": float('-inf')
        })
        
        # Sliding window analytics
        self.sliding_windows = {
            "1m": deque(maxlen=60),    # 1 minute
            "5m": deque(maxlen=300),   # 5 minutes
            "1h": deque(maxlen=3600),  # 1 hour
        }
        
        self.logger = logging.getLogger(__name__)
    
    def add_metric(self, metric: AnalyticsMetric):
        """Add metric to processing pipeline."""
        
        # Route to appropriate buffer
        if "fidelity" in metric.name.lower():
            self.fidelity_buffer.append(metric)
        elif "time" in metric.name.lower():
            self.time_buffer.append(metric)
        elif "efficiency" in metric.name.lower():
            self.efficiency_buffer.append(metric)
        elif "error" in metric.name.lower():
            self.error_buffer.append(metric)
        
        # Update running statistics
        self._update_running_stats(metric)
        
        # Update sliding windows
        self._update_sliding_windows(metric)
    
    def _update_running_stats(self, metric: AnalyticsMetric):
        """Update running statistics for metric."""
        stats = self.running_stats[metric.name]
        stats["count"] += 1
        stats["sum"] += metric.value
        stats["sum_sq"] += metric.value ** 2
        stats["min"] = min(stats["min"], metric.value)
        stats["max"] = max(stats["max"], metric.value)
    
    def _update_sliding_windows(self, metric: AnalyticsMetric):
        """Update sliding window buffers."""
        for window_name, window_buffer in self.sliding_windows.items():
            window_buffer.append(metric)
    
    def get_running_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get running statistics for a metric."""
        stats = self.running_stats[metric_name]
        if stats["count"] == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        
        mean = stats["sum"] / stats["count"]
        variance = (stats["sum_sq"] / stats["count"]) - mean ** 2
        std = np.sqrt(max(0, variance))
        
        return {
            "mean": mean,
            "std": std,
            "min": stats["min"],
            "max": stats["max"],
            "count": stats["count"]
        }
    
    def get_window_statistics(self, window: str, metric_filter: str = None) -> Dict[str, Any]:
        """Get statistics for a specific time window."""
        if window not in self.sliding_windows:
            return {}
        
        window_data = list(self.sliding_windows[window])
        if metric_filter:
            window_data = [m for m in window_data if metric_filter in m.name]
        
        if not window_data:
            return {}
        
        values = [m.value for m in window_data]
        return {
            "count": len(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99)
        }


class TrendAnalyzer:
    """Analyzer for detecting trends in quantum metrics."""
    
    def __init__(self, min_data_points: int = 10):
        self.min_data_points = min_data_points
        self.trend_history = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    def analyze_trend(self, metric_name: str, values: List[float], 
                     timestamps: List[datetime]) -> Dict[str, Any]:
        """Analyze trend in metric values over time."""
        
        if len(values) < self.min_data_points:
            return {"trend": "insufficient_data", "confidence": 0.0}
        
        # Convert timestamps to numeric values
        base_time = timestamps[0]
        time_deltas = [(ts - base_time).total_seconds() for ts in timestamps]
        
        # Linear regression for trend
        x = np.array(time_deltas)
        y = np.array(values)
        
        # Calculate trend slope
        slope, intercept, r_value, p_value = self._linear_regression(x, y)
        
        # Classify trend
        trend_classification = self._classify_trend(slope, r_value, p_value)
        
        # Detect change points
        change_points = self._detect_change_points(values)
        
        # Seasonal analysis
        seasonality = self._detect_seasonality(values, timestamps)
        
        return {
            "trend": trend_classification["trend"],
            "slope": float(slope),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "confidence": trend_classification["confidence"],
            "change_points": change_points,
            "seasonality": seasonality,
            "forecast_next": self._forecast_next_value(x, y, slope, intercept)
        }
    
    def _linear_regression(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute linear regression statistics."""
        n = len(x)
        
        # Calculate regression coefficients
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0, y_mean, 0.0, 1.0
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate correlation coefficient
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        
        if ss_tot == 0:
            r_value = 0.0
        else:
            r_value = np.sqrt(1 - ss_res / ss_tot)
            if slope < 0:
                r_value = -r_value
        
        # Simple p-value approximation
        if n > 2:
            t_stat = r_value * np.sqrt((n - 2) / (1 - r_value ** 2))
            p_value = 2 * (1 - abs(t_stat) / np.sqrt(n - 2))  # Simplified
        else:
            p_value = 1.0
        
        return slope, intercept, r_value, min(1.0, max(0.0, p_value))
    
    def _classify_trend(self, slope: float, r_value: float, p_value: float) -> Dict[str, Any]:
        """Classify trend based on statistical parameters."""
        
        confidence = abs(r_value) * (1 - p_value)  # Combined confidence metric
        
        if p_value > 0.05:
            return {"trend": "no_trend", "confidence": 0.0}
        
        if abs(slope) < 1e-6:
            return {"trend": "stable", "confidence": confidence}
        
        if slope > 0:
            if confidence > 0.8:
                return {"trend": "strong_positive", "confidence": confidence}
            elif confidence > 0.5:
                return {"trend": "moderate_positive", "confidence": confidence}
            else:
                return {"trend": "weak_positive", "confidence": confidence}
        else:
            if confidence > 0.8:
                return {"trend": "strong_negative", "confidence": confidence}
            elif confidence > 0.5:
                return {"trend": "moderate_negative", "confidence": confidence}
            else:
                return {"trend": "weak_negative", "confidence": confidence}
    
    def _detect_change_points(self, values: List[float]) -> List[int]:
        """Detect change points in time series."""
        if len(values) < 6:
            return []
        
        change_points = []
        window_size = max(3, len(values) // 10)
        
        for i in range(window_size, len(values) - window_size):
            before = values[i-window_size:i]
            after = values[i:i+window_size]
            
            # Statistical test for change point
            mean_diff = abs(np.mean(after) - np.mean(before))
            pooled_std = np.sqrt((np.var(before) + np.var(after)) / 2)
            
            if pooled_std > 0 and mean_diff / pooled_std > 1.5:  # Threshold for change
                change_points.append(i)
        
        return change_points
    
    def _detect_seasonality(self, values: List[float], timestamps: List[datetime]) -> Dict[str, Any]:
        """Detect seasonal patterns in the data."""
        if len(values) < 20:
            return {"seasonal": False, "period": None}
        
        # Simple autocorrelation-based seasonality detection
        max_lag = min(len(values) // 4, 50)
        autocorrelations = []
        
        for lag in range(1, max_lag):
            if lag >= len(values):
                break
            corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
            if not np.isnan(corr):
                autocorrelations.append(corr)
        
        if autocorrelations:
            max_corr = max(autocorrelations)
            best_lag = autocorrelations.index(max_corr) + 1
            
            if max_corr > 0.3:  # Threshold for seasonality
                return {"seasonal": True, "period": best_lag, "strength": max_corr}
        
        return {"seasonal": False, "period": None, "strength": 0.0}
    
    def _forecast_next_value(self, x: np.ndarray, y: np.ndarray, 
                           slope: float, intercept: float) -> float:
        """Forecast next value in the sequence."""
        if len(x) == 0:
            return 0.0
        
        next_x = x[-1] + (x[-1] - x[0]) / len(x)  # Assume same interval
        return float(slope * next_x + intercept)


class AnomalyDetector:
    """Advanced anomaly detection for quantum metrics."""
    
    def __init__(self, sensitivity: float = 2.0, window_size: int = 100):
        self.sensitivity = sensitivity
        self.window_size = window_size
        self.baseline_models = {}
        self.anomaly_history = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    def detect_anomaly(self, metric: AnalyticsMetric, 
                      historical_values: List[float]) -> Dict[str, Any]:
        """Detect anomalies in metric values."""
        
        if len(historical_values) < 10:
            return {"anomaly": False, "reason": "insufficient_data"}
        
        # Statistical anomaly detection
        statistical_anomaly = self._statistical_anomaly_detection(
            metric.value, historical_values
        )
        
        # Pattern-based anomaly detection
        pattern_anomaly = self._pattern_based_detection(
            metric.value, historical_values
        )
        
        # Contextual anomaly detection
        contextual_anomaly = self._contextual_anomaly_detection(metric)
        
        # Combine detection results
        anomaly_score = max(
            statistical_anomaly["score"],
            pattern_anomaly["score"],
            contextual_anomaly["score"]
        )
        
        is_anomaly = anomaly_score > self.sensitivity
        
        if is_anomaly:
            self.anomaly_history[metric.name].append({
                "timestamp": metric.timestamp,
                "value": metric.value,
                "score": anomaly_score,
                "type": self._classify_anomaly_type(metric, historical_values)
            })
        
        return {
            "anomaly": is_anomaly,
            "score": float(anomaly_score),
            "statistical": statistical_anomaly,
            "pattern": pattern_anomaly,
            "contextual": contextual_anomaly,
            "severity": self._classify_severity(anomaly_score),
            "recommendation": self._generate_recommendation(metric, anomaly_score)
        }
    
    def _statistical_anomaly_detection(self, value: float, 
                                     historical: List[float]) -> Dict[str, Any]:
        """Statistical anomaly detection using z-score and IQR."""
        
        hist_array = np.array(historical)
        mean = np.mean(hist_array)
        std = np.std(hist_array)
        
        # Z-score anomaly
        z_score = abs(value - mean) / max(std, 1e-10)
        z_anomaly = z_score > 3.0
        
        # IQR anomaly
        q1, q3 = np.percentile(hist_array, [25, 75])
        iqr = q3 - q1
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        iqr_anomaly = value < iqr_lower or value > iqr_upper
        
        # Modified Z-score (more robust)
        median = np.median(hist_array)
        mad = np.median(np.abs(hist_array - median))
        modified_z = abs(value - median) / max(mad * 0.6745, 1e-10)
        modified_anomaly = modified_z > 3.5
        
        score = max(z_score / 3.0, modified_z / 3.5) if any([z_anomaly, iqr_anomaly, modified_anomaly]) else 0
        
        return {
            "anomaly": any([z_anomaly, iqr_anomaly, modified_anomaly]),
            "score": float(min(score, 10.0)),
            "z_score": float(z_score),
            "modified_z": float(modified_z),
            "iqr_violation": iqr_anomaly
        }
    
    def _pattern_based_detection(self, value: float, 
                               historical: List[float]) -> Dict[str, Any]:
        """Pattern-based anomaly detection."""
        
        if len(historical) < 20:
            return {"anomaly": False, "score": 0.0}
        
        # Look for sudden changes in pattern
        recent = historical[-10:]
        older = historical[-20:-10]
        
        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        recent_std = np.std(recent)
        older_std = np.std(older)
        
        # Detect level shift
        level_shift = abs(recent_mean - older_mean) / max(older_std, 1e-10)
        
        # Detect variance change
        variance_change = abs(recent_std - older_std) / max(older_std, 1e-10)
        
        # Pattern break detection
        pattern_score = max(level_shift / 2.0, variance_change / 2.0)
        
        return {
            "anomaly": pattern_score > 1.0,
            "score": float(min(pattern_score, 5.0)),
            "level_shift": float(level_shift),
            "variance_change": float(variance_change)
        }
    
    def _contextual_anomaly_detection(self, metric: AnalyticsMetric) -> Dict[str, Any]:
        """Contextual anomaly detection based on metadata."""
        
        score = 0.0
        anomalies = []
        
        # Check for unusual context combinations
        if "mitigation_method" in metric.metadata:
            method = metric.metadata["mitigation_method"]
            
            # Some methods shouldn't produce extremely high fidelity improvements
            if "fidelity" in metric.name.lower() and metric.value > 0.95:
                if method in ["zne", "pec"]:  # These have theoretical limits
                    score = max(score, 2.0)
                    anomalies.append("unusually_high_fidelity")
        
        # Check noise level consistency
        if "noise_level" in metric.metadata:
            noise = metric.metadata["noise_level"]
            if "fidelity" in metric.name.lower() and noise > 0.1 and metric.value > 0.9:
                score = max(score, 1.5)
                anomalies.append("high_fidelity_high_noise")
        
        return {
            "anomaly": score > 1.0,
            "score": float(score),
            "detected_anomalies": anomalies
        }
    
    def _classify_anomaly_type(self, metric: AnalyticsMetric, 
                             historical: List[float]) -> str:
        """Classify the type of anomaly."""
        
        recent_values = historical[-5:] if len(historical) >= 5 else historical
        recent_mean = np.mean(recent_values)
        
        if metric.value > recent_mean * 1.5:
            return "spike"
        elif metric.value < recent_mean * 0.5:
            return "drop"
        elif abs(metric.value - recent_mean) > 3 * np.std(historical):
            return "outlier"
        else:
            return "drift"
    
    def _classify_severity(self, score: float) -> str:
        """Classify anomaly severity."""
        if score > 5.0:
            return "critical"
        elif score > 3.0:
            return "high"
        elif score > 1.5:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendation(self, metric: AnalyticsMetric, score: float) -> str:
        """Generate recommendation for handling anomaly."""
        
        if score > 5.0:
            return "Investigate immediately - potential system failure"
        elif score > 3.0:
            return "Review recent changes and validate measurements"
        elif score > 1.5:
            return "Monitor closely and check for patterns"
        else:
            return "Document and continue monitoring"


class RealTimeAnalyzer:
    """Real-time analytics engine for quantum error mitigation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core components
        self.metrics_processor = MetricsProcessor()
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        
        # Real-time data streams
        self.data_queue = queue.Queue(maxsize=10000)
        self.processing_thread = None
        self.running = False
        
        # Database for persistence
        self.db_path = self.config.get("database_path", "qem_analytics.db")
        self._init_database()
        
        # WebSocket server for real-time updates
        self.websocket_port = self.config.get("websocket_port", 8765)
        self.websocket_clients = set()
        
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize SQLite database for analytics data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                tags TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                score REAL NOT NULL,
                anomaly_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE
            )
        """)
        
        conn.commit()
        conn.close()
    
    def start(self):
        """Start real-time analytics processing."""
        if self.running:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        # Start WebSocket server
        asyncio.create_task(self._start_websocket_server())
        
        self.logger.info("Real-time analytics engine started")
    
    def stop(self):
        """Stop real-time analytics processing."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        
        self.logger.info("Real-time analytics engine stopped")
    
    def add_data_point(self, data_point: StreamingDataPoint):
        """Add data point to real-time processing queue."""
        try:
            self.data_queue.put(data_point, timeout=1.0)
        except queue.Full:
            self.logger.warning("Analytics queue full, dropping data point")
    
    def _processing_loop(self):
        """Main processing loop for real-time analytics."""
        
        while self.running:
            try:
                # Get data point from queue
                data_point = self.data_queue.get(timeout=1.0)
                metric = data_point.to_metric()
                
                # Process metric
                self.metrics_processor.add_metric(metric)
                
                # Store in database
                self._store_metric(metric)
                
                # Detect anomalies
                historical_values = self._get_historical_values(metric.name, limit=100)
                anomaly_result = self.anomaly_detector.detect_anomaly(metric, historical_values)
                
                if anomaly_result["anomaly"]:
                    self._store_anomaly(metric, anomaly_result)
                    self._broadcast_anomaly(metric, anomaly_result)
                
                # Analyze trends
                if len(historical_values) >= 10:
                    historical_timestamps = self._get_historical_timestamps(metric.name, limit=100)
                    trend_result = self.trend_analyzer.analyze_trend(
                        metric.name, historical_values, historical_timestamps
                    )
                    self._broadcast_trend_update(metric.name, trend_result)
                
                # Broadcast real-time update
                self._broadcast_metric_update(metric)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in analytics processing: {e}")
    
    def _store_metric(self, metric: AnalyticsMetric):
        """Store metric in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO metrics (name, value, timestamp, metadata, tags)
            VALUES (?, ?, ?, ?, ?)
        """, (
            metric.name,
            metric.value,
            metric.timestamp.isoformat(),
            json.dumps(metric.metadata),
            json.dumps(metric.tags)
        ))
        
        conn.commit()
        conn.close()
    
    def _store_anomaly(self, metric: AnalyticsMetric, anomaly_result: Dict[str, Any]):
        """Store anomaly in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO anomalies (metric_name, value, score, anomaly_type, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            metric.name,
            metric.value,
            anomaly_result["score"],
            anomaly_result.get("severity", "unknown"),
            metric.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _get_historical_values(self, metric_name: str, limit: int = 100) -> List[float]:
        """Get historical values for a metric."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT value FROM metrics 
            WHERE name = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (metric_name, limit))
        
        values = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return values[::-1]  # Return in chronological order
    
    def _get_historical_timestamps(self, metric_name: str, limit: int = 100) -> List[datetime]:
        """Get historical timestamps for a metric."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp FROM metrics 
            WHERE name = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (metric_name, limit))
        
        timestamps = [datetime.fromisoformat(row[0]) for row in cursor.fetchall()]
        conn.close()
        
        return timestamps[::-1]  # Return in chronological order
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates."""
        async def handle_client(websocket, path):
            self.websocket_clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.websocket_clients.remove(websocket)
        
        await websockets.serve(handle_client, "localhost", self.websocket_port)
        self.logger.info(f"WebSocket server started on port {self.websocket_port}")
    
    async def _broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket clients."""
        if self.websocket_clients:
            message_str = json.dumps(message)
            disconnected_clients = set()
            
            for client in self.websocket_clients:
                try:
                    await client.send(message_str)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients
    
    def _broadcast_metric_update(self, metric: AnalyticsMetric):
        """Broadcast metric update to WebSocket clients."""
        message = {
            "type": "metric_update",
            "data": metric.to_dict()
        }
        
        # Use asyncio to run the coroutine
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._broadcast_message(message))
        loop.close()
    
    def _broadcast_anomaly(self, metric: AnalyticsMetric, anomaly_result: Dict[str, Any]):
        """Broadcast anomaly detection to WebSocket clients."""
        message = {
            "type": "anomaly_detected",
            "data": {
                "metric": metric.to_dict(),
                "anomaly": anomaly_result
            }
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._broadcast_message(message))
        loop.close()
    
    def _broadcast_trend_update(self, metric_name: str, trend_result: Dict[str, Any]):
        """Broadcast trend update to WebSocket clients."""
        message = {
            "type": "trend_update",
            "data": {
                "metric_name": metric_name,
                "trend": trend_result
            }
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._broadcast_message(message))
        loop.close()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        
        # Get recent metrics
        recent_metrics = {}
        for metric_name in ["fidelity_improvement", "execution_time", "resource_efficiency"]:
            stats = self.metrics_processor.get_running_statistics(metric_name)
            window_stats = self.metrics_processor.get_window_statistics("1h", metric_name)
            recent_metrics[metric_name] = {
                "running": stats,
                "window": window_stats
            }
        
        # Get recent anomalies
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT metric_name, value, score, anomaly_type, timestamp 
            FROM anomalies 
            WHERE timestamp > datetime('now', '-24 hours')
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        
        recent_anomalies = [
            {
                "metric_name": row[0],
                "value": row[1],
                "score": row[2],
                "type": row[3],
                "timestamp": row[4]
            }
            for row in cursor.fetchall()
        ]
        conn.close()
        
        return {
            "metrics": recent_metrics,
            "anomalies": recent_anomalies,
            "system_status": "healthy" if len(recent_anomalies) == 0 else "attention_needed",
            "last_updated": datetime.now().isoformat()
        }


def create_analytics_engine(config: Dict[str, Any] = None) -> RealTimeAnalyzer:
    """Create and configure real-time analytics engine."""
    
    default_config = {
        "database_path": "qem_analytics.db",
        "websocket_port": 8765,
        "buffer_size": 10000,
        "processing_interval": 1.0,
        "anomaly_sensitivity": 2.0
    }
    
    if config:
        default_config.update(config)
    
    return RealTimeAnalyzer(default_config)


# Example usage and demo
if __name__ == "__main__":
    
    # Create analytics engine
    analyzer = create_analytics_engine()
    
    print("🔬 Real-Time Analytics Engine for QEM-Bench")
    print("=" * 60)
    
    # Start analytics engine
    print("\n🚀 Starting real-time analytics engine...")
    analyzer.start()
    
    # Simulate data stream
    print("📊 Simulating real-time data stream...")
    
    for i in range(100):
        # Simulate various metrics
        fidelity_dp = StreamingDataPoint(
            experiment_id="exp_001",
            metric_name="fidelity_improvement",
            value=0.85 + np.random.normal(0.1, 0.05),
            timestamp=datetime.now(),
            mitigation_method="zne",
            noise_level=0.05
        )
        
        time_dp = StreamingDataPoint(
            experiment_id="exp_001",
            metric_name="execution_time",
            value=np.random.exponential(2.0),
            timestamp=datetime.now(),
            mitigation_method="zne"
        )
        
        # Add some anomalies
        if i % 20 == 0:
            anomaly_dp = StreamingDataPoint(
                experiment_id="exp_001",
                metric_name="fidelity_improvement",
                value=0.95 + np.random.normal(0, 0.02),  # Unusually high
                timestamp=datetime.now(),
                mitigation_method="zne",
                noise_level=0.1  # High noise but high fidelity = anomaly
            )
            analyzer.add_data_point(anomaly_dp)
        
        analyzer.add_data_point(fidelity_dp)
        analyzer.add_data_point(time_dp)
        
        time.sleep(0.1)  # Simulate real-time stream
    
    # Get dashboard data
    time.sleep(2)  # Let processing catch up
    dashboard = analyzer.get_dashboard_data()
    
    print("\n📈 Dashboard Summary:")
    print(f"├── System Status: {dashboard['system_status']}")
    print(f"├── Recent Anomalies: {len(dashboard['anomalies'])}")
    print(f"└── Metrics Tracked: {len(dashboard['metrics'])}")
    
    # Display metrics
    for metric_name, data in dashboard['metrics'].items():
        if data['running']['count'] > 0:
            print(f"\n{metric_name.title()}:")
            print(f"├── Mean: {data['running']['mean']:.3f}")
            print(f"├── Std: {data['running']['std']:.3f}")
            print(f"└── Count: {data['running']['count']}")
    
    # Display anomalies
    if dashboard['anomalies']:
        print(f"\n🚨 Recent Anomalies:")
        for anomaly in dashboard['anomalies'][:3]:
            print(f"├── {anomaly['metric_name']}: {anomaly['value']:.3f} (score: {anomaly['score']:.2f})")
    
    # Stop analytics engine
    analyzer.stop()
    
    print("\n🎯 Real-Time Analytics Engine Demo Completed Successfully!")