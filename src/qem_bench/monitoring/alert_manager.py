"""Alert management system for QEM-Bench monitoring."""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    THRESHOLD = "threshold"
    ANOMALY = "anomaly"
    ERROR = "error"
    RESOURCE = "resource"
    PERFORMANCE = "performance"


@dataclass
class Alert:
    """An alert instance."""
    id: str
    timestamp: float
    severity: AlertSeverity
    alert_type: AlertType
    title: str
    message: str
    source: str  # Component that generated the alert
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_timestamp: Optional[float] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    
    @property
    def age_seconds(self) -> float:
        """Age of the alert in seconds."""
        return time.time() - self.timestamp
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Duration of the alert if resolved."""
        if self.resolved and self.resolved_timestamp:
            return self.resolved_timestamp - self.timestamp
        return None


@dataclass
class AlertRule:
    """Rule for triggering alerts."""
    name: str
    metric_name: str
    condition: str  # e.g., "greater_than", "less_than", "equals"
    threshold: Union[float, int, str]
    severity: AlertSeverity
    alert_type: AlertType = AlertType.THRESHOLD
    cooldown_seconds: float = 300  # Minimum time between alerts
    consecutive_violations: int = 1  # Number of consecutive violations needed
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertManagerConfig:
    """Configuration for alert management."""
    enabled: bool = True
    max_active_alerts: int = 1000
    max_alert_history: int = 10000
    default_cooldown: float = 300  # seconds
    auto_resolve_timeout: Optional[float] = None  # Auto-resolve alerts after this time
    notification_handlers: List[str] = field(default_factory=list)


class AlertManager:
    """
    Alert manager for QEM-Bench monitoring system.
    
    Manages alert rules, triggers alerts based on metrics, handles notifications,
    and provides alert resolution capabilities.
    
    Example:
        >>> alert_manager = AlertManager()
        >>> 
        >>> # Add alert rules
        >>> alert_manager.add_rule(AlertRule(
        ...     name="high_cpu_usage",
        ...     metric_name="cpu_percent",
        ...     condition="greater_than",
        ...     threshold=90.0,
        ...     severity=AlertSeverity.HIGH
        ... ))
        >>> 
        >>> # Check metrics and trigger alerts
        >>> alert_manager.check_metric("cpu_percent", 95.0, source="system_monitor")
        >>> 
        >>> # Get active alerts
        >>> active_alerts = alert_manager.get_active_alerts()
    """
    
    def __init__(self, config: Optional[AlertManagerConfig] = None):
        self.config = config or AlertManagerConfig()
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=self.config.max_alert_history)
        self._metric_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)  # Keep last 100 values for trend analysis
        )
        self._last_alert_time: Dict[str, float] = {}
        self._violation_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._notification_handlers: List[Callable[[Alert], None]] = []
        
        # Auto-resolve timer
        self._auto_resolve_timer: Optional[threading.Timer] = None
        if self.config.auto_resolve_timeout:
            self._start_auto_resolve_timer()
    
    def _start_auto_resolve_timer(self):
        """Start timer for auto-resolving old alerts."""
        def auto_resolve():
            try:
                self._auto_resolve_alerts()
            except Exception as e:
                logger.error(f"Auto-resolve error: {e}")
            finally:
                # Schedule next check
                if self.config.auto_resolve_timeout:
                    self._auto_resolve_timer = threading.Timer(
                        self.config.auto_resolve_timeout, auto_resolve
                    )
                    self._auto_resolve_timer.daemon = True
                    self._auto_resolve_timer.start()
        
        self._auto_resolve_timer = threading.Timer(
            self.config.auto_resolve_timeout, auto_resolve
        )
        self._auto_resolve_timer.daemon = True
        self._auto_resolve_timer.start()
    
    def _auto_resolve_alerts(self):
        """Auto-resolve alerts that have exceeded timeout."""
        if not self.config.auto_resolve_timeout:
            return
        
        current_time = time.time()
        alerts_to_resolve = []
        
        with self._lock:
            for alert_id, alert in self._active_alerts.items():
                if not alert.resolved and (current_time - alert.timestamp) > self.config.auto_resolve_timeout:
                    alerts_to_resolve.append(alert_id)
        
        for alert_id in alerts_to_resolve:
            self.resolve_alert(alert_id, resolved_by="auto_resolve", 
                             resolution_notes="Auto-resolved after timeout")
    
    def add_rule(self, rule: AlertRule):
        """
        Add an alert rule.
        
        Args:
            rule: AlertRule to add
        """
        with self._lock:
            self._rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """
        Remove an alert rule.
        
        Args:
            rule_name: Name of the rule to remove
        """
        with self._lock:
            if rule_name in self._rules:
                del self._rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")
    
    def get_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        with self._lock:
            return list(self._rules.values())
    
    def enable_rule(self, rule_name: str):
        """Enable an alert rule."""
        with self._lock:
            if rule_name in self._rules:
                self._rules[rule_name].enabled = True
    
    def disable_rule(self, rule_name: str):
        """Disable an alert rule."""
        with self._lock:
            if rule_name in self._rules:
                self._rules[rule_name].enabled = False
    
    def check_metric(self, metric_name: str, value: Union[float, int, str], 
                    source: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Check a metric value against alert rules.
        
        Args:
            metric_name: Name of the metric
            value: Value to check
            source: Source component
            metadata: Additional metadata
        """
        if not self.config.enabled:
            return
        
        # Store metric value for trend analysis
        timestamp = time.time()
        with self._lock:
            self._metric_history[metric_name].append((timestamp, value))
        
        # Check against all rules for this metric
        rules_to_check = []
        with self._lock:
            for rule in self._rules.values():
                if rule.enabled and rule.metric_name == metric_name:
                    rules_to_check.append(rule)
        
        for rule in rules_to_check:
            if self._evaluate_rule(rule, value):
                self._handle_rule_violation(rule, value, source, metadata or {})
            else:
                self._handle_rule_compliance(rule)
    
    def _evaluate_rule(self, rule: AlertRule, value: Union[float, int, str]) -> bool:
        """Evaluate if a rule is violated."""
        try:
            if rule.condition == "greater_than":
                return float(value) > float(rule.threshold)
            elif rule.condition == "less_than":
                return float(value) < float(rule.threshold)
            elif rule.condition == "equals":
                return value == rule.threshold
            elif rule.condition == "not_equals":
                return value != rule.threshold
            elif rule.condition == "greater_equal":
                return float(value) >= float(rule.threshold)
            elif rule.condition == "less_equal":
                return float(value) <= float(rule.threshold)
            elif rule.condition == "contains":
                return str(rule.threshold) in str(value)
            else:
                logger.warning(f"Unknown condition: {rule.condition}")
                return False
        except (ValueError, TypeError) as e:
            logger.error(f"Error evaluating rule {rule.name}: {e}")
            return False
    
    def _handle_rule_violation(self, rule: AlertRule, value: Union[float, int, str],
                              source: str, metadata: Dict[str, Any]):
        """Handle a rule violation."""
        with self._lock:
            self._violation_counts[rule.name] += 1
            
            # Check if we've met the consecutive violations requirement
            if self._violation_counts[rule.name] < rule.consecutive_violations:
                return
            
            # Check cooldown period
            last_alert_time = self._last_alert_time.get(rule.name, 0)
            if time.time() - last_alert_time < rule.cooldown_seconds:
                return
            
            # Create alert
            alert_id = f"{rule.name}_{int(time.time())}_{threading.get_ident()}"
            alert = Alert(
                id=alert_id,
                timestamp=time.time(),
                severity=rule.severity,
                alert_type=rule.alert_type,
                title=f"Alert: {rule.name}",
                message=f"Metric '{rule.metric_name}' value {value} {rule.condition} {rule.threshold}",
                source=source,
                metadata={**rule.metadata, **metadata, "metric_value": value}
            )
            
            # Store alert
            if len(self._active_alerts) < self.config.max_active_alerts:
                self._active_alerts[alert_id] = alert
            self._alert_history.append(alert)
            self._last_alert_time[rule.name] = alert.timestamp
        
        # Send notifications
        self._send_notifications(alert)
        logger.warning(f"Alert triggered: {alert.title} - {alert.message}")
    
    def _handle_rule_compliance(self, rule: AlertRule):
        """Handle when a rule is back in compliance."""
        with self._lock:
            self._violation_counts[rule.name] = 0
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        for handler in self._notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
    
    def trigger_alert(self, severity: AlertSeverity, alert_type: AlertType,
                     title: str, message: str, source: str,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Manually trigger an alert.
        
        Args:
            severity: Alert severity
            alert_type: Type of alert
            title: Alert title
            message: Alert message
            source: Source component
            metadata: Additional metadata
        
        Returns:
            Alert ID
        """
        alert_id = f"manual_{int(time.time())}_{threading.get_ident()}"
        alert = Alert(
            id=alert_id,
            timestamp=time.time(),
            severity=severity,
            alert_type=alert_type,
            title=title,
            message=message,
            source=source,
            metadata=metadata or {}
        )
        
        with self._lock:
            if len(self._active_alerts) < self.config.max_active_alerts:
                self._active_alerts[alert_id] = alert
            self._alert_history.append(alert)
        
        self._send_notifications(alert)
        logger.warning(f"Manual alert triggered: {title} - {message}")
        return alert_id
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "user",
                     resolution_notes: Optional[str] = None):
        """
        Resolve an active alert.
        
        Args:
            alert_id: ID of the alert to resolve
            resolved_by: Who resolved the alert
            resolution_notes: Notes about the resolution
        """
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_timestamp = time.time()
                alert.resolved_by = resolved_by
                alert.resolution_notes = resolution_notes
                
                # Remove from active alerts
                del self._active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {alert.title} by {resolved_by}")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None,
                         alert_type: Optional[AlertType] = None) -> List[Alert]:
        """
        Get active alerts, optionally filtered by severity or type.
        
        Args:
            severity: Filter by severity
            alert_type: Filter by alert type
        
        Returns:
            List of active Alert objects
        """
        with self._lock:
            alerts = list(self._active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(self, duration_seconds: Optional[float] = None,
                         severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """
        Get alert history.
        
        Args:
            duration_seconds: Only return alerts from this many seconds ago
            severity: Filter by severity
        
        Returns:
            List of Alert objects from history
        """
        with self._lock:
            alerts = list(self._alert_history)
        
        if duration_seconds:
            cutoff_time = time.time() - duration_seconds
            alerts = [a for a in alerts if a.timestamp >= cutoff_time]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_statistics(self, duration_seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        Get alert statistics.
        
        Args:
            duration_seconds: Analyze alerts from this many seconds ago
        
        Returns:
            Dictionary with alert statistics
        """
        alerts = self.get_alert_history(duration_seconds)
        
        if not alerts:
            return {
                'total_alerts': 0,
                'active_alerts': 0,
                'resolved_alerts': 0,
                'by_severity': {},
                'by_type': {},
                'by_source': {},
                'avg_resolution_time': 0
            }
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in alerts:
            severity_counts[alert.severity.value] += 1
        
        # Count by type
        type_counts = defaultdict(int)
        for alert in alerts:
            type_counts[alert.alert_type.value] += 1
        
        # Count by source
        source_counts = defaultdict(int)
        for alert in alerts:
            source_counts[alert.source] += 1
        
        # Resolution statistics
        resolved_alerts = [a for a in alerts if a.resolved]
        resolution_times = [a.duration_seconds for a in resolved_alerts if a.duration_seconds]
        avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
        
        return {
            'total_alerts': len(alerts),
            'active_alerts': len([a for a in alerts if not a.resolved]),
            'resolved_alerts': len(resolved_alerts),
            'by_severity': dict(severity_counts),
            'by_type': dict(type_counts),
            'by_source': dict(source_counts),
            'avg_resolution_time': avg_resolution_time,
            'resolution_rate': len(resolved_alerts) / len(alerts) * 100 if alerts else 0
        }
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler function."""
        self._notification_handlers.append(handler)
    
    def remove_notification_handler(self, handler: Callable[[Alert], None]):
        """Remove a notification handler function."""
        if handler in self._notification_handlers:
            self._notification_handlers.remove(handler)
    
    def clear_alerts(self, severity: Optional[AlertSeverity] = None):
        """
        Clear alerts.
        
        Args:
            severity: If specified, only clear alerts of this severity
        """
        with self._lock:
            if severity:
                # Clear specific severity
                alerts_to_remove = [
                    alert_id for alert_id, alert in self._active_alerts.items()
                    if alert.severity == severity
                ]
                for alert_id in alerts_to_remove:
                    del self._active_alerts[alert_id]
            else:
                # Clear all active alerts
                self._active_alerts.clear()
    
    def export_alerts(self, filepath: str, duration_seconds: Optional[float] = None):
        """
        Export alert data to a file.
        
        Args:
            filepath: Path to export file (JSON format)
            duration_seconds: If specified, only export recent alerts
        """
        alerts = self.get_alert_history(duration_seconds)
        active_alerts = self.get_active_alerts()
        statistics = self.get_alert_statistics(duration_seconds)
        
        export_data = {
            'export_timestamp': time.time(),
            'duration_filter_seconds': duration_seconds,
            'statistics': statistics,
            'active_alerts': [],
            'alert_history': []
        }
        
        # Convert active alerts
        for alert in active_alerts:
            alert_data = {
                'id': alert.id,
                'timestamp': alert.timestamp,
                'severity': alert.severity.value,
                'alert_type': alert.alert_type.value,
                'title': alert.title,
                'message': alert.message,
                'source': alert.source,
                'metadata': alert.metadata,
                'resolved': alert.resolved,
                'age_seconds': alert.age_seconds
            }
            export_data['active_alerts'].append(alert_data)
        
        # Convert alert history (sample to avoid huge files)
        sample_size = min(1000, len(alerts))
        for alert in alerts[:sample_size]:
            alert_data = {
                'id': alert.id,
                'timestamp': alert.timestamp,
                'severity': alert.severity.value,
                'alert_type': alert.alert_type.value,
                'title': alert.title,
                'message': alert.message,
                'source': alert.source,
                'metadata': alert.metadata,
                'resolved': alert.resolved,
                'resolved_timestamp': alert.resolved_timestamp,
                'resolved_by': alert.resolved_by,
                'resolution_notes': alert.resolution_notes,
                'duration_seconds': alert.duration_seconds
            }
            export_data['alert_history'].append(alert_data)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported alert data to {filepath}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._auto_resolve_timer:
            self._auto_resolve_timer.cancel()