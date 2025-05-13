"""
Notification collector for the system exporter

This module provides a collector for system notifications and alerts.
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional
from enum import Enum, auto

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class AlertCategory(Enum):
    """Alert categories"""
    SYSTEM = auto()
    HARDWARE = auto()
    NETWORK = auto()
    TRADING = auto()
    PERFORMANCE = auto()
    SECURITY = auto()
    OTHER = auto()

class Alert:
    """Class representing an alert"""
    
    def __init__(self, message: str, severity: AlertSeverity, category: AlertCategory,
                timestamp: Optional[float] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize an alert
        
        Args:
            message: Alert message
            severity: Alert severity
            category: Alert category
            timestamp: Alert timestamp (defaults to current time)
            details: Additional alert details
        """
        self.message = message
        self.severity = severity
        self.category = category
        self.timestamp = timestamp or time.time()
        self.details = details or {}
        self.id = f"{int(self.timestamp * 1000)}_{hash(message) % 10000}"
        self.acknowledged = False
        self.acknowledged_at = None
        self.acknowledged_by = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "id": self.id,
            "message": self.message,
            "severity": self.severity.name,
            "category": self.category.name,
            "timestamp": self.timestamp,
            "details": self.details,
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at,
            "acknowledged_by": self.acknowledged_by
        }
    
    def acknowledge(self, user: str = "system"):
        """
        Acknowledge the alert
        
        Args:
            user: User who acknowledged the alert
        """
        self.acknowledged = True
        self.acknowledged_at = time.time()
        self.acknowledged_by = user

class NotificationCollector:
    """Collector for system notifications and alerts"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the notification collector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.running = False
        self.collection_thread = None
        self.collection_interval = config.get("notification_collector", {}).get("collection_interval", 60)  # seconds
        
        # Initialize alert storage
        self.alerts = []
        self.max_alerts = config.get("notification_collector", {}).get("max_alerts", 1000)
        self.alerts_lock = threading.Lock()
        
        # Alert thresholds
        self.thresholds = self._load_thresholds()
        
        # Initialize notification handlers
        self.notification_handlers = []
        self._setup_notification_handlers()
    
    def _load_thresholds(self) -> Dict[str, Any]:
        """Load alert thresholds from config"""
        thresholds = self.config.get("notification_collector", {}).get("thresholds", {})
        
        # Set default thresholds if not specified
        if not thresholds:
            thresholds = {
                "cpu_usage": 90.0,  # CPU usage percentage
                "memory_usage": 90.0,  # Memory usage percentage
                "disk_usage": 90.0,  # Disk usage percentage
                "gpu_temperature": 85.0,  # GPU temperature in Celsius
                "network_errors": 100,  # Network errors per minute
                "latency": 1000.0,  # Latency in milliseconds
                "error_rate": 5.0,  # Error rate percentage
            }
        
        return thresholds
    
    def _setup_notification_handlers(self):
        """Set up notification handlers"""
        # Add console handler
        self.notification_handlers.append(self._console_handler)
        
        # Add log handler
        self.notification_handlers.append(self._log_handler)
        
        # Add custom handlers from config
        handlers = self.config.get("notification_collector", {}).get("handlers", [])
        for handler in handlers:
            if handler == "email" and self.config.get("notification_collector", {}).get("email", {}).get("enabled", False):
                self.notification_handlers.append(self._email_handler)
            elif handler == "webhook" and self.config.get("notification_collector", {}).get("webhook", {}).get("enabled", False):
                self.notification_handlers.append(self._webhook_handler)
    
    def start(self):
        """Start the collector"""
        if self.running:
            logging.warning("Notification collector already running")
            return
        
        self.running = True
        
        # Start collection thread
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logging.info("Notification collector started")
    
    def stop(self):
        """Stop the collector"""
        if not self.running:
            logging.warning("Notification collector not running")
            return
        
        self.running = False
        
        # Wait for collection thread to finish
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
        
        logging.info("Notification collector stopped")
    
    def _collection_loop(self):
        """Collection loop for periodic collection"""
        while self.running:
            try:
                # Check for new alerts based on metrics
                self._check_alerts()
            except Exception as e:
                logging.error(f"Error checking alerts: {str(e)}")
            
            # Sleep until next collection
            time.sleep(self.collection_interval)
    
    def _check_alerts(self):
        """Check for new alerts based on metrics"""
        # This method would typically check various metrics and generate alerts
        # For now, it's a placeholder that would be implemented based on specific metrics
        pass
    
    def add_alert(self, message: str, severity: AlertSeverity, category: AlertCategory,
                 details: Optional[Dict[str, Any]] = None) -> Alert:
        """
        Add a new alert
        
        Args:
            message: Alert message
            severity: Alert severity
            category: Alert category
            details: Additional alert details
            
        Returns:
            The created alert
        """
        # Create alert
        alert = Alert(message, severity, category, details=details)
        
        # Add to alerts list
        with self.alerts_lock:
            self.alerts.append(alert)
            
            # Trim alerts if exceeding maximum
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
        
        # Notify handlers
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Error in notification handler: {str(e)}")
        
        return alert
    
    def get_alerts(self, severity: Optional[str] = None, category: Optional[str] = None,
                  acknowledged: Optional[bool] = None, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get alerts with filtering and pagination
        
        Args:
            severity: Filter by severity
            category: Filter by category
            acknowledged: Filter by acknowledged status
            limit: Maximum number of alerts to return
            offset: Offset for pagination
            
        Returns:
            List of alert dictionaries
        """
        with self.alerts_lock:
            # Apply filters
            filtered_alerts = self.alerts
            
            if severity:
                try:
                    severity_enum = AlertSeverity[severity.upper()]
                    filtered_alerts = [alert for alert in filtered_alerts if alert.severity == severity_enum]
                except KeyError:
                    pass
            
            if category:
                try:
                    category_enum = AlertCategory[category.upper()]
                    filtered_alerts = [alert for alert in filtered_alerts if alert.category == category_enum]
                except KeyError:
                    pass
            
            if acknowledged is not None:
                filtered_alerts = [alert for alert in filtered_alerts if alert.acknowledged == acknowledged]
            
            # Sort by timestamp (newest first)
            sorted_alerts = sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)
            
            # Apply pagination
            paginated_alerts = sorted_alerts[offset:offset + limit]
            
            # Convert to dictionaries
            return [alert.to_dict() for alert in paginated_alerts]
    
    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """
        Get alert by ID
        
        Args:
            alert_id: Alert ID
            
        Returns:
            Alert dictionary or None if not found
        """
        with self.alerts_lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    return alert.to_dict()
            
            return None
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """
        Acknowledge an alert
        
        Args:
            alert_id: Alert ID
            user: User who acknowledged the alert
            
        Returns:
            True if alert was found and acknowledged, False otherwise
        """
        with self.alerts_lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledge(user)
                    return True
            
            return False
    
    def get_recent_alerts(self, minutes: int = 60, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent alerts
        
        Args:
            minutes: Number of minutes to look back
            severity: Filter by severity
            
        Returns:
            List of recent alert dictionaries
        """
        # Calculate cutoff time
        cutoff_time = time.time() - (minutes * 60)
        
        with self.alerts_lock:
            # Filter alerts by time
            recent_alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
            
            # Apply severity filter if specified
            if severity:
                try:
                    severity_enum = AlertSeverity[severity.upper()]
                    recent_alerts = [alert for alert in recent_alerts if alert.severity == severity_enum]
                except KeyError:
                    pass
            
            # Sort by timestamp (newest first)
            sorted_alerts = sorted(recent_alerts, key=lambda x: x.timestamp, reverse=True)
            
            # Convert to dictionaries
            return [alert.to_dict() for alert in sorted_alerts]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """
        Get statistics about alerts
        
        Returns:
            Dictionary with alert statistics
        """
        with self.alerts_lock:
            # Count alerts by severity
            severity_counts = {}
            for alert in self.alerts:
                severity = alert.severity.name
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count alerts by category
            category_counts = {}
            for alert in self.alerts:
                category = alert.category.name
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Count acknowledged vs unacknowledged
            acknowledged_count = sum(1 for alert in self.alerts if alert.acknowledged)
            unacknowledged_count = len(self.alerts) - acknowledged_count
            
            # Get time range
            timestamps = [alert.timestamp for alert in self.alerts]
            time_range = {
                "start": min(timestamps) if timestamps else None,
                "end": max(timestamps) if timestamps else None
            }
            
            return {
                "total_alerts": len(self.alerts),
                "severity_counts": severity_counts,
                "category_counts": category_counts,
                "acknowledged_count": acknowledged_count,
                "unacknowledged_count": unacknowledged_count,
                "time_range": time_range
            }
    
    # Notification handlers
    
    def _console_handler(self, alert: Alert):
        """
        Console notification handler
        
        Args:
            alert: Alert to handle
        """
        print(f"[ALERT] [{alert.severity.name}] [{alert.category.name}] {alert.message}")
    
    def _log_handler(self, alert: Alert):
        """
        Log notification handler
        
        Args:
            alert: Alert to handle
        """
        if alert.severity == AlertSeverity.INFO:
            logging.info(f"[ALERT] [{alert.category.name}] {alert.message}")
        elif alert.severity == AlertSeverity.WARNING:
            logging.warning(f"[ALERT] [{alert.category.name}] {alert.message}")
        elif alert.severity == AlertSeverity.ERROR:
            logging.error(f"[ALERT] [{alert.category.name}] {alert.message}")
        elif alert.severity == AlertSeverity.CRITICAL:
            logging.critical(f"[ALERT] [{alert.category.name}] {alert.message}")
    
    def _email_handler(self, alert: Alert):
        """
        Email notification handler
        
        Args:
            alert: Alert to handle
        """
        # This would be implemented to send email notifications
        # For now, it's a placeholder
        pass
    
    def _webhook_handler(self, alert: Alert):
        """
        Webhook notification handler
        
        Args:
            alert: Alert to handle
        """
        # This would be implemented to send webhook notifications
        # For now, it's a placeholder
        pass
