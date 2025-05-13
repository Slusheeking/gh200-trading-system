"""
System Exporter for GH200 Trading System

This module provides a comprehensive system exporter that collects and exports
various metrics from the trading system, including system metrics, hardware metrics,
portfolio metrics, trade data, logs, and alerts.
"""

import os
import time
import threading
import logging
import json
from typing import Dict, Any, List, Optional

# Import collectors
from src.monitoring.collectors.system_metrics_collector import SystemMetricsCollector
from src.monitoring.collectors.hardware_metrics_collector import HardwareMetricsCollector
from src.monitoring.collectors.log_collector import LogCollector
from src.monitoring.collectors.notification_collector import NotificationCollector
from src.monitoring.collectors.alpaca_portfolio_collector import AlpacaPortfolioCollector
from src.monitoring.collectors.trade_metrics_collector import TradeMetricsCollector
from src.monitoring.collectors.yahoo_finance_client import YahooFinanceClient

class SystemExporter:
    """
    Main system exporter class that orchestrates the collection and export of metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the system exporter
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.running = False
        self.export_thread = None
        self.export_interval = config.get("export_interval", 60)  # seconds
        
        # Initialize collectors
        self.system_metrics_collector = SystemMetricsCollector(config)
        self.hardware_metrics_collector = HardwareMetricsCollector(config)
        self.log_collector = LogCollector(config)
        self.notification_collector = NotificationCollector(config)
        self.alpaca_portfolio_collector = AlpacaPortfolioCollector(config)
        self.trade_metrics_collector = TradeMetricsCollector(config)
        self.yahoo_finance_client = YahooFinanceClient(config)
        
        logging.info("System exporter initialized")
    
    def start(self):
        """Start the system exporter"""
        if self.running:
            logging.warning("System exporter already running")
            return
        
        self.running = True
        
        # Start collectors
        self.system_metrics_collector.start()
        self.hardware_metrics_collector.start()
        self.log_collector.start()
        self.notification_collector.start()
        self.alpaca_portfolio_collector.start()
        self.trade_metrics_collector.start()
        
        # Start export thread if periodic export is enabled
        if self.config.get("enable_periodic_export", False):
            self.export_thread = threading.Thread(target=self._export_loop, daemon=True)
            self.export_thread.start()
        
        logging.info("System exporter started")
    
    def stop(self):
        """Stop the system exporter"""
        if not self.running:
            logging.warning("System exporter not running")
            return
        
        self.running = False
        
        # Stop collectors
        self.system_metrics_collector.stop()
        self.hardware_metrics_collector.stop()
        self.log_collector.stop()
        self.notification_collector.stop()
        self.alpaca_portfolio_collector.stop()
        self.trade_metrics_collector.stop()
        
        # Wait for export thread to finish
        if self.export_thread and self.export_thread.is_alive():
            self.export_thread.join(timeout=5.0)
        
        logging.info("System exporter stopped")
    
    def _export_loop(self):
        """Export loop for periodic exporting"""
        while self.running:
            try:
                self.export_metrics()
            except Exception as e:
                logging.error(f"Error exporting metrics: {str(e)}")
            
            # Sleep until next export
            time.sleep(self.export_interval)
    
    def export_metrics(self):
        """Export all metrics"""
        # Collect all metrics
        metrics = {
            "timestamp": int(time.time()),
            "system_metrics": self.system_metrics_collector.get_metrics(),
            "hardware_metrics": self.hardware_metrics_collector.get_metrics(),
            "portfolio_metrics": self.alpaca_portfolio_collector.get_metrics(),
            "active_trades": self.trade_metrics_collector.get_active_trades(),
            "alerts": self.notification_collector.get_recent_alerts()
        }
        
        # Export to configured destinations
        self._export_to_file(metrics)
        
        logging.debug("Metrics exported")
        return metrics
    
    def _export_to_file(self, metrics: Dict[str, Any]):
        """Export metrics to file"""
        if not self.config.get("export_to_file", False):
            return
        
        # Get export directory
        export_dir = self.config.get("export_directory", "exports")
        os.makedirs(export_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = metrics["timestamp"]
        filename = f"metrics_{timestamp}.json"
        filepath = os.path.join(export_dir, filename)
        
        # Write metrics to file
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logging.debug(f"Metrics exported to {filepath}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get system metrics
        
        Returns:
            Dictionary with system metrics
        """
        return self.system_metrics_collector.get_metrics()
    
    def get_hardware_metrics(self) -> Dict[str, Any]:
        """
        Get hardware metrics
        
        Returns:
            Dictionary with hardware metrics
        """
        return self.hardware_metrics_collector.get_metrics()
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Get portfolio metrics
        
        Returns:
            Dictionary with portfolio metrics
        """
        return self.alpaca_portfolio_collector.get_metrics()
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """
        Get active trades
        
        Returns:
            List of active trades
        """
        return self.trade_metrics_collector.get_active_trades()
    
    def get_trade_history(self, limit: int = 100, offset: int = 0, 
                         symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get trade history
        
        Args:
            limit: Maximum number of trades to return
            offset: Offset for pagination
            symbol: Filter by symbol
            
        Returns:
            List of historical trades
        """
        return self.trade_metrics_collector.get_historical_trades(
            limit=limit, offset=offset, symbol=symbol
        )
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        Get trade statistics
        
        Returns:
            Dictionary with trade statistics
        """
        return self.trade_metrics_collector.get_trade_statistics()
    
    def get_logs(self, level: Optional[str] = None, limit: int = 100, 
                offset: int = 0, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get logs
        
        Args:
            level: Filter by log level
            limit: Maximum number of logs to return
            offset: Offset for pagination
            component: Filter by component
            
        Returns:
            List of log entries
        """
        return self.log_collector.get_logs(
            level=level, limit=limit, offset=offset, component=component
        )
    
    def get_alerts(self, severity: Optional[str] = None, category: Optional[str] = None,
                  acknowledged: Optional[bool] = None, limit: int = 100, 
                  offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get alerts
        
        Args:
            severity: Filter by severity
            category: Filter by category
            acknowledged: Filter by acknowledged status
            limit: Maximum number of alerts to return
            offset: Offset for pagination
            
        Returns:
            List of alerts
        """
        return self.notification_collector.get_alerts(
            severity=severity, category=category, acknowledged=acknowledged,
            limit=limit, offset=offset
        )
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """
        Acknowledge an alert
        
        Args:
            alert_id: Alert ID
            user: User who acknowledged the alert
            
        Returns:
            True if alert was found and acknowledged, False otherwise
        """
        return self.notification_collector.acknowledge_alert(alert_id, user)
    
    def get_chart_data(self, symbol: str, period: str = "1y", interval: str = "1d", 
                      chart_type: str = "candlestick", include_indicators: bool = True) -> Dict[str, Any]:
        """
        Get chart data for a symbol
        
        Args:
            symbol: Stock symbol
            period: Time period
            interval: Data interval
            chart_type: Chart type (candlestick, line, ohlc)
            include_indicators: Whether to include technical indicators
            
        Returns:
            Dictionary with chart data
        """
        return self.yahoo_finance_client.get_chart_data(
            symbol, period, interval, chart_type, include_indicators
        )
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with quote data
        """
        return self.yahoo_finance_client.get_quote(symbol)
    
    def get_daily_pnl(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get daily profit/loss
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with daily profit/loss
        """
        return self.trade_metrics_collector.get_daily_pnl(start_date=start_date, end_date=end_date)
    
    def get_pnl_calendar(self, year: Optional[int] = None, month: Optional[int] = None) -> Dict[str, Any]:
        """
        Get profit/loss calendar
        
        Args:
            year: Year (e.g., 2025)
            month: Month (1-12)
            
        Returns:
            Dictionary with profit/loss calendar
        """
        return self.trade_metrics_collector.get_pnl_calendar(year=year, month=month)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics
        
        Returns:
            Dictionary with all metrics
        """
        return {
            "timestamp": int(time.time()),
            "system_metrics": self.get_system_metrics(),
            "hardware_metrics": self.get_hardware_metrics(),
            "portfolio_metrics": self.get_portfolio_metrics(),
            "active_trades": self.get_active_trades(),
            "trade_statistics": self.get_trade_statistics(),
            "daily_pnl": self.get_daily_pnl(),
            "alerts": self.notification_collector.get_recent_alerts()
        }
