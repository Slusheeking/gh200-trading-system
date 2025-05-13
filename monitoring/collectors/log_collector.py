"""
Log collector for the system exporter

This module provides a collector for system logs, interfacing with the existing logging system.
"""

import os
import time
import threading
import logging
import re
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

class LogCollector:
    """Collector for system logs"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the log collector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.running = False
        self.collection_thread = None
        self.collection_interval = config.get("log_collector", {}).get("collection_interval", 60)  # seconds
        
        # Get log file path from config
        logging_config = config.get("logging", {})
        self.log_file = logging_config.get("file", "logs/trading_system.log")
        self.log_path = Path(self.log_file)
        self.log_dir = self.log_path.parent
        
        # Log rotation settings
        self.max_files = logging_config.get("max_files", 10)
        self.rotate_logs_daily = logging_config.get("rotate_logs_daily", False)
        self.enable_log_compression = logging_config.get("enable_log_compression", False)
        
        # Log parsing settings
        self.log_format = logging_config.get("log_format", "[%(asctime)s] [%(levelname)s] [%(thread)d] %(message)s")
        self.use_structured_logging = logging_config.get("use_structured_logging", False)
        
        # Initialize log cache
        self.log_cache = []
        self.log_cache_size = config.get("log_collector", {}).get("cache_size", 1000)
        self.log_cache_lock = threading.Lock()
        
        # Compile regex for log parsing
        self._compile_log_regex()
    
    def _compile_log_regex(self):
        """Compile regex for parsing log entries"""
        if self.use_structured_logging:
            # No regex needed for JSON logs
            self.log_regex = None
        else:
            # Create regex based on log format
            # This is a simplified version that works for common formats
            self.log_regex = re.compile(
                r'\[(.*?)\] \[(.*?)\] \[(.*?)\](?: \[(.*?)\])?(?: \[(.*?)\])? (.*)'
            )
    
    def start(self):
        """Start the collector"""
        if self.running:
            logging.warning("Log collector already running")
            return
        
        self.running = True
        
        # Start collection thread
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logging.info("Log collector started")
    
    def stop(self):
        """Stop the collector"""
        if not self.running:
            logging.warning("Log collector not running")
            return
        
        self.running = False
        
        # Wait for collection thread to finish
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
        
        logging.info("Log collector stopped")
    
    def _collection_loop(self):
        """Collection loop for periodic collection"""
        # Keep track of file position
        file_position = 0
        current_file = self.log_path
        
        while self.running:
            try:
                # Check if log file exists
                if not current_file.exists():
                    time.sleep(self.collection_interval)
                    continue
                
                # Open log file and seek to last position
                with open(current_file, "r") as f:
                    # Get file size
                    f.seek(0, os.SEEK_END)
                    file_size = f.tell()
                    
                    # If file size is smaller than last position, file was rotated
                    if file_size < file_position:
                        file_position = 0
                    
                    # Seek to last position
                    f.seek(file_position)
                    
                    # Read new lines
                    new_lines = f.readlines()
                    
                    # Update file position
                    file_position = f.tell()
                
                # Parse new log entries
                if new_lines:
                    self._parse_log_entries(new_lines)
            
            except Exception as e:
                logging.error(f"Error collecting logs: {str(e)}")
            
            # Sleep until next collection
            time.sleep(self.collection_interval)
    
    def _parse_log_entries(self, lines: List[str]):
        """
        Parse log entries from lines
        
        Args:
            lines: List of log lines
        """
        entries = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            try:
                if self.use_structured_logging:
                    # Parse JSON log entry
                    entry = json.loads(line)
                    
                    # Ensure required fields
                    if "timestamp" not in entry:
                        entry["timestamp"] = datetime.now().isoformat()
                    if "level" not in entry:
                        entry["level"] = "INFO"
                    if "message" not in entry:
                        entry["message"] = ""
                    
                    entries.append(entry)
                else:
                    # Parse log entry using regex
                    match = self.log_regex.match(line)
                    if match:
                        groups = match.groups()
                        
                        # Extract fields based on log format
                        timestamp = groups[0]
                        level = groups[1]
                        thread_id = groups[2]
                        logger = groups[3] if len(groups) > 3 else None
                        file_info = groups[4] if len(groups) > 4 else None
                        message = groups[5] if len(groups) > 5 else groups[3]
                        
                        entry = {
                            "timestamp": timestamp,
                            "level": level,
                            "thread_id": thread_id,
                            "logger": logger,
                            "file_info": file_info,
                            "message": message
                        }
                        
                        entries.append(entry)
                    else:
                        # Fallback for lines that don't match the regex
                        entries.append({
                            "timestamp": datetime.now().isoformat(),
                            "level": "INFO",
                            "message": line
                        })
            except Exception as e:
                logging.error(f"Error parsing log entry: {str(e)}")
        
        # Add entries to cache
        if entries:
            with self.log_cache_lock:
                self.log_cache.extend(entries)
                
                # Trim cache if it exceeds the maximum size
                if len(self.log_cache) > self.log_cache_size:
                    self.log_cache = self.log_cache[-self.log_cache_size:]
    
    def get_logs(self, level: Optional[str] = None, limit: int = 100, offset: int = 0, 
                component: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get logs with filtering and pagination
        
        Args:
            level: Filter by log level (e.g., "INFO", "ERROR")
            limit: Maximum number of logs to return
            offset: Offset for pagination
            component: Filter by component/logger name
            
        Returns:
            List of log entries
        """
        with self.log_cache_lock:
            # Apply filters
            filtered_logs = self.log_cache
            
            if level:
                filtered_logs = [log for log in filtered_logs if log.get("level", "").upper() == level.upper()]
            
            if component:
                filtered_logs = [log for log in filtered_logs if component.lower() in (log.get("logger", "").lower() or "")]
            
            # Apply pagination
            paginated_logs = filtered_logs[offset:offset + limit]
            
            # Return a copy to avoid threading issues
            return paginated_logs.copy()
    
    def get_log_stats(self) -> Dict[str, Any]:
        """
        Get statistics about logs
        
        Returns:
            Dictionary with log statistics
        """
        with self.log_cache_lock:
            # Count logs by level
            level_counts = {}
            for log in self.log_cache:
                level = log.get("level", "UNKNOWN").upper()
                level_counts[level] = level_counts.get(level, 0) + 1
            
            # Get time range
            timestamps = [log.get("timestamp") for log in self.log_cache if "timestamp" in log]
            time_range = {
                "start": min(timestamps) if timestamps else None,
                "end": max(timestamps) if timestamps else None
            }
            
            return {
                "total_logs": len(self.log_cache),
                "level_counts": level_counts,
                "time_range": time_range
            }
    
    def search_logs(self, query: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Search logs for a query string
        
        Args:
            query: Query string to search for
            case_sensitive: Whether the search is case sensitive
            
        Returns:
            List of matching log entries
        """
        with self.log_cache_lock:
            if not case_sensitive:
                query = query.lower()
                return [log for log in self.log_cache if query in log.get("message", "").lower()]
            else:
                return [log for log in self.log_cache if query in log.get("message", "")]
    
    def get_error_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get error logs
        
        Args:
            limit: Maximum number of logs to return
            
        Returns:
            List of error log entries
        """
        return self.get_logs(level="ERROR", limit=limit)
    
    def get_recent_logs(self, minutes: int = 5, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent logs
        
        Args:
            minutes: Number of minutes to look back
            level: Filter by log level
            
        Returns:
            List of recent log entries
        """
        with self.log_cache_lock:
            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            cutoff_str = cutoff_time.isoformat()
            
            # Filter logs by time
            recent_logs = [log for log in self.log_cache if log.get("timestamp", "") >= cutoff_str]
            
            # Apply level filter if specified
            if level:
                recent_logs = [log for log in recent_logs if log.get("level", "").upper() == level.upper()]
            
            return recent_logs.copy()
