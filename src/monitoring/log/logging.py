"""
Simple, low-latency logging for trading systems.

Uses Python's standard logging module with minimal overhead.
Configuration is loaded from system.yaml.
"""

import logging
import sys
import time
import os
import gzip
import json
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import importlib.util
import threading
import uuid
from functools import wraps

# Standard log levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# Global trace ID for distributed tracing
_thread_local = threading.local()

def _gzip_rotator(source, dest):
    """
    Compress rotated log files with gzip.
    
    Args:
        source: Source file path
        dest: Destination file path
    """
    with open(source, 'rb') as f_in:
        with gzip.open(f"{dest}.gz", 'wb') as f_out:
            f_out.writelines(f_in)
    os.remove(source)  # Remove the original file after compression

# Import config_loader dynamically to avoid circular imports
def _get_config():
    """Get configuration from config_loader"""
    try:
        # Check if config_loader module exists in the config package
        config_loader_path = Path(__file__).parents[3] / "config" / "config_loader.py"
        if config_loader_path.exists():
            spec = importlib.util.spec_from_file_location("config_loader", config_loader_path)
            config_loader = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_loader)
            return config_loader.get_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
    
    # Return empty dict if config_loader not available
    return {}

def setup_logger(name="trading_system", level=None, log_to_console=True):
    """
    Set up and configure a logger based on system.yaml configuration.
    
    Args:
        name: Logger name and log file name
        level: Minimum logging level (overrides config if provided)
        log_to_console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    # Get configuration
    config = _get_config()
    logging_config = config.get("logging", {})
    
    # Check for component-specific log level
    component_levels = logging_config.get("component_levels", {})
    component_level = None
    for component_name, component_level_str in component_levels.items():
        if component_name in name:
            component_level = getattr(logging, component_level_str, None)
            break
    
    # Get log level from config or use provided level or default to INFO
    if level is None:
        if component_level is not None:
            level = component_level
        else:
            level_str = logging_config.get("level", "INFO")
            level = getattr(logging, level_str, INFO)
    
    # Get log file path from config or use default
    log_file = logging_config.get("file", "logs/trading_system.log")
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_dir = log_path.parent
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Get log format from config or use default
    log_format = logging_config.get("log_format", "[%(asctime)s] [%(levelname)s] [%(thread)d] %(message)s")
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Use structured logging if enabled
    use_structured_logging = logging_config.get("use_structured_logging", False)
    if use_structured_logging:
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": self.formatTime(record, self.datefmt),
                    "level": record.levelname,
                    "thread_id": record.thread,
                    "logger": record.name,
                    "file": record.filename,
                    "line": record.lineno,
                    "message": record.getMessage(),
                }
                
                # Add trace ID if available
                if hasattr(record, 'trace_id'):
                    log_data["trace_id"] = record.trace_id
                
                # Add any extra attributes
                if hasattr(record, 'extra'):
                    log_data.update(record.extra)
                
                return json.dumps(log_data)
        
        formatter = JsonFormatter()
    
    # Configure file handler based on settings
    max_file_size_mb = logging_config.get("max_file_size_mb", 100)
    max_files = logging_config.get("max_files", 10)
    rotate_logs_daily = logging_config.get("rotate_logs_daily", False)
    enable_log_compression = logging_config.get("enable_log_compression", False)
    
    if rotate_logs_daily:
        # Use time-based rotation
        file_handler = TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=max_files
        )
        
        # Add compression if enabled
        if enable_log_compression:
            file_handler.rotator = _gzip_rotator
    else:
        # Use size-based rotation
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=max_files
        )
        
        # Add compression if enabled
        if enable_log_compression:
            file_handler.rotator = _gzip_rotator
    
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (optional)
    if log_to_console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)
    
    # Log configuration details
    logger.debug(f"Logger configured with level={logging.getLevelName(level)}, file={log_file}")
    logger.debug(f"Max file size: {max_file_size_mb}MB, Max files: {max_files}")
    logger.debug(f"Structured logging: {use_structured_logging}, Daily rotation: {rotate_logs_daily}, Compression: {enable_log_compression}")
    
    return logger


# Create the global logger
logger = setup_logger()

# Log performance metrics if enabled
if _get_config().get("logging", {}).get("log_latency_stats", False):
    logger.info("Performance logging enabled")

# Distributed tracing functions
def get_trace_id():
    """
    Get the current trace ID for distributed tracing.
    
    Returns:
        Current trace ID or None if not in a trace context
    """
    if not hasattr(_thread_local, 'trace_id'):
        return None
    return _thread_local.trace_id

def set_trace_id(trace_id=None):
    """
    Set the trace ID for the current thread.
    
    Args:
        trace_id: Trace ID to set, or generate a new one if None
    
    Returns:
        The trace ID that was set
    """
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    _thread_local.trace_id = trace_id
    return trace_id

def clear_trace_id():
    """Clear the trace ID for the current thread."""
    if hasattr(_thread_local, 'trace_id'):
        delattr(_thread_local, 'trace_id')

def trace_context(func=None, trace_id=None):
    """
    Decorator to create a trace context for a function.
    
    Args:
        func: Function to decorate
        trace_id: Optional trace ID to use
        
    Returns:
        Decorated function
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            old_trace_id = get_trace_id()
            new_trace_id = set_trace_id(trace_id)
            try:
                logger.debug(f"Starting trace {new_trace_id} for {f.__name__}")
                return f(*args, **kwargs)
            finally:
                logger.debug(f"Ending trace {new_trace_id} for {f.__name__}")
                if old_trace_id:
                    set_trace_id(old_trace_id)
                else:
                    clear_trace_id()
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)


# Enhanced timer for performance measurement
def time_function(func=None, track_percentiles=False, log_level=INFO):
    """
    Decorator to measure and log function execution time.
    
    Args:
        func: Function to time
        track_percentiles: Whether to track percentiles for this function
        log_level: Log level to use for timing messages
        
    Returns:
        Wrapped function that logs execution time
    """
    # Store execution times for percentile tracking
    execution_times = []
    
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Get trace ID if available
            trace_id = get_trace_id()
            trace_info = f" [trace_id={trace_id}]" if trace_id else ""
            
            # Time the function
            start = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                # Log execution time
                logger.log(log_level, f"{f.__name__} executed in {elapsed_ms:.2f} ms{trace_info}")
                
                # Track percentiles if enabled
                if track_percentiles:
                    execution_times.append(elapsed_ms)
                    # Log percentiles every 100 executions
                    if len(execution_times) % 100 == 0:
                        _log_percentiles(f.__name__, execution_times)
                
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error(f"{f.__name__} failed after {elapsed_ms:.2f} ms{trace_info}: {str(e)}")
                raise
        
        # Store execution times on the wrapper for external access
        wrapper.execution_times = execution_times
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)

def _log_percentiles(func_name, times):
    """Log percentiles for a function's execution times."""
    if not times:
        return
    
    import numpy as np
    times_array = np.array(times)
    percentiles = [50, 90, 95, 99, 99.9]
    percentile_values = {p: float(np.percentile(times_array, p)) for p in percentiles}
    
    logger.info(f"Latency percentiles for {func_name} (ms):")
    for p, v in percentile_values.items():
        logger.info(f"  p{p}: {v:.2f}")


# Convenience functions
def debug(message, *args, **kwargs):
    """Log debug message."""
    logger.debug(message, *args, **kwargs)


def info(message, *args, **kwargs):
    """Log info message."""
    logger.info(message, *args, **kwargs)


def warning(message, *args, **kwargs):
    """Log warning message."""
    logger.warning(message, *args, **kwargs)


def error(message, *args, **kwargs):
    """Log error message."""
    logger.error(message, *args, **kwargs)


def critical(message, *args, **kwargs):
    """Log critical message."""
    logger.critical(message, *args, **kwargs)
