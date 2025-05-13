"""
GH200 Trading System Logging Package

This package provides simple, low-latency logging facilities for the trading system.
It offers:
- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- High-performance logging with minimal overhead
- Thread-safe logging capabilities
- Performance measurement utilities for function timing
- Structured logging for easier analysis and monitoring

The logging package is optimized for high-frequency trading environments where
logging overhead must be minimized while still providing comprehensive diagnostics.
"""

from .logging import (
    # Log levels
    DEBUG, INFO, WARNING, ERROR, CRITICAL,
    
    # Logger functions
    setup_logger, logger,
    
    # Convenience functions
    debug, info, warning, error, critical,
    
    # Performance measurement
    time_function
)

# Default exports when using "from log import *"
__all__ = [
    'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL',
    'setup_logger', 'logger',
    'debug', 'info', 'warning', 'error', 'critical',
    'time_function'
]
