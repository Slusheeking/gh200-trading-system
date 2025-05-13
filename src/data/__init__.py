"""
Data Package for GH200 Trading System

This package provides clients for the Polygon.io API and a market data processor.
It includes:
- PolygonRestAPI: Client for the Polygon.io REST API for historical data and reference data
- PolygonWebSocket: Client for the Polygon.io WebSocket API for real-time market data
- MarketDataProcessor: Processor for market data with GPU acceleration for high-performance analysis

The data package is responsible for fetching, processing, and managing market data
from external sources, with optimizations for high-frequency trading applications.
"""

from .polygon_rest_api import PolygonRestAPI
from .polygon_websocket import PolygonWebSocket
from .market_data_processor import MarketDataProcessor

# Define __all__ for explicit exports
__all__ = [
    'PolygonRestAPI',
    'PolygonWebSocket',
    'MarketDataProcessor',
]
