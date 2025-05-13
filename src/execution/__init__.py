"""
Execution Package for GH200 Trading System

This package provides execution capabilities for trading via Alpaca Markets.
It includes:
- AlpacaRestAPI: Client for the Alpaca Markets REST API
- ExecutionEngine: Engine for executing trades with GPU acceleration
- Order-related classes and enums for trade execution
"""

from .alpaca_rest_api import (
    AlpacaRestAPI,
    Order,
    BracketOrder,
    OrderResponse,
    OrderStatus,
    OrderSide,
    OrderType,
    OrderClass,
    TimeInForce
)

from .alpaca_execution import (
    ExecutionEngine,
    Signal,
    OrderSide as ExecutionOrderSide,
    OrderType as ExecutionOrderType,
    OrderClass as ExecutionOrderClass,
    TimeInForce as ExecutionTimeInForce,
    OrderStatus as ExecutionOrderStatus
)

# Define __all__ for explicit exports
__all__ = [
    # From alpaca_rest_api
    'AlpacaRestAPI',
    'Order',
    'BracketOrder',
    'OrderResponse',
    'OrderStatus',
    'OrderSide',
    'OrderType',
    'OrderClass',
    'TimeInForce',
    
    # From alpaca_execution
    'ExecutionEngine',
    'Signal',
    'ExecutionOrderSide',
    'ExecutionOrderType',
    'ExecutionOrderClass',
    'ExecutionTimeInForce',
    'ExecutionOrderStatus',
]
