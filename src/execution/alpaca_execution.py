"""
Alpaca execution module for the trading system.

This module provides the order definitions and execution engine that handles
trade execution using the Alpaca API directly.
"""

import time
import logging
import threading
import numpy as np
import uuid
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any

# Try to import cupy for GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

from src.execution.alpaca_rest_api import AlpacaRestAPI


#
# Order-related classes and enums
#

class OrderSide(str, Enum):
    """Order side enum"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enum"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderClass(str, Enum):
    """Order class enum"""
    SIMPLE = "simple"
    BRACKET = "bracket"
    OCO = "oco"  # One-Cancels-Other
    OTO = "oto"  # One-Triggers-Other


class TimeInForce(str, Enum):
    """Time in force enum"""
    DAY = "day"
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


class OrderStatus(str, Enum):
    """Order status enum"""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Class representing an order"""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    limit_price: float = 0.0
    stop_price: float = 0.0
    time_in_force: TimeInForce = TimeInForce.DAY
    client_order_id: str = ""
    order_class: OrderClass = OrderClass.SIMPLE
    take_profit_price: float = 0.0
    stop_loss_price: float = 0.0
    trail_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for API requests"""
        result = {
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.type.value,
            "qty": str(self.quantity),
            "time_in_force": self.time_in_force.value,
        }
        
        # Add optional fields
        if self.limit_price > 0:
            result["limit_price"] = str(self.limit_price)
        
        if self.stop_price > 0:
            result["stop_price"] = str(self.stop_price)
        
        if self.client_order_id:
            result["client_order_id"] = self.client_order_id
        
        # Handle bracket orders
        if self.order_class == OrderClass.BRACKET:
            result["order_class"] = "bracket"
            
            if self.take_profit_price > 0:
                result["take_profit"] = {
                    "limit_price": str(self.take_profit_price)
                }
            
            if self.stop_loss_price > 0:
                stop_loss = {
                    "stop_price": str(self.stop_loss_price)
                }
                
                # Add trailing stop if enabled
                if self.trail_percent > 0:
                    stop_loss["trail_percent"] = str(self.trail_percent)
                
                result["stop_loss"] = stop_loss
        
        return result


@dataclass
class BracketOrder:
    """Class representing a bracket order"""
    entry: Order


@dataclass
class OrderResponse:
    """Class representing an order response"""
    order_id: str = ""
    client_order_id: str = ""
    symbol: str = ""
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    status_message: str = ""
    
    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> 'OrderResponse':
        """Create OrderResponse from API response"""
        try:
            return cls(
                order_id=response.get("id", ""),
                client_order_id=response.get("client_order_id", ""),
                symbol=response.get("symbol", ""),
                status=OrderStatus(response.get("status", "new")),
                filled_quantity=float(response.get("filled_qty", "0") or "0"),
                filled_price=float(response.get("filled_avg_price", "0") or "0"),
                status_message=response.get("status_message", "")
            )
        except Exception as e:
            # Return a rejected order response if parsing fails
            return cls(
                client_order_id=response.get("client_order_id", ""),
                symbol=response.get("symbol", ""),
                status=OrderStatus.REJECTED,
                status_message=f"Failed to parse API response: {str(e)}"
            )


@dataclass
class Signal:
    """Class representing a trading signal"""
    symbol: str
    type: str  # "BUY" or "SELL"
    price: float
    position_size: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    confidence: float = 0.0
    timestamp: int = 0


#
# Execution Engine
#

class ExecutionEngine:
    """
    Execution engine for the trading system with GPU acceleration support
    
    This class handles the execution of trading signals by submitting orders
    to the Alpaca API.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the execution engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Get trading configuration
        trading_config = config.get("trading", {})
        orders_config = trading_config.get("orders", {})
        risk_config = trading_config.get("risk", {})
        
        # Set order defaults
        self.default_order_type = orders_config.get("default_order_type", "market")
        self.use_bracket_orders = orders_config.get("use_bracket_orders", True)
        self.time_in_force = orders_config.get("time_in_force", "day")
        
        # Set risk management defaults
        self.default_stop_loss_pct = risk_config.get("default_stop_loss_pct", 2.0)
        self.default_take_profit_pct = risk_config.get("default_take_profit_pct", 4.0)
        self.use_trailing_stop = risk_config.get("use_trailing_stop", False)
        self.trailing_stop_pct = risk_config.get("trailing_stop_pct", 1.0)
        
        # Initialize Alpaca API client
        self.alpaca_api = AlpacaRestAPI(config)
        
        # Thread management
        self.thread_id = None
        
        # GPU support
        self.use_gpu = config.get("performance", {}).get("use_gpu", False) and HAS_CUPY
        if self.use_gpu:
            try:
                # Initialize CuPy for GPU acceleration
                self.xp = cp
                logging.info("GPU acceleration enabled for execution engine")
            except Exception:
                logging.warning("CuPy initialization failed, falling back to NumPy")
                self.xp = np
                self.use_gpu = False
        else:
            self.xp = np
            logging.info("Using CPU for execution engine")
    
    def initialize(self) -> bool:
        """
        Initialize the execution engine
        
        Returns:
            True if initialization was successful, False otherwise
        """
        logging.info("Initializing execution engine")
        
        # Initialize Alpaca API client
        if not self.alpaca_api.initialize():
            logging.error("Failed to initialize Alpaca API client")
            return False
        
        logging.info("Execution engine initialized successfully")
        return True
    
    def execute_trades(self, validated_signals: List[Signal]) -> None:
        """
        Execute trades based on validated signals
        
        Args:
            validated_signals: List of validated trading signals
        """
        # Start timing
        start_time = time.time()
        
        # Skip if no signals
        if not validated_signals:
            return
        
        # Execute each signal
        for signal in validated_signals:
            try:
                # Create order
                if self.use_bracket_orders:
                    # Create bracket order
                    bracket_order = self.create_bracket_order(signal)
                    
                    # Submit bracket order
                    response = self.alpaca_api.submit_order(bracket_order.entry)
                    
                    # Handle response
                    self.handle_order_response(response)
                else:
                    # Create simple order
                    order = self.create_order(signal)
                    
                    # Submit order
                    response = self.alpaca_api.submit_order(order)
                    
                    # Handle response
                    self.handle_order_response(response)
            except Exception as e:
                logging.error(f"Error executing trade for {signal.symbol}: {str(e)}")
        
        # End timing
        end_time = time.time()
        duration = (end_time - start_time) * 1_000_000  # Convert to microseconds
        
        logging.info(f"Trade execution completed in {duration:.2f} µs for {len(validated_signals)} signals")
    
    def set_thread_affinity(self, core_id: int) -> None:
        """
        Set thread affinity for the execution thread
        
        Args:
            core_id: CPU core ID to pin the thread to
        """
        # Store thread ID
        self.thread_id = threading.get_ident()
        
        # Set affinity (platform-specific)
        try:
            if hasattr(threading, "current_thread"):
                current_thread = threading.current_thread()
                if hasattr(current_thread, "name"):
                    logging.info(f"Setting thread affinity for {current_thread.name} to core {core_id}")
            
            # Try to use platform-specific thread affinity setting
            try:
                import psutil
                p = psutil.Process()
                p.cpu_affinity([core_id])
                logging.info(f"Thread affinity set to core {core_id}")
            except (ImportError, AttributeError):
                logging.warning(f"Could not set thread affinity to core {core_id}")
        except Exception as e:
            logging.error(f"Error setting thread affinity: {str(e)}")
    
    def create_order(self, signal: Signal) -> Order:
        """
        Create an order from a signal
        
        Args:
            signal: Trading signal
            
        Returns:
            Order object
        """
        # Set basic order info
        order = Order(
            symbol=signal.symbol,
            side=OrderSide.BUY if signal.type == "BUY" else OrderSide.SELL,
            type=OrderType.MARKET if self.default_order_type == "market" else OrderType.LIMIT,
            order_class=OrderClass.SIMPLE,
            time_in_force=TimeInForce(self.time_in_force)
        )
        
        # Set quantity
        order.quantity = signal.position_size / signal.price
        
        # Set limit price if applicable
        if order.type == OrderType.LIMIT:
            # For buy orders, limit price is slightly above current price
            # For sell orders, limit price is slightly below current price
            slippage_factor = 0.001  # 0.1% slippage
            order.limit_price = signal.price * (1.0 + slippage_factor) if order.side == OrderSide.BUY else signal.price * (1.0 - slippage_factor)
        
        # Generate client order ID
        order.client_order_id = f"{signal.symbol}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        return order
    
    def create_bracket_order(self, signal: Signal) -> BracketOrder:
        """
        Create a bracket order from a signal
        
        Args:
            signal: Trading signal
            
        Returns:
            Bracket order object
        """
        # Create entry order
        entry = self.create_order(signal)
        entry.order_class = OrderClass.BRACKET
        
        # Set take profit price
        take_profit_pct = (signal.take_profit - signal.price) / signal.price * 100.0 if signal.take_profit > 0.0 else self.default_take_profit_pct
        
        entry.take_profit_price = signal.price * (1.0 + take_profit_pct / 100.0) if entry.side == OrderSide.BUY else signal.price * (1.0 - take_profit_pct / 100.0)
        
        # Set stop loss price
        stop_loss_pct = (signal.price - signal.stop_loss) / signal.price * 100.0 if signal.stop_loss > 0.0 else self.default_stop_loss_pct
        
        entry.stop_loss_price = signal.price * (1.0 - stop_loss_pct / 100.0) if entry.side == OrderSide.BUY else signal.price * (1.0 + stop_loss_pct / 100.0)
        
        # Set trailing stop if enabled
        if self.use_trailing_stop:
            entry.trail_percent = self.trailing_stop_pct
        
        return BracketOrder(entry=entry)
    
    def handle_order_response(self, response: OrderResponse) -> None:
        """
        Handle order response
        
        Args:
            response: Order response
        """
        # Log order response
        logging.info(f"Order {response.order_id} for {response.symbol} submitted with status {response.status}")
        
        # Handle different statuses
        if response.status == OrderStatus.NEW or response.status == OrderStatus.PARTIALLY_FILLED:
            # Order accepted, nothing to do
            pass
        elif response.status == OrderStatus.FILLED:
            # Order filled, update position
            logging.info(f"Order {response.order_id} filled at {response.filled_price}")
        elif response.status == OrderStatus.REJECTED:
            # Order rejected, log error
            logging.error(f"Order {response.order_id} rejected: {response.status_message}")
        elif response.status == OrderStatus.CANCELED or response.status == OrderStatus.EXPIRED:
            # Order canceled or expired, log warning
            logging.warning(f"Order {response.order_id} {response.status}: {response.status_message}")
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the execution engine
        
        This method should be called when the engine is no longer needed
        to ensure proper release of resources.
        """
        logging.info("Cleaning up execution engine resources")
        
        # Clean up Alpaca API client
        if hasattr(self, 'alpaca_api'):
            self.alpaca_api.cleanup()
        
        # Clean up GPU resources if used
        if self.use_gpu and cp is not None:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                logging.debug("GPU memory released")
            except Exception as e:
                logging.error(f"Error releasing GPU memory: {str(e)}")
