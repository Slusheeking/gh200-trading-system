"""
Enhanced Fast Exit Strategy Implementation

This module provides a high-performance, rule-based exit strategy system
with optimized algorithms for minimal latency. It directly integrates with
the trading system and market data for unified state management and low-latency
decision making.
"""

import time
import numpy as np
import logging
import threading
import os
import json
import uuid
import queue
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import backoff
import concurrent.futures

# Try to import cupy for GPU acceleration if available
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

# Import project modules with better error handling
try:
    from monitoring.log import setup_logger
    # Direct config loading from JSON instead of using get_config
    import json
except ImportError:
    # Fallback to standard logging if custom modules not available
    from logging import getLogger as setup_logger


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
    GTX = "gtx"  # Post-Only


class OrderStatus(str, Enum):
    """Order status enum"""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    PENDING_NEW = "pending_new"
    PENDING_CANCEL = "pending_cancel"


@dataclass
class Order:
    """Class representing an order with improved defaults and validation"""
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
    extended_hours: bool = False
    
    def __post_init__(self):
        """Validate order parameters after initialization"""
        # Generate client order ID if not provided
        if not self.client_order_id:
            self.client_order_id = f"{self.side.value}_{self.symbol}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        # Validate required fields
        if not self.symbol:
            raise ValueError("Symbol is required")
        
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        # Validate price fields for limit and stop orders
        if self.type == OrderType.LIMIT and self.limit_price <= 0:
            raise ValueError("Limit price must be positive for limit orders")
        
        if self.type == OrderType.STOP and self.stop_price <= 0:
            raise ValueError("Stop price must be positive for stop orders")
        
        if self.type == OrderType.STOP_LIMIT:
            if self.stop_price <= 0:
                raise ValueError("Stop price must be positive for stop-limit orders")
            if self.limit_price <= 0:
                raise ValueError("Limit price must be positive for stop-limit orders")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for API requests with improved field handling"""
        result = {
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.type.value,
            "qty": str(self.quantity),
            "time_in_force": self.time_in_force.value,
        }
        
        # Add optional fields with proper type conversion
        if self.limit_price > 0:
            result["limit_price"] = str(round(self.limit_price, 4))
        
        if self.stop_price > 0:
            result["stop_price"] = str(round(self.stop_price, 4))
        
        if self.client_order_id:
            result["client_order_id"] = self.client_order_id
        
        # Add extended hours flag if enabled
        if self.extended_hours:
            result["extended_hours"] = True
        
        # Handle bracket orders
        if self.order_class == OrderClass.BRACKET:
            result["order_class"] = "bracket"
            
            if self.take_profit_price > 0:
                result["take_profit"] = {
                    "limit_price": str(round(self.take_profit_price, 4))
                }
            
            if self.stop_loss_price > 0:
                stop_loss = {
                    "stop_price": str(round(self.stop_loss_price, 4))
                }
                
                # Add trailing stop if enabled
                if self.trail_percent > 0:
                    stop_loss["trail_percent"] = str(self.trail_percent)
                
                result["stop_loss"] = stop_loss
                
        # Handle OCO (One-Cancels-Other) orders
        elif self.order_class == OrderClass.OCO:
            result["order_class"] = "oco"
            if self.take_profit_price > 0:
                result["take_profit"] = {
                    "limit_price": str(round(self.take_profit_price, 4))
                }
            if self.stop_loss_price > 0:
                result["stop_loss"] = {
                    "stop_price": str(round(self.stop_loss_price, 4))
                }
        
        return result


@dataclass
class OrderResponse:
    """Class representing an order response with enhanced status handling"""
    order_id: str = ""
    client_order_id: str = ""
    symbol: str = ""
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    status_message: str = ""
    submission_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    order_type: OrderType = OrderType.MARKET
    side: OrderSide = OrderSide.BUY
    
    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> 'OrderResponse':
        """Create OrderResponse from API response with improved error handling"""
        try:
            # Extract order status with fallback to NEW
            status_str = response.get("status", "new").lower()
            try:
                status = OrderStatus(status_str)
            except ValueError:
                status = OrderStatus.NEW
                
            # Handle order type conversion
            order_type_str = response.get("type", "market").lower()
            try:
                order_type = OrderType(order_type_str)
            except ValueError:
                order_type = OrderType.MARKET
                
            # Handle order side conversion
            side_str = response.get("side", "buy").lower()
            try:
                side = OrderSide(side_str)
            except ValueError:
                side = OrderSide.BUY
                
            return cls(
                order_id=response.get("id", ""),
                client_order_id=response.get("client_order_id", ""),
                symbol=response.get("symbol", ""),
                status=status,
                filled_quantity=float(response.get("filled_qty", "0") or "0"),
                filled_price=float(response.get("filled_avg_price", "0") or "0"),
                status_message=response.get("status_message", ""),
                order_type=order_type,
                side=side,
                submission_time=time.time(),
                last_update_time=time.time()
            )
        except Exception as e:
            # Return a rejected order response if parsing fails
            return cls(
                client_order_id=response.get("client_order_id", ""),
                symbol=response.get("symbol", ""),
                status=OrderStatus.REJECTED,
                status_message=f"Failed to parse API response: {str(e)}",
                submission_time=time.time(),
                last_update_time=time.time()
            )


@dataclass
class Position:
    """Enhanced class representing a trading position with performance metrics"""
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    current_price: float = 0.0
    entry_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    highest_price: float = 0.0
    lowest_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived fields after creation"""
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price
        if self.lowest_price == 0.0:
            self.lowest_price = self.entry_price
        self._update_pnl()
    
    def update(self, current_price: float) -> None:
        """Update position with current price and recalculate metrics"""
        self.current_price = current_price
        self.last_update_time = time.time()
        
        # Update high/low watermarks
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price = min(self.lowest_price, current_price)
        
        # Update PnL calculations
        self._update_pnl()
    
    def _update_pnl(self) -> None:
        """Update PnL calculations based on current price"""
        if self.current_price <= 0 or self.entry_price <= 0:
            return
            
        if self.side == OrderSide.BUY:
            # Long position
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity
            self.unrealized_pnl_percent = ((self.current_price / self.entry_price) - 1.0) * 100.0
        else:
            # Short position
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.quantity
            self.unrealized_pnl_percent = ((self.entry_price / self.current_price) - 1.0) * 100.0
    
    def get_duration_seconds(self) -> float:
        """Get position duration in seconds"""
        return self.last_update_time - self.entry_time
    
    def get_max_drawdown_percent(self) -> float:
        """Calculate maximum drawdown percentage for this position"""
        if self.side == OrderSide.BUY:
            # For long positions, drawdown is from highest price
            if self.highest_price <= 0:
                return 0.0
            return ((self.highest_price - self.lowest_price) / self.highest_price) * 100.0
        else:
            # For short positions, drawdown is from lowest price
            if self.lowest_price <= 0:
                return 0.0
            return ((self.highest_price - self.lowest_price) / self.lowest_price) * 100.0


@dataclass
class Signal:
    """Enhanced class representing a trading signal with additional metadata"""
    symbol: str
    type: str  # "ENTRY", "EXIT", "ADJUST"
    direction: str  # "BUY" or "SELL"
    price: float
    position_size: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    confidence: float = 0.0
    timestamp: int = 0
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    indicators: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expiration: Optional[int] = None  # Signal expiration timestamp (0 = no expiration)
    
    def __post_init__(self):
        """Initialize default values and validate signal"""
        if self.timestamp == 0:
            self.timestamp = int(time.time() * 1_000_000_000)  # nanoseconds
            
        if self.indicators is None:
            self.indicators = {}
            
        if self.metadata is None:
            self.metadata = {}
            
        # Default exit reasons for EXIT signals
        if self.type == "EXIT" and "exit_reason" not in self.indicators:
            self.indicators["exit_reason"] = "unspecified"
    
    def is_valid(self) -> bool:
        """Check if signal is valid and not expired"""
        if not self.symbol or not self.type or not self.direction:
            return False
            
        if self.price <= 0 or self.position_size <= 0:
            return False
            
        # Check expiration if set
        if self.expiration and self.expiration < int(time.time() * 1_000_000_000):
            return False
            
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for serialization"""
        return {
            "symbol": self.symbol,
            "type": self.type,
            "direction": self.direction,
            "price": self.price,
            "position_size": self.position_size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "signal_id": self.signal_id,
            "indicators": self.indicators,
            "metadata": self.metadata,
            "expiration": self.expiration
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Create Signal from dictionary"""
        return cls(
            symbol=data.get("symbol", ""),
            type=data.get("type", ""),
            direction=data.get("direction", ""),
            price=data.get("price", 0.0),
            position_size=data.get("position_size", 0.0),
            stop_loss=data.get("stop_loss", 0.0),
            take_profit=data.get("take_profit", 0.0),
            confidence=data.get("confidence", 0.0),
            timestamp=data.get("timestamp", 0),
            signal_id=data.get("signal_id", str(uuid.uuid4())),
            indicators=data.get("indicators", {}),
            metadata=data.get("metadata", {}),
            expiration=data.get("expiration", None)
        )


class AsyncExitSignalProcessor:
    """Asynchronous processor for exit signals to minimize latency"""
    
    def __init__(self, 
                 exit_strategy: 'EnhancedExitStrategy',
                 max_queue_size: int = 100,
                 workers: int = 2):
        """
        Initialize the async processor
        
        Args:
            exit_strategy: Reference to the parent exit strategy
            max_queue_size: Maximum size of the signal queue
            workers: Number of worker threads
        """
        self.exit_strategy = exit_strategy
        self.logger = exit_strategy.logger
        self.signal_queue = queue.Queue(maxsize=max_queue_size)
        self.workers = workers
        self.running = False
        self.worker_threads = []
        
    def start(self):
        """Start the signal processor"""
        if self.running:
            return
            
        self.running = True
        
        # Start worker threads
        for i in range(self.workers):
            thread = threading.Thread(
                target=self._process_signals_worker,
                name=f"ExitSignalProcessor-{i}",
                daemon=True
            )
            thread.start()
            self.worker_threads.append(thread)
            
        self.logger.info(f"Started async exit signal processor with {self.workers} workers")
        
    def stop(self):
        """Stop the signal processor"""
        self.running = False
        
        # Add poison pills to the queue
        for _ in range(self.workers):
            try:
                self.signal_queue.put(None, block=False)
            except queue.Full:
                pass
        
        # Wait for threads to terminate
        for thread in self.worker_threads:
            thread.join(timeout=2.0)
            
        self.worker_threads = []
        self.logger.info("Stopped async exit signal processor")
        
    def queue_exit_signal(self, signal: Signal) -> bool:
        """
        Queue an exit signal for processing
        
        Args:
            signal: Exit signal to process
            
        Returns:
            True if signal was queued, False if queue was full
        """
        if not self.running:
            return False
            
        try:
            self.signal_queue.put(signal, block=False)
            return True
        except queue.Full:
            self.logger.warning(f"Exit signal queue full, dropping signal for {signal.symbol}")
            return False
            
    def _process_signals_worker(self):
        """Worker thread for processing exit signals"""
        while self.running:
            try:
                # Get signal from queue
                signal = self.signal_queue.get(block=True, timeout=1.0)
                
                # Check for poison pill
                if signal is None:
                    break
                    
                # Process signal
                try:
                    start_time = time.perf_counter()
                    self.exit_strategy._execute_exit_signal(signal)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    self.logger.debug(f"Processed exit signal for {signal.symbol} in {elapsed_ms:.2f}ms")
                except Exception as e:
                    self.logger.error(f"Error processing exit signal: {e}", exc_info=True)
                finally:
                    # Mark task as done
                    self.signal_queue.task_done()
                    
            except queue.Empty:
                # Timeout, check if still running
                continue
            except Exception as e:
                self.logger.error(f"Error in exit signal processor: {e}", exc_info=True)


class AlpacaAPI:
    """REST API client for Alpaca Markets with high-performance optimizations and connection pooling"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the REST client
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.running = False
        
        # Load environment variables
        load_dotenv()
        
        # Initialize API keys from environment variables first
        self.alpaca_api_key = os.getenv("ALPACA_API_KEY", "")
        self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.alpaca_endpoint = os.getenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2")
        
        # Get API keys from config as fallback
        alpaca_config = config.get("data_sources", {}).get("alpaca", {})
        if alpaca_config.get("enabled", False):
            # Only use config values if env vars aren't set
            if not self.alpaca_api_key:
                self.alpaca_api_key = alpaca_config.get("api_key", "")
            if not self.alpaca_secret_key:
                self.alpaca_secret_key = alpaca_config.get("secret_key", "")
            if not self.alpaca_endpoint:
                self.alpaca_endpoint = alpaca_config.get("endpoint", "https://paper-api.alpaca.markets/v2")
        
        # Thread management
        self.thread_affinity = -1
        self.mutex = threading.Lock()
        
        # Performance settings from config
        perf_config = config.get("performance", {})
        self.max_workers = perf_config.get("processor_threads", min(32, os.cpu_count() * 2))
        
        # Configure HTTP session with retries and timeouts
        self.session = requests.Session()
        
        # Configure retry strategy from system.yaml
        alpaca_config = config.get("data_sources", {}).get("alpaca", {})
        retry_config = alpaca_config.get("http", {}).get("retry", {})
        max_retries = retry_config.get("max_retries", 3)
        backoff_factor = retry_config.get("backoff_factor", 0.3)
        status_forcelist = retry_config.get("status_forcelist", [429, 500, 502, 503, 504])
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=["GET", "POST", "DELETE", "PATCH"]
        )
        
        # Create multiple adapters for connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers * 2
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Configure timeouts from system.yaml
        timeout_config = alpaca_config.get("http", {}).get("timeout", {})
        self.connect_timeout = timeout_config.get("connect", 3.0)
        self.read_timeout = timeout_config.get("read", 10.0)
        
        # Set up authentication headers
        self.headers = {
            "APCA-API-KEY-ID": self.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret_key,
            "Content-Type": "application/json"
        }
        
        # Create thread pool for parallel requests
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="AlpacaAPI-Worker"
        )
        
        # Cache for account and positions data
        self.account_cache = {}
        self.account_cache_time = 0
        self.account_cache_ttl = 5.0  # 5 seconds TTL
        
        self.positions_cache = {}
        self.positions_cache_time = 0
        self.positions_cache_ttl = 2.0  # 2 seconds TTL
        
        # Order cache to reduce redundant requests
        self.order_cache = {}
        self.order_cache_lock = threading.Lock()
        self.order_cache_cleanup_thread = None
        self.order_cache_ttl = 60.0  # 60 seconds TTL
        
        # Circuit breaker to prevent excessive API calls during outages
        self.circuit_breaker_engaged = False
        self.circuit_breaker_reset_time = 0
        self.consecutive_errors = 0
        self.error_threshold = 5
        self.circuit_breaker_cooldown = 60  # 60 seconds cooldown
        self.circuit_breaker_lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger("AlpacaAPI")
        
        self.logger.info(f"Configured high-performance HTTP client with max_workers={self.max_workers}, " +
                    f"max_retries={max_retries}, backoff_factor={backoff_factor}, " +
                    f"connect_timeout={self.connect_timeout}s, read_timeout={self.read_timeout}s")
    
    def initialize(self) -> bool:
        """Initialize the REST client and start background threads"""
        self.logger.info("Initializing Alpaca REST API client")
        
        # Check if API keys are available
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            self.logger.warning("No API keys configured for Alpaca REST client")
            return False
        
        # Test connection
        try:
            account = self.get_account()
            self.logger.info(f"Connected to Alpaca API. Account ID: {account.get('id')}, Status: {account.get('status')}")
            
            # Start background threads
            self._start_background_threads()
            
            self.running = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca API: {str(e)}")
            return False
    
    def _start_background_threads(self):
        """Start background maintenance threads"""
        # Start order cache cleanup thread
        self.order_cache_cleanup_thread = threading.Thread(
            target=self._order_cache_cleanup_worker,
            daemon=True,
            name="OrderCacheCleanup"
        )
        self.order_cache_cleanup_thread.start()
    
    def _order_cache_cleanup_worker(self):
        """Background worker to clean up expired order cache entries"""
        while True:
            try:
                # Sleep for a while
                time.sleep(30)
                
                # Clean up expired entries
                current_time = time.time()
                with self.order_cache_lock:
                    expired_order_ids = []
                    for order_id, (order_data, cache_time) in self.order_cache.items():
                        if current_time - cache_time > self.order_cache_ttl:
                            expired_order_ids.append(order_id)
                    
                    # Remove expired entries
                    for order_id in expired_order_ids:
                        del self.order_cache[order_id]
                        
                    if expired_order_ids:
                        self.logger.debug(f"Cleaned up {len(expired_order_ids)} expired order cache entries")
            
            except Exception as e:
                self.logger.error(f"Error in order cache cleanup: {str(e)}")
    
    def _check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker is engaged
        
        Returns:
            True if circuit breaker is engaged, False otherwise
        """
        with self.circuit_breaker_lock:
            if self.circuit_breaker_engaged:
                current_time = time.time()
                if current_time > self.circuit_breaker_reset_time:
                    # Reset circuit breaker
                    self.circuit_breaker_engaged = False
                    self.consecutive_errors = 0
                    self.logger.info("Circuit breaker reset")
                    return False
                else:
                    # Circuit breaker still engaged
                    return True
            
            return False
    
    def _record_error(self):
        """Record an API error and engage circuit breaker if threshold is reached"""
        with self.circuit_breaker_lock:
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.error_threshold:
                self.circuit_breaker_engaged = True
                self.circuit_breaker_reset_time = time.time() + self.circuit_breaker_cooldown
                self.logger.warning(f"Circuit breaker engaged for {self.circuit_breaker_cooldown} seconds after {self.consecutive_errors} consecutive errors")
    
    def _record_success(self):
        """Record a successful API call and reset consecutive error counter"""
        with self.circuit_breaker_lock:
            self.consecutive_errors = 0
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, json.JSONDecodeError),
        max_tries=3,
        giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code < 500,
        factor=2,
        jitter=backoff.full_jitter
    )
    def get_account(self) -> Dict[str, Any]:
        """
        Get account information with caching
        
        Returns:
            Account information
        """
        # Check circuit breaker
        if self._check_circuit_breaker():
            self.logger.warning("Circuit breaker engaged. Returning cached account or empty dict")
            return self.account_cache or {}
        
        # Check cache
        current_time = time.time()
        if self.account_cache and current_time - self.account_cache_time < self.account_cache_ttl:
            return self.account_cache.copy()
        
        url = f"{self.alpaca_endpoint}/account"
        
        try:
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            response.raise_for_status()
            
            # Update cache
            self.account_cache = response.json()
            self.account_cache_time = current_time
            
            # Record success
            self._record_success()
            
            return self.account_cache.copy()
        except requests.exceptions.RequestException as e:
            # Record error
            self._record_error()
            
            self.logger.error(f"Error getting account information: {str(e)}")
            
            # Return cache if available, otherwise re-raise
            if self.account_cache:
                return self.account_cache.copy()
            raise
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, json.JSONDecodeError),
        max_tries=3,
        giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code < 500,
        factor=2,
        jitter=backoff.full_jitter
    )
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all positions with caching
        
        Returns:
            List of positions
        """
        # Check circuit breaker
        if self._check_circuit_breaker():
            self.logger.warning("Circuit breaker engaged. Returning cached positions or empty list")
            return list(self.positions_cache.values()) if self.positions_cache else []
        
        # Check cache
        current_time = time.time()
        if self.positions_cache and current_time - self.positions_cache_time < self.positions_cache_ttl:
            return list(self.positions_cache.values())
        
        url = f"{self.alpaca_endpoint}/positions"
        
        try:
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            response.raise_for_status()
            
            # Update cache
            positions = response.json()
            
            # Transform to dictionary for faster lookups by symbol
            positions_dict = {}
            for position in positions:
                symbol = position.get("symbol")
                if symbol:
                    positions_dict[symbol] = position
            
            self.positions_cache = positions_dict
            self.positions_cache_time = current_time
            
            # Record success
            self._record_success()
            
            return positions
        except requests.exceptions.RequestException as e:
            # Record error
            self._record_error()
            
            self.logger.error(f"Error getting positions: {str(e)}")
            
            # Return cache if available, otherwise re-raise
            if self.positions_cache:
                return list(self.positions_cache.values())
            raise
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position for a specific symbol with caching
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position information or None if not found
        """
        # Check circuit breaker
        if self._check_circuit_breaker():
            self.logger.warning(f"Circuit breaker engaged. Returning cached position for {symbol} or None")
            return self.positions_cache.get(symbol)
        
        # Check cache first
        current_time = time.time()
        if self.positions_cache and current_time - self.positions_cache_time < self.positions_cache_ttl:
            return self.positions_cache.get(symbol)
        
        # If not in cache, try direct API call first
        url = f"{self.alpaca_endpoint}/positions/{symbol}"
        
        try:
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            response.raise_for_status()
            
            # Get position data
            position = response.json()
            
            # Update cache
            self.positions_cache[symbol] = position
            self.positions_cache_time = current_time
            
            # Record success
            self._record_success()
            
            return position
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Position not found, update cache
                self.positions_cache[symbol] = None
                return None
            
            # For other errors, try to get all positions as fallback
            try:
                self.get_positions()
                return self.positions_cache.get(symbol)
            except Exception:
                # Record error
                self._record_error()
                
                self.logger.error(f"Error getting position for {symbol}: {str(e)}")
                return None
        except Exception as e:
            # Record error
            self._record_error()
            
            self.logger.error(f"Error getting position for {symbol}: {str(e)}")
            return None
    
    def _execute_order_future(self, order: Order) -> Tuple[Order, OrderResponse]:
        """
        Submit an order and return the order and response
        (Used as worker function for map with thread pool)
        
        Args:
            order: Order to submit
            
        Returns:
            Tuple of (order, response)
        """
        response = self.submit_order(order)
        return (order, response)
    
    def submit_orders_batch(self, orders: List[Order]) -> Dict[str, OrderResponse]:
        """
        Submit multiple orders in parallel
        
        Args:
            orders: List of orders to submit
            
        Returns:
            Dictionary mapping client_order_id to order response
        """
        if not orders:
            return {}
            
        # Execute orders in parallel using thread pool
        futures = []
        for order in orders:
            future = self.thread_pool.submit(self._execute_order_future, order)
            futures.append(future)
        
        # Collect results
        results = {}
        for future in concurrent.futures.as_completed(futures):
            try:
                order, response = future.result()
                results[order.client_order_id] = response
            except Exception as e:
                self.logger.error(f"Error in batch order submission: {str(e)}")
                
        return results
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, json.JSONDecodeError),
        max_tries=3,
        giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code < 500,
        factor=2,
        jitter=backoff.full_jitter
    )
    def submit_order(self, order: Order) -> OrderResponse:
        """
        Submit an order
        
        Args:
            order: Order to submit
            
        Returns:
            Order response
        """
        # Check circuit breaker
        if self._check_circuit_breaker():
            self.logger.warning("Circuit breaker engaged. Returning rejected order response")
            return OrderResponse(
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                status=OrderStatus.REJECTED,
                status_message="Circuit breaker engaged due to API issues"
            )
        
        url = f"{self.alpaca_endpoint}/orders"
        
        # Convert order to API payload
        payload = order.to_dict()
        
        try:
            response = self.session.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            response.raise_for_status()
            order_data = response.json()
            
            # Create order response
            order_response = OrderResponse.from_api_response(order_data)
            
            # Cache the order
            with self.order_cache_lock:
                self.order_cache[order_response.order_id] = (order_data, time.time())
            
            # Record success
            self._record_success()
            
            self.logger.info(f"Order submitted: {order_response.order_id} for {order_response.symbol}")
            return order_response
        except requests.exceptions.RequestException as e:
            # Record error
            self._record_error()
            
            self.logger.error(f"Error submitting order: {str(e)}")
            
            # Create error response
            error_message = str(e)
            if hasattr(e, "response") and e.response:
                try:
                    error_data = e.response.json()
                    error_message = error_data.get("message", str(e))
                except json.JSONDecodeError:
                    pass
            
            order_response = OrderResponse(
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                status=OrderStatus.REJECTED,
                status_message=error_message
            )
            
            return order_response
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, json.JSONDecodeError),
        max_tries=3,
        giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code < 500,
        factor=2,
        jitter=backoff.full_jitter
    )
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        # Check circuit breaker
        if self._check_circuit_breaker():
            self.logger.warning(f"Circuit breaker engaged. Unable to cancel order {order_id}")
            return False
        
        url = f"{self.alpaca_endpoint}/orders/{order_id}"
        
        try:
            response = self.session.delete(
                url,
                headers=self.headers,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            response.raise_for_status()
            
            # Remove from cache
            with self.order_cache_lock:
                if order_id in self.order_cache:
                    del self.order_cache[order_id]
            
            # Record success
            self._record_success()
            
            self.logger.info(f"Order {order_id} canceled")
            return True
        except requests.exceptions.RequestException as e:
            # Record error
            self._record_error()
            
            self.logger.error(f"Error canceling order {order_id}: {str(e)}")
            return False
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, json.JSONDecodeError),
        max_tries=3,
        giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code < 500,
        factor=2,
        jitter=backoff.full_jitter
    )
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order details with caching
        
        Args:
            order_id: Order ID to get
            
        Returns:
            Order details or None if not found
        """
        # Check circuit breaker
        if self._check_circuit_breaker():
            self.logger.warning(f"Circuit breaker engaged. Returning cached order {order_id} or None")
            with self.order_cache_lock:
                cached = self.order_cache.get(order_id)
                return cached[0] if cached else None
        
        # Check cache
        with self.order_cache_lock:
            cached = self.order_cache.get(order_id)
            if cached:
                order_data, cache_time = cached
                # Use cache if relatively fresh
                if time.time() - cache_time < 5.0:  # 5 second TTL for orders
                    return order_data
        
        url = f"{self.alpaca_endpoint}/orders/{order_id}"
        
        try:
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            response.raise_for_status()
            order_data = response.json()
            
            # Update cache
            with self.order_cache_lock:
                self.order_cache[order_id] = (order_data, time.time())
            
            # Record success
            self._record_success()
            
            return order_data
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Order not found, remove from cache
                with self.order_cache_lock:
                    if order_id in self.order_cache:
                        del self.order_cache[order_id]
                return None
            
            # Record error for other HTTP errors
            self._record_error()
            
            self.logger.error(f"Error getting order {order_id}: {str(e)}")
            
            # Return cached version if available
            with self.order_cache_lock:
                cached = self.order_cache.get(order_id)
                return cached[0] if cached else None
        except Exception as e:
            # Record error
            self._record_error()
            
            self.logger.error(f"Error getting order {order_id}: {str(e)}")
            
            # Return cached version if available
            with self.order_cache_lock:
                cached = self.order_cache.get(order_id)
                return cached[0] if cached else None
    
    def cleanup(self):
        """
        Clean up resources used by the REST client
        """
        self.running = False
        self.logger.info("Cleaning up AlpacaAPI resources")
        
        # Close HTTP session
        if hasattr(self, 'session'):
            self.session.close()
            self.logger.debug("HTTP session closed")
        
        # Shutdown thread pool
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
            self.logger.debug("Thread pool shut down")
        
        # Force garbage collection of any large buffers
        import gc
        gc.collect()
        self.logger.debug("Memory buffers released")


class EnhancedExitStrategy:
    """
    High-performance, rule-based exit strategy system optimized for minimal latency.
    Provides sophisticated exit logic with multiple condition types and real-time
    position tracking. Enhanced with bracket order support and volatility-adjusted parameters.
    """
    
    def __init__(self, config=None):
        """
        Initialize the exit strategy with parameters from config.
        
        Args:
            config: Configuration dictionary (optional)
        """
        # Set up logger
        self.logger = setup_logger("exit_strategy")
        self.logger.info("Creating Enhanced Exit Strategy with Bracket Order Support")
        
        # Load configuration
        if config is None:
            # Load exit strategy specific settings from dedicated config file
            exit_strategy_config_path = "/home/ubuntu/gh200-trading-system/config/fast_exit_strategy_settings.json"
            try:
                with open(exit_strategy_config_path, "r") as f:
                    config = json.load(f)
                self.logger.info(f"Loaded exit strategy settings from {exit_strategy_config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load exit strategy settings from {exit_strategy_config_path}: {e}")
                # No fallback, raise error if config can't be loaded
                raise RuntimeError(f"Could not load required configuration from {exit_strategy_config_path}")
        
        self.config = config
        
        # Extract exit parameters from config
        trading_config = config.get("trading", {})
        exit_config = trading_config.get("exit", {})
        
        # Exit thresholds
        self.profit_target_pct = exit_config.get("profit_target_pct", 2.0)
        self.profit_target_scale = exit_config.get("profit_target_scale", 0.5)  # Scale targets based on volatility
        self.stop_loss_pct = exit_config.get("stop_loss_pct", 1.0)
        self.stop_loss_scale = exit_config.get("stop_loss_scale", 0.5)  # Scale stop losses based on volatility
        self.trailing_stop_pct = exit_config.get("trailing_stop_pct", 0.5)
        self.trailing_stop_activation_pct = exit_config.get("trailing_stop_activation_pct", 0.8)  # % of profit target
        self.max_holding_time_minutes = exit_config.get("max_holding_time_minutes", 240)
        
        # Time-based exit settings
        self.partial_exit_time_pct = exit_config.get("partial_exit_time_pct", 50.0)  # Take partial profits at 50% time
        self.partial_exit_size_pct = exit_config.get("partial_exit_size_pct", 50.0)  # Take 50% of position size
        self.enable_time_based_exits = exit_config.get("enable_time_based_exits", True)
        
        # Technical indicator thresholds
        self.rsi_overbought = exit_config.get("rsi_overbought", 70)
        self.rsi_oversold = exit_config.get("rsi_oversold", 30)
        self.volume_spike_threshold = exit_config.get("volume_spike_threshold", 2.0)
        
        # Volatility-based adjustments
        self.volatility_multiplier = exit_config.get("volatility_multiplier", 1.5)
        self.bb_exit_threshold = exit_config.get("bb_exit_threshold", 0.9)  # Exit when price reaches 90% of BB width
        
        # Advanced exit rules
        self.enable_volume_filter = exit_config.get("enable_volume_filter", True)
        self.volume_filter_threshold = exit_config.get("volume_filter_threshold", 0.7)  # Minimum volume compared to avg
        self.enable_price_rejection = exit_config.get("enable_price_rejection", True)
        self.reversal_candle_threshold = exit_config.get("reversal_candle_threshold", 1.5)  # Reversal candle size
        
        # Risk management settings
        self.max_drawdown_pct = exit_config.get("max_drawdown_pct", 3.0)  # Maximum allowed drawdown
        self.daily_loss_limit_pct = exit_config.get("daily_loss_limit_pct", 5.0)  # Daily loss threshold
        self.min_profit_target_multiplier = exit_config.get("min_profit_target_multiplier", 1.5)  # Min risk/reward
        self.enable_correlation_exits = exit_config.get("enable_correlation_exits", True)  # Exit correlated positions
        
        # Performance optimization
        self.use_vectorized_calculations = True
        self.check_interval_seconds = exit_config.get("check_exit_interval_seconds", 10)
        self.max_concurrent_exits = exit_config.get("max_concurrent_exits", 10)
        self.partial_exit_enabled = exit_config.get("partial_exit_enabled", True)
        
        # Position tracking
        self.active_positions = {}  # Track active positions: symbol -> Position
        self.position_history = {}  # Track position history: symbol -> List[Position]
        self.position_lock = threading.Lock()
        
        # Exit signal tracking
        self.exit_signals = {}  # Track exit signals: signal_id -> Signal
        self.exit_signal_lock = threading.Lock()
        
        # Daily performance tracking
        self.daily_stats = {
            "date": time.strftime("%Y-%m-%d"),
            "realized_pnl": 0.0,
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "total_fees": 0.0
        }
        self.daily_stats_lock = threading.Lock()
        
        # Thread management
        self.main_thread = None
        self.running = False
        self.cpu_affinity = None
        
        # Initialize Alpaca API client
        self.alpaca_api = AlpacaAPI(self.config)
        
        # Initialize async exit signal processor
        self.signal_processor = AsyncExitSignalProcessor(self)
        
        # Ticker data cache for efficient access
        self.ticker_data_cache = {}
        self.ticker_data_cache_lock = threading.Lock()
        
        # Advanced optimization
        self.gpu_enabled = HAS_CUPY and exit_config.get("use_gpu", False)
        
        # Feature importance for exit reasons (learned from previous exits)
        self.exit_reason_importance = {
            "profit_target": 0.25,
            "stop_loss": 0.20,
            "trailing_stop": 0.15,
            "time_exit": 0.10,
            "technical_indicators": 0.15,
            "volatility_expansion": 0.05,
            "volume_spike": 0.05,
            "correlation_exit": 0.05,
        }
        
        # Status indicators
        self.status = "initialized"
        self.last_check_time = 0
        self.exit_count = 0
        
        # Order tracking for bracket orders
        self.order_lock = threading.Lock()
        self.tracked_orders = {}  # order_id -> order_info
        
        # Extract bracket configuration
        bracket_config = trading_config.get("bracket", {})
        
        # Base parameters for bracket orders
        self.profit_target_base_pct = bracket_config.get("profit_target_pct", 2.0)
        self.stop_loss_base_pct = bracket_config.get("stop_loss_pct", 1.0)
        
        # Volatility adjustment factors
        self.use_volatility_adjustment = bracket_config.get("use_volatility_adjustment", True)
        self.volatility_scaling_factor = bracket_config.get("volatility_scaling_factor", 0.5)
        self.min_profit_target_pct = bracket_config.get("min_profit_target_pct", 0.5)
        self.max_profit_target_pct = bracket_config.get("max_profit_target_pct", 5.0)
        self.min_stop_loss_pct = bracket_config.get("min_stop_loss_pct", 0.3)
        self.max_stop_loss_pct = bracket_config.get("max_stop_loss_pct", 3.0)
        
        # Risk/reward requirements
        self.min_risk_reward_ratio = bracket_config.get("min_risk_reward_ratio", 1.5)
        
        self.logger.info(f"Enhanced Exit Strategy initialized with profit_target={self.profit_target_pct}%, "
                        f"stop_loss={self.stop_loss_pct}%, trailing_stop={self.trailing_stop_pct}%, "
                        f"bracket orders enabled with volatility adjustment={self.use_volatility_adjustment}")
    
    def set_thread_affinity(self, core_id: int) -> None:
        """
        Set thread affinity for the exit strategy thread for improved performance.
        
        Args:
            core_id: CPU core ID to pin the thread to
        """
        self.cpu_affinity = core_id
        self.logger.info(f"Thread affinity set to core {core_id}")
    
    def initialize(self) -> bool:
        """
        Initialize the enhanced exit strategy and start processing.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        self.logger.info("Initializing Enhanced Exit Strategy")
        
        # Initialize Alpaca API client
        if not self.alpaca_api.initialize():
            self.logger.error("Failed to initialize Alpaca API client")
            return False
        
        # Start async signal processor
        self.signal_processor.start()
        
        # Initialize active positions from current portfolio
        try:
            self._load_active_positions()
        except Exception as e:
            self.logger.error(f"Error loading active positions: {str(e)}")
        
        # Start exit strategy thread if not already running
        if not self.running:
            self.running = True
            self.main_thread = threading.Thread(
                target=self._run_exit_strategy_loop,
                daemon=True,
                name="ExitStrategyMain"
            )
            self.main_thread.start()
        
        self.status = "running"
        self.logger.info("Enhanced Exit Strategy initialized successfully")
        return True
    
    def _load_active_positions(self) -> None:
        """
        Load active positions from broker to initialize state.
        """
        self.logger.info("Loading active positions from broker")
        
        try:
            # Get current positions from Alpaca
            positions = self.alpaca_api.get_positions()
            
            # Process positions into our tracking system
            with self.position_lock:
                for pos_data in positions:
                    symbol = pos_data.get("symbol", "")
                    if not symbol:
                        continue
                    
                    # Determine position side
                    qty = float(pos_data.get("qty", "0"))
                    side = OrderSide.BUY if qty > 0 else OrderSide.SELL
                    
                    # Create position object
                    position = Position(
                        symbol=symbol,
                        side=side,
                        entry_price=float(pos_data.get("avg_entry_price", "0")),
                        quantity=abs(qty),
                        current_price=float(pos_data.get("current_price", "0")),
                        # Use current timestamp as approximate entry time for existing positions
                        entry_time=time.time()
                    )
                    
                    # Store position
                    self.active_positions[symbol] = position
            
            self.logger.info(f"Loaded {len(positions)} active positions from broker")
        except Exception as e:
            self.logger.error(f"Error loading active positions: {str(e)}")
            raise
    
    def _run_exit_strategy_loop(self) -> None:
        """
        Main loop for the exit strategy. Periodically checks for exit conditions.
        """
        self.logger.info("Starting exit strategy monitoring loop")
        
        # Set CPU affinity if specified
        if self.cpu_affinity is not None:
            try:
                import psutil
                p = psutil.Process()
                p.cpu_affinity([self.cpu_affinity])
                self.logger.info(f"Set thread affinity to core {self.cpu_affinity}")
            except (ImportError, AttributeError, ValueError) as e:
                self.logger.warning(f"Could not set thread affinity: {str(e)}")
        
        while self.running:
            try:
                start_time = time.perf_counter()
                
                # Check active positions for exit conditions
                if self.active_positions:
                    self._check_all_positions()
                
                # Record last check time
                self.last_check_time = time.time()
                
                # Calculate processing time
                processing_time = (time.perf_counter() - start_time) * 1000  # ms
                
                # Sleep for the remaining interval
                sleep_time = max(0.01, self.check_interval_seconds - (processing_time / 1000))
                time.sleep(sleep_time)
                
                # Log periodic status if needed
                if self.exit_count > 0 and self.exit_count % 10 == 0:
                    self.logger.info(f"Exit strategy processed {self.exit_count} exits - "
                                    f"monitoring {len(self.active_positions)} positions")
                    
            except Exception as e:
                self.logger.error(f"Error in exit strategy loop: {str(e)}", exc_info=True)
                time.sleep(1)  # Sleep briefly to avoid tight error loops
    
    def _check_all_positions(self) -> None:
        """
        Check all active positions for exit conditions.
        """
        # Get symbols to check
        with self.position_lock:
            symbols = list(self.active_positions.keys())
        
        if not symbols:
            return
        
        # Get latest market data for all symbols
        try:
            market_data = self._get_market_data(symbols)
            if not market_data:
                self.logger.warning("Failed to get market data for exit strategy")
                return
            
            # Process each position with current market data
            for symbol in symbols:
                try:
                    # Update position with latest data
                    self._update_position(symbol, market_data)
                    
                    # Check exit conditions
                    exit_reason, exit_confidence = self._check_exit_conditions(symbol, market_data)
                    
                    # Generate exit signal if needed
                    if exit_reason:
                        self._generate_exit_signal(symbol, exit_reason, exit_confidence, market_data)
                except Exception as e:
                    self.logger.error(f"Error checking position {symbol}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error processing positions: {str(e)}")
    
    def _get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get current market data for the specified symbols.
        This method should integrate with your market data provider.
        
        Args:
            symbols: List of symbols to get data for
            
        Returns:
            Dictionary containing market data
        """
        # This is a placeholder method - in a real implementation, you would
        # integrate with your market data provider to get current prices, indicators, etc.
        # The method should return a dictionary with symbol -> data mapping
        
        # In production, replace this with actual market data retrieval
        # For now, simulate with random price updates
        market_data = {
            "timestamp": int(time.time() * 1_000_000_000),  # nanoseconds
            "symbol_data": {}
        }
        
        # Update price for each symbol
        for symbol in symbols:
            # Try to get from cache first
            cached_data = self.ticker_data_cache.get(symbol, {})
            
            # Simulate new data with small random changes from cached values
            last_price = cached_data.get("last_price", 100.0)
            price_change = last_price * np.random.normal(0, 0.001)  # Small random change
            new_price = max(0.01, last_price + price_change)
            
            # Create symbol data
            symbol_data = {
                "symbol": symbol,
                "last_price": new_price,
                "bid_price": new_price * 0.999,
                "ask_price": new_price * 1.001,
                "bid_ask_spread": new_price * 0.002,
                "high_price": cached_data.get("high_price", new_price * 1.01),
                "low_price": cached_data.get("low_price", new_price * 0.99),
                "volume": cached_data.get("volume", 10000) * (1 + np.random.normal(0, 0.05)),
                "vwap": new_price * (1 + np.random.normal(0, 0.001)),
                
                # Technical indicators
                "rsi_14": cached_data.get("rsi_14", 50.0) + np.random.normal(0, 1.0),
                "macd": cached_data.get("macd", 0.0) + np.random.normal(0, 0.1),
                "macd_signal": cached_data.get("macd_signal", 0.0) + np.random.normal(0, 0.05),
                "bb_upper": new_price * 1.02,
                "bb_middle": new_price,
                "bb_lower": new_price * 0.98,
                
                # Additional metrics
                "volume_acceleration": np.random.normal(0, 0.5),
                "price_change_5m": np.random.normal(0, 0.2),
                "momentum_1m": np.random.normal(0, 0.1),
                
                # Volatility metrics 
                "atr": cached_data.get("atr", new_price * 0.01),
                "volatility": cached_data.get("volatility", 0.2)
            }
            
            # Update market data
            market_data["symbol_data"][symbol] = symbol_data
            
            # Update cache
            with self.ticker_data_cache_lock:
                self.ticker_data_cache[symbol] = symbol_data
        
        return market_data
    
    def _update_position(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """
        Update position tracking with latest market data.
        
        Args:
            symbol: Symbol to update
            market_data: Current market data
        """
        if symbol not in market_data.get("symbol_data", {}):
            return
            
        # Get position
        with self.position_lock:
            position = self.active_positions.get(symbol)
            if not position:
                return
            
            # Update with current price
            symbol_data = market_data["symbol_data"][symbol]
            current_price = symbol_data.get("last_price", 0.0)
            
            position.update(current_price)
    
    def _check_exit_conditions(self, symbol: str, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Check all exit conditions for a position.
        
        Args:
            symbol: Symbol to check
            market_data: Current market data
            
        Returns:
            Tuple of (exit_reason, confidence) where exit_reason is empty if no exit,
            and confidence is between 0.0 and 1.0
        """
        if symbol not in market_data.get("symbol_data", {}):
            return "", 0.0
            
        # Get position and symbol data
        with self.position_lock:
            position = self.active_positions.get(symbol)
            if not position:
                return "", 0.0
        
        symbol_data = market_data["symbol_data"][symbol]
        current_timestamp = market_data.get("timestamp", 0)
        
        # Enhanced exit conditions with confidence levels
        exit_conditions = []
        
        # 1. Profit target exit
        profit_target_exit, pt_confidence = self._check_profit_target_exit(position, symbol_data)
        if profit_target_exit:
            exit_conditions.append(("profit_target", pt_confidence))
            
        # 2. Stop loss exit
        stop_loss_exit, sl_confidence = self._check_stop_loss_exit(position, symbol_data)
        if stop_loss_exit:
            exit_conditions.append(("stop_loss", sl_confidence))
            
        # 3. Trailing stop exit
        trailing_stop_exit, ts_confidence = self._check_trailing_stop_exit(position, symbol_data)
        if trailing_stop_exit:
            exit_conditions.append(("trailing_stop", ts_confidence))
            
        # 4. Time-based exit
        time_exit, time_confidence = self._check_time_based_exit(position, current_timestamp)
        if time_exit:
            exit_conditions.append(("time_exit", time_confidence))
            
        # 5. Technical indicator exit
        tech_exit, tech_confidence = self._check_technical_indicator_exit(position, symbol_data)
        if tech_exit:
            exit_conditions.append(("technical_indicators", tech_confidence))
            
        # 6. Volatility expansion exit
        vol_exit, vol_confidence = self._check_volatility_exit(position, symbol_data)
        if vol_exit:
            exit_conditions.append(("volatility_expansion", vol_confidence))
            
        # 7. Volume profile exit
        volume_exit, vol_confidence = self._check_volume_exit(position, symbol_data)
        if volume_exit:
            exit_conditions.append(("volume_spike", vol_confidence))
            
        # 8. Risk management exit
        risk_exit, risk_confidence = self._check_risk_management_exit(position, symbol_data)
        if risk_exit:
            exit_conditions.append(("risk_management", risk_confidence))
            
        # Determine best exit reason based on conditions and importances
        if exit_conditions:
            # Sort by confidence and importance
            exit_conditions.sort(key=lambda x: x[1] * self.exit_reason_importance.get(x[0], 0.1), reverse=True)
            return exit_conditions[0]
            
        # No exit condition met
        return "", 0.0
    
    def _check_profit_target_exit(self, position: Position, symbol_data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if position should be exited based on profit target.
        
        Args:
            position: Trading position
            symbol_data: Current market data for the symbol
            
        Returns:
            Tuple of (should_exit, confidence)
        """
        # Calculate adjusted profit target based on volatility
        volatility = symbol_data.get("volatility", 0.0)
        
        # Adjust profit target based on volatility if available
        if volatility > 0:
            adjusted_profit_target = self.profit_target_pct * (1.0 + (volatility - 0.2) * self.profit_target_scale)
        else:
            adjusted_profit_target = self.profit_target_pct
        
        # Check if profit threshold exceeded
        if position.unrealized_pnl_percent >= adjusted_profit_target:
            # Calculate confidence based on how far past the target we are
            excess = position.unrealized_pnl_percent - adjusted_profit_target
            confidence = min(0.95, 0.85 + (excess / adjusted_profit_target) * 0.1)
            return True, confidence
            
        return False, 0.0
    
    def _check_stop_loss_exit(self, position: Position, symbol_data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if position should be exited based on stop loss.
        
        Args:
            position: Trading position
            symbol_data: Current market data for the symbol
            
        Returns:
            Tuple of (should_exit, confidence)
        """
        # Calculate adjusted stop loss based on volatility
        volatility = symbol_data.get("volatility", 0.0)
        
        # Adjust stop loss based on volatility if available
        if volatility > 0:
            adjusted_stop_loss = self.stop_loss_pct * (1.0 + (volatility - 0.2) * self.stop_loss_scale)
        else:
            adjusted_stop_loss = self.stop_loss_pct
        
        # Check if loss threshold exceeded
        if position.unrealized_pnl_percent <= -adjusted_stop_loss:
            # Calculate confidence based on how far past the stop loss we are
            excess = abs(position.unrealized_pnl_percent) - adjusted_stop_loss
            confidence = min(0.98, 0.90 + (excess / adjusted_stop_loss) * 0.08)
            return True, confidence
            
        return False, 0.0
    
    def _check_trailing_stop_exit(self, position: Position, symbol_data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if position should be exited based on trailing stop.
        
        Args:
            position: Trading position
            symbol_data: Current market data for the symbol
            
        Returns:
            Tuple of (should_exit, confidence)
        """
        # Only apply trailing stop if position has reached activation threshold
        activation_threshold = self.profit_target_pct * self.trailing_stop_activation_pct / 100.0
        
        # Check if position has ever reached the activation threshold
        if position.side == OrderSide.BUY:
            # For long positions
            highest_pnl_pct = ((position.highest_price / position.entry_price) - 1.0) * 100.0
            if highest_pnl_pct < activation_threshold:
                return False, 0.0
                
            # Calculate trailing stop level
            trail_level = position.highest_price * (1.0 - self.trailing_stop_pct / 100.0)
            
            # Check if current price has fallen below trailing stop
            if position.current_price <= trail_level:
                # Calculate confidence - higher if we've dropped significantly below trail level
                breach_pct = (trail_level - position.current_price) / trail_level * 100.0
                confidence = min(0.95, 0.85 + breach_pct * 0.1)
                return True, confidence
                
        elif position.side == OrderSide.SELL:
            # For short positions
            lowest_pnl_pct = ((position.entry_price / position.lowest_price) - 1.0) * 100.0
            if lowest_pnl_pct < activation_threshold:
                return False, 0.0
                
            # Calculate trailing stop level
            trail_level = position.lowest_price * (1.0 + self.trailing_stop_pct / 100.0)
            
            # Check if current price has risen above trailing stop
            if position.current_price >= trail_level:
                # Calculate confidence - higher if we've moved significantly above trail level
                breach_pct = (position.current_price - trail_level) / trail_level * 100.0
                confidence = min(0.95, 0.85 + breach_pct * 0.1)
                return True, confidence
                
        return False, 0.0
    
    def _check_time_based_exit(self, position: Position, current_timestamp: int) -> Tuple[bool, float]:
        """
        Check if position should be exited based on time.
        
        Args:
            position: Trading position
            current_timestamp: Current timestamp in nanoseconds
            
        Returns:
            Tuple of (should_exit, confidence)
        """
        if not self.enable_time_based_exits:
            return False, 0.0
            
        # Convert timestamps to compatible format
        position_duration_seconds = position.get_duration_seconds()
        max_duration_seconds = self.max_holding_time_minutes * 60
        
        # Check if position duration exceeds max holding time
        if position_duration_seconds >= max_duration_seconds:
            # Calculate confidence based on how far past the time limit we are
            overtime_ratio = position_duration_seconds / max_duration_seconds - 1.0
            confidence = min(0.90, 0.80 + overtime_ratio * 0.1)
            return True, confidence
            
        # Check for partial time-based exit
        if self.partial_exit_enabled:
            partial_time_threshold = max_duration_seconds * (self.partial_exit_time_pct / 100.0)
            
            # Check if position has reached partial exit time and is profitable
            if (position_duration_seconds >= partial_time_threshold and 
                position.unrealized_pnl_percent > 0):
                overtime_ratio = position_duration_seconds / partial_time_threshold - 1.0
                confidence = min(0.75, 0.65 + overtime_ratio * 0.1)
                return True, confidence
                
        return False, 0.0
    
    def _check_technical_indicator_exit(self, position: Position, symbol_data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if position should be exited based on technical indicators.
        
        Args:
            position: Trading position
            symbol_data: Current market data for the symbol
            
        Returns:
            Tuple of (should_exit, confidence)
        """
        # Initialize score and signal count to measure consensus
        exit_score = 0.0
        total_indicators = 0
        
        # Check RSI
        rsi = symbol_data.get("rsi_14", 50.0)
        if position.side == OrderSide.BUY and rsi >= self.rsi_overbought:
            # Overbought condition for long position
            rsi_score = min(1.0, (rsi - self.rsi_overbought) / (100 - self.rsi_overbought))
            exit_score += rsi_score * 0.25
            total_indicators += 1
        elif position.side == OrderSide.SELL and rsi <= self.rsi_oversold:
            # Oversold condition for short position
            rsi_score = min(1.0, (self.rsi_oversold - rsi) / self.rsi_oversold)
            exit_score += rsi_score * 0.25
            total_indicators += 1
            
        # Check MACD
        macd = symbol_data.get("macd", 0.0)
        macd_signal = symbol_data.get("macd_signal", 0.0)
        if position.side == OrderSide.BUY and macd < macd_signal:
            # Bearish MACD crossover for long position
            macd_diff = macd_signal - macd
            macd_score = min(1.0, macd_diff / 0.5)  # Normalize
            exit_score += macd_score * 0.2
            total_indicators += 1
        elif position.side == OrderSide.SELL and macd > macd_signal:
            # Bullish MACD crossover for short position
            macd_diff = macd - macd_signal
            macd_score = min(1.0, macd_diff / 0.5)  # Normalize
            exit_score += macd_score * 0.2
            total_indicators += 1
            
        # Check Bollinger Bands
        bb_upper = symbol_data.get("bb_upper", 0.0)
        bb_lower = symbol_data.get("bb_lower", 0.0)
        current_price = symbol_data.get("last_price", 0.0)
        
        if bb_upper > bb_lower:  # Ensure valid BB values
            bb_width = bb_upper - bb_lower
            if bb_width > 0:
                bb_position = (current_price - bb_lower) / bb_width
                
                if position.side == OrderSide.BUY and bb_position >= self.bb_exit_threshold:
                    # Price near upper BB for long position
                    bb_score = min(1.0, (bb_position - self.bb_exit_threshold) / (1.0 - self.bb_exit_threshold))
                    exit_score += bb_score * 0.2
                    total_indicators += 1
                elif position.side == OrderSide.SELL and bb_position <= (1.0 - self.bb_exit_threshold):
                    # Price near lower BB for short position
                    bb_score = min(1.0, ((1.0 - self.bb_exit_threshold) - bb_position) / (1.0 - self.bb_exit_threshold))
                    exit_score += bb_score * 0.2
                    total_indicators += 1
        
        # Calculate overall confidence if we have any signals
        if total_indicators > 0:
            # Average score across all indicators that triggered
            avg_score = exit_score / total_indicators
            
            # Add bonus for multiple confirming indicators
            indicator_consensus = total_indicators / 3.0  # 3 is the max number of indicators we check
            
            # Final confidence calculation
            confidence = avg_score * (0.7 + 0.3 * indicator_consensus)
            
            # If confidence exceeds threshold, return exit signal
            if confidence > 0.65:
                return True, min(0.9, confidence)
                
        return False, 0.0
    
    def _check_volatility_exit(self, position: Position, symbol_data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if position should be exited based on volatility changes.
        
        Args:
            position: Trading position
            symbol_data: Current market data for the symbol
            
        Returns:
            Tuple of (should_exit, confidence)
        """
        # Check if we have volatility data
        volatility = symbol_data.get("volatility", 0.0)
        
        if volatility <= 0:
            return False, 0.0
            
        # Higher volatility = faster exits
        
        # For long positions, check for momentum loss in volatile conditions
        if position.side == OrderSide.BUY:
            momentum = symbol_data.get("momentum_1m", 0.0)
            
            # If volatility is high and momentum is negative
            if volatility > 0.3 and momentum < -0.2:
                confidence = min(0.85, 0.70 + abs(momentum) * 0.5)
                return True, confidence
                
        # For short positions, check for momentum gain in volatile conditions
        elif position.side == OrderSide.SELL:
            momentum = symbol_data.get("momentum_1m", 0.0)
            
            # If volatility is high and momentum is positive
            if volatility > 0.3 and momentum > 0.2:
                confidence = min(0.85, 0.70 + momentum * 0.5)
                return True, confidence
                
        return False, 0.0
    
    def _check_volume_exit(self, position: Position, symbol_data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if position should be exited based on volume patterns.
        
        Args:
            position: Trading position
            symbol_data: Current market data for the symbol
            
        Returns:
            Tuple of (should_exit, confidence)
        """
        if not self.enable_volume_filter:
            return False, 0.0
            
        # Check volume spike
        volume = symbol_data.get("volume", 0.0)
        avg_volume = symbol_data.get("avg_volume", volume)
        
        if avg_volume > 0 and volume / avg_volume >= self.volume_spike_threshold:
            # Check price action direction with volume spike
            price_change = symbol_data.get("price_change_5m", 0.0)
            
            # For long positions, exit on volume spike with negative price change
            if position.side == OrderSide.BUY and price_change < 0:
                vol_ratio = min(5.0, volume / avg_volume) / 5.0
                confidence = min(0.85, 0.70 + vol_ratio * 0.15)
                return True, confidence
                
            # For short positions, exit on volume spike with positive price change
            elif position.side == OrderSide.SELL and price_change > 0:
                vol_ratio = min(5.0, volume / avg_volume) / 5.0
                confidence = min(0.85, 0.70 + vol_ratio * 0.15)
                return True, confidence
                
        return False, 0.0
    
    def _check_risk_management_exit(self, position: Position, symbol_data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if position should be exited based on risk management rules.
        
        Args:
            position: Trading position
            symbol_data: Current market data for the symbol
            
        Returns:
            Tuple of (should_exit, confidence)
        """
        # Check maximum drawdown
        max_drawdown = position.get_max_drawdown_percent()
        if max_drawdown > self.max_drawdown_pct:
            dd_excess = max_drawdown / self.max_drawdown_pct - 1.0
            confidence = min(0.95, 0.85 + dd_excess * 0.1)
            return True, confidence
            
        # Check daily loss limit
        with self.daily_stats_lock:
            daily_pnl = self.daily_stats["realized_pnl"]
            current_pnl = position.unrealized_pnl
            
            # Project total PnL if we exit this position
            projected_pnl = daily_pnl + current_pnl
            
            # Calculate as percentage of account
            try:
                account = self.alpaca_api.get_account()
                equity = float(account.get("equity", "0"))
                
                if equity > 0:
                    projected_pnl_pct = projected_pnl / equity * 100.0
                    
                    # If projected PnL would exceed daily loss limit
                    if projected_pnl_pct < -self.daily_loss_limit_pct:
                        confidence = 0.9  # High confidence for risk management
                        return True, confidence
            except Exception as e:
                self.logger.error(f"Error checking account equity: {str(e)}")
                
        return False, 0.0
    
    def _generate_exit_signal(self, symbol: str, exit_reason: str, confidence: float, 
                             market_data: Dict[str, Any]) -> None:
        """
        Generate and process an exit signal.
        
        Args:
            symbol: Symbol to exit
            exit_reason: Reason for exit
            confidence: Exit confidence level
            market_data: Current market data
        """
        # Get position data
        with self.position_lock:
            position = self.active_positions.get(symbol)
            if not position:
                return
                
            current_price = market_data["symbol_data"][symbol].get("last_price", 0.0)
            
            # Create exit signal
            signal = Signal(
                symbol=symbol,
                type="EXIT",
                direction="SELL" if position.side == OrderSide.BUY else "BUY",
                price=current_price,
                position_size=position.quantity,
                confidence=confidence,
                timestamp=market_data.get("timestamp", int(time.time() * 1_000_000_000))
            )
            
            # Add indicators and exit reason
            signal.indicators["exit_reason"] = exit_reason
            
            # Add market data metrics to signal
            symbol_data = market_data["symbol_data"][symbol]
            for key in ["rsi_14", "macd", "bb_upper", "bb_lower", "volume", "volatility"]:
                if key in symbol_data:
                    signal.indicators[key] = symbol_data[key]
            
            # Add position metrics
            signal.metadata["position_duration"] = position.get_duration_seconds()
            signal.metadata["unrealized_pnl"] = position.unrealized_pnl
            signal.metadata["unrealized_pnl_percent"] = position.unrealized_pnl_percent
            
            # Store in exit signals
            with self.exit_signal_lock:
                self.exit_signals[signal.signal_id] = signal
                
        # Queue for processing
        self.signal_processor.queue_exit_signal(signal)
        
        self.logger.info(f"Generated exit signal for {symbol} with reason: {exit_reason}, confidence: {confidence:.2f}")
    
    def calculate_bracket_parameters(self, symbol: str, entry_price: float,
                                    market_data: Dict[str, Any]) -> Tuple[float, float]:
        """
        Calculate volatility-adjusted bracket parameters.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            market_data: Current market data
            
        Returns:
            Tuple of (take_profit_price, stop_loss_price)
        """
        # Get base parameters
        profit_target_pct = self.profit_target_base_pct
        stop_loss_pct = self.stop_loss_base_pct
        
        # Apply volatility adjustment if enabled
        if self.use_volatility_adjustment and symbol in market_data.get("symbol_data", {}):
            symbol_data = market_data["symbol_data"][symbol]
            
            # Get volatility metrics
            volatility = symbol_data.get("volatility", None)
            
            if volatility is not None:
                # Scale profit target based on volatility
                # Higher volatility = higher profit target
                volatility_factor = volatility / 0.2  # Normalize to baseline volatility
                profit_target_pct = self.profit_target_base_pct * (
                    1.0 + (volatility_factor - 1.0) * self.volatility_scaling_factor
                )
                
                # Scale stop loss based on volatility
                # Higher volatility = wider stop loss
                stop_loss_pct = self.stop_loss_base_pct * (
                    1.0 + (volatility_factor - 1.0) * self.volatility_scaling_factor
                )
                
                # Apply min/max constraints
                profit_target_pct = max(self.min_profit_target_pct,
                                      min(self.max_profit_target_pct, profit_target_pct))
                stop_loss_pct = max(self.min_stop_loss_pct,
                                   min(self.max_stop_loss_pct, stop_loss_pct))
                
                # Ensure minimum risk/reward ratio
                if profit_target_pct / stop_loss_pct < self.min_risk_reward_ratio:
                    # Adjust profit target up to maintain minimum ratio
                    profit_target_pct = stop_loss_pct * self.min_risk_reward_ratio
        
        # Calculate actual prices
        if entry_price > 0:
            take_profit_price = entry_price * (1.0 + profit_target_pct / 100.0)
            stop_loss_price = entry_price * (1.0 - stop_loss_pct / 100.0)
            return take_profit_price, stop_loss_price
        
        return 0.0, 0.0

    def process_entry_signal(self, signal: Signal, market_data: Dict[str, Any]) -> OrderResponse:
        """
        Process an entry signal by creating a bracket order.
        
        Args:
            signal: Entry signal
            market_data: Current market data
            
        Returns:
            Order response
        """
        # Calculate bracket parameters based on volatility
        take_profit_price, stop_loss_price = self.calculate_bracket_parameters(
            signal.symbol, signal.price, market_data
        )
        
        # Create bracket order
        order = Order(
            symbol=signal.symbol,
            side=OrderSide.BUY if signal.direction == "BUY" else OrderSide.SELL,
            type=OrderType.MARKET,
            quantity=signal.position_size,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
            order_class=OrderClass.BRACKET,
            time_in_force=TimeInForce.DAY
        )
        
        # Log bracket parameters
        self.logger.info(
            f"Created bracket order for {signal.symbol}: "
            f"Entry @ {signal.price:.2f}, TP @ {take_profit_price:.2f} (+{((take_profit_price/signal.price)-1)*100:.2f}%), "
            f"SL @ {stop_loss_price:.2f} ({((stop_loss_price/signal.price)-1)*100:.2f}%)"
        )
        
        # Submit order
        response = self.alpaca_api.submit_order(order)
        
        # Track order if successful
        if response.status != OrderStatus.REJECTED:
            self._track_bracket_order(signal, response)
            
        return response
    
    def _track_bracket_order(self, signal: Signal, order_response: OrderResponse) -> None:
        """
        Track a submitted bracket order.
        
        Args:
            signal: Original signal
            order_response: Order response from API
        """
        # Store order reference in tracked orders
        with self.order_lock:
            self.tracked_orders[order_response.order_id] = {
                "signal": signal,
                "response": order_response,
                "bracket_parts": {},  # Will be populated with bracket leg IDs from callback
                "status": "active",
                "submitted_at": time.time()
            }
    
    def _execute_exit_signal(self, signal: Signal) -> bool:
        """
        Execute an exit signal by creating and submitting an order.
        
        Args:
            signal: Exit signal to execute
            
        Returns:
            True if order was submitted successfully, False otherwise
        """
        if not signal.is_valid():
            self.logger.warning(f"Invalid exit signal for {signal.symbol}")
            return False
            
        try:
            # Get current position size and details
            with self.position_lock:
                position = self.active_positions.get(signal.symbol)
                if not position:
                    self.logger.warning(f"No active position found for {signal.symbol} when executing exit")
                    return False
                    
                position_size = position.quantity
                side = position.side
            
            # Create order
            exit_order = Order(
                symbol=signal.symbol,
                side=OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY,
                type=OrderType.MARKET,  # Market order for faster execution
                quantity=position_size,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit order
            response = self.alpaca_api.submit_order(exit_order)
            
            # Handle response
            if response.status == OrderStatus.REJECTED:
                self.logger.error(f"Exit order rejected for {signal.symbol}: {response.status_message}")
                return False
                
            # Log exit
            self.logger.info(f"Exit order submitted for {signal.symbol}, order ID: {response.order_id}, "
                           f"reason: {signal.indicators.get('exit_reason', 'unknown')}")
            
            # Update position tracking
            if response.status == OrderStatus.FILLED:
                self._handle_position_exit(signal.symbol, position, response)
            
            self.exit_count += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing exit signal for {signal.symbol}: {str(e)}")
            return False
    
    def _handle_position_exit(self, symbol: str, position: Position, order_response: OrderResponse) -> None:
        """
        Handle a completed position exit by updating tracking data and statistics.
        
        Args:
            symbol: Symbol that was exited
            position: Position that was exited
            order_response: Order response for the exit
        """
        # Calculate realized P&L
        exit_price = order_response.filled_price
        exit_quantity = order_response.filled_quantity
        
        if position.side == OrderSide.BUY:
            realized_pnl = (exit_price - position.entry_price) * exit_quantity
            realized_pnl_pct = ((exit_price / position.entry_price) - 1.0) * 100.0
        else:
            realized_pnl = (position.entry_price - exit_price) * exit_quantity
            realized_pnl_pct = ((position.entry_price / exit_price) - 1.0) * 100.0
            
        # Update position with final values
        position.realized_pnl = realized_pnl
        position.current_price = exit_price
        position.update(exit_price)
        
        # Update daily stats
        with self.daily_stats_lock:
            self.daily_stats["realized_pnl"] += realized_pnl
            self.daily_stats["total_trades"] += 1
            if realized_pnl > 0:
                self.daily_stats["win_trades"] += 1
            else:
                self.daily_stats["loss_trades"] += 1
        
        # Move to position history
        with self.position_lock:
            # Remove from active positions
            if symbol in self.active_positions:
                del self.active_positions[symbol]
                
            # Add to position history
            if symbol not in self.position_history:
                self.position_history[symbol] = []
                
            self.position_history[symbol].append(position)
            
        self.logger.info(f"Position exit complete for {symbol} with realized PnL: ${realized_pnl:.2f} ({realized_pnl_pct:.2f}%)")
    
    def optimize_exits(self, active_positions: List[Signal], current_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate exit signals based on current market data.
        
        Args:
            active_positions: List of active positions as signals
            current_data: Current market data
        
        Returns:
            List of exit signals
        """
        # Start timing
        start_time = time.perf_counter()
        
        # Store exit signals
        exit_signals = []
        
        # Convert active positions to internal format if needed
        processed_positions = {}
        for pos_signal in active_positions:
            # Skip invalid signals
            if not pos_signal.symbol or pos_signal.position_size <= 0:
                continue
                
            # Create internal position object
            side = OrderSide.BUY if pos_signal.type == "BUY" else OrderSide.SELL
            
            position = Position(
                symbol=pos_signal.symbol,
                side=side,
                entry_price=pos_signal.price,
                quantity=pos_signal.position_size,
                stop_loss=pos_signal.stop_loss,
                take_profit=pos_signal.take_profit
            )
            
            processed_positions[pos_signal.symbol] = position
        
        # Process each position for exit conditions
        for symbol, position in processed_positions.items():
            # Check if symbol exists in the data
            if symbol not in current_data.get("symbol_data", {}):
                continue
            
            # Get current price and timestamp
            symbol_data = current_data["symbol_data"][symbol]
            current_price = symbol_data.get("last_price", 0.0)
            
            # Update position with current price
            position.update(current_price)
            
            # Check exit conditions
            exit_reason, exit_confidence = self._check_exit_conditions_for_signal(position, symbol_data, current_data.get("timestamp", 0))
            
            # Create exit signal if needed
            if exit_reason:
                exit_signal = Signal(
                    symbol=symbol,
                    type="EXIT",
                    direction="SELL" if position.side == OrderSide.BUY else "BUY",
                    price=current_price,
                    position_size=position.quantity,
                    confidence=exit_confidence,
                    timestamp=current_data.get("timestamp", int(time.time() * 1_000_000_000))
                )
                
                # Add exit reason to signal
                exit_signal.indicators["exit_reason"] = exit_reason
                
                # Add to signals
                exit_signals.append(exit_signal)
        
        # End timing
        end_time = time.perf_counter()
        duration = (end_time - start_time) * 1_000_000  # Convert to microseconds
        
        self.logger.info(f"Exit optimization completed in {duration:.2f} µs, "
                        f"generated {len(exit_signals)} exit signals from "
                        f"{len(active_positions)} active positions")
        
        return exit_signals
    
    def _check_exit_conditions_for_signal(self, position: Position, symbol_data: Dict[str, Any], 
                                         current_timestamp: int) -> Tuple[str, float]:
        """
        Simplified exit condition check for external signals.
        
        Args:
            position: Position object
            symbol_data: Market data for the symbol
            current_timestamp: Current timestamp
            
        Returns:
            Tuple of (exit_reason, confidence)
        """
        # Check profit target
        if position.unrealized_pnl_percent >= self.profit_target_pct:
            return "profit_target", 0.95
            
        # Check stop loss
        if position.unrealized_pnl_percent <= -self.stop_loss_pct:
            return "stop_loss", 0.95
            
        # Check trailing stop
        if position.side == OrderSide.BUY:
            # For long positions, check trailing stop
            if position.highest_price > position.entry_price:
                trail_level = position.highest_price * (1.0 - self.trailing_stop_pct / 100.0)
                if position.current_price <= trail_level:
                    return "trailing_stop", 0.90
        else:
            # For short positions, check trailing stop
            if position.lowest_price < position.entry_price:
                trail_level = position.lowest_price * (1.0 + self.trailing_stop_pct / 100.0)
                if position.current_price >= trail_level:
                    return "trailing_stop", 0.90
        
        # Check technical indicators
        rsi = symbol_data.get("rsi_14", 50.0)
        if (position.side == OrderSide.BUY and rsi >= self.rsi_overbought) or \
           (position.side == OrderSide.SELL and rsi <= self.rsi_oversold):
            return "technical_indicators", 0.75
            
        # Check position duration if timestamp available
        if current_timestamp > 0 and position.entry_time > 0:
            position_duration_seconds = position.get_duration_seconds()
            max_duration_seconds = self.max_holding_time_minutes * 60
            
            if position_duration_seconds >= max_duration_seconds:
                return "time_exit", 0.80
        
        # No exit condition met
        return "", 0.0
    
    def execute_exit_trades(self, exit_signals: List[Signal]) -> None:
        """
        Execute exit trades based on exit signals
        
        Args:
            exit_signals: List of exit signals
        """
        # Start timing
        start_time = time.perf_counter()
        
        # Skip if no signals
        if not exit_signals:
            return
        
        # Create orders for each signal
        orders = []
        for signal in exit_signals:
            # Create order
            try:
                order = Order(
                    symbol=signal.symbol,
                    side=OrderSide.SELL if signal.direction == "BUY" else OrderSide.BUY,
                    type=OrderType.MARKET,
                    quantity=signal.position_size,
                    time_in_force=TimeInForce.DAY,
                    client_order_id=f"exit_{signal.symbol}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
                )
                orders.append(order)
            except Exception as e:
                self.logger.error(f"Error creating exit order for {signal.symbol}: {str(e)}")
        
        # Submit orders in batch for efficiency
        if orders:
            try:
                # Limit batch size
                max_batch = min(self.max_concurrent_exits, len(orders))
                for i in range(0, len(orders), max_batch):
                    batch = orders[i:i+max_batch]
                    responses = self.alpaca_api.submit_orders_batch(batch)
                    
                    # Process responses
                    for client_id, response in responses.items():
                        self.logger.info(f"Exit order {response.order_id} for {response.symbol} submitted with status {response.status}")
                        
                        # Update position tracking for filled orders
                        if response.status == OrderStatus.FILLED:
                            symbol = response.symbol
                            with self.position_lock:
                                position = self.active_positions.get(symbol)
                                if position:
                                    self._handle_position_exit(symbol, position, response)
            except Exception as e:
                self.logger.error(f"Error executing batch exit trades: {str(e)}")
        
        # End timing
        end_time = time.perf_counter()
        duration = (end_time - start_time) * 1_000_000  # Convert to microseconds
        
        self.logger.info(f"Exit trade execution completed in {duration:.2f} µs for {len(exit_signals)} signals")
    
    def set_profit_target(self, profit_target_pct: float) -> None:
        """
        Set the profit target percentage.
        
        Args:
            profit_target_pct: Profit target percentage
        """
        self.profit_target_pct = profit_target_pct
        self.logger.info(f"Updated profit target to {profit_target_pct}%")
    
    def set_stop_loss(self, stop_loss_pct: float) -> None:
        """
        Set the stop loss percentage.
        
        Args:
            stop_loss_pct: Stop loss percentage
        """
        self.stop_loss_pct = stop_loss_pct
        self.logger.info(f"Updated stop loss to {stop_loss_pct}%")
    
    def set_trailing_stop(self, trailing_stop_pct: float) -> None:
        """
        Set the trailing stop percentage.
        
        Args:
            trailing_stop_pct: Trailing stop percentage
        """
        self.trailing_stop_pct = trailing_stop_pct
        self.logger.info(f"Updated trailing stop to {trailing_stop_pct}%")
    
    def set_max_holding_time(self, minutes: int) -> None:
        """
        Set the maximum holding time.
        
        Args:
            minutes: Maximum holding time in minutes
        """
        self.max_holding_time_minutes = minutes
        self.logger.info(f"Updated max holding time to {minutes} minutes")
    
    def get_active_positions(self) -> Dict[str, Position]:
        """
        Get active positions.
        
        Returns:
            Dictionary of active positions
        """
        with self.position_lock:
            return self.active_positions.copy()
    
    def get_exit_signals(self) -> Dict[str, Signal]:
        """
        Get exit signals.
        
        Returns:
            Dictionary of exit signals
        """
        with self.exit_signal_lock:
            return self.exit_signals.copy()
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """
        Get daily trading statistics.
        
        Returns:
            Dictionary of daily statistics
        """
        with self.daily_stats_lock:
            stats = self.daily_stats.copy()
            
            # Calculate win rate
            if stats["total_trades"] > 0:
                stats["win_rate"] = stats["win_trades"] / stats["total_trades"] * 100.0
            else:
                stats["win_rate"] = 0.0
                
            return stats
    
    def reset_position_tracking(self, symbol: str = None) -> None:
        """
        Reset position tracking for a symbol or all symbols.
        
        Args:
            symbol: Symbol to reset (None for all symbols)
        """
        with self.position_lock:
            if symbol:
                if symbol in self.active_positions:
                    # Move to history before deleting
                    if symbol not in self.position_history:
                        self.position_history[symbol] = []
                    
                    self.position_history[symbol].append(self.active_positions[symbol])
                    del self.active_positions[symbol]
            else:
                # Move all to history
                for sym, pos in self.active_positions.items():
                    if sym not in self.position_history:
                        self.position_history[sym] = []
                    
                    self.position_history[sym].append(pos)
                
                self.active_positions = {}
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the enhanced exit strategy
        """
        self.logger.info("Cleaning up Enhanced Exit Strategy resources")
        
        # Stop main processing loop
        self.running = False
        
        # Stop async signal processor
        self.signal_processor.stop()
        
        # Clean up Alpaca API client
        if hasattr(self, 'alpaca_api'):
            self.alpaca_api.cleanup()
        
        # Clean up GPU resources if used
        if self.gpu_enabled and HAS_CUPY and cp is not None:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                self.logger.debug("GPU memory released")
            except Exception as e:
                self.logger.error(f"Error releasing GPU memory: {str(e)}")
        
        # Wait for main thread to terminate if running
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=2.0)
            
        self.logger.info("Enhanced Exit Strategy cleanup complete")
    
    def process_trading_signals(self, signals: List[Signal], market_data: Dict[str, Any]) -> None:
        """
        Process trading signals with bracket orders.
        
        Args:
            signals: List of trading signals
            market_data: Current market data
        """
        entry_signals = [s for s in signals if s.type == "ENTRY"]
        
        # Process each entry signal with bracket orders
        for signal in entry_signals:
            # Skip invalid signals
            if not signal.is_valid():
                continue
                
            try:
                # Process signal with bracket order
                response = self.process_entry_signal(signal, market_data)
                
                # Track order
                if response.status != OrderStatus.REJECTED:
                    self._track_bracket_order(signal, response)
                    
            except Exception as e:
                self.logger.error(f"Error processing entry signal for {signal.symbol}: {str(e)}")


# Factory function to create a configured exit strategy
def create_exit_strategy(config=None):
    """
    Create and initialize an enhanced exit strategy with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized enhanced exit strategy
    """
    strategy = EnhancedExitStrategy(config)
    strategy.initialize()
    return strategy
