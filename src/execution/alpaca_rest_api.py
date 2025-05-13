"""
Alpaca REST API client implementation

This module provides a client for the Alpaca Markets REST API to execute trades.
It handles authentication, order submission, and parsing of API responses.
"""

import json
import threading
import requests
import os
import backoff
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests_futures.sessions import FuturesSession

class OrderStatus:
    """Order status constants matching C++ enum"""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderSide:
    """Order side constants matching C++ enum"""
    BUY = "buy"
    SELL = "sell"

class OrderType:
    """Order type constants matching C++ enum"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderClass:
    """Order class constants matching C++ enum"""
    SIMPLE = "simple"
    BRACKET = "bracket"
    OCO = "oco"
    OTO = "oto"

class TimeInForce:
    """Time in force constants matching C++ enum"""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"

@dataclass
class Order:
    """Class representing an order"""
    symbol: str = ""
    side: str = ""
    type: str = ""
    quantity: float = 0.0
    limit_price: float = 0.0
    stop_price: float = 0.0
    time_in_force: str = ""
    client_order_id: str = ""
    order_class: str = ""
    take_profit_price: float = 0.0
    stop_loss_price: float = 0.0
    trail_percent: float = 0.0

@dataclass
class BracketOrder:
    """Class representing a bracket order"""
    entry: Order = None

@dataclass
class OrderResponse:
    """Class representing an order response"""
    order_id: str = ""
    client_order_id: str = ""
    symbol: str = ""
    status: str = ""
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    status_message: str = ""


class AlpacaRestAPI:
    """REST API client for Alpaca Markets with high-performance optimizations"""
    
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
        self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY", "")  # Use ALPACA_SECRET_KEY from .env
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
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="AlpacaREST-Worker"
        )
        
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
        
        # Create a FuturesSession for asynchronous requests
        self.futures_session = FuturesSession(
            session=self.session,
            max_workers=self.max_workers
        )
        
        # Set up authentication headers
        self.headers = {
            "APCA-API-KEY-ID": self.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret_key,
            "Content-Type": "application/json"
        }
        
        logging.info(f"Configured high-performance HTTP client with max_workers={self.max_workers}, " +
                    f"max_retries={max_retries}, backoff_factor={backoff_factor}, " +
                    f"connect_timeout={self.connect_timeout}s, read_timeout={self.read_timeout}s")
    
    def initialize(self):
        """Initialize the REST client"""
        logging.info("Initializing Alpaca REST API client")
        
        # Check if API keys are available
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            logging.warning("No API keys configured for Alpaca REST client")
            return False
        
        # Test connection
        try:
            account = self.get_account()
            logging.info(f"Connected to Alpaca API. Account ID: {account.get('id')}, Status: {account.get('status')}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to Alpaca API: {str(e)}")
            return False
    
    def set_thread_affinity(self, core_id: int):
        """
        Set thread affinity for the API thread
        
        Args:
            core_id: CPU core ID to pin the thread to
        """
        self.thread_affinity = core_id
        logging.debug(f"Thread affinity set to core {core_id} (not implemented)")
    
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
        Get account information
        
        Returns:
            Account information
        """
        url = f"{self.alpaca_endpoint}/account"
        
        try:
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error getting account information: {str(e)}")
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
        Get all positions
        
        Returns:
            List of positions
        """
        url = f"{self.alpaca_endpoint}/positions"
        
        try:
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error getting positions: {str(e)}")
            raise
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, json.JSONDecodeError),
        max_tries=3,
        giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code < 500,
        factor=2,
        jitter=backoff.full_jitter
    )
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get position for a specific symbol
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position information
        """
        url = f"{self.alpaca_endpoint}/positions/{symbol}"
        
        try:
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                # No position for this symbol
                return {}
            logging.error(f"Error getting position for {symbol}: {str(e)}")
            raise
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, json.JSONDecodeError),
        max_tries=3,
        giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code < 500,
        factor=2,
        jitter=backoff.full_jitter
    )
    def get_orders(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all orders
        
        Args:
            status: Filter by order status
            
        Returns:
            List of orders
        """
        url = f"{self.alpaca_endpoint}/orders"
        params = {}
        
        if status:
            params["status"] = status
        
        try:
            response = self.session.get(
                url,
                headers=self.headers,
                params=params,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error getting orders: {str(e)}")
            raise
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, json.JSONDecodeError),
        max_tries=3,
        giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code < 500,
        factor=2,
        jitter=backoff.full_jitter
    )
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order by ID
        
        Args:
            order_id: Order ID
            
        Returns:
            Order information
        """
        url = f"{self.alpaca_endpoint}/orders/{order_id}"
        
        try:
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error getting order {order_id}: {str(e)}")
            raise
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, json.JSONDecodeError),
        max_tries=3,
        giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code < 500,
        factor=2,
        jitter=backoff.full_jitter
    )
    def get_order_by_client_id(self, client_order_id: str) -> Dict[str, Any]:
        """
        Get order by client order ID
        
        Args:
            client_order_id: Client order ID
            
        Returns:
            Order information
        """
        url = f"{self.alpaca_endpoint}/orders:by_client_order_id"
        params = {"client_order_id": client_order_id}
        
        try:
            response = self.session.get(
                url,
                headers=self.headers,
                params=params,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error getting order by client ID {client_order_id}: {str(e)}")
            raise
    
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
        url = f"{self.alpaca_endpoint}/orders"
        
        # Build order payload
        payload = {
            "symbol": order.symbol,
            "qty": str(order.quantity),
            "side": order.side,
            "type": order.type,
            "time_in_force": order.time_in_force,
        }
        
        # Add optional fields
        if order.limit_price > 0:
            payload["limit_price"] = str(order.limit_price)
        
        if order.stop_price > 0:
            payload["stop_price"] = str(order.stop_price)
        
        if order.client_order_id:
            payload["client_order_id"] = order.client_order_id
        
        # Handle bracket orders
        if order.order_class == OrderClass.BRACKET:
            payload["order_class"] = "bracket"
            
            if order.take_profit_price > 0:
                payload["take_profit"] = {
                    "limit_price": str(order.take_profit_price)
                }
            
            if order.stop_loss_price > 0:
                stop_loss = {
                    "stop_price": str(order.stop_loss_price)
                }
                
                # Add trailing stop if enabled
                if order.trail_percent > 0:
                    stop_loss["trail_percent"] = str(order.trail_percent)
                
                payload["stop_loss"] = stop_loss
        
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
            order_response = OrderResponse(
                order_id=order_data.get("id", ""),
                client_order_id=order_data.get("client_order_id", ""),
                symbol=order_data.get("symbol", ""),
                status=order_data.get("status", ""),
                filled_quantity=float(order_data.get("filled_qty", "0")),
                filled_price=float(order_data.get("filled_avg_price", "0") or "0"),
                status_message=order_data.get("status_message", "")
            )
            
            logging.info(f"Order submitted: {order_response.order_id} for {order_response.symbol}")
            return order_response
        except requests.exceptions.RequestException as e:
            logging.error(f"Error submitting order: {str(e)}")
            
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
        url = f"{self.alpaca_endpoint}/orders/{order_id}"
        
        try:
            response = self.session.delete(
                url,
                headers=self.headers,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            response.raise_for_status()
            logging.info(f"Order {order_id} canceled")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Error canceling order {order_id}: {str(e)}")
            return False
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, json.JSONDecodeError),
        max_tries=3,
        giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code < 500,
        factor=2,
        jitter=backoff.full_jitter
    )
    def cancel_all_orders(self) -> bool:
        """
        Cancel all open orders
        
        Returns:
            True if successful, False otherwise
        """
        url = f"{self.alpaca_endpoint}/orders"
        
        try:
            response = self.session.delete(
                url,
                headers=self.headers,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            response.raise_for_status()
            logging.info("All orders canceled")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Error canceling all orders: {str(e)}")
            return False
    
    def cleanup(self):
        """
        Clean up resources used by the REST client
        
        This method should be called when the client is no longer needed
        to ensure proper release of resources.
        """
        logging.info("Cleaning up AlpacaRestAPI resources")
        
        # Shutdown thread pool
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
            logging.debug("Thread pool shut down")
        
        # Close futures session
        if hasattr(self, 'futures_session'):
            # The underlying session will be closed by the session cleanup
            logging.debug("Futures session closed")
        
        # Close HTTP session
        if hasattr(self, 'session'):
            self.session.close()
            logging.debug("HTTP session closed")
        
        # Force garbage collection of any large buffers
        import gc
        gc.collect()
        logging.debug("Memory buffers released")
