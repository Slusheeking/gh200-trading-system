"""
API Client and Managers for Enhanced Fast Exit Strategy
"""

import os
import time
import json
import threading
import uuid
import logging
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, time as time_obj
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import (
    Order, OrderResponse, OrderSide, OrderType, OrderClass, TimeInForce, OrderStatus,
    Position, Signal, ValidationError
)
from .helpers import retry_on_error

# --- AlpacaAPI ---

class AlpacaAPI:
    """
    REST API client for Alpaca Markets with high-performance optimizations and connection pooling.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        load_dotenv()
        self.alpaca_api_key = os.getenv("ALPACA_API_KEY", "")
        self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.alpaca_endpoint = os.getenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2")
        alpaca_config = config.get("data_sources", {}).get("alpaca", {})
        if alpaca_config.get("enabled", False):
            if not self.alpaca_api_key:
                self.alpaca_api_key = alpaca_config.get("api_key", "")
            if not self.alpaca_secret_key:
                self.alpaca_secret_key = alpaca_config.get("secret_key", "")
            if not self.alpaca_endpoint:
                self.alpaca_endpoint = alpaca_config.get("endpoint", "https://paper-api.alpaca.markets/v2")
        self.thread_affinity = -1
        self.mutex = threading.Lock()
        perf_config = config.get("performance", {})
        self.max_workers = perf_config.get("processor_threads", min(32, os.cpu_count() * 2))
        self.session = requests.Session()
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
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers * 2
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        timeout_config = alpaca_config.get("http", {}).get("timeout", {})
        self.connect_timeout = timeout_config.get("connect", 3.0)
        self.read_timeout = timeout_config.get("read", 10.0)
        self.headers = {
            "APCA-API-KEY-ID": self.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret_key,
            "Content-Type": "application/json"
        }
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="AlpacaAPI-Worker"
        )
        self.account_cache = {}
        self.account_cache_time = 0
        self.account_cache_ttl = 5.0
        self.positions_cache = {}
        self.positions_cache_time = 0
        self.positions_cache_ttl = 2.0
        self.order_cache = {}
        self.order_cache_lock = threading.Lock()
        self.order_cache_cleanup_thread = None
        self.order_cache_ttl = 60.0
        self.circuit_breaker_engaged = False
        self.circuit_breaker_reset_time = 0
        self.consecutive_errors = 0
        self.error_threshold = 5
        self.circuit_breaker_cooldown = 60
        self.circuit_breaker_lock = threading.Lock()
        self.logger = logging.getLogger("AlpacaAPI")
        self.logger.info(f"Configured high-performance HTTP client with max_workers={self.max_workers}, "
                         f"max_retries={max_retries}, backoff_factor={backoff_factor}, "
                         f"connect_timeout={self.connect_timeout}s, read_timeout={self.read_timeout}s")

    def initialize(self) -> bool:
        self.logger.info("Initializing Alpaca REST API client")
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            self.logger.warning("No API keys configured for Alpaca REST client")
            return False
        try:
            account = self.get_account()
            self.logger.info(f"Connected to Alpaca API. Account ID: {account.get('id')}, Status: {account.get('status')}")
            self._start_background_threads()
            self.running = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca API: {str(e)}")
            return False

    def _start_background_threads(self):
        self.order_cache_cleanup_thread = threading.Thread(
            target=self._order_cache_cleanup_worker, daemon=True, name="OrderCacheCleanup"
        )
        self.order_cache_cleanup_thread.start()

    def _order_cache_cleanup_worker(self):
        while True:
            try:
                time.sleep(30)
                if not self.running:
                    break
                current_time = time.time()
                with self.order_cache_lock:
                    expired_order_ids = []
                    for order_id, (order_data, cache_time) in self.order_cache.items():
                        if current_time - cache_time > self.order_cache_ttl:
                            expired_order_ids.append(order_id)
                    for order_id in expired_order_ids:
                        del self.order_cache[order_id]
                    if expired_order_ids:
                        self.logger.debug(f"Cleaned up {len(expired_order_ids)} expired order cache entries")
            except Exception as e:
                self.logger.error(f"Error in order cache cleanup: {str(e)}")

    def _check_circuit_breaker(self) -> bool:
        with self.circuit_breaker_lock:
            if self.circuit_breaker_engaged:
                current_time = time.time()
                if current_time > self.circuit_breaker_reset_time:
                    self.circuit_breaker_engaged = False
                    self.consecutive_errors = 0
                    self.logger.info("Circuit breaker reset")
                    return False
                else:
                    return True
            return False

    def _record_error(self):
        with self.circuit_breaker_lock:
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.error_threshold:
                self.circuit_breaker_engaged = True
                self.circuit_breaker_reset_time = time.time() + self.circuit_breaker_cooldown
                self.logger.warning(f"Circuit breaker engaged for {self.circuit_breaker_cooldown} seconds after {self.consecutive_errors} consecutive errors")

    def _record_success(self):
        with self.circuit_breaker_lock:
            self.consecutive_errors = 0

    @retry_on_error(max_retries=3, retry_delay=0.5, allowed_exceptions=(requests.exceptions.RequestException, json.JSONDecodeError), log_errors=True)
    def get_account(self) -> Dict[str, Any]:
        if self._check_circuit_breaker():
            self.logger.warning("Circuit breaker engaged. Returning cached account or empty dict")
            return self.account_cache or {}
        current_time = time.time()
        if self.account_cache and current_time - self.account_cache_time < self.account_cache_ttl:
            return self.account_cache.copy()
        url = f"{self.alpaca_endpoint}/account"
        try:
            response = self.session.get(
                url, headers=self.headers, timeout=(self.connect_timeout, self.read_timeout)
            )
            response.raise_for_status()
            self.account_cache = response.json()
            self.account_cache_time = current_time
            self._record_success()
            return self.account_cache.copy()
        except requests.exceptions.RequestException as e:
            self._record_error()
            self.logger.error(f"Error getting account information: {str(e)}")
            if self.account_cache:
                return self.account_cache.copy()
            raise

    # ... (Other methods omitted for brevity, but should be included in the real file)
    # Methods: get_positions, get_position, submit_order, submit_orders_batch, cancel_order, get_order, get_orders, get_bracket_orders, get_clock, cleanup

# --- CapitalManager ---

class CapitalManager:
    """
    Manages trading capital limits and recycling for the GH200 Trading System.
    """

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        trading_config = config.get("trading", {})
        capital_config = trading_config.get("capital", {})
        self.daily_max_exposure: float = capital_config.get("daily_max_exposure", 25000.0)
        self.current_exposure: float = 0.0
        self.used_today: float = 0.0
        self.recycled_capital: float = 0.0
        self.max_position_size: float = capital_config.get("max_position_size", 5000.0)
        self.min_position_size: float = capital_config.get("min_position_size", 1000.0)
        self.open_positions: Dict[str, float] = {}
        self.position_history: List[Dict[str, Any]] = []
        self.capital_lock = threading.Lock()
        self.reset_hour: int = capital_config.get("reset_hour", 9)
        self.reset_minute: int = capital_config.get("reset_minute", 30)
        self.last_reset_date = datetime.now().date()
        self.logger = logging.getLogger("CapitalManager")
        self.logger.info(
            f"CapitalManager initialized with daily_max_exposure=${self.daily_max_exposure:.2f}, "
            f"max_position_size=${self.max_position_size:.2f}, min_position_size=${self.min_position_size:.2f}"
        )

    # ... (Methods omitted for brevity: get_available_capital, allocate_capital, release_capital, adjust_capital, _check_daily_reset, _reset_daily_limits, get_capital_status, update_config)

# --- BracketOrderManager ---

class BracketOrderManager:
    """
    Manager for bracket orders and related functionality.
    """

    def __init__(self, exit_strategy, config: Dict[str, Any] = None):
        self.exit_strategy = exit_strategy
        self.logger = exit_strategy.logger
        self.alpaca_api = exit_strategy.alpaca_api
        self.config = config or {}
        trading_config = self.config.get("trading", {})
        bracket_config = trading_config.get("bracket", {})
        self.profit_target_base_pct = bracket_config.get("profit_target_pct", 2.0)
        self.stop_loss_base_pct = bracket_config.get("stop_loss_pct", 1.0)
        self.use_volatility_adjustment = bracket_config.get("use_volatility_adjustment", True)
        self.volatility_scaling_factor = bracket_config.get("volatility_scaling_factor", 0.5)
        self.min_profit_target_pct = bracket_config.get("min_profit_target_pct", 0.5)
        self.max_profit_target_pct = bracket_config.get("max_profit_target_pct", 5.0)
        self.min_stop_loss_pct = bracket_config.get("min_stop_loss_pct", 0.3)
        self.max_stop_loss_pct = bracket_config.get("max_stop_loss_pct", 3.0)
        self.min_risk_reward_ratio = bracket_config.get("min_risk_reward_ratio", 1.5)
        self.tracked_orders = {}
        self.order_lock = threading.Lock()
        self.market_hours_cache = {}
        self.market_hours_cache_time = 0
        self.market_hours_cache_ttl = 3600
        self.logger.info(
            f"Bracket order manager initialized with profit_target={self.profit_target_base_pct}%, "
            f"stop_loss={self.stop_loss_base_pct}%, volatility_adjustment={self.use_volatility_adjustment}"
        )

    # ... (Methods omitted for brevity: calculate_bracket_parameters, process_entry_signal, _track_bracket_order, _identify_bracket_leg_type, update_tracked_orders, get_bracket_order_status, cancel_bracket_order, _is_market_open, cleanup)