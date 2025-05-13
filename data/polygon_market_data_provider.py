"""
Polygon Market Data Provider for HFT Model

This module combines the functionality of polygon_rest_api.py and market_data_processor.py
to provide optimized market data specifically for the HFT model with minimal latency.
"""

import os
import time
import numpy as np
import logging
import threading
import collections
from typing import Dict, List, Any, Union
import concurrent.futures
from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
import psutil 

# Use orjson if available (faster), otherwise fallback to standard json
try:
    import orjson as json_fast  # Faster JSON parsing
except ImportError:
    try:
        import ujson as json_fast  # Faster JSON parsing
    except ImportError:
        import json as json_fast

        logging.warning(
            "Using standard json module. Consider installing orjson or ujson for better performance."
        )

# GPU and acceleration imports
try:
    import cupy as cp

    HAS_GPU = True
except (ImportError, ValueError, AttributeError) as e:
    cp = np
    HAS_GPU = False
    logging.warning(
        f"GPU libraries not available or incompatible: {str(e)}. Falling back to CPU processing."
    )

# Import Redis client
try:
    from memory.redis_client import RedisClient

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logging.warning("Redis client not available. Redis integration will be disabled.")

# Import shared memory components (optional module for GPU memory optimization)
# Note: This module is not included in the project yet, but the code handles its absence gracefully
try:
    from shared_memory.gpu_memory_pool import SharedGPUMemoryPool  # type: ignore # pylance-ignore

    HAS_SHARED_MEMORY = True
except ImportError:
    HAS_SHARED_MEMORY = False
    logging.warning(
        "Shared memory components not available. Zero-copy operations will be disabled."
    )


@dataclass
class MarketSnapshot:
    """Class to store processed market data ready for HFT model"""

    symbol: str = ""
    timestamp: int = 0

    # Price data
    last_price: float = 0.0
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_ask_spread: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    volume: int = 0
    vwap: float = 0.0

    # Technical indicators (only those needed by HFT model)
    rsi_14: float = 50.0
    macd: float = 0.0
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0

    # Additional indicators for fast path
    volume_acceleration: float = 0.0
    price_change_5m: float = 0.0
    momentum_1m: float = 0.0

    # Volatility metrics (added for adaptive quantization)
    intraday_volatility: float = 0.0
    bid_ask_volatility: float = 0.0

    # Market condition flags (added for signal coordination)
    is_volatile: bool = False
    trend_direction: int = 0  # -1: downtrend, 0: neutral, 1: uptrend

    # GPU memory handle for zero-copy operations (new)
    gpu_memory_handle: Any = None

    # Convert to feature array for model input
    def to_feature_array(self) -> np.ndarray:
        """Convert snapshot to feature array for model input"""
        features = np.array(
            [
                self.last_price,
                self.bid_price,
                self.ask_price,
                self.bid_ask_spread,
                self.high_price,
                self.low_price,
                self.volume,
                self.vwap,
                self.rsi_14,
                self.macd,
                self.bb_upper,
                self.bb_middle,
                self.bb_lower,
                self.volume_acceleration,
                self.price_change_5m,
                self.momentum_1m,
                self.intraday_volatility,  # Added for adaptive quantization
                self.bid_ask_volatility,  # Added for adaptive quantization
            ],
            dtype=np.float32,
        )
        return features

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {k: v for k, v in self.__dict__.items() if k != "gpu_memory_handle"}


@dataclass
class PredictiveRequest:
    """Class to track predictive data requests"""

    symbol: str
    probability: float = 0.0
    last_requested: float = 0.0
    frequency: float = 0.0  # How often it's requested (requests per second)
    priority: int = 0  # Priority level (higher = more important)


class LRUCache:
    """
    Simple LRU cache for efficiently managing historical data
    """

    def __init__(self, max_size=1000):
        self.cache = collections.OrderedDict()
        self.max_size = max_size

    def get(self, key, default=None):
        """Get item from cache, moving it to the end if found"""
        try:
            value = self.cache.pop(key)
            self.cache[key] = value  # Move to end (most recently used)
            return value
        except KeyError:
            return default

    def put(self, key, value):
        """Add item to cache, evicting oldest item if necessary"""
        try:
            self.cache.pop(key)  # Remove if exists
        except KeyError:
            if len(self.cache) >= self.max_size:
                # Remove oldest item (first in OrderedDict)
                self.cache.popitem(last=False)
        self.cache[key] = value  # Add to end (most recently used)

    def remove(self, key):
        """Remove item from cache if it exists"""
        try:
            self.cache.pop(key)
            return True
        except KeyError:
            return False

    def keys(self):
        """Return all keys in the cache"""
        return list(self.cache.keys())

    def __len__(self):
        return len(self.cache)

    def __contains__(self, key):
        return key in self.cache


class IndicatorState:
    """
    Class to store state for incremental indicator calculations
    """

    def __init__(self, period=14):
        # General state
        self.last_update = 0  # Timestamp of last update

        # Price history (limited window)
        self.close_prices = collections.deque(maxlen=50)  # Store recent prices
        self.timestamps = collections.deque(maxlen=50)  # Store timestamps for cleanup

        # RSI state
        self.rsi_period = period
        self.rsi_gains = collections.deque(maxlen=period + 1)
        self.rsi_losses = collections.deque(maxlen=period + 1)
        self.last_price = None
        self.rsi_value = 50.0  # Default value

        # MACD state
        self.fast_ema = None
        self.slow_ema = None
        self.signal_ema = None
        self.macd_value = 0.0

        # Bollinger Bands state
        self.bb_window = 20
        self.sma = None
        self.std = None
        self.bb_upper = 0.0
        self.bb_middle = 0.0
        self.bb_lower = 0.0

        # Additional indicators
        self.volume_data = collections.deque(maxlen=10)  # For volume acceleration
        self.volume_value = 0.0

        # Volatility tracking (new)
        self.price_changes = collections.deque(maxlen=20)  # For volatility calculation
        self.bid_ask_spreads = collections.deque(maxlen=20)  # For bid-ask volatility


class PolygonMarketDataProvider:
    """
    Optimized provider that fetches market data from Polygon.io and processes it
    specifically for the HFT model with minimal latency.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the market data provider

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Load environment variables
        load_dotenv()

        # Initialize API keys from environment variables first
        self.polygon_api_key = os.getenv("POLYGON_API_KEY", "")
        self.polygon_base_url = os.getenv("POLYGON_BASE_URL", "https://api.polygon.io")

        # Get API keys from config as fallback
        polygon_config = config.get("data_sources", {}).get("polygon", {})
        if polygon_config.get("enabled", False):
            # Only use config values if env vars aren't set
            if not self.polygon_api_key:
                self.polygon_api_key = polygon_config.get("api_key", "")
            if (
                not self.polygon_base_url
                or self.polygon_base_url == "https://api.polygon.io"
            ):
                if polygon_config.get("base_url"):
                    self.polygon_base_url = polygon_config.get("base_url")

        # Initialize filtering settings from config
        market_config = config.get("trading", {}).get("market", {})
        self.min_price = market_config.get("min_price", 5.0)
        self.max_price = market_config.get("max_price", 500.0)
        self.min_volume = market_config.get("min_volume", 500000)
        self.min_market_cap = market_config.get("min_market_cap", 500000000)

        # Performance settings
        perf_config = config.get("performance", {})
        self.use_gpu = perf_config.get("use_gpu", True) and HAS_GPU
        self.max_workers = perf_config.get(
            "processor_threads", min(32, os.cpu_count() * 2)
        )
        self.batch_size = perf_config.get("batch_size", 1000)
        self.max_history_length = perf_config.get("max_history_length", 200)

        # Thread pool for parallel processing
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="PolygonMarketData-Worker"
        )

        # LRU cache for historical data to prevent memory leaks
        history_size = config.get("performance", {}).get("history_cache_size", 1000)
        self.historical_data = LRUCache(max_size=history_size)

        # Store indicator state for incremental calculations
        self.indicator_states = {}

        # Configure HTTP session with retries and timeouts
        self.session = requests.Session()

        # Configure retry strategy from system.yaml
        retry_config = polygon_config.get("http", {}).get("retry", {})
        max_retries = retry_config.get("max_retries", 3)
        backoff_factor = retry_config.get("backoff_factor", 0.3)
        status_forcelist = retry_config.get(
            "status_forcelist", [429, 500, 502, 503, 504]
        )

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=["GET"],
        )

        # Create multiple adapters for connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.max_workers,
            pool_maxsize=self.max_workers * 2,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Configure timeouts from system.yaml
        timeout_config = polygon_config.get("http", {}).get("timeout", {})
        self.connect_timeout = timeout_config.get("connect", 3.0)
        self.read_timeout = timeout_config.get("read", 10.0)

        # Rate limiting
        self.rate_limit_mutex = threading.Lock()
        self.rate_limit_remaining = 1000  # Default value
        self.rate_limit_reset = 0  # Timestamp when rate limit resets
        self.rate_limit_threshold = polygon_config.get("rate_limit_threshold", 50)

        # Create LRU cache for market cap data
        self.market_cap_cache = LRUCache(max_size=10000)
        self.market_cap_cache_expiry = 24 * 60 * 60  # 24 hours in seconds
        self.market_cap_last_update = {}  # Timestamp of last update for each symbol

        # Redis client (if available)
        self.redis_client = None
        if HAS_REDIS:
            self._initialize_redis()

        # More granular synchronization
        self.symbol_locks = {}  # Symbol-specific locks
        self.symbol_lock_mutex = threading.Lock()  # For adding to symbol_locks
        self.global_mutex = threading.Lock()  # For global state operations

        # Health monitoring
        self.last_successful_fetch = 0  # Timestamp of last successful fetch
        self.consecutive_failures = 0  # Count of consecutive fetch failures
        self.circuit_breaker_engaged = False  # Whether circuit breaker is engaged
        self.circuit_breaker_reset_time = 0  # Time when circuit breaker will reset

        # Shared memory pool for zero-copy operations (new)
        self.shared_memory_pool = None
        if HAS_SHARED_MEMORY and self.use_gpu:
            try:
                self.shared_memory_pool = SharedGPUMemoryPool(
                    initial_size_mb=perf_config.get("shared_memory_size_mb", 256),
                    device_id=perf_config.get("gpu_device_id", 0),
                )
                logging.info(
                    "Initialized shared GPU memory pool for zero-copy operations"
                )
            except Exception as e:
                logging.error(f"Failed to initialize shared memory pool: {str(e)}")

        # Predictive prefetching (new)
        self.enable_predictive_prefetch = perf_config.get(
            "enable_predictive_prefetch", True
        )
        self.prefetch_symbols = collections.defaultdict(
            lambda: PredictiveRequest(symbol="")
        )
        self.prefetch_interval = perf_config.get("prefetch_interval_seconds", 1.0)
        self.prefetch_thread = None
        self.prefetch_running = False
        self.prefetch_lock = threading.Lock()

        # Adaptive timeout management (new)
        self.enable_adaptive_timeouts = perf_config.get(
            "enable_adaptive_timeouts", True
        )
        self.response_times = collections.deque(maxlen=100)
        self.min_timeout = timeout_config.get("min_timeout", 1.0)
        self.max_timeout = timeout_config.get("max_timeout", 15.0)
        self.timeout_percentile = timeout_config.get("timeout_percentile", 95)

        # Set CPU affinity for critical threads (new)
        self.use_cpu_affinity = perf_config.get("use_cpu_affinity", True)
        self.data_fetch_core_ids = perf_config.get(
            "data_fetch_core_ids", [0, 1]
        )  # Default to first two cores

        # Initialize CPU affinity if enabled
        if self.use_cpu_affinity:
            self._initialize_cpu_affinity()

        logging.info(
            f"Polygon Market Data Provider initialized with GPU acceleration: {self.use_gpu}, Redis integration: {HAS_REDIS}, Predictive prefetch: {self.enable_predictive_prefetch}"
        )

    def _initialize_cpu_affinity(self):
        """
        Initialize CPU affinity for market data provider threads
        """
        try:
            # Get current process
            process = psutil.Process()

            # Log available CPU cores
            available_cores = list(range(psutil.cpu_count()))
            logging.info(f"Available CPU cores: {available_cores}")

            # Validate core IDs to avoid issues
            valid_core_ids = [
                core_id
                for core_id in self.data_fetch_core_ids
                if core_id in available_cores
            ]
            if not valid_core_ids:
                valid_core_ids = [0]  # Fallback to first core

            logging.info(
                f"Setting CPU affinity for main market data thread to cores: {valid_core_ids}"
            )

            # Set process affinity
            if hasattr(process, "cpu_affinity"):
                process.cpu_affinity(valid_core_ids)
            else:
                logging.warning("CPU affinity not supported on this platform")

        except Exception as e:
            logging.error(f"Failed to set CPU affinity: {str(e)}")

    def _initialize_redis(self):
        """
        Initialize Redis client for market data distribution
        """
        try:
            # Create Redis client
            self.redis_client = RedisClient(self.config)

            # Initialize client
            if self.redis_client.initialize():
                logging.info("Redis client initialized for market data provider")
            else:
                logging.warning(
                    "Failed to initialize Redis client for market data provider"
                )
                self.redis_client = None
        except Exception as e:
            logging.error(f"Error initializing Redis client: {str(e)}")
            self.redis_client = None

    def _get_symbol_lock(self, symbol: str) -> threading.Lock:
        """
        Get or create a lock for a specific symbol to reduce contention.

        Args:
            symbol: Symbol to get lock for

        Returns:
            Lock for the symbol
        """
        with self.symbol_lock_mutex:
            if symbol not in self.symbol_locks:
                self.symbol_locks[symbol] = threading.Lock()
            return self.symbol_locks[symbol]

    def _is_circuit_breaker_engaged(self) -> bool:
        """
        Check if circuit breaker is engaged to prevent too many API calls during failures

        Returns:
            True if circuit breaker is engaged
        """
        with self.global_mutex:
            current_time = time.time()

            # If circuit breaker is engaged, check if it's time to reset
            if self.circuit_breaker_engaged:
                if current_time >= self.circuit_breaker_reset_time:
                    # Reset circuit breaker
                    self.circuit_breaker_engaged = False
                    self.consecutive_failures = 0
                    logging.info("Circuit breaker reset")
                    return False
                else:
                    # Circuit breaker still engaged
                    return True

            return False

    def _engage_circuit_breaker(self, reset_seconds: int = 60):
        """
        Engage circuit breaker due to API failures

        Args:
            reset_seconds: Number of seconds until circuit breaker resets
        """
        with self.global_mutex:
            self.circuit_breaker_engaged = True
            self.circuit_breaker_reset_time = time.time() + reset_seconds
            logging.warning(
                f"Circuit breaker engaged for {reset_seconds} seconds due to API failures"
            )

    def _check_rate_limits(self, response: requests.Response):
        """
        Check API rate limits from response headers and handle if approaching limits

        Args:
            response: API response to check headers from
        """
        with self.rate_limit_mutex:
            # Extract rate limit information from headers
            try:
                self.rate_limit_remaining = int(
                    response.headers.get("X-RateLimit-Remaining", "1000")
                )
                self.rate_limit_reset = int(
                    response.headers.get("X-RateLimit-Reset", "0")
                )

                # Log warning if approaching limit
                if self.rate_limit_remaining < self.rate_limit_threshold:
                    reset_time = max(0, self.rate_limit_reset - time.time())
                    logging.warning(
                        f"Approaching Polygon API rate limit: {self.rate_limit_remaining} calls remaining. "
                        f"Resets in {reset_time:.1f} seconds"
                    )

                # If at or below limit, sleep until reset
                if (
                    self.rate_limit_remaining <= 0
                    and self.rate_limit_reset > time.time()
                ):
                    sleep_time = max(0, self.rate_limit_reset - time.time())
                    logging.warning(
                        f"Rate limited. Sleeping for {sleep_time:.2f} seconds"
                    )
                    time.sleep(sleep_time)

            except (ValueError, TypeError):
                # If headers are missing or invalid, log warning
                logging.warning("Could not parse rate limit headers from API response")

    def _update_adaptive_timeout(self, response_time: float):
        """
        Update adaptive timeout based on recent response times

        Args:
            response_time: Latest response time in seconds
        """
        if not self.enable_adaptive_timeouts:
            return

        # Add response time to history
        self.response_times.append(response_time)

        # Calculate new timeout if we have enough data
        if len(self.response_times) >= 10:
            # Use percentile to avoid outliers
            timeout = np.percentile(self.response_times, self.timeout_percentile)

            # Add safety margin
            timeout = timeout * 1.5

            # Clamp between min and max timeout
            timeout = max(self.min_timeout, min(self.max_timeout, timeout))

            # Update timeouts
            self.read_timeout = timeout

            logging.debug(
                f"Updated adaptive timeout to {timeout:.2f}s based on recent response times"
            )

    def start_predictive_prefetch(self):
        """
        Start the predictive prefetch thread to proactively fetch data
        """
        if not self.enable_predictive_prefetch:
            return

        with self.prefetch_lock:
            if self.prefetch_running:
                return

            self.prefetch_running = True
            self.prefetch_thread = threading.Thread(
                target=self._predictive_prefetch_worker,
                daemon=True,
                name="PredictivePrefetch-Thread",
            )
            self.prefetch_thread.start()
            logging.info("Started predictive prefetch thread")

    def _predictive_prefetch_worker(self):
        """
        Worker thread that periodically prefetches data for high-priority symbols
        """
        # Set CPU affinity if enabled
        if self.use_cpu_affinity and len(self.data_fetch_core_ids) > 1:
            try:
                core_id = self.data_fetch_core_ids[1]  # Use second core for prefetch

                # Set thread affinity
                p = psutil.Process()
                if hasattr(p, "cpu_affinity"):
                    p.cpu_affinity([core_id])
                    logging.debug(f"Set prefetch thread affinity to core {core_id}")
            except Exception as e:
                logging.error(
                    f"Error setting thread affinity for prefetch worker: {str(e)}"
                )

        while self.prefetch_running:
            try:
                # Sleep first to avoid immediate prefetch
                time.sleep(self.prefetch_interval)

                # Get top symbols to prefetch
                symbols_to_prefetch = self._get_prefetch_symbols()

                if symbols_to_prefetch:
                    logging.debug(
                        f"Prefetching data for {len(symbols_to_prefetch)} symbols"
                    )
                    # Fetch data but don't wait for result
                    self._prefetch_polygon_data(symbols_to_prefetch)
            except Exception as e:
                logging.error(f"Error in prefetch worker: {str(e)}")

    def _get_prefetch_symbols(self, limit: int = 10) -> List[str]:
        """
        Get top symbols to prefetch based on probability and priority

        Args:
            limit: Maximum number of symbols to return

        Returns:
            List of symbols to prefetch
        """
        with self.prefetch_lock:
            # Sort by priority and probability
            sorted_symbols = sorted(
                self.prefetch_symbols.values(),
                key=lambda x: (x.priority, x.probability, x.frequency),
                reverse=True,
            )

            # Take top N symbols
            return [
                s.symbol
                for s in sorted_symbols[:limit]
                if s.symbol and s.probability > 0.2
            ]

    def update_symbol_probability(
        self, symbol: str, probability: float, priority: int = 0
    ):
        """
        Update the probability that a symbol will be needed soon

        Args:
            symbol: Symbol to update
            probability: Probability that the symbol will be needed (0-1)
            priority: Priority level (higher = more important)
        """
        if not self.enable_predictive_prefetch:
            return

        with self.prefetch_lock:
            # Get existing request or create new one
            request = self.prefetch_symbols[symbol]
            request.symbol = symbol

            # Update request
            current_time = time.time()
            if request.last_requested > 0:
                # Update frequency
                time_diff = current_time - request.last_requested
                if time_diff > 0:
                    # Exponential moving average of frequency
                    new_freq = 1.0 / time_diff
                    request.frequency = 0.8 * request.frequency + 0.2 * new_freq

            # Update probability and timestamp
            request.probability = max(request.probability, probability)
            request.last_requested = current_time
            request.priority = max(request.priority, priority)

            # Ensure prefetch thread is running
            if not self.prefetch_running:
                self.start_predictive_prefetch()

    def _prefetch_polygon_data(self, symbols: List[str]) -> None:
        """
        Prefetch data from Polygon.io without waiting for results

        Args:
            symbols: List of symbols to prefetch
        """
        # Check if we have enough API calls remaining
        with self.rate_limit_mutex:
            if self.rate_limit_remaining <= self.rate_limit_threshold * 2:
                # Skip prefetching if approaching rate limit
                return

        # Submit prefetch task to thread pool
        self.thread_pool.submit(self._fetch_polygon_data, symbols)

    def fetch_and_process_market_data(
        self, symbols: List[str]
    ) -> Dict[str, MarketSnapshot]:
        """
        Fetch market data for specified symbols and process it for the HFT model

        Args:
            symbols: List of symbols to fetch data for

        Returns:
            Dictionary mapping symbols to processed market snapshots
        """
        # Update symbol probabilities for prefetching
        for symbol in symbols:
            self.update_symbol_probability(symbol, 1.0, 1)

        # Skip if circuit breaker is engaged
        if self._is_circuit_breaker_engaged():
            logging.warning(
                "Circuit breaker engaged. Returning cached data or empty results."
            )
            # Return cached data if available, otherwise empty dict
            return self._get_cached_snapshots(symbols)

        start_time = time.time()

        # Fetch raw market data
        raw_data = self._fetch_polygon_data(symbols)

        # Update success/failure metrics
        if not raw_data:
            with self.global_mutex:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 3:
                    # Engage circuit breaker after 3 consecutive failures
                    self._engage_circuit_breaker(reset_seconds=60)

            # Return cached data if available
            return self._get_cached_snapshots(symbols)
        else:
            with self.global_mutex:
                self.last_successful_fetch = time.time()
                self.consecutive_failures = 0

        # Process data for each symbol in parallel
        futures = {}
        for symbol, ticker_data in raw_data.items():
            futures[symbol] = self.thread_pool.submit(
                self._process_symbol_data, symbol, ticker_data
            )

        # Collect results
        processed_data = {}
        for symbol, future in futures.items():
            try:
                processed_data[symbol] = future.result()
            except Exception as e:
                logging.error(f"Error processing symbol {symbol}: {e}")

        # Cache the snapshots in Redis if enabled
        self._cache_snapshots(processed_data)

        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to ms

        # Update adaptive timeout
        self._update_adaptive_timeout(end_time - start_time)

        logging.info(
            f"Fetched and processed data for {len(processed_data)} symbols in {processing_time:.2f}ms"
        )

        return processed_data

    def _get_cached_snapshots(self, symbols: List[str]) -> Dict[str, MarketSnapshot]:
        """
        Get cached snapshots for symbols from Redis or memory

        Args:
            symbols: List of symbols to get data for

        Returns:
            Dictionary mapping symbols to market snapshots
        """
        result = {}

        # Try Redis first if available
        if self.redis_client:
            for symbol in symbols:
                try:
                    snapshot_data = self.redis_client.get(f"market_snapshot:{symbol}")
                    if snapshot_data:
                        # Deserialize the snapshot
                        snapshot = json_fast.loads(snapshot_data)
                        result[symbol] = MarketSnapshot(**snapshot)
                except Exception as e:
                    logging.warning(
                        f"Error getting cached snapshot for {symbol} from Redis: {e}"
                    )

        # Fall back to local cache for any missing symbols
        missing_symbols = [s for s in symbols if s not in result]
        if missing_symbols:
            for symbol in missing_symbols:
                state = self.indicator_states.get(symbol)
                if state and state.last_update > 0:
                    # Create snapshot from state
                    snapshot = MarketSnapshot(
                        symbol=symbol,
                        timestamp=state.last_update,
                        last_price=state.last_price or 0.0,
                        rsi_14=state.rsi_value,
                        macd=state.macd_value,
                        bb_upper=state.bb_upper,
                        bb_middle=state.bb_middle,
                        bb_lower=state.bb_lower,
                    )
                    result[symbol] = snapshot

        return result

    def _cache_snapshots(self, snapshots: Dict[str, MarketSnapshot]):
        """
        Cache market snapshots in Redis for distribution

        Args:
            snapshots: Dictionary mapping symbols to market snapshots
        """
        if not self.redis_client:
            return

        # Cache each snapshot
        for symbol, snapshot in snapshots.items():
            try:
                # Serialize the snapshot
                snapshot_dict = snapshot.to_dict()

                # Store in Redis with expiry
                self.redis_client.set(
                    f"market_snapshot:{symbol}",
                    json_fast.dumps(snapshot_dict),
                    expire_seconds=300,  # 5-minute expiry
                )

            except Exception as e:
                logging.warning(f"Error caching snapshot for {symbol} in Redis: {e}")

    def _fetch_polygon_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch data from Polygon.io with optimized data handling

        Args:
            symbols: List of symbols to fetch data for

        Returns:
            Dictionary mapping symbols to raw ticker data
        """
        # Check if we have enough API calls remaining
        with self.rate_limit_mutex:
            if self.rate_limit_remaining <= self.rate_limit_threshold:
                # If close to limit, check if reset time is near
                if (
                    self.rate_limit_reset > 0
                    and self.rate_limit_reset < time.time() + 5
                ):
                    # Reset time is less than 5 seconds away, wait for it
                    sleep_time = max(0, self.rate_limit_reset - time.time())
                    logging.info(f"Waiting {sleep_time:.2f}s for rate limit to reset")
                    time.sleep(sleep_time)

        fetch_start_time = time.time()

        # Build URL
        url = f"{self.polygon_base_url}/v2/snapshot/locale/us/markets/stocks/tickers"
        params = {"apiKey": self.polygon_api_key, "tickers": ",".join(symbols)}

        try:
            # Get current adaptive timeout values
            current_connect_timeout = self.connect_timeout
            current_read_timeout = self.read_timeout

            # Perform request with timeout
            response = self.session.get(
                url,
                params=params,
                timeout=(current_connect_timeout, current_read_timeout),
            )

            # Record response time
            response_time = time.time() - fetch_start_time

            # Check and handle rate limits
            self._check_rate_limits(response)

            # Raise for status
            response.raise_for_status()

            # Parse JSON response using fast JSON parser
            response_json = json_fast.loads(response.text)

            # Check status
            if response_json.get("status") != "OK":
                error_msg = f"API error: {response_json.get('status')}"
                logging.error(error_msg)
                return {}

            # Extract tickers and create a dictionary mapping symbols to ticker data
            tickers = response_json.get("tickers", [])
            ticker_data = {}
            for ticker in tickers:
                symbol = ticker.get("ticker", "")
                if symbol:
                    ticker_data[symbol] = ticker

            # Update adaptive timeout based on response time
            self._update_adaptive_timeout(response_time)

            return ticker_data

        except requests.exceptions.Timeout:
            logging.error("Timeout while fetching data from Polygon")
            # Double timeout for next attempt
            if self.enable_adaptive_timeouts:
                self.read_timeout = min(self.read_timeout * 2, self.max_timeout)
                logging.info(
                    f"Increased read timeout to {self.read_timeout:.2f}s after timeout"
                )
            return {}
        except requests.exceptions.ConnectionError:
            logging.error("Connection error while fetching data from Polygon")
            return {}
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logging.error("Rate limit exceeded for Polygon API")
                # Engage circuit breaker for rate limit
                self._engage_circuit_breaker(reset_seconds=60)
            else:
                logging.error(f"HTTP error while fetching data from Polygon: {e}")
            return {}
        except Exception as e:
            logging.error(f"Error fetching data from Polygon: {e}")
            return {}

    def _process_symbol_data(
        self, symbol: str, ticker_data: Dict[str, Any]
    ) -> MarketSnapshot:
        """
        Process ticker data for a single symbol

        Args:
            symbol: Symbol to process
            ticker_data: Raw ticker data from Polygon API

        Returns:
            Processed market snapshot
        """
        # Get symbol-specific lock
        symbol_lock = self._get_symbol_lock(symbol)

        # Create market snapshot
        snapshot = MarketSnapshot(symbol=symbol)

        # Set timestamp
        if "lastTrade" in ticker_data and ticker_data["lastTrade"]:
            snapshot.timestamp = ticker_data["lastTrade"].get(
                "t", int(time.time() * 1000)
            )
        else:
            snapshot.timestamp = int(time.time() * 1000)

        # Set price data
        if "lastTrade" in ticker_data and ticker_data["lastTrade"]:
            snapshot.last_price = ticker_data["lastTrade"].get("p", 0.0)

        if "lastQuote" in ticker_data and ticker_data["lastQuote"]:
            snapshot.bid_price = ticker_data["lastQuote"].get("p", 0.0)
            snapshot.ask_price = ticker_data["lastQuote"].get("P", 0.0)
            snapshot.bid_ask_spread = snapshot.ask_price - snapshot.bid_price

        if "day" in ticker_data and ticker_data["day"]:
            snapshot.high_price = ticker_data["day"].get("h", 0.0)
            snapshot.low_price = ticker_data["day"].get("l", 0.0)
            snapshot.volume = ticker_data["day"].get("v", 0)
            snapshot.vwap = ticker_data["day"].get("vw", 0.0)

        # Update indicators incrementally with lock to prevent race conditions
        with symbol_lock:
            # Update technical indicators incrementally
            self._update_indicators_incremental(symbol, snapshot)

        # Allocate shared GPU memory if available
        if self.shared_memory_pool and self.use_gpu:
            try:
                # Convert features to numpy array
                features = snapshot.to_feature_array()

                # Allocate in shared memory
                mem_handle = self.shared_memory_pool.allocate(
                    features.shape, np.float32
                )

                # Copy data to shared memory
                if mem_handle:
                    # Store handle in snapshot for later use
                    snapshot.gpu_memory_handle = mem_handle
            except Exception as e:
                logging.warning(f"Error allocating shared memory for {symbol}: {e}")
                snapshot.gpu_memory_handle = None

        return snapshot

    def _get_or_create_indicator_state(self, symbol: str) -> IndicatorState:
        """
        Get or create indicator state for a symbol

        Args:
            symbol: Symbol to get state for

        Returns:
            Indicator state for the symbol
        """
        if symbol not in self.indicator_states:
            self.indicator_states[symbol] = IndicatorState()
        return self.indicator_states[symbol]

    def _update_indicators_incremental(self, symbol: str, snapshot: MarketSnapshot):
        """
        Update technical indicators incrementally for a symbol

        Args:
            symbol: Symbol to update indicators for
            snapshot: Market snapshot to update
        """
        # Get indicator state
        state = self._get_or_create_indicator_state(symbol)

        # Skip if price is zero or unchanged since last update
        if snapshot.last_price == 0 or (
            state.last_price is not None and snapshot.last_price == state.last_price
        ):
            return

        # Update last update timestamp
        state.last_update = snapshot.timestamp

        # Add price to history
        state.close_prices.append(snapshot.last_price)
        state.timestamps.append(snapshot.timestamp)

        # Add volume to history
        state.volume_data.append(snapshot.volume)

        # Calculate price change for volatility
        if state.last_price is not None:
            price_change_pct = (
                snapshot.last_price - state.last_price
            ) / state.last_price
            state.price_changes.append(price_change_pct)

        # Add bid-ask spread to history
        if snapshot.bid_ask_spread > 0:
            state.bid_ask_spreads.append(snapshot.bid_ask_spread)

        # Update RSI
        snapshot.rsi_14 = self._calculate_rsi_incremental(state, snapshot.last_price)

        # Update MACD
        macd_results = self._calculate_macd_incremental(state, snapshot.last_price)
        snapshot.macd = macd_results

        # Update Bollinger Bands
        bb_results = self._calculate_bollinger_bands_incremental(
            state, snapshot.last_price
        )
        snapshot.bb_upper = bb_results["upper"]
        snapshot.bb_middle = bb_results["middle"]
        snapshot.bb_lower = bb_results["lower"]

        # Calculate volume acceleration
        if len(state.volume_data) >= 5:
            vol_mean_prev = sum(list(state.volume_data)[-5:-1]) / 4
            if vol_mean_prev > 0:
                vol_change = (state.volume_data[-1] / vol_mean_prev) - 1.0
                snapshot.volume_acceleration = vol_change * 100.0

        # Calculate price change (5-minute)
        if len(state.close_prices) >= 5:
            snapshot.price_change_5m = (
                (state.close_prices[-1] / state.close_prices[-5]) - 1.0
            ) * 100.0

        # Calculate momentum (1-minute)
        if len(state.close_prices) >= 10:
            snapshot.momentum_1m = state.close_prices[-1] - state.close_prices[-10]

        # Calculate intraday volatility (new)
        if len(state.price_changes) >= 5:
            snapshot.intraday_volatility = np.std(state.price_changes) * 100.0

        # Calculate bid-ask volatility (new)
        if len(state.bid_ask_spreads) >= 5:
            snapshot.bid_ask_volatility = np.std(state.bid_ask_spreads) * 100.0

        # Determine market condition flags (new)
        # Volatility flag
        if snapshot.intraday_volatility > 0.5:  # 0.5% volatility threshold
            snapshot.is_volatile = True

        # Trend direction
        if len(state.close_prices) >= 20:
            short_sma = sum(state.close_prices[-10:]) / 10
            long_sma = sum(state.close_prices[-20:]) / 20

            if short_sma > long_sma * 1.005:  # 0.5% uptrend threshold
                snapshot.trend_direction = 1
            elif short_sma < long_sma * 0.995:  # 0.5% downtrend threshold
                snapshot.trend_direction = -1
            else:
                snapshot.trend_direction = 0

        # Update state with current price
        state.last_price = snapshot.last_price

    def _calculate_rsi_incremental(
        self, state: IndicatorState, new_price: float
    ) -> float:
        """
        Calculate RSI incrementally for a symbol

        Args:
            state: Indicator state for the symbol
            new_price: Latest price

        Returns:
            Updated RSI value
        """
        # Calculate delta if last price exists
        if state.last_price is not None:
            delta = new_price - state.last_price

            # Calculate gain/loss
            gain = max(0, delta)
            loss = max(0, -delta)

            # Add to history
            state.rsi_gains.append(gain)
            state.rsi_losses.append(loss)

            # Calculate RSI once we have enough data
            if len(state.rsi_gains) >= state.rsi_period:
                avg_gain = sum(state.rsi_gains) / state.rsi_period
                avg_loss = sum(state.rsi_losses) / state.rsi_period

                # Calculate RS
                if avg_loss == 0:
                    rs = float("inf")
                else:
                    rs = avg_gain / avg_loss

                # Calculate RSI
                state.rsi_value = 100.0 - (100.0 / (1.0 + rs))

        return state.rsi_value

    def _calculate_macd_incremental(
        self, state: IndicatorState, new_price: float
    ) -> float:
        """
        Calculate MACD incrementally for a symbol

        Args:
            state: Indicator state for the symbol
            new_price: Latest price

        Returns:
            Updated MACD value
        """
        fast_period = 12
        slow_period = 26
        signal_period = 9

        # Initialize EMAs if needed
        if state.fast_ema is None:
            state.fast_ema = new_price
        if state.slow_ema is None:
            state.slow_ema = new_price
        if state.signal_ema is None:
            state.signal_ema = 0

        # Update EMAs
        alpha_fast = 2.0 / (fast_period + 1)
        alpha_slow = 2.0 / (slow_period + 1)
        alpha_signal = 2.0 / (signal_period + 1)

        # Exponential Moving Averages
        state.fast_ema = new_price * alpha_fast + state.fast_ema * (1 - alpha_fast)
        state.slow_ema = new_price * alpha_slow + state.slow_ema * (1 - alpha_slow)

        # Calculate MACD line
        macd_line = state.fast_ema - state.slow_ema

        # Update signal line
        state.signal_ema = macd_line * alpha_signal + state.signal_ema * (
            1 - alpha_signal
        )

        # MACD histogram (not used but kept for reference)
        # histogram = macd_line - state.signal_ema

        # Store MACD value
        state.macd_value = macd_line

        return macd_line

    def _calculate_bollinger_bands_incremental(
        self, state: IndicatorState, new_price: float
    ) -> Dict[str, float]:
        """
        Calculate Bollinger Bands incrementally for a symbol

        Args:
            state: Indicator state for the symbol
            new_price: Latest price

        Returns:
            Dictionary with 'upper', 'middle', and 'lower' band values
        """
        window = 20
        num_std = 2

        # Need at least window prices for a meaningful result
        prices = list(state.close_prices)

        if len(prices) >= window:
            # Calculate middle band (SMA)
            prices_window = prices[-window:]
            middle_band = sum(prices_window) / window

            # Calculate standard deviation
            squared_diff = [(p - middle_band) ** 2 for p in prices_window]
            variance = sum(squared_diff) / window
            std_dev = variance**0.5

            # Calculate upper and lower bands
            upper_band = middle_band + (std_dev * num_std)
            lower_band = middle_band - (std_dev * num_std)

            # Update state
            state.bb_upper = upper_band
            state.bb_middle = middle_band
            state.bb_lower = lower_band

        # Return values
        return {
            "upper": state.bb_upper,
            "middle": state.bb_middle,
            "lower": state.bb_lower,
        }

    def _cleanup_stale_indicator_state(self, max_age_minutes: int = 60):
        """
        Clean up stale indicator state to prevent memory leaks

        Args:
            max_age_minutes: Maximum age in minutes before state is considered stale
        """
        with self.global_mutex:
            current_time = time.time() * 1000  # Convert to ms
            max_age_ms = max_age_minutes * 60 * 1000

            # Find stale symbols
            stale_symbols = []
            for symbol, state in self.indicator_states.items():
                if current_time - state.last_update > max_age_ms:
                    stale_symbols.append(symbol)

            # Remove stale state
            for symbol in stale_symbols:
                del self.indicator_states[symbol]

                # Also remove from symbol locks
                with self.symbol_lock_mutex:
                    if symbol in self.symbol_locks:
                        del self.symbol_locks[symbol]

                # Also remove from prefetch symbols
                with self.prefetch_lock:
                    if symbol in self.prefetch_symbols:
                        del self.prefetch_symbols[symbol]

            if stale_symbols:
                logging.info(
                    f"Cleaned up indicator state for {len(stale_symbols)} stale symbols"
                )

    def prepare_model_features(
        self, symbols: List[str], reuse_buffers: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare features for model inference with memory efficiency as priority.

        Args:
            symbols: List of symbols to process
            reuse_buffers: Whether to reuse existing memory buffers

        Returns:
            Dictionary with memory-optimized feature arrays
        """
        # Get market data
        market_snapshots = self.fetch_and_process_market_data(symbols)

        # Get model configuration for normalization
        model_config = self.config.get("ml", {})
        stats_path = model_config.get("stats_path", "models/feature_stats.json")

        # Load normalization stats if available
        normalization_stats = None
        try:
            with open(stats_path, "r") as f:
                normalization_stats = json_fast.loads(f.read())
        except Exception as e:
            logging.warning(
                f"Could not load normalization stats from {stats_path}: {e}"
            )

        # Prepare feature arrays with buffer reuse
        feature_arrays = {}
        for symbol, snapshot in market_snapshots.items():
            # Check if we can reuse an existing buffer
            if reuse_buffers and self.shared_memory_pool:
                # Get or allocate buffer
                buffer = self.get_or_allocate_buffer(
                    symbol=symbol,
                    shape=(1, 18),  # Shape for single sample with 18 features
                    dtype=np.float32,
                )

                # Extract features to buffer in-place
                features = snapshot.to_feature_array()

                # Copy to buffer (in-place if possible)
                if isinstance(buffer, np.ndarray):
                    np.copyto(buffer[0], features)
                else:  # CuPy array
                    cp.copyto(buffer[0], cp.asarray(features))

                # Apply normalization in-place if stats available
                if normalization_stats:
                    self.normalize_features(buffer, normalization_stats, in_place=True)

                feature_arrays[symbol] = buffer
            else:
                # Extract features without buffer reuse
                features = snapshot.to_feature_array().reshape(1, -1)

                # Apply normalization if stats available
                if normalization_stats:
                    features = self.normalize_features(
                        features, normalization_stats, in_place=False
                    )

                # Transfer to GPU if needed
                if self.use_gpu and not isinstance(features, cp.ndarray):
                    features = cp.asarray(features)

                feature_arrays[symbol] = features

        return feature_arrays

    def normalize_features(
        self, features: np.ndarray, stats: Dict[str, Any], in_place: bool = True
    ) -> np.ndarray:
        """
        Normalize features using the model's statistics.

        Args:
            features: Raw feature array
            stats: Dictionary with normalization parameters
            in_place: Whether to perform normalization in-place

        Returns:
            Normalized features
        """
        # Extract normalization parameters
        means = stats.get("means", None)
        stds = stats.get("stds", None)

        if means is None or stds is None:
            logging.warning("Missing normalization parameters, skipping normalization")
            return features

        # Convert to appropriate array type
        if isinstance(features, cp.ndarray):
            means_array = cp.array(means, dtype=features.dtype)
            stds_array = cp.array(stds, dtype=features.dtype)
        else:
            means_array = np.array(means, dtype=features.dtype)
            stds_array = np.array(stds, dtype=features.dtype)

        # Handle zero standard deviations to avoid division by zero
        stds_array = np.where(stds_array < 1e-10, 1.0, stds_array)

        # Perform normalization
        if in_place:
            # In-place normalization (memory efficient)
            features -= means_array
            features /= stds_array
            return features
        else:
            # Create new array (less memory efficient)
            return (features - means_array) / stds_array

    def get_or_allocate_buffer(
        self, symbol: str, shape: tuple, dtype: np.dtype
    ) -> Union[np.ndarray, Any]:
        """
        Get an existing buffer or allocate a new one in the shared pool.

        Args:
            symbol: Symbol identifier
            shape: Required buffer shape
            dtype: Data type

        Returns:
            Memory buffer from the pool
        """
        if not self.shared_memory_pool:
            # If no shared pool, allocate regular array
            if self.use_gpu and HAS_GPU:
                return cp.zeros(shape, dtype=dtype)
            else:
                return np.zeros(shape, dtype=dtype)

        # Try to get existing buffer from pool
        buffer_key = f"{symbol}_features"
        buffer = self.shared_memory_pool.get_buffer(buffer_key)

        if buffer is not None:
            # Check if shape and dtype match
            if buffer.shape == shape and buffer.dtype == dtype:
                # Clear buffer (zero out)
                if isinstance(buffer, np.ndarray):
                    buffer.fill(0)
                else:  # CuPy array
                    buffer.fill(0)
                return buffer
            else:
                # Release old buffer if shape/dtype doesn't match
                self.shared_memory_pool.release_buffer(buffer_key)

        # Allocate new buffer
        new_buffer = self.shared_memory_pool.allocate(shape, dtype, buffer_key)
        return new_buffer

    def get_model_ready_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get data in a format directly consumable by the model with optimal memory usage.

        Args:
            symbols: List of symbols to get data for

        Returns:
            Dictionary with memory-optimized arrays ready for model consumption
        """
        # Get memory-optimized features
        features = self.prepare_model_features(symbols, reuse_buffers=True)

        # Format in memory layout matching model expectations
        model_config = self.config.get("ml", {})
        use_soa_layout = model_config.get("use_soa_layout", False)

        if use_soa_layout:
            # Structure of Arrays (SOA) layout - features are grouped by type
            # This is more efficient for some models and hardware
            soa_features = {}
            for symbol, feature_array in features.items():
                # Reshape to ensure we have a 2D array
                if len(feature_array.shape) == 1:
                    feature_array = feature_array.reshape(1, -1)

                # Store in SOA format
                soa_features[symbol] = {
                    "data": feature_array,
                    "shape": feature_array.shape,
                    "dtype": feature_array.dtype,
                    "layout": "soa",
                }
            return soa_features
        else:
            # Array of Structures (AOS) layout - default
            return features

    def get_market_data_for_hft_model(
        self, symbols: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Get market data specifically formatted for the HFT model

        Args:
            symbols: List of symbols to get data for

        Returns:
            Dictionary mapping symbols to feature arrays ready for model input
        """
        # Clean up stale state first
        self._cleanup_stale_indicator_state()

        # Update symbol probabilities for prefetching
        for symbol in symbols:
            self.update_symbol_probability(symbol, 1.0, 2)

        # Use the new optimized method for model-ready data
        return self.get_model_ready_data(symbols)

    def _fetch_market_caps(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch market capitalization data for a list of symbols

        Args:
            symbols: List of symbols to fetch market cap for

        Returns:
            Dictionary mapping symbols to market cap values
        """
        current_time = time.time()
        market_caps = {}

        # Check which symbols need updating
        symbols_to_fetch = []
        for symbol in symbols:
            # Use cached market cap if available and not expired
            cached_market_cap = self.market_cap_cache.get(symbol)
            last_update = self.market_cap_last_update.get(symbol, 0)

            if (
                cached_market_cap is not None
                and current_time - last_update < self.market_cap_cache_expiry
            ):
                market_caps[symbol] = cached_market_cap
            else:
                symbols_to_fetch.append(symbol)

        if not symbols_to_fetch:
            return market_caps

        # Use Polygon Reference API to fetch market cap data
        batch_size = 50  # Polygon typically limits to 50 symbols per request
        for i in range(0, len(symbols_to_fetch), batch_size):
            batch = symbols_to_fetch[i : i + batch_size]

            # Build URL
            url = f"{self.polygon_base_url}/v3/reference/tickers"
            params = {
                "apiKey": self.polygon_api_key,
                "tickers": ",".join(batch),
                "active": "true",
            }

            try:
                # Perform request with timeout
                response = self.session.get(
                    url,
                    params=params,
                    timeout=(self.connect_timeout, self.read_timeout),
                )

                # Check and handle rate limits
                self._check_rate_limits(response)

                # Raise for status
                response.raise_for_status()

                # Parse JSON response
                response_json = json_fast.loads(response.text)

                # Process results
                for ticker in response_json.get("results", []):
                    symbol = ticker.get("ticker", "")
                    market_cap = ticker.get("market_cap", None)

                    if symbol and market_cap is not None:
                        # Update cache
                        self.market_cap_cache.put(symbol, market_cap)
                        self.market_cap_last_update[symbol] = current_time

                        # Add to results
                        market_caps[symbol] = market_cap

            except Exception as e:
                logging.warning(f"Error fetching market cap data from Polygon: {e}")

        return market_caps

    def fetch_full_market_snapshot(
        self, include_otc: bool = False, apply_filters: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch a complete snapshot of the US stock market

        Args:
            include_otc: Whether to include OTC securities
            apply_filters: Whether to apply filtering criteria

        Returns:
            Complete market snapshot data
        """
        # Skip if circuit breaker is engaged
        if self._is_circuit_breaker_engaged():
            logging.warning("Circuit breaker engaged. Skipping full market snapshot.")
            return None

        # Build URL
        url = f"{self.polygon_base_url}/v2/snapshot/locale/us/markets/stocks/tickers"
        params = {
            "apiKey": self.polygon_api_key,
            "include_otc": str(include_otc).lower(),
        }

        logging.info(f"Fetching full market snapshot (include_otc={include_otc})...")

        try:
            # Perform request with timeout (use longer timeout for full snapshot)
            response = self.session.get(
                url,
                params=params,
                timeout=(self.connect_timeout, self.read_timeout * 2),
            )

            # Check and handle rate limits
            self._check_rate_limits(response)

            # Raise for status
            response.raise_for_status()

            # Parse JSON response using fast JSON parser
            response_json = json_fast.loads(response.text)

            # Check status
            if response_json.get("status") != "OK":
                error_msg = f"API error: {response_json.get('status')}"
                logging.error(error_msg)
                return None

            ticker_count = response_json.get("count", 0)
            logging.info(f"Successfully fetched snapshot for {ticker_count} tickers")

            # Apply filters if requested
            if apply_filters:
                response_json = self.filter_market_snapshot(response_json)

            # Update success metrics
            with self.global_mutex:
                self.last_successful_fetch = time.time()
                self.consecutive_failures = 0

            return response_json

        except requests.exceptions.Timeout:
            logging.error("Timeout while fetching full market snapshot")
            # Update failure metrics
            with self.global_mutex:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 3:
                    self._engage_circuit_breaker(
                        reset_seconds=120
                    )  # Longer reset for full snapshot
            return None
        except Exception as e:
            logging.error(f"Error fetching full market snapshot from Polygon: {e}")
            # Update failure metrics
            with self.global_mutex:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 3:
                    self._engage_circuit_breaker(reset_seconds=120)
            return None

    def filter_market_snapshot(self, snapshot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter market snapshot based on configured criteria

        Args:
            snapshot_data: Market snapshot data

        Returns:
            Filtered market snapshot data
        """
        if snapshot_data is None:
            return snapshot_data

        original_tickers = snapshot_data.get("tickers", [])

        # Filter counters
        below_min_price = 0
        above_max_price = 0
        below_min_volume = 0
        below_min_market_cap = 0
        missing_data = 0

        # Extract all symbols for market cap fetching
        symbols = [ticker.get("ticker", "") for ticker in original_tickers]
        market_caps = {}

        # Fetch market caps if min_market_cap filter is enabled
        if self.min_market_cap > 0:
            market_caps = self._fetch_market_caps(symbols)

        # Filter tickers based on criteria
        filtered_tickers = []

        for ticker in original_tickers:
            # Initialize filter flags
            price_ok = True
            volume_ok = True
            market_cap_ok = True

            # Extract data
            symbol = ticker.get("ticker", "UNKNOWN")
            last_price = None
            volume = None

            # Try to get price from lastTrade
            if "lastTrade" in ticker and ticker["lastTrade"]:
                last_price = ticker["lastTrade"].get("p", None)

            # If not found, try to get from day data
            if last_price is None and "day" in ticker and ticker["day"]:
                last_price = ticker["day"].get("c", None)

            # Get volume from day data
            if "day" in ticker and ticker["day"]:
                volume = ticker["day"].get("v", None)

            # Skip if no price data available
            if last_price is None:
                missing_data += 1
                continue

            # Check price against thresholds
            if self.min_price > 0 and last_price < self.min_price:
                below_min_price += 1
                price_ok = False

            if self.max_price > 0 and last_price > self.max_price:
                above_max_price += 1
                price_ok = False

            # Check volume
            if self.min_volume > 0 and (volume is None or volume < self.min_volume):
                below_min_volume += 1
                volume_ok = False

            # Check market cap
            market_cap = market_caps.get(symbol)
            if self.min_market_cap > 0 and (
                market_cap is None or market_cap < self.min_market_cap
            ):
                below_min_market_cap += 1
                market_cap_ok = False

            # If all criteria are met, keep the ticker
            if price_ok and volume_ok and market_cap_ok:
                filtered_tickers.append(ticker)

        # Update snapshot data with filtered tickers
        filtered_count = len(filtered_tickers)

        # Log filtering results
        if self.min_price > 0:
            logging.info(
                f"Filtered out {below_min_price} tickers below ${self.min_price}"
            )

        if self.max_price > 0:
            logging.info(
                f"Filtered out {above_max_price} tickers above ${self.max_price}"
            )

        if self.min_volume > 0:
            logging.info(
                f"Filtered out {below_min_volume} tickers with volume below {self.min_volume}"
            )

        if self.min_market_cap > 0:
            logging.info(
                f"Filtered out {below_min_market_cap} tickers with market cap below ${self.min_market_cap / 1000000:.1f}M"
            )

        if missing_data > 0:
            logging.info(f"Skipped {missing_data} tickers with missing price data")

        logging.info(f"Kept {filtered_count} tickers meeting all criteria")

        filtered_snapshot = snapshot_data.copy()
        filtered_snapshot["tickers"] = filtered_tickers
        filtered_snapshot["count"] = filtered_count

        return filtered_snapshot

    def get_market_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current market state

        Returns:
            Dictionary with market statistics
        """
        # Create statistics
        stats = {
            "symbols_tracked": len(self.indicator_states),
            "symbols_prefetched": 0,
            "cache_hit_rate": 0.0,
            "avg_response_time": 0.0,
            "current_timeout": self.read_timeout,
            "adaptive_timeout_enabled": self.enable_adaptive_timeouts,
            "predictive_prefetch_enabled": self.enable_predictive_prefetch,
            "rate_limit_remaining": self.rate_limit_remaining,
            "rate_limit_reset": self.rate_limit_reset,
            "circuit_breaker_engaged": self.circuit_breaker_engaged,
        }

        # Add prefetch stats
        with self.prefetch_lock:
            active_prefetch = [
                s
                for s in self.prefetch_symbols.values()
                if s.symbol and s.probability > 0.2
            ]
            stats["symbols_prefetched"] = len(active_prefetch)

            # Add top prefetched symbols
            top_symbols = sorted(
                active_prefetch,
                key=lambda x: (x.priority, x.probability, x.frequency),
                reverse=True,
            )[:5]

            stats["top_prefetched_symbols"] = [
                {
                    "symbol": s.symbol,
                    "probability": s.probability,
                    "frequency": s.frequency,
                    "priority": s.priority,
                }
                for s in top_symbols
            ]

        # Add response time stats
        if self.response_times:
            stats["avg_response_time"] = sum(self.response_times) / len(
                self.response_times
            )
            stats["max_response_time"] = max(self.response_times)
            stats["min_response_time"] = min(self.response_times)
            stats["p95_response_time"] = (
                np.percentile(self.response_times, 95)
                if len(self.response_times) >= 20
                else None
            )

        return stats

    def cleanup(self):
        """
        Clean up resources used by the market data provider

        This method should be called when the provider is no longer needed
        to ensure proper release of resources.
        """
        logging.info("Cleaning up PolygonMarketDataProvider resources")

        # Stop prefetch thread
        with self.prefetch_lock:
            self.prefetch_running = False

        if self.prefetch_thread:
            logging.debug("Waiting for prefetch thread to terminate")
            if self.prefetch_thread.is_alive():
                try:
                    self.prefetch_thread.join(timeout=2.0)
                except Exception as e:
                    logging.warning(f"Error waiting for prefetch thread: {e}")

        # Shutdown thread pool
        if hasattr(self, "thread_pool"):
            self.thread_pool.shutdown(wait=True)
            logging.debug("Thread pool shut down")

        # Close HTTP session
        if hasattr(self, "session"):
            self.session.close()
            logging.debug("HTTP session closed")

        # Clean up Redis client
        if hasattr(self, "redis_client") and self.redis_client is not None:
            try:
                self.redis_client.close()
                logging.debug("Redis client closed")
            except Exception as e:
                logging.warning(f"Error closing Redis client: {e}")

        # Clean up shared memory pool
        if self.shared_memory_pool:
            try:
                self.shared_memory_pool.cleanup()
                logging.debug("Shared memory pool cleaned up")
            except Exception as e:
                logging.warning(f"Error cleaning up shared memory pool: {e}")

        # Clean up cached data and state
        self.indicator_states.clear()
        self.market_cap_cache = LRUCache(max_size=1)  # Reset to minimal size
        self.market_cap_last_update.clear()
        self.symbol_locks.clear()

        # Force garbage collection to ensure memory is released
        if HAS_GPU:
            try:
                # Clear any cached memory in CuPy
                cp.get_default_memory_pool().free_all_blocks()
                if hasattr(cp.cuda, "set_allocator"):
                    cp.cuda.set_allocator(None)  # Reset allocator
                logging.debug("CuPy memory pools cleared")
            except Exception as e:
                logging.warning(f"Error clearing CuPy memory pools: {e}")

        import gc

        gc.collect()
        logging.info("Memory pools and caches cleared")
