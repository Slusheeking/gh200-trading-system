"""
Polygon REST API client implementation

This module provides a client for the Polygon.io REST API to fetch market data.
It handles authentication, data fetching, and parsing of API responses.
"""

import json
import time
import threading
import requests
import os
import backoff
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Callable, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Use orjson if available (faster), otherwise fallback to standard json
try:
    import orjson as json_fast  # Faster JSON parsing
except ImportError:
    try:
        import ujson as json_fast  # Faster JSON parsing
    except ImportError:
        import json as json_fast
        logging.warning("Using standard json module. Consider installing orjson or ujson for better performance.")

from requests_futures.sessions import FuturesSession

class ParsedMarketData:
    """Class to store parsed market data"""
    
    @dataclass
    class SymbolData:
        """Class to store data for a single symbol"""
        symbol: str = ""
        last_price: float = 0.0
        bid_price: float = 0.0
        ask_price: float = 0.0
        bid_ask_spread: float = 0.0
        open_price: float = 0.0
        high_price: float = 0.0
        low_price: float = 0.0
        volume: int = 0
        vwap: float = 0.0
        prev_close: float = 0.0
        timestamp: int = 0
        
        # Technical indicators
        rsi_14: float = 50.0
        macd: float = 0.0
        macd_signal: float = 0.0
        macd_histogram: float = 0.0
        bb_upper: float = 0.0
        bb_middle: float = 0.0
        bb_lower: float = 0.0
        atr: float = 0.0
        
        # Additional indicators for fast path
        avg_volume: float = 0.0
        volume_acceleration: float = 0.0
        volume_spike: float = 0.0
        volume_profile_imbalance: float = 0.0
        
        # Price dynamics
        price_change_1m: float = 0.0
        price_change_5m: float = 0.0
        momentum_1m: float = 0.0
        price_trend_strength: float = 0.0
        volume_trend_strength: float = 0.0
        volatility_ratio: float = 0.0
        volatility_change: float = 0.0
        
        # Market context
        market_regime: float = 0.0
        sector_performance: float = 0.0
        relative_strength: float = 0.0
        support_resistance_proximity: float = 0.0
        sma_cross_signal: float = 0.0
        
        # Order book metrics
        bid_ask_imbalance: float = 0.0
        bid_ask_spread_change: float = 0.0
        trade_count: int = 0
        avg_trade_size: float = 0.0
        large_trade_ratio: float = 0.0
    
    def __init__(self):
        self.symbol_data: Dict[str, ParsedMarketData.SymbolData] = {}
        self.timestamp: int = 0
        self.num_trades_processed: int = 0
        self.num_quotes_processed: int = 0


class PolygonRestAPI:
    """REST API client for Polygon.io with high-performance optimizations"""
    
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
        self.polygon_api_key = os.getenv("POLYGON_API_KEY", "")
        self.polygon_base_url = os.getenv("POLYGON_BASE_URL", "https://api.polygon.io")
        
        # Get API keys from config as fallback
        polygon_config = config.get("data_sources", {}).get("polygon", {})
        if polygon_config.get("enabled", False):
            # Only use config values if env vars aren't set
            if not self.polygon_api_key:
                self.polygon_api_key = polygon_config.get("api_key", "")
            if not self.polygon_base_url or self.polygon_base_url == "https://api.polygon.io":
                if polygon_config.get("base_url"):
                    self.polygon_base_url = polygon_config.get("base_url")
        
        # Thread management
        self.fetch_thread = None
        self.thread_affinity = -1
        self.running = False
        self.mutex = threading.Lock()
        self.cv = threading.Condition(self.mutex)
        self.fetch_interval_ms = 1000
        
        # Callback
        self.data_callback = None
        
        # Performance settings from config
        perf_config = config.get("performance", {})
        self.max_workers = perf_config.get("processor_threads", min(32, os.cpu_count() * 2))
        self.batch_size = perf_config.get("websocket_parser_batch_size", 1000)
        self.use_numpy = True
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="PolygonREST-Worker"
        )
        
        # Configure HTTP session with retries and timeouts
        self.session = requests.Session()
        
        # Configure retry strategy from system.yaml
        polygon_config = config.get("data_sources", {}).get("polygon", {})
        retry_config = polygon_config.get("http", {}).get("retry", {})
        max_retries = retry_config.get("max_retries", 3)
        backoff_factor = retry_config.get("backoff_factor", 0.3)
        status_forcelist = retry_config.get("status_forcelist", [429, 500, 502, 503, 504])
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=["GET"]
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
        timeout_config = polygon_config.get("http", {}).get("timeout", {})
        self.connect_timeout = timeout_config.get("connect", 3.0)
        self.read_timeout = timeout_config.get("read", 10.0)
        
        # Create a FuturesSession for asynchronous requests
        self.futures_session = FuturesSession(
            session=self.session,
            max_workers=self.max_workers
        )
        
        # Pre-allocate buffers for data processing
        self.symbol_buffer_size = polygon_config.get("max_symbols", 1000)
        
        logging.info(f"Configured high-performance HTTP client with max_workers={self.max_workers}, " +
                    f"max_retries={max_retries}, backoff_factor={backoff_factor}, " +
                    f"connect_timeout={self.connect_timeout}s, read_timeout={self.read_timeout}s")
    
    def initialize(self):
        """Initialize the REST client"""
        logging.info("Initializing Polygon REST API client")
        
        # Check if API keys are available
        if not self.polygon_api_key:
            logging.warning("No API keys configured for Polygon REST client")
    
    def fetch_full_market_snapshot(self) -> Future:
        """
        Fetch a full market snapshot
        
        Returns:
            Future: Future object that will contain the parsed market data
        """
        future = Future()
        
        def fetch_task():
            data = ParsedMarketData()
            
            try:
                # Use Polygon.io API if available
                if self.polygon_api_key:
                    self._fetch_polygon_data(data)
                    logging.info(f"Successfully fetched full market snapshot for {len(data.symbol_data)} symbols")
                else:
                    logging.error("No API keys available for fetching market data")
            except Exception as e:
                logging.error(f"Error fetching full market snapshot: {str(e)}")
                import traceback
                logging.debug(traceback.format_exc())
            
            future.set_result(data)
        
        thread = threading.Thread(target=fetch_task, name="PolygonRestAPI-FullSnapshot")
        thread.daemon = True
        thread.start()
        return future
    
    def fetch_symbol_data(self, symbols: List[str]) -> Future:
        """
        Fetch data for specific symbols
        
        Args:
            symbols: List of symbols to fetch data for
            
        Returns:
            Future: Future object that will contain the parsed market data
        """
        future = Future()
        
        def fetch_task():
            data = ParsedMarketData()
            
            try:
                # Use Polygon.io API if available
                if self.polygon_api_key:
                    self._fetch_polygon_data(data, symbols)
                    logging.info(f"Successfully fetched data for {len(data.symbol_data)}/{len(symbols)} symbols")
                else:
                    logging.error("No API keys available for fetching symbol data")
            except Exception as e:
                logging.error(f"Error fetching symbol data: {str(e)}")
                import traceback
                logging.debug(traceback.format_exc())
            
            future.set_result(data)
        
        thread = threading.Thread(target=fetch_task, name="PolygonRestAPI-SymbolData")
        thread.daemon = True
        thread.start()
        return future
    
    def set_thread_affinity(self, core_id: int):
        """
        Set thread affinity for the fetch thread
        
        Args:
            core_id: CPU core ID to pin the thread to
        """
        self.thread_affinity = core_id
        
        # Apply to running thread if exists
        if self.fetch_thread and self.fetch_thread.is_alive():
            # Note: Python doesn't have a direct equivalent to pinThreadToCore
            # This would require platform-specific code
            logging.debug(f"Thread affinity set to core {core_id} (not implemented)")
    
    def set_data_callback(self, callback: Callable[[ParsedMarketData], None]):
        """
        Set callback for data updates
        
        Args:
            callback: Callback function that takes a ParsedMarketData object
        """
        self.data_callback = callback
        logging.debug("Data callback set")
    
    def start_periodic_fetching(self, interval_ms: int):
        """
        Start periodically fetching market data
        
        Args:
            interval_ms: Interval between fetches in milliseconds
        """
        if self.running:
            logging.warning("Periodic fetching already running")
            return
        
        self.running = True
        self.fetch_interval_ms = interval_ms
        
        # Start fetch thread
        self.fetch_thread = threading.Thread(target=self._fetch_loop, name="PolygonRestAPI-FetchLoop")
        self.fetch_thread.daemon = True
        self.fetch_thread.start()
        
        logging.info(f"Started periodic fetching with interval {interval_ms}ms")
    
    def stop_periodic_fetching(self):
        """Stop periodically fetching market data"""
        if not self.running:
            logging.warning("Periodic fetching not running")
            return
        
        self.running = False
        with self.cv:
            self.cv.notify_all()
        
        if self.fetch_thread and self.fetch_thread.is_alive():
            self.fetch_thread.join(timeout=5.0)
            if self.fetch_thread.is_alive():
                logging.warning("Fetch thread did not terminate within timeout")
            else:
                logging.info("Periodic fetching stopped")
    
    def _fetch_loop(self):
        """Fetch loop for periodic fetching"""
        while self.running:
            # Fetch data
            data = ParsedMarketData()
            
            try:
                # Use Polygon.io API if available
                if self.polygon_api_key:
                    self._fetch_polygon_data(data)
                
                # Call callback if set
                if self.data_callback and data.symbol_data:
                    self.data_callback(data)
                    logging.debug(f"Data callback called with {len(data.symbol_data)} symbols")
            except Exception as e:
                logging.error(f"Error fetching market data: {str(e)}")
                import traceback
                logging.debug(traceback.format_exc())
            
            # Wait for next interval
            with self.cv:
                self.cv.wait(timeout=self.fetch_interval_ms / 1000.0)
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, json.JSONDecodeError),
        max_tries=3,
        giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and e.response.status_code < 500,
        factor=2,
        jitter=backoff.full_jitter
    )
    def _fetch_polygon_data(self, data: ParsedMarketData, symbols: Optional[List[str]] = None):
        """
        Fetch data from Polygon.io with parallel processing and optimized data handling
        
        Args:
            data: ParsedMarketData object to populate
            symbols: Optional list of symbols to fetch data for. If None, fetches all available symbols.
        """
        start_time = time.time()
        
        # Build URL
        url = f"{self.polygon_base_url}/v2/snapshot/locale/us/markets/stocks/tickers"
        params = {"apiKey": self.polygon_api_key}
        
        # Add symbols if specified
        if symbols:
            # For large symbol lists, split into batches to avoid URL length limits
            if len(symbols) > 100:
                return self._fetch_polygon_data_batched(data, symbols)
            
            symbol_list = ",".join(symbols)
            params["tickers"] = symbol_list
            logging.debug(f"Fetching data for specific symbols: {symbol_list}")
        else:
            logging.debug("Fetching full market snapshot")
        
        try:
            # Perform request with timeout and backoff retry
            response = self.session.get(
                url,
                params=params,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            # Raise for status to trigger backoff retry mechanism if needed
            response.raise_for_status()
            
            # Parse JSON response using fast JSON parser for better performance
            response_json = json_fast.loads(response.text)
            
            # Check status
            if response_json.get("status") != "OK":
                error_msg = f"API error: {response_json.get('status')}"
                logging.error(error_msg)
                raise Exception(error_msg)
            
            # Set timestamp
            data.timestamp = int(time.time() * 1_000_000_000)  # nanoseconds
            
            # Get tickers
            tickers = response_json.get("tickers", [])
            
            # Process tickers in parallel using thread pool
            if len(tickers) > 10:  # Only use parallel processing for larger datasets
                # Create a list to store futures
                futures = []
                
                # Submit tasks to thread pool
                for ticker in tickers:
                    futures.append(self.thread_pool.submit(self._process_ticker, ticker))
                
                # Collect results
                for future in futures:
                    symbol, symbol_data = future.result()
                    if symbol:
                        data.symbol_data[symbol] = symbol_data
            else:
                # Process sequentially for small datasets to avoid thread overhead
                for ticker in tickers:
                    symbol, symbol_data = self._process_ticker(ticker)
                    if symbol:
                        data.symbol_data[symbol] = symbol_data
            
            # Set counts
            data.num_trades_processed = len(data.symbol_data)
            data.num_quotes_processed = len(data.symbol_data)
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # Convert to ms
            
            logging.info(f"Fetched data for {len(data.symbol_data)} symbols in {processing_time:.2f}ms")
            
        except requests.exceptions.RequestException as e:
            error_msg = f"HTTP request error: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logging.error(error_msg)
            raise
    
    def _fetch_polygon_data_batched(self, data: ParsedMarketData, symbols: List[str]) -> None:
        """
        Fetch data for a large list of symbols in batches using parallel requests
        
        Args:
            data: ParsedMarketData object to populate
            symbols: List of symbols to fetch data for
        """
        batch_size = 100  # Maximum symbols per request
        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        logging.info(f"Splitting {len(symbols)} symbols into {len(batches)} batches for parallel processing")
        
        # Create a list to store futures
        futures = []
        
        # Submit batch requests in parallel
        for batch in batches:
            # Build URL for this batch
            url = f"{self.polygon_base_url}/v2/snapshot/locale/us/markets/stocks/tickers"
            params = {
                "apiKey": self.polygon_api_key,
                "tickers": ",".join(batch)
            }
            
            # Submit asynchronous request
            futures.append(self.futures_session.get(
                url,
                params=params,
                timeout=(self.connect_timeout, self.read_timeout)
            ))
        
        # Process results as they complete
        batch_data = []
        for future in futures:
            try:
                response = future.result()
                response.raise_for_status()
                
                # Parse JSON response
                response_json = json_fast.loads(response.text)
                
                if response_json.get("status") != "OK":
                    logging.error(f"API error in batch: {response_json.get('status')}")
                    continue
                
                # Add tickers to batch data
                batch_data.extend(response_json.get("tickers", []))
                
            except Exception as e:
                logging.error(f"Error processing batch: {str(e)}")
        
        # Set timestamp
        data.timestamp = int(time.time() * 1_000_000_000)  # nanoseconds
        
        # Process all tickers in parallel
        futures = []
        for ticker in batch_data:
            futures.append(self.thread_pool.submit(self._process_ticker, ticker))
        
        # Collect results
        for future in futures:
            try:
                symbol, symbol_data = future.result()
                if symbol:
                    data.symbol_data[symbol] = symbol_data
            except Exception as e:
                logging.error(f"Error processing ticker: {str(e)}")
        
        # Set counts
        data.num_trades_processed = len(data.symbol_data)
        data.num_quotes_processed = len(data.symbol_data)
        
        logging.info(f"Fetched data for {len(data.symbol_data)} symbols using batched requests")
    
    def _process_ticker(self, ticker: Dict[str, Any]) -> Tuple[str, ParsedMarketData.SymbolData]:
        """
        Process a single ticker from the API response
        
        Args:
            ticker: Ticker data from API response
            
        Returns:
            Tuple of (symbol, symbol_data)
        """
        try:
            # Create symbol data
            symbol_data = ParsedMarketData.SymbolData()
            symbol = ticker.get("ticker", "")
            symbol_data.symbol = symbol
            
            # Set price data with actual values
            if "lastTrade" in ticker and ticker["lastTrade"]:
                symbol_data.last_price = ticker["lastTrade"].get("p", 0.0)
                symbol_data.timestamp = ticker["lastTrade"].get("t", 0)
            
            if "lastQuote" in ticker and ticker["lastQuote"]:
                symbol_data.bid_price = ticker["lastQuote"].get("p", 0.0)
                symbol_data.ask_price = ticker["lastQuote"].get("P", 0.0)
                symbol_data.bid_ask_spread = symbol_data.ask_price - symbol_data.bid_price
            
            # Set day data with actual values
            if "day" in ticker and ticker["day"]:
                symbol_data.open_price = ticker["day"].get("o", 0.0)
                symbol_data.high_price = ticker["day"].get("h", 0.0)
                symbol_data.low_price = ticker["day"].get("l", 0.0)
                symbol_data.volume = ticker["day"].get("v", 0)
                symbol_data.vwap = ticker["day"].get("vw", 0.0)
            
            # Set previous day data with actual values
            if "prevDay" in ticker and ticker["prevDay"]:
                symbol_data.prev_close = ticker["prevDay"].get("c", 0.0)
            
            return symbol, symbol_data
        except Exception as e:
            logging.error(f"Error processing ticker {ticker.get('ticker', 'unknown')}: {str(e)}")
            return "", ParsedMarketData.SymbolData()
    
    def cleanup(self):
        """
        Clean up resources used by the REST client
        
        This method should be called when the client is no longer needed
        to ensure proper release of resources.
        """
        logging.info("Cleaning up PolygonRestAPI resources")
        
        # Stop periodic fetching if running
        if self.running:
            self.stop_periodic_fetching()
        
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
        
        # Release any large buffers
        if hasattr(self, 'symbol_buffer_size'):
            # Force garbage collection of any large buffers
            import gc
            gc.collect()
            logging.debug("Memory buffers released")
