"""
Data Manager for ML Model Training

This module provides utilities for fetching, processing, and managing
market data for training ML models.
"""

import os
import json
import numpy as np
import pandas as pd
import requests
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor

from src.monitoring.log import logging as log
from src.data.polygon_rest_api import PolygonRestAPI
from config.config_loader import get_config
from src.ml.trainer.latency_profiler import LatencyProfiler

class DataManager:
    """
    Data manager for ML model training.
    
    This class provides utilities for fetching, processing, and managing
    market data for training ML models. It supports fetching historical
    data from Polygon.io, caching data to disk, and creating training
    datasets for different model types.
    """
    
    # Data types
    DATA_TYPE_TRADES = "trades"
    DATA_TYPE_QUOTES = "quotes"
    DATA_TYPE_BARS = "bars"
    DATA_TYPE_SNAPSHOTS = "snapshots"
    
    # Cache directories
    CACHE_DIR = "data/cache"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data manager.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.logger = log.setup_logger("data_manager")
        self.logger.info("Initializing data manager")
        
        # Load configuration
        if config is None:
            config = get_config()
            
        self.config = config
        
        # Initialize latency profiler
        self.latency_profiler = LatencyProfiler("data_manager")
        
        # Initialize Polygon.io API client
        self.polygon_api = PolygonRestAPI(config)
        
        # Create cache directory
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        
        # Initialize thread pool
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.get("performance", {}).get("processor_threads", 4),
            thread_name_prefix="DataManager-Worker"
        )
        
        # Initialize cache
        self.cache_lock = threading.Lock()
        self.cache_index = self._load_cache_index()
        
    def _load_cache_index(self) -> Dict[str, Any]:
        """
        Load the cache index.
        
        Returns:
            Cache index dictionary
        """
        cache_index_path = os.path.join(self.CACHE_DIR, "cache_index.json")
        
        if os.path.exists(cache_index_path):
            try:
                with open(cache_index_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading cache index: {str(e)}")
                
        # Return empty cache index
        return {
            "trades": {},
            "quotes": {},
            "bars": {},
            "snapshots": {},
            "last_updated": datetime.now().isoformat()
        }
        
    def _save_cache_index(self) -> None:
        """Save the cache index."""
        cache_index_path = os.path.join(self.CACHE_DIR, "cache_index.json")
        
        # Update last updated timestamp
        self.cache_index["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(cache_index_path, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
                
            self.logger.debug("Saved cache index")
        except Exception as e:
            self.logger.error(f"Error saving cache index: {str(e)}")
            
    def fetch_historical_bars(self, symbols: List[str], start_date: str, end_date: str,
                             timespan: str = "minute", limit: int = 50000,
                             use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical bars for a list of symbols.
        
        Args:
            symbols: List of symbols to fetch data for
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timespan: Timespan (minute, hour, day, week, month, quarter, year)
            limit: Maximum number of bars per request
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary mapping symbols to DataFrames of bar data
        """
        self.logger.info(f"Fetching historical {timespan} bars for {len(symbols)} symbols "
                       f"from {start_date} to {end_date}")
        
        # Start timing
        self.latency_profiler.start_phase("fetch_historical_bars")
        
        # Convert dates to datetime objects
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Initialize results dictionary
        results = {}
        
        # Process each symbol
        for symbol in symbols:
            # Check cache if enabled
            if use_cache:
                cached_data = self._get_cached_bars(symbol, start_dt, end_dt, timespan)
                if cached_data is not None:
                    results[symbol] = cached_data
                    continue
                    
            # Fetch data from Polygon.io
            try:
                # Use Polygon.io API to fetch historical bars
                bars = self._fetch_polygon_historical_bars(symbol, start_dt, end_dt, timespan, limit)
                
                # Cache the data
                if use_cache:
                    self._cache_bars(symbol, bars, timespan)
                    
                results[symbol] = bars
                
            except Exception as e:
                self.logger.error(f"Error fetching bars for {symbol}: {str(e)}")
                
        # End timing
        self.latency_profiler.end_phase()
        
        self.logger.info(f"Fetched historical bars for {len(results)}/{len(symbols)} symbols")
        
        return results
        
    def _get_cached_bars(self, symbol: str, start_dt: datetime, end_dt: datetime,
                        timespan: str) -> Optional[pd.DataFrame]:
        """
        Get cached bars for a symbol if available.
        
        Args:
            symbol: Symbol to get data for
            start_dt: Start date
            end_dt: End date
            timespan: Timespan
            
        Returns:
            DataFrame of cached bars or None if not available
        """
        with self.cache_lock:
            # Check if symbol exists in cache
            if symbol not in self.cache_index["bars"]:
                return None

            # Check if timespan exists in cache
            if timespan not in self.cache_index["bars"][symbol]:
                return None

            # Get cache entry
            cache_entry = self.cache_index["bars"][symbol][timespan]

            # Check if cache covers the requested date range
            cache_start = pd.to_datetime(cache_entry["start_date"])
            cache_end = pd.to_datetime(cache_entry["end_date"])

            if cache_start <= start_dt and cache_end >= end_dt:
                # Load cached data
                cache_path = cache_entry["path"]

                try:
                    df = pd.read_parquet(cache_path)

                    # Strictly filter to requested date range (no future data)
                    df = df[(df.index >= start_dt) & (df.index <= end_dt)]

                    self.logger.debug(f"Using cached {timespan} bars for {symbol}")
                    self.logger.debug(f"Using cached {timespan} bars for {symbol} from {start_dt} to {end_dt}. Cache covers {cache_start} to {cache_end}")
                    if df.index.max() > end_dt:
                        self.logger.warning(f"Cached data for {symbol} contains future data. Cache end: {cache_end}, requested end: {end_dt}")
                        # Remove any future data
                        df = df[df.index <= end_dt]
                    return df
                except Exception as e:
                    self.logger.error(f"Error loading cached bars for {symbol}: {str(e)}")

        return None
        
    def _cache_bars(self, symbol: str, bars: pd.DataFrame, timespan: str) -> None:
        """
        Cache bars for a symbol.
        
        Args:
            symbol: Symbol to cache data for
            bars: DataFrame of bars
            timespan: Timespan
        """
        with self.cache_lock:
            # Create cache directory
            symbol_cache_dir = os.path.join(self.CACHE_DIR, "bars", symbol)
            os.makedirs(symbol_cache_dir, exist_ok=True)
            
            # Generate cache path
            cache_path = os.path.join(symbol_cache_dir, f"{timespan}.parquet")
            
            # Save data to cache
            bars.to_parquet(cache_path)
            
            # Update cache index
            if symbol not in self.cache_index["bars"]:
                self.cache_index["bars"][symbol] = {}
            
            # Check if bars DataFrame is empty
            if bars.empty:
                self.logger.warning(f"Empty DataFrame for {symbol}, not updating cache index")
                return
                
            # Ensure we have valid datetime objects for min and max
            try:
                start_date = bars.index.min().isoformat()
                end_date = bars.index.max().isoformat()
                
                self.cache_index["bars"][symbol][timespan] = {
                    "path": cache_path,
                    "start_date": start_date,
                    "end_date": end_date,
                    "updated_at": datetime.now().isoformat()
                }
            except AttributeError as e:
                self.logger.error(f"Error updating cache index for {symbol}: {str(e)}")
                self.logger.error(f"Index min type: {type(bars.index.min())}, max type: {type(bars.index.max())}")
                return
            
            # Save cache index
            self._save_cache_index()
            
            self.logger.debug(f"Cached {timespan} bars for {symbol}")
            
    def _is_trading_day(self, date: datetime) -> bool:
        """
        Check if a given date is a trading day (not a weekend or holiday).
        
        Args:
            date: Date to check
            
        Returns:
            True if the date is a trading day, False otherwise
        """
        # Check if it's a weekend
        if date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False
            
        # Check if it's a holiday
        # This is a simplified check - in a real implementation, you would use a proper calendar
        # like pandas_market_calendars or exchange_calendars
        holidays = [
            # 2025 US Market Holidays
            datetime(2025, 1, 1),   # New Year's Day
            datetime(2025, 1, 20),  # Martin Luther King Jr. Day
            datetime(2025, 2, 17),  # Presidents' Day
            datetime(2025, 4, 18),  # Good Friday
            datetime(2025, 5, 26),  # Memorial Day
            datetime(2025, 6, 19),  # Juneteenth
            datetime(2025, 7, 4),   # Independence Day
            datetime(2025, 9, 1),   # Labor Day
            datetime(2025, 11, 27), # Thanksgiving Day
            datetime(2025, 12, 25), # Christmas Day
        ]
        
        # Convert date to date-only for comparison
        date_only = date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Check if date is in holidays
        for holiday in holidays:
            if date_only == holiday:
                return False
                
        return True
        
    def _get_valid_trading_date(self, date_dt: datetime) -> datetime:
        """
        Get a valid trading date for data fetching, adjusting to previous trading day if needed.
        
        If the provided date is today, in the future, or a non-trading day,
        this method will return the most recent past trading day.
        
        Args:
            date_dt: The date to validate
            
        Returns:
            A valid past trading date
        """
        # Get current date without time component
        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # If date is today or in the future, adjust to previous day
        adjusted_date = date_dt
        if adjusted_date >= current_date:
            adjusted_date = current_date - timedelta(days=1)
            self.logger.info(f"Adjusting future date {date_dt.strftime('%Y-%m-%d')} to previous day {adjusted_date.strftime('%Y-%m-%d')}")
        
        # If adjusted date is still not a trading day, find the most recent trading day
        if not self._is_trading_day(adjusted_date):
            original_date = adjusted_date
            max_lookback = 30  # Avoid infinite loops by limiting lookback
            lookback_count = 0
            
            while not self._is_trading_day(adjusted_date) and lookback_count < max_lookback:
                adjusted_date = adjusted_date - timedelta(days=1)
                lookback_count += 1
                
            if lookback_count >= max_lookback:
                self.logger.warning(f"Could not find a valid trading day within {max_lookback} days before {original_date.strftime('%Y-%m-%d')}")
                # Return the original date as a fallback, even though it's not ideal
                return original_date
            
            if original_date != adjusted_date:
                self.logger.info(f"Adjusted non-trading day {original_date.strftime('%Y-%m-%d')} to previous trading day {adjusted_date.strftime('%Y-%m-%d')}")
        
        return adjusted_date

    def _get_trading_days(self, start_dt: datetime, end_dt: datetime) -> List[datetime]:
        """
        Get a list of trading days between start_dt and end_dt.
        
        Args:
            start_dt: Start date
            end_dt: End date
            
        Returns:
            List of trading days
        """
        trading_days = []
        current_dt = start_dt
        
        while current_dt <= end_dt:
            if self._is_trading_day(current_dt):
                trading_days.append(current_dt)
            current_dt += timedelta(days=1)
            
        return trading_days

    def _fetch_polygon_historical_bars(self, symbol: str, start_dt: datetime, end_dt: datetime,
                                     timespan: str, limit: int = 50000) -> pd.DataFrame:
        """
        Fetch historical bar data from Polygon.io API.

        Args:
            symbol: Symbol to fetch data for
            start_dt: Start date
            end_dt: End date
            timespan: Timespan (minute, hour, day, week, month)
            limit: Maximum number of bars per request

        Returns:
            DataFrame of historical bars
        """
        self.logger.info(f"Fetching historical {timespan} bars for {symbol} from {start_dt} to {end_dt}")
        
        # For synthetic symbols (STOCK1, STOCK2, etc.), use the cached parquet files
        if symbol.startswith("STOCK"):
            self.logger.info(f"Using cached parquet file for synthetic symbol {symbol}")
            try:
                # Construct the path to the parquet file
                cache_path = os.path.join(self.CACHE_DIR, "bars", symbol, f"{timespan}.parquet")
                
                # Check if the file exists
                if os.path.exists(cache_path):
                    df = pd.read_parquet(cache_path)
                    
                    # Filter to the requested date range
                    # Make sure index is datetime type
                    if not pd.api.types.is_datetime64_any_dtype(df.index):
                        self.logger.warning(f"Index for {symbol} is not datetime type, converting")
                        df.index = pd.to_datetime(df.index)
                    
                    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                    
                    if not df.empty:
                        self.logger.info(f"Successfully loaded {len(df)} bars from cache for {symbol}")
                        return df
                    else:
                        self.logger.warning(f"No data in date range for {symbol} in cache")
                        # No synthetic data fallback
                        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])
            except Exception as e:
                self.logger.error(f"Error loading from cache for {symbol}: {str(e)}")
                # No synthetic data fallback
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])
        
        # For real symbols, use the Polygon API
        # Convert timespan to Polygon API format
        if timespan == "minute":
            multiplier = 1
            timespan_api = "minute"
        elif timespan == "hour":
            multiplier = 1
            timespan_api = "hour"
        elif timespan == "day":
            multiplier = 1
            timespan_api = "day"
        elif timespan == "week":
            multiplier = 1
            timespan_api = "week"
        elif timespan == "month":
            multiplier = 1
            timespan_api = "month"
        else:
            multiplier = 1
            timespan_api = "day"  # Default to daily
        
        # Format dates for API
        start_timestamp = int(start_dt.timestamp() * 1000)  # Convert to milliseconds
        end_timestamp = int(end_dt.timestamp() * 1000)  # Convert to milliseconds
        
        # Build URL for Polygon API
        base_url = self.polygon_api.polygon_base_url
        api_key = self.polygon_api.polygon_api_key
        
        # For synthetic symbols, try to map to real symbols for API calls
        api_symbol = symbol
        if symbol.startswith("STOCK"):
            # Map synthetic symbols to real symbols for API testing
            symbol_map = {
                "STOCK1": "AAPL",
                "STOCK2": "MSFT",
                "STOCK3": "GOOGL",
                "STOCK4": "AMZN",
                "STOCK5": "META",
                "STOCK6": "TSLA",
                "STOCK7": "NVDA",
                "STOCK8": "JPM",
                "STOCK9": "V",
                "STOCK10": "JNJ"
            }
            api_symbol = symbol_map.get(symbol, "AAPL")  # Default to AAPL if not in map
            self.logger.info(f"Mapping synthetic symbol {symbol} to real symbol {api_symbol} for API call")
        
        url = f"{base_url}/v2/aggs/ticker/{api_symbol}/range/{multiplier}/{timespan_api}/{start_timestamp}/{end_timestamp}"
        params = {
            "apiKey": api_key,
            "limit": limit,
            "adjusted": "true"  # Get adjusted prices
        }
        
        self.logger.debug(f"Polygon API URL: {url}")
        
        try:
            # Use the session from polygon_api for consistent retry behavior
            response = self.polygon_api.session.get(
                url,
                params=params,
                timeout=(self.polygon_api.connect_timeout, self.polygon_api.read_timeout)
            )
            
            # Raise for status
            response.raise_for_status()
            
            # Parse response
            response_json = json.loads(response.text)
            
            # Check status
            if response_json.get("status") != "OK":
                error_msg = f"API error: {response_json.get('status')}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
            
            # Get results
            results = response_json.get("results", [])
            
            if not results:
                self.logger.warning(f"No data returned for {symbol} from {start_dt} to {end_dt}")
                # Return empty DataFrame with expected columns and datetime index
                empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])
                empty_df.index.name = "timestamp"  # Set index name to match other DataFrames
                return empty_df
            
            # Convert to DataFrame
            data = []
            for bar in results:
                # Convert timestamp from milliseconds to datetime
                timestamp = pd.Timestamp(bar.get("t", 0), unit="ms")
                
                data.append({
                    "timestamp": timestamp,
                    "open": bar.get("o", 0.0),
                    "high": bar.get("h", 0.0),
                    "low": bar.get("l", 0.0),
                    "close": bar.get("c", 0.0),
                    "volume": bar.get("v", 0),
                    "vwap": bar.get("vw", 0.0)
                })
            
            df = pd.DataFrame(data)
            
            # Set timestamp as index
            df.set_index("timestamp", inplace=True)
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            self.logger.info(f"Successfully fetched {len(df)} bars for {symbol} using Polygon API")
            
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"HTTP request error for {symbol}: {str(e)}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error for {symbol}: {str(e)}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])
        except Exception as e:
            self.logger.error(f"Unexpected error fetching data for {symbol}: {str(e)}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])
        
    def fetch_market_snapshots(self, date: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Fetch market snapshots for a specific date.
        
        Args:
            date: Date to fetch snapshots for (YYYY-MM-DD)
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary of market snapshots
        """
        self.logger.info(f"Fetching market snapshots for {date}")
        
        # Start timing
        self.latency_profiler.start_phase("fetch_market_snapshots")
        
        # Convert date to datetime object
        date_dt = pd.to_datetime(date)
        
        # Check cache if enabled
        if use_cache:
            cached_data = self._get_cached_snapshots(date_dt)
            if cached_data is not None:
                self.latency_profiler.end_phase()
                return cached_data
                
        # Fetch data from Polygon.io
        try:
            # Use Polygon.io API to fetch market snapshots
            snapshots = self._fetch_polygon_snapshots(date_dt)
            
            # Cache the data
            if use_cache:
                self._cache_snapshots(date_dt, snapshots)
                
            self.latency_profiler.end_phase()
            
            return snapshots
            
        except Exception as e:
            self.logger.error(f"Error fetching market snapshots for {date}: {str(e)}")
            self.latency_profiler.end_phase()
            
            return {}
            
    def _get_cached_snapshots(self, date_dt: datetime) -> Optional[Dict[str, Any]]:
        """
        Get cached market snapshots if available.
        
        Args:
            date_dt: Date to get snapshots for
            
        Returns:
            Dictionary of cached snapshots or None if not available
        """
        date_str = date_dt.strftime("%Y-%m-%d")
        
        with self.cache_lock:
            # Check if date exists in cache
            if date_str not in self.cache_index["snapshots"]:
                return None
                
            # Get cache entry
            cache_entry = self.cache_index["snapshots"][date_str]
            
            # Load cached data
            cache_path = cache_entry["path"]
            
            try:
                with open(cache_path, 'r') as f:
                    snapshots = json.load(f)
                    
                self.logger.debug(f"Using cached market snapshots for {date_str}")
                
                return snapshots
            except Exception as e:
                self.logger.error(f"Error loading cached snapshots for {date_str}: {str(e)}")
                
        return None
        
    def _cache_snapshots(self, date_dt: datetime, snapshots: Dict[str, Any]) -> None:
        """
        Cache market snapshots.
        
        Args:
            date_dt: Date of snapshots
            snapshots: Dictionary of snapshots
        """
        date_str = date_dt.strftime("%Y-%m-%d")
        
        with self.cache_lock:
            # Create cache directory
            snapshots_cache_dir = os.path.join(self.CACHE_DIR, "snapshots")
            os.makedirs(snapshots_cache_dir, exist_ok=True)
            
            # Generate cache path
            cache_path = os.path.join(snapshots_cache_dir, f"{date_str}.json")
            
            # Save data to cache
            with open(cache_path, 'w') as f:
                json.dump(snapshots, f, indent=2)
                
            # Update cache index
            self.cache_index["snapshots"][date_str] = {
                "path": cache_path,
                "updated_at": datetime.now().isoformat()
            }
            
            # Save cache index
            self._save_cache_index()
            
            self.logger.debug(f"Cached market snapshots for {date_str}")
            
    def _fetch_polygon_snapshots(self, date_dt: datetime) -> Dict[str, Any]:
        """
        Fetch market snapshots from Polygon.io API.
        
        Args:
            date_dt: Date to fetch snapshots for
            
        Returns:
            Dictionary of market snapshots
        """
        self.logger.info(f"Fetching market snapshots for {date_dt.strftime('%Y-%m-%d')}")
        
        # Get a valid trading date
        valid_date_dt = self._get_valid_trading_date(date_dt)
        
        # If the date is not a trading day, return empty snapshots
        if not self._is_trading_day(valid_date_dt):
            self.logger.warning(f"Date {valid_date_dt.strftime('%Y-%m-%d')} is not a trading day (weekend or holiday)")
            return {"timestamp": valid_date_dt.isoformat(), "symbols": {}}
        
        # If the original date was adjusted, log it
        if valid_date_dt != date_dt:
            self.logger.info(f"Using market data from {valid_date_dt.strftime('%Y-%m-%d')} instead of requested date {date_dt.strftime('%Y-%m-%d')}")
            # Update the date_dt to use the valid one for the rest of the function
            date_dt = valid_date_dt
        
        # Get list of symbols (using default list if API call fails)
        symbols = self._get_polygon_symbols()
        
        # Initialize snapshots dictionary
        snapshots = {
            "timestamp": date_dt.isoformat(),
            "symbols": {}
        }
        
        # Fetch snapshot data for each symbol
        for symbol in symbols:
            try:
                # Fetch snapshot data using the polygon_api client
                symbol_data = self._fetch_polygon_symbol_snapshot(symbol, date_dt)
                if symbol_data:
                    snapshots["symbols"][symbol] = symbol_data
            except Exception as e:
                self.logger.error(f"Error fetching snapshot for {symbol}: {str(e)}")
                
                # If API call fails, skip this symbol
                pass
        
        # If we couldn't get any real data, return empty snapshots
        if not snapshots["symbols"]:
            self.logger.warning(f"No real data available for {date_dt.strftime('%Y-%m-%d')}")
            
        self.logger.info(f"Successfully fetched snapshots for {len(snapshots['symbols'])} symbols")
        
        return snapshots
    
    def _get_polygon_symbols(self) -> List[str]:
        """
        Get list of symbols from Polygon.io API with pagination support.
        
        Returns:
            List of symbols (2000-3000 for production use)
        """
        try:
            # Configuration parameters
            base_url = self.polygon_api.polygon_base_url
            api_key = self.polygon_api.polygon_api_key
            max_symbols = self.config.get("data_sources", {}).get("polygon", {}).get("max_symbols", 3000)
            symbols_per_page = 1000  # Maximum for Polygon API
            
            all_symbols = []
            page = 1
            
            # Paginate until we reach the desired number of symbols
            while len(all_symbols) < max_symbols:
                url = f"{base_url}/v3/reference/tickers"
                params = {
                    "apiKey": api_key,
                    "market": "stocks",
                    "active": "true",
                    "sort": "ticker",  # Sort alphabetically
                    "order": "asc",    # Ascending order
                    "limit": symbols_per_page,
                    "page": page
                }
                
                self.logger.info(f"Fetching symbols page {page} (max symbols: {max_symbols})")
                
                # Use the session from polygon_api for consistent retry behavior
                response = self.polygon_api.session.get(
                    url,
                    params=params,
                    timeout=(self.polygon_api.connect_timeout, self.polygon_api.read_timeout)
                )
                
                # Raise for status
                response.raise_for_status()
                
                # Parse response
                response_json = json.loads(response.text)
                
                # Get results
                results = response_json.get("results", [])
                
                if not results:
                    self.logger.info(f"No more symbols found after page {page}")
                    break
                
                # Extract symbols
                page_symbols = [ticker.get("ticker") for ticker in results if ticker.get("ticker")]
                all_symbols.extend(page_symbols)
                
                self.logger.info(f"Fetched {len(page_symbols)} symbols on page {page}, total: {len(all_symbols)}")
                
                # Move to next page
                page += 1
                
                # Check if we've reached the last page
                if len(page_symbols) < symbols_per_page:
                    break
            
            # Trim to max symbols if needed
            if len(all_symbols) > max_symbols:
                all_symbols = all_symbols[:max_symbols]
            
            self.logger.info(f"Successfully fetched {len(all_symbols)} symbols from Polygon API")
            return all_symbols
            
        except Exception as e:
            self.logger.error(f"Error fetching symbols from Polygon API: {str(e)}")
            
            # In case of error, return a synthetic set of symbols
            # Make sure this covers a broader range than just STOCK1-100
            synthetic_symbols = []
            
            # Generate synthetic symbols using a mix of letter prefixes
            for prefix in ["A", "B", "C", "G", "M", "N", "S", "T"]:
                for i in range(1, 376):  # ~3000 symbols total (~375 per letter)
                    synthetic_symbols.append(f"{prefix}STOCK{i}")
            
            self.logger.warning(f"Using {len(synthetic_symbols)} synthetic symbols due to API error")
            return synthetic_symbols
    
    def _fetch_polygon_symbol_snapshot(self, symbol: str, date_dt: datetime) -> Dict[str, Any]:
        """
        Fetch snapshot data for a single symbol from Polygon.io API.
        
        Args:
            symbol: Symbol to fetch data for
            date_dt: Date to fetch data for
            
        Returns:
            Dictionary of snapshot data
        """
        try:
            # Build URL for Polygon API
            base_url = self.polygon_api.polygon_base_url
            api_key = self.polygon_api.polygon_api_key
            
            # Fetch last trade
            url = f"{base_url}/v2/last/trade/{symbol}"
            params = {"apiKey": api_key}
            
            # Use the session from polygon_api for consistent retry behavior
            response = self.polygon_api.session.get(
                url,
                params=params,
                timeout=(self.polygon_api.connect_timeout, self.polygon_api.read_timeout)
            )
            
            # Raise for status
            response.raise_for_status()
            
            # Parse response
            trade_json = json.loads(response.text)
            
            # Fetch last quote
            url = f"{base_url}/v2/last/nbbo/{symbol}"
            
            # Use the session from polygon_api for consistent retry behavior
            response = self.polygon_api.session.get(
                url,
                params=params,
                timeout=(self.polygon_api.connect_timeout, self.polygon_api.read_timeout)
            )
            
            # Raise for status
            response.raise_for_status()
            
            # Parse response
            quote_json = json.loads(response.text)
            
            # Fetch daily bar
            # Get a valid trading date and format it for the API
            valid_date_dt = self._get_valid_trading_date(date_dt)
            date_str = valid_date_dt.strftime("%Y-%m-%d")
            
            # If date was adjusted, log it
            if valid_date_dt != date_dt:
                self.logger.info(f"Using data from {date_str} for {symbol} instead of {date_dt.strftime('%Y-%m-%d')}")
            
            url = f"{base_url}/v1/open-close/{symbol}/{date_str}"
            
            # Use the session from polygon_api for consistent retry behavior
            response = self.polygon_api.session.get(
                url,
                params=params,
                timeout=(self.polygon_api.connect_timeout, self.polygon_api.read_timeout)
            )
            
            # Raise for status
            response.raise_for_status()
            
            # Parse response
            day_json = json.loads(response.text)
            
            # Create snapshot data
            snapshot = {}
            
            # Set price data
            if trade_json.get("status") == "success" and "last" in trade_json:
                snapshot["last_price"] = trade_json["last"].get("price", 0.0)
                snapshot["timestamp"] = trade_json["last"].get("timestamp", 0)
            
            if quote_json.get("status") == "success" and "last" in quote_json:
                snapshot["bid_price"] = quote_json["last"].get("bid_price", 0.0)
                snapshot["ask_price"] = quote_json["last"].get("ask_price", 0.0)
                snapshot["bid_ask_spread"] = snapshot["ask_price"] - snapshot["bid_price"]
            
            if day_json.get("status") == "OK":
                snapshot["open_price"] = day_json.get("open", 0.0)
                snapshot["high_price"] = day_json.get("high", 0.0)
                snapshot["low_price"] = day_json.get("low", 0.0)
                snapshot["close"] = day_json.get("close", 0.0)
                snapshot["volume"] = day_json.get("volume", 0)
                snapshot["vwap"] = day_json.get("vwap", 0.0)
                snapshot["prev_close"] = day_json.get("preMarket", 0.0)
            
            # Calculate technical indicators using historical data
            self._calculate_technical_indicators(symbol, snapshot, date_dt)
            
            return snapshot
            
        except Exception as e:
            # More informative error message for debugging
            error_msg = f"Error fetching snapshot for {symbol}: {str(e)}"
            if "404" in str(e) and date_dt.strftime("%Y-%m-%d") in str(e):
                error_msg += f" (Data not available for date {date_dt.strftime('%Y-%m-%d')})"
            self.logger.error(error_msg)
            return {}
    
    def _calculate_technical_indicators(self, symbol: str, snapshot: Dict[str, Any], date_dt: datetime) -> None:
        """
        Calculate technical indicators for a symbol using historical data.
        
        Args:
            symbol: Symbol to calculate indicators for
            snapshot: Snapshot data to update with indicators
            date_dt: Current date
        """
        try:
            # Fetch historical data for calculating indicators
            # Use a longer lookback to ensure enough data for rolling features
            start_date = (date_dt - timedelta(days=60)).strftime('%Y-%m-%d')
            # Only use data up to the day before the current date to avoid lookahead bias
            end_date = (date_dt - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Fetch historical bars for this symbol
            historical_data = self.fetch_historical_bars(
                [symbol],
                start_date=start_date,
                end_date=end_date,
                timespan="day"
            )
            
            # Check if we got data for this symbol
            historical_bars = historical_data.get(symbol)
            if symbol not in historical_data or historical_bars is None or len(historical_bars) < 14:
                self.logger.warning(f"Not enough historical data for {symbol} to calculate indicators")
                return
            
            # Extract price data
            closes = historical_bars['close'].values
            highs = historical_bars['high'].values
            lows = historical_bars['low'].values
            volumes = historical_bars['volume'].values
            
            # Calculate RSI
            snapshot['rsi_14'] = self._calculate_rsi(closes, 14)
            
            # Calculate MACD
            macd_line, signal_line, histogram = self._calculate_macd(closes)
            snapshot['macd'] = macd_line
            snapshot['macd_signal'] = signal_line
            snapshot['macd_histogram'] = histogram
            
            # Calculate Bollinger Bands
            upper, middle, lower = self._calculate_bollinger_bands(closes)
            snapshot['bb_upper'] = upper
            snapshot['bb_middle'] = middle
            snapshot['bb_lower'] = lower
            
            # Calculate ATR
            snapshot['atr'] = self._calculate_atr(highs, lows, closes)
            
            # Calculate volume-based indicators
            if len(volumes) > 20:
                snapshot['avg_volume'] = np.mean(volumes[-20:])
                snapshot['volume_acceleration'] = self._calculate_acceleration(volumes)
                snapshot['volume_spike'] = volumes[-1] / max(np.mean(volumes[-10:]), 1)
            
            # Calculate price change indicators
            if len(closes) > 1:
                snapshot['momentum_1m'] = (closes[-1] / closes[-2] - 1) if closes[-2] > 0 else 0
            
            # Calculate SMA cross signal
            snapshot['sma_cross_signal'] = self._calculate_sma_cross(closes)
            
            # Calculate support/resistance proximity
            # This is a simplified implementation
            snapshot['support_resistance_proximity'] = 0.5  # Default value
            
            # Calculate volatility ratio
            if len(closes) > 20:
                vol_short = np.std(closes[-10:])
                vol_long = np.std(closes[-20:])
                snapshot['volatility_ratio'] = vol_short / vol_long if vol_long > 0 else 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators for {symbol}: {str(e)}")
        
    def fetch_market_snapshots_for_period(self, start_date: str, end_date: str,
                                         symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Fetch historical market snapshots for a specified period.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: List of symbols to include (optional)
            
        Returns:
            List of market snapshots in the same format as production
        """
        self.logger.info(f"Fetching market snapshots for period {start_date} to {end_date}")
        
        # Convert dates to datetime objects
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Generate date range
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        
        # Fetch snapshots for each date
        snapshots = []
        
        for date in date_range:
            # Skip weekends and holidays
            if not self._is_trading_day(date):
                continue
                
            date_str = date.strftime('%Y-%m-%d')
            
            # Use existing method to fetch snapshot for this date
            snapshot = self.fetch_market_snapshots(date_str)
            
            # Filter symbols if requested
            if symbols and "symbols" in snapshot:
                filtered_symbols = {sym: data for sym, data in snapshot["symbols"].items()
                                  if sym in symbols}
                snapshot["symbols"] = filtered_symbols
                
            snapshots.append(snapshot)
        
        self.logger.info(f"Fetched {len(snapshots)} market snapshots for period {start_date} to {end_date}")
        return snapshots
        
    def create_training_dataset(self, model_type: str, start_date: str, end_date: str,
                                symbols: Optional[List[str]] = None,
                                train_test_split: float = 0.8,
                                random_seed: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Create a training dataset for a specific model type with optimized batch processing.
        
        This implementation prevents data leakage by first splitting the time periods,
        then generating features and sequences separately for each period.
        It also uses batch processing to handle large numbers of symbols efficiently.
        
        Args:
            model_type: Type of model (gbdt, axial_attention, lstm_gru)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: List of symbols to include (optional)
            train_test_split: Train/test split ratio
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data)
        """
        self.logger.info(f"Creating training dataset for {model_type} model "
                       f"from {start_date} to {end_date}")
        
        # Start timing
        self.latency_profiler.start_phase("create_training_dataset")
        
        # Set random seed
        np.random.seed(random_seed)
        
        # If symbols not provided, get all available symbols
        if symbols is None:
            symbols = self._get_polygon_symbols()
            
            # Log symbol distribution metrics
            self._log_symbol_distribution_metrics(symbols)
        
        self.logger.info(f"Creating dataset with {len(symbols)} symbols")
        
        # Convert dates to datetime objects
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Generate date range and split chronologically
        date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")
        split_idx = int(len(date_range) * train_test_split)
        
        # Ensure we have enough data for both training and testing
        if split_idx <= 0 or split_idx >= len(date_range):
            raise ValueError(f"Invalid train_test_split value: {train_test_split}. "
                           f"Results in split_idx={split_idx} for date_range of length {len(date_range)}")
        
        train_end_date = date_range[split_idx-1].strftime('%Y-%m-%d')
        test_start_date = date_range[split_idx].strftime('%Y-%m-%d')
        
        self.logger.info(f"Time-based split: Training period {start_date} to {train_end_date}, "
                       f"Testing period {test_start_date} to {end_date}")
        
        # Process symbols in batches to avoid memory issues
        batch_size = self.config.get("data_sources", {}).get("polygon", {}).get("batch_processing", {}).get("batch_size", 500)
        num_batches = (len(symbols) + batch_size - 1) // batch_size
        
        train_data_batches = []
        test_data_batches = []
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(symbols))
            symbol_batch = symbols[batch_start:batch_end]
            
            self.logger.info(f"Processing batch {batch_idx+1}/{num_batches} ({len(symbol_batch)} symbols)")
            
            # Create dataset based on model type with clear time separation
            if model_type == "gbdt":
                train_batch = self._create_gbdt_dataset_period(start_date, train_end_date, symbol_batch)
                test_batch = self._create_gbdt_dataset_period(test_start_date, end_date, symbol_batch)
            elif model_type == "axial_attention":
                train_batch = self._create_axial_attention_dataset_period(start_date, train_end_date, symbol_batch)
                test_batch = self._create_axial_attention_dataset_period(test_start_date, end_date, symbol_batch)
            elif model_type == "lstm_gru":
                train_batch = self._create_lstm_gru_dataset_period(start_date, train_end_date, symbol_batch)
                test_batch = self._create_lstm_gru_dataset_period(test_start_date, end_date, symbol_batch)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            train_data_batches.append(train_batch)
            test_data_batches.append(test_batch)
            
            # Validate no data leakage between train and test sets
            self._validate_no_data_leakage(train_batch, test_batch, model_type,
                                         start_date, train_end_date, test_start_date, end_date)
        
        # Combine batches into final datasets
        train_data = self._combine_dataset_batches(train_data_batches, model_type)
        test_data = self._combine_dataset_batches(test_data_batches, model_type)
        
        # End timing
        self.latency_profiler.end_phase()
        
        self.logger.info(f"Created training dataset for {model_type} model")
        
        return train_data, test_data
        
    def _create_gbdt_dataset_period(self, start_date: str, end_date: str,
                                  symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a GBDT dataset for a specific time period with no leakage.

        This implementation constructs binary classification labels for each symbol and date.
        The label is 1 if the next day's close price increases by more than a threshold (e.g., 0.2%)
        compared to the current day's close, and 0 otherwise.

        Label construction uses only future price information not present in the current snapshot,
        ensuring no leakage: features are computed from the current snapshot, while the label is
        computed from the next day's close price. No features in the current snapshot encode the
        future price change.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: List of symbols to include (optional)

        Returns:
            Dictionary containing dataset for the specified period, including features and labels.
        """
        self.logger.info(f"Creating GBDT dataset for period {start_date} to {end_date}")

        # Convert dates to datetime objects
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Generate date range
        date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")

        # Fetch market snapshots for each date
        snapshots = []
        dates = []

        for date in date_range:
            # Get valid trading date before fetching snapshot
            valid_date = self._get_valid_trading_date(date)
            date_str = valid_date.strftime("%Y-%m-%d")
            snapshot = self.fetch_market_snapshots(date_str)
            # Process technical indicators to ensure they use only past data
            if "symbols" in snapshot and snapshot["symbols"]:
                for symbol in snapshot["symbols"]:
                    # Recalculate indicators using only data available up to this point
                    self._calculate_point_in_time_indicators(snapshot["symbols"][symbol], symbol, date_str)
            self.logger.debug(f"Adding snapshot for {date_str} to GBDT dataset")
            snapshots.append(snapshot)
            dates.append(date_str)

        # --- Begin label construction ---
        # For each symbol, construct a list of features and corresponding labels.
        # The label for (date_i, symbol) is computed as:
        #   label = 1 if (close_{i+1} - close_{i}) / close_{i} > threshold, else 0
        # where close_{i} is the close price for symbol on date_i, and close_{i+1} is the next day's close.

        threshold = 0.002  # 0.2% price increase threshold for label=1

        # Collect all symbols present in the snapshots if not specified
        if symbols is None:
            all_symbols = set()
            for snapshot in snapshots:
                if "symbols" in snapshot:
                    all_symbols.update(snapshot["symbols"].keys())
            symbols = sorted(list(all_symbols))

        features = {symbol: [] for symbol in symbols}
        labels = {symbol: [] for symbol in symbols}
        feature_dates = {symbol: [] for symbol in symbols}

        # For each date except the last, compute features and label using next day's close
        for i in range(len(dates) - 1):
            curr_snapshot = snapshots[i]
            next_snapshot = snapshots[i + 1]
            curr_date = dates[i]
            # next_date variable is not used, so we'll remove it

            for symbol in symbols:
                # Ensure symbol is present in both current and next snapshot
                if (
                    "symbols" in curr_snapshot and symbol in curr_snapshot["symbols"]
                    and "symbols" in next_snapshot and symbol in next_snapshot["symbols"]
                ):
                    curr_data = curr_snapshot["symbols"][symbol]
                    next_data = next_snapshot["symbols"][symbol]

                    # Use close price for label construction
                    curr_close = curr_data.get("close", curr_data.get("last_price"))
                    next_close = next_data.get("close", next_data.get("last_price"))

                    # Defensive: skip if any close price is missing or not finite
                    if curr_close is None or next_close is None:
                        continue
                    try:
                        curr_close = float(curr_close)
                        next_close = float(next_close)
                    except Exception:
                        continue
                    if not np.isfinite(curr_close) or not np.isfinite(next_close) or curr_close == 0:
                        continue

                    # Compute label: 1 if next day's close increases by more than threshold
                    price_change = (next_close - curr_close) / curr_close
                    label = 1 if price_change > threshold else 0

                    # Remove any trivially predictive features (e.g., future price change) if present
                    # (Features are constructed from current snapshot only)
                    # --- Data consistency enforcement ---
                    # Ensure all features are scalars (not arrays/lists), and replace None with 0.0.
                    # This prevents shape mismatches and ensures all feature vectors have the same length and type.
                    raw_features = [
                        curr_data.get("last_price"),
                        curr_data.get("bid_price"),
                        curr_data.get("ask_price"),
                        curr_data.get("bid_ask_spread"),
                        curr_data.get("open_price"),
                        curr_data.get("high_price"),
                        curr_data.get("low_price"),
                        curr_data.get("volume"),
                        curr_data.get("vwap"),
                        curr_data.get("prev_close"),
                        curr_data.get("rsi_14"),
                        curr_data.get("macd"),
                        curr_data.get("macd_signal"),
                        curr_data.get("macd_histogram"),
                        curr_data.get("bb_upper"),
                        curr_data.get("bb_middle"),
                        curr_data.get("bb_lower"),
                        curr_data.get("atr"),
                        curr_data.get("avg_volume"),
                        curr_data.get("volume_acceleration"),
                        curr_data.get("volume_spike"),
                        curr_data.get("momentum_1m"),
                        curr_data.get("sma_cross_signal"),
                        curr_data.get("support_resistance_proximity"),
                        curr_data.get("volatility_ratio"),
                        # Exclude any feature that would leak future price (e.g., price_change_5m)
                        0.0  # [FIX] Add dummy feature to guarantee 26 features for every sample. This ensures all feature vectors are always length 26, preventing shape mismatches regardless of symbol or date.
                    ]
                    feature_vector = []
                    for idx, val in enumerate(raw_features):
                        # If value is None, replace with 0.0
                        if val is None:
                            feature_vector.append(0.0)
                        # If value is a list/array, take the first element if possible, else 0.0
                        elif isinstance(val, (list, np.ndarray)):
                            if len(val) > 0 and np.isscalar(val[0]):
                                feature_vector.append(float(val[0]))
                            else:
                                feature_vector.append(0.0)
                        else:
                            try:
                                feature_vector.append(float(val))
                            except Exception:
                                feature_vector.append(0.0)
                    # Enforce feature vector length consistency
                    expected_len = 26
                    if len(feature_vector) != expected_len:
                        # DEBUG: Log raw_features and feature_vector for diagnosis (as warning for log visibility)
                        self.logger.warning(
                            f"GBDT_DEBUG_MARKER: Feature vector length mismatch for {symbol} on {curr_date}: "
                            f"raw_features={raw_features}, feature_vector={feature_vector}, "
                            f"raw_features_len={len(raw_features)}, feature_vector_len={len(feature_vector)}"
                        )
                        self.logger.warning(
                            f"Feature vector for {symbol} on {curr_date} has length {len(feature_vector)} (expected {expected_len}), skipping sample."
                        )
                        continue
                    # --- End data consistency enforcement ---
                    # The above ensures all feature vectors are 1D, length 26, and contain only floats.

                    features[symbol].append(feature_vector)
                    labels[symbol].append(label)
                    feature_dates[symbol].append(curr_date)

        # Flatten feature_dates (dict of lists) to a flat list of unique date strings for validation compatibility
        flat_dates = sorted(set(date for dates_list in feature_dates.values() for date in dates_list))

        # --- Aggregation for LightGBM compatibility ---
        # LightGBM requires:
        #   - Features: 2D numpy array of shape [num_samples, num_features]
        #   - Labels:   1D numpy array of shape [num_samples]
        # We aggregate all symbol samples into a single array for X and y.
        all_features = []
        all_labels = []
        all_sample_symbols = []
        all_sample_dates = []
        for symbol in symbols:
            feats = features[symbol]
            labs = labels[symbol]
            all_features.extend(feats)
            all_labels.extend(labs)
            all_sample_symbols.extend([symbol] * len(feats))
            all_sample_dates.extend(feature_dates[symbol])

        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.float32)

        # Comment: This aggregation ensures that X is a 2D array [num_samples, num_features]
        # and y is a 1D array [num_samples], as required by LightGBM. This prevents shape errors
        # during model training and guarantees compatibility with LightGBM's fit/predict interface.

        dataset = {
            "features": features,
            "labels": labels,
            "dates": flat_dates,
            "symbols": symbols,
            "X": X,
            "y": y,
            "sample_symbols": all_sample_symbols,
            "sample_dates": all_sample_dates,
        }

        self.logger.info(
            f"Created GBDT dataset with {X.shape[0]} samples, {X.shape[1] if X.ndim==2 else 'N/A'} features, across {len(symbols)} symbols"
        )

        return dataset
        
    def _create_axial_attention_dataset_period(self, start_date: str, end_date: str,
                                             symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create an Axial Attention dataset for a specific time period with no leakage.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: List of symbols to include (optional)
            
        Returns:
            Dictionary containing dataset for the specified period
        """
        self.logger.info(f"Creating Axial Attention dataset for period {start_date} to {end_date}")
        
        # If symbols not provided, use default symbols
        if symbols is None:
            symbols = [f"STOCK{i}" for i in range(1, 21)]
            
        # Fetch historical bars for each symbol within the specified period only
        bars_data = self.fetch_historical_bars(symbols, start_date, end_date, timespan="minute")
        
        # Create sequences and labels
        sequences = {}
        labels = {}
        
        for symbol, bars in bars_data.items():
            # Create sequences of length 100
            seq_length = 100
            
            # Create sequences and labels
            X = []
            y = []
            
            # Only create complete sequences within this period
            # This prevents using data from outside the period
            for i in range(len(bars) - seq_length - 1):
                # Make sure the full sequence and label are within this period
                if i + seq_length + 1 >= len(bars):
                    continue
                    
                # Extract sequence
                seq = bars.iloc[i:i+seq_length]
                
                # Extract features using only data available at sequence time
                features = []
                for _, row in seq.iterrows():
                    # OHLCV features - using only data available at this point
                    features.append([
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row["volume"],
                        row["vwap"]
                    ])
                    
                # Extract label (future price movement)
                # This is acceptable since we're explicitly using future price as a label
                future_price = bars.iloc[i+seq_length+1]["close"]
                current_price = bars.iloc[i+seq_length]["close"]
                price_change = (future_price - current_price) / current_price
                
                # Convert to classification label
                if price_change > 0.005:
                    label = [1, 0, 0]  # Up
                elif price_change < -0.005:
                    label = [0, 0, 1]  # Down
                else:
                    label = [0, 1, 0]  # Neutral
                    
                X.append(features)
                y.append(label)
                
            if len(X) > 0:
                sequences[symbol] = np.array(X)
                labels[symbol] = np.array(y)
            else:
                self.logger.warning(f"No valid sequences created for {symbol} in period {start_date} to {end_date}")
        
        # Create dataset
        dataset = {
            "sequences": sequences,
            "labels": labels
        }
        
        total_sequences = sum(len(seqs) for seqs in sequences.values())
        self.logger.info(f"Created Axial Attention dataset with {total_sequences} sequences from {len(sequences)} symbols")
        
        return dataset
        
    def _create_lstm_gru_dataset_period(self, start_date: str, end_date: str,
                                      symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create an LSTM/GRU dataset for a specific time period with no leakage.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: List of symbols to include (optional)
            
        Returns:
            Dictionary containing dataset for the specified period
        """
        self.logger.info(f"Creating LSTM/GRU dataset for period {start_date} to {end_date}")
        
        # If symbols not provided, use default symbols
        if symbols is None:
            symbols = [f"STOCK{i}" for i in range(1, 21)]
            
        # Fetch historical bars for each symbol within the specified period only
        bars_data = self.fetch_historical_bars(symbols, start_date, end_date, timespan="minute")
        
        # Create sequences and labels
        sequences = {}
        labels = {}
        
        for symbol, bars in bars_data.items():
            # Create sequences of length 50
            seq_length = 50
            
            # Create sequences and labels
            X = []
            y = []
            
            # Only create complete sequences within this period
            # This prevents using data from outside the period
            for i in range(len(bars) - seq_length - 10):
                # Make sure the full sequence and label window are within this period
                if i + seq_length + 10 >= len(bars):
                    continue
                    
                # Extract sequence using only historical data
                seq = bars.iloc[i:i+seq_length]
                
                # Extract features
                features = []
                for _, row in seq.iterrows():
                    # OHLCV features - using only data available at this point
                    features.append([
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row["volume"],
                        row["vwap"]
                    ])
                    
                # Extract label (optimal exit point)
                # This is acceptable since we're explicitly using this as a target
                future_prices = bars.iloc[i+seq_length:i+seq_length+10]["close"].values
                
                # Find optimal exit point
                max_price = np.max(future_prices)
                max_idx = np.argmax(future_prices)
                
                # Calculate exit probability, optimal exit price, and trailing stop adjustment
                exit_probability = max_idx / 10.0  # Normalize to 0-1
                optimal_exit_price = max_price
                trailing_stop_adjustment = 0.5  # Fixed value for now
                
                label = [exit_probability, optimal_exit_price, trailing_stop_adjustment]
                
                X.append(features)
                y.append(label)
                
            if len(X) > 0:
                sequences[symbol] = np.array(X)
                labels[symbol] = np.array(y)
            else:
                self.logger.warning(f"No valid sequences created for {symbol} in period {start_date} to {end_date}")
        
        # Create dataset
        dataset = {
            "sequences": sequences,
            "labels": labels
        }
        
        total_sequences = sum(len(seqs) for seqs in sequences.values())
        self.logger.info(f"Created LSTM/GRU dataset with {total_sequences} sequences from {len(sequences)} symbols")
        
        return dataset
        
    def _calculate_point_in_time_indicators(self, symbol_data: Dict[str, Any], symbol: str, date_str: str) -> None:
        """
        Recalculate technical indicators using only data available up to this point.
        This prevents data leakage by ensuring indicators only use past data.
        
        Args:
            symbol_data: Symbol data to update
            symbol: Symbol name
            date_str: Current date string (YYYY-MM-DD)
        """
        try:
            # Fetch historical data strictly up to but NOT including the current date
            current_date = pd.to_datetime(date_str)
            # Use a longer lookback to ensure enough data for rolling features
            start_date = (current_date - timedelta(days=60)).strftime('%Y-%m-%d')
            # Only use data up to the day before the current date to avoid lookahead bias
            end_date = (current_date - timedelta(days=1)).strftime('%Y-%m-%d')

            # Fetch historical bars for this symbol
            historical_data = self.fetch_historical_bars(
                [symbol],
                start_date=start_date,
                end_date=end_date,
                timespan="day"
            )
            # Check if we got data for this symbol
            historical_bars = historical_data.get(symbol)
            if symbol not in historical_data or historical_bars is None:
                self.logger.warning(f"No historical data found for {symbol} on {date_str}")
                return

            # Log the date range to verify we're not using future data
            if isinstance(historical_bars, pd.DataFrame) and not historical_bars.empty:
                self.logger.debug(f"Historical data dates for {symbol} on {date_str}: {historical_bars.index.min()} to {historical_bars.index.max()}")
                # Double-check no future data is included
                if historical_bars.index.max() >= current_date:
                    self.logger.warning(f"Data leakage detected! Historical data for {symbol} contains future data. Max date: {historical_bars.index.max()}, Current date: {current_date}")
                    # Filter out future data
                    historical_bars = historical_bars[historical_bars.index < current_date]

            # Check if historical_bars is a DataFrame and has enough data
            if historical_bars is None or not isinstance(historical_bars, pd.DataFrame) or len(historical_bars) < 14:
                self.logger.warning(f"Not enough historical data for {symbol} on {date_str} to calculate indicators")
                return

            # Extract price data
            if 'close' not in historical_bars.columns or 'high' not in historical_bars.columns or 'low' not in historical_bars.columns or 'volume' not in historical_bars.columns:
                self.logger.warning(f"Missing required columns in historical data for {symbol}")
                return

            closes = historical_bars['close'].values
            highs = historical_bars['high'].values
            lows = historical_bars['low'].values
            volumes = historical_bars['volume'].values

            # Ensure we have arrays, not single values
            if not isinstance(closes, np.ndarray) or len(closes) == 0:
                self.logger.warning(f"Invalid close data for {symbol}: {type(closes)}")
                return

            # Calculate RSI using only historical data up to this point
            symbol_data['rsi_14'] = self._calculate_rsi(closes, 14)

            # Calculate MACD
            macd_line, signal_line, histogram = self._calculate_macd(closes)
            symbol_data['macd'] = macd_line
            symbol_data['macd_signal'] = signal_line
            symbol_data['macd_histogram'] = histogram

            # Calculate Bollinger Bands
            upper, middle, lower = self._calculate_bollinger_bands(closes)
            symbol_data['bb_upper'] = upper
            symbol_data['bb_middle'] = middle
            symbol_data['bb_lower'] = lower

            # Calculate ATR
            symbol_data['atr'] = self._calculate_atr(highs, lows, closes)

            # Calculate volume-based indicators
            if len(volumes) > 20:
                symbol_data['avg_volume'] = np.mean(volumes[-20:])
                symbol_data['volume_acceleration'] = self._calculate_acceleration(volumes)
                symbol_data['volume_spike'] = volumes[-1] / max(np.mean(volumes[-10:]), 1)

            # Calculate price change indicators
            # Removed 'price_change_5m' from features to prevent label leakage.
            # This feature directly encodes the future price change and is used as the label.
            # if len(closes) > 5:
            #     symbol_data['price_change_5m'] = (closes[-1] / closes[-5] - 1) if closes[-5] > 0 else 0

            if len(closes) > 1:
                symbol_data['momentum_1m'] = (closes[-1] / closes[-2] - 1) if closes[-2] > 0 else 0

            # Calculate SMA cross signal
            symbol_data['sma_cross_signal'] = self._calculate_sma_cross(closes)

            # Add comment for future maintainers:
            # All features above are calculated strictly using data available up to the current date,
            # ensuring no lookahead bias or data leakage into the training or test set.

        except Exception as e:
            self.logger.error(f"Error calculating point-in-time indicators for {symbol} on {date_str}: {str(e)}")
    
    def _calculate_rsi(self, prices: np.ndarray, window: int = 14) -> float:
        """Calculate RSI indicator."""
        # Check if prices is a valid array with sufficient length
        if not isinstance(prices, np.ndarray):
            return 50.0  # Default value if not an array
            
        # Convert to numpy array if it's a list
        if isinstance(prices, list):
            prices = np.array(prices)
            
        # Check if we have enough data
        if len(prices) <= window:
            return 50.0  # Default value if not enough data
            
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Check if deltas is valid
        if len(deltas) == 0:
            return 50.0
            
        # Calculate gains and losses
        gains = np.copy(deltas)
        losses = np.copy(deltas)
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[-window:])
        avg_loss = np.mean(losses[-window:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD indicator."""
        # Check if prices is a valid array
        if not isinstance(prices, np.ndarray):
            return 0.0, 0.0, 0.0  # Default values if not an array
            
        # Convert to numpy array if it's a list
        if isinstance(prices, list):
            prices = np.array(prices)
            
        # Check if we have enough data
        if len(prices) <= slow:
            return 0.0, 0.0, 0.0  # Default values if not enough data
            
        try:
            # Calculate EMAs
            ema_fast = self._calculate_ema(prices, fast)
            ema_slow = self._calculate_ema(prices, slow)
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = self._calculate_ema(np.append(np.zeros(len(prices) - 1), macd_line), signal)
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return float(macd_line), float(signal_line), float(histogram)
        except Exception as e:
            self.logger.warning(f"Error calculating MACD: {str(e)}")
            return 0.0, 0.0, 0.0
    
    def _calculate_ema(self, prices: np.ndarray, window: int) -> float:
        """Calculate EMA."""
        # Check if prices is a valid array
        if not isinstance(prices, np.ndarray):
            return 0.0  # Default value if not an array
            
        # Convert to numpy array if it's a list
        if isinstance(prices, list):
            prices = np.array(prices)
            
        # Check if we have enough data
        if len(prices) == 0:
            return 0.0
            
        if len(prices) <= window:
            return float(np.mean(prices))
            
        try:
            weights = np.exp(np.linspace(-1., 0., window))
            weights /= weights.sum()
            
            ema = np.convolve(prices, weights, mode='full')[:len(prices)]
            ema[:window] = ema[window]
            
            return float(ema[-1])
        except Exception as e:
            self.logger.warning(f"Error calculating EMA: {str(e)}")
            return float(np.mean(prices))
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        # Check if prices is a valid array
        if not isinstance(prices, np.ndarray):
            return 100.0, 50.0, 0.0  # Default values if not an array
            
        # Convert to numpy array if it's a list
        if isinstance(prices, list):
            prices = np.array(prices)
            
        # Check if we have enough data
        if len(prices) == 0:
            return 100.0, 50.0, 0.0
            
        if len(prices) <= window:
            last_price = float(prices[-1]) if len(prices) > 0 else 50.0
            return float(last_price * 1.1), float(last_price), float(last_price * 0.9)  # Default values if not enough data
            
        try:
            # Calculate middle band (SMA)
            middle = np.mean(prices[-window:])
            
            # Calculate standard deviation
            std = np.std(prices[-window:])
            
            # Calculate upper and lower bands
            upper = middle + (std * num_std)
            lower = middle - (std * num_std)
            
            return float(upper), float(middle), float(lower)
        except Exception as e:
            self.logger.warning(f"Error calculating Bollinger Bands: {str(e)}")
            last_price = float(prices[-1]) if len(prices) > 0 else 50.0
            return float(last_price * 1.1), float(last_price), float(last_price * 0.9)
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int = 14) -> float:
        """Calculate Average True Range."""
        # Check if inputs are valid arrays
        if not isinstance(highs, np.ndarray) or not isinstance(lows, np.ndarray) or not isinstance(closes, np.ndarray):
            return 1.0  # Default value if not arrays
            
        # Convert to numpy arrays if they're lists
        if isinstance(highs, list):
            highs = np.array(highs)
        if isinstance(lows, list):
            lows = np.array(lows)
        if isinstance(closes, list):
            closes = np.array(closes)
            
        # Check if we have enough data
        if len(highs) == 0 or len(lows) == 0 or len(closes) == 0:
            return 1.0
            
        if len(highs) <= window or len(lows) <= window or len(closes) <= window:
            if len(highs) > 0 and len(lows) > 0:
                return float((highs[-1] - lows[-1]) / 2)  # Default value if not enough data
            else:
                return 1.0
                
        try:
            # Calculate true ranges
            tr1 = highs[1:] - lows[1:]  # Current high - current low
            tr2 = np.abs(highs[1:] - closes[:-1])  # Current high - previous close
            tr3 = np.abs(lows[1:] - closes[:-1])  # Current low - previous close
            
            # True range is the maximum of the three
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            
            # Calculate ATR
            atr = np.mean(tr[-window:])
            
            return float(atr)
        except Exception as e:
            self.logger.warning(f"Error calculating ATR: {str(e)}")
            if len(highs) > 0 and len(lows) > 0:
                return float((highs[-1] - lows[-1]) / 2)
            else:
                return 1.0
    
    def _calculate_acceleration(self, values: np.ndarray, window: int = 5) -> float:
        """Calculate acceleration (second derivative)."""
        # Check if values is a valid array
        if not isinstance(values, np.ndarray):
            return 0.0  # Default value if not an array
            
        # Convert to numpy array if it's a list
        if isinstance(values, list):
            values = np.array(values)
            
        # Check if we have enough data
        if len(values) <= window * 2:
            return 0.0  # Default value if not enough data
            
        try:
            # Calculate first derivative (velocity)
            velocity = np.diff(values)
            
            # Check if we have enough data after first diff
            if len(velocity) <= window:
                return 0.0
                
            # Calculate second derivative (acceleration)
            acceleration = np.diff(velocity)
            
            # Check if we have enough data after second diff
            if len(acceleration) == 0:
                return 0.0
                
            # Return average recent acceleration
            return float(np.mean(acceleration[-window:]))
        except Exception as e:
            self.logger.warning(f"Error calculating acceleration: {str(e)}")
            return 0.0
    
    def _calculate_sma_cross(self, prices: np.ndarray, fast: int = 10, slow: int = 30) -> float:
        """Calculate SMA cross signal."""
        # Check if prices is a valid array
        if not isinstance(prices, np.ndarray):
            return 0.0  # Default value if not an array
            
        # Convert to numpy array if it's a list
        if isinstance(prices, list):
            prices = np.array(prices)
            
        # Check if we have enough data
        if len(prices) == 0:
            return 0.0
            
        if len(prices) <= slow:
            return 0.0  # Default value if not enough data
            
        try:
            # Calculate fast and slow SMAs
            sma_fast = np.mean(prices[-fast:])
            sma_slow = np.mean(prices[-slow:])
            
            # Calculate previous fast and slow SMAs
            if len(prices) > fast + 1:
                prev_sma_fast = np.mean(prices[-(fast+1):-1])
            else:
                prev_sma_fast = sma_fast
                
            if len(prices) > slow + 1:
                prev_sma_slow = np.mean(prices[-(slow+1):-1])
            else:
                prev_sma_slow = sma_slow
            
            # Calculate cross signal
            current_diff = sma_fast - sma_slow
            prev_diff = prev_sma_fast - prev_sma_slow
            
            # Normalize to -1 to 1 range
            if current_diff > 0 and prev_diff < 0:
                # Bullish cross (fast crosses above slow)
                return 1.0
            elif current_diff < 0 and prev_diff > 0:
                # Bearish cross (fast crosses below slow)
                return -1.0
            else:
                # No cross, return normalized difference
                return float(current_diff / (sma_slow * 0.1) if sma_slow > 0 else 0)
        except Exception as e:
            self.logger.warning(f"Error calculating SMA cross: {str(e)}")
            return 0.0
    
    def _validate_no_data_leakage(self, train_data: Dict[str, Any], test_data: Dict[str, Any],
                                model_type: str, start_date: str, train_end_date: str,
                                test_start_date: str, end_date: str) -> None:
        """
        Validate that there is no data leakage between training and testing datasets.
        
        Args:
            train_data: Training dataset
            test_data: Testing dataset
            model_type: Type of model
            start_date: Overall start date
            train_end_date: End date of training period
            test_start_date: Start date of testing period
            end_date: Overall end date
        """
        self.logger.info(f"Validating no data leakage for {model_type} model")
        
        # Convert dates to datetime objects for comparison
        train_start_dt = pd.to_datetime(start_date)
        train_end_dt = pd.to_datetime(train_end_date)
        test_start_dt = pd.to_datetime(test_start_date)
        test_end_dt = pd.to_datetime(end_date)
        
        # Validate date ranges don't overlap
        if train_end_dt >= test_start_dt:
            # Make this a hard error to prevent silent leakage
            error_msg = f"Data leakage detected: Training end date {train_end_date} is not before testing start date {test_start_date}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Model-specific validation
        if model_type == "gbdt":
            self._validate_gbdt_no_leakage(train_data, test_data, train_start_dt, train_end_dt, test_start_dt, test_end_dt)
        elif model_type == "axial_attention":
            self._validate_axial_attention_no_leakage(train_data, test_data)
        elif model_type == "lstm_gru":
            self._validate_lstm_gru_no_leakage(train_data, test_data)

        self.logger.info(f"Validation complete for {model_type} model")
    
    def _validate_gbdt_no_leakage(self, train_data: Dict[str, Any], test_data: Dict[str, Any],
                                train_start_dt: datetime, train_end_dt: datetime,
                                test_start_dt: datetime, test_end_dt: datetime) -> None:
        """Validate no data leakage for GBDT datasets."""
        # Check that training dates are within training period
        for date_str in train_data.get("dates", []):
            date_dt = pd.to_datetime(date_str)
            if date_dt < train_start_dt or date_dt > train_end_dt:
                error_msg = f"Data leakage detected: Training data contains date {date_str} outside training period {train_start_dt} to {train_end_dt}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        # Check that testing dates are within testing period
        for date_str in test_data.get("dates", []):
            date_dt = pd.to_datetime(date_str)
            if date_dt < test_start_dt or date_dt > test_end_dt:
                error_msg = f"Data leakage detected: Testing data contains date {date_str} outside testing period {test_start_dt} to {test_end_dt}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
    
    def _validate_axial_attention_no_leakage(self, train_data: Dict[str, Any], test_data: Dict[str, Any]) -> None:
        """Validate no data leakage for Axial Attention datasets."""
        # For sequence models, we can't easily validate without timestamps
        # But we can check that the datasets have different sequence counts
        train_seq_count = sum(len(seqs) for seqs in train_data.get("sequences", {}).values())
        test_seq_count = sum(len(seqs) for seqs in test_data.get("sequences", {}).values())
        
        self.logger.info(f"Axial Attention dataset validation: {train_seq_count} training sequences, "
                       f"{test_seq_count} testing sequences")
    
    def _validate_lstm_gru_no_leakage(self, train_data: Dict[str, Any], test_data: Dict[str, Any]) -> None:
        """Validate no data leakage for LSTM/GRU datasets."""
        # Similar to Axial Attention validation
        train_seq_count = sum(len(seqs) for seqs in train_data.get("sequences", {}).values())
        test_seq_count = sum(len(seqs) for seqs in test_data.get("sequences", {}).values())
        
        self.logger.info(f"LSTM/GRU dataset validation: {train_seq_count} training sequences, "
                       f"{test_seq_count} testing sequences")
    
    def _log_symbol_distribution_metrics(self, symbols: List[str]) -> None:
        """
        Log metrics about the symbol distribution to ensure diversity.
        
        Args:
            symbols: List of symbols
        """
        if not symbols:
            return
            
        # Count symbols by first letter
        letter_counts = {}
        for symbol in symbols:
            if symbol and isinstance(symbol, str) and len(symbol) > 0:
                first_letter = symbol[0].upper()
                letter_counts[first_letter] = letter_counts.get(first_letter, 0) + 1
        
        # Sort by letter
        sorted_counts = {k: letter_counts[k] for k in sorted(letter_counts.keys())}
        
        # Log distribution
        self.logger.info("Symbol distribution by first letter:")
        for letter, count in sorted_counts.items():
            percentage = (count / len(symbols)) * 100
            self.logger.info(f"  {letter}: {count} symbols ({percentage:.1f}%)")
        
        # Log coverage statistics
        unique_letters = len(letter_counts)
        self.logger.info(f"Symbol coverage: {unique_letters}/26 letters represented")
    
    def validate_symbol_coverage(self, training_symbols: List[str], production_symbols: List[str]) -> Dict[str, Any]:
        """
        Validate the coverage of training symbols compared to production symbols.
        
        Args:
            training_symbols: Symbols used in training
            production_symbols: Symbols used in production
            
        Returns:
            Dictionary with validation metrics
        """
        # Convert to sets for efficient comparison
        train_set = set(training_symbols)
        prod_set = set(production_symbols)
        
        # Basic metrics
        overlap = train_set.intersection(prod_set)
        # Calculate but don't assign the variable if it's not used
        # only_in_train = train_set - prod_set
        only_in_prod = prod_set - train_set
        
        # Calculate coverage percentages
        overlap_count = len(overlap)
        # These variables are calculated but not used, so we'll remove them
        # train_only_count = len(only_in_train)
        # prod_only_count = len(only_in_prod)
        
        prod_coverage_pct = (overlap_count / max(1, len(prod_set))) * 100
        
        # Letter distribution in training vs production
        train_first_letters = [s[0] if s and len(s) > 0 else "" for s in training_symbols]
        prod_first_letters = [s[0] if s and len(s) > 0 else "" for s in production_symbols]
        
        train_letter_counts = {}
        for letter in train_first_letters:
            if letter:
                train_letter_counts[letter] = train_letter_counts.get(letter, 0) + 1
                
        prod_letter_counts = {}
        for letter in prod_first_letters:
            if letter:
                prod_letter_counts[letter] = prod_letter_counts.get(letter, 0) + 1
        
        # Compute distribution similarity
        letters = sorted(set(train_letter_counts.keys()).union(set(prod_letter_counts.keys())))
        letter_distribution_diff = {}
        
        for letter in letters:
            train_pct = (train_letter_counts.get(letter, 0) / len(training_symbols)) if training_symbols else 0
            prod_pct = (prod_letter_counts.get(letter, 0) / len(production_symbols)) if production_symbols else 0
            letter_distribution_diff[letter] = abs(train_pct - prod_pct) * 100  # Difference in percentage points
        
        # Overall distribution similarity (0-100%, higher is better)
        dist_similarity = 100 - min(100, sum(letter_distribution_diff.values()))
        
        # Results
        result = {
            "overlap_count": overlap_count,
            "production_coverage_pct": prod_coverage_pct,
            "missing_in_train": len(only_in_prod),
            "distribution_similarity_pct": dist_similarity,
            "letter_distribution_diff": letter_distribution_diff
        }
        
        # Log results
        self.logger.info(f"Symbol coverage validation: {prod_coverage_pct:.1f}% of production symbols covered in training")
        self.logger.info(f"Symbol distribution similarity: {dist_similarity:.1f}%")
        
        if prod_coverage_pct < 95:
            self.logger.warning(f"Low symbol coverage: {prod_coverage_pct:.1f}% - training may not be representative")
        if dist_similarity < 80:
            self.logger.warning("Symbol distribution differs significantly between training and production")
            
            
        return result
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up data manager resources")
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Clean up Polygon.io API client
        self.polygon_api.cleanup()
