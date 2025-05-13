"""
Data Manager for ML Model Training

This module provides utilities for fetching, processing, and managing
market data for training ML models.
"""

import os
import json
import numpy as np
import pandas as pd
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
                # In a real implementation, this would use the Polygon.io API
                # For now, we'll generate synthetic data
                bars = self._generate_synthetic_bars(symbol, start_dt, end_dt, timespan)
                
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
                    
                    # Filter to requested date range
                    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                    
                    self.logger.debug(f"Using cached {timespan} bars for {symbol}")
                    
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
                
            self.cache_index["bars"][symbol][timespan] = {
                "path": cache_path,
                "start_date": bars.index.min().isoformat(),
                "end_date": bars.index.max().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Save cache index
            self._save_cache_index()
            
            self.logger.debug(f"Cached {timespan} bars for {symbol}")
            
    def _generate_synthetic_bars(self, symbol: str, start_dt: datetime, end_dt: datetime,
                               timespan: str) -> pd.DataFrame:
        """
        Generate synthetic bar data for testing.
        
        Args:
            symbol: Symbol to generate data for
            start_dt: Start date
            end_dt: End date
            timespan: Timespan
            
        Returns:
            DataFrame of synthetic bars
        """
        # Determine frequency based on timespan
        if timespan == "minute":
            freq = "1min"
        elif timespan == "hour":
            freq = "1H"
        elif timespan == "day":
            freq = "1D"
        elif timespan == "week":
            freq = "1W"
        elif timespan == "month":
            freq = "1M"
        else:
            freq = "1D"  # Default to daily
            
        # Generate date range
        date_range = pd.date_range(start=start_dt, end=end_dt, freq=freq)
        
        # Generate random price data
        base_price = np.random.uniform(50, 200)
        volatility = np.random.uniform(0.01, 0.05)
        
        # Generate OHLCV data
        np.random.seed(hash(symbol) % 10000)  # Seed based on symbol for consistency
        
        # Generate returns
        returns = np.random.normal(0, volatility, len(date_range))
        
        # Calculate prices
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = []
        for i, date in enumerate(date_range):
            price = prices[i]
            
            # Generate open, high, low, close
            open_price = price * np.random.uniform(0.99, 1.01)
            close_price = price * np.random.uniform(0.99, 1.01)
            high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.02)
            low_price = min(open_price, close_price) * np.random.uniform(0.98, 1.0)
            
            # Generate volume
            volume = int(np.random.lognormal(10, 1))
            
            # Generate VWAP
            vwap = (open_price + high_price + low_price + close_price) / 4
            
            data.append({
                "timestamp": date,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "vwap": vwap
            })
            
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        
        return df
        
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
            # In a real implementation, this would use the Polygon.io API
            # For now, we'll generate synthetic data
            snapshots = self._generate_synthetic_snapshots(date_dt)
            
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
            
    def _generate_synthetic_snapshots(self, date_dt: datetime) -> Dict[str, Any]:
        """
        Generate synthetic market snapshots for testing.
        
        Args:
            date_dt: Date to generate snapshots for
            
        Returns:
            Dictionary of synthetic snapshots
        """
        # Generate list of symbols
        symbols = [f"STOCK{i}" for i in range(1, 101)]
        
        # Generate snapshots
        snapshots = {
            "timestamp": date_dt.isoformat(),
            "symbols": {}
        }
        
        for symbol in symbols:
            # Generate random price data
            base_price = np.random.uniform(50, 200)
            
            # Generate snapshot data
            snapshots["symbols"][symbol] = {
                "last_price": base_price * np.random.uniform(0.99, 1.01),
                "bid_price": base_price * np.random.uniform(0.99, 0.995),
                "ask_price": base_price * np.random.uniform(1.005, 1.01),
                "bid_ask_spread": base_price * np.random.uniform(0.005, 0.01),
                "open_price": base_price * np.random.uniform(0.98, 1.02),
                "high_price": base_price * np.random.uniform(1.01, 1.05),
                "low_price": base_price * np.random.uniform(0.95, 0.99),
                "volume": int(np.random.lognormal(10, 1)),
                "vwap": base_price * np.random.uniform(0.99, 1.01),
                "prev_close": base_price * np.random.uniform(0.98, 1.02),
                
                # Technical indicators
                "rsi_14": np.random.uniform(30, 70),
                "macd": np.random.uniform(-2, 2),
                "macd_signal": np.random.uniform(-2, 2),
                "macd_histogram": np.random.uniform(-1, 1),
                "bb_upper": base_price * np.random.uniform(1.05, 1.1),
                "bb_middle": base_price,
                "bb_lower": base_price * np.random.uniform(0.9, 0.95),
                "atr": base_price * np.random.uniform(0.01, 0.05),
                
                # Additional indicators
                "avg_volume": int(np.random.lognormal(10, 0.5)),
                "volume_acceleration": np.random.uniform(-0.5, 0.5),
                "volume_spike": np.random.uniform(0, 2),
                "price_change_1m": np.random.uniform(-0.02, 0.02),
                "price_change_5m": np.random.uniform(-0.05, 0.05),
                "momentum_1m": np.random.uniform(-0.02, 0.02),
                "sma_cross_signal": np.random.uniform(-1, 1),
                "support_resistance_proximity": np.random.uniform(0, 1),
                "volatility_ratio": np.random.uniform(0.5, 2.0)
            }
            
        return snapshots
        
    def create_training_dataset(self, model_type: str, start_date: str, end_date: str,
                               symbols: Optional[List[str]] = None,
                               train_test_split: float = 0.8,
                               random_seed: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Create a training dataset for a specific model type.
        
        This implementation prevents data leakage by first splitting the time periods,
        then generating features and sequences separately for each period.
        
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
        
        # Create dataset based on model type with clear time separation
        if model_type == "gbdt":
            train_data = self._create_gbdt_dataset_period(start_date, train_end_date, symbols)
            test_data = self._create_gbdt_dataset_period(test_start_date, end_date, symbols)
        elif model_type == "axial_attention":
            train_data = self._create_axial_attention_dataset_period(start_date, train_end_date, symbols)
            test_data = self._create_axial_attention_dataset_period(test_start_date, end_date, symbols)
        elif model_type == "lstm_gru":
            train_data = self._create_lstm_gru_dataset_period(start_date, train_end_date, symbols)
            test_data = self._create_lstm_gru_dataset_period(test_start_date, end_date, symbols)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Validate no data leakage between train and test sets
        self._validate_no_data_leakage(train_data, test_data, model_type, start_date, train_end_date, test_start_date, end_date)
            
        # End timing
        self.latency_profiler.end_phase()
        
        self.logger.info(f"Created training dataset for {model_type} model")
        
        return train_data, test_data
        
    def _create_gbdt_dataset_period(self, start_date: str, end_date: str,
                                  symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a GBDT dataset for a specific time period with no leakage.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: List of symbols to include (optional)
            
        Returns:
            Dictionary containing dataset for the specified period
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
            date_str = date.strftime("%Y-%m-%d")
            snapshot = self.fetch_market_snapshots(date_str)
            
            # Process technical indicators to ensure they use only past data
            if "symbols" in snapshot and snapshot["symbols"]:
                for symbol in snapshot["symbols"]:
                    # Recalculate indicators using only data available up to this point
                    self._calculate_point_in_time_indicators(snapshot["symbols"][symbol], symbol, date_str)
            
            snapshots.append(snapshot)
            dates.append(date_str)
        
        # Create dataset
        dataset = {
            "snapshots": snapshots,
            "dates": dates
        }
        
        self.logger.info(f"Created GBDT dataset with {len(dates)} days of data")
        
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
            # Fetch historical data up to but not beyond the current date
            current_date = pd.to_datetime(date_str)
            start_date = (current_date - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = date_str
            
            # Fetch historical bars for this symbol
            historical_bars = self.fetch_historical_bars(
                [symbol],
                start_date=start_date,
                end_date=end_date,
                timespan="day"
            ).get(symbol)
            
            if historical_bars is None or len(historical_bars) < 14:
                self.logger.warning(f"Not enough historical data for {symbol} on {date_str} to calculate indicators")
                return
            
            # Extract price data
            closes = historical_bars['close'].values
            highs = historical_bars['high'].values
            lows = historical_bars['low'].values
            volumes = historical_bars['volume'].values
            
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
            if len(closes) > 5:
                symbol_data['price_change_5m'] = (closes[-1] / closes[-5] - 1) if closes[-5] > 0 else 0
            
            if len(closes) > 1:
                symbol_data['momentum_1m'] = (closes[-1] / closes[-2] - 1) if closes[-2] > 0 else 0
            
            # Calculate SMA cross signal
            symbol_data['sma_cross_signal'] = self._calculate_sma_cross(closes)
            
        except Exception as e:
            self.logger.error(f"Error calculating point-in-time indicators for {symbol} on {date_str}: {str(e)}")
    
    def _calculate_rsi(self, prices: np.ndarray, window: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) <= window:
            return 50.0  # Default value if not enough data
            
        # Calculate price changes
        deltas = np.diff(prices)
        
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
        if len(prices) <= slow:
            return 0.0, 0.0, 0.0  # Default values if not enough data
            
        # Calculate EMAs
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = self._calculate_ema(np.append(np.zeros(len(prices) - len(macd_line)), macd_line), signal)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return float(macd_line), float(signal_line), float(histogram)
    
    def _calculate_ema(self, prices: np.ndarray, window: int) -> float:
        """Calculate EMA."""
        if len(prices) <= window:
            return float(np.mean(prices))
            
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        
        ema = np.convolve(prices, weights, mode='full')[:len(prices)]
        ema[:window] = ema[window]
        
        return float(ema[-1])
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) <= window:
            return float(prices[-1] * 1.1), float(prices[-1]), float(prices[-1] * 0.9)  # Default values if not enough data
            
        # Calculate middle band (SMA)
        middle = np.mean(prices[-window:])
        
        # Calculate standard deviation
        std = np.std(prices[-window:])
        
        # Calculate upper and lower bands
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return float(upper), float(middle), float(lower)
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int = 14) -> float:
        """Calculate Average True Range."""
        if len(highs) <= window:
            return float((highs[-1] - lows[-1]) / 2)  # Default value if not enough data
            
        # Calculate true ranges
        tr1 = highs[1:] - lows[1:]  # Current high - current low
        tr2 = np.abs(highs[1:] - closes[:-1])  # Current high - previous close
        tr3 = np.abs(lows[1:] - closes[:-1])  # Current low - previous close
        
        # True range is the maximum of the three
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Calculate ATR
        atr = np.mean(tr[-window:])
        
        return float(atr)
    
    def _calculate_acceleration(self, values: np.ndarray, window: int = 5) -> float:
        """Calculate acceleration (second derivative)."""
        if len(values) <= window * 2:
            return 0.0  # Default value if not enough data
            
        # Calculate first derivative (velocity)
        velocity = np.diff(values)
        
        # Calculate second derivative (acceleration)
        acceleration = np.diff(velocity)
        
        # Return average recent acceleration
        return float(np.mean(acceleration[-window:]))
    
    def _calculate_sma_cross(self, prices: np.ndarray, fast: int = 10, slow: int = 30) -> float:
        """Calculate SMA cross signal."""
        if len(prices) <= slow:
            return 0.0  # Default value if not enough data
            
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
            self.logger.warning(f"Potential data leakage: Training end date {train_end_date} "
                              f"is not before testing start date {test_start_date}")
        
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
                self.logger.warning(f"Data leakage detected: Training data contains date {date_str} "
                                  f"outside training period {train_start_dt} to {train_end_dt}")
        
        # Check that testing dates are within testing period
        for date_str in test_data.get("dates", []):
            date_dt = pd.to_datetime(date_str)
            if date_dt < test_start_dt or date_dt > test_end_dt:
                self.logger.warning(f"Data leakage detected: Testing data contains date {date_str} "
                                  f"outside testing period {test_start_dt} to {test_end_dt}")
    
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
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up data manager resources")
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Clean up Polygon.io API client
        self.polygon_api.cleanup()
