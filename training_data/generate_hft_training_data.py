#!/usr/bin/env python3
"""
HFT Training Data Generator

This script fetches a complete snapshot of the US stock market from Polygon.io,
computes technical indicators, and formats the data for training and calibrating
the HFT model.
"""

import json
import time
import logging
import argparse
import os
import numpy as np
import pandas as pd
import requests
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("hft_training_data")

# Use orjson if available (faster), otherwise fallback to standard json
try:
    import orjson as json_fast
except ImportError:
    try:
        import ujson as json_fast
    except ImportError:
        import json as json_fast
        logger.warning("Using standard json module. Consider installing orjson or ujson for better performance.")


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
    
    def to_feature_array(self) -> np.ndarray:
        """Convert snapshot to feature array for model input"""
        features = np.array([
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
            self.momentum_1m
        ], dtype=np.float32)
        return features

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {k: v for k, v in self.__dict__.items()}


class IndicatorState:
    """
    Class to store state for incremental indicator calculations
    """
    def __init__(self, period=14):
        # General state
        self.last_update = 0  # Timestamp of last update
        
        # Price history (limited window)
        self.close_prices = deque(maxlen=50)  # Store recent prices
        self.timestamps = deque(maxlen=50)    # Store timestamps for cleanup
        
        # RSI state
        self.rsi_period = period
        self.rsi_gains = deque(maxlen=period+1)
        self.rsi_losses = deque(maxlen=period+1)
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
        self.volume_data = deque(maxlen=10)  # For volume acceleration
        self.volume_value = 0.0


class DirectPolygonClient:
    """
    Client for directly fetching data from Polygon.io API without dependencies
    """
    
    def __init__(self, min_price=5.0, max_price=500.0, min_volume=500000, min_market_cap=500000000, max_volume=None):
        """
        Initialize the direct Polygon client
        
        Args:
            min_price: Minimum price filter
            max_price: Maximum price filter
            min_volume: Minimum volume filter
            min_market_cap: Minimum market cap filter
        """
        # Load API key from environment
        load_dotenv()
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable not set")
            
        self.base_url = os.getenv("POLYGON_BASE_URL", "https://api.polygon.io")
        
        # Initialize filtering settings
        self.min_price = min_price
        self.max_price = max_price
        self.min_volume = min_volume
        self.min_market_cap = min_market_cap
        self.max_volume = max_volume
        
        # Indicator states for calculating technical indicators
        self.indicator_states = {}
        
        # Configure HTTP session with retries and timeouts
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        # Create adapter for connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=4,
            pool_maxsize=8
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Configure timeouts
        self.connect_timeout = 3.0
        self.read_timeout = 30.0  # Increased for full market snapshots
        
        logger.info(f"DirectPolygonClient initialized with API key: {self.api_key[:4]}...{self.api_key[-4:]}")
    
    def fetch_full_market_snapshot(self, include_otc: bool = False, apply_filters: bool = True) -> Dict[str, Any]:
        """
        Fetch a complete snapshot of the US stock market
        
        Args:
            include_otc: Whether to include OTC securities
            apply_filters: Whether to apply filtering criteria
            
        Returns:
            Complete market snapshot data
        """
        # Build URL
        url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers"
        params = {
            "apiKey": self.api_key,
            "include_otc": str(include_otc).lower()
        }
        
        logger.info(f"Fetching full market snapshot (include_otc={include_otc})...")
        
        try:
            # Perform request with timeout (use longer timeout for full snapshot)
            response = self.session.get(
                url,
                params=params,
                timeout=(self.connect_timeout, self.read_timeout)
            )
            
            # Raise for status
            response.raise_for_status()
            
            # Parse JSON response using fast JSON parser
            response_json = json_fast.loads(response.text)
            
            # Check status
            if response_json.get("status") != "OK":
                error_msg = f"API error: {response_json.get('status')}"
                logger.error(error_msg)
                return None
            
            ticker_count = response_json.get("count", 0)
            logger.info(f"Successfully fetched snapshot for {ticker_count} tickers")
            
            # Apply filters if requested
            if apply_filters:
                response_json = self.filter_market_snapshot(response_json)
                
            return response_json
            
        except requests.exceptions.Timeout:
            logger.error("Timeout while fetching full market snapshot")
            return None
        except Exception as e:
            logger.error(f"Error fetching full market snapshot from Polygon: {e}")
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
        above_max_volume = 0
        missing_data = 0
        
        # Filter tickers based on criteria
        filtered_tickers = []
        
        for ticker in original_tickers:
            # Initialize filter flags
            price_ok = True
            volume_ok = True
            
            # Extract data
            ticker_symbol = ticker.get("ticker", "UNKNOWN")
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
            
            # Check max volume if specified
            if self.max_volume is not None and volume is not None and volume > self.max_volume:
                above_max_volume += 1
                volume_ok = False
            
            # If criteria are met, keep the ticker
            if price_ok and volume_ok:
                filtered_tickers.append(ticker)
        
        # Update snapshot data with filtered tickers
        filtered_count = len(filtered_tickers)
        
        # Log filtering results
        if self.min_price > 0:
            logger.info(f"Filtered out {below_min_price} tickers below ${self.min_price}")
        
        if self.max_price > 0:
            logger.info(f"Filtered out {above_max_price} tickers above ${self.max_price}")
        
        if self.min_volume > 0:
            logger.info(f"Filtered out {below_min_volume} tickers with volume below {self.min_volume}")
        
        if self.max_volume is not None:
            logger.info(f"Filtered out {above_max_volume} tickers with volume above {self.max_volume}")
        
        if missing_data > 0:
            logger.info(f"Skipped {missing_data} tickers with missing price data")
        
        logger.info(f"Kept {filtered_count} tickers meeting all criteria")
        
        filtered_snapshot = snapshot_data.copy()
        filtered_snapshot["tickers"] = filtered_tickers
        filtered_snapshot["count"] = filtered_count
        
        return filtered_snapshot
    
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
    
    def _update_historical_data(self, symbol: str, snapshot: MarketSnapshot):
        """
        Update historical data for a symbol
        
        Args:
            symbol: Symbol to update
            snapshot: New market snapshot
        """
        state = self._get_or_create_indicator_state(symbol)
        
        # Skip if price is zero
        if snapshot.last_price == 0:
            return
        
        # Add price to history if it's not already there
        state.close_prices.append(snapshot.last_price)
        state.timestamps.append(snapshot.timestamp)
    
    def _calculate_indicators(self, symbol: str, snapshot: MarketSnapshot):
        """
        Calculate technical indicators for a symbol
        
        Args:
            symbol: Symbol to calculate indicators for
            snapshot: Market snapshot to update with indicators
        """
        state = self._get_or_create_indicator_state(symbol)
        
        # Skip if price is zero
        if snapshot.last_price == 0:
            return
        
        # Update last update timestamp
        state.last_update = snapshot.timestamp
        
        # Add volume to history
        state.volume_data.append(snapshot.volume)
        
        # Calculate RSI
        snapshot.rsi_14 = self._calculate_rsi(state, snapshot.last_price)
        
        # Calculate MACD
        snapshot.macd = self._calculate_macd(state, snapshot.last_price)
        
        # Calculate Bollinger Bands
        bb_results = self._calculate_bollinger_bands(state, snapshot.last_price)
        snapshot.bb_upper = bb_results["upper"]
        snapshot.bb_middle = bb_results["middle"]
        snapshot.bb_lower = bb_results["lower"]
        
        # Calculate volume acceleration
        if len(state.volume_data) >= 5:
            vol_mean_prev = sum(list(state.volume_data)[-5:-1]) / 4
            if vol_mean_prev > 0:
                vol_change = (state.volume_data[-1] / vol_mean_prev) - 1.0
                snapshot.volume_acceleration = vol_change * 100.0
        
        # Calculate price change (5-minute approximation based on last 5 prices)
        if len(state.close_prices) >= 5:
            snapshot.price_change_5m = ((state.close_prices[-1] / state.close_prices[-5]) - 1.0) * 100.0
        
        # Calculate momentum (1-minute approximation based on last 10 prices)
        if len(state.close_prices) >= 10:
            snapshot.momentum_1m = state.close_prices[-1] - state.close_prices[-10]
        
        # Update state with current price
        state.last_price = snapshot.last_price
    
    def _calculate_rsi(self, state: IndicatorState, new_price: float) -> float:
        """
        Calculate RSI for a symbol
        
        Args:
            state: Indicator state
            new_price: Latest price
            
        Returns:
            RSI value
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
                    rs = float('inf')
                else:
                    rs = avg_gain / avg_loss
                
                # Calculate RSI
                state.rsi_value = 100.0 - (100.0 / (1.0 + rs))
        
        return state.rsi_value
    
    def _calculate_macd(self, state: IndicatorState, new_price: float) -> float:
        """
        Calculate MACD for a symbol
        
        Args:
            state: Indicator state
            new_price: Latest price
            
        Returns:
            MACD value
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
        state.signal_ema = macd_line * alpha_signal + state.signal_ema * (1 - alpha_signal)
        
        # Store MACD value
        state.macd_value = macd_line
        
        return macd_line
    
    def _calculate_bollinger_bands(self, state: IndicatorState, new_price: float) -> Dict[str, float]:
        """
        Calculate Bollinger Bands for a symbol
        
        Args:
            state: Indicator state
            new_price: Latest price
            
        Returns:
            Dictionary with upper, middle, and lower band values
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
            std_dev = variance ** 0.5
            
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
            "lower": state.bb_lower
        }
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'session'):
            self.session.close()


def process_market_snapshot(client, snapshot_data, batch_size=1000):
    """
    Process market snapshot to create training data with technical indicators
    
    Args:
        client: DirectPolygonClient instance
        snapshot_data: Raw market snapshot data
        batch_size: Number of symbols to process in each batch
        
    Returns:
        dict: Processed training data with features
    """
    if snapshot_data is None:
        logger.error("No snapshot data to process")
        return None
    
    logger.info("Processing market snapshot to generate training data...")
    
    # Extract tickers
    tickers = snapshot_data.get("tickers", [])
    ticker_count = len(tickers)
    logger.info(f"Processing {ticker_count} tickers in batches of {batch_size}")
    
    # Process tickers in batches to avoid memory issues
    processed_tickers = []
    batch_count = (ticker_count + batch_size - 1) // batch_size  # Ceiling division
    
    for batch_idx in range(batch_count):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, ticker_count)
        batch_tickers = tickers[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx + 1}/{batch_count} ({len(batch_tickers)} tickers)")
        
        # Create market snapshots from ticker data
        market_snapshots = {}
        for ticker in batch_tickers:
            symbol = ticker["ticker"]
            try:
                # Create a basic market snapshot with available data
                snapshot = MarketSnapshot(symbol=symbol)
                
                # Set timestamp
                if "lastTrade" in ticker and ticker["lastTrade"]:
                    snapshot.timestamp = ticker["lastTrade"].get("t", int(time.time() * 1000))
                else:
                    snapshot.timestamp = int(time.time() * 1000)
                
                # Set price data
                if "lastTrade" in ticker and ticker["lastTrade"]:
                    snapshot.last_price = ticker["lastTrade"].get("p", 0.0)
                
                if "lastQuote" in ticker and ticker["lastQuote"]:
                    snapshot.bid_price = ticker["lastQuote"].get("p", 0.0)
                    snapshot.ask_price = ticker["lastQuote"].get("P", 0.0)
                    snapshot.bid_ask_spread = snapshot.ask_price - snapshot.bid_price
                
                if "day" in ticker and ticker["day"]:
                    snapshot.high_price = ticker["day"].get("h", 0.0)
                    snapshot.low_price = ticker["day"].get("l", 0.0)
                    snapshot.volume = ticker["day"].get("v", 0)
                    snapshot.vwap = ticker["day"].get("vw", 0.0)
                
                market_snapshots[symbol] = snapshot
            except Exception as e:
                logger.warning(f"Error processing ticker {symbol}: {e}")
        
        # Use the client's methods to calculate technical indicators
        try:
            # Update the historical data with the new snapshots
            for symbol, snapshot in market_snapshots.items():
                client._update_historical_data(symbol, snapshot)
            
            # Calculate technical indicators for each snapshot
            for symbol, snapshot in market_snapshots.items():
                client._calculate_indicators(symbol, snapshot)
                
                # Add the processed snapshot to our results
                ticker_data = next((t for t in batch_tickers if t["ticker"] == symbol), None)
                if ticker_data:
                    # Add computed technical indicators to the ticker data
                    ticker_data["technical_indicators"] = {
                        "rsi_14": snapshot.rsi_14,
                        "macd": snapshot.macd,
                        "bb_upper": snapshot.bb_upper,
                        "bb_middle": snapshot.bb_middle,
                        "bb_lower": snapshot.bb_lower,
                        "volume_acceleration": snapshot.volume_acceleration,
                        "price_change_5m": snapshot.price_change_5m,
                        "momentum_1m": snapshot.momentum_1m
                    }
                    processed_tickers.append(ticker_data)
        except Exception as e:
            logger.error(f"Error calculating technical indicators for batch: {e}")
    
    logger.info(f"Successfully processed {len(processed_tickers)}/{ticker_count} tickers")
    
    # Create a new snapshot object with processed tickers
    processed_snapshot = snapshot_data.copy()
    processed_snapshot["tickers"] = processed_tickers
    processed_snapshot["count"] = len(processed_tickers)
    
    return processed_snapshot


def extract_training_features(processed_snapshot):
    """
    Extract training features in the format needed by the HFT model
    
    Args:
        processed_snapshot: Processed market snapshot with technical indicators
        
    Returns:
        tuple: (features_array, symbols_list, feature_names)
    """
    logger.info("Extracting training features from processed snapshot...")
    
    # Feature names in the order expected by the HFT model
    feature_names = [
        "last_price", 
        "bid_price", 
        "ask_price", 
        "bid_ask_spread", 
        "high_price", 
        "low_price", 
        "volume", 
        "vwap", 
        "rsi_14", 
        "macd", 
        "bb_upper", 
        "bb_middle", 
        "bb_lower", 
        "volume_acceleration", 
        "price_change_5m", 
        "momentum_1m"
    ]
    
    features_list = []
    symbols_list = []
    
    for ticker in processed_snapshot.get("tickers", []):
        symbol = ticker.get("ticker")
        
        # Skip tickers without necessary data
        if not all(k in ticker for k in ["lastTrade", "lastQuote", "day", "technical_indicators"]):
            continue
        
        try:
            # Extract features in the correct order
            features = []
            
            # Price data
            features.append(ticker.get("lastTrade", {}).get("p", 0.0))  # last_price
            features.append(ticker.get("lastQuote", {}).get("p", 0.0))  # bid_price
            features.append(ticker.get("lastQuote", {}).get("P", 0.0))  # ask_price
            bid = ticker.get("lastQuote", {}).get("p", 0.0)
            ask = ticker.get("lastQuote", {}).get("P", 0.0)
            features.append(ask - bid)  # bid_ask_spread
            features.append(ticker.get("day", {}).get("h", 0.0))  # high_price
            features.append(ticker.get("day", {}).get("l", 0.0))  # low_price
            features.append(ticker.get("day", {}).get("v", 0))    # volume
            features.append(ticker.get("day", {}).get("vw", 0.0)) # vwap
            
            # Technical indicators
            tech_indicators = ticker.get("technical_indicators", {})
            features.append(tech_indicators.get("rsi_14", 50.0))
            features.append(tech_indicators.get("macd", 0.0))
            features.append(tech_indicators.get("bb_upper", 0.0))
            features.append(tech_indicators.get("bb_middle", 0.0))
            features.append(tech_indicators.get("bb_lower", 0.0))
            features.append(tech_indicators.get("volume_acceleration", 0.0))
            features.append(tech_indicators.get("price_change_5m", 0.0))
            features.append(tech_indicators.get("momentum_1m", 0.0))
            
            # Add to our collection
            features_list.append(features)
            symbols_list.append(symbol)
        except Exception as e:
            logger.warning(f"Error extracting features for {symbol}: {e}")
    
    # Convert to numpy array
    if features_list:
        features_array = np.array(features_list, dtype=np.float32)
        logger.info(f"Extracted features for {len(symbols_list)} symbols with {len(feature_names)} features per symbol")
        return features_array, symbols_list, feature_names
    else:
        logger.warning("No valid features extracted")
        return None, [], feature_names


def save_training_data(features_array, symbols_list, feature_names, output_dir, include_raw_snapshot=False, raw_snapshot=None):
    """
    Save training data to disk in formats suitable for ML training
    
    Args:
        features_array: Numpy array of features
        symbols_list: List of symbols corresponding to features
        feature_names: Names of the features
        output_dir: Directory to save the data
        include_raw_snapshot: Whether to include the raw snapshot data
        raw_snapshot: Raw snapshot data to include
        
    Returns:
        str: Path to the saved data
    """
    if features_array is None or len(symbols_list) == 0:
        logger.error("No data to save")
        return None
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol_count = len(symbols_list)
    
    # Save features as numpy array
    np_filename = f"hft_features_{timestamp}_{symbol_count}_samples.npy"
    np_filepath = output_path / np_filename
    np.save(np_filepath, features_array)
    logger.info(f"Saved feature array to {np_filepath}")
    
    # Save symbols and feature names for reference
    meta_filename = f"hft_features_{timestamp}_{symbol_count}_metadata.json"
    meta_filepath = output_path / meta_filename
    metadata = {
        "timestamp": timestamp,
        "num_samples": symbol_count,
        "feature_names": feature_names,
        "symbols": symbols_list,
        "shape": features_array.shape,
        "stats": {
            "means": features_array.mean(axis=0).tolist(),
            "stds": features_array.std(axis=0).tolist(),
            "mins": features_array.min(axis=0).tolist(),
            "maxs": features_array.max(axis=0).tolist()
        }
    }
    with open(meta_filepath, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {meta_filepath}")
    
    # Save feature stats separately for model normalization
    stats_filename = f"hft_features_{timestamp}_{symbol_count}_stats.json"
    stats_filepath = output_path / stats_filename
    feature_stats = {
        "means": features_array.mean(axis=0).tolist(),
        "stds": features_array.std(axis=0).tolist(),
        "feature_names": feature_names
    }
    with open(stats_filepath, "w") as f:
        json.dump(feature_stats, f, indent=2)
    logger.info(f"Saved feature statistics to {stats_filepath}")
    
    # Save as CSV for easy inspection and use in other tools
    csv_filename = f"hft_features_{timestamp}_{symbol_count}_samples.csv"
    csv_filepath = output_path / csv_filename
    df = pd.DataFrame(features_array, columns=feature_names)
    df["symbol"] = symbols_list
    df.to_csv(csv_filepath, index=False)
    logger.info(f"Saved CSV data to {csv_filepath}")
    
    # Optionally save the raw snapshot data
    if include_raw_snapshot and raw_snapshot:
        raw_filename = f"raw_snapshot_{timestamp}_{symbol_count}_tickers.json"
        raw_filepath = output_path / raw_filename
        with open(raw_filepath, "w") as f:
            json.dump(raw_snapshot, f, indent=2)
        logger.info(f"Saved raw snapshot data to {raw_filepath}")
    
    return str(output_path)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate training data for HFT model from Polygon market snapshots"
    )
    parser.add_argument("--include-otc", action="store_true", help="Include OTC securities in the snapshot")
    parser.add_argument("--output-dir", default="data/hft_training", help="Directory to save the training data")
    parser.add_argument("--min-price", type=float, default=5.0, help="Filter out stocks below this price (default: $5.00)")
    parser.add_argument("--max-price", type=float, default=500.0, help="Filter out stocks above this price (default: $500.00)")
    parser.add_argument("--min-volume", type=int, default=500000, help="Filter out stocks with volume below this threshold (default: 500,000)")
    parser.add_argument("--max-volume", type=int, default=None, help="Filter out stocks with volume above this threshold")
    parser.add_argument("--min-market-cap", type=float, default=500000000, help="Filter out stocks with market cap below this threshold (default: $500M)")
    parser.add_argument("--no-filter", action="store_true", help="Disable all filtering")
    parser.add_argument("--save-raw", action="store_true", help="Save the raw snapshot data as well")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    start_time = time.time()
    
    # Create the direct polygon client
    client = DirectPolygonClient(
        min_price=args.min_price,
        max_price=args.max_price,
        min_volume=args.min_volume,
        min_market_cap=args.min_market_cap,
        max_volume=args.max_volume
    )
    
    try:
        # Fetch market snapshot
        snapshot_data = client.fetch_full_market_snapshot(
            include_otc=args.include_otc,
            apply_filters=not args.no_filter
        )
        
        if snapshot_data:
            logger.info(f"Processing snapshot with {snapshot_data.get('count', 0)} tickers")
            
            # Process market snapshot to compute technical indicators
            processed_snapshot = process_market_snapshot(client, snapshot_data)
            
            # Extract features for HFT model training
            features_array, symbols_list, feature_names = extract_training_features(processed_snapshot)
            
            if features_array is not None and len(symbols_list) > 0:
                # Save training data
                output_path = save_training_data(
                    features_array, 
                    symbols_list, 
                    feature_names, 
                    args.output_dir,
                    args.save_raw,
                    snapshot_data if args.save_raw else None
                )
                
                elapsed_time = time.time() - start_time
                logger.info(f"Training data generation completed in {elapsed_time:.2f} seconds")
                logger.info(f"Generated training data for {len(symbols_list)} symbols with {len(feature_names)} features per symbol")
                logger.info(f"Data saved to {output_path}")
                return 0
            else:
                logger.error("Failed to extract valid features from snapshot data")
                return 1
        else:
            logger.error("Failed to fetch market snapshot")
            return 1
    
    finally:
        # Clean up resources
        client.cleanup()


if __name__ == "__main__":
    exit(main())