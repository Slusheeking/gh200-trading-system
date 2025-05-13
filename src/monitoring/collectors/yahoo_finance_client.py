"""
Yahoo Finance client for the system exporter

This module provides a client for fetching historical price data from Yahoo Finance.
"""

import time
import threading
import logging
import os
from typing import Dict, Any
import pandas as pd
import numpy as np

class YahooFinanceClient:
    """Client for Yahoo Finance data"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Yahoo Finance client
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Cache settings
        self.cache_dir = config.get("yahoo_finance", {}).get("cache_dir", "cache/yahoo_finance")
        self.cache_expiry = config.get("yahoo_finance", {}).get("cache_expiry", 86400)  # 24 hours
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize cache
        self.data_cache = {}
        self.cache_lock = threading.Lock()
    
    def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Get historical price data for a symbol
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with historical price data
        """
        # Check cache
        cache_key = f"{symbol}_{period}_{interval}"
        
        with self.cache_lock:
            if cache_key in self.data_cache:
                cache_entry = self.data_cache[cache_key]
                
                # Check if cache is still valid
                if time.time() - cache_entry["timestamp"] < self.cache_expiry:
                    return cache_entry["data"]
        
        # Check if data is cached on disk
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.csv")
        
        if os.path.exists(cache_file):
            # Check if cache is still valid
            if time.time() - os.path.getmtime(cache_file) < self.cache_expiry:
                try:
                    # Load data from cache
                    data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    
                    # Update memory cache
                    with self.cache_lock:
                        self.data_cache[cache_key] = {
                            "data": data,
                            "timestamp": os.path.getmtime(cache_file)
                        }
                    
                    return data
                except Exception as e:
                    logging.error(f"Error loading cached data: {str(e)}")
        
        # Fetch data from Yahoo Finance
        try:
            import yfinance as yf
            
            # Fetch data
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            
            # Check if data is empty
            if data.empty:
                logging.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Save to cache
            try:
                data.to_csv(cache_file)
                
                # Update memory cache
                with self.cache_lock:
                    self.data_cache[cache_key] = {
                        "data": data,
                        "timestamp": time.time()
                    }
            except Exception as e:
                logging.error(f"Error saving data to cache: {str(e)}")
            
            return data
        
        except Exception as e:
            logging.error(f"Error fetching data from Yahoo Finance: {str(e)}")
            return pd.DataFrame()
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with quote data
        """
        try:
            import yfinance as yf
            
            # Fetch ticker
            ticker = yf.Ticker(symbol)
            
            # Get quote
            info = ticker.info
            
            # Extract relevant fields
            quote = {
                "symbol": symbol,
                "price": info.get("regularMarketPrice", None),
                "previous_close": info.get("previousClose", None),
                "open": info.get("open", None),
                "day_high": info.get("dayHigh", None),
                "day_low": info.get("dayLow", None),
                "volume": info.get("volume", None),
                "avg_volume": info.get("averageVolume", None),
                "market_cap": info.get("marketCap", None),
                "pe_ratio": info.get("trailingPE", None),
                "eps": info.get("trailingEps", None),
                "dividend_yield": info.get("dividendYield", None),
                "52_week_high": info.get("fiftyTwoWeekHigh", None),
                "52_week_low": info.get("fiftyTwoWeekLow", None),
                "timestamp": int(time.time())
            }
            
            return quote
        
        except Exception as e:
            logging.error(f"Error fetching quote from Yahoo Finance: {str(e)}")
            return {}
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for price data
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with indicators
        """
        if data.empty:
            return data
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Simple Moving Averages
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
        
        # MACD
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        
        # Bollinger Bands
        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        df["BB_Std"] = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Middle"] + (df["BB_Std"] * 2)
        df["BB_Lower"] = df["BB_Middle"] - (df["BB_Std"] * 2)
        
        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        low_14 = df["Low"].rolling(window=14).min()
        high_14 = df["High"].rolling(window=14).max()
        
        df["Stoch_K"] = 100 * ((df["Close"] - low_14) / (high_14 - low_14))
        df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()
        
        # Average True Range (ATR)
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - df["Close"].shift()).abs()
        tr3 = (df["Low"] - df["Close"].shift()).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(window=14).mean()
        
        # On-Balance Volume (OBV)
        df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
        
        return df
    
    def get_chart_data(self, symbol: str, period: str = "1y", interval: str = "1d", 
                      chart_type: str = "candlestick", include_indicators: bool = True) -> Dict[str, Any]:
        """
        Get chart data for a symbol
        
        Args:
            symbol: Stock symbol
            period: Time period
            interval: Data interval
            chart_type: Chart type (candlestick, line, ohlc)
            include_indicators: Whether to include technical indicators
            
        Returns:
            Dictionary with chart data
        """
        # Get historical data
        data = self.get_historical_data(symbol, period, interval)
        
        if data.empty:
            return {"error": "No data available"}
        
        # Calculate indicators if requested
        if include_indicators:
            data = self.calculate_indicators(data)
        
        # Format data for chart
        if chart_type == "candlestick":
            chart_data = {
                "type": "candlestick",
                "data": [
                    {
                        "time": index.timestamp(),
                        "open": row["Open"],
                        "high": row["High"],
                        "low": row["Low"],
                        "close": row["Close"],
                        "volume": row["Volume"]
                    }
                    for index, row in data.iterrows()
                ]
            }
        elif chart_type == "ohlc":
            chart_data = {
                "type": "ohlc",
                "data": [
                    {
                        "time": index.timestamp(),
                        "open": row["Open"],
                        "high": row["High"],
                        "low": row["Low"],
                        "close": row["Close"]
                    }
                    for index, row in data.iterrows()
                ]
            }
        else:  # line chart
            chart_data = {
                "type": "line",
                "data": [
                    {
                        "time": index.timestamp(),
                        "value": row["Close"]
                    }
                    for index, row in data.iterrows()
                ]
            }
        
        # Add indicators if requested
        if include_indicators:
            indicators = {}
            
            # Moving Averages
            if "SMA_20" in data.columns:
                indicators["sma20"] = [
                    {
                        "time": index.timestamp(),
                        "value": row["SMA_20"] if not pd.isna(row["SMA_20"]) else None
                    }
                    for index, row in data.iterrows()
                ]
            
            if "SMA_50" in data.columns:
                indicators["sma50"] = [
                    {
                        "time": index.timestamp(),
                        "value": row["SMA_50"] if not pd.isna(row["SMA_50"]) else None
                    }
                    for index, row in data.iterrows()
                ]
            
            if "SMA_200" in data.columns:
                indicators["sma200"] = [
                    {
                        "time": index.timestamp(),
                        "value": row["SMA_200"] if not pd.isna(row["SMA_200"]) else None
                    }
                    for index, row in data.iterrows()
                ]
            
            # Bollinger Bands
            if "BB_Upper" in data.columns:
                indicators["bollinger"] = {
                    "upper": [
                        {
                            "time": index.timestamp(),
                            "value": row["BB_Upper"] if not pd.isna(row["BB_Upper"]) else None
                        }
                        for index, row in data.iterrows()
                    ],
                    "middle": [
                        {
                            "time": index.timestamp(),
                            "value": row["BB_Middle"] if not pd.isna(row["BB_Middle"]) else None
                        }
                        for index, row in data.iterrows()
                    ],
                    "lower": [
                        {
                            "time": index.timestamp(),
                            "value": row["BB_Lower"] if not pd.isna(row["BB_Lower"]) else None
                        }
                        for index, row in data.iterrows()
                    ]
                }
            
            # RSI
            if "RSI" in data.columns:
                indicators["rsi"] = [
                    {
                        "time": index.timestamp(),
                        "value": row["RSI"] if not pd.isna(row["RSI"]) else None
                    }
                    for index, row in data.iterrows()
                ]
            
            # MACD
            if "MACD" in data.columns:
                indicators["macd"] = {
                    "macd": [
                        {
                            "time": index.timestamp(),
                            "value": row["MACD"] if not pd.isna(row["MACD"]) else None
                        }
                        for index, row in data.iterrows()
                    ],
                    "signal": [
                        {
                            "time": index.timestamp(),
                            "value": row["MACD_Signal"] if not pd.isna(row["MACD_Signal"]) else None
                        }
                        for index, row in data.iterrows()
                    ],
                    "histogram": [
                        {
                            "time": index.timestamp(),
                            "value": row["MACD_Hist"] if not pd.isna(row["MACD_Hist"]) else None
                        }
                        for index, row in data.iterrows()
                    ]
                }
            
            chart_data["indicators"] = indicators
        
        # Add metadata
        chart_data["metadata"] = {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "last_updated": int(time.time())
        }
        
        return chart_data
