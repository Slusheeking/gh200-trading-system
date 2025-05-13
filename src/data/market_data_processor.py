"""
Market Data Processor with GPU Acceleration

This module processes market data from various sources using GPU acceleration.
It calculates technical indicators and prepares data for ML models.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import logging
import threading
from typing import Dict, Any, List, Optional
import concurrent.futures
from collections import defaultdict

# Import Redis client
try:
    from redis_integration.redis_client import RedisClient
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logging.warning("Redis client not available. Redis integration will be disabled.")

# GPU and acceleration imports
try:
    import cupy as cp
    import tensorrt as trt
    HAS_GPU = True
except (ImportError, ValueError, AttributeError) as e:
    cp = np
    HAS_GPU = False
    logging.warning(f"GPU libraries not available or incompatible: {str(e)}. Falling back to CPU processing.")

# Import data structures from our API clients
try:
    # Try relative import first (when used as a package)
    from .polygon_rest_api import ParsedMarketData
    from .polygon_websocket import MarketData, Trade
except ImportError:
    # Fall back to absolute import (when run directly)
    from polygon_rest_api import ParsedMarketData
    from polygon_websocket import MarketData, Trade

class MarketDataProcessor:
    """
    Processes market data using GPU acceleration when available.
    Calculates technical indicators and prepares data for ML models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the market data processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Performance settings
        perf_config = config.get("performance", {})
        self.use_gpu = perf_config.get("use_gpu", True) and HAS_GPU
        self.batch_size = perf_config.get("batch_size", 1000)
        self.max_history_length = perf_config.get("max_history_length", 200)
        self.thread_count = perf_config.get("processor_threads", 4)
        
        # Thread pool for parallel processing
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.thread_count
        )
        
        # Historical data storage for technical indicators
        self.historical_data = defaultdict(lambda: {
            'close': [],
            'high': [],
            'low': [],
            'volume': [],
            'timestamp': []
        })
        
        # TensorRT engine (if available)
        self.trt_engine = None
        self.trt_context = None
        if self.use_gpu:
            self._initialize_tensorrt()
        
        # Redis client (if available)
        self.redis_client = None
        if HAS_REDIS:
            self._initialize_redis()
        
        # Synchronization
        self.mutex = threading.Lock()
        
        logging.info(f"Market Data Processor initialized with GPU acceleration: {self.use_gpu}, Redis integration: {HAS_REDIS}")
    
    def _initialize_redis(self):
        """
        Initialize Redis client for market data distribution
        """
        try:
            # Create Redis client
            self.redis_client = RedisClient(self.config)
            
            # Initialize client
            if self.redis_client.initialize():
                logging.info("Redis client initialized for market data processor")
            else:
                logging.warning("Failed to initialize Redis client for market data processor")
                self.redis_client = None
        except Exception as e:
            logging.error(f"Error initializing Redis client: {str(e)}")
            self.redis_client = None
    
    def _initialize_tensorrt(self):
        """
        Initialize TensorRT for accelerated inference of technical indicators
        
        This method loads a pre-built TensorRT engine file and creates an execution context.
        The engine should be compatible with the input shape (batch_size, sequence_length, features)
        where features are typically OHLCV data.
        
        Supports INT8 quantization for improved performance.
        """
        if not HAS_GPU:
            logging.info("GPU libraries not available. TensorRT initialization skipped.")
            return
            
        try:
            # Get TensorRT configuration from config
            trt_config = self.config.get("tensorrt", {})
            
            # Get engine file path from config or use default
            engine_file_path = trt_config.get("engine_file_path", "models/technical_indicators.engine")
            
            # Get TensorRT precision mode from config (FP32, FP16, INT8)
            precision_mode = trt_config.get("precision", "FP32")
            
            # INT8 calibration file for quantization
            int8_calibration_file = trt_config.get("int8_calibration_file", "")
            use_int8 = precision_mode == "INT8" and os.path.exists(int8_calibration_file)
            
            # Get TensorRT verbosity level
            verbosity_level = trt_config.get("verbosity", "WARNING")
            logger_level = {
                "VERBOSE": trt.Logger.VERBOSE,
                "INFO": trt.Logger.INFO,
                "WARNING": trt.Logger.WARNING,
                "ERROR": trt.Logger.ERROR,
                "INTERNAL_ERROR": trt.Logger.INTERNAL_ERROR
            }.get(verbosity_level, trt.Logger.WARNING)
            
            # Check if engine file exists
            if not os.path.exists(engine_file_path):
                # If INT8 is requested but no engine exists, we'll try to build one
                if use_int8 and self._build_int8_engine(int8_calibration_file, engine_file_path):
                    logging.info(f"Built new INT8 TensorRT engine at {engine_file_path}")
                else:
                    logging.warning(f"TensorRT engine file not found at {engine_file_path}. GPU acceleration for indicators will not be available.")
                    self.use_gpu = False
                    return
            
            # Create TensorRT logger with appropriate verbosity
            logger = trt.Logger(logger_level)
            logging.info(f"Loading TensorRT engine from {engine_file_path} with {precision_mode} precision")
            
            # Load the TensorRT engine
            with open(engine_file_path, "rb") as f, trt.Runtime(logger) as runtime:
                engine_data = f.read()
                if not engine_data:
                    logging.error("Empty engine file. Cannot deserialize TensorRT engine.")
                    self.use_gpu = False
                    return
                    
                self.trt_engine = runtime.deserialize_cuda_engine(engine_data)
                if not self.trt_engine:
                    logging.error("Failed to deserialize TensorRT engine.")
                    self.use_gpu = False
                    return
                
                # Create execution context
                self.trt_context = self.trt_engine.create_execution_context()
                if not self.trt_context:
                    logging.error("Failed to create TensorRT execution context.")
                    self.use_gpu = False
                    return
                
                # Validate engine configuration
                if self.trt_engine.num_bindings < 2:
                    logging.error("Invalid TensorRT engine: Expected at least one input and one output binding.")
                    self.use_gpu = False
                    return
                
                # Log engine details
                num_inputs = sum(1 for i in range(self.trt_engine.num_bindings) if self.trt_engine.binding_is_input(i))
                num_outputs = self.trt_engine.num_bindings - num_inputs
                
                # Check if engine is using INT8
                is_int8_engine = False
                for i in range(self.trt_engine.num_bindings):
                    if self.trt_engine.binding_is_input(i):
                        dtype = self.trt_engine.get_binding_dtype(i)
                        if dtype == trt.int8:
                            is_int8_engine = True
                            break
                
                if is_int8_engine:
                    logging.info("TensorRT engine is using INT8 quantization")
                
                logging.info(f"TensorRT engine loaded successfully with {num_inputs} inputs and {num_outputs} outputs")
                
                # Validate input shape compatibility
                for i in range(self.trt_engine.num_bindings):
                    if self.trt_engine.binding_is_input(i):
                        shape = self.trt_context.get_binding_shape(i)
                        dtype = self.trt_engine.get_binding_dtype(i)
                        dtype_str = "INT8" if dtype == trt.int8 else "FP32" if dtype == trt.float32 else "FP16" if dtype == trt.float16 else "UNKNOWN"
                        logging.info(f"TensorRT input binding {i}: shape={shape}, dtype={dtype_str}, name={self.trt_engine.get_binding_name(i)}")
                        
                        # Check if the engine has dynamic shapes (-1 in dimensions)
                        has_dynamic_shape = -1 in shape
                        if has_dynamic_shape:
                            logging.info("Engine has dynamic input shape. Will set dimensions at inference time.")
                        
                        # For static shapes, validate compatibility with our data
                        if not has_dynamic_shape and len(shape) >= 3:
                            # Assuming shape is (batch_size, sequence_length, features)
                            if shape[1] > self.max_history_length:
                                logging.warning(f"TensorRT engine expects {shape[1]} data points, but max_history_length is {self.max_history_length}. Adjusting max_history_length.")
                                self.max_history_length = shape[1]
                
                # Log output shapes
                for i in range(self.trt_engine.num_bindings):
                    if not self.trt_engine.binding_is_input(i):
                        shape = self.trt_context.get_binding_shape(i)
                        dtype = self.trt_engine.get_binding_dtype(i)
                        dtype_str = "INT8" if dtype == trt.int8 else "FP32" if dtype == trt.float32 else "FP16" if dtype == trt.float16 else "UNKNOWN"
                        logging.info(f"TensorRT output binding {i}: shape={shape}, dtype={dtype_str}, name={self.trt_engine.get_binding_name(i)}")

        except Exception as e:
            logging.error(f"Failed to initialize TensorRT: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            self.use_gpu = False
    
    def _build_int8_engine(self, calibration_file: str, output_path: str) -> bool:
        """
        Build an INT8 TensorRT engine using calibration data
        
        Args:
            calibration_file: Path to calibration data file
            output_path: Path to save the engine file
            
        Returns:
            True if engine was built successfully, False otherwise
        """
        try:
            logging.info(f"Building INT8 TensorRT engine using calibration data from {calibration_file}")
            
            # Load calibration data
            with open(calibration_file, 'rb') as f:
                calibration_data = np.load(f)
                
            # Create TensorRT builder and network
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            config = builder.create_builder_config()
            
            # Set INT8 mode
            config.set_flag(trt.BuilderFlag.INT8)
            
            # Create calibrator
            calibrator = self._create_int8_calibrator(calibration_data)
            config.int8_calibrator = calibrator
            
            # Define network layers (simplified example)
            _ = network.add_input("input", trt.float32, (-1, self.max_history_length, 4))  # OHLCV data
            
            # Add network layers here (this is a placeholder - actual implementation would depend on your model)
            # ...
            
            # Mark output
            # output_tensor = ...  # Last layer's output
            # network.mark_output(output_tensor)
            
            # Build engine
            engine = builder.build_engine(network, config)
            if engine is None:
                logging.error("Failed to build INT8 TensorRT engine")
                return False
                
            # Serialize engine to file
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
                
            logging.info(f"INT8 TensorRT engine built and saved to {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error building INT8 TensorRT engine: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def _create_int8_calibrator(self, calibration_data):
        """
        Create an INT8 calibrator for TensorRT
        
        Args:
            calibration_data: Numpy array with calibration data
            
        Returns:
            TensorRT calibrator object
        """
        # This is a placeholder - actual implementation would depend on TensorRT version
        # and specific calibration requirements
        
        class Int8Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, calibration_data, batch_size, input_name):
                super().__init__()
                self.calibration_data = calibration_data
                self.batch_size = batch_size
                self.current_index = 0
                self.input_name = input_name
                self.device_input = None
                
                # Allocate GPU memory for input data
                self.device_input = cp.cuda.alloc(calibration_data[0].nbytes)
                
            def get_batch_size(self):
                return self.batch_size
                
            def get_batch(self, names):
                if self.current_index >= len(self.calibration_data):
                    return None
                    
                # Get next batch
                batch = self.calibration_data[self.current_index:self.current_index+self.batch_size]
                self.current_index += self.batch_size
                
                # Copy data to GPU
                self.device_input.copy_from_async(batch.reshape(-1))
                
                # Return pointers to input data
                return [int(self.device_input.ptr)]
                
            def read_calibration_cache(self):
                # Return cached calibration data if available
                return None
                
            def write_calibration_cache(self, cache):
                # Cache calibration data
                pass
        
        # Create calibrator with sample data
        batch_size = min(32, len(calibration_data))
        return Int8Calibrator(calibration_data, batch_size, "input")
    
    def process_rest_data(self, market_data: ParsedMarketData) -> ParsedMarketData:
        """
        Process data from REST API
        
        Args:
            market_data: Data from REST API
            
        Returns:
            Processed market data with technical indicators
        """
        # Create a copy to avoid modifying the original
        processed_data = ParsedMarketData()
        processed_data.timestamp = market_data.timestamp
        processed_data.num_trades_processed = market_data.num_trades_processed
        processed_data.num_quotes_processed = market_data.num_quotes_processed
        
        # Process each symbol in parallel
        futures = {}
        for symbol, symbol_data in market_data.symbol_data.items():
            futures[symbol] = self.thread_pool.submit(
                self._process_symbol_data, symbol, symbol_data
            )
        
        # Collect results
        for symbol, future in futures.items():
            try:
                processed_data.symbol_data[symbol] = future.result()
            except Exception as e:
                logging.error(f"Error processing symbol {symbol}: {e}")
        
        return processed_data
    
    def process_websocket_data(self, market_data: MarketData) -> MarketData:
        """
        Process data from WebSocket with enhanced high-frequency handling
        
        Args:
            market_data: Data from WebSocket
            
        Returns:
            Processed market data with real-time features
        """
        # Use more efficient data structure for high-frequency data
        # Group trades by symbol for batch processing using a pre-allocated dictionary
        trades_by_symbol = defaultdict(lambda: [None] * len(market_data.trades))
        counts = {}
        
        # Faster grouping of trades by symbol
        for trade in market_data.trades:
            symbol = trade.symbol
            if symbol not in counts:
                counts[symbol] = 0
            
            idx = counts[symbol]
            trades_by_symbol[symbol][idx] = trade
            counts[symbol] += 1
        
        # Trim arrays to actual size
        for symbol in trades_by_symbol:
            trades_by_symbol[symbol] = trades_by_symbol[symbol][:counts[symbol]]
        
        # Process each symbol in parallel using thread pool with batch size control
        futures = {}
        batch_size = self.config.get("websocket", {}).get("batch_size", 1000)
        
        for symbol, trades in trades_by_symbol.items():
            if len(trades) > 0:
                # Process in batches if there are many trades
                if len(trades) > batch_size:
                    for i in range(0, len(trades), batch_size):
                        batch = trades[i:i+batch_size]
                        futures[f"{symbol}_{i}"] = self.thread_pool.submit(
                            self._process_symbol_websocket_data, symbol, batch
                        )
                else:
                    futures[symbol] = self.thread_pool.submit(
                        self._process_symbol_websocket_data, symbol, trades
                    )
        
        # Wait for all processing to complete with improved timeout handling
        completed_futures = []
        timeout_per_future = 0.05  # Shorter timeout per future for more responsive handling
        
        for symbol, future in futures.items():
            try:
                completed_futures.append(future.result(timeout=timeout_per_future))
            except concurrent.futures.TimeoutError:
                logging.warning(f"Timeout waiting for {symbol} websocket processing - will continue in background")
            except Exception as e:
                logging.error(f"Error processing websocket data for {symbol}: {e}")
        
        # Return the processed data
        return market_data
    
    def _process_symbol_websocket_data(self, symbol: str, trades: List[Trade]):
        """
        Process websocket trades for a single symbol with optimized batch updates
        
        Args:
            symbol: Symbol to process
            trades: List of trades for the symbol
        """
        if not trades:
            return
            
        # Sort trades by timestamp to ensure correct order - use faster sorting
        trades.sort(key=lambda t: t.timestamp)
        
        # Pre-allocate arrays for batch update with exact size
        n_trades = len(trades)
        close_prices = np.empty(n_trades, dtype=float)
        high_prices = np.empty(n_trades, dtype=float)
        low_prices = np.empty(n_trades, dtype=float)
        volumes = np.empty(n_trades, dtype=float)
        timestamps = np.empty(n_trades, dtype=int)
        
        # Extract data from trades using vectorized operations
        for i, trade in enumerate(trades):
            close_prices[i] = trade.price
            high_prices[i] = trade.price
            low_prices[i] = trade.price
            volumes[i] = trade.size
            timestamps[i] = trade.timestamp
        
        # Update historical data in batch with optimized locking
        with self.mutex:
            # Convert to list for extending
            close_list = close_prices.tolist()
            high_list = high_prices.tolist()
            low_list = low_prices.tolist()
            volume_list = volumes.tolist()
            timestamp_list = timestamps.tolist()
            
            # Batch update historical data
            self.historical_data[symbol]['close'].extend(close_list)
            self.historical_data[symbol]['high'].extend(high_list)
            self.historical_data[symbol]['low'].extend(low_list)
            self.historical_data[symbol]['volume'].extend(volume_list)
            self.historical_data[symbol]['timestamp'].extend(timestamp_list)
            
            # Trim historical data if it exceeds max length - use more efficient slicing
            current_length = len(self.historical_data[symbol]['close'])
            if current_length > self.max_history_length:
                start_idx = current_length - self.max_history_length
                self.historical_data[symbol]['close'] = self.historical_data[symbol]['close'][start_idx:]
                self.historical_data[symbol]['high'] = self.historical_data[symbol]['high'][start_idx:]
                self.historical_data[symbol]['low'] = self.historical_data[symbol]['low'][start_idx:]
                self.historical_data[symbol]['volume'] = self.historical_data[symbol]['volume'][start_idx:]
                self.historical_data[symbol]['timestamp'] = self.historical_data[symbol]['timestamp'][start_idx:]
        
        # Calculate real-time features if we have enough data
        # Use a separate thread for feature calculation to avoid blocking
        if len(self.historical_data[symbol]['close']) >= 20:
            self.thread_pool.submit(self._calculate_realtime_features, symbol)
    
    def _calculate_realtime_features(self, symbol: str):
        """
        Calculate real-time features for a symbol based on websocket data
        with enhanced feature set and optimized calculations
        
        Args:
            symbol: Symbol to calculate features for
        """
        try:
            # Skip if not enough data
            if len(self.historical_data[symbol]['close']) < 20:
                return
                
            # Create DataFrames for feature calculation - use copy to avoid race conditions
            with self.mutex:
                market_data = pd.DataFrame({
                    'close': self.historical_data[symbol]['close'][-200:],  # Limit to recent data for performance
                    'high': self.historical_data[symbol]['high'][-200:],
                    'low': self.historical_data[symbol]['low'][-200:],
                    'volume': self.historical_data[symbol]['volume'][-200:],
                    'timestamp': self.historical_data[symbol]['timestamp'][-200:]
                })
            
            # Calculate basic features for real-time signals
            rsi = self._calculate_rsi(market_data['close']).iloc[-1]
            
            # Calculate MACD
            macd_data = self._calculate_macd(market_data['close'])
            macd = macd_data['macd'].iloc[-1]
            macd_signal = macd_data['signal'].iloc[-1]
            macd_hist = macd_data['histogram'].iloc[-1]
            
            # Calculate Bollinger Bands
            bb_data = self._calculate_bollinger_bands(market_data['close'])
            bb_upper = bb_data['upper'].iloc[-1]
            bb_middle = bb_data['middle'].iloc[-1]
            bb_lower = bb_data['lower'].iloc[-1]
            
            # Calculate EMAs
            ema_12 = self._calculate_ema(market_data['close'], span=12).iloc[-1]
            ema_26 = self._calculate_ema(market_data['close'], span=26).iloc[-1]
            
            # Calculate Stochastic Oscillator
            stoch_data = self._calculate_stochastic_oscillator(
                market_data['high'], market_data['low'], market_data['close'])
            stoch_k = stoch_data['k'].iloc[-1]
            stoch_d = stoch_data['d'].iloc[-1]
            
            # Calculate OBV
            obv = self._calculate_obv(market_data['close'], market_data['volume']).iloc[-1]
            
            # Detect market regime
            market_regime = self.detect_market_regime(market_data)
            
            # Create enhanced feature set
            features = {
                'timestamp': self.historical_data[symbol]['timestamp'][-1],
                'last_price': self.historical_data[symbol]['close'][-1],
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'ema_12': ema_12,
                'ema_26': ema_26,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d,
                'obv': obv,
                'market_regime': market_regime
            }
            
            # Publish to Redis if available
            self._publish_realtime_features_to_redis(symbol, features)
            
            # Feed data to ML models if configured
            self._feed_data_to_ml_models(symbol, features, market_data)
            
        except Exception as e:
            logging.error(f"Error calculating real-time features for {symbol}: {e}")
    
    def _update_historical_data(self, symbol: str, trade: Trade):
        """
        Update historical data for a symbol based on a trade
        
        Args:
            symbol: Symbol to update
            trade: Trade data
        """
        # Add trade data to historical data
        self.historical_data[symbol]['close'].append(trade.price)
        self.historical_data[symbol]['timestamp'].append(trade.timestamp)
        
        # We don't have high/low/volume in a single trade, so use price for high/low
        # and size for volume as approximations
        self.historical_data[symbol]['high'].append(trade.price)
        self.historical_data[symbol]['low'].append(trade.price)
        self.historical_data[symbol]['volume'].append(trade.size)
        
        # Trim historical data if it exceeds max length
        if len(self.historical_data[symbol]['close']) > self.max_history_length:
            self.historical_data[symbol]['close'] = self.historical_data[symbol]['close'][-self.max_history_length:]
            self.historical_data[symbol]['high'] = self.historical_data[symbol]['high'][-self.max_history_length:]
            self.historical_data[symbol]['low'] = self.historical_data[symbol]['low'][-self.max_history_length:]
            self.historical_data[symbol]['volume'] = self.historical_data[symbol]['volume'][-self.max_history_length:]
            self.historical_data[symbol]['timestamp'] = self.historical_data[symbol]['timestamp'][-self.max_history_length:]
    
    def _process_symbol_data(self, symbol: str, symbol_data: ParsedMarketData.SymbolData) -> ParsedMarketData.SymbolData:
        """
        Process data for a single symbol
        
        Args:
            symbol: Symbol to process
            symbol_data: Data for the symbol
            
        Returns:
            Processed symbol data with technical indicators
        """
        # Create a copy to avoid modifying the original
        processed_data = ParsedMarketData.SymbolData()
        processed_data.__dict__.update(symbol_data.__dict__)
        
        # Update historical data
        with self.mutex:
            if symbol_data.last_price > 0:
                self.historical_data[symbol]['close'].append(symbol_data.last_price)
                self.historical_data[symbol]['high'].append(symbol_data.high_price if symbol_data.high_price > 0 else symbol_data.last_price)
                self.historical_data[symbol]['low'].append(symbol_data.low_price if symbol_data.low_price > 0 else symbol_data.last_price)
                self.historical_data[symbol]['volume'].append(symbol_data.volume if symbol_data.volume > 0 else 0)
                self.historical_data[symbol]['timestamp'].append(symbol_data.timestamp)
                
                # Trim historical data if it exceeds max length
                if len(self.historical_data[symbol]['close']) > self.max_history_length:
                    self.historical_data[symbol]['close'] = self.historical_data[symbol]['close'][-self.max_history_length:]
                    self.historical_data[symbol]['high'] = self.historical_data[symbol]['high'][-self.max_history_length:]
                    self.historical_data[symbol]['low'] = self.historical_data[symbol]['low'][-self.max_history_length:]
                    self.historical_data[symbol]['volume'] = self.historical_data[symbol]['volume'][-self.max_history_length:]
                    self.historical_data[symbol]['timestamp'] = self.historical_data[symbol]['timestamp'][-self.max_history_length:]
        
        # Calculate technical indicators
        if self.use_gpu:
            self._calculate_indicators_gpu(symbol, processed_data)
        else:
            self._calculate_indicators_cpu(symbol, processed_data)
        
        return processed_data
    
    def _calculate_indicators_cpu(self, symbol: str, data: ParsedMarketData.SymbolData):
        """
        Calculate technical indicators using CPU
        
        Args:
            symbol: Symbol to calculate indicators for
            data: Symbol data to update with indicators
        """
        with self.mutex:
            # Get historical data arrays
            close_prices = pd.Series(self.historical_data[symbol]['close'], dtype=float)
            high_prices = pd.Series(self.historical_data[symbol]['high'], dtype=float)
            low_prices = pd.Series(self.historical_data[symbol]['low'], dtype=float)
            volume_data = pd.Series(self.historical_data[symbol]['volume'], dtype=float)
        
        # Only calculate indicators if we have enough data
        min_data_points = 52 # Increased for Ichimoku and longer SMAs
        if len(close_prices) >= min_data_points:
            try:
                # Convert numpy arrays used later to pandas Series if needed, or use pandas directly
                close_array = close_prices.values # Keep numpy array for existing logic if preferred
                high_array = high_prices.values
                low_array = low_prices.values
                volume_array = volume_data.values

                # RSI (14-period) - Using helper function
                data.rsi_14 = self._calculate_rsi(close_prices).iloc[-1]
                
                # MACD - Using helper function
                macd_results = self._calculate_macd(close_prices)
                data.macd = macd_results['macd'].iloc[-1]
                # Optionally store signal and histogram if needed in `data`
                # data.macd_signal = macd_results['signal'].iloc[-1]
                # data.macd_histogram = macd_results['histogram'].iloc[-1]

                # Bollinger Bands (20-period, 2 standard deviations) - Using helper function
                bb_results = self._calculate_bollinger_bands(close_prices)
                data.bb_upper = bb_results['upper'].iloc[-1]
                data.bb_middle = bb_results['middle'].iloc[-1]
                data.bb_lower = bb_results['lower'].iloc[-1]

                # ATR (14-period) - Simplified (using high/low arrays)
                if len(high_array) >= 14 and len(low_array) >= 14:
                    tr1 = high_prices - low_prices
                    tr2 = abs(high_prices - close_prices.shift(1))
                    tr3 = abs(low_prices - close_prices.shift(1))
                    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    data.atr = true_range.rolling(window=14).mean().iloc[-1]
                else:
                    data.atr = (data.high_price - data.low_price) * 0.1 # Fallback

                # --- Add New Indicators ---

                # Exponential Moving Averages (EMAs)
                data.ema_12 = self._calculate_ema(close_prices, span=12).iloc[-1]
                data.ema_26 = self._calculate_ema(close_prices, span=26).iloc[-1]
                data.ema_50 = self._calculate_ema(close_prices, span=50).iloc[-1]
                data.ema_200 = self._calculate_ema(close_prices, span=200).iloc[-1]

                # Stochastic Oscillator
                stoch_results = self._calculate_stochastic_oscillator(high_prices, low_prices, close_prices)
                data.stoch_k = stoch_results['k'].iloc[-1]
                data.stoch_d = stoch_results['d'].iloc[-1]

                # On-Balance Volume (OBV)
                data.obv = self._calculate_obv(close_prices, volume_data).iloc[-1]

                # Ichimoku Cloud
                ichimoku_results = self._calculate_ichimoku_cloud(high_prices, low_prices, close_prices)
                data.ichimoku_tenkan = ichimoku_results['tenkan_sen'].iloc[-1]
                data.ichimoku_kijun = ichimoku_results['kijun_sen'].iloc[-1]
                # Note: Spans A/B/Chikou are shifted, so the latest value might be NaN or represent a future/past point.
                # Decide how to handle these (e.g., store the latest non-NaN value or handle shifted nature).
                # For simplicity, storing the latest calculated value (which might be NaN if shifted)
                data.ichimoku_span_a = ichimoku_results['senkou_span_a'].iloc[-1]
                data.ichimoku_span_b = ichimoku_results['senkou_span_b'].iloc[-1]
                data.ichimoku_chikou = ichimoku_results['chikou_span'].iloc[-1]

                # Fibonacci Retracement (Based on last N periods high/low)
                # Determine the lookback period for Fib calculation (e.g., last 50 periods)
                lookback_period = 50
                recent_high = high_prices.iloc[-lookback_period:].max()
                recent_low = low_prices.iloc[-lookback_period:].min()
                fib_levels = self._calculate_fibonacci_retracement(recent_high, recent_low)
                data.fib_0_0 = fib_levels['0.0']
                data.fib_23_6 = fib_levels['0.236']
                data.fib_38_2 = fib_levels['0.382']
                data.fib_50_0 = fib_levels['0.5']
                data.fib_61_8 = fib_levels['0.618']
                data.fib_78_6 = fib_levels['0.786']
                data.fib_100_0 = fib_levels['1.0']

                # --- End New Indicators ---

                # Volume indicators (using volume_array for consistency with previous logic)
                data.avg_volume = np.mean(volume_array[-20:]) if len(volume_array) >= 20 else data.volume
                
                # Volume acceleration (rate of change)
                if len(volume_array) >= 5:
                    vol_mean_prev = np.mean(volume_array[-5:-1])
                    if vol_mean_prev > 0:
                        vol_change = (volume_array[-1] / vol_mean_prev) - 1.0
                        data.volume_acceleration = vol_change * 100.0
                    else:
                        data.volume_acceleration = 0.0 # Avoid division by zero
                
                # Volume spike detection
                if len(volume_array) >= 20:
                    vol_avg = np.mean(volume_array[-20:-1])
                    vol_std = np.std(volume_array[-20:-1])
                    if vol_std > 0:
                        data.volume_spike = (volume_array[-1] - vol_avg) / vol_std
                    else:
                         data.volume_spike = 0.0 # Avoid division by zero
                
                # Price dynamics (using close_array)
                if len(close_array) >= 5:
                    if close_array[-5] != 0: # Avoid division by zero
                        data.price_change_5m = ((close_array[-1] / close_array[-5]) - 1.0) * 100.0
                    else:
                        data.price_change_5m = 0.0
                
                # Momentum (using close_array)
                if len(close_array) >= 10:
                    data.momentum_1m = close_array[-1] - close_array[-10]
                
                # SMA cross signal (using close_array) - Already calculated EMAs, use those? Or keep SMA? Keeping SMA for now.
                if len(close_array) >= 200:
                    sma50 = np.mean(close_array[-50:])
                    sma200 = np.mean(close_array[-200:])
                    data.sma_cross_signal = sma50 - sma200
                
            except Exception as e:
                logging.warning(f"Error calculating technical indicators for {symbol}: {e}")
                # Optionally clear indicator fields in data object on error
                indicator_fields = ['rsi_14', 'macd', 'bb_upper', 'bb_middle', 'bb_lower', 'atr',
                                    'ema_12', 'ema_26', 'ema_50', 'ema_200', 'stoch_k', 'stoch_d', 'obv',
                                    'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_span_a', 'ichimoku_span_b', 'ichimoku_chikou',
                                    'fib_0_0', 'fib_23_6', 'fib_38_2', 'fib_50_0', 'fib_61_8', 'fib_78_6', 'fib_100_0',
                                    'avg_volume', 'volume_acceleration', 'volume_spike', 'price_change_5m',
                                    'momentum_1m', 'sma_cross_signal']
                for field in indicator_fields:
                    if hasattr(data, field):
                        setattr(data, field, None) # Set to None or NaN

        # --- Remove old RSI calculation ---
        # delta = np.diff(close_array)
        # --- End of indicator calculations ---
    
    def _calculate_indicators_gpu(self, symbol: str, data: ParsedMarketData.SymbolData):
        """
        Calculate technical indicators using GPU acceleration
        
        Args:
            symbol: Symbol to calculate indicators for
            data: Symbol data to update with indicators
        """
        if not HAS_GPU:
            self._calculate_indicators_cpu(symbol, data)
            return
            
        try:
            with self.mutex:
                # Get historical data arrays and transfer to GPU
                close_array = cp.array(self.historical_data[symbol]['close'], dtype=float)
                high_array = cp.array(self.historical_data[symbol]['high'], dtype=float)
                low_array = cp.array(self.historical_data[symbol]['low'], dtype=float)
                volume_array = cp.array(self.historical_data[symbol]['volume'], dtype=float)
            
            # Only calculate indicators if we have enough data and TensorRT is initialized
            if len(close_array) >= self.max_history_length and self.trt_engine and self.trt_context: # Use max_history_length as minimum data points
                try:
                    
                    input_data = cp.stack([
                        close_array[-self.max_history_length:],
                        high_array[-self.max_history_length:],
                        low_array[-self.max_history_length:],
                        volume_array[-self.max_history_length:]
                    ], axis=1)
                    
                    # Add batch dimension (batch_size=1)
                    _input_tensor = cp.expand_dims(input_data, axis=0)
                    
                    # Ensure input data is in float32 format (TensorRT typically expects float32)
                    _input_tensor = _input_tensor.astype(cp.float32)

                    # Allocate buffers for input and output
                    inputs = []
                    outputs = []
                    bindings = []
                    stream = cp.cuda.Stream()

                    # Allocate memory for each binding (input and output)
                    for binding_idx in range(self.trt_engine.num_bindings):
                        _ = self.trt_engine[binding_idx]
                        shape = self.trt_context.get_binding_shape(binding_idx)
                        dtype = trt.nptype(self.trt_engine.get_binding_dtype(binding_idx))
                        
                        # If shape contains -1 (dynamic dimension), replace with actual values
                        if -1 in shape:
                            # For input tensor, we know the actual shape
                            if self.trt_engine.binding_is_input(binding_idx):
                                shape = _input_tensor.shape
                            else:
                                shape = self.trt_context.get_binding_shape(binding_idx)
                        
                        # Calculate buffer size
                        size = trt.volume(shape) * np.dtype(dtype).itemsize
                        
                        # Allocate device memory
                        device_buffer = cp.cuda.alloc(size)
                        bindings.append(int(device_buffer.ptr))
                        
                        # Keep track of input and output buffers
                        if self.trt_engine.binding_is_input(binding_idx):
                            inputs.append(device_buffer)
                        else:
                            # Create empty array for output with the right shape and dtype
                            host_output = cp.empty(shape, dtype)
                            outputs.append((device_buffer, host_output))

                    # Transfer input data to GPU
                    inputs[0].copy_from_async(_input_tensor.reshape(-1), stream)
                    
                    # Execute inference
                    self.trt_context.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)
                    
                    # Transfer output data back to host
                    results = []
                    for device_output, host_output in outputs:
                        device_output.copy_to_async(host_output.reshape(-1), stream)
                        results.append(host_output)
                    
                    # Synchronize to ensure all operations are complete
                    stream.synchronize()
                    
                    
                    if len(results) >= 6:  # Ensure we have at least the basic indicators
                        data.rsi_14 = float(results[0][0, -1])
                        data.macd = float(results[1][0, -1])
                        data.bb_upper = float(results[2][0, -1])
                        data.bb_middle = float(results[3][0, -1])
                        data.bb_lower = float(results[4][0, -1])
                        data.atr = float(results[5][0, -1])
                        
                        # If more indicators are available in the results, process them
                        if len(results) > 6:
                            # Example: Additional indicators like volume metrics
                            data.avg_volume = float(results[6][0, -1])
                            
                            if len(results) > 7:
                                data.volume_acceleration = float(results[7][0, -1])
                            
                            if len(results) > 8:
                                data.volume_spike = float(results[8][0, -1])
                            
                            if len(results) > 9:
                                data.price_change_5m = float(results[9][0, -1])
                            
                            if len(results) > 10:
                                data.momentum_1m = float(results[10][0, -1])
                            
                            if len(results) > 11:
                                data.sma_cross_signal = float(results[11][0, -1])
                    
                    logging.debug(f"TensorRT inference completed for {symbol}")

                except Exception as e:
                    logging.error(f"Error during TensorRT inference for {symbol}: {e}")
                    # Fall back to CuPy calculation if TensorRT inference fails
                    self._calculate_indicators_cupy(symbol, data)

            else:
                # If not enough data or TensorRT not initialized, use CuPy
                self._calculate_indicators_cupy(symbol, data)

        except Exception as e:
            logging.error(f"Error calculating GPU indicators for {symbol}: {e}")
            # Fall back to CPU calculation if any other error occurs
            self._calculate_indicators_cpu(symbol, data)

    def _calculate_indicators_cupy(self, symbol: str, data: ParsedMarketData.SymbolData):
        """
        Calculate technical indicators using CuPy (GPU)
        
        Args:
            symbol: Symbol to calculate indicators for
            data: Symbol data to update with indicators
        """
        if not HAS_GPU:
            self._calculate_indicators_cpu(symbol, data)
            return
            
        try:
            with self.mutex:
                # Get historical data arrays and transfer to GPU
                close_array = cp.array(self.historical_data[symbol]['close'], dtype=float)
                high_array = cp.array(self.historical_data[symbol]['high'], dtype=float)
                low_array = cp.array(self.historical_data[symbol]['low'], dtype=float)
                volume_array = cp.array(self.historical_data[symbol]['volume'], dtype=float)

            # Only calculate indicators if we have enough data
            if len(close_array) >= 14:
                # RSI (14-period)
                delta = cp.diff(close_array)
                gain = cp.where(delta > 0, delta, 0)
                loss = cp.where(delta < 0, -delta, 0)

                avg_gain = cp.mean(gain[-14:])
                avg_loss = cp.mean(loss[-14:])

                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    data.rsi_14 = float(100 - (100 / (1 + rs)))
                else:
                    data.rsi_14 = 100.0

                # Simple Moving Averages
                if len(close_array) >= 12:
                    sma12 = float(cp.mean(close_array[-12:]))
                else:
                    sma12 = float(close_array[-1])

                if len(close_array) >= 26:
                    sma26 = float(cp.mean(close_array[-26:]))
                else:
                    sma26 = float(close_array[-1])

                # MACD
                data.macd = sma12 - sma26

                # Bollinger Bands (20-period, 2 standard deviations)
                if len(close_array) >= 20:
                    sma20 = float(cp.mean(close_array[-20:]))
                    std20 = float(cp.std(close_array[-20:]))
                    data.bb_middle = sma20
                    data.bb_upper = sma20 + 2 * std20
                    data.bb_lower = sma20 - 2 * std20

                # ATR (14-period) - Simplified
                if len(high_array) >= 14 and len(low_array) >= 14:
                    ranges = high_array[-14:] - low_array[-14:]
                    data.atr = float(cp.mean(ranges))

                # Volume indicators
                data.avg_volume = float(cp.mean(volume_array[-20:])) if len(volume_array) >= 20 else data.volume

                # Volume acceleration (rate of change)
                if len(volume_array) >= 5:
                    vol_change = (float(volume_array[-1]) / float(cp.mean(volume_array[-5:-1]))) - 1.0
                    data.volume_acceleration = vol_change * 100.0

                # Volume spike detection
                if len(volume_array) >= 20:
                    vol_avg = float(cp.mean(volume_array[-20:-1]))
                    vol_std = float(cp.std(volume_array[-20:-1]))
                    if vol_std > 0 and vol_avg > 0:
                        data.volume_spike = (float(volume_array[-1]) - vol_avg) / vol_std

                # Price dynamics
                if len(close_array) >= 5:
                    data.price_change_5m = ((float(close_array[-1]) / float(close_array[-5])) - 1.0) * 100.0

                # Momentum
                if len(close_array) >= 10:
                    data.momentum_1m = float(close_array[-1] - close_array[-10])

                # SMA cross signal (SMA 50 vs SMA 200)
                if len(close_array) >= 200:
                    sma50 = float(cp.mean(close_array[-50:]))
                    sma200 = float(cp.mean(close_array[-200:]))
                    data.sma_cross_signal = sma50 - sma200

        except Exception as e:
            logging.error(f"Error calculating CuPy indicators for {symbol}: {e}")
            # Fall back to CPU calculation
            self._calculate_indicators_cpu(symbol, data)
    def _publish_realtime_features_to_redis(self, symbol: str, features: Dict[str, Any]):
        """
        Publish real-time features to Redis
        
        Args:
            symbol: Symbol for the features
            features: Dictionary of feature values
        """
        # Skip if Redis integration is not available
        if not self.redis_client:
            return
            
        try:
            # Create channel name
            channel = f"market_data:{symbol}:features"
            
            # Convert features to JSON
            features_json = json.dumps(features)
            
            # Publish to Redis
            self.redis_client.publish(channel, features_json)
            
            # Also store the latest features in a Redis hash
            hash_key = f"market_data:{symbol}:latest"
            
            # Convert all values to strings for Redis hash
            string_features = {k: str(v) for k, v in features.items()}
            
            # Use Redis pipeline for better performance
            pipeline = self.redis_client.redis.pipeline()
            
            # Set hash values
            pipeline.hmset(hash_key, string_features)
            
            # Set expiration for the hash (1 hour)
            pipeline.expire(hash_key, 3600)
            
            # Execute pipeline
            pipeline.execute()
            
            # Cache technical indicators separately for faster access
            self._cache_technical_indicators(symbol, features)
            
        except Exception as e:
            logging.error(f"Error publishing to Redis: {e}")
    
    def _cache_technical_indicators(self, symbol: str, features: Dict[str, Any]):
        """
        Cache technical indicators in Redis for faster access
        
        Args:
            symbol: Symbol for the indicators
            features: Dictionary of feature values
        """
        if not self.redis_client:
            return
            
        try:
            # Extract technical indicators
            indicators = {
                'rsi': features.get('rsi', 0),
                'macd': features.get('macd', 0),
                'macd_signal': features.get('macd_signal', 0),
                'macd_hist': features.get('macd_hist', 0),
                'bb_upper': features.get('bb_upper', 0),
                'bb_middle': features.get('bb_middle', 0),
                'bb_lower': features.get('bb_lower', 0)
            }
            
            # Create sorted set key
            key = f"indicators:{symbol}:{int(time.time())}"
            
            # Store indicators as JSON
            self.redis_client.set(key, json.dumps(indicators), expiry=86400)  # 24 hour expiry
            
            # Add to indicator list
            list_key = f"indicator_list:{symbol}"
            self.redis_client.redis.lpush(list_key, key)
            
            # Trim list to last 1000 entries
            self.redis_client.redis.ltrim(list_key, 0, 999)
            
            # Set expiry on list
            self.redis_client.redis.expire(list_key, 86400)  # 24 hour expiry
            
        except Exception as e:
            logging.error(f"Error caching technical indicators: {e}")
    
    def initialize_redis(self, redis_client=None):
        """
        Initialize Redis integration
        
        Args:
            redis_client: Redis client instance (optional)
        
        Returns:
            True if initialization was successful, False otherwise
        """
        # If a client is provided, use it
        if redis_client:
            self.redis_client = redis_client
            logging.info("Redis integration initialized with provided client")
            return True
            
        # Otherwise, try to create a new client
        if not HAS_REDIS:
            logging.warning("Redis client not available. Redis integration will be disabled.")
            return False
            
        try:
            # Create Redis client
            self.redis_client = RedisClient(self.config)
            
            # Initialize client
            if self.redis_client.initialize():
                logging.info("Redis integration initialized for MarketDataProcessor")
                return True
            else:
                logging.warning("Failed to initialize Redis client")
                self.redis_client = None
                return False
        except Exception as e:
            logging.error(f"Error initializing Redis client: {str(e)}")
            self.redis_client = None
            return False
    
    def resample_market_data(self, market_data: pd.DataFrame, timeframe: str = '1m') -> pd.DataFrame:
        """
        Resample market data to a different timeframe
        
        Args:
            market_data: DataFrame with market data (must have DatetimeIndex)
            timeframe: Target timeframe ('1m', '5m', '15m', '1h', '1d')
            
        Returns:
            Resampled DataFrame
        """
        if not isinstance(market_data.index, pd.DatetimeIndex):
            raise ValueError("Market data must have DatetimeIndex for resampling")
            
        # Define resampling rules
        resample_rules = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '1h': '1H',
            '1d': 'D'
        }
        
        if timeframe not in resample_rules:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
            
        # Resample OHLCV data
        resampled = market_data.resample(resample_rules[timeframe]).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        return resampled
    
    def detect_market_regime(self, market_data: pd.DataFrame, window: int = 20) -> str:
        """
        Detect market regime based on volatility and trend
        
        Args:
            market_data: DataFrame with market data
            window: Window size for calculations
            
        Returns:
            Market regime ('trending_up', 'trending_down', 'ranging', 'volatile')
        """
        # Calculate volatility
        returns = market_data['close'].pct_change()
        volatility = returns.rolling(window=window).std().iloc[-1]
        
        # Calculate trend
        sma_short = market_data['close'].rolling(window=window//2).mean().iloc[-1]
        sma_long = market_data['close'].rolling(window=window).mean().iloc[-1]
        
        # Calculate True Range for volatility assessment
        # Determine regime
        high_volatility_threshold = 0.015  # 1.5% daily volatility
        
        if volatility > high_volatility_threshold:
            regime = 'volatile'
        elif sma_short > sma_long * 1.01:  # 1% above
            regime = 'trending_up'
        elif sma_short < sma_long * 0.99:  # 1% below
            regime = 'trending_down'
        else:
            regime = 'ranging'
            
        # Publish regime to Redis
        if hasattr(self, 'redis_client') and self.redis_client is not None:
            try:
                symbol = market_data.get('symbol', 'unknown')
                self.redis_client.set(f"market_regime:{symbol}", regime)
                self.redis_client.expire(f"market_regime:{symbol}", 3600)  # 1 hour expiration
            except Exception as e:
                logging.error(f"Error publishing market regime to Redis: {e}")
                
        return regime
    
    def cleanup(self):
        """
        Clean up resources used by the market data processor
        
        This method should be called when the processor is no longer needed
        to ensure proper release of GPU resources.
        """
        logging.info("Cleaning up MarketDataProcessor resources")
        
        # Clean up TensorRT resources
        if self.trt_context:
            # In newer TensorRT versions, explicitly delete the context
            del self.trt_context
            self.trt_context = None
            logging.debug("TensorRT execution context released")
            
        if self.trt_engine:
            # In newer TensorRT versions, explicitly delete the engine
            del self.trt_engine
            self.trt_engine = None
            logging.debug("TensorRT engine released")
            
        # Clean up thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            logging.debug("Thread pool shut down")
            
        # Clean up Redis client
        if hasattr(self, 'redis_client') and self.redis_client is not None:
            try:
                self.redis_client.close()
                logging.debug("Redis client closed")
            except Exception as e:
                logging.warning(f"Error closing Redis client: {e}")
            
        # Force garbage collection to ensure GPU memory is released
        if HAS_GPU:
            try:
                # Clear any cached memory in CuPy
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                logging.debug("CuPy memory pools cleared")
            except Exception as e:
                logging.warning(f"Error clearing CuPy memory pools: {e}")
                
    def _calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Series of price values
            period: RSI period (default: 14)
            
        Returns:
            Series containing RSI values
        """
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = -losses  # Make losses positive
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate Moving Average Convergence Divergence (MACD)
        
        Args:
            prices: Series of price values
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal EMA period (default: 9)
            
        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' values
        """
        # Calculate fast and slow EMAs
        fast_ema = self._calculate_ema(prices, span=fast_period)
        slow_ema = self._calculate_ema(prices, span=slow_period)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = self._calculate_ema(macd_line, span=signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Series of price values
            window: Window size for moving average (default: 20)
            num_std: Number of standard deviations (default: 2)
            
        Returns:
            Dictionary with 'upper', 'middle', and 'lower' band values
        """
        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=window).mean()
        
        # Calculate standard deviation
        std = prices.rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }
    
    def _min_max_scale(self, series):
        """Min-max scale a series to 0-1 range"""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)
        
    def _calculate_ema(self, series, span=20):
        """Calculate Exponential Moving Average"""
        return series.ewm(span=span, adjust=False).mean()

    def _calculate_stochastic_oscillator(self, high, low, close, k_period=14, d_period=3):
        """
        Calculate Stochastic Oscillator
        
        Returns:
            Dictionary with 'k' and 'd' values
        """
        # Calculate %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (3-period moving average of %K)
        d = k.rolling(window=d_period).mean()
        
        return {'k': k, 'd': d}

    def _calculate_obv(self, close, volume):
        """
        Calculate On-Balance Volume (OBV)
        
        Args:
            close: Series of closing prices
            volume: Series of volume values
            
        Returns:
            Series containing OBV values
        """
        price_change = close.diff()
        obv = pd.Series(0, index=close.index)
        
        for i in range(1, len(close)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        return obv

    def _calculate_ichimoku_cloud(self, high, low, close):
        """
        Calculate Ichimoku Cloud components
        
        Returns:
            Dictionary with Ichimoku components
        """
        # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for the past 9 periods
        nine_period_high = high.rolling(window=9).max()
        nine_period_low = low.rolling(window=9).min()
        tenkan_sen = (nine_period_high + nine_period_low) / 2
        
        # Kijun-sen (Base Line): (highest high + lowest low)/2 for the past 26 periods
        twenty_six_period_high = high.rolling(window=26).max()
        twenty_six_period_low = low.rolling(window=26).min()
        kijun_sen = (twenty_six_period_high + twenty_six_period_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 (projected 26 periods in the future)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for the past 52 periods (projected 26 periods in the future)
        fifty_two_period_high = high.rolling(window=52).max()
        fifty_two_period_low = low.rolling(window=52).min()
        senkou_span_b = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close price projected 26 periods in the past
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }

    def _calculate_fibonacci_retracement(self, high, low):
        """
        Calculate Fibonacci retracement levels
        
        Returns:
            Dictionary with retracement levels
        """
        diff = high - low
        levels = {
            '0.0': low,
            '0.236': low + 0.236 * diff,
            '0.382': low + 0.382 * diff,
            '0.5': low + 0.5 * diff,
            '0.618': low + 0.618 * diff,
            '0.786': low + 0.786 * diff,
            '1.0': high
        }
        return levels
    
    def extract_gbdt_features(self, market_data, timeframe='1m'):
        """
        Extract features specifically optimized for Gradient Boosted Decision Trees.
        These features focus on fast and efficient screening.
        
        Args:
            market_data: DataFrame containing market data
            timeframe: Time resolution of the data
            
        Returns:
            DataFrame of extracted features
        """
        # Resample data if needed
        if timeframe != '1m' and isinstance(market_data.index, pd.DatetimeIndex):
            market_data = self.resample_market_data(market_data, timeframe)
        
        features = pd.DataFrame()
        
        # Price-based features
        features['price_pct_change'] = market_data['close'].pct_change()
        features['price_std_5'] = market_data['close'].rolling(5).std()
        features['price_std_20'] = market_data['close'].rolling(20).std()
        features['price_acceleration'] = features['price_pct_change'].diff()
        features['log_return'] = np.log(market_data['close'] / market_data['close'].shift(1))
        
        # Range-based features
        features['high_low_ratio'] = market_data['high'] / market_data['low']
        features['close_to_high'] = (market_data['close'] - market_data['low']) / (market_data['high'] - market_data['low'])
        
        # Volume-based features
        features['volume_pct_change'] = market_data['volume'].pct_change()
        features['volume_ma_ratio'] = market_data['volume'] / market_data['volume'].rolling(10).mean()
        features['volume_price_trend'] = features['volume_pct_change'] * features['price_pct_change']
        features['volume_std_5'] = market_data['volume'].rolling(5).std() / market_data['volume'].rolling(5).mean()
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(market_data['close'])
        
        macd_data = self._calculate_macd(market_data['close'])
        features['macd'] = macd_data['macd']
        features['macd_signal'] = macd_data['signal']
        features['macd_hist'] = macd_data['histogram']
        
        bb_data = self._calculate_bollinger_bands(market_data['close'])
        features['bollinger_pct'] = (market_data['close'] - bb_data['middle']) / (bb_data['upper'] - bb_data['lower'])
        features['bollinger_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
        
        # Stochastic oscillator
        stoch_data = self._calculate_stochastic_oscillator(market_data['high'], market_data['low'], market_data['close'])
        features['stoch_k'] = stoch_data['k']
        features['stoch_d'] = stoch_data['d']
        
        # On-Balance Volume
        features['obv_normalized'] = self._calculate_obv(market_data['close'], market_data['volume']) / market_data['volume'].rolling(20).sum()
        
        # EMAs
        features['ema_12'] = self._calculate_ema(market_data['close'], span=12)
        features['ema_26'] = self._calculate_ema(market_data['close'], span=26)
        features['ema_cross'] = features['ema_12'] - features['ema_26']
        
        # Inter-market features if available
        if 'vix' in market_data.columns:
            features['vix_ratio'] = market_data['vix'] / market_data['vix'].rolling(10).mean()
        
        # Time-based features
        if isinstance(market_data.index, pd.DatetimeIndex):
            features['hour'] = market_data.index.hour
            features['minute'] = market_data.index.minute
            features['day_of_week'] = market_data.index.dayofweek
        
        # Market regime features
        try:
            # Detect market regime for segments of data
            window_size = min(100, len(market_data))
            regimes = []
            
            for i in range(len(market_data) - window_size + 1):
                window_data = market_data.iloc[i:i+window_size]
                regime = self.detect_market_regime(window_data)
                regimes.append(regime)
            
            # Pad with the first regime for earlier data points
            regimes = [regimes[0]] * (len(market_data) - len(regimes)) + regimes
            
            # Convert regimes to numeric features
            regime_map = {'trending_up': 1, 'trending_down': -1, 'ranging': 0, 'volatile': 2}
            features['market_regime'] = [regime_map.get(r, 0) for r in regimes]
        except Exception as e:
            logging.warning(f"Error calculating market regime features: {e}")
            features['market_regime'] = 0
        
        # Feature normalization
        numeric_features = features.select_dtypes(include=[np.number])
        for col in numeric_features.columns:
            if features[col].std() > 0:
                features[col] = (features[col] - features[col].mean()) / features[col].std()
            else:
                features[col] = 0  # Handle constant features
        
        # Drop NaN values resulting from indicators that need historical data
        features = features.dropna()
        
        # Add feature timestamp
        if isinstance(market_data.index, pd.DatetimeIndex):
            features.index = market_data.index[len(market_data) - len(features):]
        
        return features

    def extract_axial_attention_features(self, market_data, seq_length=60, timeframe='1m'):
        """
        Extract features optimized for Axial Attention models.
        These features focus on spatial relationships in the data.
        
        Args:
            market_data: DataFrame containing market data
            seq_length: Length of sequence for attention mechanism
            timeframe: Time resolution of the data
            
        Returns:
            Numpy array of extracted features with shape [samples, seq_length, features]
        """
        # Resample data if needed
        if timeframe != '1m' and isinstance(market_data.index, pd.DatetimeIndex):
            market_data = self.resample_market_data(market_data, timeframe)
            
        # Calculate basic features
        basic_features = pd.DataFrame()
        
        # Price features with normalization
        basic_features['close_normalized'] = self._min_max_scale(market_data['close'])
        basic_features['high_normalized'] = self._min_max_scale(market_data['high'])
        basic_features['low_normalized'] = self._min_max_scale(market_data['low'])
        basic_features['open_normalized'] = self._min_max_scale(market_data['open']) if 'open' in market_data else basic_features['close_normalized']
        
        # Price differences and ratios
        basic_features['high_low_diff'] = self._min_max_scale(market_data['high'] - market_data['low'])
        basic_features['close_open_diff'] = self._min_max_scale(
            market_data['close'] - market_data['open'] if 'open' in market_data else pd.Series(0, index=market_data.index)
        )
        
        # Returns and momentum
        basic_features['returns'] = market_data['close'].pct_change()
        basic_features['returns_normalized'] = self._min_max_scale(basic_features['returns'])
        
        for lag in [1, 3, 5, 10]:
            basic_features[f'momentum_{lag}'] = self._min_max_scale(market_data['close'].pct_change(lag))
        
        # Volatility features
        for window in [10, 20, 30]:
            vol = market_data['close'].rolling(window).std()
            basic_features[f'volatility_{window}'] = self._min_max_scale(vol)
        
        # Technical indicators
        basic_features['rsi'] = self._calculate_rsi(market_data['close']) / 100.0  # Normalize to 0-1
        
        macd_data = self._calculate_macd(market_data['close'])
        basic_features['macd'] = self._min_max_scale(macd_data['macd'])
        basic_features['macd_signal'] = self._min_max_scale(macd_data['signal'])
        basic_features['macd_hist'] = self._min_max_scale(macd_data['histogram'])
        
        bb_data = self._calculate_bollinger_bands(market_data['close'])
        basic_features['bb_position'] = (market_data['close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
        basic_features['bb_width'] = self._min_max_scale((bb_data['upper'] - bb_data['lower']) / bb_data['middle'])
        
        # Stochastic oscillator
        stoch_data = self._calculate_stochastic_oscillator(market_data['high'], market_data['low'], market_data['close'])
        basic_features['stoch_k'] = stoch_data['k'] / 100.0  # Normalize to 0-1
        basic_features['stoch_d'] = stoch_data['d'] / 100.0  # Normalize to 0-1
        
        # Volume features
        if 'volume' in market_data.columns:
            basic_features['volume_normalized'] = self._min_max_scale(market_data['volume'])
            basic_features['volume_change'] = self._min_max_scale(market_data['volume'].pct_change())
            basic_features['volume_ma_ratio'] = self._min_max_scale(market_data['volume'] / market_data['volume'].rolling(20).mean())
            
            # OBV
            obv = self._calculate_obv(market_data['close'], market_data['volume'])
            basic_features['obv_normalized'] = self._min_max_scale(obv)
        
        # Ichimoku Cloud components (selected)
        try:
            ichimoku = self._calculate_ichimoku_cloud(market_data['high'], market_data['low'], market_data['close'])
            basic_features['tenkan_sen'] = self._min_max_scale(ichimoku['tenkan_sen'])
            basic_features['kijun_sen'] = self._min_max_scale(ichimoku['kijun_sen'])
        except Exception as e:
            logging.warning(f"Error calculating Ichimoku features: {e}")
        
        # Time-based features if datetime index is available
        if isinstance(market_data.index, pd.DatetimeIndex):
            # Hour of day normalized to 0-1
            basic_features['hour_sin'] = np.sin(2 * np.pi * market_data.index.hour / 24)
            basic_features['hour_cos'] = np.cos(2 * np.pi * market_data.index.hour / 24)
            
            # Day of week normalized to 0-1
            basic_features['day_sin'] = np.sin(2 * np.pi * market_data.index.dayofweek / 7)
            basic_features['day_cos'] = np.cos(2 * np.pi * market_data.index.dayofweek / 7)
        
        # Fill NaN values with 0
        basic_features = basic_features.fillna(0)
        
        # Create sequence data with proper padding if needed
        if len(basic_features) < seq_length:
            # Pad with zeros if we don't have enough data
            pad_length = seq_length - len(basic_features)
            pad_df = pd.DataFrame(0, index=range(pad_length), columns=basic_features.columns)
            padded_features = pd.concat([pad_df, basic_features])
            sequence_data = [padded_features.values]
        else:
            # Create overlapping sequences
            sequence_data = []
            for i in range(len(basic_features) - seq_length + 1):
                sequence_data.append(basic_features.iloc[i:i+seq_length].values)
        
        if not sequence_data:
            return np.array([])
        
        return np.array(sequence_data)
        
    def _feed_data_to_ml_models(self, symbol: str, features: Dict[str, Any], market_data: pd.DataFrame):
        """
        Feed processed market data to ML models for inference
        
        Args:
            symbol: Symbol being processed
            features: Dictionary of calculated features
            market_data: DataFrame with raw market data
        """
        # Skip if no ML model configuration is available
        ml_config = self.config.get("ml_models", {})
        if not ml_config:
            return
            
        try:
            # Check which models are enabled
            gbdt_enabled = ml_config.get("gbdt", {}).get("enabled", False)
            axial_enabled = ml_config.get("axial_attention", {}).get("enabled", False)
            lstm_gru_enabled = ml_config.get("lstm_gru", {}).get("enabled", False)
            
            # Prepare model inputs based on enabled models
            model_inputs = {}
            
            # Extract features for GBDT model (fast path screening)
            if gbdt_enabled:
                gbdt_features = self.extract_gbdt_features(market_data)
                if not gbdt_features.empty:
                    model_inputs["gbdt"] = gbdt_features
            
            # Extract features for Axial Attention model (accurate path signals)
            if axial_enabled:
                seq_length = ml_config.get("axial_attention", {}).get("seq_length", 60)
                axial_features = self.extract_axial_attention_features(market_data, seq_length=seq_length)
                if len(axial_features) > 0:
                    model_inputs["axial_attention"] = axial_features
            
            # Extract features for LSTM/GRU model (exit optimization)
            if lstm_gru_enabled:
                seq_length = ml_config.get("lstm_gru", {}).get("seq_length", 100)
                lstm_gru_features = self.extract_lstm_gru_features(market_data, seq_length=seq_length)
                if len(lstm_gru_features) > 0:
                    model_inputs["lstm_gru"] = lstm_gru_features
            
            # If no model inputs were prepared, exit early
            if not model_inputs:
                return
                
            # Publish model inputs to Redis for external model inference
            if self.redis_client:
                # Create a channel for model inputs
                channel = f"ml_model_inputs:{symbol}"
                
                # Serialize model inputs (convert numpy arrays to lists)
                serializable_inputs = {}
                for model_type, inputs in model_inputs.items():
                    if isinstance(inputs, pd.DataFrame):
                        serializable_inputs[model_type] = inputs.to_dict(orient="list")
                    elif isinstance(inputs, np.ndarray):
                        serializable_inputs[model_type] = inputs.tolist()
                
                # Publish to Redis
                self.redis_client.publish(channel, json.dumps({
                    "timestamp": features["timestamp"],
                    "symbol": symbol,
                    "model_inputs": serializable_inputs
                }))
                
            # Execute local model inference if callbacks are registered
            for model_type, inputs in model_inputs.items():
                callback_name = f"{model_type}_inference_callback"
                if hasattr(self, callback_name) and callable(getattr(self, callback_name)):
                    # Call the model-specific inference callback
                    callback = getattr(self, callback_name)
                    try:
                        inference_result = callback(symbol, inputs)
                        
                        # Publish inference results to Redis
                        if self.redis_client and inference_result is not None:
                            result_channel = f"ml_inference_results:{symbol}:{model_type}"
                            self.redis_client.publish(result_channel, json.dumps({
                                "timestamp": features["timestamp"],
                                "symbol": symbol,
                                "model_type": model_type,
                                "result": inference_result
                            }))
                    except Exception as e:
                        logging.error(f"Error in {model_type} inference callback for {symbol}: {e}")
                        
        except Exception as e:
            logging.error(f"Error feeding data to ML models for {symbol}: {e}")
    
    def detect_time_series_patterns(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Detect common time series patterns in market data
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of detected patterns and their confidence scores
        """
        patterns = {}
        
        try:
            # Ensure we have enough data
            if len(market_data) < 10:
                return patterns
                
            close = market_data['close']
            open_prices = market_data['open'] if 'open' in market_data else close.shift(1)
            high = market_data['high']
            low = market_data['low']
            
            # Detect trend reversal patterns
            
            # Head and Shoulders pattern
            if len(close) >= 30:
                # Simplified detection - look for 3 peaks with middle peak higher
                window = min(30, len(close))
                segment = close[-window:]
                
                # Find local maxima
                peaks = []
                for i in range(1, len(segment)-1):
                    if segment.iloc[i] > segment.iloc[i-1] and segment.iloc[i] > segment.iloc[i+1]:
                        peaks.append((i, segment.iloc[i]))
                
                # Check if we have at least 3 peaks
                if len(peaks) >= 3:
                    # Sort peaks by height
                    sorted_peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
                    
                    # Check if the highest peak is in the middle position
                    peak_indices = [p[0] for p in sorted_peaks[:3]]
                    peak_indices.sort()
                    
                    if sorted_peaks[0][0] == peak_indices[1]:
                        # Middle peak is highest - potential head and shoulders
                        patterns['head_and_shoulders'] = 0.7
            
            # Double Bottom pattern
            if len(close) >= 20:
                window = min(20, len(close))
                segment = close[-window:]
                
                # Find local minima
                troughs = []
                for i in range(1, len(segment)-1):
                    if segment.iloc[i] < segment.iloc[i-1] and segment.iloc[i] < segment.iloc[i+1]:
                        troughs.append((i, segment.iloc[i]))
                
                # Check if we have at least 2 troughs
                if len(troughs) >= 2:
                    # Sort troughs by value
                    sorted_troughs = sorted(troughs, key=lambda x: x[1])
                    
                    # Check if the two lowest troughs are similar in value
                    if abs(sorted_troughs[0][1] - sorted_troughs[1][1]) / sorted_troughs[0][1] < 0.03:
                        # Two similar bottoms - potential double bottom
                        patterns['double_bottom'] = 0.8
            
            # Candlestick patterns
            
            # Doji
            body_sizes = abs(close - open_prices)
            shadow_sizes = high - low
            doji_score = np.mean(body_sizes[-3:] / shadow_sizes[-3:] < 0.1) if len(close) >= 3 else 0
            if doji_score > 0:
                patterns['doji'] = doji_score
            
            # Hammer
            if len(close) >= 1:
                i = -1  # Last candle
                body_size = abs(close.iloc[i] - open_prices.iloc[i])
                lower_shadow = min(close.iloc[i], open_prices.iloc[i]) - low.iloc[i]
                upper_shadow = high.iloc[i] - max(close.iloc[i], open_prices.iloc[i])
                
                if body_size > 0 and lower_shadow > 2 * body_size and upper_shadow < 0.2 * body_size:
                    patterns['hammer'] = 0.9
            
            # Shooting Star
            if len(close) >= 1:
                i = -1  # Last candle
                body_size = abs(close.iloc[i] - open_prices.iloc[i])
                lower_shadow = min(close.iloc[i], open_prices.iloc[i]) - low.iloc[i]
                upper_shadow = high.iloc[i] - max(close.iloc[i], open_prices.iloc[i])
                
                if body_size > 0 and upper_shadow > 2 * body_size and lower_shadow < 0.2 * body_size:
                    patterns['shooting_star'] = 0.9
            
            # Engulfing pattern
            if len(close) >= 2:
                # Bullish engulfing
                if (open_prices.iloc[-2] > close.iloc[-2] and  # Previous candle is bearish
                    close.iloc[-1] > open_prices.iloc[-1] and  # Current candle is bullish
                    open_prices.iloc[-1] < close.iloc[-2] and  # Current open below previous close
                    close.iloc[-1] > open_prices.iloc[-2]):    # Current close above previous open
                    patterns['bullish_engulfing'] = 0.85
                
                # Bearish engulfing
                if (open_prices.iloc[-2] < close.iloc[-2] and  # Previous candle is bullish
                    close.iloc[-1] < open_prices.iloc[-1] and  # Current candle is bearish
                    open_prices.iloc[-1] > close.iloc[-2] and  # Current open above previous close
                    close.iloc[-1] < open_prices.iloc[-2]):    # Current close below previous open
                    patterns['bearish_engulfing'] = 0.85
            
            # Trend patterns
            
            # Detect uptrend
            if len(close) >= 10:
                sma5 = close.rolling(5).mean()
                sma10 = close.rolling(10).mean()
                
                if sma5.iloc[-1] > sma10.iloc[-1] and close.iloc[-1] > sma5.iloc[-1]:
                    patterns['uptrend'] = 0.7
            
            # Detect downtrend
            if len(close) >= 10:
                sma5 = close.rolling(5).mean()
                sma10 = close.rolling(10).mean()
                
                if sma5.iloc[-1] < sma10.iloc[-1] and close.iloc[-1] < sma5.iloc[-1]:
                    patterns['downtrend'] = 0.7
            
            # Detect consolidation/ranging
            if len(close) >= 20:
                std20 = close.rolling(20).std()
                mean20 = close.rolling(20).mean()
                
                if std20.iloc[-1] / mean20.iloc[-1] < 0.01:  # Very low volatility
                    patterns['consolidation'] = 0.8
            
        except Exception as e:
            logging.error(f"Error detecting time series patterns: {e}")
            
        return patterns
    
    def get_features_for_model(self, symbol: str, model_type: str, timeframe: str = '1m', seq_length: Optional[int] = None) -> Any:
        """
        Get the latest calculated features suitable for a specific model type.
        
        Args:
            symbol: The stock symbol
            model_type: Type of model ('gbdt', 'axial_attention', 'lstm_gru')
            timeframe: Timeframe for features
            seq_length: Sequence length required by the model (for sequence models)
            
        Returns:
            Features in the format required by the model (DataFrame or numpy array)
        """
        with self.mutex:
            if symbol not in self.historical_data or len(self.historical_data[symbol]['close']) < 50:
                logging.warning(f"Not enough historical data for {symbol} to extract features")
                return None
                
            # Create DataFrame from historical data
            market_data = pd.DataFrame({
                'close': self.historical_data[symbol]['close'],
                'high': self.historical_data[symbol]['high'],
                'low': self.historical_data[symbol]['low'],
                'volume': self.historical_data[symbol]['volume'],
                'timestamp': self.historical_data[symbol]['timestamp']
            })
            
            # Convert timestamp to datetime if needed
            if isinstance(market_data['timestamp'].iloc[0], (int, float)):
                market_data.index = pd.to_datetime(market_data['timestamp'], unit='ms')
            
        # Extract features based on model type
        if model_type.lower() == 'gbdt':
            return self.extract_gbdt_features(market_data, timeframe=timeframe)
            
        elif model_type.lower() == 'axial_attention':
            seq_length = seq_length or 60  # Default sequence length
            return self.extract_axial_attention_features(market_data, seq_length=seq_length, timeframe=timeframe)
            
        elif model_type.lower() == 'lstm_gru':
            seq_length = seq_length or 100  # Default sequence length
            return self.extract_lstm_gru_features(market_data, seq_length=seq_length, timeframe=timeframe)
            
        else:
            logging.warning(f"Unknown model type: {model_type}")
            return None

    def extract_lstm_gru_features(self, market_data, seq_length=100, timeframe='1m'):
        """
        Extract features optimized for LSTM/GRU models.
        These features focus on temporal patterns for exit optimization.
        
        Args:
            market_data: DataFrame containing market data
            seq_length: Length of sequence for LSTM/GRU
            timeframe: Time resolution of the data
            
        Returns:
            Numpy array of extracted features with shape [samples, seq_length, features]
        """
        # Resample data if needed
        if timeframe != '1m' and isinstance(market_data.index, pd.DatetimeIndex):
            market_data = self.resample_market_data(market_data, timeframe)
            
        # Calculate basic features
        basic_features = pd.DataFrame()
        
        # Price and returns
        basic_features['returns'] = market_data['close'].pct_change()
        basic_features['log_returns'] = np.log(market_data['close'] / market_data['close'].shift(1))
        
        # Moving averages and volatility of returns
        for window in [5, 10, 20, 50]:
            basic_features[f'returns_ma_{window}'] = basic_features['returns'].rolling(window=window).mean()
            basic_features[f'returns_std_{window}'] = basic_features['returns'].rolling(window=window).std()
            basic_features[f'close_ma_{window}'] = market_data['close'].rolling(window=window).mean() / market_data['close'] - 1
        
        # Log price differences at different lags
        for lag in [1, 3, 5, 10, 20]:
            basic_features[f'log_return_{lag}'] = np.log(market_data['close'] / market_data['close'].shift(lag))
        
        # Momentum indicators
        for lag in [1, 5, 10, 20]:
            basic_features[f'momentum_{lag}'] = market_data['close'] - market_data['close'].shift(lag)
            basic_features[f'momentum_norm_{lag}'] = basic_features[f'momentum_{lag}'] / market_data['close']
        
        # Technical indicators
        basic_features['rsi'] = self._calculate_rsi(market_data['close']) / 100.0
        
        # MACD
        macd_data = self._calculate_macd(market_data['close'])
        basic_features['macd'] = macd_data['macd']
        basic_features['macd_signal'] = macd_data['signal']
        basic_features['macd_hist'] = macd_data['histogram']
        
        # Bollinger bands features
        bb_data = self._calculate_bollinger_bands(market_data['close'])
        basic_features['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
        basic_features['bb_position'] = (market_data['close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
        
        # Stochastic oscillator
        stoch_data = self._calculate_stochastic_oscillator(market_data['high'], market_data['low'], market_data['close'])
        basic_features['stoch_k'] = stoch_data['k'] / 100.0
        basic_features['stoch_d'] = stoch_data['d'] / 100.0
        
        # Volume features if available
        if 'volume' in market_data.columns:
            basic_features['volume_change'] = market_data['volume'].pct_change()
            basic_features['volume_ma_ratio'] = market_data['volume'] / market_data['volume'].rolling(20).mean()
            
            # Volume momentum
            for lag in [1, 5, 10]:
                basic_features[f'volume_momentum_{lag}'] = market_data['volume'] / market_data['volume'].shift(lag) - 1
            
            # OBV
            obv = self._calculate_obv(market_data['close'], market_data['volume'])
            basic_features['obv_change'] = obv.diff() / obv.shift(1)
            
            # Volume and price relationship
            basic_features['volume_price_corr'] = basic_features['returns'].rolling(10).corr(basic_features['volume_change'])
        
        # Add high-low range and volatility
        basic_features['high_low_ratio'] = (market_data['high'] - market_data['low']) / market_data['close']
        basic_features['high_low_std'] = basic_features['high_low_ratio'].rolling(10).std()
        
        # Price patterns
        for i in range(1, 6):
            basic_features[f'close_diff_{i}'] = market_data['close'].diff(i) / market_data['close']
        
        # Candlestick pattern features
        if 'open' in market_data.columns:
            # Doji
            basic_features['doji'] = abs(market_data['close'] - market_data['open']) / (market_data['high'] - market_data['low']) < 0.1
            
            # Hammer
            body_size = abs(market_data['close'] - market_data['open'])
            lower_shadow = np.minimum(market_data['open'], market_data['close']) - market_data['low']
            upper_shadow = market_data['high'] - np.maximum(market_data['open'], market_data['close'])
            basic_features['hammer'] = (lower_shadow > 2 * body_size) & (upper_shadow < 0.2 * body_size)
            
            # Shooting star
            basic_features['shooting_star'] = (upper_shadow > 2 * body_size) & (lower_shadow < 0.2 * body_size)
        
        # Market regime features
        try:
            window_size = min(50, len(market_data))
            regimes = []
            
            for i in range(len(market_data) - window_size + 1):
                window_data = market_data.iloc[i:i+window_size]
                regime = self.detect_market_regime(window_data)
                regimes.append(regime)
            
            # Pad with the first regime for earlier data points
            regimes = [regimes[0] if regimes else 'unknown'] * (len(market_data) - len(regimes)) + regimes
            
            # One-hot encode regimes
            regime_map = {
                'trending_up': [1, 0, 0, 0],
                'trending_down': [0, 1, 0, 0],
                'ranging': [0, 0, 1, 0],
                'volatile': [0, 0, 0, 1],
                'unknown': [0, 0, 0, 0]
            }
            
            regime_features = [regime_map.get(r, regime_map['unknown']) for r in regimes]
            basic_features['regime_trend_up'] = [r[0] for r in regime_features]
            basic_features['regime_trend_down'] = [r[1] for r in regime_features]
            basic_features['regime_ranging'] = [r[2] for r in regime_features]
            basic_features['regime_volatile'] = [r[3] for r in regime_features]
        except Exception as e:
            logging.warning(f"Error calculating market regime features: {e}")
        
        # Time-based features if datetime index is available
        if isinstance(market_data.index, pd.DatetimeIndex):
            # Hour of day - sine and cosine encoding for cyclical features
            basic_features['hour_sin'] = np.sin(2 * np.pi * market_data.index.hour / 24)
            basic_features['hour_cos'] = np.cos(2 * np.pi * market_data.index.hour / 24)
            
            # Day of week - sine and cosine encoding
            basic_features['day_sin'] = np.sin(2 * np.pi * market_data.index.dayofweek / 7)
            basic_features['day_cos'] = np.cos(2 * np.pi * market_data.index.dayofweek / 7)
        
        # Fill NaN values with 0
        basic_features = basic_features.fillna(0)
        
        # Create sequence data with proper padding if needed
        if len(basic_features) < seq_length:
            # Pad with zeros if we don't have enough data
            pad_length = seq_length - len(basic_features)
            pad_df = pd.DataFrame(0, index=range(pad_length), columns=basic_features.columns)
            padded_features = pd.concat([pad_df, basic_features])
            sequence_data = [padded_features.values]
        else:
            # Create overlapping sequences
            sequence_data = []
            for i in range(len(basic_features) - seq_length + 1):
                sequence_data.append(basic_features.iloc[i:i+seq_length].values)
        
        if not sequence_data:
            return np.array([])
        
        return np.array(sequence_data)
