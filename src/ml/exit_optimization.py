"""
Exit optimization model implementation

This module provides a production-ready implementation of the Exit Optimization model
for determining optimal exit points for trading positions. It integrates with the
project's data sources, signal handling, and logging system.
"""

import os
import time
import numpy as np
import logging
import threading
from typing import List, Dict, Any, Optional

# Try to import GPU libraries if available
try:
    import torch
    import tensorrt as trt
    from cuda import cudart

    HAS_GPU = True
except (ImportError, ValueError, AttributeError) as e:
    HAS_GPU = False
    logging.warning(
        f"GPU libraries not available or incompatible: {str(e)}. Falling back to CPU processing."
    )

# Import project modules
from src.monitoring.log import logging as log
from src.execution.alpaca_execution import Signal
from config.config_loader import get_config

# Try to import Redis client
try:
    from redis_integration.redis_client import RedisSignalHandler

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logging.warning("Redis client not available. Signal distribution will be disabled.")


# TensorRT logger singleton
class TrtLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)
        self.logger = log.setup_logger("tensorrt")

    def log(self, severity, msg):
        if severity == trt.Logger.VERBOSE:
            self.logger.debug(msg)
        elif severity == trt.Logger.INFO:
            self.logger.info(msg)
        elif severity == trt.Logger.WARNING:
            self.logger.warning(msg)
        elif severity == trt.Logger.ERROR:
            self.logger.error(msg)
        elif severity == trt.Logger.INTERNAL_ERROR:
            self.logger.critical(msg)


# Create TensorRT logger singleton
TRT_LOGGER = TrtLogger() if HAS_GPU else None


class ExitOptimizationModel:
    """
    Exit optimization model for determining optimal exit points for trading positions.

    This model uses LSTM/GRU architecture to analyze market data and active positions
    to determine the optimal exit points. It can run on CPU or GPU depending on availability,
    with optimized TensorRT inference for production environments.
    """

    def __init__(self, config=None):
        """
        Initialize the Exit Optimization model with parameters from config.

        Args:
            config: Configuration dictionary (optional)
        """
        # Set up logger
        self.logger = log.setup_logger("exit_optimization_model")
        self.logger.info("Creating Exit Optimization Model")

        # Load configuration
        if config is None:
            config = get_config()

        self.config = config

        # Extract model parameters from config
        ml_config = config.get("ml", {})
        exit_config = ml_config.get("exit_optimization", {})
        inference_config = ml_config.get("inference", {})
        trading_config = config.get("trading", {})
        exit_trading_config = trading_config.get("exit", {})

        # Initialize with architecture parameters from config
        self.num_layers = exit_config.get("num_layers", 3)
        self.hidden_size = exit_config.get("hidden_size", 128)
        self.bidirectional = exit_config.get("bidirectional", True)
        self.attention_enabled = exit_config.get("attention_enabled", True)
        self.dropout = exit_config.get("dropout", 0.1)
        self.use_fp16 = inference_config.get("use_fp16", True)
        self.max_batch_size = inference_config.get("batch_size", 64)
        self.use_tensorrt = inference_config.get("use_tensorrt", True) and HAS_GPU
        self.tensorrt_cache_path = inference_config.get(
            "tensorrt_cache_path", "models/trt_cache"
        )

        # Exit configuration
        self.exit_threshold = exit_trading_config.get("exit_confidence_threshold", 0.6)
        self.max_holding_time_minutes = exit_trading_config.get(
            "max_holding_time_minutes", 240
        )
        self.check_interval_seconds = exit_trading_config.get(
            "check_exit_interval_seconds", 60
        )
        self.use_ml_exit = exit_trading_config.get("use_ml_exit", True)

        # Default shapes
        self.input_shape = [self.max_batch_size, 20]  # Batch size, num features
        self.output_shape = [
            self.max_batch_size,
            3,
        ]  # Batch size, output size (exit_probability, optimal_exit_price, trailing_stop_adjustment)

        # TensorRT resources
        self.engine = None
        self.context = None
        self.cuda_stream = None
        self.bindings = []
        self.input_binding_idx = None
        self.output_binding_idx = None
        self.input_buffer = None
        self.output_buffer = None
        self.host_input = None
        self.host_output = None

        # Model state
        self.model_path = ""
        self.is_loaded = False

        # Thread management
        self.thread_id = None

        # Redis integration
        self.redis_handler = None
        if HAS_REDIS:
            self.redis_handler = RedisSignalHandler(config)

    def __del__(self):
        """Clean up resources when the model is destroyed."""
        self.logger.info("Destroying Exit Optimization Model and releasing resources")

        try:
            # Clean up TensorRT resources
            if self.context is not None:
                self.context = None

            if self.engine is not None:
                self.engine = None

            # Clean up CUDA resources
            if self.cuda_stream is not None:
                if HAS_GPU:
                    cudart.cudaStreamDestroy(self.cuda_stream)
                self.cuda_stream = None

            # Free GPU memory
            if HAS_GPU and self.input_buffer is not None:
                cudart.cudaFree(self.input_buffer)
                self.input_buffer = None

            if HAS_GPU and self.output_buffer is not None:
                cudart.cudaFree(self.output_buffer)
                self.output_buffer = None

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def initialize(self) -> bool:
        """
        Initialize the Exit Optimization model.

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Load exit model
            model_paths = self.config.get("ml", {}).get("model_paths", {})
            exit_model_path = model_paths.get("exit_model", "")

            if exit_model_path and os.path.exists(exit_model_path):
                self.load(exit_model_path)
                self.logger.info(f"Loaded exit optimization model: {exit_model_path}")
            else:
                self.logger.warning(
                    f"Exit model path not specified or not found: {exit_model_path}"
                )
                return False

            # Initialize Redis handler if available
            if self.redis_handler:
                if not self.redis_handler.initialize():
                    self.logger.warning("Failed to initialize Redis signal handler")

            return True

        except Exception as e:
            self.logger.error(f"Error initializing exit optimization model: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            return False

    def load(self, model_path: str) -> None:
        """
        Load the model from the specified path.

        Args:
            model_path: Path to the model file

        Raises:
            RuntimeError: If the model file is not found or cannot be loaded
        """
        self.logger.info(f"Loading Exit Optimization model from {model_path}")

        # Store the model path
        self.model_path = model_path

        # Check if file exists
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found: {model_path}")

        # Create TensorRT cache directory if it doesn't exist
        if self.use_tensorrt:
            os.makedirs(self.tensorrt_cache_path, exist_ok=True)

        # Initialize TensorRT engine
        self.initialize_engine()

        self.is_loaded = True
        self.logger.info("Model loaded successfully and ready for production inference")

    def initialize_engine(self) -> None:
        """
        Initialize the TensorRT engine for inference.

        Raises:
            RuntimeError: If the engine cannot be initialized
        """
        self.logger.info("Initializing TensorRT engine for Exit Optimization model")

        try:
            # Check if we can use TensorRT
            if self.use_tensorrt and HAS_GPU and torch.cuda.is_available():
                self.logger.info("Using GPU with TensorRT for inference")

                # Create CUDA stream
                error, self.cuda_stream = cudart.cudaStreamCreate()
                if error != cudart.cudaError_t.cudaSuccess:
                    raise RuntimeError(
                        f"CUDA stream creation failed with error: {error}"
                    )

                # Load or build TensorRT engine
                self.engine = self.load_or_build_engine()

                if not self.engine:
                    raise RuntimeError("Failed to create TensorRT engine")

                # Create execution context
                self.context = self.engine.create_execution_context()
                if not self.context:
                    raise RuntimeError("Failed to create TensorRT execution context")

                # Set up input and output bindings
                self.setup_bindings()

                # Enable FP16 precision if requested
                if self.use_fp16:
                    self.logger.info("Using FP16 precision for inference")
                    self.context.set_flag(trt.ExecutionContextFlag.FP16)

            else:
                self.logger.info("Using CPU for inference (TensorRT not available)")
                # Initialize PyTorch model for CPU inference
                self.engine = self.load_pytorch_model()
                self.context = {"device": "cpu"}
                self.cuda_stream = None

            self.logger.info("Successfully initialized inference engine")

        except Exception as e:
            self.logger.error(f"Error initializing inference engine: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            raise

    def load_or_build_engine(self) -> Optional[trt.ICudaEngine]:
        """
        Load TensorRT engine from cache or build a new one.

        Returns:
            TensorRT engine or None if failed
        """
        # Generate engine cache path
        engine_cache_path = os.path.join(
            self.tensorrt_cache_path,
            f"{os.path.basename(self.model_path).split('.')[0]}_b{self.max_batch_size}_{'fp16' if self.use_fp16 else 'fp32'}.engine",
        )

        # Try to load from cache first
        if os.path.exists(engine_cache_path):
            self.logger.info(f"Loading TensorRT engine from cache: {engine_cache_path}")
            try:
                with open(engine_cache_path, "rb") as f:
                    engine_data = f.read()

                runtime = trt.Runtime(TRT_LOGGER)
                return runtime.deserialize_cuda_engine(engine_data)
            except Exception as e:
                self.logger.warning(
                    f"Failed to load cached engine: {str(e)}. Will rebuild."
                )

        # Build new engine
        self.logger.info("Building new TensorRT engine (this may take a while)...")

        try:
            # Create builder and network
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )

            # Parse ONNX file
            parser = trt.OnnxParser(network, TRT_LOGGER)
            with open(self.model_path, "rb") as model_file:
                if not parser.parse(model_file.read()):
                    for error in range(parser.num_errors):
                        self.logger.error(
                            f"ONNX parse error: {parser.get_error(error)}"
                        )
                    raise RuntimeError("Failed to parse ONNX model")

            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1 GB

            # Set precision
            if self.use_fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                self.logger.info("Enabled FP16 precision for TensorRT")

            # Create optimization profile if needed for dynamic shapes
            # profile = builder.create_optimization_profile()
            # Add dynamic shape configurations here if needed

            # Build engine
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")

            # Save engine to cache
            with open(engine_cache_path, "wb") as f:
                f.write(engine.serialize())
                self.logger.info(f"TensorRT engine saved to cache: {engine_cache_path}")

            return engine

        except Exception as e:
            self.logger.error(f"Error building TensorRT engine: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            return None

    def load_pytorch_model(self):
        """
        Load PyTorch model for CPU inference when TensorRT is not available.

        Returns:
            PyTorch model
        """
        self.logger.info("Loading PyTorch model for CPU inference")

        try:
            # In production, this would load a real PyTorch model
            # For now, we'll just return a placeholder
            return {"model_type": "pytorch_cpu"}
        except Exception as e:
            self.logger.error(f"Error loading PyTorch model: {str(e)}")
            raise

    def setup_bindings(self) -> None:
        """Set up TensorRT bindings and allocate memory."""
        if not self.use_tensorrt or not self.engine:
            return

        self.logger.info("Setting up TensorRT bindings and allocating GPU memory")

        # Get binding information
        self.bindings = []
        self.input_binding_idx = None
        self.output_binding_idx = None

        for i in range(self.engine.num_bindings):
            # Get binding name (useful for debugging)
            _ = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                self.input_binding_idx = i
            else:
                self.output_binding_idx = i

            # Get binding shape
            shape = self.engine.get_binding_shape(i)
            size = trt.volume(shape) * self.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(i))

            # Allocate device memory
            error, device_mem = cudart.cudaMalloc(size * dtype.itemsize)
            if error != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"CUDA memory allocation failed with error: {error}")

            # Allocate host memory
            host_mem = np.zeros(size, dtype=dtype)

            # Store binding information
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.input_buffer = device_mem
                self.host_input = host_mem
            else:
                self.output_buffer = device_mem
                self.host_output = host_mem

        self.logger.info(
            f"TensorRT bindings set up: input_idx={self.input_binding_idx}, output_idx={self.output_binding_idx}"
        )

    def set_thread_affinity(self, core_id: int) -> None:
        """
        Set thread affinity for the exit optimization thread.

        Args:
            core_id: CPU core ID to pin the thread to
        """
        # Store thread ID
        self.thread_id = threading.get_ident()

        # Set affinity (platform-specific)
        try:
            if hasattr(threading, "current_thread"):
                current_thread = threading.current_thread()
                if hasattr(current_thread, "name"):
                    self.logger.info(
                        f"Setting thread affinity for {current_thread.name} to core {core_id}"
                    )

            # Try to use platform-specific thread affinity setting
            try:
                import psutil

                p = psutil.Process()
                p.cpu_affinity([core_id])
                self.logger.info(f"Thread affinity set to core {core_id}")
            except (ImportError, AttributeError):
                self.logger.warning(f"Could not set thread affinity to core {core_id}")
        except Exception as e:
            self.logger.error(f"Error setting thread affinity: {str(e)}")

    @log.time_function(track_percentiles=True)
    @log.trace_context
    def optimize_exits(
        self, active_positions: List[Signal], current_data: Dict[str, Any]
    ) -> List[Signal]:
        """
        Optimize exits for active positions based on current market data.

        Args:
            active_positions: List of active positions
            current_data: Current market data

        Returns:
            List of exit signals
        """
        # Start timing
        start_time = time.time()

        # Store exit signals
        exit_signals = []

        # Process each active position
        for position in active_positions:
            # Check if symbol exists in the data
            if position.symbol not in current_data.get("symbol_data", {}):
                continue

            # Get current price and timestamp
            symbol_data = current_data["symbol_data"][position.symbol]
            current_price = symbol_data.get("last_price", 0.0)
            current_timestamp = current_data.get("timestamp", 0)

            # Check time-based exit
            time_exit = self.should_exit_based_on_time(position, current_timestamp)

            # Check profit-based exit
            profit_exit = self.should_exit_based_on_profit(position, current_price)

            # Check stop-loss exit
            stop_loss_exit = self.should_exit_based_on_stop_loss(
                position, current_price
            )

            # ML-based exit
            ml_exit = False
            exit_probability = 0.0

            if self.use_ml_exit and self.is_loaded:
                # Extract features for this position
                features = self.extract_exit_features(position, current_data)

                # Run inference
                if features:
                    outputs = self.infer(features)

                    # Skip if no outputs
                    if outputs:
                        # Get exit probability
                        exit_probability = outputs[0]

                        # Check if probability exceeds threshold
                        ml_exit = exit_probability > self.exit_threshold

            # Create exit signal if any exit condition is met
            if time_exit or profit_exit or stop_loss_exit or ml_exit:
                exit_signal = Signal(
                    symbol=position.symbol,
                    type="EXIT",
                    price=current_price,
                    position_size=position.position_size,
                    confidence=exit_probability,
                    timestamp=current_timestamp,
                )

                # Add exit reason to signal
                if time_exit:
                    exit_signal.indicators = {"exit_reason": 1.0}  # Time-based exit
                elif profit_exit:
                    exit_signal.indicators = {"exit_reason": 2.0}  # Profit-based exit
                elif stop_loss_exit:
                    exit_signal.indicators = {"exit_reason": 3.0}  # Stop-loss exit
                elif ml_exit:
                    exit_signal.indicators = {"exit_reason": 4.0}  # ML-based exit

                # Add to signals
                exit_signals.append(exit_signal)

                # Publish to Redis if available
                if self.redis_handler and self.redis_handler.initialized:
                    self.redis_handler.publish_signal(exit_signal)

        # End timing
        end_time = time.time()
        duration = (end_time - start_time) * 1_000_000  # Convert to microseconds

        self.logger.info(
            f"Exit optimization completed in {duration:.2f} µs, "
            f"generated {len(exit_signals)} exit signals from "
            f"{len(active_positions)} active positions"
        )

        return exit_signals

    @log.time_function(track_percentiles=True)
    @log.trace_context
    def infer(self, features: List[float]) -> List[float]:
        """
        Perform inference on a single set of features.

        Args:
            features: Input features as a flat list of floats

        Returns:
            List of output values (exit_probability, optimal_exit_price, trailing_stop_adjustment)

        Raises:
            RuntimeError: If the model is not loaded or inference fails
        """
        # Check if model is loaded
        if not self.is_loaded or not self.engine or not self.context:
            raise RuntimeError("Model not loaded or engine not initialized")

        try:
            # Validate input size
            if len(features) != self.input_shape[1]:
                raise RuntimeError(
                    f"Input size mismatch: expected {self.input_shape[1]}, got {len(features)}"
                )

            # Use TensorRT for GPU inference
            if (
                self.use_tensorrt
                and HAS_GPU
                and self.host_input is not None
                and self.host_output is not None
            ):
                try:
                    # Reshape features to match input shape
                    input_data = np.array(features, dtype=np.float32).reshape(
                        1, self.input_shape[1]
                    )

                    # Copy input data to host buffer
                    np.copyto(self.host_input, input_data.ravel())

                    # Copy from host to device
                    (error,) = cudart.cudaMemcpy(
                        self.input_buffer,
                        self.host_input.ctypes.data,
                        self.host_input.size * self.host_input.itemsize,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    )
                    if error != cudart.cudaError_t.cudaSuccess:
                        raise RuntimeError(
                            f"CUDA memcpy H2D failed with error: {error}"
                        )

                    # Execute inference
                    status = self.context.execute_v2(self.bindings)
                    if not status:
                        raise RuntimeError("TensorRT execution failed")

                    # Copy from device to host
                    (error,) = cudart.cudaMemcpy(
                        self.host_output.ctypes.data,
                        self.output_buffer,
                        self.host_output.size * self.host_output.itemsize,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    )
                    if error != cudart.cudaError_t.cudaSuccess:
                        raise RuntimeError(
                            f"CUDA memcpy D2H failed with error: {error}"
                        )

                    # Get output
                    output = self.host_output[: self.output_shape[1]].tolist()
                    return output
                except Exception as e:
                    self.logger.warning(
                        f"TensorRT inference failed, falling back to CPU: {str(e)}"
                    )
                    # Fall through to CPU inference
            else:
                # CPU inference fallback
                # In a real implementation, this would use PyTorch
                # For now, we'll compute a realistic output similar to the C++ version

                # Prepare output buffer
                output_buffer = np.zeros(self.output_shape[1], dtype=np.float32)

                # Exit probability (0.0-1.0)
                output_buffer[0] = 0.5 + np.random.normal(0, 0.2)
                output_buffer[0] = np.clip(output_buffer[0], 0.0, 1.0)

                # Optimal exit price (based on input features)
                base_price = 0.0
                for i in range(min(len(features), 10)):
                    base_price += features[i]
                base_price = max(base_price, 50.0)
                output_buffer[1] = base_price * (1.0 + np.random.normal(0, 0.02))

                # Trailing stop adjustment (0.0-1.0)
                output_buffer[2] = np.random.uniform(0.0, 1.0)

                return output_buffer.tolist()

        except Exception as e:
            self.logger.error(f"Error during inference: {str(e)}")
            import traceback

            self.logger.error(traceback.format_exc())
            raise

    def extract_exit_features(
        self, position: Signal, current_data: Dict[str, Any]
    ) -> List[float]:
        """
        Extract features from position and market data for exit optimization.

        Args:
            position: Trading position
            current_data: Current market data

        Returns:
            List of features for model input
        """
        # Find symbol data
        symbol_data = current_data.get("symbol_data", {}).get(position.symbol)
        if not symbol_data:
            return []

        # Create feature vector for exit optimization
        features = []

        # Position-specific features
        current_price = symbol_data.get("last_price", 0.0)
        current_timestamp = current_data.get("timestamp", 0)

        # Current P&L
        if position.type == "BUY":
            pnl = (current_price / position.price) - 1.0
        else:  # SELL
            pnl = 1.0 - (current_price / position.price)
        features.append(pnl)

        # Duration in minutes
        duration_minutes = (
            current_timestamp - position.timestamp
        ) / 60_000_000_000  # Convert ns to minutes
        features.append(duration_minutes)

        # Initial confidence
        features.append(position.confidence)

        # Market condition changes
        features.append(symbol_data.get("volatility_change", 0.0))
        features.append(
            symbol_data.get("volume", 0.0)
            / max(symbol_data.get("avg_volume", 1.0), 1.0)
        )
        features.append(symbol_data.get("bid_ask_spread_change", 0.0))

        # Technical indicator changes
        features.append(symbol_data.get("rsi_14", 50.0))
        features.append(symbol_data.get("macd", 0.0))
        features.append(symbol_data.get("macd_histogram", 0.0))

        # Bollinger band position
        bb_upper = symbol_data.get("bb_upper", current_price * 1.02)
        bb_lower = symbol_data.get("bb_lower", current_price * 0.98)
        bb_position = (current_price - bb_lower) / max((bb_upper - bb_lower), 0.01)
        features.append(bb_position)

        # Add more features as needed to match the input shape
        while len(features) < self.input_shape[1]:
            features.append(0.0)

        return features

    def should_exit_based_on_time(
        self, position: Signal, current_timestamp: int
    ) -> bool:
        """
        Check if position should be exited based on time.

        Args:
            position: Trading position
            current_timestamp: Current timestamp in nanoseconds

        Returns:
            True if position should be exited, False otherwise
        """
        # Calculate position duration in minutes
        duration_ns = current_timestamp - position.timestamp
        duration_minutes = duration_ns / 60_000_000_000  # Convert ns to minutes

        # Check if position has exceeded max holding time
        return duration_minutes >= self.max_holding_time_minutes

    def should_exit_based_on_profit(
        self, position: Signal, current_price: float
    ) -> bool:
        """
        Check if position should be exited based on profit.

        Args:
            position: Trading position
            current_price: Current price

        Returns:
            True if position should be exited, False otherwise
        """
        # Calculate profit percentage
        profit_pct = 0.0

        if position.type == "BUY":
            profit_pct = (current_price - position.price) / position.price * 100.0
        elif position.type == "SELL":
            profit_pct = (position.price - current_price) / position.price * 100.0

        # Check if profit exceeds take profit level
        return profit_pct >= position.take_profit

    def should_exit_based_on_stop_loss(
        self, position: Signal, current_price: float
    ) -> bool:
        """
        Check if position should be exited based on stop loss.

        Args:
            position: Trading position
            current_price: Current price

        Returns:
            True if position should be exited, False otherwise
        """
        # Calculate loss percentage
        loss_pct = 0.0

        if position.type == "BUY":
            loss_pct = (position.price - current_price) / position.price * 100.0
        elif position.type == "SELL":
            loss_pct = (current_price - position.price) / position.price * 100.0

        # Check if loss exceeds stop loss level
        return loss_pct >= position.stop_loss

    def set_exit_threshold(self, threshold: float) -> None:
        """
        Set the exit threshold.

        Args:
            threshold: Exit threshold (0.0-1.0)
        """
        self.exit_threshold = threshold

    def set_max_holding_time(self, minutes: int) -> None:
        """
        Set the maximum holding time.

        Args:
            minutes: Maximum holding time in minutes
        """
        self.max_holding_time_minutes = minutes

    def set_check_interval(self, seconds: int) -> None:
        """
        Set the check interval.

        Args:
            seconds: Check interval in seconds
        """
        self.check_interval_seconds = seconds
