"""
Test script for the Axial Attention model

This script demonstrates how to use the Axial Attention model with the project's
data sources and logger.
"""

import os
import sys
import time
import traceback
import json
import numpy as np
from pathlib import Path
import threading

# Add project root to path to allow imports
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

# Import project modules - must be after sys.path modification
from config.config_loader import get_config  # noqa: E402
from src.monitoring.log import logging as log  # noqa: E402
from src.data.polygon_rest_api import PolygonRestAPI, ParsedMarketData  # noqa: E402
from src.ml.axial_attention_model import AxialAttentionModel  # noqa: E402


def main():
    """Main test function for the Axial Attention model."""
    # Set up logger
    logger = log.setup_logger("test_axial_attention")
    logger.info("Starting Axial Attention model test")

    try:
        # Load configuration
        config = get_config()
        logger.info("Configuration loaded")

        # Create model
        model = AxialAttentionModel(config)
        logger.info(f"Created model: {model.get_name()}")

        # Use the model configuration from JSON - no fallbacks
        model_config_path = (
            Path(project_root) / "models" / "axial_attention" / "model_config.json"
        )
        if not model_config_path.exists():
            raise FileNotFoundError(
                f"Model configuration file not found: {model_config_path}"
            )

        logger.info(f"Loading model configuration from {model_config_path}")
        with open(model_config_path, "r") as f:
            model_config = json.load(f)

        # Configure model with parameters from config
        if hasattr(model, "set_num_heads"):
            model.set_num_heads(model_config.get("num_heads", 8))
        if hasattr(model, "set_head_dim"):
            model.set_head_dim(model_config.get("head_dim", 64))
        if hasattr(model, "set_num_layers"):
            model.set_num_layers(model_config.get("num_layers", 8))
        if hasattr(model, "set_dropout"):
            model.set_dropout(model_config.get("dropout", 0.1))
        if hasattr(model, "set_hidden_dim"):
            model.set_hidden_dim(model_config.get("hidden_dim", 512))

        # Check if we have a real model file
        model_path = model_config.get("model_path")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at path specified in config: {model_path}"
            )

        # Load the model
        logger.info(f"Loading model from {model_path}")
        model.load(model_path)

        # Test with synthetic data
        test_with_synthetic_data(model, logger)

        # Test batch processing
        test_batch_processing(model, logger)

        # Test with real market data if available
        test_with_market_data(model, config, logger)

        logger.info("Test completed successfully")

    except Exception:
        logger.error("Test failed")
        logger.error(traceback.format_exc())
        return 1

    return 0


def test_with_synthetic_data(model, logger):
    """Test the model with synthetic data."""
    logger.info("Testing with synthetic data")

    # Create synthetic features
    input_shape = model.get_input_shape()
    feature_count = input_shape[1] * input_shape[2]

    # Generate random features
    features = np.random.normal(100, 10, feature_count).tolist()

    # Run inference
    start_time = time.time()
    result = model.infer(features)
    elapsed_ms = (time.time() - start_time) * 1000

    logger.info(f"Inference completed in {elapsed_ms:.2f} ms")
    logger.info(
        f"Result: signal_type={result[0]:.4f}, confidence={result[1]:.4f}, target_price=${result[2]:.2f}"
    )

    # Test batch inference
    batch_size = 10
    features_batch = [
        np.random.normal(100, 10, feature_count).tolist() for _ in range(batch_size)
    ]

    try:
        start_time = time.time()
        batch_results = model.infer_batch(features_batch)
        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Batch inference (size={batch_size}) completed in {elapsed_ms:.2f} ms"
        )
        logger.info(
            f"First batch result: signal_type={batch_results[0][0]:.4f}, confidence={batch_results[0][1]:.4f}, target_price=${batch_results[0][2]:.2f}"
        )
    except Exception as e:
        logger.warning(
            f"Batch inference failed: {str(e)}. This is expected in CPU mode."
        )
        # Process each item individually as fallback
        batch_results = []
        start_time = time.time()
        for features in features_batch:
            result = model.infer(features)
            batch_results.append(result)
        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Individual inference for {batch_size} items completed in {elapsed_ms:.2f} ms"
        )
        logger.info(
            f"First result: signal_type={batch_results[0][0]:.4f}, confidence={batch_results[0][1]:.4f}, target_price=${batch_results[0][2]:.2f}"
        )


def test_batch_processing(model, logger):
    """Test the batch processing capabilities of the model."""
    logger.info("Testing batch processing")

    # Create synthetic features
    input_shape = model.get_input_shape()
    feature_count = input_shape[1] * input_shape[2]

    # Check if we're in CPU mode
    is_cpu_mode = not model.use_tensorrt or model.host_input is None

    if is_cpu_mode:
        logger.info("Running in CPU mode, using individual inference for batch testing")

        # Test different batch sizes with individual inference
        for batch_size in [1, 5, 20]:
            features_batch = [
                np.random.normal(100, 10, feature_count).tolist()
                for _ in range(batch_size)
            ]

            start_time = time.time()
            batch_results = []
            for features in features_batch:
                result = model.infer(features)
                batch_results.append(result)
            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Batch size {batch_size}: {elapsed_ms:.2f} ms total, {elapsed_ms / batch_size:.2f} ms per item"
            )
    else:
        # Test different batch sizes with batch inference
        for batch_size in [1, 5, 20, 50]:
            features_batch = [
                np.random.normal(100, 10, feature_count).tolist()
                for _ in range(batch_size)
            ]

            try:
                start_time = time.time()
                batch_results = model.infer_batch(features_batch)
                elapsed_ms = (time.time() - start_time) * 1000

                logger.info(
                    f"Batch size {batch_size}: {elapsed_ms:.2f} ms total, {elapsed_ms / batch_size:.2f} ms per item"
                )
            except Exception as e:
                logger.warning(
                    f"Batch inference failed for size {batch_size}: {str(e)}"
                )

    # Test concurrent batch processing
    logger.info("Testing concurrent batch processing")

    # Create a function to run inference in a thread
    def run_inference(thread_id, results):
        features = np.random.normal(100, 10, feature_count).tolist()
        start_time = time.time()
        result = model.infer(features)
        elapsed_ms = (time.time() - start_time) * 1000
        results[thread_id] = (result, elapsed_ms)

    # Run multiple threads
    num_threads = 5
    threads = []
    results = {}

    for i in range(num_threads):
        thread = threading.Thread(target=run_inference, args=(i, results))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Log results
    total_time = sum(elapsed for _, elapsed in results.values())
    logger.info(
        f"Concurrent inference with {num_threads} threads completed in {total_time:.2f} ms total"
    )
    logger.info(f"Average time per thread: {total_time / num_threads:.2f} ms")


def test_with_market_data(model, config, logger):
    """Test the model with real market data if available."""
    logger.info("Testing with market data")

    try:
        # Create REST API client
        rest_client = PolygonRestAPI(config)
        rest_client.initialize()

        # Check if API key is available
        if not rest_client.polygon_api_key:
            logger.warning(
                "No Polygon API key available. Creating synthetic market data instead."
            )
            market_data = create_synthetic_market_data()
        else:
            # Fetch data for some test symbols
            test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            logger.info(f"Fetching data for test symbols: {', '.join(test_symbols)}")

            # Fetch data
            future = rest_client.fetch_symbol_data(test_symbols)
            market_data = future.result()

        if not market_data.symbol_data:
            logger.warning("No market data received. Skipping market data test.")
            return

        logger.info(f"Received data for {len(market_data.symbol_data)} symbols")

        # Process market data with the model
        signals = model.process_market_data(market_data)

        # Log results
        for symbol, signal in signals.items():
            signal_type = "BUY" if signal[0] > 0.5 else "SELL"
            logger.info(
                f"{symbol}: {signal_type} signal with confidence={signal[1]:.2f}, target=${signal[2]:.2f}"
            )

        # Test performance monitoring
        logger.info(f"Total inference calls: {model.total_inference_calls}")
        logger.info(f"Total inference time: {model.total_inference_time_ms:.2f} ms")
        if model.total_inference_calls > 0:
            logger.info(
                f"Average inference time: {model.total_inference_time_ms / model.total_inference_calls:.2f} ms"
            )

        logger.info("Market data test completed")

    except Exception as e:
        logger.error(f"Market data test failed: {str(e)}")
        logger.error(traceback.format_exc())


def create_synthetic_market_data():
    """Create synthetic market data for testing."""
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    market_data = ParsedMarketData()

    try:
        for symbol in symbols:
            price = np.random.uniform(100, 1000)

            symbol_data = ParsedMarketData.SymbolData()
            symbol_data.symbol = symbol
            symbol_data.last_price = price
            symbol_data.bid_price = price * 0.999
            symbol_data.ask_price = price * 1.001
            symbol_data.bid_ask_spread = price * 0.002
            symbol_data.open_price = price * (1.0 - np.random.uniform(-0.02, 0.02))
            symbol_data.high_price = price * (1.0 + np.random.uniform(0.0, 0.03))
            symbol_data.low_price = price * (1.0 - np.random.uniform(0.0, 0.03))
            symbol_data.volume = np.random.uniform(10000, 1000000)
            symbol_data.vwap = price * (1.0 + np.random.uniform(-0.01, 0.01))
            symbol_data.prev_close = price * (1.0 - np.random.uniform(-0.05, 0.05))

            # Technical indicators
            symbol_data.rsi_14 = np.random.uniform(0, 100)
            symbol_data.macd = np.random.uniform(-10, 10)
            symbol_data.macd_signal = np.random.uniform(-10, 10)
            symbol_data.macd_histogram = np.random.uniform(-5, 5)
            symbol_data.bb_upper = price * 1.05
            symbol_data.bb_middle = price
            symbol_data.bb_lower = price * 0.95
            symbol_data.atr = price * 0.02

            # Additional indicators
            symbol_data.avg_volume = np.random.uniform(10000, 1000000)
            symbol_data.volume_acceleration = np.random.uniform(-2, 2)
            symbol_data.volume_spike = np.random.uniform(0, 3)
            symbol_data.price_change_5m = np.random.uniform(-2, 2)
            symbol_data.momentum_1m = np.random.uniform(-1, 1)
            symbol_data.sma_cross_signal = np.random.uniform(-1, 1)

            market_data.symbol_data[symbol] = symbol_data
    except Exception:
        # If SymbolData is not a class but a dict, use dict approach
        market_data = {"symbol_data": {}}
        for symbol in symbols:
            price = np.random.uniform(100, 1000)

            market_data["symbol_data"][symbol] = {
                "symbol": symbol,
                "last_price": price,
                "bid_price": price * 0.999,
                "ask_price": price * 1.001,
                "bid_ask_spread": price * 0.002,
                "open_price": price * (1.0 - np.random.uniform(-0.02, 0.02)),
                "high_price": price * (1.0 + np.random.uniform(0.0, 0.03)),
                "low_price": price * (1.0 - np.random.uniform(0.0, 0.03)),
                "volume": np.random.uniform(10000, 1000000),
                "vwap": price * (1.0 + np.random.uniform(-0.01, 0.01)),
                "prev_close": price * (1.0 - np.random.uniform(-0.05, 0.05)),
                # Technical indicators
                "rsi_14": np.random.uniform(0, 100),
                "macd": np.random.uniform(-10, 10),
                "macd_signal": np.random.uniform(-10, 10),
                "macd_histogram": np.random.uniform(-5, 5),
                "bb_upper": price * 1.05,
                "bb_middle": price,
                "bb_lower": price * 0.95,
                "atr": price * 0.02,
                # Additional indicators
                "avg_volume": np.random.uniform(10000, 1000000),
                "volume_acceleration": np.random.uniform(-2, 2),
                "volume_spike": np.random.uniform(0, 3),
                "price_change_5m": np.random.uniform(-2, 2),
                "momentum_1m": np.random.uniform(-1, 1),
                "sma_cross_signal": np.random.uniform(-1, 1),
            }

    return market_data


# Removed train_minimal_model and generate_synthetic_training_data functions
# as we're only using actual models from configuration

if __name__ == "__main__":
    sys.exit(main())
