"""
Test script for the GBDT model

This script demonstrates how to use the GBDT model with the project's
data sources and logger.
"""

import os
import sys
import time
import traceback
import json
import numpy as np
import threading
from pathlib import Path
from typing import Dict

# Add project root to path to allow imports
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

# Import project modules - must be after sys.path modification
from config.config_loader import get_config  # noqa: E402
from src.monitoring.log import logging as log  # noqa: E402
from src.ml.gbdt_model import GBDTModel  # noqa: E402

def main():
    """Main test function for the GBDT model."""
    # Set up logger
    logger = log.setup_logger("test_gbdt_model")
    logger.info("Starting GBDT model test")
    
    try:
        # Load configuration
        config = get_config()
        logger.info("Configuration loaded")
        
        # Create model
        model = GBDTModel(config)
        logger.info(f"Created model: {model.get_name()}")
        
        # Use the model configuration from JSON - no fallbacks
        model_config_path = Path(project_root) / "models" / "gbdt" / "model_config.json"
        if not model_config_path.exists():
            raise FileNotFoundError(f"Model configuration file not found: {model_config_path}")
            
        logger.info(f"Loading model configuration from {model_config_path}")
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
            
        # Configure model with parameters from config
        if hasattr(model, 'set_num_trees'):
            model.set_num_trees(model_config.get("num_trees", 250))
        if hasattr(model, 'set_max_depth'):
            model.set_max_depth(model_config.get("max_depth", 10))
        if hasattr(model, 'set_learning_rate'):
            model.set_learning_rate(model_config.get("learning_rate", 0.03))
        if hasattr(model, 'set_feature_fraction'):
            model.set_feature_fraction(model_config.get("feature_fraction", 0.75))
        
        # Check if we have a real model file
        model_path = model_config.get("model_path")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at path specified in config: {model_path}")
        
        # Load the model
        logger.info(f"Loading model from {model_path}")
        model.load(model_path)
        
        # Test with synthetic data
        test_with_synthetic_data(model, logger)
        
        # Test batch processing
        test_batch_processing(model, logger)
        
        # Test market data processing
        test_market_data_processing(model, logger)
        
        # Test threading
        test_threading(model, logger)
        
        logger.info("Test completed successfully")
    
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

# Removed create_dummy_model function as we're only using actual models from configuration

def test_with_synthetic_data(model, logger):
    """Test the model with synthetic data."""
    logger.info("Testing with synthetic data")
    
    # Create synthetic features
    input_shape = model.get_input_shape()
    feature_count = input_shape[1]
    
    # Generate random features
    features = np.random.normal(0, 1, feature_count).tolist()
    
    # Run inference
    start_time = time.time()
    result = model.infer(features)
    elapsed_ms = (time.time() - start_time) * 1000
    
    logger.info(f"Inference completed in {elapsed_ms:.2f} ms")
    logger.info(f"Result: {result}")
    
    # Test model parameter setters
    logger.info("Testing model parameter setters")
    
    # Save original values
    original_num_trees = model.num_trees
    original_max_depth = model.max_depth
    original_learning_rate = model.learning_rate
    
    # Set new values
    model.set_num_trees(original_num_trees + 10)
    model.set_max_depth(original_max_depth + 2)
    model.set_learning_rate(original_learning_rate * 1.1)
    
    # Verify changes
    logger.info(f"Updated num_trees: {model.num_trees}")
    logger.info(f"Updated max_depth: {model.max_depth}")
    logger.info(f"Updated learning_rate: {model.learning_rate}")
    
    # Run inference with new parameters
    result_new = model.infer(features)
    logger.info(f"Result with new parameters: {result_new}")
    
    # Restore original values
    model.set_num_trees(original_num_trees)
    model.set_max_depth(original_max_depth)
    model.set_learning_rate(original_learning_rate)
    
    # Get feature importance
    feature_importance = model.get_feature_importance()
    if feature_importance:
        # Get top 5 features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("Top 5 features by importance:")
        for feature, importance in top_features:
            logger.info(f"  - {feature}: {importance:.4f}")

def test_batch_processing(model, logger):
    """Test batch processing with the model."""
    logger.info("Testing batch processing")
    
    # Create synthetic features
    input_shape = model.get_input_shape()
    feature_count = input_shape[1]
    
    # Generate random features for batch
    batch_size = 10
    features_batch = [np.random.normal(0, 1, feature_count).tolist() for _ in range(batch_size)]
    
    # Run batch inference
    try:
        start_time = time.time()
        batch_results = model.infer_batch(features_batch)
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Batch inference (size={batch_size}) completed in {elapsed_ms:.2f} ms")
        logger.info(f"Average time per item: {elapsed_ms / batch_size:.2f} ms")
        logger.info(f"First batch result: {batch_results[0]}")
        
        # Test different batch sizes
        for batch_size in [1, 5, 20, 50]:
            features_batch = [np.random.normal(0, 1, feature_count).tolist() for _ in range(batch_size)]
            
            start_time = time.time()
            model.infer_batch(features_batch)
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Batch size {batch_size}: {elapsed_ms:.2f} ms total, {elapsed_ms / batch_size:.2f} ms per item")
    except Exception as e:
        logger.warning(f"Batch inference failed: {str(e)}. Falling back to individual inference.")
        # Process each item individually as fallback
        batch_results = []
        start_time = time.time()
        for features in features_batch:
            result = model.infer(features)
            batch_results.append(result)
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Individual inference for {batch_size} items completed in {elapsed_ms:.2f} ms")
        logger.info(f"Average time per item: {elapsed_ms / batch_size:.2f} ms")
        logger.info(f"First result: {batch_results[0]}")
    
    # Test performance monitoring
    logger.info(f"Total inference calls: {model.total_inference_calls}")
    logger.info(f"Total inference time: {model.total_inference_time_ms:.2f} ms")
    if model.total_inference_calls > 0:
        logger.info(f"Average inference time: {model.total_inference_time_ms / model.total_inference_calls:.2f} ms")

def test_market_data_processing(model, logger):
    """Test market data processing with the model."""
    logger.info("Testing market data processing")
    
    # Create synthetic market data
    market_data = create_synthetic_market_data()
    
    # Process market data
    start_time = time.time()
    signals = model.process_market_data(market_data)
    elapsed_ms = (time.time() - start_time) * 1000
    
    logger.info(f"Market data processing completed in {elapsed_ms:.2f} ms")
    logger.info(f"Generated {len(signals)} signals")
    
    # Log signals
    for symbol, signal in signals.items():
        logger.info(f"{symbol}: score={signal[0]:.4f}")

def test_threading(model, logger):
    """Test threading with the model."""
    logger.info("Testing threading")
    
    # Create a function to run inference in a thread
    def run_inference(thread_id, results):
        input_shape = model.get_input_shape()
        feature_count = input_shape[1]
        features = np.random.normal(0, 1, feature_count).tolist()
        
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
    logger.info(f"Concurrent inference with {num_threads} threads completed in {total_time:.2f} ms total")
    logger.info(f"Average time per thread: {total_time / num_threads:.2f} ms")

def create_synthetic_market_data() -> Dict:
    """Create synthetic market data for testing."""
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    market_data = {}
    
    for symbol in symbols:
        price = np.random.uniform(100, 1000)
        
        # Create feature dictionary
        features = {
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
            "sma_cross_signal": np.random.uniform(-1, 1)
        }
        
        market_data[symbol] = features
    
    return market_data

# Removed train_minimal_model and generate_synthetic_training_data functions
# as we're only using actual models from configuration

if __name__ == "__main__":
    sys.exit(main())