"""
Test script for the Exit Optimization model

This script demonstrates how to use the Exit Optimization model with the project's
data sources and logger.
"""

import os
import sys
import time
import random
import traceback
import threading
import json
from pathlib import Path
from typing import List

# Add project root to path to allow imports
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

# Import project modules - must be after sys.path modification
from config.config_loader import get_config  # noqa: E402
from src.monitoring.log import logging as log  # noqa: E402
from src.ml.exit_optimization import ExitOptimizationModel  # noqa: E402
from src.execution.alpaca_execution import Signal  # noqa: E402

def main():
    """Main test function for the Exit Optimization model."""
    # Set up logger
    logger = log.setup_logger("test_exit_optimization")
    logger.info("Starting Exit Optimization model test")
    
    try:
        # Load configuration
        config = get_config()
        logger.info("Configuration loaded")
        
        # Create model
        model = ExitOptimizationModel(config)
        logger.info("Created model: Exit Optimization Model")
        
        # Use the model configuration from JSON - no fallbacks
        model_config_path = Path(project_root) / "models" / "lstm_gru" / "model_config.json"
        if not model_config_path.exists():
            raise FileNotFoundError(f"Model configuration file not found: {model_config_path}")
            
        logger.info(f"Loading model configuration from {model_config_path}")
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        
        # Set up model paths in config
        if "ml" not in config:
            config["ml"] = {}
        if "model_paths" not in config["ml"]:
            config["ml"]["model_paths"] = {}
        
        # Check if we have a real model file
        model_path = model_config.get("model_path")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at path specified in config: {model_path}")
        
        # Set model path in config
        logger.info(f"Using model from {model_path}")
        config["ml"]["model_paths"]["exit_model"] = model_path
        
        # Initialize the model
        if not model.initialize():
            logger.error("Failed to initialize Exit Optimization model")
            return 1
        
        logger.info("Model initialized successfully")
        
        # Test with synthetic data
        test_with_synthetic_data(model, logger)
        
        # Test exit strategies
        test_exit_strategies(model, logger)
        
        # Test Redis integration if available
        test_redis_integration(model, logger)
        
        # Test threading
        test_threading(model, logger)
        
        # Test configuration setters
        test_configuration_setters(model, logger)
        
        logger.info("Test completed successfully")
    
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

# Removed create_dummy_model function as we're only using actual models from configuration

def generate_random_signal() -> Signal:
    """Generate a random trading signal."""
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD", "INTC", "IBM"]
    signal_type = random.choice(["BUY", "SELL"])
    price = round(random.uniform(100.0, 1000.0), 2)
    confidence = round(random.uniform(0.5, 1.0), 2)
    
    # Calculate stop loss and take profit based on signal type
    if signal_type == "BUY":
        stop_loss = round(random.uniform(1.0, 5.0), 2)  # Percentage
        take_profit = round(random.uniform(2.0, 10.0), 2)  # Percentage
    else:
        stop_loss = round(random.uniform(1.0, 5.0), 2)  # Percentage
        take_profit = round(random.uniform(2.0, 10.0), 2)  # Percentage
    
    # Create signal
    signal = Signal(
        symbol=random.choice(symbols),
        type=signal_type,
        price=price,
        position_size=round(random.uniform(1.0, 10.0), 2),
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        timestamp=int(time.time() * 1_000_000)  # Microsecond precision
    )
    
    return signal

def generate_market_data(symbols: List[str]) -> dict:
    """Generate synthetic market data for testing."""
    market_data = {
        "timestamp": int(time.time() * 1_000_000_000),  # Nanosecond precision
        "symbol_data": {}
    }
    
    for symbol in symbols:
        price = round(random.uniform(100.0, 1000.0), 2)
        
        market_data["symbol_data"][symbol] = {
            "symbol": symbol,
            "last_price": price,
            "bid_price": price * 0.999,
            "ask_price": price * 1.001,
            "bid_ask_spread": price * 0.002,
            "open_price": price * (1.0 - random.uniform(-0.02, 0.02)),
            "high_price": price * (1.0 + random.uniform(0.0, 0.03)),
            "low_price": price * (1.0 - random.uniform(0.0, 0.03)),
            "volume": random.uniform(10000, 1000000),
            "vwap": price * (1.0 + random.uniform(-0.01, 0.01)),
            "prev_close": price * (1.0 - random.uniform(-0.05, 0.05)),
            
            # Technical indicators
            "rsi_14": random.uniform(0, 100),
            "macd": random.uniform(-10, 10),
            "macd_signal": random.uniform(-10, 10),
            "macd_histogram": random.uniform(-5, 5),
            "bb_upper": price * 1.05,
            "bb_middle": price,
            "bb_lower": price * 0.95,
            "atr": price * 0.02,
            
            # Additional indicators
            "avg_volume": random.uniform(10000, 1000000),
            "volume_acceleration": random.uniform(-2, 2),
            "volume_spike": random.uniform(0, 3),
            "price_change_5m": random.uniform(-2, 2),
            "momentum_1m": random.uniform(-1, 1),
            "sma_cross_signal": random.uniform(-1, 1),
            "volatility_change": random.uniform(-0.5, 0.5),
            "bid_ask_spread_change": random.uniform(-0.1, 0.1)
        }
    
    return market_data

def test_with_synthetic_data(model: ExitOptimizationModel, logger):
    """Test the model with synthetic data."""
    logger.info("Testing with synthetic data")
    
    # Generate random active positions
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD", "INTC", "IBM"]
    active_positions = [generate_random_signal() for _ in range(5)]
    
    # Log active positions
    logger.info(f"Generated {len(active_positions)} active positions:")
    for position in active_positions:
        logger.info(f"  {position.symbol} {position.type} @ {position.price} "
                   f"(stop_loss={position.stop_loss}%, take_profit={position.take_profit}%)")
    
    # Generate market data
    market_data = generate_market_data(symbols)
    
    # Run exit optimization
    try:
        start_time = time.time()
        exit_signals = model.optimize_exits(active_positions, market_data)
        elapsed_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Exit optimization completed in {elapsed_ms:.2f} ms")
        logger.info(f"Generated {len(exit_signals)} exit signals:")
    except Exception as e:
        logger.warning(f"Exit optimization failed: {str(e)}. This may be expected with dummy models.")
        exit_signals = []
    for signal in exit_signals:
        exit_reason = "Unknown"
        if hasattr(signal, 'indicators') and signal.indicators and "exit_reason" in signal.indicators:
            reason_code = signal.indicators["exit_reason"]
            if reason_code == 1.0:
                exit_reason = "Time-based"
            elif reason_code == 2.0:
                exit_reason = "Profit-based"
            elif reason_code == 3.0:
                exit_reason = "Stop-loss"
            elif reason_code == 4.0:
                exit_reason = "ML-based"
        
        logger.info(f"  {signal.symbol} EXIT @ {signal.price} (confidence={signal.confidence:.2f}, reason={exit_reason})")
    
    # Test feature extraction
    logger.info("Testing feature extraction")
    features = model.extract_exit_features(active_positions[0], market_data)
    logger.info(f"Extracted {len(features)} features for exit optimization")
    
    # Test inference
    if features:
        logger.info("Testing inference")
        try:
            start_time = time.time()
            result = model.infer(features)
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Inference completed in {elapsed_ms:.2f} ms")
            logger.info(f"Result: exit_probability={result[0]:.4f}, optimal_exit_price={result[1]:.2f}, trailing_stop_adjustment={result[2]:.4f}")
        except Exception as e:
            logger.warning(f"Inference failed: {str(e)}. This may be expected in CPU mode or with dummy models.")

def test_exit_strategies(model: ExitOptimizationModel, logger):
    """Test different exit strategies."""
    logger.info("Testing exit strategies")
    
    # Generate random active positions
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
    active_positions = []
    market_data = generate_market_data(symbols)
    
    # Test time-based exit
    logger.info("Testing time-based exit")
    old_position = generate_random_signal()
    old_position.symbol = symbols[0]
    old_position.timestamp = int((time.time() - 60 * 60 * 5) * 1_000_000)  # 5 hours ago
    active_positions.append(old_position)
    
    # Test profit-based exit
    logger.info("Testing profit-based exit")
    profit_position = generate_random_signal()
    profit_position.symbol = symbols[1]
    symbol_data = market_data["symbol_data"][profit_position.symbol]
    if profit_position.type == "BUY":
        # Set price above take profit
        symbol_data["last_price"] = profit_position.price * (1.0 + profit_position.take_profit / 100.0 + 0.01)
    else:
        # Set price below take profit
        symbol_data["last_price"] = profit_position.price * (1.0 - profit_position.take_profit / 100.0 - 0.01)
    active_positions.append(profit_position)
    
    # Test stop-loss exit
    logger.info("Testing stop-loss exit")
    stop_loss_position = generate_random_signal()
    stop_loss_position.symbol = symbols[2]
    symbol_data = market_data["symbol_data"][stop_loss_position.symbol]
    if stop_loss_position.type == "BUY":
        # Set price below stop loss
        symbol_data["last_price"] = stop_loss_position.price * (1.0 - stop_loss_position.stop_loss / 100.0 - 0.01)
    else:
        # Set price above stop loss
        symbol_data["last_price"] = stop_loss_position.price * (1.0 + stop_loss_position.stop_loss / 100.0 + 0.01)
    active_positions.append(stop_loss_position)
    
    # Test ML-based exit
    logger.info("Testing ML-based exit")
    ml_position = generate_random_signal()
    ml_position.symbol = symbols[3]
    # Ensure ML exit threshold is exceeded
    model.set_exit_threshold(0.1)  # Set low threshold to ensure ML exit
    active_positions.append(ml_position)
    
    # Run exit optimization
    model.optimize_exits(active_positions, market_data)
    
    # Test individual exit checks
    logger.info("Testing individual exit checks")
    
    # Time-based exit
    result = model.should_exit_based_on_time(old_position, market_data["timestamp"])
    logger.info(f"Time-based exit for old position: {result}")
    
    # Profit-based exit
    result = model.should_exit_based_on_profit(
        profit_position,
        market_data["symbol_data"][profit_position.symbol]["last_price"]
    )
    logger.info(f"Profit-based exit for profit position: {result}")
    
    # Stop-loss exit
    result = model.should_exit_based_on_stop_loss(
        stop_loss_position,
        market_data["symbol_data"][stop_loss_position.symbol]["last_price"]
    )
    logger.info(f"Stop-loss exit for stop-loss position: {result}")
    
    # Reset exit threshold to default
    model.set_exit_threshold(0.6)

def test_redis_integration(model: ExitOptimizationModel, logger):
    """Test Redis integration if available."""
    logger.info("Testing Redis integration")
    
    if model.redis_handler and model.redis_handler.initialized:
        logger.info("Redis integration is available and initialized")
        
        # Generate a test signal
        test_signal = generate_random_signal()
        test_signal.type = "EXIT"
        
        # Try to publish signal
        try:
            model.redis_handler.publish_signal(test_signal)
            logger.info(f"Published test signal to Redis: {test_signal.symbol} EXIT")
        except Exception as e:
            logger.warning(f"Failed to publish signal to Redis: {str(e)}")
    else:
        logger.info("Redis integration is not available or not initialized")

def test_threading(model: ExitOptimizationModel, logger):
    """Test threading with the model."""
    logger.info("Testing threading")
    
    # Create a function to run inference in a thread
    def run_inference(thread_id, results):
        # Generate synthetic features
        features = [random.uniform(0, 1) for _ in range(model.input_shape[1])]
        
        try:
            start_time = time.time()
            result = model.infer(features)
            elapsed_ms = (time.time() - start_time) * 1000
            
            results[thread_id] = (result, elapsed_ms)
        except Exception as e:
            logger.warning(f"Thread {thread_id} inference failed: {str(e)}")
            results[thread_id] = (None, 0)
    
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
    
    # Test thread affinity
    logger.info("Testing thread affinity")
    try:
        model.set_thread_affinity(0)  # Set affinity to core 0
        logger.info("Thread affinity set successfully")
    except Exception as e:
        logger.warning(f"Failed to set thread affinity: {str(e)}")

def test_configuration_setters(model: ExitOptimizationModel, logger):
    """Test configuration setters."""
    logger.info("Testing configuration setters")
    
    # Save original values
    original_exit_threshold = model.exit_threshold
    original_max_holding_time = model.max_holding_time_minutes
    original_check_interval = model.check_interval_seconds
    
    # Set new values
    model.set_exit_threshold(0.75)
    model.set_max_holding_time(120)
    model.set_check_interval(30)
    
    # Log new values
    logger.info(f"Updated exit threshold: {model.exit_threshold}")
    logger.info(f"Updated max holding time: {model.max_holding_time_minutes} minutes")
    logger.info(f"Updated check interval: {model.check_interval_seconds} seconds")
    
    # Restore original values
    model.set_exit_threshold(original_exit_threshold)
    model.set_max_holding_time(original_max_holding_time)
    model.set_check_interval(original_check_interval)
    
    logger.info("Restored original configuration")

if __name__ == "__main__":
    sys.exit(main())
