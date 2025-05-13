"""
Test script for the LSTM/GRU model

This script demonstrates how to use the LSTM/GRU model with the project's
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
from src.ml.lstm_gru_model import LstmGruModel  # noqa: E402

def main():
    """Main test function for the LSTM/GRU model."""
    # Set up logger
    logger = log.setup_logger("test_lstm_gru_model")
    logger.info("Starting LSTM/GRU model test")
    
    try:
        # Load configuration
        config = get_config()
        logger.info("Configuration loaded")
        
        # Create model
        model = LstmGruModel(config)
        logger.info(f"Created model: {model.get_name()}")
        
        # Use the model configuration from JSON - no fallbacks
        model_config_path = Path(project_root) / "models" / "lstm_gru" / "model_config.json"
        if not model_config_path.exists():
            raise FileNotFoundError(f"Model configuration file not found: {model_config_path}")
            
        logger.info(f"Loading model configuration from {model_config_path}")
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
            
        # Configure model with parameters from config
        if hasattr(model, 'set_num_layers'):
            model.set_num_layers(model_config.get("num_layers", 4))
        if hasattr(model, 'set_hidden_size'):
            model.set_hidden_size(model_config.get("hidden_size", 256))
        if hasattr(model, 'set_bidirectional'):
            model.set_bidirectional(model_config.get("bidirectional", True))
        if hasattr(model, 'set_attention_enabled'):
            model.set_attention_enabled(model_config.get("attention_enabled", True))
        
        # Check if we have a real model file
        model_path = model_config.get("model_path")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at path specified in config: {model_path}")
        
        # Load the model
        logger.info(f"Loading model from {model_path}")
        model.load(model_path)
        logger.info("Model loaded successfully")
        
        # Test with synthetic data
        test_with_synthetic_data(model, logger)
        
        # Test batch processing
        test_batch_processing(model, logger)
        
        # Test position state persistence
        test_position_state_persistence(model, logger)
        
        # Test model configuration
        test_model_configuration(model, logger)
        
        # Test threading
        test_threading(model, logger)
        
        logger.info("Test completed successfully")
    
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0

# Removed create_dummy_model function as we're only using actual models from configuration

def generate_synthetic_features() -> List[float]:
    """Generate synthetic features for testing."""
    # Generate 20 features (seq_length) * 10 (feature_dim) = 200 features
    features = []
    
    # Position ID (encoded as a feature)
    features.append(random.randint(1, 100))
    
    # Position duration (in minutes)
    features.append(random.uniform(1, 500))
    
    # Unrealized P&L (as a percentage)
    features.append(random.uniform(-0.1, 0.1))
    
    # Price momentum
    features.append(random.uniform(-0.05, 0.05))
    
    # Volatility
    features.append(random.uniform(0.01, 0.2))
    
    # Volume ratio
    features.append(random.uniform(0.5, 3.0))
    
    # Current price
    current_price = random.uniform(100, 1000)
    features.append(current_price)
    
    # Entry price
    entry_price = current_price * (1 + random.uniform(-0.1, 0.1))
    features.append(entry_price)
    
    # Fill the rest with random values
    for _ in range(192):  # 200 - 8 = 192
        features.append(random.uniform(-1, 1))
    
    return features

def test_with_synthetic_data(model: LstmGruModel, logger):
    """Test the model with synthetic data."""
    logger.info("Testing with synthetic data")
    
    # Generate synthetic features
    features = generate_synthetic_features()
    
    # Log features
    logger.info(f"Generated features with shape: {len(features)}")
    logger.info(f"Position ID: {int(features[0])}")
    logger.info(f"Position duration: {features[1]:.2f} minutes")
    logger.info(f"Unrealized P&L: {features[2]:.2%}")
    logger.info(f"Price momentum: {features[3]:.4f}")
    logger.info(f"Current price: {features[6]:.2f}")
    logger.info(f"Entry price: {features[7]:.2f}")
    
    # Test single inference
    logger.info("Testing single inference")
    start_time = time.time()
    result = model.infer(features)
    elapsed_ms = (time.time() - start_time) * 1000
    
    logger.info(f"Inference completed in {elapsed_ms:.2f} ms")
    logger.info(f"Result: {result}")
    logger.info(f"Exit probability: {result[0]:.2%}")
    logger.info(f"Optimal exit price: {result[1]:.2f}")
    logger.info(f"Trailing stop adjustment: {result[2]:.2f}")
    
    # Test input and output shapes
    input_shape = model.get_input_shape()
    output_shape = model.get_output_shape()
    
    logger.info(f"Model input shape: {input_shape}")
    logger.info(f"Model output shape: {output_shape}")

def test_batch_processing(model: LstmGruModel, logger):
    """Test batch processing with the model."""
    logger.info("Testing batch processing")
    
    # Test batch inference
    batch_sizes = [1, 5, 10, 20]
    
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size {batch_size}")
        features_batch = [generate_synthetic_features() for _ in range(batch_size)]
        
        try:
            start_time = time.time()
            batch_result = model.infer_batch(features_batch)
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Batch inference completed in {elapsed_ms:.2f} ms for {batch_size} samples")
            logger.info(f"Average time per sample: {elapsed_ms / batch_size:.2f} ms")
            logger.info(f"Batch results: {len(batch_result)} results")
            
            if batch_size > 0:
                logger.info(f"First result: exit_probability={batch_result[0][0]:.2f}, "
                          f"optimal_exit_price={batch_result[0][1]:.2f}, "
                          f"trailing_stop_adjustment={batch_result[0][2]:.2f}")
        except Exception as e:
            logger.warning(f"Batch inference failed for size {batch_size}: {str(e)}. Falling back to individual inference.")
            # Process each item individually as fallback
            batch_result = []
            start_time = time.time()
            for features in features_batch:
                result = model.infer(features)
                batch_result.append(result)
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Individual inference for {batch_size} items completed in {elapsed_ms:.2f} ms")
            logger.info(f"Average time per sample: {elapsed_ms / batch_size:.2f} ms")
            logger.info(f"Batch results: {len(batch_result)} results")
            
            if batch_size > 0:
                logger.info(f"First result: exit_probability={batch_result[0][0]:.2f}, "
                          f"optimal_exit_price={batch_result[0][1]:.2f}, "
                          f"trailing_stop_adjustment={batch_result[0][2]:.2f}")

def test_position_state_persistence(model: LstmGruModel, logger):
    """Test position state persistence."""
    logger.info("Testing position state persistence")
    
    # Create features with the same position ID
    position_id = random.randint(1, 100)
    features1 = generate_synthetic_features()
    features1[0] = position_id
    
    features2 = generate_synthetic_features()
    features2[0] = position_id
    
    # Run inference twice with the same position ID
    result1 = model.infer(features1)
    result2 = model.infer(features2)
    
    logger.info(f"Position ID: {position_id}")
    logger.info(f"First inference result: {result1}")
    logger.info(f"Second inference result: {result2}")
    logger.info(f"Position state maintained: {position_id in model.position_states}")
    
    # Check position state content
    if position_id in model.position_states:
        state = model.position_states[position_id]
        logger.info(f"Position state: {state[:3]}...")  # Show first 3 elements
    
    # Test with multiple positions
    logger.info("Testing with multiple positions")
    position_ids = [random.randint(1, 100) for _ in range(5)]
    
    for pos_id in position_ids:
        features = generate_synthetic_features()
        features[0] = pos_id
        model.infer(features)
    
    logger.info(f"Number of position states: {len(model.position_states)}")
    logger.info(f"Position IDs in state: {['pos_' + str(int(pid)) for pid in position_ids]}")

def test_model_configuration(model: LstmGruModel, logger):
    """Test model configuration."""
    logger.info("Testing model configuration")
    
    # Save original configuration
    original_num_layers = model.num_layers
    original_hidden_size = model.hidden_size
    original_bidirectional = model.bidirectional
    original_attention_enabled = model.attention_enabled
    
    # Change model parameters
    model.set_num_layers(4)
    model.set_hidden_size(256)
    model.set_bidirectional(False)
    model.set_attention_enabled(False)
    
    # Log new configuration
    logger.info("Updated configuration:")
    logger.info(f"  - Num layers: {model.num_layers}")
    logger.info(f"  - Hidden size: {model.hidden_size}")
    logger.info(f"  - Bidirectional: {model.bidirectional}")
    logger.info(f"  - Attention enabled: {model.attention_enabled}")
    
    # Reconfigure model
    model.configure_model()
    
    # Run inference with new configuration
    features = generate_synthetic_features()
    result = model.infer(features)
    
    logger.info(f"Result with new configuration: {result}")
    
    # Restore original configuration
    model.set_num_layers(original_num_layers)
    model.set_hidden_size(original_hidden_size)
    model.set_bidirectional(original_bidirectional)
    model.set_attention_enabled(original_attention_enabled)
    model.configure_model()
    
    logger.info("Restored original configuration")

def test_threading(model: LstmGruModel, logger):
    """Test threading with the model."""
    logger.info("Testing threading")
    
    # Create a function to run inference in a thread
    def run_inference(thread_id, results):
        features = generate_synthetic_features()
        
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

# Removed train_minimal_model and generate_synthetic_training_data functions
# as we're only using actual models from configuration

if __name__ == "__main__":
    sys.exit(main())