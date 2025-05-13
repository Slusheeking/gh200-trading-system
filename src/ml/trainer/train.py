#!/usr/bin/env python3
"""
Model Training Coordinator

This script coordinates the training of all models in the GH200 trading system.
It handles data preparation, model training, evaluation, and versioning.
"""

import os
import sys
import json
import argparse
import datetime
import time

# Import required GPU libraries - system is GPU-only
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except ImportError as e:
    raise RuntimeError(f"GPU-only mode requires CUDA and TensorRT: {str(e)}")

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
sys.path.insert(0, project_root)

from src.monitoring.log import logging as log # noqa: E402
from config.config_loader import get_config # noqa: E402
from src.ml.trainer.gbdt_trainer import GBDTTrainer # noqa: E402
from src.ml.trainer.axial_attention_trainer import AxialAttentionTrainer # noqa: E402
from src.ml.trainer.lstm_gru_trainer import LSTMGRUTrainer # noqa: E402
from src.ml.trainer.model_version_manager import ModelVersionManager # noqa: E402

# Set GPU and TensorRT flags - always true in GPU-only mode
HAS_GPU = True
HAS_TENSORRT = True

def initialize_production_environment(config):
    """
    Initialize production-identical environment for training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if initialization was successful, False otherwise
    """
    logger = log.setup_logger("production_env_init")
    
    try:
        # Initialize TensorRT for GPU-only operation
        logger.info("Initializing GPU-only environment with TensorRT")
        
        # Log GPU and TensorRT availability
        logger.info("CUDA is available for GPU acceleration")
        logger.info(f"TensorRT version {trt.__version__} is available for acceleration")
        
        # Check available GPU memory
        free_mem, total_mem = cuda.mem_get_info()
        free_mem_gb = free_mem / (1024 ** 3)
        total_mem_gb = total_mem / (1024 ** 3)
        logger.info(f"GPU memory: {free_mem_gb:.2f} GB free / {total_mem_gb:.2f} GB total")
        
        # Verify minimum GPU memory requirements (8GB)
        if total_mem_gb < 8:
            raise RuntimeError(f"Insufficient GPU memory: {total_mem_gb:.2f}GB available, minimum 8GB required")
            
        # Create TensorRT cache directory if it doesn't exist
        tensorrt_cache_path = config.get("export", {}).get("tensorrt_cache_path", "models/trt_cache")
        os.makedirs(tensorrt_cache_path, exist_ok=True)
        logger.info(f"TensorRT cache directory: {tensorrt_cache_path}")
        
        # Always enable TensorRT in GPU-only mode
        config["use_tensorrt"] = True
        
        # Log TensorRT configuration
        tensorrt_config = config.get("tensorrt", {})
        logger.info(f"TensorRT configuration: {tensorrt_config}")
        
        # Create cache directory if it doesn't exist
        cache_path = config.get("data_sources", {}).get("cache_path", "data/cache")
        os.makedirs(cache_path, exist_ok=True)
        logger.info(f"Market data cache directory: {cache_path}")
        
        # Log market data processing configuration
        market_data_config = config.get("data_sources", {})
        logger.info(f"Market data processing configuration: {market_data_config}")
        
        logger.info("GPU-only production environment initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing GPU-only production environment: {str(e)}")
        # In GPU-only mode, we don't fall back to CPU - we fail
        raise RuntimeError(f"GPU-only mode requires CUDA and TensorRT: {str(e)}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train trading system models")
    
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Models to train (comma-separated, or 'all'): gbdt,axial_attention,lstm_gru",
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for training data (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for training data (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Symbols to include in training (comma-separated)",
    )
    
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version to assign to trained models (default: auto-generated)",
    )
    
    parser.add_argument(
        "--no-active",
        action="store_true",
        help="Don't set trained models as active",
    )
    
    parser.add_argument(
        "--use-all-symbols",
        action="store_true",
        help="Use all available symbols (default: use symbols from config)",
    )
    
    parser.add_argument(
        "--no-tensorrt",
        action="store_true",
        help="Disable TensorRT acceleration",
    )
    
    parser.add_argument(
        "--measure-latency",
        action="store_true",
        help="Measure processing latency for each snapshot",
    )
    
    parser.add_argument(
        "--force-active",
        action="store_true",
        help="Force setting models as active even if there's an issue",
    )
    
    parser.add_argument(
        "--ensure-unique",
        action="store_true",
        help="Ensure version is unique by adding microseconds",
    )
    
    return parser.parse_args()


def get_date_range(args, config):
    """Get date range for training data"""
    # If dates are provided, use them
    if args.start_date and args.end_date:
        return args.start_date, args.end_date
    
    # Otherwise, use default date range from config or last 90 days
    ml_config = config.get("ml", {})
    training_config = ml_config.get("training", {})
    
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    days_back = training_config.get("default_training_days", 90)
    
    start_date = (
        datetime.datetime.now() - datetime.timedelta(days=days_back)
    ).strftime("%Y-%m-%d")
    
    return start_date, end_date


def get_symbols(args, config):
    """Get symbols for training data"""
    # If symbols are provided, use them
    if args.symbols:
        return args.symbols.split(",")
    
    # Otherwise, use default symbols from config
    ml_config = config.get("ml", {})
    training_config = ml_config.get("training", {})
    
    return training_config.get("default_symbols", None)


def get_version(args):
    """Get version for trained models"""
    # If version is provided, use it
    if args.version:
        # Validate that the provided version is in semver format
        try:
            import semver
            semver.parse(args.version)
            return args.version
        except (ImportError, ValueError):
            # If semver validation fails, convert to semver format
            pass
    
    # Generate a version based on current date and time in semver format (MAJOR.MINOR.PATCH)
    now = datetime.datetime.now()
    # Use year as major version, month+day as minor, hour+minute+second as patch
    # Adding seconds ensures uniqueness even for rapid consecutive runs
    major = now.year
    minor = now.month * 100 + now.day  # Combine month and day (e.g., Jan 15 = 115)
    patch = now.hour * 10000 + now.minute * 100 + now.second  # Include seconds for uniqueness
    
    version = f"{major}.{minor}.{patch}"
    
    # Add microseconds to ensure absolute uniqueness if needed
    if args and getattr(args, 'ensure_unique', False):
        version = f"{version}.{now.microsecond}"
    
    return version


def train_gbdt_model(config, start_date, end_date, symbols, version, set_as_active):
    """Train GBDT model with production-identical processing"""
    logger = log.setup_logger("train_coordinator")
    logger.info("Training GBDT model with production-identical processing")
    
    # Record start time
    start_time = time.time()
    
    # Initialize trainer
    trainer = GBDTTrainer(config)
    
    # Log processing status
    logger.info("Using production-identical market data processing for training")
    
    # Train and evaluate model
    results = trainer.train_and_evaluate(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        version=version,
        set_as_active=set_as_active,
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    results["training_time_seconds"] = training_time
    
    # Log results
    logger.info(f"GBDT model training completed with version {version} in {training_time:.2f} seconds")
    logger.info(f"Performance metrics: {results['performance']['evaluation']}")
    
    # Validate symbol coverage if using all symbols
    if symbols is None and "symbol_coverage" in results:
        coverage = results.get("symbol_coverage", {}).get("production_coverage_pct", 0)
        logger.info(f"Symbol coverage: {coverage:.1f}% of production symbols covered in training")
    
    return results


def train_axial_attention_model(config, start_date, end_date, symbols, version, set_as_active):
    """Train Axial Attention model with production-identical processing"""
    logger = log.setup_logger("train_coordinator")
    logger.info("Training Axial Attention model with production-identical processing")
    
    # Record start time
    start_time = time.time()
    
    # Initialize trainer
    trainer = AxialAttentionTrainer(config)
    
    # Log processing status
    logger.info("Using production-identical market data processing for training")
    
    # Train and evaluate model
    results = trainer.train_and_evaluate(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        version=version,
        set_as_active=set_as_active,
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    results["training_time_seconds"] = training_time
    
    # Log results
    logger.info(f"Axial Attention model training completed with version {version} in {training_time:.2f} seconds")
    logger.info(f"Performance metrics: {results['performance']['evaluation']}")
    
    # Validate symbol coverage if using all symbols
    if symbols is None and "symbol_coverage" in results:
        coverage = results.get("symbol_coverage", {}).get("production_coverage_pct", 0)
        logger.info(f"Symbol coverage: {coverage:.1f}% of production symbols covered in training")
    
    return results


def train_lstm_gru_model(config, start_date, end_date, symbols, version, set_as_active):
    """Train LSTM/GRU model with production-identical processing"""
    logger = log.setup_logger("train_coordinator")
    logger.info("Training LSTM/GRU model with production-identical processing")
    
    # Record start time
    start_time = time.time()
    
    # Initialize trainer
    trainer = LSTMGRUTrainer(config)
    
    # Log processing status
    logger.info("Using production-identical market data processing for training")
    
    # Train and evaluate model
    results = trainer.train_and_evaluate(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        version=version,
        set_as_active=set_as_active,
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    results["training_time_seconds"] = training_time
    
    # Log results
    logger.info(f"LSTM/GRU model training completed with version {version} in {training_time:.2f} seconds")
    logger.info(f"Performance metrics: {results['performance']['evaluation']}")
    
    # Validate symbol coverage if using all symbols
    if symbols is None and "symbol_coverage" in results:
        coverage = results.get("symbol_coverage", {}).get("production_coverage_pct", 0)
        logger.info(f"Symbol coverage: {coverage:.1f}% of production symbols covered in training")
    
    return results


def save_training_summary(results, version):
    """Save training summary to file with enhanced information"""
    logger = log.setup_logger("train_coordinator")
    
    # Create summary directory
    summary_dir = os.path.join("models", "training_summaries")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Add system information to results - GPU-only mode
    results["system_info"] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "gpu_available": True,  # Always true in GPU-only mode
        "gpu_enabled": True,    # Always true in GPU-only mode
        "tensorrt_available": True,  # Always true in GPU-only mode
        "tensorrt_enabled": True,    # Always true in GPU-only mode
        "tensorrt_version": trt.__version__
    }
    
    # Add GPU hardware information
    try:
        free_mem, total_mem = cuda.mem_get_info()
        results["system_info"]["gpu_memory"] = {
            "free_gb": free_mem / (1024 ** 3),
            "total_gb": total_mem / (1024 ** 3)
        }
    except Exception as e:
        logger.error(f"Error getting GPU information in GPU-only mode: {str(e)}")
            
    # Add latency metrics summary if available
    for model_type, model_results in results.get("models", {}).items():
        if "performance" in model_results and "latency_metrics" in model_results.get("performance", {}):
            latency_metrics = model_results["performance"]["latency_metrics"]
            logger.info(f"Latency metrics for {model_type}: "
                      f"avg={latency_metrics.get('avg_snapshot_latency_ms', 0):.2f}ms")
    
    # Generate summary file path
    summary_path = os.path.join(summary_dir, f"training_summary_{version}.json")
    
    # Save summary
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Enhanced training summary saved to {summary_path}")


def main():
    """Main function"""
    # Set up logging
    logger = log.setup_logger("train_coordinator")
    logger.info("Starting model training coordinator with production-identical processing")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = get_config()
    
    # In GPU-only mode, TensorRT is always enabled
    config["use_tensorrt"] = True
    logger.info("TensorRT acceleration enabled in GPU-only mode (--no-tensorrt flag ignored)")
    
    # Configure latency measurement
    if args.measure_latency:
        config["measure_latency"] = True
        logger.info("Latency measurement enabled via command line")
    else:
        config["measure_latency"] = False
    
    # Initialize production environment - will raise an error if GPU is not available
    initialize_production_environment(config)
    
    # Pass latency measurement configuration to training market processor
    if config.get("measure_latency", False):
        if "ml" not in config:
            config["ml"] = {}
        if "training" not in config["ml"]:
            config["ml"]["training"] = {}
        config["ml"]["training"]["measure_latency"] = True
        logger.info("Configured training market processor to measure latency")
    
    # Get date range
    start_date, end_date = get_date_range(args, config)
    logger.info(f"Training period: {start_date} to {end_date}")
    
    # Get symbols
    if args.use_all_symbols:
        logger.info("Using all available symbols for training")
        symbols = None  # Let the trainer fetch all symbols
    else:
        symbols = get_symbols(args, config)
        if symbols:
            logger.info(f"Training with symbols: {', '.join(symbols[:10])}... (total: {len(symbols)})")
        else:
            logger.info("Training with default symbols")
    
    # Get version - always generate a new version based on current timestamp
    # This ensures each training run gets a unique version
    version = get_version(args)
    logger.info(f"Using version: {version}")
    
    # Determine which models to train
    models_to_train = args.models.lower().split(",") if args.models != "all" else ["gbdt", "axial_attention", "lstm_gru"]
    logger.info(f"Models to train: {', '.join(models_to_train)}")
    
    # Set active flag - default to True to ensure models are set as active
    set_as_active = not args.no_active or args.force_active
    if args.force_active:
        logger.info("Force active flag set - models will be set as active regardless of no-active flag")
    logger.info(f"Set as active: {set_as_active}")
    
    # Verify the model will be saved and set as active
    if set_as_active:
        logger.info(f"Models will be saved with version {version} and set as active")
    else:
        logger.warning("Models will be saved but NOT set as active - use with caution")
    
    # Initialize results dictionary - GPU-only mode
    results = {
        "version": version,
        "training_period": {
            "start_date": start_date,
            "end_date": end_date
        },
        "gpu_enabled": True,  # Always true in GPU-only mode
        "tensorrt_enabled": True,  # Always true in GPU-only mode
        "measure_latency": config.get("measure_latency", False),
        "models": {}
    }
    
    # Train models
    try:
        # Train GBDT model
        if "gbdt" in models_to_train:
            logger.info("Starting GBDT model training")
            gbdt_results = train_gbdt_model(
                config, start_date, end_date, symbols, version, set_as_active
            )
            results["models"]["gbdt"] = gbdt_results
            logger.info("GBDT model training completed")
        
        # Train Axial Attention model
        if "axial_attention" in models_to_train:
            logger.info("Starting Axial Attention model training")
            axial_results = train_axial_attention_model(
                config, start_date, end_date, symbols, version, set_as_active
            )
            results["models"]["axial_attention"] = axial_results
            logger.info("Axial Attention model training completed")
        
        # Train LSTM/GRU model
        if "lstm_gru" in models_to_train:
            logger.info("Starting LSTM/GRU model training")
            lstm_results = train_lstm_gru_model(
                config, start_date, end_date, symbols, version, set_as_active
            )
            results["models"]["lstm_gru"] = lstm_results
            logger.info("LSTM/GRU model training completed")
        
        # Save training summary
        save_training_summary(results, version)
        
        # Verify models were saved and set as active
        if set_as_active:
            verify_models_active(models_to_train, version)
        
        logger.info("All model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

def verify_models_active(models_to_train, expected_version):
    """Verify that models were saved and set as active"""
    logger = log.setup_logger("train_coordinator")
    logger.info(f"Verifying models were saved and set as active with version {expected_version}")
    
    # Initialize version manager
    version_manager = ModelVersionManager()
    
    # Check each model
    all_active = True
    for model_type in models_to_train:
        active_version = version_manager.get_active_version(model_type)
        if active_version == expected_version:
            logger.info(f"✅ {model_type} model is active with correct version {active_version}")
        else:
            logger.error(f"❌ {model_type} model active version is {active_version}, expected {expected_version}")
            all_active = False
            
            # Try to fix it
            try:
                logger.info(f"Attempting to set {model_type} model version {expected_version} as active")
                version_manager.set_active_version(model_type, expected_version)
                logger.info(f"Successfully set {model_type} model version {expected_version} as active")
            except Exception as e:
                logger.error(f"Failed to set {model_type} model version {expected_version} as active: {str(e)}")
    
    if all_active:
        logger.info("All models are active with correct versions")
    else:
        logger.warning("Some models are not active with correct versions")


if __name__ == "__main__":
    main()