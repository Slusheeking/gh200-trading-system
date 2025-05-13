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

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
sys.path.insert(0, project_root)

from src.monitoring.log import logging as log
from config.config_loader import get_config
from src.ml.trainer.gbdt_trainer import GBDTTrainer
from config.config_loader import get_config
from src.ml.trainer.gbdt_trainer import GBDTTrainer
from src.ml.trainer.axial_attention_trainer import AxialAttentionTrainer
from src.ml.trainer.lstm_gru_trainer import LSTMGRUTrainer
from src.ml.trainer.model_version_manager import ModelVersionManager


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
    # Use year as major version, month+day as minor, hour+minute as patch
    major = now.year
    minor = now.month * 100 + now.day  # Combine month and day (e.g., Jan 15 = 115)
    patch = now.hour * 100 + now.minute  # Combine hour and minute
    
    return f"{major}.{minor}.{patch}"


def train_gbdt_model(config, start_date, end_date, symbols, version, set_as_active):
    """Train GBDT model"""
    logger = log.setup_logger("train_coordinator")
    logger.info("Training GBDT model")
    
    # Initialize trainer
    trainer = GBDTTrainer(config)
    
    # Train and evaluate model
    results = trainer.train_and_evaluate(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        version=version,
        set_as_active=set_as_active,
    )
    
    # Log results
    logger.info(f"GBDT model training completed with version {version}")
    logger.info(f"Performance metrics: {results['performance']['evaluation']}")
    
    return results


def train_axial_attention_model(config, start_date, end_date, symbols, version, set_as_active):
    """Train Axial Attention model"""
    logger = log.setup_logger("train_coordinator")
    logger.info("Training Axial Attention model")
    
    # Initialize trainer
    trainer = AxialAttentionTrainer(config)
    
    # Train and evaluate model
    results = trainer.train_and_evaluate(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        version=version,
        set_as_active=set_as_active,
    )
    
    # Log results
    logger.info(f"Axial Attention model training completed with version {version}")
    logger.info(f"Performance metrics: {results['performance']['evaluation']}")
    
    return results


def train_lstm_gru_model(config, start_date, end_date, symbols, version, set_as_active):
    """Train LSTM/GRU model"""
    logger = log.setup_logger("train_coordinator")
    logger.info("Training LSTM/GRU model")
    
    # Initialize trainer
    trainer = LSTMGRUTrainer(config)
    
    # Train and evaluate model
    results = trainer.train_and_evaluate(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        version=version,
        set_as_active=set_as_active,
    )
    
    # Log results
    logger.info(f"LSTM/GRU model training completed with version {version}")
    logger.info(f"Performance metrics: {results['performance']['evaluation']}")
    
    return results


def save_training_summary(results, version):
    """Save training summary to file"""
    logger = log.setup_logger("train_coordinator")
    
    # Create summary directory
    summary_dir = os.path.join("models", "training_summaries")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Generate summary file path
    summary_path = os.path.join(summary_dir, f"training_summary_{version}.json")
    
    # Save summary
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Training summary saved to {summary_path}")


def main():
    """Main function"""
    # Set up logging
    logger = log.setup_logger("train_coordinator")
    logger.info("Starting model training coordinator")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = get_config()
    
    # Get date range
    start_date, end_date = get_date_range(args, config)
    logger.info(f"Training period: {start_date} to {end_date}")
    
    # Get symbols
    symbols = get_symbols(args, config)
    if symbols:
        logger.info(f"Training with symbols: {', '.join(symbols)}")
    else:
        logger.info("Training with default symbols")
    
    # Get version
    version = get_version(args)
    logger.info(f"Using version: {version}")
    
    # Determine which models to train
    models_to_train = args.models.lower().split(",") if args.models != "all" else ["gbdt", "axial_attention", "lstm_gru"]
    logger.info(f"Models to train: {', '.join(models_to_train)}")
    
    # Set active flag
    set_as_active = not args.no_active
    logger.info(f"Set as active: {set_as_active}")
    
    # Initialize results dictionary
    results = {
        "version": version,
        "training_period": {
            "start_date": start_date,
            "end_date": end_date
        },
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
        
        logger.info("All model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()