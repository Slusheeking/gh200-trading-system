"""
Base Model Trainer

This module provides a base class for all model trainers in the system.
It defines common functionality and interfaces that all trainers should implement.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.monitoring.log import logging as log
from config.config_loader import get_config
from src.ml.trainer.latency_profiler import LatencyProfiler
from src.ml.trainer.model_version_manager import ModelVersionManager
from src.ml.trainer.data_manager import DataManager
from src.ml.trainer.training_market_processor import TrainingMarketDataProcessor


class ModelTrainer(ABC):
    """
    Base class for all model trainers.

    This abstract class defines the common interface and functionality
    for all model trainers in the system. Specific model trainers should
    inherit from this class and implement the abstract methods.
    """

    def __init__(self, model_type: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model trainer.

        Args:
            model_type: Type of model (gbdt, axial_attention, lstm_gru)
            config: Configuration dictionary (optional)
        """
        self.model_type = model_type
        self.logger = log.setup_logger(f"{model_type}_trainer")
        self.logger.info(f"Initializing {model_type} trainer")

        # Load configuration
        if config is None:
            config = get_config()

        self.config = config

        # Extract model-specific configuration
        ml_config = config.get("ml", {})
        self.model_config = ml_config.get(self._get_config_key(), {})
        self.training_config = ml_config.get("training", {})
        self.export_config = ml_config.get("export", {})

        # Initialize components
        self.latency_profiler = LatencyProfiler(f"{model_type}_trainer")
        self.version_manager = ModelVersionManager(config)
        self.data_manager = DataManager(config)
        
        # Initialize market data processor for production-identical processing
        self.market_processor = TrainingMarketDataProcessor(config)
        self.logger.info("Initialized market data processor for production-identical processing")

        # Initialize model state
        self.model = None
        self.is_trained = False
        self.training_metadata = {}
        self.performance_metrics = {}

    def _get_config_key(self) -> str:
        """
        Get the configuration key for this model type.

        Returns:
            Configuration key
        """
        if self.model_type == "gbdt":
            return "fast_path"
        elif self.model_type == "axial_attention":
            return "accurate_path"
        elif self.model_type == "lstm_gru":
            return "exit_optimization"
        else:
            return self.model_type

    @abstractmethod
    def build_model(self) -> Any:
        """
        Build the model architecture.

        Returns:
            Model instance
        """
        pass

    @abstractmethod
    def train(
        self,
        train_data: Dict[str, Any],
        validation_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train the model on the provided data.

        Args:
            train_data: Training data
            validation_data: Validation data (optional)

        Returns:
            Training history/metrics
        """
        pass

    @abstractmethod
    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            test_data: Test data

        Returns:
            Evaluation metrics
        """
        pass

    @abstractmethod
    def save(self, version: str) -> str:
        """
        Save the trained model with the specified version.

        Args:
            version: Version string (e.g., "1.0.0")

        Returns:
            Path to saved model
        """
        pass

    @abstractmethod
    def load(self, version: Optional[str] = None) -> None:
        """
        Load a trained model.

        Args:
            version: Version to load (optional, defaults to active version)
        """
        pass

    def train_and_evaluate(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        version: Optional[str] = None,
        set_as_active: bool = True,
    ) -> Dict[str, Any]:
        """
        Train and evaluate the model on data from the specified date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: List of symbols to include (optional)
            version: Version to assign to the trained model (optional)
            set_as_active: Whether to set the trained model as active

        Returns:
            Dictionary with training and evaluation results
        """
        self.logger.info(
            f"Training and evaluating {self.model_type} model "
            f"on data from {start_date} to {end_date} using production-identical processing"
        )

        # Start timing
        self.latency_profiler.start_phase("train_and_evaluate")

        # Create training dataset using production-identical processing
        train_test_split = self.training_config.get("train_test_split", 0.8)
        random_seed = self.training_config.get("random_seed", 42)

        self.logger.info(
            f"Creating training dataset with production-identical processing and train/test split {train_test_split}"
        )
        
        # Use the new method that leverages the market data processor
        train_data, test_data = self.prepare_training_data(
            start_date,
            end_date,
            symbols,
            train_test_split
        )

        # Build model
        self.logger.info("Building model architecture")
        self.model = self.build_model()

        # Train model
        self.logger.info("Training model with production-identical features")
        self.latency_profiler.start_phase("training")
        training_history = self.train(train_data)
        training_time = self.latency_profiler.end_phase()

        self.is_trained = True

        # Evaluate model
        self.logger.info("Evaluating model with production-identical features")
        self.latency_profiler.start_phase("evaluation")
        evaluation_metrics = self.evaluate(test_data)
        evaluation_time = self.latency_profiler.end_phase()

        # Collect performance metrics
        self.performance_metrics = {
            "training": training_history,
            "evaluation": evaluation_metrics,
            "latency": {
                "training_time_ms": training_time,
                "evaluation_time_ms": evaluation_time,
            },
        }

        # Collect metadata
        self.training_metadata = {
            "model_type": self.model_type,
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_range": {"start_date": start_date, "end_date": end_date},
            "symbols": symbols,
            "train_test_split": train_test_split,
            "random_seed": random_seed,
            "model_parameters": self._get_model_parameters(),
            "training_parameters": self._get_training_parameters(),
        }

        # Save model if version is provided
        if version is not None:
            self.logger.info(f"Saving model as version {version}")
            # Save the model first
            try:
                model_dir = self.save(version)
                self.logger.info(f"Successfully saved model to {model_dir}")
                
                # Limit the number of model versions to keep only the top 10
                self.logger.info(f"Limiting {self.model_type} models to top 10 versions")
                self.version_manager.limit_model_versions(self.model_type, 10)
                
                # Automatically promote the best model
                if set_as_active:
                    self.logger.info(f"Promoting best {self.model_type} model to active version")
                    best_version = self.version_manager.promote_best_model(self.model_type)
                    if best_version:
                        self.logger.info(f"Promoted {self.model_type} model v{best_version} to active version")
                    else:
                        self.logger.warning("Could not promote best model, setting current version as active")
                        self.version_manager.set_active_version(self.model_type, version)
                
                # Clean up training data to keep the system clean
                self.logger.info("Cleaning up training data")
                self.version_manager.clean_training_data()
                
            except Exception as e:
                self.logger.error(f"Error saving model: {str(e)}")
                raise

        # End timing
        self.latency_profiler.end_phase()

        # Create result dictionary
        result = {
            "metadata": self.training_metadata,
            "performance": self.performance_metrics,
            "latency_profile": self.latency_profiler.create_latency_profile(),
        }

        self.logger.info(
            f"Completed training and evaluation of {self.model_type} model"
        )

        return result
        
    def prepare_training_data(self, start_date: str, end_date: str,
                             symbols: Optional[List[str]] = None,
                             train_test_split: float = 0.8) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Prepare training data using production-identical processing
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            symbols: Optional list of symbols to include
            train_test_split: Ratio for train/test split
            
        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        self.logger.info(f"Preparing training data for {self.model_type} with production-identical processing")
        
        # Get raw market snapshots from data manager
        snapshots = self.data_manager.fetch_market_snapshots_for_period(
            start_date, end_date, symbols)
        
        # Process snapshots using production-identical processor
        processed_snapshots = self.market_processor.process_historical_snapshots(snapshots)
        
        # Split into training and validation sets
        split_idx = int(len(processed_snapshots) * train_test_split)
        
        # Process snapshots and extract features for this model type
        train_data = self.market_processor.batch_process_snapshots(
            processed_snapshots[:split_idx], self.model_type)
        
        test_data = self.market_processor.batch_process_snapshots(
            processed_snapshots[split_idx:], self.model_type)
        
        # Add raw snapshots for reference
        train_data["snapshots"] = processed_snapshots[:split_idx]
        test_data["snapshots"] = processed_snapshots[split_idx:]
        
        self.logger.info(f"Prepared training data with {len(train_data['X'])} training samples "
                       f"and {len(test_data['X'])} testing samples")
        
        return train_data, test_data

    def _get_model_parameters(self) -> Dict[str, Any]:
        """
        Get model parameters for metadata.

        Returns:
            Dictionary of model parameters
        """
        # Default implementation returns model_config
        return self.model_config

    def _get_training_parameters(self) -> Dict[str, Any]:
        """
        Get training parameters for metadata.

        Returns:
            Dictionary of training parameters
        """
        # Default implementation returns training_config
        return self.training_config

    def benchmark_inference(
        self, num_samples: int = 1000, batch_sizes: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark inference performance.

        Args:
            num_samples: Number of samples to use for benchmarking
            batch_sizes: List of batch sizes to benchmark (optional)

        Returns:
            Dictionary with benchmark results
        """
        self.logger.info(
            f"Benchmarking inference performance with {num_samples} samples"
        )

        if not self.is_trained:
            self.logger.warning("Model is not trained, loading active version")
            self.load()

        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32, 64, 128]

        # Start timing
        self.latency_profiler.start_phase("benchmark_inference")

        # Generate synthetic data for benchmarking
        benchmark_data = self._generate_benchmark_data(num_samples)

        # Benchmark each batch size
        results = {}

        for batch_size in batch_sizes:
            self.logger.info(f"Benchmarking batch size {batch_size}")

            # Prepare batches
            num_batches = (num_samples + batch_size - 1) // batch_size
            batches = []

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batches.append(benchmark_data[start_idx:end_idx])

            # Warm up
            self._benchmark_inference_batch(batches[0])

            # Benchmark
            self.latency_profiler.start_timer(f"batch_size_{batch_size}")

            batch_latencies = []
            for batch in batches:
                start_time = time.time()
                self._benchmark_inference_batch(batch)
                elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
                batch_latencies.append(elapsed_time)

            total_time = self.latency_profiler.stop_timer(f"batch_size_{batch_size}")

            # Calculate statistics
            batch_latencies = np.array(batch_latencies)

            results[str(batch_size)] = {
                "total_time_ms": total_time,
                "avg_batch_time_ms": float(np.mean(batch_latencies)),
                "p50_ms": float(np.percentile(batch_latencies, 50)),
                "p95_ms": float(np.percentile(batch_latencies, 95)),
                "p99_ms": float(np.percentile(batch_latencies, 99)),
                "min_ms": float(np.min(batch_latencies)),
                "max_ms": float(np.max(batch_latencies)),
                "throughput_samples_per_sec": num_samples / (total_time / 1000),
            }

        # End timing
        self.latency_profiler.end_phase()

        # Create benchmark report
        benchmark_report = {
            "model_type": self.model_type,
            "num_samples": num_samples,
            "batch_sizes": batch_sizes,
            "results": results,
            "optimal_batch_size": self._find_optimal_batch_size(results),
        }

        self.logger.info(
            f"Completed inference benchmarking for {self.model_type} model"
        )

        return benchmark_report

    @abstractmethod
    def _generate_benchmark_data(self, num_samples: int) -> List[Any]:
        """
        Generate synthetic data for benchmarking.

        Args:
            num_samples: Number of samples to generate

        Returns:
            List of benchmark data samples
        """
        pass

    @abstractmethod
    def _benchmark_inference_batch(self, batch: List[Any]) -> List[Any]:
        """
        Run inference on a batch of data for benchmarking.

        Args:
            batch: Batch of data

        Returns:
            Inference results
        """
        pass

    def _find_optimal_batch_size(self, results: Dict[str, Dict[str, float]]) -> int:
        """
        Find the optimal batch size based on throughput.

        Args:
            results: Benchmark results

        Returns:
            Optimal batch size
        """
        max_throughput = 0
        optimal_batch_size = 1

        for batch_size, metrics in results.items():
            throughput = metrics["throughput_samples_per_sec"]
            if throughput > max_throughput:
                max_throughput = throughput
                optimal_batch_size = int(batch_size)

        return optimal_batch_size
