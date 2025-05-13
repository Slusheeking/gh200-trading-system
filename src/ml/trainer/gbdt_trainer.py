"""
GBDT Trainer Implementation

This module provides a trainer for Gradient Boosted Decision Trees models
using LightGBM as the backend. It handles training, evaluation, and model
management for the fast path in the trading system.
"""

import numpy as np
import os
import json
import logging
import importlib.util
from typing import Dict, List, Any, Optional
from src.ml.trainer.base_trainer import ModelTrainer

# Try to import LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, ValueError, AttributeError) as e:
    HAS_LIGHTGBM = False
    logging.warning(f"LightGBM not available or incompatible: {str(e)}. GBDT training will not function.")

# Check for TensorRT and PyCUDA availability using importlib
HAS_TENSORRT = False
if (importlib.util.find_spec("tensorrt") is not None and
    importlib.util.find_spec("pycuda") is not None):
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        # Initialize CUDA by accessing the module rather than importing it directly
        cuda.init()  # Initialize CUDA driver
        HAS_TENSORRT = True
    except (ImportError, ValueError, AttributeError) as e:
        logging.warning(f"TensorRT not available or incompatible: {str(e)}. Falling back to CPU inference.")


class GBDTTrainer(ModelTrainer):
    """
    Trainer for Gradient Boosted Decision Trees models.
    
    This trainer uses LightGBM to train GBDT models for the fast path
    in the trading system. It handles data preparation, training,
    evaluation, and model management.
    """
    
    def __init__(self, config=None):
        """
        Initialize the GBDT trainer.
        
        Args:
            config: Configuration dictionary (optional)
        """
        super().__init__("gbdt", config)
        
        # Check if LightGBM is available
        if not HAS_LIGHTGBM:
            self.logger.error("LightGBM is not available. GBDT training will not function.")
            raise ImportError("LightGBM is required for GBDT training")
        
        # Initialize model parameters from config
        self.num_trees = self.model_config.get("num_trees", 150)
        self.max_depth = self.model_config.get("max_depth", 8)
        self.learning_rate = self.model_config.get("learning_rate", 0.05)
        self.feature_fraction = self.model_config.get("feature_fraction", 0.8)
        self.bagging_fraction = self.model_config.get("bagging_fraction", 0.7)
        self.num_leaves = self.model_config.get("num_leaves", 31)
        self.early_stopping_rounds = self.model_config.get("early_stopping_rounds", 10)
        self.num_boost_round = self.model_config.get("num_boost_round", 100)
        self.bagging_freq = self.model_config.get("bagging_freq", 5)
        self.objective = self.model_config.get("objective", "binary")
        self.metric = self.model_config.get("metric", "auc")
        
        # Initialize feature names
        self.feature_names = []
        
        # TensorRT configuration
        self.use_tensorrt = self.training_config.get("use_tensorrt", True) and HAS_TENSORRT
        self.tensorrt_fp16 = self.training_config.get("export", {}).get("tensorrt_fp16", True)
        self.tensorrt_cache_path = self.training_config.get("export", {}).get("tensorrt_cache_path", "models/trt_cache")
        
        # Create TensorRT cache directory if it doesn't exist
        if self.use_tensorrt:
            os.makedirs(self.tensorrt_cache_path, exist_ok=True)
            
        # TensorRT engine and context
        self.trt_engine = None
        self.trt_context = None
        self.trt_input_shape = None
        self.trt_output_shape = None
        
        self.logger.info(f"Initialized GBDT trainer with parameters: "
                       f"trees={self.num_trees}, "
                       f"depth={self.max_depth}, "
                       f"learning_rate={self.learning_rate}, "
                       f"use_tensorrt={self.use_tensorrt}")
    
    def train_and_evaluate(self, start_date: str, end_date: str,
                         symbols: Optional[List[str]] = None,
                         version: Optional[str] = None,
                         set_as_active: bool = True) -> Dict[str, Any]:
        """
        Train and evaluate the model with enhanced symbol verification.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            symbols: List of symbols to include (optional)
            version: Version to assign to trained model (optional)
            set_as_active: Whether to set the trained model as active
            
        Returns:
            Dictionary with training and evaluation results
        """
        self.logger.info(f"Training and evaluating GBDT model from {start_date} to {end_date}")
        
        # Start timing
        self.latency_profiler.start_phase("train_and_evaluate")
        
        # Get training configuration
        train_test_split = self.training_config.get("train_test_split", 0.8)
        random_seed = self.training_config.get("random_seed", 42)
        
        # Get production symbols for comparison prior to training
        production_symbols = self._get_production_symbols()
        
        # Create training dataset
        train_data, test_data = self.data_manager.create_training_dataset(
            self.model_type,
            start_date,
            end_date,
            symbols,
            train_test_split,
            random_seed
        )
        
        # Get unique symbols from training data
        training_symbols = []
        if self.model_type == "gbdt" and "sample_symbols" in train_data:
            training_symbols = list(set(train_data["sample_symbols"]))
        
        # Log the number of unique symbols used for training
        self.logger.info(f"Number of unique symbols used for training: {len(training_symbols)}")
        # Validate training vs. production symbol coverage
        if production_symbols and training_symbols:
            coverage_metrics = self.data_manager.validate_symbol_coverage(
                training_symbols, production_symbols)
                
            # Add metrics to training metadata
            self.training_metadata["symbol_coverage"] = coverage_metrics
            
            # Log warning if coverage is low
            if coverage_metrics["production_coverage_pct"] < 90:
                self.logger.warning(
                    f"Low symbol coverage: {coverage_metrics['production_coverage_pct']:.1f}% of production symbols covered in training. "
                    f"This may affect model performance in production."
                )
        
        # Build model architecture
        self.model = self.build_model()
        
        # Train model
        training_history = self.train(train_data, test_data)
        
        # Evaluate model
        evaluation_metrics = self.evaluate(test_data)
        
        # Generate version if not provided
        if version is None:
            version = self.version_manager.generate_version()
            
        # Save model
        model_dir = self.save(version)
        
        # Set as active if requested
        if set_as_active:
            self.version_manager.set_active_version(self.model_type, version)
            
        # Create results dictionary
        results = {
            "version": version,
            "model_dir": model_dir,
            "training": training_history,
            "performance": {
                "evaluation": evaluation_metrics
            },
            "num_training_symbols": len(training_symbols)
        }
        
        # Store training metadata
        self.training_metadata.update({
            "start_date": start_date,
            "end_date": end_date,
            "train_test_split": train_test_split,
            "random_seed": random_seed,
            "feature_names": self.feature_names,
            "feature_importance": self.get_feature_importance(),
            "num_training_symbols": len(training_symbols)
        })
        
        # Store performance metrics
        self.performance_metrics.update(evaluation_metrics)
        
        # End timing
        self.latency_profiler.end_phase()
        
        self.logger.info(f"GBDT model training and evaluation completed with version {version}")
        
        return results
        
    def build_model(self) -> Dict[str, Any]:
        """
        Build the GBDT model architecture.
        
        Returns:
            LightGBM parameters dictionary
        """
        self.logger.info("Building GBDT model")
        
        # Check if GPU is available
        use_gpu = self.training_config.get("use_gpu", True)
        
        # Create LightGBM parameters
        params = {
            'objective': self.objective,
            'metric': self.metric,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'feature_fraction': self.feature_fraction,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'num_threads': self.training_config.get("num_threads", 4),
            'verbose': -1
        }
        
        # Add GPU parameters if enabled
        if use_gpu:
            try:
                # Check if GPU is available in LightGBM
                if 'gpu_platform_id' in lgb.LGBMRegressor().get_params():
                    self.logger.info("Enabling GPU acceleration for LightGBM")
                    params.update({
                        'device': 'gpu',
                        'gpu_platform_id': 0,
                        'gpu_device_id': 0,
                        'gpu_use_dp': False,  # Use double precision for better accuracy
                    })
                else:
                    self.logger.warning("LightGBM was not built with GPU support. Using CPU instead.")
            except Exception as e:
                self.logger.warning(f"Error configuring GPU for LightGBM: {str(e)}. Using CPU instead.")
        
        self.logger.info(f"Built GBDT model with parameters: {params}")
        
        return params
    
    def _get_production_symbols(self) -> List[str]:
        """
        Get the list of symbols used in production.
        
        Returns:
            List of production symbols
        """
        try:
            # Try to get the symbol list from a standard location
            production_symbols_path = os.path.join(self.config.get("data_dir", "data"), "production_symbols.json")
            
            if os.path.exists(production_symbols_path):
                with open(production_symbols_path, 'r') as f:
                    data = json.load(f)
                    return data.get("symbols", [])
            
            # If file doesn't exist, fallback to API to get current symbols
            self.logger.info("Production symbols file not found, querying API")
            return self.data_manager._get_polygon_symbols()
            
        except Exception as e:
            self.logger.error(f"Error getting production symbols: {str(e)}")
            return []
    
    def train(self, train_data: Dict[str, Any], validation_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the GBDT model on the provided data.
        
        Args:
            train_data: Training data
            validation_data: Validation data (optional)
            
        Returns:
            Training history/metrics
        """
        self.logger.info("Training GBDT model with production-identical features")
        
        # Start timing
        self.latency_profiler.start_phase("training")
        
        # Use pre-processed features from the market data processor
        X_train = train_data["X"]
        y_train = train_data["y"]
        
        # Create LightGBM dataset
        lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        
        # Process validation data if provided
        lgb_valid = None
        if validation_data is not None:
            X_valid = validation_data["X"]
            y_valid = validation_data["y"]
            lgb_valid = lgb.Dataset(X_valid, label=y_valid, feature_name=self.feature_names)
            
        # Set up early stopping callback
        callbacks = []
        if lgb_valid is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
            callbacks.append(lgb.log_evaluation(period=10))
        
        # Train model
        self.logger.info(f"Starting LightGBM training with {self.num_boost_round} rounds")
        
        # Get model parameters
        params = self.model if isinstance(self.model, dict) else self.build_model()
        
        # Train model
        evals_result = {}
        
        # Check if the LightGBM version supports evals_result
        import inspect
        train_params = inspect.signature(lgb.train).parameters
        
        if 'evals_result' in train_params:
            self.model = lgb.train(
                params,
                lgb_train,
                num_boost_round=self.num_boost_round,
                valid_sets=[lgb_train, lgb_valid] if lgb_valid is not None else [lgb_train],
                valid_names=['train', 'valid'] if lgb_valid is not None else ['train'],
                callbacks=callbacks,
                evals_result=evals_result
            )
        else:
            # Older version of LightGBM that doesn't support evals_result
            self.logger.warning("LightGBM version doesn't support evals_result, training without it")
            self.model = lgb.train(
                params,
                lgb_train,
                num_boost_round=self.num_boost_round,
                valid_sets=[lgb_train, lgb_valid] if lgb_valid is not None else [lgb_train],
                valid_names=['train', 'valid'] if lgb_valid is not None else ['train'],
                callbacks=callbacks
            )
        
        # End timing
        training_time = self.latency_profiler.end_phase()
        
        # Extract feature importance
        importance = self.model.feature_importance(importance_type='split')
        feature_importance = {}
        for i, name in enumerate(self.feature_names):
            feature_importance[name] = float(importance[i])
        
        # Create training history
        history = {
            "training_time_ms": training_time,
            "num_iterations": self.model.current_iteration(),
            "feature_importance": feature_importance
        }
        
        # Add evals_result if available
        if evals_result:
            history["evals_result"] = evals_result
        
        self.logger.info(f"Completed GBDT training in {training_time:.2f} ms "
                       f"with {self.model.current_iteration()} iterations")
        
        return history
    
    def _prepare_lgb_dataset(self, data: Dict[str, Any]) -> lgb.Dataset:
        """
        Prepare LightGBM dataset from training data.
        
        Args:
            data: Training data dictionary
            
        Returns:
            LightGBM Dataset
        """
        # Extract features and labels from snapshots
        features = []
        labels = []
        
        for i, snapshot in enumerate(data.get("snapshots", [])):
            if "symbols" not in snapshot:
                continue
                
            for symbol, symbol_data in snapshot["symbols"].items():
                # Extract features from symbol data
                symbol_features = []
                
                # Add basic price features
                symbol_features.append(symbol_data.get("last_price", 0))
                symbol_features.append(symbol_data.get("bid_price", 0))
                symbol_features.append(symbol_data.get("ask_price", 0))
                symbol_features.append(symbol_data.get("bid_ask_spread", 0))
                symbol_features.append(symbol_data.get("open_price", 0))
                symbol_features.append(symbol_data.get("high_price", 0))
                symbol_features.append(symbol_data.get("low_price", 0))
                symbol_features.append(symbol_data.get("volume", 0))
                symbol_features.append(symbol_data.get("vwap", 0))
                symbol_features.append(symbol_data.get("prev_close", 0))
                
                # Add technical indicators
                symbol_features.append(symbol_data.get("rsi_14", 50))
                symbol_features.append(symbol_data.get("macd", 0))
                symbol_features.append(symbol_data.get("macd_signal", 0))
                symbol_features.append(symbol_data.get("macd_histogram", 0))
                symbol_features.append(symbol_data.get("bb_upper", 0))
                symbol_features.append(symbol_data.get("bb_middle", 0))
                symbol_features.append(symbol_data.get("bb_lower", 0))
                symbol_features.append(symbol_data.get("atr", 0))
                
                # Add additional indicators
                symbol_features.append(symbol_data.get("avg_volume", 0))
                symbol_features.append(symbol_data.get("volume_acceleration", 0))
                symbol_features.append(symbol_data.get("volume_spike", 0))
                # Removed price_change_5m from features to prevent data leakage
                symbol_features.append(symbol_data.get("momentum_1m", 0))
                symbol_features.append(symbol_data.get("sma_cross_signal", 0))
                
                # Generate synthetic label for training
                # This uses price_change_5m as the target variable, not as a feature
                price_change = symbol_data.get("price_change_5m", 0)
                label = 1 if price_change > 0.005 else 0
                
                features.append(symbol_features)
                labels.append(label)
        
        # Convert to numpy arrays
        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)

        # Log the label distribution for leakage/debugging analysis
        unique, counts = np.unique(y, return_counts=True)
        label_dist = dict(zip(unique.tolist(), counts.tolist()))
        self.logger.info(f"GBDT label distribution: {label_dist}")

        # Store feature names if not already set
        if not self.feature_names:
            self.feature_names = [
                "last_price", "bid_price", "ask_price", "bid_ask_spread",
                "open_price", "high_price", "low_price", "volume", "vwap", "prev_close",
                "rsi_14", "macd", "macd_signal", "macd_histogram",
                "bb_upper", "bb_middle", "bb_lower", "atr",
                "avg_volume", "volume_acceleration", "volume_spike",
                # Removed price_change_5m from feature names
                "momentum_1m", "sma_cross_signal"
            ]
        
        # Create LightGBM dataset
        return lgb.Dataset(X, label=y, feature_name=self.feature_names)
    
    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("Evaluating GBDT model with production-identical features")
        
        # Start timing
        self.latency_profiler.start_phase("evaluation")
        
        # Check if model is trained
        if self.model is None:
            raise RuntimeError("Model is not trained")
        
        # Use pre-processed features from the market data processor
        X = test_data["X"]
        y = test_data["y"]
        
        # Make predictions - use TensorRT if available
        if self.use_tensorrt and self.trt_engine is not None and self.trt_context is not None:
            self.logger.info("Using TensorRT for inference")
            y_pred = self._tensorrt_inference(X)
        else:
            self.logger.info("Using LightGBM for inference")
            y_pred = self.model.predict(X)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Convert predictions to binary
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred_binary)),
            "precision": float(precision_score(y, y_pred_binary, zero_division=0)),
            "recall": float(recall_score(y, y_pred_binary, zero_division=0)),
            "f1": float(f1_score(y, y_pred_binary, zero_division=0)),
            "auc": float(roc_auc_score(y, y_pred)) if len(np.unique(y)) > 1 else 0.5
        }
        
        # End timing
        evaluation_time = self.latency_profiler.end_phase()
        
        # Add evaluation time to metrics
        metrics["evaluation_time_ms"] = evaluation_time
        
        self.logger.info(f"Completed GBDT evaluation in {evaluation_time:.2f} ms "
                       f"with metrics: {metrics}")
        
        return metrics
        
    def _tensorrt_inference(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference using TensorRT engine.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not HAS_TENSORRT or self.trt_engine is None or self.trt_context is None:
            self.logger.warning("TensorRT not available, falling back to LightGBM")
            return self.model.predict(X)
            
        try:
            # Allocate device memory for inputs and outputs
            input_shape = X.shape
            batch_size = input_shape[0]
            
            # Allocate output buffer
            output_size = batch_size
            output_buffer = np.zeros(output_size, dtype=np.float32)
            
            # Allocate device memory
            d_input = cuda.mem_alloc(X.astype(np.float32).nbytes)
            d_output = cuda.mem_alloc(output_buffer.nbytes)
            
            # Create CUDA stream
            stream = cuda.Stream()
            
            # Copy input data to device
            cuda.memcpy_htod_async(d_input, X.astype(np.float32), stream)
            
            # Set input shape if dynamic
            if -1 in self.trt_input_shape:
                self.trt_context.set_binding_shape(0, (batch_size, X.shape[1]))
                
            # Run inference
            bindings = [int(d_input), int(d_output)]
            self.trt_context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            
            # Copy output data to host
            cuda.memcpy_dtoh_async(output_buffer, d_output, stream)
            
            # Synchronize stream
            stream.synchronize()
            
            return output_buffer
            
        except Exception as e:
            self.logger.error(f"Error during TensorRT inference: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Fall back to LightGBM
            self.logger.warning("Falling back to LightGBM for inference")
            return self.model.predict(X)
    
    def save(self, version: str) -> str:
        """
        Save the trained model with the specified version.
        
        Args:
            version: Version string (e.g., "1.0.0")
            
        Returns:
            Path to saved model
        """
        self.logger.info(f"Saving GBDT model version {version} with production-compatible format")
        
        # Check if model is trained
        if self.model is None:
            raise RuntimeError("Model is not trained")
        
        # Save model using ModelVersionManager
        model_dir = self.version_manager.save_model(
            "gbdt",
            version,
            self.model,
            self.training_metadata,
            self.performance_metrics,
            self.latency_profiler.create_latency_profile()
        )
        
        # Export model for production use
        self._export_model_for_production(model_dir)
        
        self.logger.info(f"Saved GBDT model version {version} to {model_dir}")
        
        return model_dir
        
    def _export_to_tensorrt(self, model_dir: str, version: str) -> None:
        """
        Export the trained model to TensorRT format.
        
        Args:
            model_dir: Directory where the model is saved
            version: Version string
        """
        # Since we're removing ONNX, we'll just log a message about TensorRT
        self.logger.info("TensorRT export via ONNX has been removed from the codebase")
        
        # Instead, we'll focus on direct model export
        self._export_model_for_production(model_dir)
            
    def _export_model_for_production(self, model_dir: str) -> None:
        """
        Export model in a format suitable for production use.
        
        Args:
            model_dir: Directory where the model is saved
        """
        self.logger.info(f"Exporting model for production use to {model_dir}")
        
        # Save model in native LightGBM format
        model_path = os.path.join(model_dir, "model.pkl")
        self.model.save_model(model_path)
        
        self.logger.info(f"Successfully exported model to {model_path}")
            
    def _initialize_tensorrt(self, engine_file: str) -> None:
        """
        Initialize TensorRT engine from file.
        
        Args:
            engine_file: Path to TensorRT engine file
        """
        if not HAS_TENSORRT:
            self.logger.warning("TensorRT not available, skipping initialization")
            return
            
        self.logger.info(f"Initializing TensorRT engine from {engine_file}")
        
        try:
            # Create TensorRT runtime and load engine
            trt_logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(trt_logger)
            
            with open(engine_file, 'rb') as f:
                engine_data = f.read()
                
            self.trt_engine = runtime.deserialize_cuda_engine(engine_data)
            if self.trt_engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            # Create execution context
            self.trt_context = self.trt_engine.create_execution_context()
            if self.trt_context is None:
                raise RuntimeError("Failed to create TensorRT execution context")
                
            # Get input and output shapes
            for i in range(self.trt_engine.num_bindings):
                if self.trt_engine.binding_is_input(i):
                    self.trt_input_shape = self.trt_context.get_binding_shape(i)
                    self.logger.info(f"TensorRT input shape: {self.trt_input_shape}")
                else:
                    self.trt_output_shape = self.trt_context.get_binding_shape(i)
                    self.logger.info(f"TensorRT output shape: {self.trt_output_shape}")
                    
            self.logger.info("TensorRT engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing TensorRT engine: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.trt_engine = None
            self.trt_context = None
    
    def load(self, version: Optional[str] = None) -> None:
        """
        Load a trained model.
        
        Args:
            version: Version to load (optional, defaults to active version)
        """
        self.logger.info(f"Loading GBDT model version {version if version else 'active'}")
        
        # Load model using ModelVersionManager
        model, metadata = self.version_manager.load_model("gbdt", version)
        
        # Set model and metadata
        self.model = model
        self.training_metadata = metadata
        
        # Extract feature names from metadata if available
        if "feature_names" in metadata:
            self.feature_names = metadata["feature_names"]
        
        self.logger.info(f"Loaded GBDT model version {version if version else 'active'}")
    
    def _generate_benchmark_data(self, num_samples: int) -> List[Any]:
        """
        Generate synthetic data for benchmarking.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of benchmark data samples
        """
        # Generate random feature data
        X = np.random.rand(num_samples, len(self.feature_names))
        
        return X
    
    def _benchmark_inference_batch(self, batch: List[Any]) -> List[Any]:
        """
        Run inference on a batch of data for benchmarking.
        
        Args:
            batch: Batch of data
            
        Returns:
            Inference results
        """
        # Check if model is trained
        if self.model is None:
            raise RuntimeError("Model is not trained")
        
        # Convert batch to numpy array
        X = np.array(batch)
        
        # Run inference
        return self.model.predict(X)