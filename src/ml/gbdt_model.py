"""
GBDT model implementation for fast path inference

This module provides a production-ready implementation of the Gradient Boosted Decision Trees
model for market data processing and signal generation. It integrates with the
project's data sources and logging system, with optimized LightGBM inference.
"""

import os
import time
import numpy as np
import logging
from typing import List, Dict
import threading

# Try to import LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, ValueError, AttributeError) as e:
    HAS_LIGHTGBM = False
    logging.warning(f"LightGBM not available or incompatible: {str(e)}. GBDT model will not function.")

# Import project modules
from src.monitoring.log import logging as log
from config.config_loader import get_config

class GBDTModel:
    """
    Production-ready GBDT model for market data processing and signal generation.
    
    This model uses LightGBM for efficient gradient boosting decision trees
    to process market data and generate trading signals. It's optimized for
    low-latency inference as part of the fast path in the trading system.
    """
    
    def __init__(self, config=None):
        """
        Initialize the GBDT model with parameters from config.
        
        Args:
            config: Configuration dictionary (optional)
        """
        # Set up logger
        self.logger = log.setup_logger("gbdt_model")
        self.logger.info("Creating GBDT model")
        
        # Load configuration
        if config is None:
            config = get_config()
            
        # Get hardware configuration
        hardware_config = config.get("hardware", {})
        
        # Load model-specific configuration
        import json
        import os
        from pathlib import Path
        
        # Get base model directory
        project_root = Path(__file__).parents[2]  # Go up 2 levels from src/ml
        gbdt_config_path = project_root / "models" / "gbdt" / "model_config.json"
        training_config_path = project_root / "models" / "training_config.json"
        
        # Extract model parameters from config as fallback
        ml_config = config.get("ml", {})
        
        # Load model-specific configuration
        if os.path.exists(gbdt_config_path):
            with open(gbdt_config_path, 'r') as f:
                fast_path_config = json.load(f)
        else:
            self.logger.warning(f"GBDT config file not found: {gbdt_config_path}")
            fast_path_config = ml_config.get("fast_path", {})
            
        # Load training configuration for inference settings
        if os.path.exists(training_config_path):
            with open(training_config_path, 'r') as f:
                training_config = json.load(f)
                inference_config = training_config.get("inference", {})
        else:
            self.logger.warning(f"Training config file not found: {training_config_path}")
            inference_config = ml_config.get("inference", {})
        
        # Initialize with architecture parameters from config
        self.num_trees = fast_path_config.get("num_trees", 150)
        self.max_depth = fast_path_config.get("max_depth", 8)
        self.learning_rate = fast_path_config.get("learning_rate", 0.05)
        self.feature_fraction = fast_path_config.get("feature_fraction", 0.8)
        self.bagging_fraction = fast_path_config.get("bagging_fraction", 0.7)
        self.num_leaves = fast_path_config.get("num_leaves", 31)
        self.early_stopping_rounds = fast_path_config.get("early_stopping_rounds", 10)
        self.num_boost_round = fast_path_config.get("num_boost_round", 100)
        self.bagging_freq = fast_path_config.get("bagging_freq", 5)
        self.objective = fast_path_config.get("objective", "binary")
        self.metric = fast_path_config.get("metric", "auc")
        self.use_fp16 = inference_config.get("use_fp16", True)
        self.max_batch_size = inference_config.get("batch_size", 64)
        
        # Hardware settings
        self.device = hardware_config.get("device", "cuda:0")
        self.memory_limit_mb = hardware_config.get("memory_limit_mb", 80000)
        self.use_tensor_cores = hardware_config.get("use_tensor_cores", True)
        
        # NUMA settings
        self.numa_aware = hardware_config.get("numa_aware", False)
        self.numa_node_mapping = hardware_config.get("numa_node_mapping", {}).get("inference", 0)
        
        # Default shapes - will be updated when model is loaded
        self.input_shape = [self.max_batch_size, 20]  # Batch size, num features
        self.output_shape = [self.max_batch_size, 1]  # Batch size, output size
        
        # Model state
        self.model_path = ""
        self.is_loaded = False
        self.booster = None
        
        # Feature information
        self.feature_names = []
        self.feature_importance = {}
        
        # Threading settings
        self.num_threads = inference_config.get("inference_threads", 2)
        
        # Performance monitoring
        self.total_inference_calls = 0
        self.total_inference_time_ms = 0
        self.last_performance_log = time.time()
        self.performance_log_interval = 60  # Log performance stats every 60 seconds
        
        # Batch processing
        self.batch_lock = threading.Lock()
    
    def __del__(self):
        """Clean up resources when the model is destroyed."""
        self.logger.info("Destroying GBDT model and releasing resources")
        
        try:
            # Clean up LightGBM resources
            if self.booster is not None:
                self.booster = None
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    def load(self, model_path: str) -> None:
        """
        Load the model from the specified path.
        
        Args:
            model_path: Path to the model file
        
        Raises:
            RuntimeError: If the model file is not found or cannot be loaded
        """
        self.logger.info(f"Loading GBDT model from {model_path}")
        
        # Store the model path
        self.model_path = model_path
        
        # Check if file exists
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found: {model_path}")
        
        # Check if LightGBM is available
        if not HAS_LIGHTGBM:
            raise RuntimeError("LightGBM is not available. Cannot load GBDT model.")
        
        # Load the LightGBM model
        self._load_booster(model_path)
        
        # Extract model metadata
        self._extract_model_metadata()
        
        self.is_loaded = True
        self.logger.info("Model loaded successfully and ready for production inference")
    
    def _load_booster(self, model_path: str) -> None:
        """
        Load the LightGBM booster from file.
        
        Args:
            model_path: Path to the model file
            
        Raises:
            RuntimeError: If the model cannot be loaded
        """
        try:
            # Set NUMA affinity if enabled
            if self.numa_aware:
                try:
                    import ctypes
                    libnuma = ctypes.CDLL('libnuma.so.1')
                    if isinstance(self.numa_node_mapping, int):
                        numa_node = self.numa_node_mapping
                    elif isinstance(self.numa_node_mapping, str) and '-' in self.numa_node_mapping:
                        # If range is specified, use the first node in the range
                        numa_node = int(self.numa_node_mapping.split('-')[0])
                    else:
                        numa_node = 0
                        
                    self.logger.info(f"Setting NUMA affinity to node {numa_node}")
                    libnuma.numa_run_on_node(numa_node)
                except Exception as e:
                    self.logger.warning(f"Failed to set NUMA affinity: {str(e)}")
            
            # Load model from file
            self.booster = lgb.Booster(model_file=model_path)
            
            # Set number of threads - use more threads based on CPU core allocation
            self.num_threads = min(self.num_threads * 2, 16)  # Increase threads but cap at 16
            self.booster.params["num_threads"] = self.num_threads
            
            # Set learning rate
            self.booster.params["learning_rate"] = self.learning_rate
            
            # Set feature fraction
            self.booster.params["feature_fraction"] = self.feature_fraction
            
            # Set bagging fraction
            self.booster.params["bagging_fraction"] = self.bagging_fraction
            
            self.logger.info(f"Loaded LightGBM model with parameters: "
                           f"trees={self.num_trees}, "
                           f"depth={self.max_depth}, "
                           f"learning_rate={self.learning_rate}, "
                           f"feature_fraction={self.feature_fraction}, "
                           f"bagging_fraction={self.bagging_fraction}")
            
        except Exception as e:
            self.logger.error(f"Failed to load LightGBM model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load LightGBM model: {str(e)}")
    
    def _extract_model_metadata(self) -> None:
        """
        Extract metadata from the loaded model.
        
        Raises:
            RuntimeError: If the model is not loaded
        """
        if self.booster is None:
            raise RuntimeError("Booster not initialized")
        
        try:
            # Get feature names
            self.feature_names = self.booster.feature_name()
            
            # Update input shape
            self.input_shape[1] = len(self.feature_names)
            
            # Get feature importance
            importance = self.booster.feature_importance(importance_type='split')
            
            # Store feature importance
            self.feature_importance = {}
            for i, name in enumerate(self.feature_names):
                if i < len(importance):
                    self.feature_importance[name] = float(importance[i])
            
            self.logger.info(f"Model has {len(self.feature_names)} features")
            
        except Exception as e:
            self.logger.error(f"Failed to extract model metadata: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to extract model metadata: {str(e)}")
    
    @log.time_function(track_percentiles=True)
    @log.trace_context
    def infer(self, features: List[float]) -> List[float]:
        """
        Perform inference on a single set of features.
        
        Args:
            features: Input features as a flat list of floats
        
        Returns:
            List of output values (prediction score)
        
        Raises:
            RuntimeError: If the model is not loaded or inference fails
        """
        # Check if model is loaded
        if not self.is_loaded or self.booster is None:
            raise RuntimeError("Model not loaded")
        
        # Track performance
        start_time = time.time()
        
        try:
            # Validate input size
            if len(features) != self.input_shape[1]:
                raise RuntimeError(f"Input size mismatch: expected {self.input_shape[1]}, got {len(features)}")
            
            # Reshape features for LightGBM
            features_array = np.array(features).reshape(1, -1)
            
            # Run prediction
            prediction = self.booster.predict(features_array)
            
            # Convert to list
            result = prediction.tolist()
            if isinstance(result, list) and len(result) == 1:
                return result
            else:
                return [float(result)]
            
        except Exception as e:
            self.logger.error(f"Error during inference: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            # Update performance metrics
            elapsed_ms = (time.time() - start_time) * 1000
            with self.batch_lock:
                self.total_inference_calls += 1
                self.total_inference_time_ms += elapsed_ms
                
                # Log performance stats periodically
                current_time = time.time()
                if current_time - self.last_performance_log > self.performance_log_interval:
                    avg_latency = self.total_inference_time_ms / max(1, self.total_inference_calls)
                    self.logger.info(f"Performance stats: {self.total_inference_calls} inferences, "
                                   f"avg latency: {avg_latency:.2f} ms")
                    self.last_performance_log = current_time
    
    @log.time_function(track_percentiles=True)
    @log.trace_context
    def infer_batch(self, features_batch: List[List[float]]) -> List[List[float]]:
        """
        Perform inference on a batch of feature sets.
        
        This method efficiently processes multiple inputs at once using
        LightGBM's batch processing capabilities.
        
        Args:
            features_batch: List of input feature sets
        
        Returns:
            List of output values for each input in the batch
        
        Raises:
            RuntimeError: If the model is not loaded or inference fails
        """
        # Check if model is loaded
        if not self.is_loaded or self.booster is None:
            raise RuntimeError("Model not loaded")
        
        # Track performance
        start_time = time.time()
        batch_size = len(features_batch)
        
        try:
            # Validate batch size
            if not features_batch:
                return []
            
            # Cap batch size to max_batch_size
            if batch_size > self.max_batch_size:
                self.logger.warning(f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}, truncating")
                features_batch = features_batch[:self.max_batch_size]
                batch_size = self.max_batch_size
            
            # Validate input dimensions
            for features in features_batch:
                if len(features) != self.input_shape[1]:
                    raise RuntimeError(f"Input size mismatch in batch: expected {self.input_shape[1]}, got {len(features)}")
            
            # Convert to numpy array
            features_array = np.array(features_batch)
            
            # Run prediction
            predictions = self.booster.predict(features_array)
            
            # Convert to list of lists
            if predictions.ndim == 1:
                return [[float(p)] for p in predictions]
            else:
                return predictions.tolist()
            
        except Exception as e:
            self.logger.error(f"Error during batch inference: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        
        finally:
            # Log batch performance
            elapsed_ms = (time.time() - start_time) * 1000
            self.logger.debug(f"Batch inference of {batch_size} items completed in {elapsed_ms:.2f} ms "
                            f"({elapsed_ms/max(1, batch_size):.2f} ms per item)")
    
    def get_name(self) -> str:
        """
        Get the name of the model.
        
        Returns:
            Model name
        """
        return "GBDT Model (LightGBM - Fast Path)"
    
    def get_input_shape(self) -> List[int]:
        """
        Get the input shape of the model.
        
        Returns:
            Input shape as [batch_size, num_features]
        """
        return self.input_shape
    
    def get_output_shape(self) -> List[int]:
        """
        Get the output shape of the model.
        
        Returns:
            Output shape as [batch_size, output_size]
        """
        return self.output_shape
    
    def set_num_trees(self, num_trees: int) -> None:
        """
        Set the number of trees.
        
        Args:
            num_trees: Number of trees
        """
        self.num_trees = num_trees
        
        # Update booster parameter if loaded
        if self.booster is not None:
            self.booster.params["num_trees"] = num_trees
    
    def set_max_depth(self, max_depth: int) -> None:
        """
        Set the maximum depth of trees.
        
        Args:
            max_depth: Maximum depth
        """
        self.max_depth = max_depth
        
        # Update booster parameter if loaded
        if self.booster is not None:
            self.booster.params["max_depth"] = max_depth
    
    def set_feature_names(self, feature_names: List[str]) -> None:
        """
        Set the feature names.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
    
    def set_num_threads(self, num_threads: int) -> None:
        """
        Set the number of threads for inference.
        
        Args:
            num_threads: Number of threads
        """
        self.num_threads = num_threads
        
        # Update booster parameter if loaded
        if self.booster is not None:
            self.booster.params["num_threads"] = num_threads
    
    def set_learning_rate(self, learning_rate: float) -> None:
        """
        Set the learning rate.
        
        Args:
            learning_rate: Learning rate
        """
        self.learning_rate = learning_rate
        
        # Update booster parameter if loaded
        if self.booster is not None:
            self.booster.params["learning_rate"] = learning_rate
    
    def set_feature_fraction(self, feature_fraction: float) -> None:
        """
        Set the feature fraction.
        
        Args:
            feature_fraction: Feature fraction
        """
        self.feature_fraction = feature_fraction
        
        # Update booster parameter if loaded
        if self.booster is not None:
            self.booster.params["feature_fraction"] = feature_fraction
    
    def set_bagging_fraction(self, bagging_fraction: float) -> None:
        """
        Set the bagging fraction.
        
        Args:
            bagging_fraction: Bagging fraction
        """
        self.bagging_fraction = bagging_fraction
        
        # Update booster parameter if loaded
        if self.booster is not None:
            self.booster.params["bagging_fraction"] = bagging_fraction
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the feature importance.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        return self.feature_importance
    
    def process_market_data(self, market_data: Dict) -> Dict[str, List[float]]:
        """
        Process market data and generate signals.
        
        This method extracts features from market data and runs inference
        to generate trading signals.
        
        Args:
            market_data: Market data dictionary with symbol data
        
        Returns:
            Dictionary mapping symbols to signals
        """
        self.logger.info("Processing market data for fast path screening")
        
        results = {}
        
        # Process each symbol
        for symbol, symbol_data in market_data.items():
            try:
                # Extract features from symbol data
                features = self._extract_features(symbol_data)
                
                # Run inference
                if features:
                    signal = self.infer(features)
                    results[symbol] = signal
                    
                    # Log high-confidence signals
                    if signal[0] > 0.8:  # High confidence
                        self.logger.info(f"High confidence signal for {symbol}: score={signal[0]:.2f}")
            
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {str(e)}")
        
        self.logger.info(f"Generated fast path signals for {len(results)} symbols")
        return results
    
    def _extract_features(self, symbol_data: Dict) -> List[float]:
        """
        Extract features from symbol data for model input.
        
        Args:
            symbol_data: Data for a single symbol
        
        Returns:
            List of features for model input
        """
        # Initialize feature vector with zeros
        features = [0.0] * self.input_shape[1]
        
        # Map feature names to indices
        feature_map = {name: i for i, name in enumerate(self.feature_names)}
        
        # Fill in available features
        for feature_name, value in symbol_data.items():
            if feature_name in feature_map:
                features[feature_map[feature_name]] = float(value)
        
        return features