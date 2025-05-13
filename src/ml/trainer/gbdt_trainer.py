"""
GBDT Trainer Implementation

This module provides a trainer for Gradient Boosted Decision Trees models
using LightGBM as the backend. It handles training, evaluation, and model
management for the fast path in the trading system.
"""

import numpy as np
from typing import Dict, List, Any, Optional

# Try to import LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, ValueError, AttributeError) as e:
    HAS_LIGHTGBM = False
    import logging
    logging.warning(f"LightGBM not available or incompatible: {str(e)}. GBDT training will not function.")

from src.ml.trainer.base_trainer import ModelTrainer

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
        
        self.logger.info(f"Initialized GBDT trainer with parameters: "
                       f"trees={self.num_trees}, "
                       f"depth={self.max_depth}, "
                       f"learning_rate={self.learning_rate}")
    
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
    
    def train(self, train_data: Dict[str, Any], validation_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the GBDT model on the provided data.
        
        Args:
            train_data: Training data
            validation_data: Validation data (optional)
            
        Returns:
            Training history/metrics
        """
        self.logger.info("Training GBDT model")
        
        # Start timing
        self.latency_profiler.start_phase("training")
        
        # Process training data
        lgb_train = self._prepare_lgb_dataset(train_data)
        
        # Process validation data if provided
        lgb_valid = None
        if validation_data is not None:
            lgb_valid = self._prepare_lgb_dataset(validation_data)
            
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
                symbol_features.append(symbol_data.get("price_change_5m", 0))
                symbol_features.append(symbol_data.get("momentum_1m", 0))
                symbol_features.append(symbol_data.get("sma_cross_signal", 0))
                
                # Generate synthetic label for training
                # In a real implementation, this would use actual labels
                price_change = symbol_data.get("price_change_5m", 0)
                label = 1 if price_change > 0.005 else 0
                
                features.append(symbol_features)
                labels.append(label)
        
        # Convert to numpy arrays
        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        
        # Store feature names if not already set
        if not self.feature_names:
            self.feature_names = [
                "last_price", "bid_price", "ask_price", "bid_ask_spread",
                "open_price", "high_price", "low_price", "volume", "vwap", "prev_close",
                "rsi_14", "macd", "macd_signal", "macd_histogram",
                "bb_upper", "bb_middle", "bb_lower", "atr",
                "avg_volume", "volume_acceleration", "volume_spike",
                "price_change_5m", "momentum_1m", "sma_cross_signal"
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
        self.logger.info("Evaluating GBDT model")
        
        # Start timing
        self.latency_profiler.start_phase("evaluation")
        
        # Check if model is trained
        if self.model is None:
            raise RuntimeError("Model is not trained")
        
        # Process test data
        lgb_test = self._prepare_lgb_dataset(test_data)
        
        # Get feature data and labels
        X = lgb_test.data
        y = lgb_test.label
        
        # Make predictions
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
    
    def save(self, version: str) -> str:
        """
        Save the trained model with the specified version.
        
        Args:
            version: Version string (e.g., "1.0.0")
            
        Returns:
            Path to saved model
        """
        self.logger.info(f"Saving GBDT model version {version}")
        
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
        
        self.logger.info(f"Saved GBDT model version {version} to {model_dir}")
        
        return model_dir
    
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