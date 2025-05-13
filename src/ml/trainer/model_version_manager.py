"""
Model Version Manager

This module provides utilities for managing model versions, including saving,
loading, and tracking model versions with their associated metadata and
performance metrics.
"""

import os
import json
import shutil
from typing import Dict, Any, List, Optional, Tuple
import semver
from datetime import datetime

from src.monitoring.log import logging as log
from config.config_loader import get_config

class ModelVersionManager:
    """
    Model version manager for ML models.
    
    This class provides utilities for managing model versions, including saving,
    loading, and tracking model versions with their associated metadata and
    performance metrics.
    """
    
    # Model types
    MODEL_TYPE_GBDT = "gbdt"
    MODEL_TYPE_AXIAL_ATTENTION = "axial_attention"
    MODEL_TYPE_LSTM_GRU = "lstm_gru"
    
    # File names
    METADATA_FILENAME = "metadata.json"
    PERFORMANCE_FILENAME = "performance.json"
    LATENCY_FILENAME = "latency_profile.json"
    REGISTRY_FILENAME = "model_registry.json"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model version manager.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.logger = log.setup_logger("model_version_manager")
        self.logger.info("Initializing model version manager")
        
        # Load configuration
        if config is None:
            config = get_config()
            
        self.config = config
        
        # Get base model directory from config
        ml_config = config.get("ml", {})
        self.base_model_dir = ml_config.get("model_base_dir", "models")
        
        # Ensure base directory exists
        os.makedirs(self.base_model_dir, exist_ok=True)
        
        # Set registry path
        self.registry_path = os.path.join(self.base_model_dir, self.REGISTRY_FILENAME)
        
        # Check if registry exists, if not create it
        if not os.path.exists(self.registry_path):
            self._init_registry()
        else:
            self.logger.info(f"Using existing model registry at {self.registry_path}")
        
    def _init_registry(self) -> None:
        """Initialize the model registry if it doesn't exist."""
        if not os.path.exists(self.registry_path):
            # Create default registry
            registry = {
                "models": {
                    self.MODEL_TYPE_GBDT: {
                        "active_version": None,
                        "versions": []
                    },
                    self.MODEL_TYPE_AXIAL_ATTENTION: {
                        "active_version": None,
                        "versions": []
                    },
                    self.MODEL_TYPE_LSTM_GRU: {
                        "active_version": None,
                        "versions": []
                    }
                },
                "last_updated": datetime.now().isoformat()
            }
            
            # Save registry
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
                
            self.logger.info(f"Created new model registry at {self.registry_path}")
        else:
            self.logger.info(f"Using existing model registry at {self.registry_path}")
            
    def _load_registry(self) -> Dict[str, Any]:
        """
        Load the model registry.
        
        Returns:
            Model registry dictionary
        """
        try:
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)
                
            return registry
        except Exception as e:
            self.logger.error(f"Error loading model registry: {str(e)}")
            # Return empty registry
            return {
                "models": {
                    self.MODEL_TYPE_GBDT: {"active_version": None, "versions": []},
                    self.MODEL_TYPE_AXIAL_ATTENTION: {"active_version": None, "versions": []},
                    self.MODEL_TYPE_LSTM_GRU: {"active_version": None, "versions": []}
                },
                "last_updated": datetime.now().isoformat()
            }
            
    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """
        Save the model registry.
        
        Args:
            registry: Model registry dictionary
        """
        # Update last updated timestamp
        registry["last_updated"] = datetime.now().isoformat()
        
        try:
            # Create backup of existing registry
            if os.path.exists(self.registry_path):
                backup_path = f"{self.registry_path}.bak"
                shutil.copy2(self.registry_path, backup_path)
                
            # Save registry
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
                
            self.logger.info(f"Saved model registry to {self.registry_path}")
        except Exception as e:
            self.logger.error(f"Error saving model registry: {str(e)}")
            
    def get_model_dir(self, model_type: str, version: str) -> str:
        """
        Get the directory for a specific model version.
        
        Args:
            model_type: Type of model (gbdt, axial_attention, lstm_gru)
            version: Model version (e.g., "1.0.0")
            
        Returns:
            Path to model directory
        """
        return os.path.join(self.base_model_dir, model_type, f"v{version}")
        
    def save_model(self, model_type: str, version: str, model_data: Any,
                  metadata: Dict[str, Any], performance: Dict[str, Any],
                  latency_profile: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a model with its metadata and performance metrics.
        
        Args:
            model_type: Type of model (gbdt, axial_attention, lstm_gru)
            version: Model version (e.g., "1.0.0")
            model_data: Model data to save
            metadata: Model metadata
            performance: Model performance metrics
            latency_profile: Model latency profile (optional)
            
        Returns:
            Path to saved model directory
        """
        # Validate model type
        if model_type not in [self.MODEL_TYPE_GBDT, self.MODEL_TYPE_AXIAL_ATTENTION, self.MODEL_TYPE_LSTM_GRU]:
            raise ValueError(f"Invalid model type: {model_type}")
            
        # Validate version format
        try:
            semver.parse(version)
        except ValueError:
            raise ValueError(f"Invalid version format: {version}. Must be semver format (e.g., 1.0.0)")
            
        # Get model directory
        model_dir = self.get_model_dir(model_type, version)
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model data
        model_path = self._save_model_data(model_type, model_dir, model_data)
        
        # Add model path to metadata
        metadata["model_path"] = model_path
        metadata["saved_at"] = datetime.now().isoformat()
        
        # Save metadata
        metadata_path = os.path.join(model_dir, self.METADATA_FILENAME)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save performance metrics
        performance_path = os.path.join(model_dir, self.PERFORMANCE_FILENAME)
        with open(performance_path, 'w') as f:
            json.dump(performance, f, indent=2)
            
        # Save latency profile if provided
        if latency_profile:
            latency_path = os.path.join(model_dir, self.LATENCY_FILENAME)
            with open(latency_path, 'w') as f:
                json.dump(latency_profile, f, indent=2)
                
        # Update registry
        self._update_registry(model_type, version, metadata, performance)
        
        self.logger.info(f"Saved {model_type} model v{version} to {model_dir}")
        
        return model_dir
        
    def _save_model_data(self, model_type: str, model_dir: str, model_data: Any) -> str:
        """
        Save model data based on model type.
        
        Args:
            model_type: Type of model
            model_dir: Directory to save model to
            model_data: Model data to save
            
        Returns:
            Path to saved model file
        """
        if model_type == self.MODEL_TYPE_GBDT:
            # Save LightGBM model
            model_path = os.path.join(model_dir, "model.pkl")
            if hasattr(model_data, 'save_model'):
                model_data.save_model(model_path)
            else:
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
        elif model_type == self.MODEL_TYPE_AXIAL_ATTENTION:
            # Save PyTorch model
            model_path = os.path.join(model_dir, "model.pt")
            if hasattr(model_data, 'state_dict'):
                import torch
                torch.save(model_data.state_dict(), model_path)
            else:
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
        elif model_type == self.MODEL_TYPE_LSTM_GRU:
            # Save PyTorch model
            model_path = os.path.join(model_dir, "model.pt")
            if hasattr(model_data, 'state_dict'):
                import torch
                torch.save(model_data.state_dict(), model_path)
            else:
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        return model_path
        
    def _update_registry(self, model_type: str, version: str, 
                        metadata: Dict[str, Any], performance: Dict[str, Any]) -> None:
        """
        Update the model registry with a new model version.
        
        Args:
            model_type: Type of model
            version: Model version
            metadata: Model metadata
            performance: Model performance metrics
        """
        # Load registry
        registry = self._load_registry()
        
        # Get model registry
        model_registry = registry["models"].get(model_type, {"active_version": None, "versions": []})
        
        # Check if version already exists
        version_exists = False
        for v in model_registry["versions"]:
            if v["version"] == version:
                # Update existing version
                v["metadata"] = metadata
                v["performance"] = performance
                v["updated_at"] = datetime.now().isoformat()
                version_exists = True
                break
                
        if not version_exists:
            # Add new version
            model_registry["versions"].append({
                "version": version,
                "metadata": metadata,
                "performance": performance,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
            
        # Sort versions by semver
        model_registry["versions"] = sorted(
            model_registry["versions"],
            key=lambda v: semver.VersionInfo.parse(v["version"]),
            reverse=True
        )
        
        # Update registry
        registry["models"][model_type] = model_registry
        
        # Save registry
        self._save_registry(registry)
        
    def set_active_version(self, model_type: str, version: str) -> None:
        """
        Set the active version for a model type.
        
        Args:
            model_type: Type of model
            version: Model version to set as active
        """
        # Load registry
        registry = self._load_registry()
        
        # Get model registry
        model_registry = registry["models"].get(model_type)
        if not model_registry:
            raise ValueError(f"Model type not found in registry: {model_type}")
            
        # Check if version exists
        version_exists = False
        for v in model_registry["versions"]:
            if v["version"] == version:
                version_exists = True
                break
                
        if not version_exists:
            raise ValueError(f"Version {version} not found for model type {model_type}")
            
        # Set active version
        model_registry["active_version"] = version
        
        # Update registry
        registry["models"][model_type] = model_registry
        
        # Save registry
        self._save_registry(registry)
        
        # Update model config JSON to point to the new model
        self._update_model_config(model_type, version)
        
        self.logger.info(f"Set active version for {model_type} to v{version}")
        
    def _update_model_config(self, model_type: str, version: str) -> None:
        """
        Update model config JSON file to point to the new model.
        
        Args:
            model_type: Type of model
            version: Model version
        """
        # Get model path
        model_dir = self.get_model_dir(model_type, version)
        
        # Determine model file based on model type
        if model_type == self.MODEL_TYPE_GBDT:
            model_file = "model.pkl"
            config_path = os.path.join(self.base_model_dir, "gbdt", "model_config.json")
        elif model_type == self.MODEL_TYPE_AXIAL_ATTENTION:
            model_file = "model.pt"
            config_path = os.path.join(self.base_model_dir, "axial_attention", "model_config.json")
        elif model_type == self.MODEL_TYPE_LSTM_GRU:
            model_file = "model.pt"
            config_path = os.path.join(self.base_model_dir, "lstm_gru", "model_config.json")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
                
        # Full model path
        model_path = os.path.join(model_dir, model_file)
        
        # Update model config JSON file
        if os.path.exists(config_path):
            try:
                # Load existing config
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Update model path
                config["model_path"] = model_path
                
                # Save updated config
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                self.logger.info(f"Updated model config at {config_path} with model_path = {model_path}")
            except Exception as e:
                self.logger.error(f"Error updating model config: {str(e)}")
        else:
            self.logger.warning(f"Model config file not found: {config_path}")
        
    def get_active_version(self, model_type: str) -> Optional[str]:
        """
        Get the active version for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Active version or None if no active version
        """
        # Load registry
        registry = self._load_registry()
        
        # Get model registry
        model_registry = registry["models"].get(model_type)
        if not model_registry:
            return None
            
        return model_registry.get("active_version")
        
    def get_model_versions(self, model_type: str) -> List[Dict[str, Any]]:
        """
        Get all versions for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            List of model versions with metadata
        """
        # Load registry
        registry = self._load_registry()
        
        # Get model registry
        model_registry = registry["models"].get(model_type, {"versions": []})
        
        return model_registry.get("versions", [])
        
    def get_model_metadata(self, model_type: str, version: str) -> Dict[str, Any]:
        """
        Get metadata for a specific model version.
        
        Args:
            model_type: Type of model
            version: Model version
            
        Returns:
            Model metadata
        """
        # Get model directory
        model_dir = self.get_model_dir(model_type, version)
        
        # Check if metadata file exists
        metadata_path = os.path.join(model_dir, self.METADATA_FILENAME)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        return metadata
        
    def get_model_performance(self, model_type: str, version: str) -> Dict[str, Any]:
        """
        Get performance metrics for a specific model version.
        
        Args:
            model_type: Type of model
            version: Model version
            
        Returns:
            Model performance metrics
        """
        # Get model directory
        model_dir = self.get_model_dir(model_type, version)
        
        # Check if performance file exists
        performance_path = os.path.join(model_dir, self.PERFORMANCE_FILENAME)
        if not os.path.exists(performance_path):
            raise FileNotFoundError(f"Performance file not found: {performance_path}")
            
        # Load performance metrics
        with open(performance_path, 'r') as f:
            performance = json.load(f)
            
        return performance
        
    def get_model_latency_profile(self, model_type: str, version: str) -> Optional[Dict[str, Any]]:
        """
        Get latency profile for a specific model version.
        
        Args:
            model_type: Type of model
            version: Model version
            
        Returns:
            Model latency profile or None if not available
        """
        # Get model directory
        model_dir = self.get_model_dir(model_type, version)
        
        # Check if latency profile file exists
        latency_path = os.path.join(model_dir, self.LATENCY_FILENAME)
        if not os.path.exists(latency_path):
            return None
            
        # Load latency profile
        with open(latency_path, 'r') as f:
            latency_profile = json.load(f)
            
        return latency_profile
        
    def load_model(self, model_type: str, version: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a model with its metadata.
        
        Args:
            model_type: Type of model
            version: Model version (optional, defaults to active version)
            
        Returns:
            Tuple of (model, metadata)
        """
        # Get version to load
        if version is None:
            version = self.get_active_version(model_type)
            if version is None:
                raise ValueError(f"No active version found for model type: {model_type}")
                
        # Get model directory
        model_dir = self.get_model_dir(model_type, version)
        
        # Load metadata
        metadata = self.get_model_metadata(model_type, version)
        
        # Load model based on model type
        if model_type == self.MODEL_TYPE_GBDT:
            # Load LightGBM model
            model_path = os.path.join(model_dir, "model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
        elif model_type == self.MODEL_TYPE_AXIAL_ATTENTION:
            # Load PyTorch model
            model_path = os.path.join(model_dir, "model.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            import torch
            model = torch.load(model_path)
            
        elif model_type == self.MODEL_TYPE_LSTM_GRU:
            # Load PyTorch model
            model_path = os.path.join(model_dir, "model.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            import torch
            model = torch.load(model_path)
                
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.logger.info(f"Loaded {model_type} model v{version} from {model_dir}")
        
        return model, metadata
        
    def limit_model_versions(self, model_type: str, max_versions: int = 10) -> None:
        """
        Limit the number of model versions kept for a model type.
        Keeps only the top N versions based on performance metrics.
        
        Args:
            model_type: Type of model
            max_versions: Maximum number of versions to keep
        """
        self.logger.info(f"Limiting {model_type} models to top {max_versions} versions")
        
        # Load registry
        registry = self._load_registry()
        
        # Get model registry
        model_registry = registry["models"].get(model_type)
        if not model_registry or not model_registry.get("versions"):
            self.logger.info(f"No versions found for model type: {model_type}")
            return
            
        versions = model_registry["versions"]
        
        # If we don't have more than the max, no need to limit
        if len(versions) <= max_versions:
            self.logger.info(f"Only {len(versions)} versions exist for {model_type}, no cleanup needed")
            return
            
        # Sort versions by performance metrics (higher is better)
        # First try to use AUC as the metric, then accuracy, then any available metric
        def get_performance_score(version_info):
            perf = version_info.get("performance", {})
            eval_metrics = perf.get("evaluation", {})
            
            # Try to get AUC first
            if "auc" in eval_metrics:
                return float(eval_metrics["auc"])
            # Then try accuracy
            elif "accuracy" in eval_metrics:
                return float(eval_metrics["accuracy"])
            # Then try f1
            elif "f1" in eval_metrics:
                return float(eval_metrics["f1"])
            # Then try any other numeric metric
            else:
                for key, value in eval_metrics.items():
                    if isinstance(value, (int, float)):
                        return float(value)
                return 0.0
        
        # Sort versions by performance score (descending)
        sorted_versions = sorted(versions, key=get_performance_score, reverse=True)
        
        # Keep only the top N versions
        versions_to_keep = sorted_versions[:max_versions]
        versions_to_delete = sorted_versions[max_versions:]
        
        # Update registry with versions to keep
        model_registry["versions"] = versions_to_keep
        registry["models"][model_type] = model_registry
        
        # Save registry
        self._save_registry(registry)
        
        # Delete model directories for versions to delete
        for version_info in versions_to_delete:
            version = version_info["version"]
            model_dir = self.get_model_dir(model_type, version)
            
            # Check if this is the active version
            active_version = model_registry.get("active_version")
            if active_version == version:
                self.logger.warning(f"Cannot delete active version {version} for {model_type}")
                continue
                
            # Delete model directory
            try:
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                    self.logger.info(f"Deleted {model_type} model v{version} directory: {model_dir}")
            except Exception as e:
                self.logger.error(f"Error deleting model directory {model_dir}: {str(e)}")
                
        self.logger.info(f"Kept top {len(versions_to_keep)} versions for {model_type}, deleted {len(versions_to_delete)} versions")
        
    def promote_best_model(self, model_type: str) -> Optional[str]:
        """
        Promote the best performing model to be the active version.
        
        Args:
            model_type: Type of model
            
        Returns:
            Version of the promoted model or None if no models available
        """
        self.logger.info(f"Promoting best {model_type} model to active version")
        
        # Load registry
        registry = self._load_registry()
        
        # Get model registry
        model_registry = registry["models"].get(model_type)
        if not model_registry or not model_registry.get("versions"):
            self.logger.info(f"No versions found for model type: {model_type}")
            return None
            
        versions = model_registry["versions"]
        
        # Sort versions by performance metrics (higher is better)
        def get_performance_score(version_info):
            perf = version_info.get("performance", {})
            eval_metrics = perf.get("evaluation", {})
            
            # Try to get AUC first
            if "auc" in eval_metrics:
                return float(eval_metrics["auc"])
            # Then try accuracy
            elif "accuracy" in eval_metrics:
                return float(eval_metrics["accuracy"])
            # Then try f1
            elif "f1" in eval_metrics:
                return float(eval_metrics["f1"])
            # Then try any other numeric metric
            else:
                for key, value in eval_metrics.items():
                    if isinstance(value, (int, float)):
                        return float(value)
                return 0.0
        
        # Sort versions by performance score (descending)
        sorted_versions = sorted(versions, key=get_performance_score, reverse=True)
        
        if not sorted_versions:
            self.logger.warning(f"No versions with performance metrics found for {model_type}")
            return None
            
        # Get best version
        best_version = sorted_versions[0]["version"]
        
        # Set as active version
        self.set_active_version(model_type, best_version)
        
        self.logger.info(f"Promoted {model_type} model v{best_version} to active version")
        
        return best_version
        
    def clean_training_data(self) -> None:
        """
        Clean up training data to keep the system clean.
        Deletes cached data and temporary files.
        """
        self.logger.info("Cleaning up training data")
        
        # Clean up cache directory
        cache_dir = os.path.join("data", "cache")
        if os.path.exists(cache_dir):
            try:
                # Delete all files in cache directory
                for filename in os.listdir(cache_dir):
                    file_path = os.path.join(cache_dir, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        
                self.logger.info(f"Cleaned up cache directory: {cache_dir}")
            except Exception as e:
                self.logger.error(f"Error cleaning cache directory: {str(e)}")
                
        # Clean up temporary files
        temp_dirs = ["tmp", "temp"]
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    self.logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    self.logger.error(f"Error cleaning temporary directory {temp_dir}: {str(e)}")