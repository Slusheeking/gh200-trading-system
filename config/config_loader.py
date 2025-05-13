"""
Configuration Loader for GH200 Trading System

This module loads and parses configuration files from both YAML and JSON sources.
It provides access to the configuration settings for all components.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class ConfigLoader:
    """
    Configuration loader for the GH200 Trading System
    
    This class loads and parses configuration files from both YAML and JSON sources.
    It provides access to the configuration settings for all components.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader
        
        Args:
            config_path: Path to the main configuration file. If None, uses default path.
        """
        self.config: Dict[str, Any] = {}
        self.project_root = Path(__file__).parents[1]  # Go up 1 level from config
        
        # Default config path is in the settings directory at the project root
        if config_path is None:
            config_path = str(self.project_root / "settings" / "system.yaml")
        
        self.config_path = config_path
        
        # Load environment variables from .env file
        dotenv_path = self.project_root / ".env"
        load_dotenv(dotenv_path=dotenv_path)
        
        # Load configurations
        self._load_yaml_config()
        self._load_model_configs()
        
    def _load_yaml_config(self):
        """Load the configuration from the YAML file"""
        try:
            # Check if file exists
            if not os.path.exists(self.config_path):
                logging.error(f"Configuration file not found: {self.config_path}")
                return
            
            # Load YAML file
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            # Process environment variables in the config
            self._process_env_vars(self.config)
            
            logging.info(f"Loaded YAML configuration from {self.config_path}")
        except Exception as e:
            logging.error(f"Error loading YAML configuration: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
            
    def _load_model_configs(self):
        """Load model configurations from JSON files"""
        try:
            # Initialize ML config if it doesn't exist
            if "ml" not in self.config:
                self.config["ml"] = {}
                
            # Load model registry
            registry_path = self.project_root / "models" / "model_registry.json"
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as file:
                    registry = json.load(file)
                self.config["ml"]["model_registry"] = registry
                logging.info(f"Loaded model registry from {registry_path}")
                
            # Load training config
            training_config_path = self.project_root / "models" / "training_config.json"
            if os.path.exists(training_config_path):
                with open(training_config_path, 'r') as file:
                    training_config = json.load(file)
                self.config["ml"]["training"] = training_config
                logging.info(f"Loaded training config from {training_config_path}")
                
            # Load model-specific configs
            model_types = ["gbdt", "axial_attention", "lstm_gru"]
            for model_type in model_types:
                config_path = self.project_root / "models" / model_type / "model_config.json"
                if os.path.exists(config_path):
                    with open(config_path, 'r') as file:
                        model_config = json.load(file)
                    
                    # Map model type to config key
                    if model_type == "gbdt":
                        config_key = "fast_path"
                    elif model_type == "axial_attention":
                        config_key = "accurate_path"
                    elif model_type == "lstm_gru":
                        config_key = "exit_optimization"
                    else:
                        config_key = model_type
                        
                    self.config["ml"][config_key] = model_config
                    
                    # Extract features if available
                    if "features" in model_config and model_type == "gbdt":
                        self.config["ml"]["fast_path_features"] = model_config["features"]
                        
                    logging.info(f"Loaded {model_type} config from {config_path}")
                    
            # Extract model paths
            self.config["ml"]["model_paths"] = {
                "fast_path": self.config["ml"].get("fast_path", {}).get("model_path", ""),
                "accurate_path": self.config["ml"].get("accurate_path", {}).get("model_path", ""),
                "exit_model": self.config["ml"].get("exit_optimization", {}).get("model_path", "")
            }
            
        except Exception as e:
            logging.error(f"Error loading model configurations: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
    
    def _process_env_vars(self, config_dict: Dict[str, Any]):
        """
        Process environment variables in the configuration
        
        This method recursively searches for strings in the format "${ENV_VAR}"
        and replaces them with the corresponding environment variable value.
        
        Args:
            config_dict: Configuration dictionary to process
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                self._process_env_vars(value)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Extract environment variable name
                env_var = value[2:-1]
                
                # Get environment variable value
                env_value = os.getenv(env_var)
                
                if env_value is not None:
                    # Replace with environment variable value
                    config_dict[key] = env_value
                else:
                    # Try to get from .env file directly (in case it wasn't loaded properly)
                    from dotenv import dotenv_values
                    
                    project_root = Path(__file__).parents[1]
                    dotenv_path = project_root / ".env"
                    env_values = dotenv_values(dotenv_path=dotenv_path)
                    
                    if env_var in env_values:
                        config_dict[key] = env_values[env_var]
                    else:
                        logging.warning(f"Environment variable not found: {env_var}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the full configuration dictionary
        
        Returns:
            Configuration dictionary
        """
        return self.config
    
    def get(self, *keys, default=None):
        """
        Get a configuration value by key path
        
        Args:
            *keys: Key path to the configuration value
            default: Default value to return if the key is not found
            
        Returns:
            Configuration value or default if not found
        """
        current = self.config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current

# Create a singleton instance
config_loader = ConfigLoader()

def get_config() -> Dict[str, Any]:
    """
    Get the full configuration dictionary
    
    Returns:
        Configuration dictionary
    """
    return config_loader.get_config()

def get(*keys, default=None):
    """
    Get a configuration value by key path
    
    Args:
        *keys: Key path to the configuration value
        default: Default value to return if the key is not found
        
    Returns:
        Configuration value or default if not found
    """
    return config_loader.get(*keys, default=default)
