#!/usr/bin/env python3
"""
Set Active Models

This script sets the active models to the latest versions in the model registry.
It's useful for fixing issues where models are not being set as active correctly.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir))
sys.path.insert(0, project_root)

from src.monitoring.log import logging as log
from src.ml.trainer.model_version_manager import ModelVersionManager

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Set active models")
    
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Models to set as active (comma-separated, or 'all'): gbdt,axial_attention,lstm_gru",
    )
    
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Specific version to set as active (default: latest version)",
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and versions",
    )
    
    return parser.parse_args()

def list_models(version_manager: ModelVersionManager):
    """List available models and versions"""
    logger = log.setup_logger("set_active_models")
    logger.info("Listing available models and versions")
    
    model_types = ["gbdt", "axial_attention", "lstm_gru"]
    
    for model_type in model_types:
        versions = version_manager.get_model_versions(model_type)
        active_version = version_manager.get_active_version(model_type)
        
        logger.info(f"Model type: {model_type}")
        logger.info(f"  Active version: {active_version}")
        logger.info(f"  Available versions:")
        
        for version_info in versions:
            version = version_info["version"]
            created_at = version_info.get("created_at", "Unknown")
            is_active = version == active_version
            
            logger.info(f"    {version} (created: {created_at}) {'[ACTIVE]' if is_active else ''}")
        
        logger.info("")

def get_latest_version(version_manager: ModelVersionManager, model_type: str) -> Optional[str]:
    """Get the latest version for a model type"""
    versions = version_manager.get_model_versions(model_type)
    
    if not versions:
        return None
    
    # Versions are already sorted by semver in descending order
    return versions[0]["version"]

def set_active_models(args):
    """Set active models"""
    logger = log.setup_logger("set_active_models")
    logger.info("Setting active models")
    
    # Initialize version manager
    version_manager = ModelVersionManager()
    
    # List models if requested
    if args.list:
        list_models(version_manager)
        return
    
    # Determine which models to set as active
    model_types = args.models.lower().split(",") if args.models != "all" else ["gbdt", "axial_attention", "lstm_gru"]
    logger.info(f"Setting active models for: {', '.join(model_types)}")
    
    # Set active version for each model
    for model_type in model_types:
        # Get version to set as active
        version = args.version
        if version is None:
            version = get_latest_version(version_manager, model_type)
            if version is None:
                logger.warning(f"No versions found for {model_type}, skipping")
                continue
        
        # Check if version exists
        versions = version_manager.get_model_versions(model_type)
        version_exists = any(v["version"] == version for v in versions)
        
        if not version_exists:
            logger.warning(f"Version {version} not found for {model_type}, skipping")
            continue
        
        # Set active version
        try:
            logger.info(f"Setting {model_type} model version {version} as active")
            version_manager.set_active_version(model_type, version)
            logger.info(f"Successfully set {model_type} model version {version} as active")
        except Exception as e:
            logger.error(f"Error setting {model_type} model version {version} as active: {str(e)}")
    
    # Verify active versions
    logger.info("Verifying active versions")
    for model_type in model_types:
        active_version = version_manager.get_active_version(model_type)
        expected_version = args.version or get_latest_version(version_manager, model_type)
        
        if active_version == expected_version:
            logger.info(f"✅ {model_type} model is active with correct version {active_version}")
        else:
            logger.error(f"❌ {model_type} model active version is {active_version}, expected {expected_version}")

def main():
    """Main function"""
    args = parse_arguments()
    set_active_models(args)

if __name__ == "__main__":
    main()