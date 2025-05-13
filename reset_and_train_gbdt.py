#!/usr/bin/env python3
"""
Reset and Retrain GBDT Model

This script resets the GBDT model by removing existing model files and then
retrains the model to ensure accurate metrics.
"""

import os
import sys
import shutil
import subprocess
import json
import datetime
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(script_dir)
sys.path.insert(0, project_root)

def reset_gbdt_model():
    """
    Reset the GBDT model by removing existing model files.
    """
    print("Resetting GBDT model...")
    
    # Get the latest version directory
    models_dir = os.path.join(project_root, "models", "gbdt")
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        print(f"Models directory {models_dir} does not exist. Nothing to reset.")
        return
    
    # Find all version directories
    version_dirs = [d for d in os.listdir(models_dir) 
                   if os.path.isdir(os.path.join(models_dir, d)) 
                   and d.startswith('v')]
    
    if not version_dirs:
        print("No version directories found. Nothing to reset.")
        return
    
    # Sort version directories to find the latest
    version_dirs.sort(reverse=True)
    
    # Print the versions that will be removed
    print(f"Found {len(version_dirs)} version directories:")
    for version_dir in version_dirs:
        print(f"  - {version_dir}")
    
    # Ask for confirmation
    confirm = input("Do you want to remove these directories? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return
    
    # Remove the version directories
    for version_dir in version_dirs:
        dir_path = os.path.join(models_dir, version_dir)
        try:
            shutil.rmtree(dir_path)
            print(f"Removed {dir_path}")
        except Exception as e:
            print(f"Error removing {dir_path}: {str(e)}")
    
    # Update model registry to remove GBDT entries
    registry_file = os.path.join(project_root, "models", "model_registry.json")
    if os.path.exists(registry_file):
        try:
            with open(registry_file, 'r') as f:
                registry = json.load(f)
            
            # Check if GBDT is in the registry
            if "models" in registry and "gbdt" in registry["models"]:
                # Remove GBDT from registry
                del registry["models"]["gbdt"]
                
                # Update last_updated timestamp
                registry["last_updated"] = datetime.datetime.now().isoformat()
                
                # Save updated registry
                with open(registry_file, 'w') as f:
                    json.dump(registry, f, indent=2)
                
                print(f"Updated model registry: removed GBDT entries")
            else:
                print("GBDT not found in model registry")
        except Exception as e:
            print(f"Error updating model registry: {str(e)}")
    
    print("GBDT model reset complete")

def train_gbdt_model():
    """
    Train a new GBDT model.
    
    Note: The GBDT trainer has been fixed to prevent data leakage where 'price_change_5m'
    was being used both as a feature and to generate the target label. This fix ensures
    more realistic and reliable model performance metrics.
    """
    print("Training new GBDT model...")
    
    # Run the training script
    try:
        # Use subprocess to run the training script
        cmd = [sys.executable, os.path.join(project_root, "src", "ml", "trainer", "train.py"), "--models", "gbdt"]
        
        # Run the command
        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Get return code
        return_code = process.poll()
        
        # Check if training was successful
        if return_code == 0:
            print("GBDT model training completed successfully")
        else:
            # Print error output
            error_output = process.stderr.read()
            print(f"Error during GBDT model training (return code {return_code}):")
            print(error_output)
            return False
        
        return True
    except Exception as e:
        print(f"Error running training script: {str(e)}")
        return False

def main():
    """
    Main function.
    """
    print("Starting GBDT model reset and retraining...")
    
    # Reset GBDT model
    reset_gbdt_model()
    
    # Train new GBDT model
    success = train_gbdt_model()
    
    if success:
        print("GBDT model has been reset and retrained successfully")
    else:
        print("GBDT model retraining failed")
        sys.exit(1)

if __name__ == "__main__":
    main()