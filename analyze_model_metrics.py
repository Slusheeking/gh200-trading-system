import os
import json
import glob
import pandas as pd
from tabulate import tabulate
from collections import defaultdict

def analyze_model_metrics():
    """
    Analyze and display metrics for all models in the system.
    """
    print("Analyzing model metrics for all models...\n")
    
    # Get base models directory
    models_dir = "models"
    
    # Dictionary to store metrics by model type
    metrics_by_type = defaultdict(list)
    
    # Find all model types
    model_types = [d for d in os.listdir(models_dir) 
                  if os.path.isdir(os.path.join(models_dir, d)) 
                  and not d.startswith('.')]
    
    print(f"Found model types: {', '.join(model_types)}\n")
    
    # Process each model type
    for model_type in model_types:
        model_type_dir = os.path.join(models_dir, model_type)
        
        # Find all version directories
        version_dirs = [d for d in os.listdir(model_type_dir) 
                       if os.path.isdir(os.path.join(model_type_dir, d)) 
                       and d.startswith('v')]
        
        if not version_dirs:
            print(f"No versions found for {model_type}")
            continue
            
        print(f"Found {len(version_dirs)} versions for {model_type}")
        
        # Process each version
        for version in version_dirs:
            version_dir = os.path.join(model_type_dir, version)
            
            # Check for performance.json
            perf_file = os.path.join(version_dir, "performance.json")
            if not os.path.exists(perf_file):
                continue
                
            # Load performance metrics
            try:
                with open(perf_file, 'r') as f:
                    perf_data = json.load(f)
                
                # Load metadata if available
                metadata = {}
                metadata_file = os.path.join(version_dir, "metadata.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                # Combine metrics with version info
                metrics = {
                    "model_type": model_type,
                    "version": version,
                    "timestamp": metadata.get("timestamp", "N/A"),
                    "training_duration": metadata.get("training_duration", "N/A")
                }
                
                # Add all performance metrics
                metrics.update(perf_data)
                
                # Add to the collection
                metrics_by_type[model_type].append(metrics)
                
            except Exception as e:
                print(f"Error processing {perf_file}: {str(e)}")
    
    # Display metrics for each model type
    for model_type, metrics_list in metrics_by_type.items():
        print(f"\n{'-'*80}")
        print(f"Metrics for {model_type.upper()} models:")
        print(f"{'-'*80}")
        
        if not metrics_list:
            print("No metrics available")
            continue
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(metrics_list)
        
        # Sort by version (assuming version format is consistent)
        df = df.sort_values(by="version", ascending=False)
        
        # Select columns to display
        display_columns = ["version", "timestamp"]
        
        # Add common metric columns based on what's available
        metric_columns = []
        
        # For GBDT models
        if model_type == "gbdt":
            potential_metrics = [
                "accuracy", "precision", "recall", "f1_score", "auc", 
                "training_time", "inference_time_avg", "inference_time_p95",
                "training_samples", "test_samples"
            ]
            metric_columns = [col for col in potential_metrics if col in df.columns]
        
        # For axial_attention models
        elif model_type == "axial_attention":
            potential_metrics = [
                "accuracy", "precision", "recall", "f1_score", 
                "training_loss", "validation_loss", "training_time"
            ]
            metric_columns = [col for col in potential_metrics if col in df.columns]
        
        # For lstm_gru models
        elif model_type == "lstm_gru":
            potential_metrics = [
                "mse", "mae", "rmse", "r2_score", 
                "training_loss", "validation_loss", "training_time"
            ]
            metric_columns = [col for col in potential_metrics if col in df.columns]
        
        # Display the metrics
        display_columns.extend(metric_columns)
        
        # Format the table
        if len(df) > 0:
            # Format numeric columns
            for col in metric_columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if "time" in col and "training_time" not in col:
                        # Format time in milliseconds
                        df[col] = df[col].apply(lambda x: f"{x*1000:.2f}ms" if pd.notnull(x) else "N/A")
                    elif col in ["accuracy", "precision", "recall", "f1_score", "auc"]:
                        # Format percentages
                        df[col] = df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
                    else:
                        # Format other numeric values
                        df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
            
            # Print the table
            print(tabulate(df[display_columns], headers="keys", tablefmt="grid", showindex=False))
            
            # Print the best model based on primary metric
            if "accuracy" in df.columns:
                best_idx = df["accuracy"].astype(float).idxmax()
                best_version = df.loc[best_idx, "version"]
                best_accuracy = df.loc[best_idx, "accuracy"]
                print(f"\nBest model: {best_version} with accuracy {best_accuracy}")
            elif "f1_score" in df.columns:
                best_idx = df["f1_score"].astype(float).idxmax()
                best_version = df.loc[best_idx, "version"]
                best_f1 = df.loc[best_idx, "f1_score"]
                print(f"\nBest model: {best_version} with F1 score {best_f1}")
        else:
            print("No metrics available to display")
    
    # Check for model registry
    registry_file = os.path.join(models_dir, "model_registry.json")
    if os.path.exists(registry_file):
        try:
            with open(registry_file, 'r') as f:
                registry = json.load(f)
            
            print(f"\n{'-'*80}")
            print("Currently active models from registry:")
            print(f"{'-'*80}")
            
            for model_type, info in registry.items():
                if isinstance(info, dict) and "active_version" in info:
                    print(f"{model_type}: {info['active_version']}")
        except Exception as e:
            print(f"Error reading model registry: {str(e)}")

if __name__ == "__main__":
    analyze_model_metrics()