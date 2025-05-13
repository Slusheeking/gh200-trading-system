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
    
    # Track processed versions to avoid duplicates
    processed_versions = defaultdict(set)
    
    # First, check the model registry for comprehensive information
    registry_file = os.path.join(models_dir, "model_registry.json")
    if os.path.exists(registry_file):
        try:
            with open(registry_file, 'r') as f:
                registry = json.load(f)
            
            # Extract metrics from registry
            if "models" in registry:
                for model_type, model_info in registry["models"].items():
                    if "versions" in model_info:
                        for version_info in model_info["versions"]:
                            version = version_info.get("version", "unknown")
                            metadata = version_info.get("metadata", {})
                            performance = version_info.get("performance", {})
                            
                            # Initialize metrics with version info
                            metrics = {
                                "model_type": model_type,
                                "version": version,
                                "timestamp": metadata.get("created_at", metadata.get("training_date", "N/A")),
                                "name": metadata.get("name", "N/A"),
                                "description": metadata.get("description", "N/A"),
                                "source": "registry"
                            }
                            
                            # Extract performance metrics
                            if isinstance(performance, dict):
                                # Direct metrics at top level
                                for key, value in performance.items():
                                    if key not in metrics and key not in ["training", "evaluation", "latency"]:
                                        metrics[key] = value
                                
                                # Nested metrics in evaluation section
                                if "evaluation" in performance:
                                    eval_metrics = performance["evaluation"]
                                    metrics.update({
                                        "accuracy": eval_metrics.get("accuracy", "N/A"),
                                        "precision": eval_metrics.get("precision", "N/A"),
                                        "recall": eval_metrics.get("recall", "N/A"),
                                        "f1_score": eval_metrics.get("f1", eval_metrics.get("f1_score", "N/A")),
                                        "auc": eval_metrics.get("auc", "N/A"),
                                        "evaluation_time_ms": eval_metrics.get("evaluation_time_ms", "N/A")
                                    })
                                
                                # Nested metrics in training section
                                if "training" in performance:
                                    train_metrics = performance["training"]
                                    metrics.update({
                                        "training_time_ms": train_metrics.get("training_time_ms", "N/A"),
                                        "num_iterations": train_metrics.get("num_iterations", "N/A")
                                    })
                                    
                                    # Extract feature importance if available
                                    if "feature_importance" in train_metrics:
                                        # Find top 3 features by importance
                                        feature_imp = train_metrics["feature_importance"]
                                        sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
                                        top_features = sorted_features[:3]
                                        
                                        for i, (feature, importance) in enumerate(top_features):
                                            metrics[f"top_feature_{i+1}"] = feature
                                            metrics[f"importance_{i+1}"] = importance
                            
                            # Add to the collection and mark as processed
                            metrics_by_type[model_type].append(metrics)
                            processed_versions[model_type].add(version)
        except Exception as e:
            print(f"Error reading model registry: {str(e)}")
    
    # Then, check individual performance files for any models not in registry
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
            print(f"No version directories found for {model_type}")
            continue
            
        print(f"Found {len(version_dirs)} version directories for {model_type}")
        
        # Process each version
        for version in version_dirs:
            # Extract version number without 'v' prefix
            version_num = version[1:] if version.startswith('v') else version
            
            # Skip if we already have this version from registry
            if version_num in processed_versions[model_type]:
                continue
                
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
                
                # Initialize metrics with version info
                metrics = {
                    "model_type": model_type,
                    "version": version_num,  # Store without 'v' prefix for consistency
                    "timestamp": metadata.get("timestamp", "N/A"),
                    "training_duration": metadata.get("training_duration", "N/A"),
                    "source": "performance_file"
                }
                
                # Extract evaluation metrics
                if "evaluation" in perf_data:
                    eval_metrics = perf_data["evaluation"]
                    metrics.update({
                        "accuracy": eval_metrics.get("accuracy", "N/A"),
                        "precision": eval_metrics.get("precision", "N/A"),
                        "recall": eval_metrics.get("recall", "N/A"),
                        "f1_score": eval_metrics.get("f1", "N/A"),
                        "auc": eval_metrics.get("auc", "N/A"),
                        "evaluation_time_ms": eval_metrics.get("evaluation_time_ms", "N/A")
                    })
                
                # Extract training metrics
                if "training" in perf_data:
                    train_metrics = perf_data["training"]
                    metrics.update({
                        "training_time_ms": train_metrics.get("training_time_ms", "N/A"),
                        "num_iterations": train_metrics.get("num_iterations", "N/A")
                    })
                    
                    # Extract feature importance if available
                    if "feature_importance" in train_metrics:
                        # Find top 3 features by importance
                        feature_imp = train_metrics["feature_importance"]
                        sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
                        top_features = sorted_features[:3]
                        
                        for i, (feature, importance) in enumerate(top_features):
                            metrics[f"top_feature_{i+1}"] = feature
                            metrics[f"importance_{i+1}"] = importance
                
                # Extract latency metrics
                if "latency" in perf_data:
                    latency_metrics = perf_data["latency"]
                    metrics.update({
                        "inference_time_ms": latency_metrics.get("inference_time_ms", 
                                            latency_metrics.get("evaluation_time_ms", "N/A"))
                    })
                
                # Add to the collection
                metrics_by_type[model_type].append(metrics)
                processed_versions[model_type].add(version_num)
                
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
        
        # Select columns to display based on model type
        display_columns = ["version"]
        
        # Add name and description if available
        if "name" in df.columns and not df["name"].isna().all():
            display_columns.append("name")
        
        # Add common metric columns based on what's available
        if "accuracy" in df.columns:
            display_columns.extend(["accuracy", "precision", "recall", "f1_score", "auc"])
        
        if "training_time_ms" in df.columns:
            display_columns.append("training_time_ms")
        
        if "evaluation_time_ms" in df.columns:
            display_columns.append("evaluation_time_ms")
            
        if "inference_time_ms" in df.columns:
            display_columns.append("inference_time_ms")
            
        if "num_iterations" in df.columns:
            display_columns.append("num_iterations")
            
        # Add top features if available
        feature_cols = [col for col in df.columns if col.startswith("top_feature_")]
        if feature_cols and not df[feature_cols[0]].isna().all():
            display_columns.extend(feature_cols)
        
        # Filter to only include columns that exist in the DataFrame
        display_columns = [col for col in display_columns if col in df.columns]
        
        # Format the table
        if len(df) > 0:
            # Format numeric columns
            for col in df.columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if "time_ms" in col:
                        # Format time in milliseconds
                        df[col] = df[col].apply(lambda x: f"{x:.2f}ms" if pd.notnull(x) and x != "N/A" else "N/A")
                    elif col in ["accuracy", "precision", "recall", "f1_score", "auc"]:
                        # Format percentages
                        df[col] = df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) and x != "N/A" else "N/A")
                    elif col.startswith("importance_"):
                        # Format feature importance
                        df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) and x != "N/A" else "N/A")
                    else:
                        # Format other numeric values
                        df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) and x != "N/A" else "N/A")
            
            # Print the table
            print(tabulate(df[display_columns], headers="keys", tablefmt="grid", showindex=False))
            
            # Print the best model based on primary metric
            if "accuracy" in df.columns:
                try:
                    # Convert accuracy to numeric, handling non-numeric values
                    numeric_accuracy = pd.to_numeric(df["accuracy"].replace("N/A", float('nan')), errors='coerce')
                    if not numeric_accuracy.isna().all():
                        best_idx = numeric_accuracy.idxmax()
                        best_version = df.loc[best_idx, "version"]
                        best_accuracy = df.loc[best_idx, "accuracy"]
                        print(f"\nBest model: {best_version} with accuracy {best_accuracy}")
                except Exception as e:
                    print(f"Could not determine best model: {str(e)}")
            elif "f1_score" in df.columns:
                try:
                    # Convert f1_score to numeric, handling non-numeric values
                    numeric_f1 = pd.to_numeric(df["f1_score"].replace("N/A", float('nan')), errors='coerce')
                    if not numeric_f1.isna().all():
                        best_idx = numeric_f1.idxmax()
                        best_version = df.loc[best_idx, "version"]
                        best_f1 = df.loc[best_idx, "f1_score"]
                        print(f"\nBest model: {best_version} with F1 score {best_f1}")
                except Exception as e:
                    print(f"Could not determine best model: {str(e)}")
        else:
            print("No metrics available to display")
    
    # Display active models from registry
    if os.path.exists(registry_file):
        try:
            with open(registry_file, 'r') as f:
                registry = json.load(f)
            
            print(f"\n{'-'*80}")
            print("Currently active models from registry:")
            print(f"{'-'*80}")
            
            if "models" in registry:
                for model_type, info in registry["models"].items():
                    if "active_version" in info:
                        print(f"{model_type}: {info['active_version']}")
        except Exception as e:
            print(f"Error reading model registry: {str(e)}")

if __name__ == "__main__":
    analyze_model_metrics()