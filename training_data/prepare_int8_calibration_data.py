#!/usr/bin/env python3
"""
Prepare INT8 Calibration Data for TensorRT

This script combines multiple market snapshot datasets into a single
calibration dataset for TensorRT INT8 quantization, ensuring diverse
representation across price ranges and other features.
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("int8_calibration")

def load_numpy_dataset(filepath: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load a dataset from a numpy file with its corresponding metadata
    
    Args:
        filepath: Path to the numpy (.npy) file
        
    Returns:
        Tuple of (features_array, symbols)
    """
    # Load the numpy array
    features_array = np.load(filepath)
    
    # Load the metadata to get the symbols
    meta_filepath = filepath.replace('_samples.npy', '_metadata.json')
    if os.path.exists(meta_filepath):
        with open(meta_filepath, 'r') as f:
            metadata = json.load(f)
            symbols = metadata.get('symbols', [])
    else:
        # Generate default symbols if metadata not found
        symbols = [f"UNKNOWN_{i}" for i in range(len(features_array))]
    
    return features_array, symbols

def find_datasets(directory: str, pattern: str = "*_samples.npy") -> List[str]:
    """
    Find all dataset files matching the pattern in the specified directory
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        
    Returns:
        List of matching filepaths
    """
    path = Path(directory)
    return [str(f) for f in path.glob(pattern)]

def combine_datasets(filepaths: List[str], max_samples: int = 10000) -> Tuple[np.ndarray, List[str]]:
    """
    Combine multiple datasets into a single calibration dataset
    
    Args:
        filepaths: List of dataset file paths
        max_samples: Maximum number of samples to include
        
    Returns:
        Tuple of (combined_array, combined_symbols)
    """
    all_features = []
    all_symbols = []
    feature_names = None
    
    # Load each dataset
    for filepath in filepaths:
        features_array, symbols = load_numpy_dataset(filepath)
        
        # Get stats file to log dataset characteristics
        stats_filepath = filepath.replace('_samples.npy', '_stats.json')
        if os.path.exists(stats_filepath):
            with open(stats_filepath, 'r') as f:
                stats = json.load(f)
                feature_names = stats.get('feature_names', [])
                means = stats.get('means', [])
                stds = stats.get('stds', [])
                
                # Log dataset characteristics
                logger.info(f"Dataset {os.path.basename(filepath)}: {len(symbols)} samples")
                if means and feature_names:
                    logger.info(f"  Price range: mean={means[0]:.2f}, std={stds[0]:.2f}")
                    logger.info(f"  Volume: mean={means[6]:.0f}, std={stds[6]:.0f}")
        
        all_features.append(features_array)
        all_symbols.extend(symbols)
    
    # Combine datasets
    combined_features = np.vstack(all_features)
    combined_symbols = all_symbols
    
    # Limit to max_samples if needed by randomly sampling
    if len(combined_symbols) > max_samples:
        logger.info(f"Sampling {max_samples} records from {len(combined_symbols)} total samples")
        indices = np.random.choice(len(combined_symbols), max_samples, replace=False)
        combined_features = combined_features[indices]
        combined_symbols = [combined_symbols[i] for i in indices]
    
    logger.info(f"Combined dataset: {combined_features.shape[0]} samples with {combined_features.shape[1]} features")
    
    return combined_features, combined_symbols

def save_calibration_dataset(features_array: np.ndarray, symbols: List[str], 
                           feature_names: List[str], output_dir: str,
                           filename_prefix: str = "int8_calibration") -> str:
    """
    Save the calibration dataset
    
    Args:
        features_array: Features array
        symbols: Symbol list
        feature_names: Feature names
        output_dir: Output directory
        filename_prefix: Prefix for output filenames
        
    Returns:
        Path to the saved calibration data
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filenames
    sample_count = len(symbols)
    
    # Save as NPY for TensorRT calibration
    npy_filename = f"{filename_prefix}_{sample_count}_samples.npy"
    npy_filepath = output_path / npy_filename
    np.save(npy_filepath, features_array)
    logger.info(f"Saved calibration data to {npy_filepath}")
    
    # Save as CSV for easy inspection
    csv_filename = f"{filename_prefix}_{sample_count}_samples.csv"
    csv_filepath = output_path / csv_filename
    df = pd.DataFrame(features_array, columns=feature_names)
    df["symbol"] = symbols
    df.to_csv(csv_filepath, index=False)
    logger.info(f"Saved CSV data to {csv_filepath}")
    
    # Save metadata for reference
    meta_filename = f"{filename_prefix}_{sample_count}_metadata.json"
    meta_filepath = output_path / meta_filename
    metadata = {
        "num_samples": sample_count,
        "feature_names": feature_names,
        "symbols": symbols,
        "shape": features_array.shape,
        "stats": {
            "means": features_array.mean(axis=0).tolist(),
            "stds": features_array.std(axis=0).tolist(),
            "mins": features_array.min(axis=0).tolist(),
            "maxs": features_array.max(axis=0).tolist()
        }
    }
    with open(meta_filepath, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {meta_filepath}")
    
    return str(output_path)

def create_calibrator_class_code(calibration_data_path: str, output_file: str, 
                                class_name: str = "HFTCalibrator") -> str:
    """
    Create a TensorRT calibrator class code file
    
    Args:
        calibration_data_path: Path to calibration data file
        output_file: Path to output Python file
        class_name: Name of calibrator class
        
    Returns:
        Path to the generated code file
    """
    code = f"""#!/usr/bin/env python3
\"\"\"
TensorRT INT8 Calibrator for HFT Model

This module provides a custom calibrator for TensorRT INT8 calibration,
using market data for accurate quantization of the HFT model.
\"\"\"

import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class {class_name}(trt.IInt8EntropyCalibrator2):
    \"\"\"
    INT8 Calibrator for HFT Model
    
    This calibrator uses real market data to calibrate the INT8 quantization
    for the HFT model, ensuring optimal performance without accuracy loss.
    \"\"\"
    
    def __init__(self, calibration_data_path, batch_size=32, cache_file="calibration.cache"):
        \"\"\"
        Initialize the calibrator
        
        Args:
            calibration_data_path: Path to calibration data file (.npy)
            batch_size: Batch size for calibration
            cache_file: Path to calibration cache file
        \"\"\"
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        
        # Load calibration data
        self.data = np.load(calibration_data_path)
        self.num_samples = self.data.shape[0]
        self.feature_count = self.data.shape[1]
        self.current_index = 0
        
        # Allocate device memory for a batch
        self.device_input = cuda.mem_alloc(self.batch_size * self.feature_count * np.dtype(np.float32).itemsize)
        
        # Create a stream to copy data to device
        self.stream = cuda.Stream()
    
    def get_batch_size(self):
        \"\"\"Return the batch size\"\"\"
        return self.batch_size
    
    def get_batch(self, names):
        \"\"\"
        Get the next batch of calibration data
        
        Args:
            names: Input tensor names
            
        Returns:
            A list of device memory pointers or None if no more batches
        \"\"\"
        if self.current_index + self.batch_size > self.num_samples:
            return None
        
        # Get a batch of data
        batch = self.data[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        
        # Copy data to device
        cuda.memcpy_htod_async(self.device_input, np.ascontiguousarray(batch.astype(np.float32)), self.stream)
        self.stream.synchronize()
        
        # Return pointer to device memory
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        \"\"\"
        Read calibration cache from file
        
        Returns:
            Calibration cache or None if not available
        \"\"\"
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        \"\"\"
        Write calibration cache to file
        
        Args:
            cache: Calibration cache to write
        \"\"\"
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# Example usage:
# calibrator = {class_name}(
#     calibration_data_path="{os.path.basename(calibration_data_path)}",
#     batch_size=32,
#     cache_file="hft_calibration.cache"
# )
"""
    
    with open(output_file, "w") as f:
        f.write(code)
        
    logger.info(f"Created calibrator class at {output_file}")
    return output_file

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Prepare INT8 calibration data for TensorRT"
    )
    parser.add_argument("--input-dir", default="data/hft_training/calibration", 
                      help="Directory containing training datasets")
    parser.add_argument("--output-dir", default="data/hft_training/int8_calibration", 
                      help="Directory to save calibration data")
    parser.add_argument("--max-samples", type=int, default=5000, 
                      help="Maximum number of samples to include in calibration dataset")
    parser.add_argument("--generate-code", action="store_true", 
                      help="Generate TensorRT calibrator class code")
    args = parser.parse_args()
    
    # Find datasets
    dataset_files = find_datasets(args.input_dir)
    if not dataset_files:
        logger.error(f"No datasets found in {args.input_dir}")
        return 1
    
    logger.info(f"Found {len(dataset_files)} datasets in {args.input_dir}")
    
    # Combine datasets
    features_array, symbols = combine_datasets(dataset_files, args.max_samples)
    
    # Get feature names from any stats file
    feature_names = ["last_price", "bid_price", "ask_price", "bid_ask_spread", 
                   "high_price", "low_price", "volume", "vwap", 
                   "rsi_14", "macd", "bb_upper", "bb_middle", "bb_lower", 
                   "volume_acceleration", "price_change_5m", "momentum_1m"]
    
    for file in dataset_files:
        stats_file = file.replace('_samples.npy', '_stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                if 'feature_names' in stats:
                    feature_names = stats['feature_names']
                    break
    
    # Save calibration dataset
    save_calibration_dataset(
        features_array, symbols, feature_names, args.output_dir
    )
    
    # Generate calibrator class code if requested
    if args.generate_code:
        calibration_data_path = os.path.join(args.output_dir, f"int8_calibration_{len(symbols)}_samples.npy")
        output_file = os.path.join(args.output_dir, "hft_int8_calibrator.py")
        create_calibrator_class_code(calibration_data_path, output_file)
    
    logger.info("INT8 calibration data preparation complete")
    return 0

if __name__ == "__main__":
    exit(main())