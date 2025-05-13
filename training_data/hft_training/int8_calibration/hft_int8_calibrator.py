#!/usr/bin/env python3
"""
TensorRT INT8 Calibrator for HFT Model

This module provides a custom calibrator for TensorRT INT8 calibration,
using market data for accurate quantization of the HFT model.
"""

import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class HFTCalibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 Calibrator for HFT Model
    
    This calibrator uses real market data to calibrate the INT8 quantization
    for the HFT model, ensuring optimal performance without accuracy loss.
    """
    
    def __init__(self, calibration_data_path, batch_size=32, cache_file="calibration.cache"):
        """
        Initialize the calibrator
        
        Args:
            calibration_data_path: Path to calibration data file (.npy)
            batch_size: Batch size for calibration
            cache_file: Path to calibration cache file
        """
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
        """Return the batch size"""
        return self.batch_size
    
    def get_batch(self, names):
        """
        Get the next batch of calibration data
        
        Args:
            names: Input tensor names
            
        Returns:
            A list of device memory pointers or None if no more batches
        """
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
        """
        Read calibration cache from file
        
        Returns:
            Calibration cache or None if not available
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """
        Write calibration cache to file
        
        Args:
            cache: Calibration cache to write
        """
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# Example usage:
# calibrator = HFTCalibrator(
#     calibration_data_path="int8_calibration_5000_samples.npy",
#     batch_size=32,
#     cache_file="hft_calibration.cache"
# )
