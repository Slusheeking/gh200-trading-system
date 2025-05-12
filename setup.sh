#!/bin/bash

# Set up environment for optimal performance on GH200
echo "Setting up environment for ONNX Runtime with TensorRT on GH200"

# Create cache directories
mkdir -p /workspace/trt_cache
chmod 777 /workspace/trt_cache

# Set environment variables for TensorRT optimization
export ORT_TENSORRT_MAX_WORKSPACE_SIZE=4294967296  # 4GB workspace
export ORT_TENSORRT_FP16_ENABLE=1
export ORT_TENSORRT_ENGINE_CACHE_ENABLE=1
export ORT_TENSORRT_ENGINE_CACHE_PATH=/workspace/trt_cache

# Set environment variables for ARM64 optimization
export GOMP_CPU_AFFINITY="0-71"  # Set affinity for all Grace CPU cores
export OMP_NUM_THREADS=72
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Optimize GPU settings
nvidia-smi -i 0 -ac 1815,1815

# Check ONNX Runtime availability
python3 -c "import onnxruntime as ort; print('Available providers:', ort.get_available_providers())"

echo "Environment setup complete"
