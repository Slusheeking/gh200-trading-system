#!/bin/bash

# Setup and Run Script for GH200 ONNX Runtime with TensorRT
# This script runs all the necessary steps in sequence

set -e  # Exit on error
echo "Starting setup for GH200 ONNX Runtime with TensorRT"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or with sudo"
  exit 1
fi

# Step 1: Install prerequisites
echo "Step 1: Installing prerequisites..."
./install_prerequisites.sh
if [ $? -ne 0 ]; then
  echo "Error: Prerequisites installation failed"
  exit 1
fi
echo "Prerequisites installation completed successfully"

# Step 2: Build Docker image
echo "Step 2: Building Docker image..."
docker-compose build
if [ $? -ne 0 ]; then
  echo "Error: Docker image build failed"
  exit 1
fi
echo "Docker image built successfully"

# Step 3: Start the container
echo "Step 3: Starting the container..."
docker-compose up -d
if [ $? -ne 0 ]; then
  echo "Error: Container startup failed"
  exit 1
fi
echo "Container started successfully"

# Step 4: Copy and run the test script
echo "Step 4: Copying test script to container..."
docker cp test_onnx_runtime.py onnxrt-tensorrt-gh200:/workspace/
if [ $? -ne 0 ]; then
  echo "Warning: Failed to copy test script to container"
else
  docker exec onnxrt-tensorrt-gh200 chmod +x /workspace/test_onnx_runtime.py
  echo "Running comprehensive test script to verify setup..."
  docker exec onnxrt-tensorrt-gh200 python3 /workspace/test_onnx_runtime.py
  if [ $? -ne 0 ]; then
    echo "Warning: Test script execution returned an error"
  else
    echo "Test script executed successfully"
  fi
fi

# Also run the basic test script
echo "Running basic test script..."
docker exec onnxrt-tensorrt-gh200 python3 /workspace/test_onnx.py

# Step 5: Show container status
echo "Step 5: Checking container status..."
docker ps | grep onnxrt-tensorrt-gh200
echo ""

echo "Setup completed successfully!"
echo "You can now use the ONNX Runtime with TensorRT on your GH200 system."
echo ""
echo "To run the inference service with your model:"
echo "  docker exec -d onnxrt-tensorrt-gh200 python3 /workspace/inference_service.py --model /workspace/models/your_model.onnx --tensorrt"
echo ""
echo "To optimize your ONNX models:"
echo "  docker exec onnxrt-tensorrt-gh200 python3 /workspace/onnx_optimizer.py --input /workspace/models/model.onnx --output /workspace/models/optimized_model.onnx --level 2"
echo ""
echo "To benchmark inference performance:"
echo "  docker exec onnxrt-tensorrt-gh200 python3 /workspace/tensorrt_provider.py --model /workspace/models/model.onnx --iterations 100"
echo ""
echo "For more information, please refer to the README.md file."
