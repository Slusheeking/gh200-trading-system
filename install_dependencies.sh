#!/bin/bash
# Install dependencies for the hybrid HFT model architecture

set -e  # Exit on error

echo "Installing dependencies for hybrid HFT model architecture..."

# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    libcurl4-openssl-dev \
    libssl-dev \
    libboost-all-dev \
    libtbb-dev \
    libyaml-cpp-dev \
    nlohmann-json3-dev \
    libspdlog-dev \
    libpython3-dev

# Install Python dependencies
pip3 install --upgrade pip
pip3 install \
    numpy \
    pandas \
    torch \
    onnx \
    onnxruntime-gpu \
    lightgbm \
    scikit-learn \
    flask \
    pybind11 \
    matplotlib \
    tensorboard

# Install TensorRT if not already installed
if ! python3 -c "import tensorrt" &> /dev/null; then
    echo "TensorRT not found, installing..."
    # Check if we're on ARM64 (GH200)
    if [ "$(uname -m)" = "aarch64" ]; then
        # Install TensorRT for ARM64
        pip3 install --extra-index-url https://pypi.nvidia.com tensorrt
    else
        # Install TensorRT for x86_64
        pip3 install --extra-index-url https://pypi.nvidia.com tensorrt
    fi
fi

# Install pybind11 for C++
if [ ! -d "pybind11" ]; then
    echo "Cloning pybind11..."
    git clone https://github.com/pybind/pybind11.git
    cd pybind11
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
    sudo make install
    cd ../..
fi

# Create directories
mkdir -p models/trt_cache
mkdir -p data/training
mkdir -p data/market_snapshots

echo "Dependencies installed successfully!"
echo "You can now build the system with:"
echo "  mkdir -p build && cd build && cmake .. && make -j$(nproc)"