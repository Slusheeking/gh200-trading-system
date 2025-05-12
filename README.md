# GH200 Trading System - Hybrid HFT Architecture

This project implements a high-performance trading system optimized for the NVIDIA GH200 Grace Hopper Superchip. It features a hybrid high-frequency trading (HFT) model architecture designed to efficiently process market data and generate trading signals with ultra-low latency.

## Architecture Overview

The hybrid HFT model architecture consists of three main components:

1. **Fast Path Model**: A lightweight gradient boosting decision tree model that quickly scans the entire market (10,000+ stocks) to identify potential trading opportunities.
2. **Accurate Path Model**: A more sophisticated neural network with axial attention for detailed analysis on selected candidates.
3. **Exit Optimization Model**: An LSTM-based model for optimizing exit points for active positions.

![Hybrid Architecture](docs/images/hybrid_architecture.png)

## Key Features

- **Multi-stage Filtering**: Efficiently process thousands of stocks in real-time
- **TensorRT Optimization**: Maximum inference performance on NVIDIA hardware
- **Low Latency**: Sub-millisecond end-to-end processing
- **Python Integration**: Seamless C++ and Python interoperability
- **Configurable Risk Management**: Advanced position sizing and stop-loss mechanisms
- **Real-time Monitoring**: Live performance metrics and system health

## Documentation

- [Hybrid HFT Architecture Overview](docs/HYBRID_HFT_ARCHITECTURE.md) - High-level design and components
- [Implementation Guide](docs/HYBRID_HFT_IMPLEMENTATION.md) - Detailed implementation and usage instructions
- [System Hardware Requirements](docs/SYSTEM_HARDWARE.md) - Hardware specifications and optimization

## Installation

### Prerequisites

- NVIDIA GH200 (or compatible GPU with CUDA support)
- CUDA 12.0 or higher
- TensorRT 8.6 or higher
- Ubuntu 22.04 or higher

### Dependencies Installation

```bash
./install_dependencies.sh
```

### Building the System

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Running the System

```bash
./run_hybrid_system.sh
```

Additional command-line options:

```bash
./run_hybrid_system.sh --config config/system.yaml --log-level info --mode live --data-source polygon
```

## Model Training

The system requires three different models:

```bash
# Train fast path model
python3 python/model_trainer.py --model-type fast_path --data-path data/training --output-dir models

# Train accurate path model
python3 python/model_trainer.py --model-type accurate_path --data-path data/training --output-dir models

# Train exit optimization model
python3 python/model_trainer.py --model-type exit_optimization --data-path data/training --output-dir models
```

## Directory Structure

- **config/**: Configuration files
- **data/**: Market data and training datasets
- **docs/**: Documentation files
- **include/**: C++ header files
- **models/**: Model files (ONNX, TensorRT)
- **monitoring/**: Real-time monitoring system
- **python/**: Python modules for model training and optimization
- **scripts/**: Utility scripts
- **src/**: C++ source code
- **tools/**: Development and profiling tools

## License

This project is licensed under the terms of the license included in the repository.

## Acknowledgments

- NVIDIA for GH200 architecture
- TensorRT team for acceleration libraries
- TA-Lib for technical analysis functions
