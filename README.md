# GH200 Trading System

A high-performance, low-latency trading system designed for the NVIDIA GH200 Grace Hopper Superchip, optimized for AI-driven algorithmic trading.

## Overview

This trading system leverages the computational power of NVIDIA's GH200 Grace Hopper Superchip to perform real-time market data processing, feature extraction, ML inference, risk management, and trade execution with ultra-low latency.

## Key Features

- **High-Performance Architecture**: Optimized for the GH200 Grace Hopper Superchip
- **Real-Time Market Data Processing**: Efficient WebSocket client for market data ingestion
- **CUDA-Accelerated Feature Extraction**: Parallel processing of market data
- **ML-Driven Trading Signals**: Real-time inference using PyTorch models
- **Risk Management**: Position tracking and risk controls
- **Low-Latency Execution**: Optimized order execution engine
- **Real-Time Monitoring**: Dashboard for system metrics and performance

## System Components

- **Data Ingestion**: WebSocket client for market data feeds
- **Feature Extraction**: CUDA-accelerated feature calculation
- **ML Inference**: PyTorch models for signal generation
- **Risk Management**: Position tracking and risk controls
- **Execution Engine**: Order management and execution
- **Monitoring Dashboard**: Real-time system metrics and performance visualization

## Monitoring Dashboard

The system includes a real-time monitoring dashboard that displays:

- System metrics (CPU, memory, GPU usage)
- Latency metrics for each component
- Trading metrics (positions, signals, trades, P&L)

Access the dashboard at http://localhost:3000 when the system is running.

## Getting Started

### Prerequisites

- NVIDIA GH200 Grace Hopper Superchip or compatible hardware
- CUDA Toolkit 12.0+
- CMake 3.20+
- Node.js 18+ (for monitoring dashboard)
- MongoDB
- Redis

### Building the System

```bash
# Clone the repository
git clone https://github.com/slusheeking/gh200-trading-system.git
cd gh200-trading-system

# Build the system
cmake .
make
```

### Running the System

```bash
# Start the system
./start.sh

# Stop the system
./stop.sh
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
