# GH200 Trading System Monitoring

This directory contains the monitoring system for the GH200 Trading System. The monitoring system provides real-time metrics visualization and alerting for the trading system.

## Architecture

The monitoring system consists of the following components:

1. **Metrics Publisher** - A C++ component in the trading system that publishes metrics to a shared file (`/tmp/trading_metrics.json`).
2. **Metrics API** - A Node.js server that reads metrics from the shared file and provides them via HTTP API and WebSocket.
3. **Dashboard** - A web-based dashboard that visualizes the metrics in real-time.

## Directory Structure

```
monitoring/
├── api/                  # Metrics API server (Port 8000)
│   ├── metrics_api.js    # API server implementation
│   └── metrics_collector.js # Collects metrics from trading system
│
├── dashboard/            # Dashboard UI (Port 3000)
│   ├── index.html        # Main dashboard page
│   ├── css/              # CSS styles
│   ├── js/               # JavaScript files
│   └── assets/           # Static assets
│
├── shared/               # Shared utilities
│   ├── metrics_buffer.js # Shared metrics buffer
│   └── config.js         # Monitoring configuration
│
└── scripts/              # Utility scripts
    ├── start_monitoring.sh # Start monitoring services
    ├── stop_monitoring.sh  # Stop monitoring services
    └── setup_tunnel.sh     # Set up SSH/ngrok tunnel
```

## Metrics

The monitoring system tracks the following metrics:

### System Metrics
- CPU Usage (%)
- Memory Usage (MB)
- GPU Usage (%)

### Latency Metrics
- Data Ingestion Latency (ms)
- ML Inference Latency (ms)
- Risk Check Latency (ms)
- Execution Latency (ms)
- End-to-End Latency (ms)

### Trading Metrics
- Active Positions
- Trading Signals
- Executed Trades
- Profit and Loss (PnL)

## Setup and Usage

### Prerequisites

- Node.js 14+
- npm

### Installation

1. Install dependencies:
   ```
   cd monitoring
   npm install
   ```

### Starting the Monitoring System

To start the monitoring system:

```
cd monitoring/scripts
./start_monitoring.sh
```

This will start:
- Metrics API server on port 8000
- Dashboard server on port 3000

### Stopping the Monitoring System

To stop the monitoring system:

```
cd monitoring/scripts
./stop_monitoring.sh
```

### Accessing the Dashboard

The dashboard is available at:
- Local: http://localhost:3000
- Remote: Configured via ngrok tunnel (see below)

### Setting Up Remote Access

To set up remote access via ngrok:

```
cd monitoring/scripts
./setup_tunnel.sh
```

This will create secure tunnels to the Metrics API and Dashboard servers.

## Production Deployment

In production, the monitoring system runs alongside the trading system. The metrics are published by the trading system's C++ component (`metrics_publisher.cpp`) to a shared file, which is then read by the monitoring system.

The monitoring system is designed to have minimal impact on the trading system's performance, with the metrics collection happening asynchronously.

## Troubleshooting

If the dashboard is not showing any metrics:

1. Check if the trading system is running and publishing metrics
2. Verify that the Metrics API server is running (`ps aux | grep metrics_api`)
3. Check the logs in the `monitoring/logs` directory