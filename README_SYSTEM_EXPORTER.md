# GH200 Trading System Exporter

This module provides a comprehensive system exporter for the GH200 Trading System. It collects and exports various metrics from the trading system, including:

- System metrics
- Hardware metrics
- Portfolio metrics
- Trade data
- Logs
- Alerts
- Market data

The system exporter exposes these metrics through a REST API, which can be accessed locally or remotely through an ngrok tunnel.

## Features

- **System Metrics**: CPU usage, memory usage, disk usage, network usage, etc.
- **Hardware Metrics**: GPU usage, temperature, power consumption, etc.
- **Portfolio Metrics**: Account balance, equity, buying power, positions, etc.
- **Trade Data**: Active trades, historical trades, trade statistics, etc.
- **Logs**: System logs, application logs, error logs, etc.
- **Alerts**: System alerts, trading alerts, etc.
- **Market Data**: Historical price data, technical indicators, etc.

## Installation

1. Make sure you have Python 3.8+ installed
2. Install ngrok: `sudo apt install -y ngrok`
3. Install required Python packages: `pip install -r requirements.txt`

## Configuration

The system exporter uses the configuration from `settings/system.yaml`. You can modify this file to customize the behavior of the system exporter.

### API Key

The API key is used to authenticate requests to the API. It is set in the configuration file:

```yaml
api_key: "2vB4tKLZnRJTMvPr9lw46ELUTyr_qBaN5g6Eti66dN3m4LTJ"
```

### Ngrok Configuration

The ngrok tunnel is configured with the following settings:

```yaml
ngrok:
  auth_token: "2vB4mEpkOKCPryJJTqcnQZu17mU_2mHUjAc8Gp4egYp8iDVRJ"
  api_key: "2vB4tKLZnRJTMvPr9lw46ELUTyr_qBaN5g6Eti66dN3m4LTJ"
  region: "us"
```

### API Server Configuration

The API server is configured with the following settings:

```yaml
api_server:
  host: "0.0.0.0"
  port: 8000
```

## Usage

### Starting the System Exporter

There are two ways to start the system exporter:

#### 1. Direct Execution

To start the system exporter with the ngrok tunnel directly, run:

```bash
./start_exporter_api.sh
```

This will start the system exporter, API server, and ngrok tunnel in the foreground. The ngrok tunnel URL will be displayed in the console.

#### 2. Systemd Service (Recommended for 24/7 Operation)

To run the system exporter as a systemd service that will automatically start on boot and run 24/7:

1. Install the service:
   ```bash
   sudo ./install_services.sh
   ```

2. Start the service:
   ```bash
   sudo ./start_exporter_service.sh
   ```

3. View the logs and get the ngrok tunnel URL:
   ```bash
   sudo journalctl -u system-exporter-api.service -f
   ```

4. Stop the service when needed:
   ```bash
   sudo ./stop_exporter_service.sh
   ```

### Accessing the API

The API can be accessed through the ngrok tunnel URL. For example:

```
https://abcd1234.ngrok.io
```

All API endpoints require authentication with the API key. The API key should be provided in the `X-API-Key` header.

### API Endpoints

#### Health Check

```
GET /health
```

Returns the health status of the API server.

#### System Metrics

```
GET /metrics/system
```

Returns system metrics.

#### Hardware Metrics

```
GET /metrics/hardware
```

Returns hardware metrics.

#### Portfolio Metrics

```
GET /metrics/portfolio
```

Returns portfolio metrics.

#### Active Trades

```
GET /trades/active
```

Returns active trades.

#### Trade History

```
GET /trades/history
```

Returns historical trades.

Parameters:
- `limit`: Maximum number of trades to return (default: 100)
- `offset`: Offset for pagination (default: 0)
- `symbol`: Filter by symbol (optional)

#### Trade Statistics

```
GET /trades/statistics
```

Returns trade statistics.

#### Daily Profit/Loss

```
GET /trades/daily-pnl
```

Returns daily profit/loss data.

Parameters:
- `start_date`: Start date in YYYY-MM-DD format (optional)
- `end_date`: End date in YYYY-MM-DD format (optional)

#### Profit/Loss Calendar

```
GET /trades/pnl-calendar
```

Returns profit/loss calendar data.

Parameters:
- `year`: Year (e.g., 2025) (optional, defaults to current year)
- `month`: Month (1-12) (optional, if provided returns a monthly calendar, otherwise returns a yearly calendar)

#### Logs

```
GET /logs
```

Returns logs.

Parameters:
- `level`: Filter by log level (optional)
- `limit`: Maximum number of logs to return (default: 100)
- `offset`: Offset for pagination (default: 0)
- `component`: Filter by component (optional)

#### Alerts

```
GET /alerts
```

Returns alerts.

Parameters:
- `severity`: Filter by severity (optional)
- `category`: Filter by category (optional)
- `acknowledged`: Filter by acknowledged status (optional)
- `limit`: Maximum number of alerts to return (default: 100)
- `offset`: Offset for pagination (default: 0)

#### Acknowledge Alert

```
POST /alerts/{alert_id}/acknowledge
```

Acknowledges an alert.

Parameters:
- `alert_id`: Alert ID
- `user`: User who acknowledged the alert (default: "system")

#### Chart Data

```
GET /market/chart/{symbol}
```

Returns chart data for a symbol.

Parameters:
- `symbol`: Stock symbol
- `period`: Time period (default: "1y")
- `interval`: Data interval (default: "1d")
- `chart_type`: Chart type (default: "candlestick")
- `include_indicators`: Whether to include technical indicators (default: true)

#### Quote

```
GET /market/quote/{symbol}
```

Returns current quote for a symbol.

Parameters:
- `symbol`: Stock symbol

#### All Metrics

```
GET /metrics/all
```

Returns all metrics.

## Example API Requests

### Get System Metrics

```bash
curl -X GET "https://abcd1234.ngrok.io/metrics/system" \
     -H "X-API-Key: 2vB4tKLZnRJTMvPr9lw46ELUTyr_qBaN5g6Eti66dN3m4LTJ"
```

### Get Active Trades

```bash
curl -X GET "https://abcd1234.ngrok.io/trades/active" \
     -H "X-API-Key: 2vB4tKLZnRJTMvPr9lw46ELUTyr_qBaN5g6Eti66dN3m4LTJ"
```

### Get Daily Profit/Loss

```bash
curl -X GET "https://abcd1234.ngrok.io/trades/daily-pnl?start_date=2025-01-01&end_date=2025-05-13" \
     -H "X-API-Key: 2vB4tKLZnRJTMvPr9lw46ELUTyr_qBaN5g6Eti66dN3m4LTJ"
```

### Get Profit/Loss Calendar

```bash
curl -X GET "https://abcd1234.ngrok.io/trades/pnl-calendar?year=2025&month=5" \
     -H "X-API-Key: 2vB4tKLZnRJTMvPr9lw46ELUTyr_qBaN5g6Eti66dN3m4LTJ"
```

### Get Chart Data for AAPL

```bash
curl -X GET "https://abcd1234.ngrok.io/market/chart/AAPL?period=1mo&interval=1d" \
     -H "X-API-Key: 2vB4tKLZnRJTMvPr9lw46ELUTyr_qBaN5g6Eti66dN3m4LTJ"
```

## Architecture

The system exporter consists of the following components:

1. **System Exporter**: The main component that orchestrates the collection and export of metrics.
2. **Collectors**: Components that collect specific types of metrics.
   - System Metrics Collector
   - Hardware Metrics Collector
   - Log Collector
   - Notification Collector
   - Alpaca Portfolio Collector
   - Trade Metrics Collector
   - Yahoo Finance Client
3. **API Server**: A FastAPI server that exposes the metrics through a REST API.
4. **Ngrok Tunnel**: A tunnel that exposes the API server to the internet.

## Security Considerations

- The API is protected by an API key. Make sure to keep the API key secret.
- The ngrok tunnel is secured with HTTPS.
- The API server only binds to localhost by default, so it's not directly accessible from the internet.

## Troubleshooting

### API Server Not Starting

- Check if the port is already in use: `sudo lsof -i :8000`
- Check if the API server is running: `ps aux | grep uvicorn`

### Ngrok Tunnel Not Starting

- Check if ngrok is installed: `ngrok version`
- Check if the ngrok auth token is valid: `ngrok authtoken`
- Check if ngrok is already running: `ps aux | grep ngrok`

### API Requests Failing

- Check if the API server is running: `ps aux | grep uvicorn`
- Check if the ngrok tunnel is running: `ps aux | grep ngrok`
- Check if the API key is correct
- Check if the endpoint URL is correct
