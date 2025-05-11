/**
 * real_metrics.js - Generate real metrics from the trading system
 * 
 * This script collects real performance metrics from the trading system
 * and writes them to the metrics file for the dashboard to display.
 */

const fs = require('fs');
const path = require('path');
const { promisify } = require('util');
const exec = promisify(require('child_process').exec);
const config = require('./shared/config');

// Define metrics file path - must match the path in metrics_collector.js
const metricsFilePath = '/tmp/trading_metrics.json';

// Configuration
const METRICS_INTERVAL = 1000; // 1 second
const LOG_FILE = path.join(__dirname, '..', 'logs', 'trading_system.log');
const POSITION_FILE = path.join(__dirname, '..', 'logs', 'positions.json');

// Initialize metrics object
let metrics = {
  system: {
    cpu_usage: 0,
    memory_usage: 0,
    gpu_usage: 0
  },
  latency: {
    data_ingestion: 0,
    ml_inference: 0,
    risk_check: 0,
    execution: 0,
    end_to_end: 0
  },
  trading: {
    positions: 0,
    signals: 0,
    trades: 0,
    pnl: 0
  },
  timestamp: Date.now()
};

/**
 * Get system metrics using system commands
 */
async function getSystemMetrics() {
  try {
    // Get CPU usage
    const { stdout: cpuStdout } = await exec("top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4}'");
    metrics.system.cpu_usage = parseFloat(cpuStdout.trim());

    // Get memory usage (in MB)
    const { stdout: memStdout } = await exec("free -m | grep Mem | awk '{print $3}'");
    metrics.system.memory_usage = parseFloat(memStdout.trim());

    // Get GPU usage if available (using nvidia-smi if available)
    try {
      const { stdout: gpuStdout } = await exec("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits");
      metrics.system.gpu_usage = parseFloat(gpuStdout.trim());
    } catch (error) {
      // If nvidia-smi fails, use a placeholder value or previous value
      metrics.system.gpu_usage = metrics.system.gpu_usage || 0;
    }
  } catch (error) {
    console.error('Error getting system metrics:', error.message);
  }
}

/**
 * Parse trading system log to extract latency metrics
 */
function parseLogForLatencyMetrics() {
  try {
    if (!fs.existsSync(LOG_FILE)) {
      console.log(`Log file not found: ${LOG_FILE}`);
      return;
    }

    // Read the last 100 lines of the log file
    const logContent = fs.readFileSync(LOG_FILE, 'utf8');
    const lines = logContent.split('\n').slice(-100);

    // Extract latency metrics from log lines
    let latencyMetrics = {
      data_ingestion: [],
      ml_inference: [],
      risk_check: [],
      execution: [],
      end_to_end: []
    };

    for (const line of lines) {
      if (line.includes('LATENCY')) {
        if (line.includes('data_ingestion')) {
          const match = line.match(/data_ingestion: (\d+\.\d+)/);
          if (match) latencyMetrics.data_ingestion.push(parseFloat(match[1]));
        } else if (line.includes('ml_inference')) {
          const match = line.match(/ml_inference: (\d+\.\d+)/);
          if (match) latencyMetrics.ml_inference.push(parseFloat(match[1]));
        } else if (line.includes('risk_check')) {
          const match = line.match(/risk_check: (\d+\.\d+)/);
          if (match) latencyMetrics.risk_check.push(parseFloat(match[1]));
        } else if (line.includes('execution')) {
          const match = line.match(/execution: (\d+\.\d+)/);
          if (match) latencyMetrics.execution.push(parseFloat(match[1]));
        } else if (line.includes('end_to_end')) {
          const match = line.match(/end_to_end: (\d+\.\d+)/);
          if (match) latencyMetrics.end_to_end.push(parseFloat(match[1]));
        }
      }
    }

    // Calculate average latency for each metric
    for (const key in latencyMetrics) {
      const values = latencyMetrics[key];
      if (values.length > 0) {
        const sum = values.reduce((a, b) => a + b, 0);
        metrics.latency[key] = sum / values.length;
      }
    }
  } catch (error) {
    console.error('Error parsing log for latency metrics:', error.message);
  }
}

/**
 * Get trading metrics from position file or logs
 */
function getTradingMetrics() {
  try {
    // Try to read positions from position file
    if (fs.existsSync(POSITION_FILE)) {
      const positionData = JSON.parse(fs.readFileSync(POSITION_FILE, 'utf8'));
      metrics.trading.positions = positionData.positions ? positionData.positions.length : 0;
      metrics.trading.pnl = positionData.total_pnl || 0;
    }

    // Parse log for trading signals and trades
    if (fs.existsSync(LOG_FILE)) {
      const logContent = fs.readFileSync(LOG_FILE, 'utf8');
      const lines = logContent.split('\n').slice(-200);

      // Count signals and trades in recent log entries
      let signalCount = 0;
      let tradeCount = 0;

      const lastMinuteTimestamp = Date.now() - 60000; // Last minute
      
      for (const line of lines) {
        const match = line.match(/\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]/);
        if (match) {
          const lineTimestamp = new Date(match[1]).getTime();
          if (lineTimestamp > lastMinuteTimestamp) {
            if (line.includes('SIGNAL')) signalCount++;
            if (line.includes('TRADE')) tradeCount++;
          }
        }
      }

      metrics.trading.signals = signalCount;
      metrics.trading.trades = tradeCount;
    }
  } catch (error) {
    console.error('Error getting trading metrics:', error.message);
  }
}

/**
 * Write metrics to file
 */
function writeMetricsToFile() {
  try {
    metrics.timestamp = Date.now();
    fs.writeFileSync(metricsFilePath, JSON.stringify(metrics));
    console.log('Metrics written to file:', metrics);
  } catch (error) {
    console.error('Error writing metrics to file:', error.message);
  }
}

/**
 * Main function to collect and write metrics
 */
async function collectAndWriteMetrics() {
  try {
    await getSystemMetrics();
    parseLogForLatencyMetrics();
    getTradingMetrics();
    writeMetricsToFile();
  } catch (error) {
    console.error('Error in metrics collection:', error.message);
  }
}

// Run metrics collection at regular intervals
console.log('Starting real metrics collection...');
collectAndWriteMetrics(); // Run immediately
setInterval(collectAndWriteMetrics, METRICS_INTERVAL);

// Handle process termination
process.on('SIGINT', () => {
  console.log('Stopping real metrics collection...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('Stopping real metrics collection...');
  process.exit(0);
});