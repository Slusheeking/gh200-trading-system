/**
 * Shared metrics buffer for monitoring components
 * Provides a circular buffer for storing time-series metrics
 */

const config = require('./config');

class MetricsBuffer {
    constructor(maxSize = config.metricsBuffer.maxSize) {
        this.buffer = [];
        this.maxSize = maxSize;
        this.currentIndex = 0;
        
        // Initialize with empty metrics
        this.currentMetrics = {
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
    }
    
    // Update current metrics
    update(metrics) {
        // Update system metrics if provided
        if (metrics.system) {
            Object.assign(this.currentMetrics.system, metrics.system);
        }
        
        // Update latency metrics if provided
        if (metrics.latency) {
            Object.assign(this.currentMetrics.latency, metrics.latency);
        }
        
        // Update trading metrics if provided
        if (metrics.trading) {
            Object.assign(this.currentMetrics.trading, metrics.trading);
        }
        
        // Update timestamp
        this.currentMetrics.timestamp = Date.now();
        
        // Add to buffer
        this.addToBuffer({...this.currentMetrics});
        
        return this.currentMetrics;
    }
    
    // Add metrics to buffer
    addToBuffer(metrics) {
        if (this.buffer.length < this.maxSize) {
            this.buffer.push(metrics);
        } else {
            this.buffer[this.currentIndex] = metrics;
            this.currentIndex = (this.currentIndex + 1) % this.maxSize;
        }
    }
    
    // Get current metrics
    getCurrent() {
        return {...this.currentMetrics};
    }
    
    // Get historical metrics
    getHistory(count = 100) {
        if (this.buffer.length <= count) {
            return [...this.buffer];
        }
        
        // Calculate start index
        const startIndex = (this.currentIndex - count + this.maxSize) % this.maxSize;
        
        // Collect metrics
        const result = [];
        for (let i = 0; i < count; i++) {
            const index = (startIndex + i) % this.maxSize;
            if (this.buffer[index]) {
                result.push(this.buffer[index]);
            }
        }
        
        return result;
    }
    
    // Clear buffer
    clear() {
        this.buffer = [];
        this.currentIndex = 0;
    }
}

// Create singleton instance
const metricsBuffer = new MetricsBuffer();

module.exports = metricsBuffer;