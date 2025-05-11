/**
 * Metrics collector for GH200 Trading System
 * Collects metrics from the trading system and stores them in Redis and MongoDB
 */

const fs = require('fs');
const path = require('path');
const metricsBuffer = require('../shared/metrics_buffer');
const config = require('../shared/config');
const dbConfig = require('../shared/db_config');

// Metrics file path (shared with trading system)
const METRICS_FILE = '/tmp/trading_metrics.json';

// Read metrics from file
function readMetricsFile() {
    try {
        if (fs.existsSync(METRICS_FILE)) {
            const data = fs.readFileSync(METRICS_FILE, 'utf8');
            return JSON.parse(data);
        }
    } catch (error) {
        console.error('Error reading metrics file:', error);
    }
    
    return null;
}

// Store metrics in Redis
async function storeMetricsInRedis(metrics) {
    try {
        const redisClient = dbConfig.getRedisClient();
        
        // Store current metrics
        await redisClient.set('current_metrics', JSON.stringify(metrics));
        
        // Add to recent metrics list (keep last 100)
        await redisClient.lPush('recent_metrics', JSON.stringify(metrics));
        await redisClient.lTrim('recent_metrics', 0, 99);
        
        return true;
    } catch (error) {
        console.error('Error storing metrics in Redis:', error);
        return false;
    }
}

// Store metrics in MongoDB
async function storeMetricsInMongoDB(metrics) {
    try {
        // Only store metrics every minute to avoid overwhelming the database
        if (metrics.timestamp % 60000 < 1000) {
            const db = await dbConfig.getMongoDb();
            const metricsCollection = db.collection('metrics');
            
            await metricsCollection.insertOne({
                ...metrics,
                created_at: new Date()
            });
        }
        
        return true;
    } catch (error) {
        console.error('Error storing metrics in MongoDB:', error);
        return false;
    }
}

// Collect metrics
async function collectMetrics() {
    // Read metrics from file
    const metrics = readMetricsFile();
    
    if (metrics) {
        // Update metrics buffer (for backward compatibility)
        metricsBuffer.update(metrics);
        
        // Store in Redis for real-time access
        await storeMetricsInRedis(metrics);
        
        // Store in MongoDB for historical analysis
        await storeMetricsInMongoDB(metrics);
    }
}

// Start metrics collection
function startCollection() {
    console.log('Starting metrics collection...');
    
    // Collect metrics periodically
    setInterval(collectMetrics, config.intervals.metricsCollection);
}

// Export functions
module.exports = {
    startCollection,
    readMetricsFile,
    storeMetricsInRedis,
    storeMetricsInMongoDB
};

// Start collection if run directly
if (require.main === module) {
    startCollection();
}