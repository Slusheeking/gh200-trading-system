/**
 * Shared configuration for monitoring components
 */

// Load environment variables
require('dotenv').config({ path: '../../.env' });

// Default configuration
const config = {
    // Server ports
    ports: {
        metricsApi: process.env.METRICS_API_PORT || 8000,
        dashboard: process.env.DASHBOARD_PORT || 3000
    },
    
    // Update intervals (ms)
    intervals: {
        metricsCollection: 100,  // How often to collect metrics
        clientBroadcast: 1000    // How often to broadcast to clients
    },
    
    // Metrics buffer configuration
    metricsBuffer: {
        maxSize: 10000,          // Maximum number of data points to store
        flushInterval: 60000     // How often to flush old data (ms)
    },
    
    // Database configuration
    database: {
        redis: {
            host: process.env.REDIS_HOST || 'localhost',
            port: parseInt(process.env.REDIS_PORT || '6379')
        },
        mongodb: {
            uri: process.env.MONGODB_URI || 'mongodb://trading_user:trading_password@localhost:27017/trading_system',
            options: {
                useNewUrlParser: true,
                useUnifiedTopology: true
            }
        }
    },
    
    // Ngrok configuration
    ngrok: {
        authToken: process.env.NGROK_AUTH_TOKEN,
        apiKey: process.env.NGROK_API_KEY
    },
    
    // Logging configuration
    logging: {
        level: process.env.LOG_LEVEL || 'info',
        file: '../logs/monitoring.log'
    }
};

module.exports = config;