/**
 * Database configuration for GH200 Trading System Monitoring
 * Provides connections to Redis and MongoDB
 * Reads credentials from secure credentials file
 */

const redis = require('redis');
const { MongoClient } = require('mongodb');
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config({ path: path.join(__dirname, '../../.env') });

// Read secure credentials file
function readCredentials() {
  try {
    // Path to secure credentials file
    const credentialsPath = '/etc/trading-system/credentials';
    
    // Check if file exists and is readable
    if (fs.existsSync(credentialsPath)) {
      const data = fs.readFileSync(credentialsPath, 'utf8');
      const credentials = {};
      
      // Parse credentials file (format: KEY=VALUE)
      data.split('\n').forEach(line => {
        if (line.trim() && line.includes('=')) {
          const [key, value] = line.split('=');
          credentials[key.trim()] = value.trim();
        }
      });
      
      return credentials;
    }
  } catch (error) {
    console.warn('Could not read secure credentials file:', error.message);
    console.warn('Falling back to environment variables');
  }
  
  // Return empty object if file can't be read
  return {};
}

// Get credentials
const credentials = readCredentials();

// Default configuration
const config = {
  redis: {
    host: credentials.REDIS_HOST || process.env.REDIS_HOST || 'localhost',
    port: parseInt(credentials.REDIS_PORT || process.env.REDIS_PORT || '6379'),
    options: {
      retry_strategy: (options) => {
        if (options.error && options.error.code === 'ECONNREFUSED') {
          // End reconnecting on a specific error
          console.error('Redis connection refused. Please check if Redis is running.');
          return new Error('Redis server refused connection');
        }
        if (options.total_retry_time > 1000 * 60 * 60) {
          // End reconnecting after a specific timeout
          return new Error('Redis retry time exhausted');
        }
        if (options.attempt > 10) {
          // End reconnecting with built in error
          return undefined;
        }
        // Reconnect after
        return Math.min(options.attempt * 100, 3000);
      }
    }
  },
  mongodb: {
    uri: credentials.MONGODB_URI || process.env.MONGODB_URI || 'mongodb://trading_user:trading_password@localhost:27017/trading_system',
    options: {
      useNewUrlParser: true,
      useUnifiedTopology: true,
      connectTimeoutMS: 10000,
      socketTimeoutMS: 45000,
    }
  }
};

// Redis client
let redisClient = null;

// MongoDB client
let mongoClient = null;
let db = null;

/**
 * Initialize Redis connection
 * @returns {Object} Redis client
 */
function getRedisClient() {
  if (!redisClient) {
    console.log('Connecting to Redis...');
    redisClient = redis.createClient({
      host: config.redis.host,
      port: config.redis.port,
      retry_strategy: config.redis.options.retry_strategy
    });

    redisClient.on('error', (err) => {
      console.error('Redis error:', err);
    });

    redisClient.on('connect', () => {
      console.log('Connected to Redis');
    });
  }
  return redisClient;
}

/**
 * Initialize MongoDB connection
 * @returns {Promise<Object>} MongoDB database instance
 */
async function getMongoDb() {
  if (!db) {
    try {
      console.log('Connecting to MongoDB...');
      mongoClient = new MongoClient(config.mongodb.uri, config.mongodb.options);
      await mongoClient.connect();
      db = mongoClient.db();
      console.log('Connected to MongoDB');
    } catch (err) {
      console.error('MongoDB connection error:', err);
      throw err;
    }
  }
  return db;
}

/**
 * Close database connections
 */
async function closeConnections() {
  if (redisClient) {
    console.log('Closing Redis connection...');
    redisClient.quit();
    redisClient = null;
  }

  if (mongoClient) {
    console.log('Closing MongoDB connection...');
    await mongoClient.close();
    mongoClient = null;
    db = null;
  }
}

module.exports = {
  config,
  getRedisClient,
  getMongoDb,
  closeConnections
};