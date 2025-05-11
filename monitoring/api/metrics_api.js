// Metrics API server
// Serves metrics data via HTTP API on port 8000

const express = require('express');
const cors = require('cors');
const http = require('http');
const socketIo = require('socket.io');
const metricsCollector = require('./metrics_collector');
const metricsBuffer = require('../shared/metrics_buffer');
const dbConfig = require('../shared/db_config');
const config = require('../shared/config');

// Create Express app
const app = express();
app.use(cors());
app.use(express.json());

// Create HTTP server
const server = http.createServer(app);

// Create Socket.IO server
const io = socketIo(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

// Get current metrics from Redis
async function getCurrentMetrics() {
    try {
        const redisClient = dbConfig.getRedisClient();
        const data = await redisClient.get('current_metrics');
        
        if (data) {
            return JSON.parse(data);
        }
        
        // Fallback to buffer if Redis fails
        return metricsBuffer.getCurrent();
    } catch (error) {
        console.error('Error getting current metrics from Redis:', error);
        // Fallback to buffer
        return metricsBuffer.getCurrent();
    }
}

// Get historical metrics from Redis
async function getHistoricalMetrics(count = 100) {
    try {
        const redisClient = dbConfig.getRedisClient();
        const data = await redisClient.lRange('recent_metrics', 0, count - 1);
        
        if (data && data.length > 0) {
            return data.map(item => JSON.parse(item));
        }
        
        // Fallback to buffer if Redis fails
        return metricsBuffer.getHistory(count);
    } catch (error) {
        console.error('Error getting historical metrics from Redis:', error);
        // Fallback to buffer
        return metricsBuffer.getHistory(count);
    }
}

// Get metrics from MongoDB for a specific time range
async function getMetricsFromMongoDB(startTime, endTime, limit = 1000) {
    try {
        const db = await dbConfig.getMongoDb();
        const metricsCollection = db.collection('metrics');
        
        const query = {};
        if (startTime) {
            query.timestamp = { $gte: parseInt(startTime) };
        }
        if (endTime) {
            query.timestamp = { ...query.timestamp, $lte: parseInt(endTime) };
        }
        
        return await metricsCollection.find(query)
            .sort({ timestamp: -1 })
            .limit(limit)
            .toArray();
    } catch (error) {
        console.error('Error getting metrics from MongoDB:', error);
        return [];
    }
}

// API routes
app.get('/api/metrics/current', async (req, res) => {
    const metrics = await getCurrentMetrics();
    res.json(metrics);
});

app.get('/api/metrics/history', async (req, res) => {
    const count = req.query.count ? parseInt(req.query.count) : 100;
    const metrics = await getHistoricalMetrics(count);
    res.json(metrics);
});

app.get('/api/metrics/range', async (req, res) => {
    const { start, end, limit } = req.query;
    const metrics = await getMetricsFromMongoDB(start, end, limit ? parseInt(limit) : 1000);
    res.json(metrics);
});

app.get('/api/status', (req, res) => {
    res.json({
        status: 'ok',
        timestamp: Date.now(),
        services: {
            redis: !!dbConfig.getRedisClient(),
            mongodb: !!dbConfig.getMongoDb()
        }
    });
});

// Socket.IO connection
io.on('connection', async (socket) => {
    console.log('Client connected');
    
    // Send current metrics on connection
    const currentMetrics = await getCurrentMetrics();
    socket.emit('metrics', currentMetrics);
    
    // Handle disconnect
    socket.on('disconnect', () => {
        console.log('Client disconnected');
    });
});

// Broadcast metrics to clients
async function broadcastMetrics() {
    const metrics = await getCurrentMetrics();
    io.emit('metrics', metrics);
}

// Start server
const PORT = config.ports.metricsApi;
server.listen(PORT, () => {
    console.log(`Metrics API server running on port ${PORT}`);
    
    // Start metrics collection
    metricsCollector.startCollection();
    
    // Start broadcasting metrics to clients
    setInterval(broadcastMetrics, config.intervals.clientBroadcast);
});