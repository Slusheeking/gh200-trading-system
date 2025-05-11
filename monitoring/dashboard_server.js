/**
 * Dashboard Server
 * Serves the dashboard static files on port 3000
 */

const express = require('express');
const path = require('path');
const config = require('./shared/config');

// Create Express app
const app = express();

// Serve static files from the dashboard directory
app.use(express.static(path.join(__dirname, 'dashboard')));

// Start server
const PORT = config.ports.dashboard;
app.listen(PORT, () => {
    console.log(`Dashboard server running on port ${PORT}`);
});