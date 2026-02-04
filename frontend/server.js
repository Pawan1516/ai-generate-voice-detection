/**
 * Simple Express Server for Frontend
 * Serves static files on port 3000
 * Proxies API requests to backend on port 8000
 */

const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files from current directory
app.use(express.static(__dirname));

// Serve index.html for root
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Fallback to index.html for SPA routing
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Start server
app.listen(PORT, () => {
    console.log(`
╔════════════════════════════════════════════════════╗
║  🎤 AI VOICE DETECTION - FRONTEND SERVER          ║
╚════════════════════════════════════════════════════╝

✅ Frontend running on:  http://localhost:${PORT}
✅ Backend API on:       http://localhost:8000

📋 Make sure backend is running:
   cd backend
   python -m uvicorn main:app --host 127.0.0.1 --port 8000

🌐 Open your browser: http://localhost:${PORT}

Press CTRL+C to stop
    `);
});
