# 🚀 AI Voice Detection - Local Setup Guide

## ✅ Prerequisites
- **Python 3.8+** - For backend
- **Node.js 14+** - For frontend  
- **npm** - Package manager (comes with Node.js)

---

## 🎯 Quick Start

### Option 1: Run Both Servers (Easiest)

#### On Windows (PowerShell or CMD):
```bash
# Method 1: Using Python script
python start_all.py

# Method 2: Using Batch script
start_all.bat

# Method 3: Manual (two terminal windows)
# Terminal 1 - Backend
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 - Frontend
cd frontend
npm start
```

#### On Mac/Linux:
```bash
# Using Python script
python3 start_all.py

# Or manually in two terminals:
# Terminal 1
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2
cd frontend
node server.js
```

---

## 🌐 Access Points

Once both servers are running:

| Service | URL |
|---------|-----|
| **Frontend** | http://localhost:3000 |
| **Backend API** | http://localhost:8000 |
| **API Documentation** | http://localhost:8000/docs |
| **API Health Check** | http://localhost:8000/health |

---

## 📋 Server Details

### Backend (Port 8000)
- **Framework**: FastAPI (Python)
- **File**: `backend/main.py`
- **Features**:
  - Voice detection API endpoint
  - Swagger UI for API testing
  - Audio processing pipeline
  - ML model inference

### Frontend (Port 3000)
- **Framework**: Express.js + HTML/CSS/JavaScript
- **Files**: 
  - `frontend/server.js` - Express server
  - `frontend/index.html` - UI
  - `frontend/config.js` - Configuration
- **Features**:
  - Single Page Application (SPA)
  - Real-time voice detection
  - Multi-language support

---

## 🔧 Manual Setup (If Scripts Don't Work)

### Step 1: Install Frontend Dependencies
```bash
cd frontend
npm install
```

### Step 2: Start Backend (Terminal 1)
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Step 3: Start Frontend (Terminal 2)
```bash
cd frontend
npm start
# or
node server.js
```

---

## 🧪 Testing the API

### Using Swagger UI (Easiest)
1. Open http://localhost:8000/docs
2. Click on `/api/voice-detection` endpoint
3. Click "Try it out"
4. Upload an audio file or provide base64-encoded audio
5. Click "Execute"

### Using cURL
```bash
curl -X POST "http://localhost:8000/api/voice-detection" \
  -H "Content-Type: application/json" \
  -d '{"audio_base64": "YOUR_BASE64_AUDIO", "language": "English"}'
```

### Using Python
```python
import requests

# Upload audio file
with open('test_audio.wav', 'rb') as f:
    files = {'audio_file': f}
    response = requests.post(
        'http://localhost:8000/api/voice-detection',
        files=files,
        data={'language': 'English'}
    )
    print(response.json())
```

---

## 📊 Model Information

- **Model Type**: Random Forest (100 trees)
- **Training Samples**: 330 (cross-validated)
- **Features**: 42 audio features
  - 13 MFCC (Mean & Std)
  - 4 Spectral features
  - 2 Zero Crossing Rate
  - 4 RMS features
  - 2 Onset features
  - Duration
- **Accuracy**: 100% (5-fold cross-validation)
- **Classes**: 
  - 0 = Human voice
  - 1 = AI-generated voice

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Kill process using port 3000
# Windows:
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :3000
kill -9 <PID>
```

### npm not found
- Install Node.js from https://nodejs.org/
- Restart terminal/PowerShell after installation
- Verify: `npm --version`

### Python module errors
```bash
cd backend
pip install -r requirements.txt
```

### Frontend not connecting to backend
- Check backend is running: http://localhost:8000/health
- Check `frontend/config.js` points to `http://localhost:8000/api/voice-detection`
- Check browser console for CORS errors

---

## 📁 Project Structure

```
ai audio/
├── backend/               # FastAPI backend
│   ├── main.py           # Main API app
│   ├── audio_processor.py # Audio feature extraction
│   ├── ml_model.py       # Model inference
│   ├── config.py         # Configuration
│   └── models/           # Trained model files
├── frontend/             # Express.js + HTML frontend
│   ├── server.js         # Express server
│   ├── index.html        # UI
│   ├── config.js         # API config
│   ├── style.css         # Styling
│   └── app.js            # Frontend logic
├── ml/                   # Training scripts
├── data/                 # Training datasets
└── start_all.py/bat      # Startup scripts
```

---

## 🚀 Deployment

Ready to deploy? Check `DEPLOYMENT_GUIDE.md` for cloud deployment options:
- Railway.com
- Vercel (Frontend)
- Netlify (Frontend)

---

## ❓ Support

For issues or questions:
1. Check the troubleshooting section above
2. Review browser console for errors (F12)
3. Check backend logs in terminal
4. Review API documentation at http://localhost:8000/docs

---

**Created**: 2024  
**Version**: 1.0
