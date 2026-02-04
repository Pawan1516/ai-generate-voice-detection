"""
FIXED FastAPI Backend with proper inference
Uses VoiceClassifierFixed with:
✅ Threshold 0.65
✅ Spectral flatness safety override
✅ Consistent preprocessing
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
import base64
import io
import logging
import time
import sys
import os
from typing import Optional

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SUPPORTED_LANGUAGES, API_KEY, LANGUAGE_API_KEYS
from audio_processor_fixed import AudioProcessorFixed
from inference_fixed import VoiceClassifierFixed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Detection API - FIXED",
    description="Detect AI-generated voices vs human voices with corrected preprocessing",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
audio_processor = AudioProcessorFixed()
voice_classifier = VoiceClassifierFixed()

# Mount static files (frontend)
frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


# Request/Response Models
class VoiceDetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

    @validator('language')
    def validate_language(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f'Language must be one of: {", ".join(SUPPORTED_LANGUAGES)}')
        return v

    @validator('audioFormat')
    def validate_format(cls, v):
        allowed_formats = ['wav', 'mp3', 'audio/wav', 'audio/mpeg']
        if v.lower() not in allowed_formats:
            raise ValueError(f'Audio format must be one of: wav, mp3')
        return v.lower()

    @validator('audioBase64')
    def validate_base64(cls, v):
        if not v or len(v) < 100:
            raise ValueError('Invalid or empty audio data')
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError('Invalid Base64 encoding')
        return v


class SuccessResponse(BaseModel):
    status: str = "success"
    language: str
    classification: str
    confidenceScore: float
    method: str
    spectral_flatness: float = 0.0
    

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str


# API Key validation
def verify_api_key(x_api_key: Optional[str] = Header(None)):
    valid_keys = [API_KEY]
    for lang_config in LANGUAGE_API_KEYS.values():
        valid_keys.append(lang_config["api_key"])

    if not x_api_key or x_api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key


# Endpoints
@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "message": "AI Voice Detection API (FIXED)",
        "status": "running",
        "version": "2.0.0",
        "supported_languages": SUPPORTED_LANGUAGES,
        "preprocessing": "LOCKED librosa settings",
        "threshold": 0.65,
        "safety_override": "Spectral flatness < 0.02 = HUMAN"
    }


@app.post("/api/voice-detection", response_model=SuccessResponse)
async def detect_voice(request: VoiceDetectionRequest):
    """
    ✅ FIXED Voice detection endpoint
    
    Features:
    - Consistent preprocessing (Step 1-2)
    - Correct label mapping (Step 3)
    - Threshold 0.65 (Step 4)
    - Spectral flatness safety override (Step 6)
    """
    request_start = time.time()
    
    try:
        logger.info(f"Processing voice detection request for language: {request.language}")
        
        # Decode Base64 audio
        try:
            audio_bytes = base64.b64decode(request.audioBase64)
        except Exception as e:
            logger.error(f"Base64 decoding failed: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid Base64 audio data")
        
        # Validate audio size
        if len(audio_bytes) < 1000:
            raise HTTPException(status_code=400, detail="Audio file too small")
        
        if len(audio_bytes) > 15 * 1024 * 1024:  # 15MB limit
            raise HTTPException(status_code=400, detail="Audio file too large")
        
        # ✅ Use FIXED inference with safety overrides
        result = voice_classifier.predict_with_safety(audio_bytes)
        
        if result.get('classification') == 'ERROR':
            raise HTTPException(status_code=500, detail=result.get('error'))
        
        # Extract results
        classification = result['classification']
        confidence = result['confidence']
        method = result.get('method', 'unknown')
        spectral_flatness = result.get('spectral_flatness', 0.0)
        
        # Log results
        total_time = time.time() - request_start
        logger.info(
            f"✅ Detection: {classification} | "
            f"Confidence: {confidence:.4f} | "
            f"Method: {method} | "
            f"Time: {total_time:.3f}s"
        )
        
        return SuccessResponse(
            language=request.language,
            classification=classification,
            confidenceScore=round(confidence, 4),
            method=method,
            spectral_flatness=round(spectral_flatness, 4)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/")
async def root():
    """Serve frontend index.html"""
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'index.html')
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Frontend not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
