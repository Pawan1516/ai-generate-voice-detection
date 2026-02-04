# Forced reload for model parity
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, validator
import base64
import os
import sys
import librosa
import numpy as np
import io
import logging
import time
from typing import Optional

# Universal Import Fix for Deployment
# Adds current directory to sys.path so 'import config' etc works regardless of CWD
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from config import SUPPORTED_LANGUAGES, API_KEY, CLASSIFICATION_THRESHOLD, LANGUAGE_API_KEYS
from audio_processor import AudioProcessor
from hybrid_detector import HybridVoiceDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Detection API - Hybrid Detection",
    description="Detect AI-generated voices using hybrid ML + artifact analysis for high-quality neural TTS",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize hybrid detector
audio_processor = AudioProcessor()
hybrid_detector = HybridVoiceDetector()

# Mount static files (frontend) - serve CSS, JS, etc
frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Request model
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
        allowed_formats = ['mp3', 'audio/mpeg']
        if v.lower() not in allowed_formats:
            raise ValueError(f'Audio format must be: mp3')
        return v.lower()

    @validator('audioBase64')
    def validate_base64(cls, v):
        if not v or len(v) < 100:
            raise ValueError('Invalid or empty audio data')
        try:
            # Test decode
            base64.b64decode(v)
        except Exception:
            raise ValueError('Invalid Base64 encoding')
        return v

# Response models
class SuccessResponse(BaseModel):
    status: str = "success"
    language: str
    classification: str
    confidenceScore: float
    explanation: str

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str

# API Key validation
def verify_api_key(x_api_key: Optional[str] = Header(None)):
    # Gather all valid keys
    valid_keys = [API_KEY]
    for lang_config in LANGUAGE_API_KEYS.values():
        valid_keys.append(lang_config["api_key"])
    
    # DEBUG: Print keys (remove in production)
    print(f"DEBUG: Received Key: '{x_api_key}'")
    print(f"DEBUG: Valid Keys: {valid_keys}")

    if not x_api_key or x_api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key

# Health check endpoint (not used as root serves static files)
@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "message": "AI Voice Detection API",
        "status": "running",
        "version": "1.0.0",
        "supported_languages": SUPPORTED_LANGUAGES
    }

# Main detection endpoint
@app.post("/api/voice-detection", response_model=SuccessResponse)
async def detect_voice(request: VoiceDetectionRequest):
    """
    Voice detection endpoint with latency optimization
    
    Performance target: < 2 seconds total response time
    - Audio decoding: < 0.2s
    - Feature extraction: < 1s (limited to 5 seconds of audio)
    - Model inference: < 0.1s
    """
    request_start = time.time()
    
    try:
        logger.info(f"Processing voice detection request for language: {request.language}")
        
        # Decode and Load Audio once for all processors (CRITICAL FOR RAM ON RENDER)
        load_start = time.time()
        try:
            audio_bytes = base64.b64decode(request.audioBase64)
            # Use soundfile/librosa to load from bytes once
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            logger.info(f"Audio loaded successfully: {len(y)} samples")
        except Exception as e:
            logger.error(f"Audio loading failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to load audio: {str(e)}")
        load_time = time.time() - load_start
        
        # Check for silence (using pre-loaded array)
        silence_start = time.time()
        try:
            is_silent, silence_percentage = audio_processor.detect_silence(y, silence_threshold=0.1)
            if is_silent:
                logger.warning(f"Audio rejected: {silence_percentage:.1%} silence detected")
        except Exception as e:
            logger.warning(f"Silence detection failed: {str(e)}, continuing")
        silence_time = time.time() - silence_start
        
        # Run hybrid detection (using pre-loaded array)
        inference_start = time.time()
        try:
            result = hybrid_detector.predict_hybrid(y, sr=16000)
            
            if result.get('classification') == 'ERROR':
                error_msg = result.get('error', 'Unknown error')
                logger.error(f"Hybrid detection failed: {error_msg}")
                raise HTTPException(status_code=500, detail=f"Detection failed: {error_msg}")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Hybrid detection failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
        inference_time = time.time() - inference_start
        
        # Parse hybrid results
        classification = result['classification']
        confidence_score = result['confidence']
        ml_score = result['breakdown']['ml_score']
        artifact_score = result['breakdown']['artifact_score']
        final_score = result['breakdown']['final_score']
        reasoning = result.get('reasoning', '')
        
        print(f"DEBUG: Hybrid Detection - Classification: {classification}, "
              f"Confidence: {confidence_score:.4f}, ML: {ml_score:.4f}, "
              f"Artifacts: {artifact_score:.4f}, Final: {final_score:.4f}")
        
        # Generate detailed explanation
        explanation = generate_hybrid_explanation(
            classification, ml_score, artifact_score, confidence_score, reasoning
        )
        
        # Calculate total time
        total_time = time.time() - request_start
        
        # Log performance metrics
        logger.info(
            f"Hybrid Detection complete: {classification} | "
            f"Confidence: {confidence_score:.2f} | "
            f"ML: {ml_score:.2f} | Artifacts: {artifact_score:.2f} | "
            f"Total: {total_time:.3f}s | "
            f"Decode: {decode_time:.3f}s | "
            f"Silence: {silence_time:.3f}s | "
            f"Features: {features_time:.3f}s | "
            f"Inference: {inference_time:.3f}s"
        )
        
        response = SuccessResponse(
            language=request.language,
            classification=classification,
            confidenceScore=round(confidence_score, 2),
            explanation=explanation
        )
        
        # Add timing info to logs (for debugging)
        logger.debug(f"Latency breakdown - Total: {total_time:.3f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

def generate_hybrid_explanation(
    classification: str, 
    ml_score: float, 
    artifact_score: float, 
    confidence: float,
    reasoning: str
) -> str:
    """Generate explanation for hybrid detection results"""
    
    if classification == "AI_GENERATED":
        base = f"AI-generated voice detected with {confidence:.1%} confidence. "
        details = f"ML model: {ml_score:.1%} AI probability, Artifact analysis: {artifact_score:.1%}. "
        return base + details + reasoning
    else:  # HUMAN
        base = f"Human voice detected with {confidence:.1%} confidence. "
        details = f"ML model: {ml_score:.1%} AI probability, Artifact analysis: {artifact_score:.1%}. "
        return base + details + reasoning

def generate_explanation(classification: str, features: dict, probability: float) -> str:
    """Generate human-readable explanation based on features"""
    
    explanations = []
    
    # Analyze pitch variance
    if features.get('pitch_variance', 0) < 50:
        explanations.append("low pitch variation")
    elif features.get('pitch_variance', 0) > 200:
        explanations.append("high pitch variation")
    
    # Analyze spectral features
    if features.get('spectral_centroid_mean', 0) < 1000:
        explanations.append("uniform spectral patterns")
    
    # Analyze zero crossing rate
    if features.get('zcr_mean', 0) < 0.05:
        explanations.append("smooth signal transitions")
    
    # Analyze energy variance
    if features.get('energy_variance', 0) < 0.01:
        explanations.append("consistent energy levels")
    
    if classification == "AI_GENERATED":
        base = "AI-generated characteristics detected: "
        if not explanations:
            explanations.append("synthetic voice patterns identified")
    else:
        base = "Human voice characteristics detected: "
        if not explanations:
            explanations.append("natural speech patterns identified")
    
    return base + ", ".join(explanations) + f" (confidence: {probability:.1%})"

# Error handler for HTTP exceptions
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )

# GLOBAL Error handler for ANY crash (Ensure we always return JSON)
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    import traceback
    error_trace = traceback.format_exc()
    logger.error(f"CRITIAL SERVER ERROR: {str(exc)}\n{error_trace}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": f"Server processing error: {str(exc)}"}
    )

# Wildcard route - serve index.html for root or missing routes
@app.get("/")
@app.get("/{file_path:path}")
async def serve_frontend(file_path: str = ""):
    """Unified frontend server: serves assets if they exist, otherwise index.html"""
    # Explicitly block API paths from being served as HTML (Prevents JSON parse errors)
    if file_path.startswith("api/"):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": f"API endpoint '/{file_path}' not found"}
        )
        
    if not file_path or file_path == "/":
        file_path = "index.html"
    
    full_path = os.path.join(frontend_path, file_path)
    
    if os.path.isfile(full_path):
        return FileResponse(full_path)
    
    # For SPA routing: if file doesn't exist, serve index.html
    index_file = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
        
    raise HTTPException(status_code=404, detail="Frontend assets not found")

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
