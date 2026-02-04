import os
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
except ImportError:
    # dotenv not installed, using defaults
    pass

# API Configuration
# General API credentials
CLIENT_ID = os.getenv("CLIENT_ID", "mdc_57097cbe5d1584a7a133b636c51e5e4f")
API_KEY = os.getenv("API_KEY", "ff568c59718453585c4eb6507283633e318793cb9213adab671c87318070a951")

# Language-specific API credentials
LANGUAGE_API_KEYS = {
    "Telugu": {
        "client_id": os.getenv("TELUGU_CLIENT_ID", "mdc_0ee72f87271d16e307501a80e522ea64"),
        "api_key": os.getenv("TELUGU_API_KEY", "96244c2f911b90550b708dfc1023bd1c4bd0eca9a1826b5026f01427ce269cac")
    }
}

def get_api_key_for_language(language):
    """Get API key for specific language, fallback to general API key"""
    if language in LANGUAGE_API_KEYS:
        return LANGUAGE_API_KEYS[language]["api_key"]
    return API_KEY

def get_client_id_for_language(language):
    """Get client ID for specific language, fallback to general client ID"""
    if language in LANGUAGE_API_KEYS:
        return LANGUAGE_API_KEYS[language]["client_id"]
    return CLIENT_ID

# Supported languages (evaluated)
SUPPORTED_LANGUAGES = [
    "Tamil",
    "English",
    "Hindi",
    "Malayalam",
    "Telugu"
]

# Audio processing parameters
SAMPLE_RATE = 16000  # 16kHz
N_MFCC = 13  # Number of MFCC coefficients
HOP_LENGTH = 512
N_FFT = 2048

# Classification threshold
# Confidence < 0.20 (20%) = AI_GENERATED
# Confidence > 0.80 (80%) = HUMAN
CLASSIFICATION_THRESHOLD = 0.5  # Not used anymore, keeping for compatibility

# Model path
MODEL_PATH = "models/voice_detection_model.pkl"

# Feature extraction parameters
FEATURE_CONFIG = {
    'mfcc': {
        'n_mfcc': N_MFCC,
        'hop_length': HOP_LENGTH,
        'n_fft': N_FFT
    },
    'spectral': {
        'hop_length': HOP_LENGTH,
        'n_fft': N_FFT
    }
}
