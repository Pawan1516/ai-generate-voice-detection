"""
Compliance Test Script - Hackathon Specification
"""
import requests
import base64
import os

# Configuration from images
API_URL = "http://localhost:8000/api/voice-detection"
API_KEY = "ff568c59718453585c4eb6507283633e318793cb9213adab671c87318070a951"  # From .env / config

def test_compliance():
    print("="*60)
    print("HACKATHON COMPLIANCE TEST")
    print("="*60)
    
    # Sample data
    try:
        sample_path = 'data/train/human/english/0_jackson_0.wav' # Using wav locally for test, but will call as mp3
        with open(sample_path, 'rb') as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        print(f"Failed to load sample: {e}")
        return

    # 1. Correct Request
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_b64
    }
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    print("\n[1] Testing Strict Compliance Request...")
    response = requests.post(API_URL, json=payload, headers=headers)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Response Body: {data}")
        
        # Verify schema
        required_fields = ["status", "language", "classification", "confidenceScore", "explanation"]
        missing = [f for f in required_fields if f not in data]
        
        if not missing:
            print("✅ SCHEMA MATCHES EXACTLY!")
        else:
            print(f"❌ MISSING FIELDS: {missing}")
            
        # Verify no extra fields (optional but good for strict compliance)
        extra = [f for f in data if f not in required_fields]
        if extra:
            print(f"⚠️ EXTRA FIELDS (NOT IN SPEC): {extra}")

    # 2. Test Invalid Format (WAV should fail now)
    payload_wav = payload.copy()
    payload_wav["audioFormat"] = "wav"
    print("\n[2] Testing Invalid Format (WAV - should fail)...")
    response_wav = requests.post(API_URL, json=payload_wav, headers=headers)
    print(f"Status: {response_wav.status_code} (Expected 422 or 400)")

    # 3. Test Invalid Language
    payload_lang = payload.copy()
    payload_lang["language"] = "Kannada"
    print("\n[3] Testing Invalid Language (Kannada - should fail)...")
    response_lang = requests.post(API_URL, json=payload_lang, headers=headers)
    print(f"Status: {response_lang.status_code} (Expected 422 or 400)")

if __name__ == "__main__":
    test_compliance()
