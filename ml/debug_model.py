"""
Debug script to check model predictions
"""
import os
import sys
import pickle
import numpy as np
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from audio_processor import AudioProcessor

# Load the model
model_path = 'backend/models/voice_detection_model.pkl'
scaler_path = 'backend/models/voice_detection_scaler.pkl'

print("="*70)
print("🔍 MODEL DEBUG - CHECK PREDICTIONS")
print("="*70)

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"\n✓ Model loaded: {model}")
    print(f"  Classes: {model.classes_}")
    print(f"  Class labels: 0=Human, 1=AI")
    
    # Test with a real audio file
    audio_processor = AudioProcessor()
    
    # Find an AI file
    ai_files = glob.glob('data/train/ai_generated/**/*.wav', recursive=True)
    ai_files.extend(glob.glob('data/train/ai_generated/**/*.mp3', recursive=True))
    
    # Find a human file
    human_files = glob.glob('data/train/human/**/*.wav', recursive=True)
    human_files.extend(glob.glob('data/train/human/**/*.mp3', recursive=True))
    
    print(f"\n📂 Testing with actual data files:")
    print(f"   Found {len(ai_files)} AI files")
    print(f"   Found {len(human_files)} human files")
    
    # Test AI file
    if ai_files:
        ai_file = ai_files[0]
        print(f"\n🤖 Testing AI file: {os.path.basename(ai_file)}")
        
        with open(ai_file, 'rb') as f:
            audio_bytes = f.read()
        
        features_dict = audio_processor.extract_features(audio_bytes)
        features_list = list(features_dict.values())
        features_scaled = scaler.transform([features_list])
        
        pred_class = model.predict(features_scaled)[0]
        pred_proba = model.predict_proba(features_scaled)[0]
        
        print(f"   Features extracted: {len(features_list)}")
        print(f"   Prediction class: {pred_class} (0=Human, 1=AI)")
        print(f"   Probabilities: {pred_proba}")
        print(f"     - P(Human): {pred_proba[0]:.4f}")
        print(f"     - P(AI): {pred_proba[1]:.4f}")
        
        if pred_class == 1:
            print(f"   ✅ CORRECT - Detected as AI")
        else:
            print(f"   ❌ WRONG - Detected as Human (should be AI)")
    
    # Test human file
    if human_files:
        human_file = human_files[0]
        print(f"\n👤 Testing Human file: {os.path.basename(human_file)}")
        
        with open(human_file, 'rb') as f:
            audio_bytes = f.read()
        
        features_dict = audio_processor.extract_features(audio_bytes)
        features_list = list(features_dict.values())
        features_scaled = scaler.transform([features_list])
        
        pred_class = model.predict(features_scaled)[0]
        pred_proba = model.predict_proba(features_scaled)[0]
        
        print(f"   Features extracted: {len(features_list)}")
        print(f"   Prediction class: {pred_class} (0=Human, 1=AI)")
        print(f"   Probabilities: {pred_proba}")
        print(f"     - P(Human): {pred_proba[0]:.4f}")
        print(f"     - P(AI): {pred_proba[1]:.4f}")
        
        if pred_class == 0:
            print(f"   ✅ CORRECT - Detected as Human")
        else:
            print(f"   ❌ WRONG - Detected as AI (should be Human)")

else:
    print(f"❌ Model not found: {model_path}")

print("\n" + "="*70)
