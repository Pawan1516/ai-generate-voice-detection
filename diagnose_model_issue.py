"""
Diagnose why human voices are being classified as AI
This script will test the model with known human voices and analyze the problem
"""

import os
import sys
import numpy as np
import pickle
from pathlib import Path
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from audio_processor import AudioProcessor

def analyze_model():
    print("\n" + "="*70)
    print("🔍 MODEL DIAGNOSIS - Human Voice Misclassification Issue")
    print("="*70)
    
    # Load model
    model_path = 'backend/models/voice_detection_model.pkl'
    scaler_path = 'backend/models/voice_detection_scaler.pkl'
    
    print(f"\n📦 Loading Model...")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    
    if not os.path.exists(model_path):
        print(f"   ❌ Model not found!")
        return
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract actual model from dict if needed
    if isinstance(model_data, dict):
        model = model_data.get('model', model_data)
    else:
        model = model_data
    
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            if isinstance(scaler_data, dict):
                scaler = scaler_data.get('scaler', scaler_data)
            else:
                scaler = scaler_data
    
    # Check model properties
    print(f"\n📊 Model Properties:")
    print(f"   Type: {type(model)}")
    if hasattr(model, 'classes_'):
        print(f"   Classes: {model.classes_}")
    if hasattr(model, 'n_estimators'):
        print(f"   N Estimators: {model.n_estimators}")
    if hasattr(model, 'max_depth'):
        print(f"   Max Depth: {model.max_depth}")
    if hasattr(model, 'n_features_in_'):
        print(f"   Features expected: {model.n_features_in_}")
    
    # Test on known human voices
    print(f"\n🧪 Testing on Known Human Voices...")
    
    processor = AudioProcessor()
    human_dir = 'data/train/human'
    ai_dir = 'data/train/ai'
    
    # Get sample files
    human_files = glob.glob(f"{human_dir}/**/*.wav", recursive=True)[:5]
    ai_files = glob.glob(f"{ai_dir}/**/*.wav", recursive=True)[:5]
    
    print(f"\n👤 Human Voice Samples ({len(human_files)} files):")
    print("-" * 70)
    
    human_predictions = []
    for audio_file in human_files:
        try:
            with open(audio_file, 'rb') as f:
                audio_bytes = f.read()
            
            features = processor.extract_features(audio_bytes)
            if features is None:
                print(f"   ❌ {Path(audio_file).name} - Failed to extract features")
                continue
            
            # Prepare feature vector
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            if feature_vector.shape[1] != model.n_features_in_:
                print(f"   ❌ {Path(audio_file).name} - Feature mismatch: {feature_vector.shape[1]} vs {model.n_features_in_}")
                continue
            
            # Scale if scaler exists
            if scaler is not None:
                feature_vector = scaler.transform(feature_vector)
            
            # Predict
            probs = model.predict_proba(feature_vector)[0]
            pred = model.predict(feature_vector)[0]
            
            human_predictions.append((probs[0], probs[1], pred))
            
            pred_label = "HUMAN ✓" if pred == 0 else "AI ✗"
            print(f"   {pred_label} | Human: {probs[0]:.2%} | AI: {probs[1]:.2%} | {Path(audio_file).name}")
            
        except Exception as e:
            print(f"   ❌ {Path(audio_file).name} - Error: {str(e)}")
    
    print(f"\n🤖 AI Voice Samples ({len(ai_files)} files):")
    print("-" * 70)
    
    ai_predictions = []
    for audio_file in ai_files:
        try:
            with open(audio_file, 'rb') as f:
                audio_bytes = f.read()
            
            features = processor.extract_features(audio_bytes)
            if features is None:
                print(f"   ❌ {Path(audio_file).name} - Failed to extract features")
                continue
            
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            if feature_vector.shape[1] != model.n_features_in_:
                print(f"   ❌ {Path(audio_file).name} - Feature mismatch")
                continue
            
            if scaler is not None:
                feature_vector = scaler.transform(feature_vector)
            
            probs = model.predict_proba(feature_vector)[0]
            pred = model.predict(feature_vector)[0]
            
            ai_predictions.append((probs[0], probs[1], pred))
            
            pred_label = "HUMAN ✗" if pred == 0 else "AI ✓"
            print(f"   {pred_label} | Human: {probs[0]:.2%} | AI: {probs[1]:.2%} | {Path(audio_file).name}")
            
        except Exception as e:
            print(f"   ❌ {Path(audio_file).name} - Error: {str(e)}")
    
    # Summary
    print(f"\n📈 Analysis Summary:")
    print("-" * 70)
    
    if human_predictions:
        human_correct = sum(1 for h, a, p in human_predictions if p == 0)
        print(f"   Human samples: {human_correct}/{len(human_predictions)} correctly classified")
        human_avg_confidence = np.mean([h for h, a, p in human_predictions])
        print(f"   Average human confidence: {human_avg_confidence:.2%}")
    
    if ai_predictions:
        ai_correct = sum(1 for h, a, p in ai_predictions if p == 1)
        print(f"   AI samples: {ai_correct}/{len(ai_predictions)} correctly classified")
        ai_avg_confidence = np.mean([a for h, a, p in ai_predictions])
        print(f"   Average AI confidence: {ai_avg_confidence:.2%}")
    
    # Feature analysis
    print(f"\n🔧 Feature Importance (Top 10):")
    print("-" * 70)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-10:][::-1]
        for idx in top_indices:
            print(f"   Feature {idx}: {importances[idx]:.4f}")
    
    print("\n" + "="*70)
    print("Issues Found:")
    print("="*70)
    
    if human_predictions:
        misclassified = sum(1 for h, a, p in human_predictions if p != 0)
        if misclassified > 0:
            print(f"\n❌ {misclassified}/{len(human_predictions)} human voices misclassified as AI")
            print("\n   SOLUTION: Retrain model with better data quality")
            print("   - Check human training data for AI voices mixed in")
            print("   - Verify feature extraction is consistent")
            print("   - Check class balance in training data")

if __name__ == '__main__':
    try:
        analyze_model()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
