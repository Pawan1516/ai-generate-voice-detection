#!/usr/bin/env python3
"""
🧪 TEST THE TRAINED MODEL
Quick validation of AI vs Human voice detection
"""

import os
import sys
import requests
import base64
import glob
from pathlib import Path

def test_api(audio_file, expected_label):
    """Test a single audio file against the API"""
    
    try:
        # Read audio file
        with open(audio_file, 'rb') as f:
            audio_data = f.read()
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Prepare request
        payload = {
            "audioBase64": audio_base64,
            "audioFormat": "wav",
            "language": "english"
        }
        
        # Make API request
        response = requests.post(
            "http://localhost:8001/api/detect",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get('prediction')
            confidence = result.get('confidence', 0)
            
            # Parse prediction (0=human, 1=AI)
            is_ai = prediction == 'AI'
            
            # Check if correct
            expected_ai = expected_label == 'AI'
            correct = is_ai == expected_ai
            
            status = "✅" if correct else "❌"
            
            print(f"{status} {Path(audio_file).name:30} → {prediction:8} (conf: {confidence:.2%})")
            
            return correct
        else:
            print(f"❌ {Path(audio_file).name:30} → API Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ {Path(audio_file).name:30} → Error: {str(e)[:50]}")
        return False

def main():
    """Run model tests"""
    
    print("\n" + "="*70)
    print("🧪 TESTING TRAINED MODEL")
    print("="*70)
    
    data_dir = "../data/test"
    
    # Test AI-generated voices
    print("\n🤖 Testing AI-generated voices:")
    print("-" * 70)
    
    ai_files = glob.glob(os.path.join(data_dir, "ai_generated", "**", "*.wav"), recursive=True)
    ai_files.extend(glob.glob(os.path.join(data_dir, "ai_generated", "**", "*.mp3"), recursive=True))
    
    ai_correct = 0
    ai_tested = 0
    
    for audio_file in ai_files[:10]:  # Test first 10
        if test_api(audio_file, "AI"):
            ai_correct += 1
        ai_tested += 1
    
    if ai_tested > 0:
        print(f"\nAI accuracy: {ai_correct}/{ai_tested} ({100*ai_correct/ai_tested:.1f}%)")
    
    # Test human voices
    print("\n👤 Testing human voices:")
    print("-" * 70)
    
    human_files = glob.glob(os.path.join(data_dir, "human", "**", "*.wav"), recursive=True)
    human_files.extend(glob.glob(os.path.join(data_dir, "human", "**", "*.mp3"), recursive=True))
    
    human_correct = 0
    human_tested = 0
    
    for audio_file in human_files[:10]:  # Test first 10
        if test_api(audio_file, "human"):
            human_correct += 1
        human_tested += 1
    
    if human_tested > 0:
        print(f"\nHuman accuracy: {human_correct}/{human_tested} ({100*human_correct/human_tested:.1f}%)")
    
    # Summary
    print("\n" + "="*70)
    total_correct = ai_correct + human_correct
    total_tested = ai_tested + human_tested
    
    if total_tested > 0:
        overall_acc = 100 * total_correct / total_tested
        print(f"📊 Overall Accuracy: {total_correct}/{total_tested} ({overall_acc:.1f}%)")
        print("="*70)
        
        if overall_acc >= 95:
            print("🏆 EXCELLENT! Model is ready for production!")
        elif overall_acc >= 80:
            print("✅ Good performance! Model is ready.")
        else:
            print("⚠️  Consider improving the model...")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⛔ Test interrupted by user")
        sys.exit(0)
