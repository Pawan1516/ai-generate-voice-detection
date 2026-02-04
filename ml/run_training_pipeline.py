#!/usr/bin/env python3
"""
🚀 HACKATHON TRAINING PIPELINE
Generate AI voices → Train model → Ready for inference
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n{'='*70}")
    print(f"▶️  {description}")
    print(f"{'='*70}")
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(__file__) or '.',
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"\n✅ {description} - SUCCESS")
            return True
        else:
            print(f"\n❌ {description} - FAILED (exit code: {result.returncode})")
            return False
    except Exception as e:
        print(f"\n❌ Error running {description}: {e}")
        return False

def main():
    """Run the complete training pipeline"""
    
    print("\n" + "🎤 "*20)
    print("HACKATHON TRAINING PIPELINE")
    print("Generate AI Voices → Train Model → Deploy Ready")
    print("🎤 "*20)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ml_dir = script_dir
    
    # Step 1: Generate AI voices
    print("\n\n📌 STEP 1: GENERATE AI TRAINING DATA")
    print("-" * 70)
    
    generate_script = os.path.join(ml_dir, 'generate_training_data.py')
    if os.path.exists(generate_script):
        success = run_command(
            [sys.executable, generate_script],
            "Generating AI voice training data (Coqui XTTS)"
        )
        
        if not success:
            print("\n⚠️  Skipping AI voice generation (TTS not installed)")
            print("   You can manually generate voices later or continue with existing data")
    
    # Step 2: Train the model
    print("\n\n📌 STEP 2: TRAIN AI VS HUMAN DETECTION MODEL")
    print("-" * 70)
    
    train_script = os.path.join(ml_dir, 'train_model_optimized.py')
    if os.path.exists(train_script):
        success = run_command(
            [sys.executable, train_script],
            "Training voice detection model"
        )
        
        if not success:
            print("\n❌ Model training failed!")
            print("   Check data directory structure and audio files")
            return 1
    else:
        print(f"❌ Training script not found: {train_script}")
        return 1
    
    # Step 3: Summary
    print("\n\n" + "="*70)
    print("🏆 TRAINING PIPELINE COMPLETE")
    print("="*70)
    
    print("\n✅ Your model is now ready!")
    print("\n📂 Model location: backend/models/")
    print("   - voice_detection_model.pkl")
    print("   - voice_detection_scaler.pkl")
    
    print("\n🚀 Next steps:")
    print("   1. Start the API: python backend/main.py")
    print("   2. Test with Postman or frontend")
    print("   3. Submit to judges!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
