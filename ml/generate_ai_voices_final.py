#!/usr/bin/env python3
"""
Generate AI voices using Coqui TTS for training data
Improved version with error handling
"""

import os
import sys
import subprocess

def install_tts():
    """Install TTS if not present"""
    try:
        import TTS
        print("✓ TTS already installed")
        return True
    except ImportError:
        print("Installing Coqui TTS...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "TTS", "pysoundfile"],
                timeout=120
            )
            print("✓ TTS installed")
            return True
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            return False

def generate_ai_voices():
    """Generate AI voices for training"""
    
    if not install_tts():
        print("\n⚠️  TTS installation failed. Skipping AI voice generation.")
        print("You can still use existing training data.")
        return False
    
    from TTS.api import TTS
    import numpy as np
    
    print("\n" + "="*70)
    print("🎤 GENERATING AI VOICES WITH COQUI XTTS")
    print("="*70)
    
    # Text samples for different languages
    LANG_TEXTS = {
        "english": [
            "Hello, this is a test voice.",
            "Artificial intelligence is the future.",
            "Voice detection is important for security.",
            "Machine learning models need training data.",
            "Digital voice technology is advancing rapidly.",
            "This audio is artificially generated.",
            "Testing voice authentication systems.",
            "Neural networks can synthesize speech.",
        ],
        "hindi": [
            "नमस्ते, यह एक परीक्षण आवाज है।",
            "कृत्रिम बुद्धिमत्ता भविष्य है।",
            "वॉयस पहचान सुरक्षा के लिए महत्वपूर्ण है।",
            "मशीन लर्निंग को प्रशिक्षण डेटा की आवश्यकता है।",
        ],
    }
    
    try:
        print("\n📥 Loading Coqui XTTS model (multilingual)...")
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=False,
            progress_bar=False
        )
        print("✓ Model loaded")
        
        base_dir = "data/train/ai_generated"
        os.makedirs(base_dir, exist_ok=True)
        
        total = 0
        
        for lang, texts in LANG_TEXTS.items():
            lang_dir = os.path.join(base_dir, lang)
            os.makedirs(lang_dir, exist_ok=True)
            
            print(f"\n🗣️  Generating {lang.upper()} voices ({len(texts)} texts × 10 variations)...")
            
            for text_idx, text in enumerate(texts, 1):
                for var_idx in range(10):  # 10 variations per text
                    filename = f"ai_{text_idx:02d}_{var_idx:02d}.wav"
                    filepath = os.path.join(lang_dir, filename)
                    
                    if os.path.exists(filepath):
                        continue
                    
                    try:
                        tts.tts_to_file(
                            text=text,
                            file_path=filepath,
                            language=lang
                        )
                        total += 1
                        if total % 20 == 0:
                            print(f"  Generated {total} samples...")
                    except Exception as e:
                        print(f"  ⚠️  Error generating {filename}: {str(e)[:50]}")
                        continue
        
        print(f"\n✓ Generated {total} AI voice samples total")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = generate_ai_voices()
    sys.exit(0 if success else 1)
