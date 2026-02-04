"""
DIAGNOSTIC: Test which AI voices are misclassified as human
Helps calibrate the spectral flatness threshold
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.audio_processor_fixed import AudioProcessorFixed
from backend.inference_fixed import VoiceClassifierFixed


def diagnose_ai_misclassification():
    """
    Find which AI voices are being misclassified as HUMAN
    """
    print("\n" + "="*70)
    print("🔍 DIAGNOSING AI VOICE MISCLASSIFICATION")
    print("="*70)
    
    classifier = VoiceClassifierFixed()
    processor = AudioProcessorFixed()
    
    # AI voice directories
    ai_dirs = [
        "data/train/ai_generated/english",
        "data/train/ai_generated/hindi",
        "data/train/ai_generated/tamil",
        "data/train/ai_generated/telugu",
    ]
    
    misclassified = []
    correct = []
    
    for ai_dir in ai_dirs:
        if not os.path.exists(ai_dir):
            continue
        
        print(f"\n📂 Testing {ai_dir.split('/')[-1].upper()} AI voices...")
        
        audio_files = glob.glob(os.path.join(ai_dir, "*.wav"))[:10]  # Test first 10
        
        for filepath in audio_files:
            try:
                with open(filepath, 'rb') as f:
                    audio_bytes = f.read()
                
                result = classifier.predict_with_safety(audio_bytes)
                
                # Extract audio for spectral analysis
                y, sr = processor.load_audio_consistent(audio_bytes)
                flatness = processor.extract_spectral_flatness(y)
                
                if result['classification'] == 'HUMAN':
                    # MISCLASSIFIED
                    misclassified.append({
                        'file': os.path.basename(filepath),
                        'language': ai_dir.split('/')[-1],
                        'flatness': flatness,
                        'confidence': result['confidence'],
                        'method': result.get('method', 'unknown')
                    })
                    print(f"   ❌ {os.path.basename(filepath)}: HUMAN (flatness={flatness:.4f})")
                else:
                    # CORRECT
                    correct.append({
                        'file': os.path.basename(filepath),
                        'flatness': flatness
                    })
                    print(f"   ✅ {os.path.basename(filepath)}: AI (flatness={flatness:.4f})")
            
            except Exception as e:
                print(f"   ⚠️  Error processing {filepath}: {e}")
                continue
    
    # Summary
    print("\n" + "="*70)
    print("📊 DIAGNOSIS RESULTS")
    print("="*70)
    
    print(f"\n✅ Correctly classified as AI: {len(correct)}")
    print(f"❌ Misclassified as HUMAN: {len(misclassified)}")
    
    if misclassified:
        print(f"\n🔴 PROBLEM AI VOICES (False Negatives):")
        
        # Group by language
        by_lang = {}
        for item in misclassified:
            lang = item['language']
            if lang not in by_lang:
                by_lang[lang] = []
            by_lang[lang].append(item)
        
        for lang, items in sorted(by_lang.items()):
            print(f"\n   {lang.upper()}:")
            avg_flatness = np.mean([i['flatness'] for i in items])
            print(f"   Average Spectral Flatness: {avg_flatness:.4f}")
            print(f"   Threshold (current): 0.02")
            print(f"   Suggested new threshold: {avg_flatness * 0.8:.4f}")
    
    if correct:
        print(f"\n✅ CORRECT AI VOICES:")
        avg_flatness = np.mean([i['flatness'] for i in correct])
        print(f"   Average Spectral Flatness: {avg_flatness:.4f}")
    
    print("\n" + "="*70)
    print("💡 RECOMMENDATION")
    print("="*70)
    
    if misclassified:
        min_ai_flatness = min([i['flatness'] for i in misclassified])
        print(f"\nLowest AI flatness: {min_ai_flatness:.4f}")
        print(f"Current threshold: 0.02")
        print(f"\n❌ PROBLEM: AI voices have flatness > 0.02, but safety override triggers at < 0.02")
        print(f"✅ SOLUTION: Lower the spectral flatness threshold to {min_ai_flatness * 0.9:.4f}")
        print(f"            OR disable safety override for AI detection")
        print(f"            OR adjust threshold logic entirely")
    else:
        print("\n✅ All AI voices correctly classified!")
        print("   Safety override is working correctly")


if __name__ == '__main__':
    diagnose_ai_misclassification()
