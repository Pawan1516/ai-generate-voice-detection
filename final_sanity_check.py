"""
Final Sanity Check for Deterministic Voice Detection
"""
import sys
import os
import glob
import numpy as np
import librosa

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from backend.hybrid_detector import HybridVoiceDetector

import json

def run_sanity_check():
    print("Running deterministic sanity check...")
    
    hd = HybridVoiceDetector()
    
    # 1. Find samples
    human_dir = 'data/train/human'
    ai_dir = 'data/train/ai_generated'
    
    human_samples = glob.glob(os.path.join(human_dir, '**', '*.wav'), recursive=True)[:5]
    ai_samples = glob.glob(os.path.join(ai_dir, '**', '*.wav'), recursive=True)[:5]
    
    all_tests = [
        (s, "HUMAN") for s in human_samples
    ] + [
        (s, "AI_GENERATED") for s in ai_samples
    ]
    
    results = []
    success_count = 0
    
    for path, expected in all_tests:
        try:
            with open(path, 'rb') as f:
                audio_bytes = f.read()
            
            res = hd.predict_hybrid(audio_bytes)
            
            actual = res['classification']
            confidence = res['confidence']
            match = (actual == expected)
            
            if match:
                success_count += 1
            
            results.append({
                "file": os.path.basename(path),
                "expected": expected,
                "actual": actual,
                "match": match,
                "confidence": confidence,
                "ml_score": res['breakdown']['ml_score'],
                "final_score": res['breakdown']['final_score']
            })
            
            print(f"Tested {os.path.basename(path)}: {'OK' if match else 'FAIL'}")
            
        except Exception as e:
            print(f"Error testing {path}: {e}")

    # Save to JSON
    with open('final_results.json', 'w') as f:
        json.dump({
            "summary": f"{success_count}/{len(all_tests)} matches",
            "results": results
        }, f, indent=2)
    
    print(f"Done. {success_count}/{len(all_tests)} matches. Results saved to final_results.json")

if __name__ == "__main__":
    run_sanity_check()
