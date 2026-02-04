"""
Test hybrid detector on sample files
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.hybrid_detector import HybridVoiceDetector
import glob

def test_hybrid_detector():
    print("="*70)
    print("🔬 TESTING HYBRID AI VOICE DETECTOR")
    print("="*70)
    
    hd = HybridVoiceDetector()
    
    # Test human voice
    print("\n📋 Testing HUMAN voices...")
    human_files = glob.glob('data/train/human/**/*.wav', recursive=True)[:3]
    
    for i, file in enumerate(human_files, 1):
        print(f"\n{i}. Testing: {os.path.basename(file)}")
        try:
            with open(file, 'rb') as f:
                audio = f.read()
            
            result = hd.predict_hybrid(audio)
            
            if result.get('classification') == 'ERROR':
                print(f"   ❌ Error: {result.get('error')}")
                continue
            
            print(f"   Classification: {result['classification']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   ML Score (AI prob): {result['breakdown']['ml_score']:.2%}")
            print(f"   Artifact Score: {result['breakdown']['artifact_score']:.2%}")
            print(f"   Final Score: {result['breakdown']['final_score']:.2%}")
            print(f"   Reasoning: {result['reasoning']}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test AI voice
    print("\n\n📋 Testing AI-GENERATED voices...")
    ai_files = glob.glob('data/train/ai_generated/**/*.wav', recursive=True)[:3]
    
    for i, file in enumerate(ai_files, 1):
        print(f"\n{i}. Testing: {os.path.basename(file)}")
        try:
            with open(file, 'rb') as f:
                audio = f.read()
            
            result = hd.predict_hybrid(audio)
            
            if result.get('classification') == 'ERROR':
                print(f"   ❌ Error: {result.get('error')}")
                continue
            
            print(f"   Classification: {result['classification']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   ML Score (AI prob): {result['breakdown']['ml_score']:.2%}")
            print(f"   Artifact Score: {result['breakdown']['artifact_score']:.2%}")
            print(f"   Final Score: {result['breakdown']['final_score']:.2%}")
            print(f"   Reasoning: {result['reasoning']}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*70)
    print("✅ TESTING COMPLETE")
    print("="*70)

if __name__ == '__main__':
    test_hybrid_detector()
