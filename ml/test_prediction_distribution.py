
import sys
import os
import glob
import numpy as np
import pickle

# Add backend to path to import AudioProcessor
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from audio_processor import AudioProcessor
from config import MODEL_PATH

def main():
    print("Loading model...")
    try:
        with open(f"../backend/{MODEL_PATH}", 'rb') as f:
            model_data = pickle.load(f)
            if isinstance(model_data, dict):
                model = model_data['model']
            else:
                model = model_data
            print(f"Model loaded successfully. Type: {type(model)}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    processor = AudioProcessor()
    
    # Test on HQ Edge AI files
    test_dir = "../data/train/ai_generated/hq_edge"
    files = glob.glob(os.path.join(test_dir, "*.mp3"))[:10] # Test 10 files
    
    print(f"\nTesting {len(files)} HQ AI files...")
    
    probabilities = []
    
    for f in files:
        try:
            with open(f, 'rb') as af:
                audio_bytes = af.read()
            
            features = processor.extract_features(audio_bytes)
            # Prepare for prediction (values only)
            feat_vector = np.array(list(features.values())).reshape(1, -1)
            
            # Predict
            probs = model.predict_proba(feat_vector)[0]
            ai_index = list(model.classes_).index(1)
            p_ai = probs[ai_index]
            
            probabilities.append(p_ai)
            print(f"{os.path.basename(f)} -> P(AI) = {p_ai:.4f}")
            
        except Exception as e:
            print(f"Error {f}: {e}")
            
    print("\n--- Summary ---")
    print(f"Mean P(AI): {np.mean(probabilities):.4f}")
    if np.mean(probabilities) > 0.5:
        print("✅ Model correctly identifies these as AI (Avg > 0.5)")
    else:
        print("❌ Model thinks these are Human (Avg < 0.5)")

if __name__ == "__main__":
    main()
