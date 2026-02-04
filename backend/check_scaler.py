import pickle
import os

backend_dir = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(backend_dir, 'models', 'voice_detection_scaler.pkl')

print(f"Path: {scaler_path}")
if os.path.exists(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    if hasattr(scaler, 'n_features_in_'):
        print(f"Scaler expects {scaler.n_features_in_} features.")
    else:
        print("Scaler has no n_features_in_ attribute.")
else:
    print("Scaler file not found.")
