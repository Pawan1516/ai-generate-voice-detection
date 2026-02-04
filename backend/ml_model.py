import pickle
import numpy as np
from typing import Dict
import os
from config import MODEL_PATH

class VoiceClassifier:
    """ML model for classifying AI-generated vs human voices"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load trained model from disk"""
        # Get absolute path to model
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(backend_dir, MODEL_PATH)
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    
                # Check if it's wrapped in a dict or direct model
                if isinstance(model_data, dict) and 'model' in model_data:
                    self.model = model_data['model']
                    self.feature_names = model_data.get('feature_names')
                else:
                    # Direct model object
                    self.model = model_data
                    self.feature_names = None
                
                # Load scaler if available
                scaler_path = os.path.join(backend_dir, 'models', 'voice_detection_scaler.pkl')
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                
                print(f"Model loaded successfully from {model_path}")
                if self.scaler:
                    print("Scaler also loaded")
            except Exception as e:
                print(f"Warning: Could not load model: {str(e)}")
                raise e # Hard fail if model is broken, per "no hard coding" requirement
        else:
            print(f"Model file not found at {model_path}.")
            print("Running in DEPLOYMENT MODE (Fallback). initializing dummy model for API health.")
            # Initialize a simple dummy model for API health
            from sklearn.dummy import DummyClassifier
            self.model = DummyClassifier(strategy="constant", constant=0) # Default to Human
            self.model.fit([[0] * 13], [0]) # Mock fit
            self.feature_names = [f'mfcc_{i}_mean' for i in range(13)]
            self.scaler = None
            print("WARNING: Using DUMMY model. Predictions will be constant.")
    
    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Predict if voice is AI-generated or human
        """
        if self.model is None:
             raise RuntimeError("Model not loaded")

        # Use real trained model ONLY
        feature_vector = self._prepare_features(features)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(feature_vector)[0]
        
        # Find which class is AI (class 1) in the model's class order
        ai_class_index = list(self.model.classes_).index(1)
        ml_probability = probabilities[ai_class_index]
        probability = ml_probability
        
        # Log for debugging
        print(f"Classes: {self.model.classes_}, AI index: {ai_class_index}, Probabilities: {probabilities}, P(AI)={ml_probability:.3f}")
        
        return {
            'probability': float(probability),
            'features_used': len(features)
        }
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare features for model input"""
        
        # CRITICAL: Ensure feature order matches training
        if self.feature_names is not None:
            # Use specific order from trained model
            feature_vector = []
            for name in self.feature_names:
                # Use strict 0.0 default if missing to avoid crashing
                feature_vector.append(features.get(name, 0.0))
            return np.array(feature_vector).reshape(1, -1)
        else:
            # Fallback for legacy models or dummy: Sort keys alphabetically
            # (Matches train_model.py default behavior: self.feature_names = sorted(X[0].keys()))
            sorted_keys = sorted(features.keys())
            feature_vector = np.array([features[k] for k in sorted_keys])
            return feature_vector.reshape(1, -1)
