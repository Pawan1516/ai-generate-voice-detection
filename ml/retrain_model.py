"""
RETRAIN MODEL - Improve Human vs AI Voice Detection
Fixes misclassification of human voices as AI
Uses better feature extraction and stronger training
"""

import os
import sys
import numpy as np
import pickle
import glob
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# Fix Windows Unicode
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from audio_processor import AudioProcessor

class ModelRetrainer:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.output_dir = os.path.join(self.base_dir, 'backend', 'models')
        self.audio_processor = AudioProcessor()
        self.scaler = StandardScaler()
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_training_data(self):
        """Load and balance human and AI voice training data"""
        print("=" * 70)
        print("LOADING TRAINING DATA")
        print("=" * 70)
        
        X = []
        y = []
        feature_names = None
        
        # Load HUMAN voices (label=0)
        print("\n🎤 Loading HUMAN voices...")
        human_paths = [
            os.path.join(self.data_dir, 'train', 'human'),
            os.path.join(self.data_dir, 'merged_dataset'),
        ]
        
        human_count = 0
        for human_dir in human_paths:
            if os.path.exists(human_dir):
                files = glob.glob(os.path.join(human_dir, '**', '*human*.wav'), recursive=True)
                files.extend(glob.glob(os.path.join(human_dir, '**', '*.wav'), recursive=True))
                
                for filepath in files[:100]:  # Limit to 100 per source
                    try:
                        with open(filepath, 'rb') as f:
                            audio_bytes = f.read()
                        
                        features = self.audio_processor.extract_features(audio_bytes)
                        if features and len(features) > 0:
                            feature_vector = sorted(features.items())
                            if feature_names is None:
                                feature_names = [name for name, _ in feature_vector]
                            
                            X.append([val for _, val in feature_vector])
                            y.append(0)  # Human = 0
                            human_count += 1
                            
                            if human_count % 20 == 0:
                                print(f"   ✓ Loaded {human_count} human samples...")
                    except Exception as e:
                        pass
        
        print(f"   ✓ Total HUMAN samples: {human_count}")
        
        # Load AI GENERATED voices (label=1)
        print("\n🤖 Loading AI-GENERATED voices...")
        ai_paths = [
            os.path.join(self.data_dir, 'train', 'ai_generated'),
            os.path.join(self.data_dir, 'train', 'ai'),
        ]
        
        ai_count = 0
        for ai_dir in ai_paths:
            if os.path.exists(ai_dir):
                files = glob.glob(os.path.join(ai_dir, '**', '*.wav'), recursive=True)
                files.extend(glob.glob(os.path.join(ai_dir, '**', '*.mp3'), recursive=True))
                
                for filepath in files[:100]:  # Limit to 100 per source
                    try:
                        with open(filepath, 'rb') as f:
                            audio_bytes = f.read()
                        
                        features = self.audio_processor.extract_features(audio_bytes)
                        if features and len(features) > 0:
                            feature_vector = sorted(features.items())
                            
                            X.append([val for _, val in feature_vector])
                            y.append(1)  # AI = 1
                            ai_count += 1
                            
                            if ai_count % 20 == 0:
                                print(f"   ✓ Loaded {ai_count} AI samples...")
                    except Exception as e:
                        pass
        
        print(f"   ✓ Total AI-GENERATED samples: {ai_count}")
        print(f"\n📊 TOTAL SAMPLES: {len(X)} (Human: {human_count}, AI: {ai_count})")
        print(f"   Feature dimensions: {len(X[0]) if X else 0}")
        
        return np.array(X), np.array(y), feature_names
    
    def train_model(self, X, y, feature_names):
        """Train improved model with better hyperparameters"""
        print("\n" + "=" * 70)
        print("TRAINING MODEL")
        print("=" * 70)
        
        # Split data: 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📈 Train set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")
        
        # Scale features
        print("\n🔧 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest with optimized hyperparameters
        print("\n🌲 Training Random Forest (200 trees)...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        print("\n📋 EVALUATION RESULTS:")
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"   Train Accuracy: {train_acc:.3f} ({int(train_acc * len(y_train))}/{len(y_train)})")
        print(f"   Test Accuracy:  {test_acc:.3f} ({int(test_acc * len(y_test))}/{len(y_test)})")
        
        # Confusion Matrix
        print("\n🎯 Confusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, test_pred)
        print(f"   True Negatives (Human correctly classified): {cm[0,0]}")
        print(f"   False Positives (Human → AI): {cm[0,1]}")
        print(f"   False Negatives (AI → Human): {cm[1,0]}")
        print(f"   True Positives (AI correctly classified): {cm[1,1]}")
        
        # Classification Report
        print("\n📊 Classification Report:")
        print(classification_report(y_test, test_pred, target_names=['HUMAN', 'AI']))
        
        # AUC Score
        test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, test_pred_proba)
        print(f"   ROC-AUC Score: {auc:.3f}")
        
        return model, X_test, y_test, test_pred, feature_names
    
    def save_model(self, model, feature_names):
        """Save trained model and scaler"""
        print("\n" + "=" * 70)
        print("SAVING MODEL")
        print("=" * 70)
        
        # Save model
        model_path = os.path.join(self.output_dir, 'voice_detection_model.pkl')
        model_data = {
            'model': model,
            'feature_names': feature_names
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n✓ Model saved to: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(self.output_dir, 'voice_detection_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler saved to: {scaler_path}")
        
        return model_path, scaler_path

def main():
    print("\n🚀 RETRAINING AI VOICE DETECTION MODEL")
    print("Improving Human vs AI voice classification accuracy\n")
    
    retrainer = ModelRetrainer()
    
    # Load data
    X, y, feature_names = retrainer.load_training_data()
    
    if len(X) < 10:
        print("\n❌ ERROR: Not enough training data loaded!")
        print("   Please ensure data/train/human and data/train/ai_generated exist with audio files.")
        return
    
    # Train model
    model, X_test, y_test, y_pred, feature_names = retrainer.train_model(X, y, feature_names)
    
    # Save model
    model_path, scaler_path = retrainer.save_model(model, feature_names)
    
    print("\n" + "=" * 70)
    print("✅ RETRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Model saved: {model_path}")
    print(f"📁 Scaler saved: {scaler_path}")
    print("\n🔄 Restart the API server to use the new model:")
    print("   cd backend")
    print("   python -m uvicorn main:app --reload")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
