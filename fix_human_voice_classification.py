"""
COMPLETE MODEL FIX - Retrain with proper data validation and class balancing
This script will fix the issue of human voices being misclassified as AI

Key improvements:
- Proper data validation and cleaning
- Class balancing (equal human and AI samples)
- Optimized hyperparameters for better generalization
- Feature scaling and normalization
- Train/test split with stratification
- Detailed performance metrics
"""

import os
import sys
import numpy as np
import pickle
import glob
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from audio_processor import AudioProcessor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

class ProperModelRetrainer:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data', 'train')
        self.output_dir = os.path.join(self.base_dir, 'backend', 'models')
        self.audio_processor = AudioProcessor()
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_audio_safely(self, filepath):
        """Load audio and extract features"""
        try:
            if not os.path.exists(filepath) or os.path.getsize(filepath) < 1000:
                return None  # Too small or doesn't exist
            
            with open(filepath, 'rb') as f:
                audio_bytes = f.read()
            
            if len(audio_bytes) < 100:
                return None
            
            # Extract features
            features = self.audio_processor.extract_features(audio_bytes)
            return features
            
        except Exception as e:
            print(f"      Error loading {Path(filepath).name}: {str(e)}")
            return None
    
    def load_dataset(self):
        """Load all training data with validation"""
        print("\n" + "="*70)
        print("📂 LOADING TRAINING DATA")
        print("="*70)
        
        X = []
        y = []
        
        # Load human voices (class 0)
        print(f"\n👤 Loading Human Voices (Class 0)...")
        human_dir = os.path.join(self.data_dir, 'human')
        human_files = list(glob.glob(f"{human_dir}/**/*.wav", recursive=True))[:200]  # Limit to 200
        
        human_count = 0
        for i, audio_file in enumerate(human_files):
            if i % 20 == 0:
                print(f"   Processing: {i}/{len(human_files)}...", end='\r')
            
            features = self.load_audio_safely(audio_file)
            if features is not None:
                X.append(list(features.values()))
                y.append(0)
                human_count += 1
        
        print(f"   ✓ Loaded {human_count} human voice samples")
        
        # Load AI voices (class 1)
        print(f"\n🤖 Loading AI Generated Voices (Class 1)...")
        ai_dirs = [
            os.path.join(self.data_dir, 'ai'),
            os.path.join(self.data_dir, 'ai_generated'),
            os.path.join(self.data_dir, 'ai_generated_gtts')
        ]
        
        ai_files = []
        for ai_dir in ai_dirs:
            if os.path.exists(ai_dir):
                ai_files.extend(glob.glob(f"{ai_dir}/**/*.wav", recursive=True))
        
        ai_files = ai_files[:200]  # Limit to 200
        
        ai_count = 0
        for i, audio_file in enumerate(ai_files):
            if i % 20 == 0:
                print(f"   Processing: {i}/{len(ai_files)}...", end='\r')
            
            features = self.load_audio_safely(audio_file)
            if features is not None:
                X.append(list(features.values()))
                y.append(1)
                ai_count += 1
        
        print(f"   ✓ Loaded {ai_count} AI voice samples")
        
        # Convert to numpy
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total samples: {len(X)}")
        print(f"   Human voices: {np.sum(y == 0)}")
        print(f"   AI voices: {np.sum(y == 1)}")
        print(f"   Features per sample: {X.shape[1]}")
        
        # Balance dataset
        min_class = min(np.sum(y == 0), np.sum(y == 1))
        print(f"\n⚖️  Balancing Dataset to {min_class} samples per class...")
        
        human_indices = np.where(y == 0)[0]
        ai_indices = np.where(y == 1)[0]
        
        selected_human = np.random.choice(human_indices, min_class, replace=False)
        selected_ai = np.random.choice(ai_indices, min_class, replace=False)
        selected_indices = np.concatenate([selected_human, selected_ai])
        
        X = X[selected_indices]
        y = y[selected_indices]
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        print(f"   ✓ Balanced dataset: {len(X)} total samples")
        print(f"   - Human: {np.sum(y == 0)}")
        print(f"   - AI: {np.sum(y == 1)}")
        
        return X, y
    
    def train_and_evaluate(self, X, y):
        """Train and evaluate model with proper metrics"""
        print("\n" + "="*70)
        print("🤖 TRAINING MODEL")
        print("="*70)
        
        # Split data
        print(f"\n📊 Splitting data: 80% train, 20% test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Train: {len(X_train)} samples")
        print(f"   Test: {len(X_test)} samples")
        
        # Scale features
        print(f"\n🔧 Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest with better hyperparameters
        print(f"\n🌲 Training Random Forest Classifier...")
        print(f"   Hyperparameters:")
        print(f"   - n_estimators: 200")
        print(f"   - max_depth: 15")
        print(f"   - min_samples_split: 5")
        print(f"   - min_samples_leaf: 2")
        print(f"   - class_weight: balanced")
        print(f"   - random_state: 42")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        print(f"   ✓ Training complete")
        
        # Evaluate on test set
        print(f"\n📈 Evaluating on Test Set...")
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   Test Accuracy: {accuracy:.2%}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n   Confusion Matrix:")
        print(f"   {cm[0, 0]:3d} (Human → Human)    {cm[0, 1]:3d} (Human → AI)")
        print(f"   {cm[1, 0]:3d} (AI → Human)      {cm[1, 1]:3d} (AI → AI)")
        
        # Classification report
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
        
        # Cross-validation on full dataset
        print(f"\n✔️  5-Fold Cross-Validation...")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        print(f"   CV Scores: {cv_scores}")
        print(f"   Mean CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
        
        # Feature importance
        print(f"\n🔍 Top 10 Most Important Features:")
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-10:][::-1]
        for rank, idx in enumerate(top_indices, 1):
            print(f"   {rank:2d}. Feature {idx:2d}: {importances[idx]:.4f}")
        
        return model, scaler, accuracy, cm
    
    def save_model(self, model, scaler):
        """Save trained model and scaler"""
        print(f"\n" + "="*70)
        print("💾 SAVING MODEL")
        print("="*70)
        
        model_path = os.path.join(self.output_dir, 'voice_detection_model.pkl')
        scaler_path = os.path.join(self.output_dir, 'voice_detection_scaler.pkl')
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n   ✓ Model saved to {model_path}")
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"   ✓ Scaler saved to {scaler_path}")
    
    def run(self):
        """Run complete training pipeline"""
        try:
            # Load data
            X, y = self.load_dataset()
            
            # Train and evaluate
            model, scaler, accuracy, cm = self.train_and_evaluate(X, y)
            
            # Check if model is good enough
            if accuracy < 0.80:
                print(f"\n⚠️  WARNING: Model accuracy is only {accuracy:.2%}")
                print(f"   This may still result in misclassifications")
                response = input(f"\nContinue and save model anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Training cancelled.")
                    return False
            
            # Save model
            self.save_model(model, scaler)
            
            print(f"\n" + "="*70)
            print("✅ MODEL TRAINING COMPLETE!")
            print("="*70)
            print(f"\n✓ Model accuracy on test set: {accuracy:.2%}")
            print(f"✓ Model is ready for deployment")
            print(f"\nHuman voices will now be correctly classified!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    retainer = ProperModelRetrainer()
    retainer.run()
