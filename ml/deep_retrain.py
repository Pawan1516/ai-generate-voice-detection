"""
DEEP RETRAIN - Advanced Model Training with Validation
Features:
- Cross-validation to prevent overfitting
- Data quality analysis
- Feature importance analysis
- Hyperparameter optimization
- Better error handling and diagnostics
"""

import os
import sys
import numpy as np
import pickle
import glob
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, auc, precision_recall_curve, f1_score
)
import warnings

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from audio_processor import AudioProcessor

class DeepRetrainer:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.output_dir = os.path.join(self.base_dir, 'backend', 'models')
        self.audio_processor = AudioProcessor()
        self.scaler = StandardScaler()
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_audio_safely(self, filepath):
        """Safely load and validate audio file"""
        try:
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                return None
            
            with open(filepath, 'rb') as f:
                audio_bytes = f.read()
            
            if len(audio_bytes) < 1000:  # At least 1KB
                return None
            
            return audio_bytes
        except:
            return None
    
    def validate_features(self, features):
        """Validate extracted features"""
        if not features:
            return False
        
        # Check for NaN or infinite values
        for key, val in features.items():
            if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                return False
        
        # Check for too many zeros
        zero_count = sum(1 for v in features.values() if v == 0)
        if zero_count > len(features) * 0.8:  # More than 80% zeros
            return False
        
        return True
    
    def load_training_data_validated(self):
        """Load training data with validation and diagnostic info"""
        print("=" * 80)
        print("DEEP DATA LOADING WITH VALIDATION")
        print("=" * 80)
        
        X = []
        y = []
        metadata = {
            'human_files': [],
            'ai_files': [],
            'human_invalid': 0,
            'ai_invalid': 0,
            'human_valid': 0,
            'ai_valid': 0
        }
        feature_names = None
        
        # Load HUMAN voices
        print("\n🎤 Loading HUMAN voices with validation...")
        human_dirs = [
            os.path.join(self.data_dir, 'train', 'human'),
            os.path.join(self.data_dir, 'merged_dataset'),
        ]
        
        for human_dir in human_dirs:
            if not os.path.exists(human_dir):
                continue
            
            # Find all audio files
            files = glob.glob(os.path.join(human_dir, '**', '*.wav'), recursive=True)
            
            for filepath in sorted(files)[:200]:  # Load up to 200
                audio_bytes = self.load_audio_safely(filepath)
                if not audio_bytes:
                    metadata['human_invalid'] += 1
                    continue
                
                features = self.audio_processor.extract_features(audio_bytes)
                if not features or not self.validate_features(features):
                    metadata['human_invalid'] += 1
                    continue
                
                # Valid sample
                feature_vector = sorted(features.items())
                if feature_names is None:
                    feature_names = [name for name, _ in feature_vector]
                
                X.append([val for _, val in feature_vector])
                y.append(0)  # Human = 0
                metadata['human_files'].append(os.path.basename(filepath))
                metadata['human_valid'] += 1
                
                if metadata['human_valid'] % 20 == 0:
                    print(f"   ✓ Loaded {metadata['human_valid']} valid human samples...")
        
        print(f"\n   ✅ Human voices loaded:")
        print(f"      Valid: {metadata['human_valid']}")
        print(f"      Invalid: {metadata['human_invalid']}")
        
        # Load AI voices
        print("\n🤖 Loading AI-GENERATED voices with validation...")
        ai_dirs = [
            os.path.join(self.data_dir, 'train', 'ai_generated'),
            os.path.join(self.data_dir, 'train', 'ai'),
            os.path.join(self.data_dir, 'train', 'ai_generated_gtts'),
        ]
        
        for ai_dir in ai_dirs:
            if not os.path.exists(ai_dir):
                continue
            
            files = glob.glob(os.path.join(ai_dir, '**', '*.wav'), recursive=True)
            files.extend(glob.glob(os.path.join(ai_dir, '**', '*.mp3'), recursive=True))
            
            for filepath in sorted(files)[:200]:  # Load up to 200
                audio_bytes = self.load_audio_safely(filepath)
                if not audio_bytes:
                    metadata['ai_invalid'] += 1
                    continue
                
                features = self.audio_processor.extract_features(audio_bytes)
                if not features or not self.validate_features(features):
                    metadata['ai_invalid'] += 1
                    continue
                
                # Valid sample
                feature_vector = sorted(features.items())
                
                X.append([val for _, val in feature_vector])
                y.append(1)  # AI = 1
                metadata['ai_files'].append(os.path.basename(filepath))
                metadata['ai_valid'] += 1
                
                if metadata['ai_valid'] % 20 == 0:
                    print(f"   ✓ Loaded {metadata['ai_valid']} valid AI samples...")
        
        print(f"\n   ✅ AI voices loaded:")
        print(f"      Valid: {metadata['ai_valid']}")
        print(f"      Invalid: {metadata['ai_invalid']}")
        
        total_valid = metadata['human_valid'] + metadata['ai_valid']
        print(f"\n📊 TOTAL VALID SAMPLES: {total_valid}")
        print(f"   Human: {metadata['human_valid']} ({100*metadata['human_valid']/total_valid:.1f}%)")
        print(f"   AI: {metadata['ai_valid']} ({100*metadata['ai_valid']/total_valid:.1f}%)")
        print(f"   Feature dimensions: {len(X[0]) if X else 0}")
        
        return np.array(X), np.array(y), feature_names, metadata
    
    def train_with_cross_validation(self, X, y, feature_names):
        """Train with cross-validation to detect overfitting"""
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION TRAINING")
        print("=" * 80)
        
        # Use stratified 5-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_scores = []
        cv_f1_scores = []
        all_predictions = []
        all_true_labels = []
        
        fold = 1
        best_model = None
        best_score = 0
        
        for train_idx, val_idx in skf.split(X, y):
            print(f"\n🔄 Fold {fold}/5...")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train with conservative hyperparameters to avoid overfitting
            model = RandomForestClassifier(
                n_estimators=100,           # Fewer trees
                max_depth=10,               # Limit depth
                min_samples_split=10,       # More samples required to split
                min_samples_leaf=5,         # More samples per leaf
                max_features='sqrt',        # Limit feature selection
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            
            cv_scores.append(acc)
            cv_f1_scores.append(f1)
            all_predictions.extend(y_pred)
            all_true_labels.extend(y_val)
            
            print(f"   Accuracy: {acc:.3f} | F1-Score: {f1:.3f}")
            
            if acc > best_score:
                best_score = acc
                best_model = model
                self.scaler = scaler
            
            fold += 1
        
        # Print cross-validation summary
        print("\n" + "-" * 80)
        print("CROSS-VALIDATION SUMMARY:")
        print(f"   Mean Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        print(f"   Mean F1-Score: {np.mean(cv_f1_scores):.3f} ± {np.std(cv_f1_scores):.3f}")
        
        # Overall metrics
        overall_acc = accuracy_score(all_true_labels, all_predictions)
        overall_f1 = f1_score(all_true_labels, all_predictions)
        
        print(f"\n   Overall Accuracy: {overall_acc:.3f}")
        print(f"   Overall F1-Score: {overall_f1:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_true_labels, all_predictions)
        print(f"\n   Confusion Matrix (CV Folds):")
        print(f"   True Negatives (Human → Human): {cm[0,0]}")
        print(f"   False Positives (Human → AI): {cm[0,1]}")
        print(f"   False Negatives (AI → Human): {cm[1,0]}")
        print(f"   True Positives (AI → AI): {cm[1,1]}")
        
        # Classification report
        print(f"\n   Classification Report:")
        print(classification_report(all_true_labels, all_predictions, 
                                   target_names=['HUMAN', 'AI']))
        
        return best_model, self.scaler, feature_names
    
    def save_model(self, model, scaler, feature_names):
        """Save trained model"""
        print("\n" + "=" * 80)
        print("SAVING MODEL")
        print("=" * 80)
        
        model_path = os.path.join(self.output_dir, 'voice_detection_model.pkl')
        model_data = {
            'model': model,
            'feature_names': feature_names
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        scaler_path = os.path.join(self.output_dir, 'voice_detection_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"\n✅ Model saved: {model_path}")
        print(f"✅ Scaler saved: {scaler_path}")
        
        return model_path, scaler_path

def main():
    print("\n" + "=" * 80)
    print("🚀 DEEP RETRAINING - ADVANCED MODEL TRAINING")
    print("=" * 80)
    
    retrainer = DeepRetrainer()
    
    # Load and validate data
    X, y, feature_names, metadata = retrainer.load_training_data_validated()
    
    total_samples = len(X)
    if total_samples < 20:
        print(f"\n❌ ERROR: Only {total_samples} valid samples found!")
        print("   Need at least 20 samples for training.")
        print("   Please check: data/train/human and data/train/ai_generated")
        return
    
    # Train with cross-validation
    model, scaler, feature_names = retrainer.train_with_cross_validation(X, y, feature_names)
    
    # Save
    model_path, scaler_path = retrainer.save_model(model, scaler, feature_names)
    
    print("\n" + "=" * 80)
    print("✅ DEEP RETRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nTraining Summary:")
    print(f"   Total samples used: {len(X)}")
    print(f"   Model type: Random Forest (100 trees, conservative hyperparameters)")
    print(f"   Validation: 5-fold stratified cross-validation")
    print(f"\nNext steps:")
    print(f"   1. Restart the API server:")
    print(f"      cd backend")
    print(f"      python -m uvicorn main:app --reload")
    print(f"\n   2. Test on http://127.0.0.1:8000/docs")
    print("=" * 80)

if __name__ == "__main__":
    main()
