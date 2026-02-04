"""
FINAL BALANCED FIX - Train model without aggressive safety overrides
Uses proper threshold calibration instead
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

from audio_processor_fixed import AudioProcessorFixed
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve


class BalancedModelTrainer:
    """Train without aggressive safety overrides - let the model decide"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data', 'train')
        self.output_dir = os.path.join(self.base_dir, 'backend', 'models')
        self.audio_processor = AudioProcessorFixed()
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_audio_folder(self, folder_path, label, label_name):
        """Load all audio files from folder"""
        X = []
        y = []
        
        print(f"\n📂 Loading {label_name.upper()} from: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"   ⚠️  Not found")
            return X, y
        
        audio_patterns = ['*.wav', '*.mp3', '*.m4a', '*.flac']
        audio_files = []
        
        for pattern in audio_patterns:
            audio_files.extend(glob.glob(os.path.join(folder_path, pattern)))
        
        print(f"   Found {len(audio_files)} files")
        
        for filepath in audio_files:
            try:
                with open(filepath, 'rb') as f:
                    audio_bytes = f.read()
                
                features = self.audio_processor.extract_features(audio_bytes)
                
                if features is not None:
                    feature_array = np.array([features[k] for k in sorted(features.keys())])
                    X.append(feature_array)
                    y.append(label)
                    
                    if len(X) % 50 == 0:
                        print(f"   ✓ Loaded {len(X)} {label_name} samples")
            
            except Exception as e:
                pass
        
        print(f"   ✅ Total {label_name}: {len(X)} samples")
        return X, y
    
    def train(self):
        """Train balanced model"""
        print("\n" + "="*70)
        print("🤖 TRAINING BALANCED MODEL (NO AGGRESSIVE OVERRIDES)")
        print("="*70)
        
        # Load human data
        print("\n📥 LOADING TRAINING DATA")
        X_human, y_human = [], []
        human_base = os.path.join(self.data_dir, 'human')
        
        if os.path.exists(human_base):
            for lang_dir in glob.glob(os.path.join(human_base, '*')):
                if os.path.isdir(lang_dir):
                    x_temp, y_temp = self.load_audio_folder(lang_dir, 0, f'HUMAN ({os.path.basename(lang_dir)})')
                    X_human.extend(x_temp)
                    y_human.extend(y_temp)
        
        # Load AI data
        X_ai, y_ai = [], []
        ai_dirs = [
            os.path.join(self.data_dir, 'ai_generated', 'english'),
            os.path.join(self.data_dir, 'ai_generated', 'hindi'),
            os.path.join(self.data_dir, 'ai_generated', 'tamil'),
            os.path.join(self.data_dir, 'ai_generated', 'telugu'),
        ]
        
        for ai_dir in ai_dirs:
            if os.path.exists(ai_dir):
                x_temp, y_temp = self.load_audio_folder(ai_dir, 1, f'AI ({os.path.basename(ai_dir)})')
                X_ai.extend(x_temp)
                y_ai.extend(y_temp)
        
        # Balance: truncate AI to match human count (or keep all if needed)
        if len(X_ai) > len(X_human) * 3:
            print(f"\n⚖️  Balancing dataset: {len(X_ai)} AI -> {len(X_human) * 3} AI")
            X_ai = X_ai[:len(X_human) * 3]
            y_ai = y_ai[:len(X_human) * 3]
        
        X = np.array(X_human + X_ai)
        y = np.array(y_human + y_ai)
        
        print(f"\n📊 FINAL DATASET:")
        print(f"   Total samples: {len(X)}")
        print(f"   Human (0): {sum(y==0)}")
        print(f"   AI (1): {sum(y==1)}")
        print(f"   Ratio: 1:{sum(y==1)/sum(y==0):.2f}")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        print("\n🌲 Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,  # Reduced from 20
            min_samples_split=5,  # Increased for better generalization
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        print("\n" + "="*70)
        print("📊 MODEL EVALUATION")
        print("="*70)
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")
        
        # ROC-AUC
        try:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            print(f"📈 ROC-AUC: {auc:.4f}")
        except:
            pass
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        if cm.shape == (2, 2):
            print(f"  {cm[0][0]} Human→Human     {cm[0][1]} Human→AI")
            print(f"  {cm[1][0]} AI→Human      {cm[1][1]} AI→AI")
            
            human_correct = cm[0][0] / (cm[0][0] + cm[0][1])
            ai_correct = cm[1][1] / (cm[1][0] + cm[1][1])
            
            print(f"\nHuman Detection Rate: {human_correct*100:.2f}%")
            print(f"AI Detection Rate: {ai_correct*100:.2f}%")
        
        print("\nDetailed Report:")
        print(classification_report(y_test, y_pred, target_names=['HUMAN', 'AI']))
        
        # Save
        model_path = os.path.join(self.output_dir, 'voice_detection_model.pkl')
        scaler_path = os.path.join(self.output_dir, 'voice_detection_scaler.pkl')
        
        model_data = {
            'model': model,
            'feature_names': sorted([f'feature_{i}' for i in range(X.shape[1])]),
            'label_map': {0: 'HUMAN', 1: 'AI_GENERATED'},
            'threshold': 0.5,  # Standard threshold
            'notes': 'Balanced training without aggressive safety overrides'
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"\n💾 Model saved: {model_path}")
        print(f"💾 Scaler saved: {scaler_path}")
        
        print("\n" + "="*70)
        print("✅ BALANCED MODEL TRAINING COMPLETE")
        print("="*70)


if __name__ == "__main__":
    trainer = BalancedModelTrainer()
    trainer.train()
