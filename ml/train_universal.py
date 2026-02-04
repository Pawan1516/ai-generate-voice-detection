"""
🚀 TRAIN UNIVERSAL MODEL (Robust Random Forest)
The most stable and GENERALIZABLE configuration.
Uses 1000 trees but with regularization to prevent overfitting on specific artifacts.
"""

import os
import sys
import numpy as np
import pickle
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Fix for Windows Unicode errors
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

try:
    from audio_processor import AudioProcessor
except ImportError:
    print("❌ Could not import backend modules.")
    sys.exit(1)

class UniversalTrainer:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.output_dir = os.path.join(self.base_dir, 'backend', 'models')
        self.audio_processor = AudioProcessor()
        self.scaler = StandardScaler()
        os.makedirs(self.output_dir, exist_ok=True)

    def load_files(self, directory, label):
        features = []
        labels = []
        # Support all formats
        files = glob.glob(os.path.join(directory, '**', '*.wav'), recursive=True)
        files.extend(glob.glob(os.path.join(directory, '**', '*.mp3'), recursive=True))
        
        print(f"   Found {len(files)} files in {directory}")
        
        for i, f in enumerate(files):
            try:
                with open(f, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                
                feat_dict = self.audio_processor.extract_features(audio_bytes)
                if feat_dict:
                    # Map to ordered feature vector (ensure order is same)
                    feat_vec = [feat_dict[f'mel_{j}_mean'] for j in range(128)] + \
                               [feat_dict[f'mel_{j}_std'] for j in range(128)]
                    features.append(feat_vec)
                    labels.append(label)
            except Exception:
                pass
            
            if (i + 1) % 500 == 0: # Changed from 200 to 500
                print(f"     Processed {i+1}/{len(files)}")
                
        return np.array(features), np.array(labels)

    def train(self):
        print("="*60)
        print("🌍  STARTING UNIVERSAL TRAINING (Robust config)")
        print("="*60)
        
        # 1. Load Data
        human_dirs = [
            os.path.join(self.data_dir, 'train', 'human'),
            os.path.join(self.data_dir, 'validation', 'human')
        ]
        ai_dir = os.path.join(self.data_dir, 'train', 'ai_generated')
        
        print("\n1️⃣  Loading ALL Data (including Validation set for more Humans)...")
        
        X_h_list, y_h_list = [], []
        for h_dir in human_dirs:
            if os.path.exists(h_dir):
                xh, yh = self.load_files(h_dir, 0)
                if len(xh) > 0:
                    X_h_list.append(xh)
                    y_h_list.append(yh)
        
        X_h = np.vstack(X_h_list) if X_h_list else np.empty((0, 256))
        y_h = np.concatenate(y_h_list) if y_h_list else np.empty((0,))
        
        X_a, y_a = self.load_files(ai_dir, 1)
        
        # 2. Balanced Selection
        print(f"\n2️⃣  Balancing Dataset (Human: {len(X_h)} vs AI: {len(X_a)})...")
        
        # Limit AI samples to prevent overwhelming the small human set
        # But pick the most diverse ones (original + some augmented)
        max_ai = min(len(X_a), len(X_h) * 6) # Increased to 6:1 to favor AI sensitivity
        if len(X_a) > max_ai:
            np.random.seed(42)
            indices = np.random.choice(len(X_a), max_ai, replace=False)
            X_a = X_a[indices]
            y_a = y_a[indices]
        
        X = np.vstack([X_h, X_a])
        y = np.concatenate([y_h, y_a])
        
        print(f"   Final Training Set Shape: {X.shape}")
        
        # 3. Split & Scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 4. Train Model (The Universal Config)
        print("\n4️⃣  Training Universal Forest (Optimized for AI sensitivity)...")
        clf = RandomForestClassifier(
            n_estimators=1000,
            max_depth=20,            # Increased depth for better feature separation
            min_samples_split=6,     # Lowered to allow more specific splits
            min_samples_leaf=2,      # Lowered for higher specificity
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        
        clf.fit(X_train_scaled, y_train)
        
        # 5. Evaluate
        print("\n4️⃣  Evaluation:")
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"   Accuracy: {acc:.2%}")
        print("\n   Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # 6. Save
        print("\n5️⃣  Saving Universal Model...")
        with open(os.path.join(self.output_dir, 'voice_detection_model.pkl'), 'wb') as f:
            pickle.dump(clf, f)
        with open(os.path.join(self.output_dir, 'voice_detection_scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
            
        print("   ✅ Universal Model Saved!")

if __name__ == "__main__":
    UniversalTrainer().train()
