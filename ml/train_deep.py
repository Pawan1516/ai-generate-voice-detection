"""
🚀 DEEP TRAINING: Neural Network Classifier (TensorFlow/Keras)
Architecture: 4-Layer MLP with BatchNorm and Dropout
Target: improved generalizability for high-quality AI voices.
"""

import os
import sys
import numpy as np
import pickle
import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Fix for Windows Unicode errors
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

try:
    from audio_processor import AudioProcessor
except ImportError:
    print("❌ Could not import backend modules.")
    sys.exit(1)

class DeepTrainer:
    def __init__(self, data_dir='data', output_dir='backend/models'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.ap = AudioProcessor()
        self.scaler = StandardScaler()
        os.makedirs(output_dir, exist_ok=True)

    def load_files(self, directory, label):
        files = glob.glob(os.path.join(directory, '**', '*.wav'), recursive=True)
        files.extend(glob.glob(os.path.join(directory, '**', '*.mp3'), recursive=True))
        
        print(f"   Found {len(files)} files in {directory}")
        features = []
        labels = []
        
        # Limit processing for speed if needed, but for "Deep Train" we want more
        for i, f in enumerate(files):
            try:
                with open(f, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                
                feat_dict = self.ap.extract_features(audio_bytes)
                if feat_dict:
                    # Convert dict to vector (ensure order is same)
                    feat_vec = [feat_dict[f'mel_{j}_mean'] for j in range(128)] + \
                               [feat_dict[f'mel_{j}_std'] for j in range(128)]
                    features.append(feat_vec)
                    labels.append(label)
            except Exception:
                pass
            
            if (i + 1) % 500 == 0:
                print(f"     Processed {i+1}/{len(files)}")
                
        return np.array(features), np.array(labels)

    def train(self):
        print("="*60)
        print("🧠  STARTING DEEP TRAINING (Neural Network)")
        print("="*60)
        
        # 1. Load Data
        human_dir = os.path.join(self.data_dir, 'train', 'human')
        ai_dir = os.path.join(self.data_dir, 'train', 'ai_generated')
        
        print("\n1️⃣  Loading Data (including Augmented Samples)...")
        X_h, y_h = self.load_files(human_dir, 0)
        X_a, y_a = self.load_files(ai_dir, 1)
        
        if len(X_h) == 0 or len(X_a) == 0:
            print("❌ Error: Missing data.")
            return

        # Combine
        X = np.vstack([X_h, X_a])
        y = np.concatenate([y_h, y_a])
        
        print(f"\n2️⃣  Total Dataset: {X.shape} (AI: {np.sum(y==1)}, Human: {np.sum(y==0)})")
        
        # 2. Split & Scale
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # 3. Build Model
        print("\n3️⃣  Building Deep Neural Network...")
        model = Sequential([
            # Input Layer (256 features)
            Dense(512, activation='relu', input_shape=(X.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden Layers
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(64, activation='relu'),
            
            # Output Layer
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # 4. Train
        print("\n4️⃣  Training...")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint(os.path.join(self.output_dir, 'voice_detection_deep_model.h5'), 
                            save_best_only=True)
        ]
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # 5. Save Scaler (Neural Network needs the specific scaler)
        with open(os.path.join(self.output_dir, 'voice_detection_scaler_deep.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
            
        print(f"\n✅ Deep Training Complete. Model & Scaler saved.")
        
        # 6. Evaluation
        y_pred = (model.predict(X_val_scaled) > 0.5).astype(int)
        acc = accuracy_score(y_val, y_pred)
        print(f"\n🎯 Final Validation Accuracy: {acc:.2%}")
        print("\n   Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))

if __name__ == "__main__":
    DeepTrainer().train()
