"""
🚀 TRAIN AI vs HUMAN VOICE DETECTION MODEL
Optimized for hackathon - Fast & Accurate
"""

import os
import sys
import numpy as np
import pickle
import glob
from pathlib import Path
import time
from datetime import datetime

# Fix for Windows Unicode errors
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )
    from audio_processor import AudioProcessor
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install -r backend/requirements.txt")
    sys.exit(1)

class VoiceDetectionModelTrainer:
    """Train AI vs Human Voice Detection Model"""
    
    def __init__(self, data_dir='../data', output_dir='../backend/models'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.audio_processor = AudioProcessor()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        os.makedirs(output_dir, exist_ok=True)
    
    def load_audio_features(self, audio_file, label):
        """Extract features from audio file"""
        try:
            # Read audio file as bytes
            with open(audio_file, 'rb') as f:
                audio_bytes = f.read()
            
            features_dict = self.audio_processor.extract_features(audio_bytes)
            
            if features_dict is not None and len(features_dict) > 0:
                # Convert dict to ordered feature vector
                features_list = list(features_dict.values())
                return features_list, label
        except Exception as e:
            print(f"Error loading {os.path.basename(audio_file)}: {e}")
            pass
        return None, None
    
    def collect_training_data(self, ai_dir='train/ai_generated', human_dir='train/human', limit_per_category=None):
        """Collect and load training data"""
        print("\n" + "="*70)
        print("📊 COLLECTING TRAINING DATA")
        print("="*70)
        
        X = []
        y = []
        
        # Load AI-generated voices (label = 1)
        # Check multiple possible directories
        ai_dirs_to_check = [ai_dir, 'train/ai']
        
        print(f"\n🤖 Loading AI-generated voices...")
        ai_files = []
        
        for adir in ai_dirs_to_check:
            ai_path = os.path.join(self.data_dir, adir)
            if os.path.exists(ai_path):
                print(f"   Checking {adir}...")
                found = glob.glob(os.path.join(ai_path, '**', '*.wav'), recursive=True)
                found.extend(glob.glob(os.path.join(ai_path, '**', '*.mp3'), recursive=True))
                print(f"   Found {len(found)} files in {adir}")
                ai_files.extend(found)
        
        ai_files = list(set(ai_files))  # Remove duplicates
        
        if limit_per_category:
            ai_files = ai_files[:limit_per_category]
        
        print(f"   Total unique AI audio files: {len(ai_files)}")
        
        loaded_count = 0
        for idx, audio_file in enumerate(ai_files, 1):
            features, label = self.load_audio_features(audio_file, 1)  # AI = 1
            if features is not None:
                X.append(features)
                y.append(label)
                loaded_count += 1
            
            if idx % 50 == 0:
                print(f"   Processed {idx}/{len(ai_files)} AI files ({loaded_count} successful)")
        
        print(f"   ✓ Loaded {loaded_count} AI voice samples")
        
        # Load Human voices (label = 0)
        human_path = os.path.join(self.data_dir, human_dir)
        if os.path.exists(human_path):
            print(f"\n👤 Loading human voices from {human_dir}...")
            human_files = glob.glob(os.path.join(human_path, '**', '*.wav'), recursive=True)
            human_files.extend(glob.glob(os.path.join(human_path, '**', '*.mp3'), recursive=True))
            human_files = list(set(human_files))  # Remove duplicates
            
            if limit_per_category:
                human_files = human_files[:limit_per_category]
            
            print(f"   Found {len(human_files)} human audio files")
            
            loaded_count = 0
            for idx, audio_file in enumerate(human_files, 1):
                features, label = self.load_audio_features(audio_file, 0)  # Human = 0
                if features is not None:
                    X.append(features)
                    y.append(label)
                    loaded_count += 1
                
                if idx % 50 == 0:
                    print(f"   Processed {idx}/{len(human_files)} human files ({loaded_count} successful)")
            
            print(f"   ✓ Loaded {loaded_count} human voice samples")
        
        if len(X) == 0:
            print("\n❌ No audio files found. Please check data directories.")
            return None, None
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n📈 Dataset Summary:")
        print(f"   Total samples: {len(X)}")
        print(f"   AI voices: {np.sum(y == 1)}")
        print(f"   Human voices: {np.sum(y == 0)}")
        print(f"   Features per sample: {X.shape[1]}")
        
        return X, y
    
    def train(self, ai_dir='train/ai_generated', human_dir='train/human', 
              test_size=0.2, limit_per_category=None, model_type='random_forest'):
        """Train the detection model"""
        
        print("\n" + "="*70)
        print("🎯 TRAINING AI VS HUMAN VOICE DETECTION MODEL")
        print("="*70)
        
        # Collect data
        X, y = self.collect_training_data(ai_dir, human_dir, limit_per_category)
        
        if X is None:
            print("\n❌ Training failed: No data collected")
            return False
        
        # Split data
        print(f"\n🔄 Splitting data (train: {100-int(test_size*100)}%, test: {int(test_size*100)}%)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        print("⚙️  Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"\n🤖 Training {model_type} classifier...")
        start_time = time.time()
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=0
            )
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        print(f"   ✓ Training completed in {training_time:.2f} seconds")
        
        # Evaluate
        print("\n📊 EVALUATION RESULTS")
        print("-" * 70)
        
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"\n🎯 Accuracy:")
        print(f"   Training: {train_acc:.2%}")
        print(f"   Testing:  {test_acc:.2%}")
        
        print(f"\n📈 Classification Metrics (Test Set):")
        print(f"   Precision: {precision_score(y_test, y_pred_test):.2%}")
        print(f"   Recall:    {recall_score(y_test, y_pred_test):.2%}")
        print(f"   F1-Score:  {f1_score(y_test, y_pred_test):.2%}")
        
        print(f"\n🔍 Confusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"   True Negatives:  {cm[0][0]}")
        print(f"   False Positives: {cm[0][1]}")
        print(f"   False Negatives: {cm[1][0]}")
        print(f"   True Positives:  {cm[1][1]}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_test, 
                                   target_names=['Human', 'AI'],
                                   digits=3))
        
        # Save model
        self.save_model()
        
        print("\n" + "="*70)
        print(f"✓ MODEL TRAINING COMPLETE")
        print("="*70)
        
        return True
    
    def save_model(self):
        """Save trained model and scaler"""
        model_path = os.path.join(self.output_dir, 'voice_detection_model.pkl')
        scaler_path = os.path.join(self.output_dir, 'voice_detection_scaler.pkl')
        
        print(f"\n💾 Saving model...")
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"   ✓ Model saved to {model_path}")
        print(f"   ✓ Scaler saved to {scaler_path}")

def main():
    """Main training routine"""
    
    print("\n" + "# "*30)
    print("AI vs HUMAN VOICE DETECTION - MODEL TRAINING")
    print("# "*30)
    
    # Calculate base directory (project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    trainer = VoiceDetectionModelTrainer(
        data_dir=os.path.join(base_dir, 'data'),
        output_dir=os.path.join(base_dir, 'backend', 'models')
    )
    
    # Train with full dataset (or limited for quick testing)
    success = trainer.train(
        ai_dir='train/ai_generated',
        human_dir='train/human',
        test_size=0.2,
        limit_per_category=None,  # Set to number to limit (e.g., 500)
        model_type='random_forest'
    )
    
    if success:
        print("\n[SUCCESS] Training successful! Model is ready for inference.")
        return 0
    else:
        print("\n[FAILED] Training failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
