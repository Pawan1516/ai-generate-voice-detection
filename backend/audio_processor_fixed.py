"""
FIXED Audio Feature Extraction Module
Ensures IDENTICAL preprocessing between training and inference
Uses consistent librosa settings locked across all operations
"""

import numpy as np
import librosa
import soundfile as sf
from io import BytesIO


class AudioProcessorFixed:
    """
    FIXED: Extract audio features with LOCKED settings for training and inference
    This ensures no preprocessing mismatch
    """
    
    # ✅ LOCKED SETTINGS (DO NOT CHANGE)
    SR = 16000              # Sample rate
    DURATION = 4            # Pad/trim to exactly 4 seconds
    N_MELS = 128            # Mel spectrogram bins
    N_FFT = 1024            # FFT window
    HOP_LENGTH = 256        # Hop length
    N_MFCC = 13             # MFCC coefficients
    
    def __init__(self):
        self.sr = self.SR
        self.duration = self.DURATION
    
    def load_audio_consistent(self, audio_input, sr=None):
        """
        ✅ STEP 1: CONSISTENT AUDIO LOADING
        Works for both file paths and bytes
        Always returns normalized audio at exact duration
        """
        if sr is None:
            sr = self.sr
        
        try:
            # Load audio
            if isinstance(audio_input, (str, bytes)):
                if isinstance(audio_input, str):
                    # File path
                    y, _ = librosa.load(audio_input, sr=sr, mono=True)
                else:
                    # Bytes (from HTTP request)
                    y, _ = librosa.load(BytesIO(audio_input), sr=sr, mono=True)
            else:
                raise ValueError("Invalid audio input type")
            
            # Ensure exact duration (CRITICAL for consistency)
            target_len = sr * self.duration  # 4 seconds at 16kHz = 64000 samples
            
            if len(y) > target_len:
                # Trim to exact length
                y = y[:target_len]
            elif len(y) < target_len:
                # Pad with zeros to exact length
                y = np.pad(y, (0, target_len - len(y)), mode='constant')
            
            return y, sr
        
        except Exception as e:
            print(f"❌ Audio loading failed: {e}")
            raise
    
    def extract_mel_spectrogram(self, audio, sr=None):
        """
        ✅ STEP 2: MEL SPECTROGRAM WITH LOCKED SETTINGS
        These parameters MUST match training exactly
        """
        if sr is None:
            sr = self.sr
        
        try:
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=self.N_MELS,
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH
            )
            
            # Convert to dB scale
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # Normalize to 0-1 range
            mel_normalized = (mel_db + 80) / 80  # Assuming range -80 to 0 dB
            mel_normalized = np.clip(mel_normalized, 0, 1)
            
            return mel_normalized
        
        except Exception as e:
            print(f"❌ Mel spectrogram extraction failed: {e}")
            raise
    
    def extract_spectral_flatness(self, audio):
        """
        ✅ STEP 6 HELPER: Spectral flatness for human voice detection override
        Lower flatness = more tonal = human voice
        """
        try:
            flatness = librosa.feature.spectral_flatness(y=audio)
            return float(np.mean(flatness))
        except Exception:
            return 0.5  # Neutral default
    
    def extract_features(self, audio_bytes):
        """
        ✅ MAIN: Extract 42 features with LOCKED preprocessing
        Returns feature dictionary with consistent ordering
        """
        try:
            if audio_bytes is None or len(audio_bytes) == 0:
                return None
            
            # ✅ STEP 1: Consistent loading
            y, sr = self.load_audio_consistent(audio_bytes)
            
            # ✅ STEP 2: Locked mel spectrogram
            mel = self.extract_mel_spectrogram(y, sr)
            
            features = {}
            
            # 1. === MFCC (13 means + 13 stds = 26 features) ===
            try:
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.N_MFCC)
                
                for i in range(self.N_MFCC):
                    features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
                    features[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
            except Exception as e:
                print(f"MFCC extraction failed: {e}")
                for i in range(self.N_MFCC):
                    features[f'mfcc_{i}_mean'] = 0.0
                    features[f'mfcc_{i}_std'] = 0.0
            
            # 2. === Spectral Features (4 features) ===
            try:
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
                features['spectral_centroid_std'] = float(np.std(spectral_centroid))
                
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
                features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            except Exception:
                features['spectral_centroid_mean'] = 0.0
                features['spectral_centroid_std'] = 0.0
                features['spectral_rolloff_mean'] = 0.0
                features['spectral_rolloff_std'] = 0.0
            
            # 3. === Zero Crossing Rate (2 features) ===
            try:
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                features['zcr_mean'] = float(np.mean(zcr))
                features['zcr_std'] = float(np.std(zcr))
            except Exception:
                features['zcr_mean'] = 0.0
                features['zcr_std'] = 0.0
            
            # 4. === Energy / RMS (4 features) ===
            try:
                rms = librosa.feature.rms(y=y)[0]
                features['rms_mean'] = float(np.mean(rms))
                features['rms_std'] = float(np.std(rms))
                features['rms_min'] = float(np.min(rms))
                features['rms_max'] = float(np.max(rms))
            except Exception:
                features['rms_mean'] = 0.0
                features['rms_std'] = 0.0
                features['rms_min'] = 0.0
                features['rms_max'] = 0.0
            
            # 5. === Onset Strength (2 features) ===
            try:
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                features['onset_mean'] = float(np.mean(onset_env))
                features['onset_std'] = float(np.std(onset_env))
            except Exception:
                features['onset_mean'] = 0.0
                features['onset_std'] = 0.0
            
            # 6. === Duration (1 feature) ===
            features['duration'] = float(len(y) / sr)
            
            # Pad to 42 features exactly
            num_features = len(features)
            if num_features < 42:
                for i in range(42 - num_features):
                    features[f'padding_{i}'] = 0.0
            
            return features
        
        except Exception as e:
            print(f"❌ Feature extraction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
