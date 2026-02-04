"""
Audio Feature Extraction Module
Extracts MFCC, spectral, and temporal features from audio files
"""

import numpy as np
import librosa
import soundfile as sf
from io import BytesIO


class AudioProcessor:
    """Extract audio features for AI vs Human voice detection"""
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def detect_silence(self, audio_bytes, silence_threshold=0.01):
        """
        Detect if audio is mostly silence
        
        Args:
            audio_bytes: Raw audio data
            silence_threshold: RMS threshold below which is considered silence
            
        Returns:
            tuple: (is_silent: bool, silence_percentage: float)
        """
        try:
            # Load audio
            y, sr = librosa.load(BytesIO(audio_bytes), sr=self.sr)
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y)[0]
            
            # Count silent frames (normalized RMS < threshold)
            # Typically RMS is not normalized 0-1 easily without reference.
            # But let's assume threshold is small (e.g., 0.01)
            
            # Normalize RMS to 0-1 roughly
            if np.max(rms) > 0:
                rms_norm = rms / np.max(rms)
            else:
                rms_norm = rms
                
            silent_frames = np.sum(rms < silence_threshold)
            total_frames = len(rms)
            
            if total_frames == 0:
                return True, 1.0
                
            silence_percentage = silent_frames / total_frames
            
            # If > 80% silence, flag as silent
            is_silent = silence_percentage > 0.8
            
            return is_silent, silence_percentage
            
        except Exception as e:
            # Default to not silent on error
            print(f"Error in detect_silence: {e}")
            return False, 0.0

    def extract_features(self, audio_bytes):
        """
        FIXED STANDARD PIPELINE:
        1. Load as 16kHz mono
        2. Trim/Pad to 4 seconds (64000 samples)
        3. Mel Spectrogram (128 mels, 1024 n_fft)
        4. Power to DB
        5. Normalize
        """
        try:
            if audio_bytes is None or len(audio_bytes) == 0:
                return None
            
            # 1. Load 16kHz Mono
            y, sr = librosa.load(BytesIO(audio_bytes), sr=16000, mono=True)
            
            # 2. Fixed Duration (4s = 64000 samples)
            target_len = 64000
            if len(y) > target_len:
                y = y[:target_len]
            else:
                y = np.pad(y, (0, target_len - len(y)))
            
            # 3. Mel Spectrogram
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=256
            )
            
            # 4. Convert to DB
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # 5. Normalize (-80 to 0 scale simplified)
            mel_norm = (mel_db + 80) / 80
            
            # Flatten to vector for Random Forest
            # 128 mels * ~251 frames (for 4s at 256 hop)
            # To keep features manageable and fixed-size: 
            # we'll use statistical summaries of the mel bands
            features = {}
            for i in range(128):
                features[f'mel_{i}_mean'] = float(np.mean(mel_norm[i]))
                features[f'mel_{i}_std'] = float(np.std(mel_norm[i]))
            
            return features
        
        except Exception as e:
            print(f"Standard Preprocess Error: {str(e)}")
            return None
        
        except Exception as e:
            print(f"AudioProcessor Error: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return None instead of crashing
            return None
