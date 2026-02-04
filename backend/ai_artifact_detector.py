"""
AI Artifact Detector - Detects AI-specific artifacts in audio

This module analyzes audio for characteristics that are common in AI-generated voices:
- Over-smooth harmonics (high spectral flatness)
- Too consistent waveforms (low zero-crossing rate)
- Missing high-frequency noise (low spectral centroid)
"""

import numpy as np
import librosa
from typing import Dict, Tuple
import io
import soundfile as sf


class AIArtifactDetector:
    """Detects AI-specific artifacts in audio signals"""
    
    def __init__(self):
        # Thresholds tuned for AI voice detection
        self.flatness_threshold = 0.04  # AI voices tend to be flatter
        self.zcr_threshold = 0.03       # AI voices have lower ZCR
        self.centroid_threshold = 1500  # AI voices have lower centroid
    
    def analyze(self, audio_bytes: bytes, sr: int = 16000) -> Dict[str, float]:
        """
        Analyze audio for AI artifacts
        
        Args:
            audio_bytes: Raw audio data
            sr: Sample rate (default 16000)
            
        Returns:
            Dictionary with artifact scores and metrics
        """
        # Load audio
        try:
            audio, sample_rate = sf.read(io.BytesIO(audio_bytes))
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if needed
            if sample_rate != sr:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=sr)
        except Exception as e:
            print(f"Error loading audio for artifact detection: {e}")
            return self._default_scores()
        
        # Extract artifact features
        try:
            # 1. Spectral Flatness (over-smooth harmonics indicator)
            flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
            
            # 2. Zero-Crossing Rate (consistency indicator)
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            # 3. Spectral Centroid (high-frequency noise indicator)
            centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            
            # Calculate artifact score (0-3)
            artifact_score = 0
            
            if flatness > self.flatness_threshold:
                artifact_score += 1  # Too smooth
            
            if zcr < self.zcr_threshold:
                artifact_score += 1  # Too consistent
            
            if centroid < self.centroid_threshold:
                artifact_score += 1  # Missing high frequencies
            
            return {
                'artifact_score': artifact_score,
                'spectral_flatness': float(flatness),
                'zero_crossing_rate': float(zcr),
                'spectral_centroid': float(centroid),
                'flatness_flag': flatness > self.flatness_threshold,
                'zcr_flag': zcr < self.zcr_threshold,
                'centroid_flag': centroid < self.centroid_threshold
            }
            
        except Exception as e:
            print(f"Error extracting artifact features: {e}")
            return self._default_scores()
    
    def _default_scores(self) -> Dict[str, float]:
        """Return default scores when analysis fails"""
        return {
            'artifact_score': 0,
            'spectral_flatness': 0.0,
            'zero_crossing_rate': 0.0,
            'spectral_centroid': 0.0,
            'flatness_flag': False,
            'zcr_flag': False,
            'centroid_flag': False
        }
    
    def get_explanation(self, artifacts: Dict[str, float]) -> str:
        """Generate human-readable explanation of artifact detection"""
        score = artifacts['artifact_score']
        
        if score == 0:
            return "No AI artifacts detected - natural voice characteristics"
        elif score == 1:
            flags = []
            if artifacts['flatness_flag']:
                flags.append("smooth harmonics")
            if artifacts['zcr_flag']:
                flags.append("consistent waveform")
            if artifacts['centroid_flag']:
                flags.append("limited frequency range")
            return f"Minor AI artifact: {', '.join(flags)}"
        elif score == 2:
            return "Moderate AI artifacts - likely synthetic voice"
        else:  # score == 3
            return "Strong AI artifacts - highly likely synthetic voice"
