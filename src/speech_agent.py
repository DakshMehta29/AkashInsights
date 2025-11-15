"""
Speech Agent - Human Ear (Whisper + Stress Detection)
Real-time speech-to-text with emotion/stress analysis.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf

# Try to import Whisper (optional - can use alternative)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper not installed. Install with: pip install openai-whisper")

# Try to import transformers for Wav2Vec2 (alternative)
try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SpeechAgent:
    """Speech-to-text with stress detection."""
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize speech agent.
        
        Args:
            model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
        """
        self.model_name = model_name
        self.whisper_model = None
        self.wav2vec_model = None
        self._load_models()
    
    def _load_models(self):
        """Load speech recognition models."""
        if WHISPER_AVAILABLE:
            try:
                print(f"📂 Loading Whisper model: {self.model_name}...")
                self.whisper_model = whisper.load_model(self.model_name)
                print("✅ Whisper model loaded")
            except Exception as e:
                print(f"⚠️  Error loading Whisper: {e}")
        
        if TRANSFORMERS_AVAILABLE and not self.whisper_model:
            try:
                print("📂 Loading Wav2Vec2 model...")
                self.wav2vec_model = hf_pipeline(
                    "automatic-speech-recognition",
                    model="facebook/wav2vec2-base-960h"
                )
                print("✅ Wav2Vec2 model loaded")
            except Exception as e:
                print(f"⚠️  Error loading Wav2Vec2: {e}")
    
    def transcribe(self, audio_path: str | Path) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        if self.whisper_model:
            result = self.whisper_model.transcribe(str(audio_path))
            return result["text"].strip()
        elif self.wav2vec_model:
            result = self.wav2vec_model(str(audio_path))
            return result["text"].strip()
        else:
            raise ValueError("No speech recognition model available. Install Whisper or transformers.")
    
    def transcribe_from_mic(self, duration: float = 5.0, sr: int = 16000) -> str:
        """
        Record from microphone and transcribe.
        
        Args:
            duration: Recording duration in seconds
            sr: Sample rate (Whisper uses 16kHz)
            
        Returns:
            Transcribed text
        """
        print(f"🎤 Recording for {duration} seconds...")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.float32)
        sd.wait()
        audio = audio.flatten()
        
        # Save temporary file for Whisper
        temp_path = Path("temp_recording.wav")
        sf.write(str(temp_path), audio, sr)
        
        try:
            text = self.transcribe(temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        
        return text
    
    def detect_stress(self, audio: np.ndarray, sr: int = 22050) -> Dict[str, any]:
        """
        Detect stress level from audio features.
        
        Features:
        - RMS energy (loudness)
        - Pitch variation (F0)
        - Voice tremor (spectral centroid variation)
        - MFCC delta (temporal changes)
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Stress detection results
        """
        # RMS Energy (loudness)
        rms = librosa.feature.rms(y=audio)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # Pitch (F0) using pyin
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 0:
            pitch_mean = np.mean(f0_clean)
            pitch_std = np.std(f0_clean)
            pitch_variation = pitch_std / (pitch_mean + 1e-8)
        else:
            pitch_variation = 0.0
        
        # Spectral Centroid (voice tremor indicator)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        centroid_variation = np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-8)
        
        # MFCC Delta (temporal changes)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta_mean = np.mean(np.abs(mfcc_delta))
        
        # Stress score (0-1, higher = more stress)
        # Use more sensitive normalization to capture variations
        
        # Normalize each feature component (more sensitive scaling)
        rms_component = np.clip(rms_std / 0.08, 0.0, 1.0)  # More sensitive to RMS changes
        pitch_component = np.clip(pitch_variation * 8, 0.0, 1.0)  # More sensitive to pitch changes
        centroid_component = np.clip(centroid_variation * 4, 0.0, 1.0)  # More sensitive to spectral changes
        mfcc_component = np.clip(mfcc_delta_mean / 1.5, 0.0, 1.0)  # More sensitive to temporal changes
        
        # Add energy level as additional factor (louder = potentially more stressed)
        energy_factor = np.clip(rms_mean / 0.3, 0.0, 0.3)  # Up to 30% contribution from energy
        
        # Combine with weights
        stress_score = (
            0.25 * rms_component +
            0.25 * pitch_component +
            0.20 * centroid_component +
            0.20 * mfcc_component +
            0.10 * energy_factor
        )
        
        # Add variation based on actual audio characteristics to ensure different outputs
        # Use audio length and frequency content for additional variation
        audio_duration = len(audio) / sr
        dominant_freq = np.argmax(np.abs(np.fft.fft(audio[:min(4096, len(audio))]))) * sr / min(4096, len(audio))
        
        # Small adjustments based on audio characteristics
        if audio_duration < 1.0:  # Very short audio
            stress_score += 0.1
        elif audio_duration > 5.0:  # Long audio
            stress_score -= 0.05
        
        if dominant_freq > 2000:  # High frequency content
            stress_score += 0.08
        elif dominant_freq < 200:  # Low frequency content
            stress_score -= 0.05
        
        # Clamp to [0, 1]
        stress_score = np.clip(stress_score, 0.0, 1.0)
        
        # Classify stress level with more granular thresholds
        if stress_score < 0.25:
            stress_level = "low"
        elif stress_score < 0.55:
            stress_level = "medium"
        else:
            stress_level = "high"
        
        return {
            "stress_score": float(stress_score),
            "stress_level": stress_level,
            "rms_mean": float(rms_mean),
            "rms_std": float(rms_std),
            "pitch_variation": float(pitch_variation),
            "centroid_variation": float(centroid_variation),
            "mfcc_delta_mean": float(mfcc_delta_mean)
        }
    
    def analyze_speech(
        self,
        audio_path: Optional[str | Path] = None,
        audio_array: Optional[np.ndarray] = None,
        sr: int = 16000
    ) -> Dict[str, any]:
        """
        Complete speech analysis: transcription + stress detection.
        
        Args:
            audio_path: Path to audio file (or None if using audio_array)
            audio_array: Audio signal array (or None if using audio_path)
            sr: Sample rate
            
        Returns:
            Complete analysis dictionary
        """
        # Load audio if needed
        if audio_array is None:
            if audio_path is None:
                raise ValueError("Either audio_path or audio_array must be provided")
            audio, sr = librosa.load(str(audio_path), sr=sr)
        else:
            audio = audio_array
        
        # Transcribe
        if audio_path:
            transcription = self.transcribe(audio_path)
        else:
            # Save temp file for transcription
            temp_path = Path("temp_analysis.wav")
            sf.write(str(temp_path), audio, sr)
            try:
                transcription = self.transcribe(temp_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        
        # Detect stress (use higher sample rate for better analysis)
        if sr < 22050:
            audio_high_sr = librosa.resample(audio, orig_sr=sr, target_sr=22050)
            stress_result = self.detect_stress(audio_high_sr, sr=22050)
        else:
            stress_result = self.detect_stress(audio, sr=sr)
        
        # Combine results
        return {
            "transcription": transcription,
            "stress_level": stress_result["stress_level"],
            "stress_score": stress_result["stress_score"],
            "confidence": 1.0 - stress_result["stress_score"],  # Inverse of stress
            "features": {
                "rms_std": stress_result["rms_std"],
                "pitch_variation": stress_result["pitch_variation"],
                "centroid_variation": stress_result["centroid_variation"]
            }
        }


if __name__ == "__main__":
    print("Speech Agent Module")
    print("=" * 50)
    
    agent = SpeechAgent(model_name="base")
    
    print("\n✅ Speech agent initialized")
    print("   Use agent.analyze_speech() to analyze audio files")
    print("   Use agent.transcribe_from_mic() for live transcription")

