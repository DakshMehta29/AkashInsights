"""
Acoustic Preprocessing Module - Machine Ear Agent
Extracts MFCCs, Mel Spectrograms, and FFT features from audio/sensor data.
Includes data augmentation for robust training.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional

import librosa
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler

# Consistent randomness
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_audio(file_path: str | Path, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load audio file using librosa.
    
    Args:
        file_path: Path to audio file (.wav, .mp3, etc.)
        sr: Target sample rate
        
    Returns:
        audio: Audio signal array
        sr: Sample rate
    """
    audio, sr = librosa.load(str(file_path), sr=sr)
    return audio, sr


def load_sensor_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load CMAPSS-style sensor data from .txt file.
    Converts sensor readings to pseudo-audio for acoustic analysis.
    
    Args:
        file_path: Path to .txt sensor data
        
    Returns:
        DataFrame with sensor columns
    """
    # NASA CMAPSS column names
    columns = [
        "unit_number", "time_in_cycles",
        "op_setting_1", "op_setting_2", "op_setting_3"
    ] + [f"sensor_{i}" for i in range(1, 22)]
    
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=columns)
    return df


def sensor_to_audio(sensor_data: pd.DataFrame, sensor_col: str = "sensor_1") -> np.ndarray:
    """
    Convert sensor time series to audio-like signal for acoustic analysis.
    
    Args:
        sensor_data: DataFrame with sensor readings
        sensor_col: Column name to convert
        
    Returns:
        Audio-like signal array
    """
    signal_data = sensor_data[sensor_col].values.astype(np.float32)
    # Normalize to [-1, 1] range
    signal_data = (signal_data - signal_data.mean()) / (signal_data.std() + 1e-8)
    return signal_data


def extract_mfcc(
    audio: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128
) -> np.ndarray:
    """
    Extract Mel-Frequency Cepstral Coefficients (MFCCs).
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel filter banks
        
    Returns:
        MFCC features (n_mfcc, time_frames)
    """
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    return mfccs


def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128
) -> np.ndarray:
    """
    Extract Mel Spectrogram.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel filter banks
        
    Returns:
        Mel spectrogram (n_mels, time_frames)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    # Convert to dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def extract_fft_features(
    audio: np.ndarray,
    n_fft: int = 2048
) -> np.ndarray:
    """
    Extract FFT-based spectral features.
    
    Args:
        audio: Audio signal
        n_fft: FFT window size
        
    Returns:
        FFT magnitude spectrum
    """
    fft = np.fft.fft(audio, n=n_fft)
    magnitude = np.abs(fft[:n_fft // 2])
    return magnitude


def time_stretch(audio: np.ndarray, rate: float = 1.2) -> np.ndarray:
    """
    Time-stretch augmentation (speed up/slow down without pitch change).
    
    Args:
        audio: Audio signal
        rate: Stretch rate (>1.0 = faster, <1.0 = slower)
        
    Returns:
        Time-stretched audio
    """
    return librosa.effects.time_stretch(audio, rate=rate)


def add_noise(audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """
    Add Gaussian noise for augmentation.
    
    Args:
        audio: Audio signal
        noise_factor: Noise strength (0.0-1.0)
        
    Returns:
        Noisy audio
    """
    noise = np.random.randn(len(audio)) * noise_factor
    return audio + noise


def pitch_shift(audio: np.ndarray, sr: int = 22050, n_steps: int = 2) -> np.ndarray:
    """
    Pitch shift augmentation.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_steps: Number of semitones to shift
        
    Returns:
        Pitch-shifted audio
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def random_gain(audio: np.ndarray, min_gain: float = 0.5, max_gain: float = 1.5) -> np.ndarray:
    """
    Random gain augmentation.
    
    Args:
        audio: Audio signal
        min_gain: Minimum gain multiplier
        max_gain: Maximum gain multiplier
        
    Returns:
        Gain-adjusted audio
    """
    gain = np.random.uniform(min_gain, max_gain)
    return audio * gain


def augment_audio(audio: np.ndarray, sr: int = 22050) -> list[np.ndarray]:
    """
    Apply multiple augmentations to create augmented samples.
    
    Args:
        audio: Original audio signal
        sr: Sample rate
        
    Returns:
        List of augmented audio samples
    """
    augmented = [audio]  # Include original
    
    # Time stretch
    augmented.append(time_stretch(audio, rate=1.1))
    augmented.append(time_stretch(audio, rate=0.9))
    
    # Noise
    augmented.append(add_noise(audio, noise_factor=0.005))
    augmented.append(add_noise(audio, noise_factor=0.01))
    
    # Pitch shift
    augmented.append(pitch_shift(audio, sr=sr, n_steps=1))
    augmented.append(pitch_shift(audio, sr=sr, n_steps=-1))
    
    # Random gain
    augmented.append(random_gain(audio, 0.7, 1.3))
    
    return augmented


def extract_all_features(
    audio: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 13,
    n_mels: int = 128
) -> dict[str, np.ndarray]:
    """
    Extract all acoustic features (MFCC, Mel Spectrogram, FFT).
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of mel filter banks
        
    Returns:
        Dictionary with all feature arrays
    """
    features = {
        "mfcc": extract_mfcc(audio, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels),
        "mel_spectrogram": extract_mel_spectrogram(audio, sr=sr, n_mels=n_mels),
        "fft": extract_fft_features(audio)
    }
    return features


def pad_or_truncate(array: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or truncate array to target length.
    
    Args:
        array: Input array
        target_length: Target length
        
    Returns:
        Padded/truncated array
    """
    if len(array) > target_length:
        return array[:target_length]
    elif len(array) < target_length:
        padding = target_length - len(array)
        return np.pad(array, (0, padding), mode="constant")
    return array


def prepare_features(
    audio: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    time_frames: int = 128
) -> np.ndarray:
    """
    Prepare mel spectrogram features for model input.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mels: Number of mel filter banks
        time_frames: Target time frames for model input
        
    Returns:
        Feature array shaped (n_mels, time_frames, 1)
    """
    # Extract mel spectrogram
    mel_spec = extract_mel_spectrogram(audio, sr=sr, n_mels=n_mels)
    
    # Transpose to (time, frequency) and pad/truncate time dimension
    mel_spec = mel_spec.T  # (time, frequency)
    
    # Pad or truncate each frequency bin's time dimension
    if mel_spec.shape[0] > time_frames:
        mel_spec = mel_spec[:time_frames, :]
    elif mel_spec.shape[0] < time_frames:
        padding = time_frames - mel_spec.shape[0]
        mel_spec = np.pad(mel_spec, ((0, padding), (0, 0)), mode="constant")
    
    # Transpose back to (frequency, time) and add channel dimension
    mel_spec = mel_spec.T  # Back to (frequency, time)
    mel_spec = np.expand_dims(mel_spec, axis=-1)  # Add channel dimension
    
    return mel_spec


if __name__ == "__main__":
    # Example usage
    print("Acoustic Preprocessing Module")
    print("=" * 50)
    
    # Test with dummy audio
    dummy_audio = np.random.randn(22050)  # 1 second at 22.05kHz
    
    features = extract_all_features(dummy_audio)
    print(f"MFCC shape: {features['mfcc'].shape}")
    print(f"Mel Spectrogram shape: {features['mel_spectrogram'].shape}")
    print(f"FFT shape: {features['fft'].shape}")
    
    print("\n✅ Acoustic preprocessing module ready!")

