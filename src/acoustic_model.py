"""
Acoustic Model Training - CNN/CRNN for Fault Detection
Trains deep learning model to classify engine health states.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

try:
    from .acoustic_preprocessing import extract_all_features, load_audio, augment_audio, prepare_features
except ImportError:
    from acoustic_preprocessing import extract_all_features, load_audio, augment_audio, prepare_features

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Model configuration
N_MFCC = 13
N_MELS = 128
TIME_FRAMES = 128  # Fixed time dimension for input
N_CLASSES = 4  # Normal, Fault1, Fault2, Fault3


def build_cnn_model(input_shape: Tuple[int, int, int], n_classes: int = 4) -> keras.Model:
    """
    Build CNN model for acoustic fault classification.
    
    Args:
        input_shape: (height, width, channels) - e.g., (n_mels, time_frames, 1)
        n_classes: Number of fault classes
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(n_classes, activation="softmax", name="predictions")
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def build_crnn_model(input_shape: Tuple[int, int, int], n_classes: int = 4) -> keras.Model:
    """
    Build CRNN (CNN + RNN) model for temporal acoustic patterns.
    
    Args:
        input_shape: (height, width, channels)
        n_classes: Number of fault classes
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # CNN layers for feature extraction
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Reshape for RNN (remove height dimension, keep time)
        layers.Reshape((input_shape[1] // 4, -1)),
        
        # RNN layers
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(n_classes, activation="softmax", name="predictions")
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


# prepare_features is now imported from acoustic_preprocessing


def load_dataset(data_dir: Path, use_augmentation: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and prepare dataset from directory structure.
    
    Expected structure:
    data_dir/
        normal/
            *.wav or *.txt
        fault1/
            *.wav or *.txt
        fault2/
            *.wav or *.txt
        fault3/
            *.wav or *.txt
    
    Args:
        data_dir: Root directory with class subdirectories
        use_augmentation: Whether to apply data augmentation
        
    Returns:
        X: Feature arrays (n_samples, height, width, channels)
        y: Labels (n_samples,)
    """
    X_list = []
    y_list = []
    
    class_dirs = ["normal", "fault1", "fault2", "fault3"]
    label_encoder = LabelEncoder()
    label_encoder.fit(class_dirs)
    
    for class_idx, class_name in enumerate(class_dirs):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"⚠️  Warning: {class_dir} not found, skipping...")
            continue
        
        files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.txt"))
        
        for file_path in files:
            try:
                if file_path.suffix == ".wav":
                    audio, sr = load_audio(file_path)
                else:
                    # Assume sensor data - convert to audio-like signal
                    import pandas as pd
                    df = pd.read_csv(file_path, sep=r"\s+", header=None)
                    audio = df.iloc[:, -1].values.astype(np.float32)  # Use last sensor
                    sr = 22050
                
                # Prepare features (using model's expected dimensions)
                features = prepare_features(audio, sr, n_mels=N_MELS, time_frames=TIME_FRAMES)
                X_list.append(features)
                y_list.append(class_idx)
                
                # Augmentation
                if use_augmentation:
                    augmented = augment_audio(audio, sr)
                    for aug_audio in augmented[1:]:  # Skip original
                        aug_features = prepare_features(aug_audio, sr, n_mels=N_MELS, time_frames=TIME_FRAMES)
                        X_list.append(aug_features)
                        y_list.append(class_idx)
                        
            except Exception as e:
                print(f"⚠️  Error processing {file_path}: {e}")
                continue
    
    if len(X_list) == 0:
        raise ValueError("No data files found! Please check data directory structure.")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"✅ Loaded {len(X)} samples with shape {X.shape}")
    return X, y


def train_model(
    data_dir: Path,
    model_type: str = "cnn",
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: float = 0.2,
    model_save_path: Optional[Path] = None
) -> keras.Model:
    """
    Train acoustic fault detection model.
    
    Args:
        data_dir: Directory with class subdirectories
        model_type: "cnn" or "crnn"
        epochs: Training epochs
        batch_size: Batch size
        validation_split: Validation split ratio
        model_save_path: Path to save trained model
        
    Returns:
        Trained Keras model
    """
    print("=" * 60)
    print("Training Acoustic Fault Detection Model")
    print("=" * 60)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    X, y = load_dataset(data_dir, use_augmentation=True)
    
    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Build model
    print(f"\n🏗️  Building {model_type.upper()} model...")
    input_shape = X_train.shape[1:]
    
    if model_type.lower() == "cnn":
        model = build_cnn_model(input_shape, n_classes=N_CLASSES)
    elif model_type.lower() == "crnn":
        model = build_crnn_model(input_shape, n_classes=N_CLASSES)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'cnn' or 'crnn'")
    
    model.summary()
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            str(model_save_path) if model_save_path else "models/acoustic_model_best.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print("\n🚀 Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Save final model
    if model_save_path:
        model.save(str(model_save_path))
        print(f"\n💾 Model saved to: {model_save_path}")
    else:
        model.save("models/acoustic_model.h5")
        print("\n💾 Model saved to: models/acoustic_model.h5")
    
    return model


if __name__ == "__main__":
    # Example training (requires data directory structure)
    data_dir = Path("../data/acoustic")
    models_dir = Path("../models")
    models_dir.mkdir(exist_ok=True)
    
    if data_dir.exists() and any((data_dir / d).exists() for d in ["normal", "fault1", "fault2"]):
        print("Starting training...")
        model = train_model(
            data_dir=data_dir,
            model_type="cnn",
            epochs=50,
            model_save_path=models_dir / "acoustic_model.h5"
        )
    else:
        print("⚠️  Data directory not found or empty.")
        print("Expected structure: data/acoustic/{normal,fault1,fault2,fault3}/*.wav")
        print("\nCreating dummy model for testing...")
        # Create dummy model architecture
        dummy_model = build_cnn_model((N_MELS, TIME_FRAMES, 1), n_classes=N_CLASSES)
        models_dir.mkdir(exist_ok=True)
        dummy_model.save(models_dir / "acoustic_model.h5")
        print("✅ Dummy model saved (ready for real training when data is available)")

