"""
Preprocess CMAPSS data for Remaining Useful Life (RUL) prediction.

This module:
- Loads cleaned CSVs
- Computes RUL per unit (max_cycle - current_cycle)
- Splits into train/validation sets (80/20)
- Scales features with MinMaxScaler (fit on train only)
- Saves numpy arrays and scaler for model training

Outputs:
  data/processed/
    X_train.npy, y_train.npy, X_val.npy, y_val.npy
    feature_names.json
    scaler.pkl

Usage:
    python scripts/preprocess.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

RANDOM_SEED = 42
VAL_SIZE = 0.2


def load_cleaned_data(data_dir: Path) -> pd.DataFrame:
    train_csv = data_dir / "train_cleaned.csv"
    if not train_csv.exists():
        raise FileNotFoundError(
            f"Missing cleaned data at {train_csv}. Run scripts/load_data.py first."
        )
    df = pd.read_csv(train_csv)
    # Enforce dtypes
    df["unit_number"] = df["unit_number"].astype(int)
    df["time_in_cycles"] = df["time_in_cycles"].astype(int)
    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Remaining Useful Life (RUL) per unit.
    RUL = max_cycle_for_unit - current_cycle
    """
    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df = df.copy()
    df["RUL"] = (max_cycle - df["time_in_cycles"]).astype(int)
    return df


def select_feature_target_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select numerical features and target RUL.
    Keep 'unit_number' and 'time_in_cycles' as features as they are predictive in baseline.
    """
    feature_cols = [c for c in df.columns if c not in {"RUL"}]
    X = df[feature_cols]
    y = df["RUL"]
    return X, y


def split_scale_save(
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
) -> Dict[str, str]:
    """
    Split data, scale features, and persist arrays and scaler.
    Returns paths to outputs for convenience.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=RANDOM_SEED, shuffle=True
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Save arrays
    np.save(out_dir / "X_train.npy", X_train_scaled)
    np.save(out_dir / "y_train.npy", y_train.to_numpy())
    np.save(out_dir / "X_val.npy", X_val_scaled)
    np.save(out_dir / "y_val.npy", y_val.to_numpy())

    # Save scaler
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save feature names for reference
    feature_names = list(X.columns)
    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    return {
        "X_train": str(out_dir / "X_train.npy"),
        "y_train": str(out_dir / "y_train.npy"),
        "X_val": str(out_dir / "X_val.npy"),
        "y_val": str(out_dir / "y_val.npy"),
        "scaler": str(out_dir / "scaler.pkl"),
        "feature_names": str(out_dir / "feature_names.json"),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"

    df = load_cleaned_data(data_dir)
    df = add_rul(df)
    X, y = select_feature_target_columns(df)

    outputs = split_scale_save(X, y, processed_dir)
    print("Preprocessing complete. Saved:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()


