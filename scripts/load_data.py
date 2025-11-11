"""
Load and clean NASA CMAPSS Turbofan Engine Degradation data.

This script:
- Loads train and test .txt files
- Assigns NASA-specified column names
- Removes columns that are entirely zero or empty
- Saves cleaned CSVs to data/train_cleaned.csv and data/test_cleaned.csv

Usage:
    python scripts/load_data.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Consistent randomness across the project
RANDOM_SEED = 42


def get_cmapss_columns() -> List[str]:
    """
    CMAPSS FD001 schema:
    26 columns total:
      - unit_number (1)
      - time_in_cycles (1)
      - op_setting_1 .. op_setting_3 (3)
      - sensor_1 .. sensor_21 (21)
    """
    cols = [
        "unit_number",
        "time_in_cycles",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
    ]
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    return cols + sensor_cols


def load_txt_as_dataframe(path: Path) -> pd.DataFrame:
    """
    Load a CMAPSS .txt file into a DataFrame with assigned columns.
    Files are space-separated with possible variable whitespace and no header.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Read using delim_whitespace to handle irregular spaces
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=get_cmapss_columns(),
        engine="python",
    )

    # Some CMAPSS dumps may include trailing empty columns; drop unnamed columns if any
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Ensure proper dtypes
    df["unit_number"] = df["unit_number"].astype(int)
    df["time_in_cycles"] = df["time_in_cycles"].astype(int)

    return df


def drop_all_zero_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are entirely zero or NaN (excluding id and cycle).
    """
    protected = {"unit_number", "time_in_cycles"}
    cols_to_check = [c for c in df.columns if c not in protected]

    is_all_zero_or_nan = (
        (df[cols_to_check].fillna(0) == 0).all(axis=0) | df[cols_to_check].isna().all(axis=0)
    )
    drop_cols = list(is_all_zero_or_nan[is_all_zero_or_nan].index)

    cleaned = df.drop(columns=drop_cols)
    return cleaned


def save_cleaned_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    raw_dir = data_dir / "CMaps"

    # File names (FD001 as baseline)
    train_txt = raw_dir / "train_FD001.txt"
    test_txt = raw_dir / "test_FD001.txt"

    # Output cleaned CSVs
    train_csv = data_dir / "train_cleaned.csv"
    test_csv = data_dir / "test_cleaned.csv"

    # Load
    train_df = load_txt_as_dataframe(train_txt)
    test_df = load_txt_as_dataframe(test_txt)

    # Clean (drop all-zero columns using train reference, then align test)
    train_df_clean = drop_all_zero_columns(train_df)
    # Align test to the same columns as train
    test_df_clean = test_df[train_df_clean.columns]

    # Save
    save_cleaned_csv(train_df_clean, train_csv)
    save_cleaned_csv(test_df_clean, test_csv)

    print(f"Saved cleaned train to: {train_csv}")
    print(f"Saved cleaned test to:  {test_csv}")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()


