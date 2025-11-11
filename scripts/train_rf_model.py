"""
Train a RandomForest baseline for Remaining Useful Life (RUL) prediction.

This script:
- Ensures preprocessed data exists (runs preprocessing if needed)
- Trains RandomForestRegressor on scaled features
- Evaluates using MAE, RMSE, R^2
- Saves model to models/rf_model.pkl
- Writes metrics to results/evaluation_report.txt

Usage:
    python scripts/train_rf_model.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Local modules
try:
    from scripts.evaluate import regression_metrics
except ImportError:
    # Allow execution via `%run` from notebooks by adjusting sys.path to project root
    import sys
    from pathlib import Path as _Path
    sys.path.append(str(_Path(__file__).resolve().parents[1]))
    from scripts.evaluate import regression_metrics

RANDOM_SEED = 42


def ensure_preprocessed(project_root: Path) -> Path:
    """
    Ensure that preprocessed arrays exist. If not, run preprocessing.
    """
    processed_dir = project_root / "data" / "processed"
    required = [
        processed_dir / "X_train.npy",
        processed_dir / "y_train.npy",
        processed_dir / "X_val.npy",
        processed_dir / "y_val.npy",
        processed_dir / "feature_names.json",
        processed_dir / "scaler.pkl",
    ]
    if all(p.exists() for p in required):
        return processed_dir

    # Run preprocessing
    from scripts.preprocess import main as preprocess_main

    preprocess_main()
    return processed_dir


def load_processed(processed_dir: Path):
    X_train = np.load(processed_dir / "X_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    X_val = np.load(processed_dir / "X_val.npy")
    y_val = np.load(processed_dir / "y_val.npy")
    feature_names = json.loads((processed_dir / "feature_names.json").read_text(encoding="utf-8"))
    return X_train, y_train, X_val, y_val, feature_names


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor with sensible baseline hyperparameters.
    """
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)


def write_report(metrics: Dict[str, float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "RandomForest Regressor - RUL Prediction",
        "",
        f"MAE:  {metrics['MAE']:.4f}",
        f"RMSE: {metrics['RMSE']:.4f}",
        f"R^2:  {metrics['R2']:.4f}",
        "",
        "Goal: MAE < 20 cycles",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_feature_importance(model: RandomForestRegressor, feature_names, out_path: Path) -> None:
    import pandas as pd

    out_path.parent.mkdir(parents=True, exist_ok=True)
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(out_path, index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    results_dir = project_root / "results"
    processed_dir = ensure_preprocessed(project_root)

    X_train, y_train, X_val, y_val, feature_names = load_processed(processed_dir)
    model = train_model(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    metrics = regression_metrics(y_val, y_pred)

    # Persist artifacts
    save_model(model, models_dir / "rf_model.pkl")
    write_report(metrics, results_dir / "evaluation_report.txt")
    save_feature_importance(model, feature_names, results_dir / "feature_importance.csv")

    print("Training complete.")
    print(f"Model saved to:     {models_dir / 'rf_model.pkl'}")
    print(f"Report saved to:    {results_dir / 'evaluation_report.txt'}")
    print(f"Importance saved to:{results_dir / 'feature_importance.csv'}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()


