"""
Utility Functions - Shared helpers for AkashInsights
"""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np


def get_project_root() -> Path:
    """Get project root directory."""
    current = Path(__file__).resolve()
    # Navigate up from src/utils.py to project root
    if current.parent.name == "src":
        return current.parent.parent
    return current.parent


def ensure_dir(path: Path | str):
    """Ensure directory exists, create if not."""
    Path(path).mkdir(parents=True, exist_ok=True)


def compute_hash(data: str | bytes) -> str:
    """
    Compute SHA256 hash of data.
    
    Args:
        data: String or bytes to hash
        
    Returns:
        Hexadecimal hash string
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format datetime as ISO string.
    
    Args:
        dt: Datetime object (default: now)
        
    Returns:
        ISO formatted string
    """
    if dt is None:
        dt = datetime.now()
    return dt.isoformat()


def save_json(data: Dict[str, Any], file_path: Path | str):
    """Save dictionary to JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(file_path: Path | str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize score to [0, 1] range."""
    return np.clip((score - min_val) / (max_val - min_val + 1e-8), 0.0, 1.0)


def get_status_color(status: str) -> str:
    """
    Get color code for status (for Streamlit/UI).
    
    Args:
        status: "safe", "caution", or "critical"
        
    Returns:
        Hex color code
    """
    colors = {
        "safe": "#28a745",  # Green
        "caution": "#ffc107",  # Yellow
        "critical": "#dc3545"  # Red
    }
    return colors.get(status.lower(), "#6c757d")  # Gray default


def format_confidence(confidence: float) -> str:
    """Format confidence as percentage string."""
    return f"{confidence * 100:.1f}%"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


if __name__ == "__main__":
    print("Utils Module")
    print("=" * 50)
    print(f"Project root: {get_project_root()}")
    print(f"Hash test: {compute_hash('test')}")
    print(f"Timestamp: {format_timestamp()}")

