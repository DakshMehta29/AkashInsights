"""
Dashboard Module - Helper functions for Streamlit app
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import hashlib
import json
import numpy as np

from utils import compute_hash, format_timestamp, get_status_color


class MaintenanceLog:
    """Blockchain-like maintenance log using SQLite."""
    
    def __init__(self, db_path: Path | str = "data/maintenance_log.db"):
        """
        Initialize maintenance log database.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                hash TEXT NOT NULL UNIQUE,
                previous_hash TEXT,
                machine_status TEXT,
                fault_prediction TEXT,
                stress_level TEXT,
                composite_score REAL,
                system_status TEXT,
                data_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_record(
        self,
        machine_status: str,
        fault_prediction: str,
        stress_level: str,
        composite_score: float,
        system_status: str,
        additional_data: Optional[Dict] = None
    ) -> str:
        """
        Add maintenance record with hash chain.
        
        Args:
            machine_status: Machine health status
            fault_prediction: Predicted fault type
            stress_level: Human stress level
            composite_score: Composite health score
            system_status: Overall system status
            additional_data: Additional metadata
            
        Returns:
            Record hash
        """
        # Get previous hash
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT hash FROM maintenance_log ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()
        previous_hash = result[0] if result else "0" * 64  # Genesis block
        
        # Create data string
        data_dict = {
            "timestamp": format_timestamp(),
            "machine_status": machine_status,
            "fault_prediction": fault_prediction,
            "stress_level": stress_level,
            "composite_score": composite_score,
            "system_status": system_status
        }
        if additional_data:
            data_dict.update(additional_data)
        
        data_json = json.dumps(data_dict, sort_keys=True)
        
        # Compute hash
        hash_input = f"{previous_hash}{data_json}"
        record_hash = compute_hash(hash_input)
        
        # Insert record
        cursor.execute("""
            INSERT INTO maintenance_log 
            (timestamp, hash, previous_hash, machine_status, fault_prediction, 
             stress_level, composite_score, system_status, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            format_timestamp(),
            record_hash,
            previous_hash,
            machine_status,
            fault_prediction,
            stress_level,
            composite_score,
            system_status,
            data_json
        ))
        
        conn.commit()
        conn.close()
        
        return record_hash
    
    def get_recent_records(self, limit: int = 10) -> List[Dict]:
        """
        Get recent maintenance records.
        
        Args:
            limit: Number of records to retrieve
            
        Returns:
            List of record dictionaries
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, hash, previous_hash, machine_status, 
                   fault_prediction, stress_level, composite_score, system_status
            FROM maintenance_log
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        
        records = []
        for row in cursor.fetchall():
            records.append({
                "timestamp": row[0],
                "hash": row[1],
                "previous_hash": row[2],
                "machine_status": row[3],
                "fault_prediction": row[4],
                "stress_level": row[5],
                "composite_score": row[6],
                "system_status": row[7]
            })
        
        conn.close()
        return records
    
    def verify_chain(self) -> bool:
        """
        Verify hash chain integrity.
        
        Returns:
            True if chain is valid
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, hash, previous_hash, data_json FROM maintenance_log ORDER BY id")
        records = cursor.fetchall()
        
        if len(records) == 0:
            conn.close()
            return True
        
        previous_hash = "0" * 64
        for record_id, hash_val, prev_hash, data_json in records:
            if prev_hash != previous_hash:
                conn.close()
                return False
            
            hash_input = f"{previous_hash}{data_json}"
            computed_hash = compute_hash(hash_input)
            if computed_hash != hash_val:
                conn.close()
                return False
            
            previous_hash = hash_val
        
        conn.close()
        return True


class EmissionsAgent:
    """Mock emissions reduction agent based on engine health."""
    
    @staticmethod
    def recommend_optimization(anomaly_prob: float, current_altitude: float = 35000) -> Dict:
        """
        Recommend fuel/emission optimizations based on engine health.
        
        Args:
            anomaly_prob: Probability of anomaly (0-1)
            current_altitude: Current flight altitude in feet
            
        Returns:
            Optimization recommendations
        """
        if anomaly_prob < 0.30:
            # Healthy engine - optimize for efficiency
            optimal_altitude = current_altitude + 2000  # Climb slightly
            throttle_reduction = 0.02  # Reduce throttle by 2%
            fuel_savings_pct = np.random.uniform(3.0, 8.0)
            emission_reduction_pct = fuel_savings_pct * 0.95  # ~95% correlation
        else:
            # Unhealthy - maintain current or reduce load
            optimal_altitude = current_altitude
            throttle_reduction = 0.0
            fuel_savings_pct = 0.0
            emission_reduction_pct = 0.0
        
        return {
            "optimal_altitude_ft": float(optimal_altitude),
            "throttle_reduction": float(throttle_reduction),
            "fuel_savings_pct": float(fuel_savings_pct),
            "emission_reduction_pct": float(emission_reduction_pct),
            "co2_reduction_kg": float(fuel_savings_pct * 0.5),  # Mock: 0.5 kg per % savings
            "recommendation": "optimize" if anomaly_prob < 0.30 else "maintain"
        }


if __name__ == "__main__":
    print("Dashboard Utilities")
    print("=" * 50)
    
    # Test maintenance log
    log = MaintenanceLog()
    hash1 = log.add_record(
        machine_status="normal",
        fault_prediction="none",
        stress_level="low",
        composite_score=0.85,
        system_status="safe"
    )
    print(f"✅ Added record: {hash1[:16]}...")
    
    records = log.get_recent_records(5)
    print(f"✅ Retrieved {len(records)} records")
    
    is_valid = log.verify_chain()
    print(f"✅ Chain valid: {is_valid}")

