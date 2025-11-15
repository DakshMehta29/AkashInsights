"""
Composite Health Engine - Fusion Agent
Combines Machine Ear (acoustic) + Human Ear (speech stress) for unified health score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from .acoustic_inference import AcousticInference, predict_audio
    from .speech_agent import SpeechAgent
except ImportError:
    from acoustic_inference import AcousticInference, predict_audio
    from speech_agent import SpeechAgent

# Weight configuration
MACHINE_WEIGHT = 0.6
HUMAN_STRESS_WEIGHT = 0.4

# Status thresholds
THRESHOLD_SAFE = 0.7
THRESHOLD_CAUTION = 0.4


class CompositeHealthEngine:
    """Fusion engine combining acoustic and speech analysis."""
    
    def __init__(
        self,
        acoustic_model_path: Optional[str | Path] = None,
        speech_model_name: str = "base"
    ):
        """
        Initialize composite health engine.
        
        Args:
            acoustic_model_path: Path to acoustic model
            speech_model_name: Whisper model name
        """
        if acoustic_model_path is None:
            acoustic_model_path = Path("models/acoustic_model.h5")
        
        self.acoustic_model_path = Path(acoustic_model_path)
        self.acoustic_inference = None
        self.speech_agent = None
        
        self._load_models()
    
    def _load_models(self):
        """Load acoustic and speech models."""
        if self.acoustic_model_path.exists():
            try:
                self.acoustic_inference = AcousticInference(self.acoustic_model_path)
                print("✅ Acoustic model loaded")
            except Exception as e:
                print(f"⚠️  Error loading acoustic model: {e}")
        else:
            print(f"⚠️  Acoustic model not found: {self.acoustic_model_path}")
        
        try:
            self.speech_agent = SpeechAgent(model_name="base")
            print("✅ Speech agent loaded")
        except Exception as e:
            print(f"⚠️  Error loading speech agent: {e}")
    
    def compute_machine_score(self, acoustic_result: Dict) -> float:
        """
        Convert acoustic prediction to normalized machine health score (0-1).
        
        Args:
            acoustic_result: Result from acoustic inference
            
        Returns:
            Machine health score (1.0 = perfect, 0.0 = critical fault)
        """
        if acoustic_result["is_normal"]:
            # Normal state: confidence = health score
            return float(acoustic_result["confidence"])
        else:
            # Fault detected: invert confidence (lower confidence in fault = worse)
            fault_confidence = acoustic_result["confidence"]
            # Map to health: high fault confidence = low health
            health_score = 1.0 - fault_confidence
            return max(0.0, health_score)
    
    def compute_human_stress_index(self, stress_level: str, stress_score: float) -> float:
        """
        Convert stress level to normalized index (0-1).
        
        Args:
            stress_level: "low", "medium", "high"
            stress_score: Raw stress score (0-1)
            
        Returns:
            Human stress index (1.0 = no stress, 0.0 = high stress)
        """
        # Inverse of stress: high stress = low index
        return 1.0 - stress_score
    
    def compute_composite_score(
        self,
        machine_score: float,
        human_stress_index: float,
        machine_weight: float = MACHINE_WEIGHT,
        human_weight: float = HUMAN_STRESS_WEIGHT
    ) -> float:
        """
        Compute weighted composite health score.
        
        Args:
            machine_score: Machine health score (0-1)
            human_stress_index: Human stress index (0-1)
            machine_weight: Weight for machine component
            human_weight: Weight for human component
            
        Returns:
            Composite score (0-1)
        """
        composite = (machine_weight * machine_score) + (human_weight * human_stress_index)
        return np.clip(composite, 0.0, 1.0)
    
    def get_system_status(self, composite_score: float) -> str:
        """
        Determine system status from composite score.
        
        Args:
            composite_score: Composite health score (0-1)
            
        Returns:
            Status: "safe", "caution", or "critical"
        """
        if composite_score >= THRESHOLD_SAFE:
            return "safe"
        elif composite_score >= THRESHOLD_CAUTION:
            return "caution"
        else:
            return "critical"
    
    def analyze_complete(
        self,
        audio_path: Optional[str | Path] = None,
        speech_text: Optional[str] = None,
        speech_stress: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Complete health analysis combining acoustic and speech.
        
        Args:
            audio_path: Path to audio file for acoustic analysis
            speech_text: Pre-transcribed speech text (optional)
            speech_stress: Pre-computed stress analysis (optional)
            
        Returns:
            Complete health analysis dictionary
        """
        results = {
            "machine_analysis": None,
            "human_analysis": None,
            "composite_score": 0.0,
            "system_status": "unknown"
        }
        
        # Machine Ear Analysis
        if audio_path and self.acoustic_inference:
            try:
                try:
                    from .acoustic_preprocessing import load_audio as load_audio_func
                except ImportError:
                    from acoustic_preprocessing import load_audio as load_audio_func
                
                audio, sr = load_audio_func(audio_path)
                machine_result = self.acoustic_inference.predict(audio, sr)
                results["machine_analysis"] = machine_result
                machine_score = self.compute_machine_score(machine_result)
            except Exception as e:
                print(f"⚠️  Acoustic analysis error: {e}")
                machine_score = 0.5  # Default neutral
        else:
            machine_score = 0.5  # Default if no audio
        
        # Human Ear Analysis
        if speech_stress:
            human_stress_index = self.compute_human_stress_index(
                speech_stress.get("stress_level", "medium"),
                speech_stress.get("stress_score", 0.5)
            )
            results["human_analysis"] = speech_stress
        elif speech_text and self.speech_agent:
            try:
                # Analyze speech text (would need audio for full analysis)
                # For now, use default if only text provided
                human_stress_index = 0.7  # Default moderate
                results["human_analysis"] = {
                    "transcription": speech_text,
                    "stress_level": "medium",
                    "stress_score": 0.3
                }
            except Exception as e:
                print(f"⚠️  Speech analysis error: {e}")
                human_stress_index = 0.5
        else:
            human_stress_index = 0.7  # Default optimistic
        
        # Composite Score
        composite_score = self.compute_composite_score(machine_score, human_stress_index)
        system_status = self.get_system_status(composite_score)
        
        results["machine_score"] = float(machine_score)
        results["human_stress_index"] = float(human_stress_index)
        results["composite_score"] = float(composite_score)
        results["system_status"] = system_status
        
        return results


def get_health_status(
    acoustic_result: Optional[Dict] = None,
    stress_result: Optional[Dict] = None
) -> Dict[str, any]:
    """
    Convenience function for quick health status.
    
    Args:
        acoustic_result: Acoustic prediction result
        stress_result: Speech stress analysis result
        
    Returns:
        Health status dictionary
    """
    engine = CompositeHealthEngine()
    
    machine_score = 0.5
    if acoustic_result:
        machine_score = engine.compute_machine_score(acoustic_result)
    
    human_stress_index = 0.7
    if stress_result:
        human_stress_index = engine.compute_human_stress_index(
            stress_result.get("stress_level", "medium"),
            stress_result.get("stress_score", 0.3)
        )
    
    composite_score = engine.compute_composite_score(machine_score, human_stress_index)
    system_status = engine.get_system_status(composite_score)
    
    return {
        "composite_score": float(composite_score),
        "system_status": system_status,
        "machine_score": float(machine_score),
        "human_stress_index": float(human_stress_index)
    }


if __name__ == "__main__":
    print("Composite Health Engine")
    print("=" * 50)
    
    engine = CompositeHealthEngine()
    
    # Example analysis
    print("\n📊 Example Health Analysis:")
    print("   (Requires audio file and/or speech input)")
    print("\n✅ Composite engine ready for fusion analysis")

