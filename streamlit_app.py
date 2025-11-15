"""
AkashInsights - Dual-Agent Aerospace AI System
Streamlit Dashboard for Machine Ear + Human Ear Integration
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import tempfile
import os

# Import modules
try:
    from src.acoustic_inference import AcousticInference, predict_audio, predict_from_mic
    from src.speech_agent import SpeechAgent
    from src.composite_engine import CompositeHealthEngine
    from src.translator import Translator
    from src.dashboard import MaintenanceLog, EmissionsAgent
    from src.utils import get_status_color, format_confidence, format_timestamp
except ImportError:
    # Fallback for direct imports
    from acoustic_inference import AcousticInference, predict_audio, predict_from_mic
    from speech_agent import SpeechAgent
    from composite_engine import CompositeHealthEngine
    from translator import Translator
    from dashboard import MaintenanceLog, EmissionsAgent
    from utils import get_status_color, format_confidence, format_timestamp

# Page config
st.set_page_config(
    page_title="AkashInsights - Aircraft Health Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .status-safe { background-color: #28a745; color: white; padding: 0.5rem; border-radius: 5px; }
    .status-caution { background-color: #ffc107; color: black; padding: 0.5rem; border-radius: 5px; }
    .status-critical { background-color: #dc3545; color: white; padding: 0.5rem; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "acoustic_model_loaded" not in st.session_state:
    st.session_state.acoustic_model_loaded = False
if "speech_agent_loaded" not in st.session_state:
    st.session_state.speech_agent_loaded = False
if "composite_engine_loaded" not in st.session_state:
    st.session_state.composite_engine_loaded = False
if "maintenance_log" not in st.session_state:
    st.session_state.maintenance_log = MaintenanceLog()
if "voice_command_history" not in st.session_state:
    st.session_state.voice_command_history = []


def load_models():
    """Load all AI models."""
    model_path = Path("models/acoustic_model.h5")
    
    if not st.session_state.acoustic_model_loaded and model_path.exists():
        try:
            st.session_state.acoustic_inference = AcousticInference(model_path)
            st.session_state.acoustic_model_loaded = True
        except Exception as e:
            st.error(f"Error loading acoustic model: {e}")
    
    if not st.session_state.speech_agent_loaded:
        try:
            st.session_state.speech_agent = SpeechAgent(model_name="base")
            st.session_state.speech_agent_loaded = True
        except Exception as e:
            st.warning(f"Speech agent not available: {e}")
    
    if not st.session_state.composite_engine_loaded:
        try:
            st.session_state.composite_engine = CompositeHealthEngine()
            st.session_state.composite_engine_loaded = True
        except Exception as e:
            st.warning(f"Composite engine not available: {e}")


def get_status_badge(status: str) -> str:
    """Get HTML badge for status."""
    status_class = f"status-{status.lower()}"
    return f'<div class="{status_class}"><strong>SYSTEM STATE: {status.upper()}</strong></div>'


def process_voice_command(command: str) -> str:
    """Process voice commands."""
    command_lower = command.lower()
    
    if "engine status" in command_lower or "show engine" in command_lower:
        return "navigate_to_machine"
    elif "translate" in command_lower or "translation" in command_lower:
        return "navigate_to_translation"
    elif "acoustic" in command_lower or "sound" in command_lower:
        return "navigate_to_machine"
    elif "stress" in command_lower or "crew" in command_lower:
        return "navigate_to_crew"
    else:
        return "unknown"


# Main App
def main():
    # Header
    st.markdown('<div class="main-header">✈️ AkashInsights - Aircraft Health Intelligence</div>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading AI models..."):
        load_models()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Navigation")
        page = st.radio(
            "Select Page",
            ["🏠 Dashboard", "🔊 Machine Health", "👥 Crew Communication", "📊 Analytics", "⚙️ Settings"]
        )
        
        st.header("🎤 Voice Commands")
        voice_input = st.text_input("Say a command:", placeholder="e.g., 'Show engine status'")
        if st.button("Process Command") and voice_input:
            action = process_voice_command(voice_input)
            st.session_state.voice_command_history.append({
                "command": voice_input,
                "action": action,
                "timestamp": format_timestamp()
            })
            if action != "unknown":
                st.success(f"Command recognized: {action}")
                if action.startswith("navigate_to"):
                    st.info(f"Navigate to: {action.replace('navigate_to_', '')}")
            else:
                st.warning("Command not recognized. Try: 'Show engine status', 'Translate message', etc.")
        
        if st.session_state.voice_command_history:
            with st.expander("Command History"):
                for cmd in st.session_state.voice_command_history[-5:]:
                    st.text(f"{cmd['timestamp']}: {cmd['command']}")
    
    # Main content based on page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🔊 Machine Health":
        show_machine_health()
    elif page == "👥 Crew Communication":
        show_crew_communication()
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "⚙️ Settings":
        show_settings()


def show_dashboard():
    """Main dashboard with unified status."""
    st.header("📊 System Overview")
    
    # System Status Banner
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Get latest status (mock for demo)
        latest_status = "safe"  # Would come from latest analysis
        status_color = get_status_color(latest_status)
        st.markdown(
            f'<div style="background-color: {status_color}; color: white; padding: 1rem; '
            f'border-radius: 10px; text-align: center; font-size: 1.5rem; font-weight: bold;">'
            f'🛡️ SYSTEM STATE: {latest_status.upper()}</div>',
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Machine Health", "92%", "↑ 2%")
    with col2:
        st.metric("Crew Stress", "Low", "↓ 5%")
    with col3:
        st.metric("Composite Score", "0.87", "↑ 0.03")
    with col4:
        st.metric("Faults Detected", "0", "Normal")
    
    st.divider()
    
    # Recent Activity
    st.subheader("📋 Recent Maintenance Log")
    if st.session_state.maintenance_log:
        records = st.session_state.maintenance_log.get_recent_records(5)
        if records:
            df = pd.DataFrame(records)
            st.dataframe(df[["timestamp", "machine_status", "fault_prediction", "system_status"]], use_container_width=True)
        else:
            st.info("No maintenance records yet. Run analyses to generate logs.")
    
    # Emission Optimization
    st.subheader("🌱 Emission Optimization & Fuel Efficiency")
    
    with st.expander("ℹ️ How It Works", expanded=False):
        st.markdown("""
        **Emission Optimization Agent** analyzes engine health to recommend fuel-efficient flight parameters:
        
        - **Healthy Engine (Anomaly < 30%)**: System recommends slight altitude increase and throttle reduction
          - Optimal altitude: +2000 ft above current
          - Throttle reduction: ~2%
          - Expected fuel savings: 3-8%
          - CO₂ reduction: Proportional to fuel savings
        
        - **Unhealthy Engine (Anomaly ≥ 30%)**: System recommends maintaining current parameters
          - No optimization (safety first)
          - Monitor engine health before optimization
        
        **Benefits**:
        - Reduce fuel consumption by 3-8% on healthy engines
        - Lower CO₂ emissions proportionally
        - Extend engine life through optimal operating conditions
        - Real-time recommendations based on acoustic analysis
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        anomaly_prob = st.slider(
            "Anomaly Probability", 
            0.0, 1.0, 0.15, 0.01,
            help="Probability of engine anomaly (from acoustic analysis). Lower = healthier engine."
        )
        st.caption(f"Engine Status: {'🟢 Healthy' if anomaly_prob < 0.3 else '🟡 Caution' if anomaly_prob < 0.7 else '🔴 Critical'}")
    with col2:
        altitude = st.slider(
            "Current Altitude (ft)", 
            30000, 40000, 35000, 500,
            help="Current flight altitude in feet. Typical cruising altitude: 35,000-40,000 ft"
        )
        st.caption(f"Current: {altitude:,} ft")
    
    if st.button("🚀 Get Optimization Recommendations", type="primary"):
        with st.spinner("Calculating optimal parameters..."):
            recommendations = EmissionsAgent.recommend_optimization(anomaly_prob, altitude)
            
            if recommendations['recommendation'] == 'optimize':
                st.success("✅ Optimization Available - Engine is healthy enough for efficiency improvements")
            else:
                st.warning("⚠️ Optimization Not Recommended - Maintain current parameters for safety")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Fuel Savings", 
                    f"{recommendations['fuel_savings_pct']:.2f}%",
                    help="Estimated percentage reduction in fuel consumption"
                )
            with col2:
                st.metric(
                    "CO₂ Reduction", 
                    f"{recommendations['co2_reduction_kg']:.2f} kg",
                    help="Estimated CO₂ emissions reduction per flight hour"
                )
            with col3:
                st.metric(
                    "Optimal Altitude", 
                    f"{recommendations['optimal_altitude_ft']:.0f} ft",
                    delta=f"{recommendations['optimal_altitude_ft'] - altitude:+.0f} ft",
                    help="Recommended altitude for optimal efficiency"
                )
            with col4:
                throttle_display = f"{recommendations['throttle_reduction']*100:.1f}%"
                st.metric(
                    "Throttle Reduction",
                    throttle_display,
                    help="Recommended throttle reduction for fuel efficiency"
                )
            
            # Detailed breakdown
            with st.expander("📊 Detailed Analysis"):
                st.json(recommendations)
                
                # Visual representation
                if recommendations['fuel_savings_pct'] > 0:
                    st.markdown("**Estimated Annual Impact (assuming 1000 flight hours/year):**")
                    annual_fuel_savings = recommendations['fuel_savings_pct'] * 1000  # Simplified calculation
                    annual_co2 = recommendations['co2_reduction_kg'] * 1000
                    st.info(f"💰 Fuel Savings: ~{annual_fuel_savings:.0f}% annually | 🌍 CO₂ Reduction: ~{annual_co2:.0f} kg/year")


def show_machine_health():
    """Machine Health Monitor tab."""
    st.header("🔊 Machine Health Monitor")
    
    tab1, tab2 = st.tabs(["📁 Upload Audio", "🎤 Live Recording"])
    
    with tab1:
        st.info("💡 **Tip**: Demo audio files are available in `data/acoustic/` folder. Try `demo_normal.wav`, `demo_fault1.wav`, or `demo_fault2.wav`")
        uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "flac", "txt"], help="Upload engine sound file for fault detection analysis")
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                if st.session_state.acoustic_model_loaded:
                    # Predict
                    result = predict_audio(tmp_path)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Prediction Results")
                        st.metric("Predicted Class", result["predicted_class"])
                        st.metric("Confidence", format_confidence(result["confidence"]))
                        st.metric("Status", result["anomaly_status"].upper())
                        
                        # Show classification method
                        method = result.get("method", "unknown")
                        if method == "trained_model":
                            st.success("✅ Using trained CNN model")
                        elif method == "feature_based":
                            st.info("ℹ️ Using feature-based analysis (analyzes audio characteristics)")
                            if "features_used" in result:
                                with st.expander("Audio Features Analyzed"):
                                    st.json(result["features_used"])
                        else:
                            st.warning("⚠️ Using fallback classifier")
                        
                        # Status badge
                        status_color = get_status_color(result["anomaly_status"])
                        st.markdown(
                            f'<div style="background-color: {status_color}; color: white; '
                            f'padding: 1rem; border-radius: 5px; text-align: center;">'
                            f'Status: {result["anomaly_status"].upper()}</div>',
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.subheader("Class Probabilities")
                        prob_df = pd.DataFrame({
                            "Class": list(result["all_probabilities"].keys()),
                            "Probability": list(result["all_probabilities"].values())
                        })
                        fig = px.bar(prob_df, x="Class", y="Probability", color="Probability",
                                   color_continuous_scale="RdYlGn_r")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Spectrogram visualization (placeholder)
                    st.subheader("📈 Audio Spectrogram")
                    st.info("Spectrogram visualization would go here (requires librosa plotting)")
                    
                    # Log to maintenance
                    if st.button("Log to Maintenance Record"):
                        hash_val = st.session_state.maintenance_log.add_record(
                            machine_status=result["predicted_class"],
                            fault_prediction=result["predicted_class"],
                            stress_level="N/A",
                            composite_score=result["confidence"] if result["is_normal"] else 1.0 - result["confidence"],
                            system_status=result["anomaly_status"]
                        )
                        st.success(f"Record logged! Hash: {hash_val[:16]}...")
                else:
                    st.warning("Acoustic model not loaded. Please ensure models/acoustic_model.h5 exists.")
            
            finally:
                os.unlink(tmp_path)
    
    with tab2:
        st.subheader("🎤 Live Microphone Recording")
        st.info("💡 **Note**: Make sure your microphone is connected and permissions are granted. Play engine sounds or speak near the microphone.")
        
        duration = st.slider("Recording Duration (seconds)", 1.0, 10.0, 3.0, 0.5, help="How long to record audio from microphone")
        
        if st.button("🎙️ Start Recording", type="primary"):
            if st.session_state.acoustic_model_loaded:
                try:
                    import sounddevice as sd
                    import soundfile as sf
                    import numpy as np
                    from pathlib import Path
                    
                    st.info(f"🎤 Recording for {duration} seconds... Speak or play audio now!")
                    
                    # Record audio
                    sr = 22050
                    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.float32)
                    sd.wait()
                    audio = audio.flatten()
                    
                    st.success("✅ Recording complete! Processing...")
                    
                    # Save temporary file
                    temp_path = Path("temp_mic_recording.wav")
                    sf.write(str(temp_path), audio, sr)
                    
                    # Predict
                    result = predict_audio(temp_path)
                    
                    # Cleanup
                    if temp_path.exists():
                        temp_path.unlink()
                    
                    st.subheader("📊 Prediction Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Class", result["predicted_class"])
                        st.metric("Confidence", format_confidence(result["confidence"]))
                    with col2:
                        st.metric("Status", result["anomaly_status"].upper())
                        st.metric("Source", result.get("source", "microphone"))
                    
                    # Show classification method
                    method = result.get("method", "unknown")
                    if method == "trained_model":
                        st.success("✅ Using trained CNN model")
                    elif method == "feature_based":
                        st.info("ℹ️ Using feature-based analysis - analyzing actual audio characteristics")
                    else:
                        st.warning("⚠️ Using fallback classifier")
                    
                    # Show probabilities
                    st.subheader("Class Probabilities")
                    prob_df = pd.DataFrame({
                        "Class": list(result["all_probabilities"].keys()),
                        "Probability": list(result["all_probabilities"].values())
                    })
                    fig = px.bar(prob_df, x="Class", y="Probability", color="Probability",
                               color_continuous_scale="RdYlGn_r", title="Fault Classification Probabilities")
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Recording error: {e}")
                    st.info("💡 **Troubleshooting**:\n- Check microphone permissions\n- Ensure microphone is connected\n- Try a different duration")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
            else:
                st.warning("⚠️ Acoustic model not loaded. Please ensure `models/acoustic_model.h5` exists.")


def show_crew_communication():
    """Crew Communication Monitor tab."""
    st.header("👥 Crew Communication Monitor")
    
    tab1, tab2, tab3 = st.tabs(["🎤 Live Transcription", "📝 Text Analysis", "🌐 Translation"])
    
    with tab1:
        st.subheader("Real-time Speech-to-Text")
        duration = st.slider("Recording Duration", 1.0, 10.0, 5.0, 0.5)
        
        if st.button("🎙️ Record & Transcribe"):
            if st.session_state.speech_agent_loaded:
                with st.spinner("Recording and processing..."):
                    try:
                        import sounddevice as sd
                        import soundfile as sf
                        import librosa
                        import numpy as np
                        from pathlib import Path
                        
                        # Record audio
                        sr = 16000
                        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.float32)
                        sd.wait()
                        audio = audio.flatten()
                        
                        # Save temporary file for transcription
                        temp_path = Path("temp_recording.wav")
                        sf.write(str(temp_path), audio, sr)
                        
                        # Transcribe
                        text = st.session_state.speech_agent.transcribe(temp_path)
                        st.success("Transcription complete!")
                        st.text_area("Transcribed Text", text, height=100, key="transcribed_text")
                        
                        # Stress analysis on recorded audio
                        st.subheader("Stress Analysis")
                        try:
                            # Use higher sample rate for better analysis
                            audio_high_sr = librosa.resample(audio, orig_sr=sr, target_sr=22050)
                            stress_result = st.session_state.speech_agent.detect_stress(audio_high_sr, sr=22050)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Stress Level", stress_result["stress_level"].upper())
                                st.metric("Stress Score", f"{stress_result['stress_score']:.3f}")
                            with col2:
                                st.metric("RMS Std", f"{stress_result['rms_std']:.4f}")
                                st.metric("Pitch Variation", f"{stress_result['pitch_variation']:.4f}")
                            
                            # Stress gauge
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=stress_result["stress_score"] * 100,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Stress Level"},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 30], 'color': "lightgreen"},
                                        {'range': [30, 70], 'color': "yellow"},
                                        {'range': [70, 100], 'color': "red"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 70
                                    }
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as stress_error:
                            st.warning(f"Stress analysis error: {stress_error}")
                        
                        # Cleanup
                        if temp_path.exists():
                            temp_path.unlink()
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.warning("Speech agent not loaded.")
    
    with tab2:
        st.subheader("Text-based Stress Detection")
        st.info("💡 Upload an audio file below for stress analysis, or use the Live Transcription tab to record audio.")
        
        uploaded_audio = st.file_uploader("Upload Audio File for Stress Analysis", type=["wav", "mp3", "flac"], key="stress_audio")
        
        if uploaded_audio is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_audio.name) as tmp_file:
                tmp_file.write(uploaded_audio.read())
                tmp_path = tmp_file.name
            
            try:
                if st.session_state.speech_agent_loaded:
                    import librosa
                    import numpy as np
                    
                    # Load audio
                    audio, sr = librosa.load(tmp_path, sr=22050)
                    
                    # Perform stress analysis
                    stress_result = st.session_state.speech_agent.detect_stress(audio, sr=sr)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Stress Level", stress_result["stress_level"].upper())
                        st.metric("Stress Score", f"{stress_result['stress_score']:.3f}")
                    with col2:
                        st.metric("RMS Std", f"{stress_result['rms_std']:.4f}")
                        st.metric("Pitch Variation", f"{stress_result['pitch_variation']:.4f}")
                    
                    # Stress gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=stress_result["stress_score"] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Stress Level"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional features
                    with st.expander("Detailed Features"):
                        st.json(stress_result)
                else:
                    st.warning("Speech agent not loaded. Cannot perform stress analysis.")
            except Exception as e:
                st.error(f"Error analyzing stress: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                os.unlink(tmp_path)
        else:
            st.info("👆 Please upload an audio file to analyze stress levels.")
    
    with tab3:
        st.subheader("🌐 Multilingual Translation")
        text_to_translate = st.text_area("Text to translate:", height=100, placeholder="Enter text to translate...")
        target_lang = st.selectbox("Target Language", 
                                  ["hindi", "tamil", "bengali", "telugu", "marathi", "english"])
        
        if st.button("Translate"):
            if not text_to_translate.strip():
                st.warning("Please enter some text to translate.")
            else:
                try:
                    translator = Translator()
                    
                    if translator.model is None:
                        st.error("❌ Translation model not available. Please install:")
                        st.code("pip install deep-translator")
                        st.info("💡 **Note**: `deep-translator` works with Python 3.13+. For older Python versions, you can also try `googletrans==4.0.0rc1`")
                    else:
                        with st.spinner("Translating..."):
                            result = translator.translate_text(text_to_translate, target_lang)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Original")
                                st.write(result.get("original", text_to_translate))
                                if "source_lang" in result:
                                    st.caption(f"Detected: {result['source_lang']}")
                            with col2:
                                st.subheader("Translated")
                                translated_text = result.get("translated", "Translation unavailable")
                                st.write(translated_text)
                                if "backend" in result:
                                    st.caption(f"Backend: {result['backend']}")
                            
                            if "error" in result:
                                st.warning(f"⚠️ Translation error: {result['error']}")
                                st.info("💡 Try installing: pip install googletrans==4.0.0rc1")
                            elif translated_text == text_to_translate and target_lang != "english":
                                st.warning("⚠️ Translation may have failed. The output is the same as input.")
                                st.info("💡 Try installing a translation backend or check your internet connection.")
                except ImportError as e:
                    st.error(f"❌ Translation module import error: {e}")
                    st.info("💡 Make sure the translator module is in the src/ directory.")
                except Exception as e:
                    st.error(f"❌ Translation error: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())


def show_analytics():
    """Analytics and historical data."""
    st.header("📊 Analytics Dashboard")
    
    # Maintenance log analytics
    st.subheader("Maintenance History")
    records = st.session_state.maintenance_log.get_recent_records(50)
    
    if records:
        df = pd.DataFrame(records)
        
        # Status distribution
        col1, col2 = st.columns(2)
        with col1:
            status_counts = df["system_status"].value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index,
                        title="System Status Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Composite score over time
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            fig = px.line(df, x="timestamp", y="composite_score",
                         title="Composite Health Score Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        # Full table
        st.subheader("Full Log")
        st.dataframe(df, use_container_width=True)
        
        # Chain verification
        is_valid = st.session_state.maintenance_log.verify_chain()
        if is_valid:
            st.success("✅ Blockchain integrity verified!")
        else:
            st.error("❌ Chain integrity compromised!")
    else:
        st.info("No analytics data yet. Run analyses to generate logs.")


def show_settings():
    """Settings and configuration."""
    st.header("⚙️ Settings")
    
    st.subheader("Model Configuration")
    st.info("Model paths and configurations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Acoustic Model:**")
        st.code("models/acoustic_model.h5")
        st.write(f"Status: {'✅ Loaded' if st.session_state.acoustic_model_loaded else '❌ Not loaded'}")
    
    with col2:
        st.write("**Speech Agent:**")
        st.code("Whisper base model")
        st.write(f"Status: {'✅ Loaded' if st.session_state.speech_agent_loaded else '❌ Not loaded'}")
    
    st.subheader("System Information")
    st.json({
        "Python Version": sys.version,
        "Project Root": str(Path(__file__).parent),
        "Models Directory": str(Path("models").resolve())
    })


if __name__ == "__main__":
    main()

