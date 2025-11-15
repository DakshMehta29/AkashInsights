# ✈️ AkashInsights: Dual-Agent Aerospace AI System

**Aircraft Health Intelligence Platform** combining **Machine Ear** (acoustic fault detection) + **Human Ear** (speech recognition & stress analysis) for comprehensive aerospace health monitoring.

---

## 🎯 Project Overview

AkashInsights is an end-to-end AI system that:
- **Predicts engine faults** through acoustic analysis (Machine Ear)
- **Monitors crew stress** via speech recognition (Human Ear)
- **Fuses insights** for unified health scoring
- **Provides real-time dashboard** for monitoring and decision-making
- **Supports multilingual** communication (Make-in-India initiative)

---

## 📊 Week 2 Achievements

- ✅ Completed project ideation & architecture planning
- ✅ Created GitHub repo: [AkashInsights](https://github.com/DakshMehta29/AkashInsights)
- ✅ Downloaded datasets (CMAPSS + acoustic + speech)
- ✅ Set up project folder structure
- ✅ Preprocessed dataset samples
- ✅ Built initial Machine Ear MFCC pipeline
- ✅ Added baseline acoustic anomaly detection model
- ✅ Integrated Whisper for speech-to-text
- ✅ Created initial dual-agent Streamlit prototype
- ✅ Trained RandomForest baseline (MAE: 11.05 cycles, R²: 0.942)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AkashInsights Platform                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  Machine Ear     │         │  Human Ear       │         │
│  │  (Agent 1)      │         │  (Agent 2)      │         │
│  ├──────────────────┤         ├──────────────────┤         │
│  │ • Acoustic CNN   │         │ • Whisper STT    │         │
│  │ • Fault Detection│         │ • Stress Analysis│         │
│  │ • RUL Prediction │         │ • Translation    │         │
│  └────────┬─────────┘         └────────┬─────────┘         │
│           │                             │                   │
│           └──────────┬──────────────────┘                   │
│                      │                                      │
│           ┌──────────▼──────────┐                          │
│           │ Composite Engine    │                          │
│           │ (Fusion Agent)      │                          │
│           │ • Weighted Scoring  │                          │
│           │ • Status: Safe/Caution/Critical                │
│           └──────────┬──────────┘                          │
│                      │                                      │
│           ┌──────────▼──────────┐                          │
│           │ Streamlit Dashboard  │                          │
│           │ • Real-time Monitor │                          │
│           │ • Analytics         │                          │
│           │ • Blockchain Log    │                          │
│           └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
AkashInsights/
│
├── data/
│   ├── CMaps/                    # NASA CMAPSS raw data
│   │   ├── train_FD001.txt
│   │   └── test_FD001.txt
│   ├── acoustic/                 # Acoustic training data
│   │   ├── normal/
│   │   ├── fault1/
│   │   ├── fault2/
│   │   └── fault3/
│   ├── speech/                   # Speech samples
│   ├── train_cleaned.csv
│   ├── test_cleaned.csv
│   └── processed/               # Preprocessed arrays
│
├── src/
│   ├── acoustic_preprocessing.py # MFCC, Mel Spec, FFT extraction
│   ├── acoustic_model.py         # CNN/CRNN training
│   ├── acoustic_inference.py     # Real-time prediction
│   ├── speech_agent.py           # Whisper + stress detection
│   ├── translator.py             # Multilingual (IndicTrans)
│   ├── composite_engine.py      # Fusion agent
│   ├── dashboard.py              # Blockchain log + emissions
│   └── utils.py                  # Helper functions
│
├── scripts/                      # Week 2 baseline scripts
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_rf_model.py
│   └── evaluate.py
│
├── notebooks/
│   ├── 01_load_and_clean.ipynb
│   └── 02_train_model.ipynb
│
├── models/
│   ├── rf_model.pkl              # Week 2 baseline
│   └── acoustic_model.h5         # Week 3 CNN model
│
├── results/
│   ├── evaluation_report.txt
│   └── feature_importance.csv
│
├── streamlit_app.py              # Main dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/DakshMehta29/AkashInsights.git
cd AkashInsights

# Install dependencies
pip install -r requirements.txt

# Optional: Install Whisper (if not included in requirements)
pip install openai-whisper

# Optional: Install IndicTrans for better Indian language support
pip install indicTrans
```

### 2. Prepare Data

Place your datasets in the appropriate directories:

- **CMAPSS data**: `data/CMaps/train_FD001.txt`, `test_FD001.txt`
- **Acoustic data**: `data/acoustic/{normal,fault1,fault2,fault3}/*.wav`
- **Speech samples**: `data/speech/*.wav` (optional)

### 3. Train Models

#### Week 2 Baseline (RUL Prediction):
```bash
# Load and clean data
python scripts/load_data.py

# Preprocess
python scripts/preprocess.py

# Train RandomForest
python scripts/train_rf_model.py
```

#### Week 3 Acoustic Model:
```bash
# Train CNN/CRNN for fault detection
python -c "from src.acoustic_model import train_model; from pathlib import Path; train_model(Path('data/acoustic'), model_type='cnn', epochs=50, model_save_path=Path('models/acoustic_model.h5'))"
```

### 4. Run Dashboard

```bash
streamlit run streamlit_app.py
```

Access at: `http://localhost:8501`

---

## 🧠 Core Components

### 1. Machine Ear (Acoustic Agent)

**File**: `src/acoustic_preprocessing.py`, `src/acoustic_model.py`, `src/acoustic_inference.py`

**Features**:
- MFCC, Mel Spectrogram, FFT feature extraction
- Data augmentation (time-stretch, noise, pitch shift, gain)
- CNN/CRNN models for fault classification
- Real-time inference from audio files or microphone

**Usage**:
```python
from src.acoustic_inference import predict_audio, predict_from_mic

# Predict from file
result = predict_audio("data/acoustic/test.wav")
print(f"Class: {result['predicted_class']}, Confidence: {result['confidence']}")

# Predict from microphone
result = predict_from_mic(duration=3.0)
```

**Classes**: Normal, Fault1, Fault2, Fault3

**Target Accuracy**: >90%

---

### 2. Human Ear (Speech Agent)

**File**: `src/speech_agent.py`

**Features**:
- Whisper-based speech-to-text
- Stress detection (RMS energy, pitch variation, voice tremor, MFCC delta)
- Real-time transcription from microphone

**Usage**:
```python
from src.speech_agent import SpeechAgent

agent = SpeechAgent(model_name="base")
result = agent.analyze_speech(audio_path="speech.wav")
print(f"Transcription: {result['transcription']}")
print(f"Stress Level: {result['stress_level']}")
```

**Output**:
```json
{
  "transcription": "...",
  "stress_level": "low/medium/high",
  "stress_score": 0.35,
  "confidence": 0.94
}
```

---

### 3. Translation Module (Make-in-India)

**File**: `src/translator.py`

**Supported Languages**: Hindi, Tamil, Bengali, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu

**Usage**:
```python
from src.translator import Translator

translator = Translator()
result = translator.translate_text("Engine status is normal", "hindi")
print(result["translated"])
```

**Backends** (priority order):
1. IndicTrans (best for Indian languages)
2. IndicBERT (transformers-based)
3. Google Translate (fallback)

---

### 4. Composite Health Engine

**File**: `src/composite_engine.py`

**Fusion Formula**:
```
composite_score = 0.6 * machine_score + 0.4 * human_stress_index
```

**Status Levels**:
- **Safe**: composite_score ≥ 0.7
- **Caution**: 0.4 ≤ composite_score < 0.7
- **Critical**: composite_score < 0.4

**Usage**:
```python
from src.composite_engine import CompositeHealthEngine

engine = CompositeHealthEngine()
result = engine.analyze_complete(
    audio_path="engine.wav",
    speech_stress={"stress_level": "low", "stress_score": 0.2}
)
print(f"System Status: {result['system_status']}")
```

---

### 5. Streamlit Dashboard

**File**: `streamlit_app.py`

**Features**:
- **Machine Health Monitor**: Upload audio, live recording, spectrograms, fault prediction
- **Crew Communication**: Live transcription, stress analysis, translation
- **Analytics**: Maintenance history, composite score trends, blockchain verification
- **Emission Optimization**: Fuel savings recommendations based on engine health
- **Voice Commands**: "Show engine status", "Translate message", etc.

**Tabs**:
1. 🏠 Dashboard - System overview, status banner, quick stats
2. 🔊 Machine Health - Audio upload, live recording, predictions
3. 👥 Crew Communication - Speech transcription, stress detection, translation
4. 📊 Analytics - Historical data, charts, blockchain log
5. ⚙️ Settings - Model configuration, system info

---

## 🔐 Advanced Features

### Blockchain-like Maintenance Log

**File**: `src/dashboard.py` → `MaintenanceLog` class

- SHA256 hash chain for each record
- Timestamp, fault prediction, stress level, composite score
- Chain integrity verification
- SQLite database storage

**Usage**:
```python
from src.dashboard import MaintenanceLog

log = MaintenanceLog()
hash_val = log.add_record(
    machine_status="normal",
    fault_prediction="none",
    stress_level="low",
    composite_score=0.85,
    system_status="safe"
)
is_valid = log.verify_chain()  # True
```

---

### Emission Reduction Agent

**File**: `src/dashboard.py` → `EmissionsAgent` class

- Recommends optimal altitude and throttle based on engine health
- Estimates fuel savings (3-8%) and CO₂ reduction
- Mock simulation for demonstration

**Usage**:
```python
from src.dashboard import EmissionsAgent

recommendations = EmissionsAgent.recommend_optimization(
    anomaly_prob=0.15,
    current_altitude=35000
)
print(f"Fuel Savings: {recommendations['fuel_savings_pct']}%")
```

---

## 📈 Model Performance

### Week 2 Baseline (RandomForest)
- **MAE**: 11.05 cycles ✅ (Goal: <20)
- **RMSE**: 16.28 cycles
- **R²**: 0.942

### Week 3 Acoustic Model (Target)
- **Accuracy**: >90% (fault classification)
- **Classes**: Normal, Fault1, Fault2, Fault3

---

## 🛠️ Development

### Running Tests

```bash
# Test acoustic preprocessing
python -c "from src.acoustic_preprocessing import extract_all_features; import numpy as np; features = extract_all_features(np.random.randn(22050)); print('✅ Preprocessing works')"

# Test speech agent
python src/speech_agent.py

# Test composite engine
python src/composite_engine.py
```

### Code Style

- PEP8 compliant
- Type hints where applicable
- Docstrings for all functions
- Modular design

---

## 📝 Dataset Sources

1. **NASA CMAPSS**: Turbofan Engine Degradation Simulation
   - Source: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
   - Files: `train_FD001.txt`, `test_FD001.txt`

2. **Acoustic Data**: Engine sound samples (user-provided)
   - Structure: `data/acoustic/{normal,fault1,fault2,fault3}/*.wav`

3. **Speech Data**: Crew communication samples (user-provided)
   - Location: `data/speech/*.wav`

---

## 🎯 Future Enhancements

- [ ] Real-time streaming audio analysis
- [ ] Advanced fusion architectures (attention-based)
- [ ] Multi-engine fleet monitoring
- [ ] Mobile app integration
- [ ] Cloud deployment (AWS/Azure)
- [ ] Edge device optimization (TensorFlow Lite)

---

## 🤝 Contributing

PRs welcome! Please:
- Follow PEP8 style guide
- Add docstrings to new functions
- Include tests for new features
- Update README if adding major features

---

## 📄 License

This project is part of an academic/research initiative. Please cite appropriately if used in research.

---

## 👨‍💻 Author

**Daksh Mehta**
- GitHub: [@DakshMehta29](https://github.com/DakshMehta29)
- Repository: [AkashInsights](https://github.com/DakshMehta29/AkashInsights)

---

## 🙏 Acknowledgments

- NASA for CMAPSS dataset
- OpenAI for Whisper model
- Librosa team for audio processing
- Streamlit for dashboard framework
- IndicTrans for multilingual support

---

**Built with ❤️ for Aerospace AI Innovation**
