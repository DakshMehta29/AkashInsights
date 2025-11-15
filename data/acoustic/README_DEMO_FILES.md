# Demo Audio Files

This directory contains demo audio files for testing the AkashInsights system.

## Available Demo Files

### Machine Health Analysis (Acoustic)
- **normal/demo_normal.wav** - Simulated normal engine sound (440 Hz sine wave)
- **fault1/demo_fault1.wav** - Simulated fault type 1 (multiple frequencies)
- **fault2/demo_fault2.wav** - Simulated fault type 2 (higher frequency harmonics)

### How to Use
1. Go to **Machine Health Monitor** tab in Streamlit
2. Click **Upload Audio File**
3. Select one of the demo files from `data/acoustic/`
4. View the prediction results

## Getting Real Audio Data

### Option 1: NASA CMAPSS (Sensor Data)
- Already available in `data/CMaps/`
- Convert sensor readings to audio-like signals for analysis

### Option 2: Record Your Own
- Record engine sounds using a microphone
- Save as `.wav` format (recommended: 22050 Hz sample rate)
- Place in appropriate folder: `normal/`, `fault1/`, `fault2/`, or `fault3/`

### Option 3: Download from Public Datasets
- **MIMII Dataset**: https://zenodo.org/record/3384388
- **DCASE Challenge**: https://dcase.community/challenge
- **Aircraft Engine Sound Database**: Various research repositories

## Speech/Stress Analysis Demo

For speech analysis, you can:
1. Use your microphone in the **Live Transcription** tab
2. Record any speech audio (3-10 seconds recommended)
3. The system will analyze stress levels automatically

## File Format Requirements
- **Format**: WAV (recommended), MP3, FLAC
- **Sample Rate**: 22050 Hz (optimal), 16000 Hz (minimum)
- **Duration**: 1-10 seconds
- **Channels**: Mono (1 channel) preferred

