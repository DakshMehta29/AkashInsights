## AkashInsights: Aircraft Health Intelligence (Aerospace + AI + Recognition/Speech)

Baseline: Predict Remaining Useful Life (RUL) of turbofan engines using NASA CMAPSS data (FD001). This repo sets up a clean, modular pipeline and notebooks, and prepares for future Week 3 integration of acoustic models (Librosa + CNN).

### Dataset
- NASA CMAPSS Turbofan Engine Degradation Simulation
- FD001 subset (train/test .txt)
- Source: `https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/`

Place the following files in `data/`:
- `train_FD001.txt`
- `test_FD001.txt`

### Repository Structure
```
AkashInsights/
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── train_cleaned.csv
│   ├── test_cleaned.csv
│   └── processed/
│       ├── X_train.npy, y_train.npy, X_val.npy, y_val.npy
│       ├── feature_names.json
│       └── scaler.pkl
├── notebooks/
│   ├── 01_load_and_clean.ipynb
│   ├── 02_train_model.ipynb
├── scripts/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── train_rf_model.py
│   ├── evaluate.py
├── models/
│   └── rf_model.pkl
├── results/
│   ├── evaluation_report.txt
│   └── feature_importance.csv
└── README.md
```

### Environment
Install Python 3.9+ and the required packages:

```bash
pip install -U numpy pandas scikit-learn seaborn matplotlib
```

### Steps to Run
1) Load and clean raw data:
```bash
python scripts/load_data.py
```
This creates `data/train_cleaned.csv` and `data/test_cleaned.csv`.

2) Preprocess (add RUL, scale, split) [optional to run explicitly]:
```bash
python scripts/preprocess.py
```
The training script will run this automatically if needed.

3) Train RandomForest baseline and evaluate:
```bash
python scripts/train_rf_model.py
```
Artifacts:
- Model: `models/rf_model.pkl`
- Metrics: `results/evaluation_report.txt` (MAE, RMSE, R²)
- Feature importance: `results/feature_importance.csv`

4) Explore notebooks (recommended):
- `notebooks/01_load_and_clean.ipynb`: preview data, RUL distribution, feature correlations
- `notebooks/02_train_model.ipynb`: training demo, feature importance, evaluation

### Model Details
- Algorithm: RandomForestRegressor (baseline)
- Target: Remaining Useful Life (RUL)
- Inputs: 3 operational settings + up to 21 sensor signals (auto-pruned if constant zero)
- Metrics: MAE, RMSE, R²
- Baseline Goal: **MAE < 20 cycles**

### Future Roadmap (Week 3): Acoustic Agent Integration
- Add `Librosa` for audio feature extraction (MFCCs, spectral features)
- Train CNN on engine sound data for anomaly detection and RUL refinement
- Fuse acoustic insights with sensor data (late fusion or learned fusion)

### Optional: Streamlit (Stretch)
A simple real-time predictor can be added (`app.py`) to input sensor values and predict RUL/anomaly. Future work will include audio upload and inference.

### Notes
- All file paths are relative
- Random seed fixed at 42 for reproducibility
- Code is modular and commented; adheres to PEP8

### Contributing
PRs are welcome. Please keep code modular and add comments explaining non-obvious logic. 


