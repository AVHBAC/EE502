# 🎵 Speech Emotion Recognition (SER) with Traditional Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

> **Speech emotion recognition using MFCC features with traditional machine learning approaches achieving up to 79% accuracy.**

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [📊 Performance](#-performance)
- [🛠️ Installation](#️-installation)
- [📁 Dataset Setup](#-dataset-setup)
- [🚀 Quick Start](#-quick-start)
- [🔧 Advanced Usage](#-advanced-usage)
- [📈 Experiment Results](#-experiment-results)
- [🏗️ Architecture](#️-architecture)
- [📋 API Reference](#-api-reference)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

## 🎯 Overview

This repository implements a Speech Emotion Recognition (SER) system that classifies emotions from audio signals using Mel-Frequency Cepstral Coefficients (MFCC) features. The system employs **traditional machine learning** techniques only:

- **Random Forest Classifier** - Best performer at 79% accuracy
- **Support Vector Machine (SVM)** - 76% accuracy
- **K-Nearest Neighbors (KNN)** - 74% accuracy
- **Logistic Regression** - 74% accuracy
- **Audio Feature Extraction** - MFCC, ZCR, Chroma, RMS, Mel Spectrogram
- **Simple Data Augmentation** - Noise injection, time stretching

### 🎪 Supported Emotions
- 😠 **Angry**
- 😨 **Fear** 
- 😊 **Happy**
- 😢 **Sad**
- 😮 **Surprise**
- 🤢 **Disgust**
- 😐 **Neutral**
- 😌 **Calm**

## ✨ Features

### 🔥 Core Capabilities
- **Multi-dataset Support**: RAVDESS, CREMA-D, TESS, SAVEE
- **Comprehensive Feature Extraction**: 162 audio features (ZCR + Chroma + MFCC + RMS + Mel)
- **Traditional ML Models**: Random Forest, SVM, KNN, Logistic Regression
- **Complete Evaluation**: Confusion matrices, classification reports, accuracy metrics
- **Production Ready**: Model persistence with joblib, easy deployment

### 🎯 Implementation Highlights
- **Feature Engineering**: 20 MFCCs, Zero Crossing Rate, Chroma features, RMS, Mel Spectrogram
- **Data Augmentation**: Simple noise injection and time stretching
- **Interpretability**: Feature importance analysis with Random Forest
- **Fast Training**: Efficient scikit-learn implementations
- **Low Resource Requirements**: Suitable for edge devices and embedded systems
- **Traditional ML Only**: No neural networks or deep learning methods

### 🎙️ Audio Preprocessing Details

**Parameters**:
- Sample Rate: 22,050 Hz
- Duration: 2.5 seconds per clip
- Offset: 0.6 seconds (skips initial silence/artifacts)

**Why We Don't Use Energy-Based Silence Trimming**:

Unlike some audio processing pipelines, this project **intentionally preserves silence** in audio clips for several important reasons:

1. **Emotional Information in Silence**: Pauses and silence patterns carry significant emotional meaning:
   - Hesitation and pauses in **fear** and **sad** emotions
   - Brief pauses for emphasis in **angry** speech
   - Calm emotions naturally have more quiet moments

2. **Consistent Feature Dimensions**: Fixed 2.5-second duration ensures:
   - All samples produce exactly 162 features
   - Uniform input dimensions for ML models
   - Simplified batch processing

3. **Dataset Quality**: Professional emotion datasets (RAVDESS, CREMA-D, TESS, SAVEE) are studio-recorded with minimal non-speech silence, making aggressive trimming unnecessary

4. **Simplicity**: Focusing on core ML classification rather than complex audio preprocessing techniques

**See EXPERIMENT_RESULTS.md for more details on preprocessing decisions.**

## 📊 Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **79%** | 81% | 78% | 79% |
| **SVM** | **76%** | 77% | 75% | 76% |
| **KNN** | **74%** | 75% | 74% | 74% |
| **Logistic Regression** | **74%** | 74% | 74% | 74% |

**🎯 Best Model**: **Random Forest at 79% accuracy**

## 🛠️ Installation

### Prerequisites
- **Python**: 3.8+ (3.11 recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 5GB free space

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/MFCC_SER-Advanced.git
cd MFCC_SER-Advanced
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n ser_env python=3.11
conda activate ser_env

# Or using venv
python -m venv ser_env
source ser_env/bin/activate  # Linux/Mac
# ser_env\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import sklearn; import librosa; import pandas; print('✅ All packages installed successfully')"
```

## 📁 Dataset Setup

### Supported Datasets

This project supports multiple standard emotion recognition datasets:

#### 1. **RAVDESS** (Recommended)
- **Download**: [RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- **Structure**: 
```
dataset/ravdess-emotional-speech-audio/
└── audio_speech_actors_01-24/
    ├── Actor_01/
    ├── Actor_02/
    └── ...
```

#### 2. **CREMA-D**
- **Download**: [CREMA-D Dataset](https://www.kaggle.com/datasets/ejlok1/cremad)
- **Structure**:
```
dataset/cremad/
└── AudioWAV/
    ├── 1001_DFA_ANG_XX.wav
    ├── 1001_DFA_HAP_XX.wav
    └── ...
```

#### 3. **TESS**  
- **Download**: [TESS Dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- **Structure**:
```
dataset/toronto-emotional-speech-set-tess/
└── TESS Toronto emotional speech set data/
    ├── OAF_angry/
    ├── OAF_happy/
    └── ...
```

#### 4. **SAVEE**
- **Download**: [SAVEE Dataset](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)
- **Structure**:
```
dataset/surrey-audiovisual-expressed-emotion-savee/
└── ALL/
    ├── DC_a01.wav
    ├── DC_h01.wav
    └── ...
```

### 📥 Quick Dataset Setup

**Option 1: Manual Download**
1. Download datasets from the links above
2. Extract to `dataset/` folder following the structure shown
3. Ensure file paths match the expected structure

**Option 2: Kaggle API (Recommended)**
```bash
# Install Kaggle API
pip install kaggle

# Download RAVDESS (example)
kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio
unzip ravdess-emotional-speech-audio.zip -d dataset/

# Verify structure
python -c "import os; print('✅ Dataset found' if os.path.exists('dataset') else '❌ Dataset missing')"
```

## 🚀 Quick Start

### 1. Run Jupyter Notebook (Recommended)
```bash
# Interactive notebook with all steps
jupyter notebook Speech_Emotion_Recognition_ML.ipynb
```

### 2. Run Baseline Comparison Script
```bash
# Automated ML comparison (5-10 minutes)
python baseline_comparison.py
```

## 🔧 Advanced Usage

### Custom Model Training
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# Train custom Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Save model
joblib.dump(rf_model, 'custom_rf_model.pkl')
```

### Feature Extraction
```python
import librosa
import numpy as np

def extract_features(audio_path):
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, duration=2.5, offset=0.6)

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = np.mean(mfcc.T, axis=0)

    # Extract other features
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y))

    return np.concatenate([mfcc, [zcr], chroma, [rms]])
```

### Data Augmentation
```python
import librosa
import numpy as np

# Add noise
def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

# Time stretching
def time_stretch(data, stretch_rate=0.8):
    return librosa.effects.time_stretch(data, rate=stretch_rate)
```

## 📈 Experiment Results

### Performance Metrics
```
Random Forest:           79% accuracy
SVM:                     76% accuracy
KNN:                     74% accuracy
Logistic Regression:     74% accuracy
```

### Comprehensive Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **79%** | **0.81** | **0.78** | **0.79** |
| **SVM** | **76%** | **0.77** | **0.75** | **0.76** |
| **KNN** | **74%** | **0.75** | **0.74** | **0.74** |
| **Logistic Regression** | **74%** | **0.74** | **0.74** | **0.74** |

### Visualizations
All experiments generate comprehensive visualizations:
- Confusion matrices
- Feature importance analysis
- Classification reports
- Model comparison plots

## 🏗️ Architecture

### Project Structure
```
MFCC_SER-main/
├── 📁 dataset/                          # Audio datasets
│   ├── ravdess-emotional-speech-audio/
│   ├── cremad/
│   ├── toronto-emotional-speech-set-tess/
│   └── surrey-audiovisual-expressed-emotion-savee/
├── 📁 models/                           # Saved models (joblib format)
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   ├── knn_model.pkl
│   └── lr_model.pkl
├── 📁 results/                          # Experiment results
│   ├── confusion_matrices/
│   └── performance_reports/
├── 📄 requirements.txt                  # Dependencies
├── 📄 Speech_Emotion_Recognition_ML.ipynb  # ML-only notebook
├── 🐍 baseline_comparison.py           # ML model comparison
├── 🐍 pdf_analysis_and_visualization.py # Visualization utilities
├── 🐍 setup.py                         # Package setup
├── 📄 EXECUTION_ORDER_GUIDE.md         # Detailed instructions
├── 📄 PROJECT_SUMMARY.md               # Project summary
└── 📄 README.md                        # This file
```

### Core Components

#### 1. **Data Processing Pipeline**
- **Audio Loading**: Librosa-based loading with configurable parameters
- **Feature Extraction**: 162 audio features (MFCC, ZCR, Chroma, RMS, Mel)
- **Augmentation**: Simple noise injection and time stretching
- **Preprocessing**: Scaling, encoding, train/test splitting

#### 2. **Machine Learning Models**
```python
# Available models (scikit-learn)
models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(kernel='rbf'),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}
```

#### 3. **Evaluation Framework**
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Visualizations**: Confusion matrices, classification reports
- **Cross-validation**: K-fold validation for robust estimates
- **Model Persistence**: Save/load models with joblib

## 📋 API Reference

### Core Functions

#### Feature Extraction
```python
import librosa
import numpy as np

def extract_mfcc_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050, duration=2.5, offset=0.6)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc.T, axis=0)
```

#### Model Training
```python
from sklearn.ensemble import RandomForestClassifier
import joblib

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'rf_model.pkl')

# Load model
loaded_model = joblib.load('rf_model.pkl')
```

#### Model Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix

# Predictions
y_pred = model.predict(X_test)

# Metrics
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### Configuration Options

#### Audio Processing
```python
AUDIO_CONFIG = {
    'sample_rate': 22050,
    'duration': 2.5,
    'offset': 0.6,
    'n_mfcc': 20,
    'n_fft': 2048,
    'hop_length': 512
}
```

#### Model Configuration
```python
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 30,
        'min_samples_split': 5
    },
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale'
    },
    'knn': {
        'n_neighbors': 5,
        'weights': 'distance'
    }
}
```

## 🔍 Troubleshooting

### Common Issues & Solutions

#### 1. **Memory Errors**
```bash
# Reduce dataset size or process in batches
# Traditional ML models use minimal memory
```

#### 2. **Dataset Loading Errors**
```bash
# Verify dataset structure
python -c "import pandas as pd; print(pd.read_csv('data_path.csv').head())"

# Regenerate data paths
python -c "
import os
import pandas as pd
# Add your dataset regeneration code here
"
```

#### 3. **Package Conflicts**
```bash
# Clean reinstall
pip install --upgrade scikit-learn librosa pandas numpy
```

#### 4. **Insufficient Disk Space**
```bash
# Clean temporary files
rm -rf logs/*.log results/temp/
# Monitor space during augmentation
watch -n 5 'df -h .'
```

### Performance Optimization

#### For Large Datasets
```python
# Process data in batches
from sklearn.model_selection import train_test_split

# Use partial_fit for incremental learning (if supported)
# Or subsample for faster training
X_sample = X[:10000]  # Use subset for testing
y_sample = y[:10000]
```

#### For Limited Resources
```python
# Reduce feature dimensions
from sklearn.decomposition import PCA

pca = PCA(n_components=50)  # Reduce from 162 to 50 features
X_reduced = pca.fit_transform(X_train)
```

## 📊 Monitoring & Logging

### Training Progress
```bash
# Monitor model files
watch -n 10 'ls -lth *.pkl | head -5'

# System resources
htop
```

### Experiment Tracking
```python
# Track experiment results
import json

results = {
    'model': 'Random Forest',
    'accuracy': 0.79,
    'timestamp': '2025-10-01',
    'parameters': {'n_estimators': 200}
}

with open('experiment_log.json', 'a') as f:
    json.dump(results, f)
    f.write('\n')
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/MFCC_SER-Advanced.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 . --max-line-length=88
black . --line-length=88
```

### Areas for Contribution
- 🆕 Additional audio augmentation techniques
- 🏗️ New traditional ML algorithms
- 📊 Advanced evaluation metrics
- 🗃️ Support for more datasets
- ⚡ Performance optimizations
- 📝 Documentation improvements
- 🧪 Additional test coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Datasets
- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song
- **CREMA-D**: Crowd-sourced Emotional Multimodal Actors Dataset  
- **TESS**: Toronto Emotional Speech Set
- **SAVEE**: Surrey Audio-Visual Expressed Emotion Database

### Libraries & Frameworks
- **Scikit-learn**: Machine learning algorithms and utilities
- **Librosa**: Audio processing and feature extraction
- **NumPy/Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization

### Research References
```bibtex
@inproceedings{livingstone2018ravdess,
  title={The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)},
  author={Livingstone, Steven R and Russo, Frank A},
  booktitle={PLoS one},
  year={2018}
}

@inproceedings{cao2014crema,
  title={CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset},
  author={Cao, Houwei and Cooper, David G and Keutmann, Michael K},
  booktitle={IEEE transactions on affective computing},
  year={2014}
}
```

## 🚀 Getting Started Checklist

Before running the experiment, make sure you have:

- [ ] **Python 3.8+** installed
- [ ] **At least 10GB free disk space**
- [ ] **Stable internet connection** for package downloads
- [ ] **Dataset downloaded** and placed in correct structure
- [ ] **Virtual environment** created and activated
- [ ] **All dependencies** installed (`pip install -r requirements.txt`)
- [ ] **Sufficient time** allocated (30-60 minutes for full experiment)

### Quick Verification
```bash
# Verify setup
python -c "
import sklearn
import librosa
import pandas as pd
import numpy as np
print('✅ All core packages working')
print(f'scikit-learn: {sklearn.__version__}')
print(f'librosa: {librosa.__version__}')
"

# Check dataset
python -c "
import os
datasets = ['ravdess-emotional-speech-audio', 'cremad', 'toronto-emotional-speech-set-tess', 'surrey-audiovisual-expressed-emotion-savee']
found = [d for d in datasets if os.path.exists(f'dataset/{d}')]
print(f'✅ Found datasets: {found}')
print(f'❌ Missing datasets: {[d for d in datasets if d not in found]}')
"
```

---

## 🎯 Ready to Start?

### Quick Start (Jupyter Notebook)
```bash
jupyter notebook Speech_Emotion_Recognition_ML.ipynb
```

### Run ML Comparison (5-10 minutes)
```bash
python baseline_comparison.py
```

---

**🎉 Happy Experimenting! For questions, issues, or contributions, please open an issue or submit a pull request.**

---

<div align="center">

**⭐ Star this repository if it helped you!**

**🔗 [Report Bug](https://github.com/yourusername/MFCC_SER-Advanced/issues) • [Request Feature](https://github.com/yourusername/MFCC_SER-Advanced/issues) • [Contribute](CONTRIBUTING.md)**

Made with ❤️ for the Speech Processing Community

</div>