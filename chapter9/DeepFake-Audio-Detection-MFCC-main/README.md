# DeepFake Audio Detection using MFCC Features

[![GitHub license](https://img.shields.io/github/license/Yash-Sukhdeve/DeepFake-Audio-Detection-MFCC)](https://github.com/Yash-Sukhdeve/DeepFake-Audio-Detection-MFCC/blob/main/LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-orange.svg)](https://scikit-learn.org/)
[![librosa](https://img.shields.io/badge/librosa-0.9.1-green.svg)](https://librosa.org/)

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Dataset Preparation](#dataset-preparation)
6. [Usage](#usage)
   - [Training the Model](#training-the-model)
   - [Web Application](#web-application)
   - [Command Line Usage](#command-line-usage)
7. [Project Structure](#project-structure)
8. [Model Performance](#model-performance)
9. [API Documentation](#api-documentation)
10. [Troubleshooting](#troubleshooting)
11. [Contributing](#contributing)
12. [Citation](#citation)
13. [License](#license)

## Overview

This project implements a deepfake audio detection system using **Mel-frequency cepstral coefficients (MFCC)** features and **Support Vector Machine (SVM)** classification. The system can distinguish between genuine and synthetically generated (deepfake) audio files with high accuracy.

### Key Technologies
- **Feature Extraction**: MFCC (Mel-frequency cepstral coefficients)
- **Machine Learning**: Support Vector Machine with linear kernel
- **Audio Processing**: librosa library
- **Web Interface**: Flask web application
- **Model Persistence**: joblib for model serialization

This project was developed during the AIAmplify Hackathon and is based on the research paper:

> A. Hamza et al., "Deepfake Audio Detection via MFCC Features Using Machine Learning," in IEEE Access, vol. 10, pp. 134018-134028, 2022, doi: 10.1109/ACCESS.2022.3231480.

## Features

✅ **Audio Feature Extraction**: Robust MFCC feature extraction from WAV files  
✅ **Machine Learning Classification**: SVM-based binary classification  
✅ **Web Interface**: User-friendly Flask web application  
✅ **Command Line Interface**: Direct script execution for batch processing  
✅ **Model Persistence**: Trained models saved for reuse  
✅ **Cross-Platform**: Compatible with Windows, macOS, and Linux  
✅ **Batch Processing**: Support for analyzing multiple audio files  
✅ **Real-time Analysis**: Fast inference on new audio samples  

## System Requirements

### Hardware Requirements
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: At least 2GB free space (more for larger datasets)
- **CPU**: Multi-core processor recommended for faster training

### Software Requirements
- **Python**: 3.8+ (Tested with Python 3.11)
- **Operating System**: Windows 10+, macOS 10.14+, or Linux Ubuntu 18.04+
- **Audio Libraries**: FFmpeg (automatically installed with librosa)

## Installation

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Yash-Sukhdeve/DeepFake-Audio-Detection-MFCC.git
cd DeepFake-Audio-Detection-MFCC
```

### Step 2: Set up Python Environment

#### Option A: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

#### Option B: Using Conda

```bash
# Create conda environment
conda create -n deepfake-detection python=3.11
conda activate deepfake-detection
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

#### Core Dependencies
The main dependencies include:
- `librosa==0.9.1` - Audio processing and MFCC extraction
- `scikit-learn>=1.3.0` - Machine learning algorithms
- `numpy>=1.21.0` - Numerical computations
- `Flask>=2.0.0` - Web application framework
- `joblib>=1.0.0` - Model serialization

### Step 4: Verify Installation

```bash
# Test the installation
python -c "import librosa, sklearn, numpy, flask; print('All dependencies installed successfully!')"
```

## Dataset Preparation

### Directory Structure
Create the following directory structure for your audio dataset:

```
DeepFake-Audio-Detection-MFCC/
├── real_audio/
│   ├── sample_real_1.wav
│   ├── sample_real_2.wav
│   └── deepfake_audio/
│       ├── sample_fake_1.wav
│       ├── sample_fake_2.wav
│       └── sample_fake_3.wav
├── toanalyze/
│   └── test_audio.wav
└── uploads/ (created automatically)
```

### Audio File Requirements
- **Format**: WAV files only (`.wav` extension)
- **Sample Rate**: Any (automatically resampled by librosa)
- **Duration**: No strict limits, but 2-30 seconds recommended for best results
- **Quality**: Higher quality audio generally yields better results

### Sample Datasets
- **Fake-or-Real Dataset**: Available from the original research
- **ASVspoof Dataset**: Alternative dataset for testing
- **Custom Dataset**: You can use your own genuine and synthetic audio files

### Data Collection Guidelines
1. **Genuine Audio**: Use recordings from real human speakers
2. **Synthetic Audio**: Use AI-generated speech from tools like:
   - Text-to-Speech systems
   - Voice cloning software
   - AI voice generators

## Usage

### Training the Model

#### Step 1: Prepare Your Dataset
Place genuine audio files in `real_audio/` and synthetic audio files in `real_audio/deepfake_audio/`.

#### Step 2: Run Training Script
```bash
python main.py
```

**What happens during training:**
1. **Feature Extraction**: MFCC features extracted from all audio files
2. **Data Preprocessing**: Features standardized using StandardScaler
3. **Model Training**: SVM classifier trained on extracted features
4. **Model Evaluation**: Performance metrics calculated on test set
5. **Model Saving**: Trained model and scaler saved as `.pkl` files

#### Step 3: Analyze Test Audio
When prompted, provide the path to a WAV file you want to analyze:
```
Enter the path of the .wav file to analyze: toanalyze/test_audio.wav
```

### Web Application

#### Start the Flask Server
```bash
python app.py
```

The web application will be available at: `http://localhost:5001`

#### Using the Web Interface
1. **Open Browser**: Navigate to `http://localhost:5001`
2. **Upload Audio**: Click "Choose File" and select a WAV file
3. **Analyze**: Click "Analyze" button
4. **View Results**: See classification result (Genuine/Deepfake)

#### Web Application Features
- Drag-and-drop file upload
- Real-time processing
- Responsive design
- Error handling for invalid files
- Automatic file cleanup after analysis

### Command Line Usage

#### Analyze Single Audio File
```bash
python -c "from main import analyze_audio; analyze_audio('path/to/audio.wav')"
```

#### Batch Analysis
Create a script for batch processing:
```python
import os
from main import analyze_audio

audio_dir = "path/to/audio/files"
for filename in os.listdir(audio_dir):
    if filename.endswith('.wav'):
        filepath = os.path.join(audio_dir, filename)
        print(f"Analyzing: {filename}")
        analyze_audio(filepath)
```

## Project Structure

```
DeepFake-Audio-Detection-MFCC/
│
├── main.py                 # Core training and analysis script
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
│
├── templates/             # Flask templates
│   ├── index.html         # Main upload page
│   └── result.html        # Results display page
│
├── real_audio/            # Genuine audio samples
│   ├── *.wav             # Genuine audio files
│   └── deepfake_audio/   # Synthetic audio samples
│       └── *.wav         # Deepfake audio files
│
├── toanalyze/             # Test audio files
│   └── *.wav             # Files to be analyzed
│
├── uploads/               # Temporary upload directory (auto-created)
├── svm_model.pkl         # Trained SVM model (created after training)
├── scaler.pkl            # Feature scaler (created after training)
└── venv/                 # Virtual environment (if used)
```

## Model Performance

### MFCC Features
- **Number of Coefficients**: 13 (default)
- **FFT Window Size**: 2048 samples
- **Hop Length**: 512 samples
- **Feature Vector**: Mean MFCC values across time

### SVM Configuration
- **Kernel**: Linear
- **Feature Scaling**: StandardScaler (mean=0, std=1)
- **Train/Test Split**: 80/20
- **Cross-Validation**: Stratified sampling

### Expected Performance
Based on the original research and testing:
- **Accuracy**: 85-95% (depending on dataset quality)
- **Precision**: High for both genuine and synthetic classes
- **Recall**: Balanced performance across classes
- **F1-Score**: Consistent across different test sets

### Performance Factors
- **Audio Quality**: Higher quality audio improves accuracy
- **Dataset Balance**: Equal numbers of genuine/synthetic samples recommended
- **Audio Duration**: 2-10 second clips often work best
- **Speaker Diversity**: More diverse speakers in training improve generalization

## API Documentation

### Core Functions

#### `extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512)`
Extracts MFCC features from an audio file.

**Parameters:**
- `audio_path` (str): Path to the WAV audio file
- `n_mfcc` (int): Number of MFCC coefficients to extract (default: 13)
- `n_fft` (int): FFT window size (default: 2048)
- `hop_length` (int): Hop length for STFT (default: 512)

**Returns:**
- `numpy.ndarray`: Mean MFCC feature vector

#### `analyze_audio(input_audio_path)`
Analyzes an audio file using the trained model.

**Parameters:**
- `input_audio_path` (str): Path to the audio file to analyze

**Returns:**
- Prints classification result to console

#### `train_model(X, y)`
Trains the SVM model on the provided features and labels.

**Parameters:**
- `X` (numpy.ndarray): Feature matrix
- `y` (numpy.ndarray): Label vector (0=genuine, 1=deepfake)

### Flask API Endpoints

#### `GET /`
Serves the main upload page.

#### `POST /`
Processes uploaded audio file and returns analysis result.

**Form Data:**
- `audio_file`: WAV file upload

**Response:**
- Renders result page with classification

## Troubleshooting

### Common Issues and Solutions

#### 1. **ImportError: No module named 'librosa'**
```bash
# Solution: Install librosa and its dependencies
pip install librosa
# If on Linux, you might need:
sudo apt-get install ffmpeg
```

#### 2. **Error loading audio file**
```bash
# Ensure file is in WAV format
ffmpeg -i input.mp3 output.wav
```

#### 3. **Model file not found**
```bash
# Make sure you've trained the model first
python main.py
# This creates svm_model.pkl and scaler.pkl
```

#### 4. **Low accuracy results**
**Solutions:**
- Increase training dataset size
- Ensure dataset balance (equal genuine/fake samples)
- Check audio quality and format consistency
- Verify MFCC parameter settings

#### 5. **Web application not starting**
```bash
# Check if port 5001 is available
netstat -an | grep 5001

# Try a different port
# Modify app.py: app.run(debug=True, port=5002)
```

#### 6. **Memory issues with large datasets**
**Solutions:**
- Process audio files in batches
- Reduce number of MFCC coefficients
- Use shorter audio clips
- Increase system RAM

### Debugging Tips

#### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Check Audio File Properties
```python
import librosa
y, sr = librosa.load('audio_file.wav')
print(f"Duration: {len(y)/sr:.2f}s, Sample Rate: {sr}Hz")
```

#### Verify Model Performance
```python
# Check model accuracy after training
from sklearn.metrics import classification_report
# Add to train_model() function
print(classification_report(y_test, y_pred))
```

## Contributing

We welcome contributions to improve this project! Here's how you can help:

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests** (if applicable)
5. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Create a Pull Request**

### Areas for Contribution

- **Algorithm Improvements**: Better feature extraction methods
- **Web Interface**: Enhanced UI/UX design
- **Performance**: Optimization for speed and memory usage
- **Documentation**: Additional examples and tutorials
- **Testing**: Unit tests and integration tests
- **Datasets**: Curated datasets for training and testing

### Code Style Guidelines

- Follow PEP 8 Python style guide
- Add docstrings to all functions
- Include type hints where appropriate
- Write descriptive commit messages
- Add comments for complex algorithms

## Citation

If you use this project in your research, please cite:

### Original Research Paper
```bibtex
@article{hamza2022deepfake,
  title={Deepfake Audio Detection via MFCC Features Using Machine Learning},
  author={Hamza, A. and others},
  journal={IEEE Access},
  volume={10},
  pages={134018--134028},
  year={2022},
  publisher={IEEE},
  doi={10.1109/ACCESS.2022.3231480}
}
```

### This Implementation
```bibtex
@misc{deepfake-audio-detection-mfcc,
  title={DeepFake Audio Detection using MFCC Features},
  author={Yash Sukhdeve and contributors},
  year={2024},
  url={https://github.com/Yash-Sukhdeve/DeepFake-Audio-Detection-MFCC}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ Liability
- ❌ Warranty

## Acknowledgments

- **Original Research**: A. Hamza et al. for the foundational research
- **AIAmplify Hackathon**: Event that inspired this project
- **Contributors**: 
  - Noor Chauhan
  - [Abhishek Khadgi](https://github.com/abhis-hek)
  - Omkar Sapkal
  - Himanshi Shinde
  - Furqan Ali

## Contact

For questions, suggestions, or collaborations:

- **GitHub**: [Yash-Sukhdeve](https://github.com/Yash-Sukhdeve)
- **Issues**: [Report issues here](https://github.com/Yash-Sukhdeve/DeepFake-Audio-Detection-MFCC/issues)
- **Discussions**: [Join discussions](https://github.com/Yash-Sukhdeve/DeepFake-Audio-Detection-MFCC/discussions)

---

**⭐ If you find this project helpful, please give it a star on GitHub!**
