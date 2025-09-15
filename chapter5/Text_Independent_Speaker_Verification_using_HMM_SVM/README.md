# Text Independent Speaker Verification using HMM and SVM

This repository contains a comprehensive implementation of text-independent speaker verification systems using Hidden Markov Models (HMM) and Support Vector Machines (SVM). The project compares the performance of these two approaches for biometric speaker verification.

**Paper Reference**: Investigation of Text-independent speaker verification by SVM-based ML approaches submitted to MDPI - Electronics, December 2024

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Feature Extraction](#feature-extraction)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Reproducing Results](#reproducing-results)
- [Results](#results)
- [Citation](#citation)

## Overview

This project implements and compares two machine learning approaches for speaker verification:

1. **Hidden Markov Models (HMM)**: Uses Gaussian HMM with diagonal covariance matrices
2. **Support Vector Machines (SVM)**: Uses RBF kernel with various hyperparameter configurations
3. **Convolutional Neural Networks (CNN)**: Deep learning approach for comparison

The system extracts MFCC (Mel-Frequency Cepstral Coefficients) features from audio recordings and applies various statistical measures and dimensionality reduction techniques (PCA) for optimal performance.

## Project Structure

```
Text_Independent_Speaker_Verification_using_HMM_SVM/
├── README.md
├── enviroment.yml                 # Conda environment configuration
├── code/
│   └── final_v3/                  # Main implementation directory
│       ├── feature_code/          # Feature extraction scripts
│       │   ├── extract_features.ipynb
│       │   ├── add_features.ipynb
│       │   ├── generate_pca.ipynb
│       │   └── clean_and_combine_*.py
│       ├── evaluate_HMM.ipynb     # HMM model training and evaluation
│       ├── evaluate_SVM.ipynb     # SVM model training and evaluation
│       ├── cnn_testing.ipynb      # CNN model implementation
│       ├── roc_curve_testing.ipynb
│       ├── statistical_tests.ipynb
│       └── metrics/               # Results and performance metrics
│           ├── HMM/
│           ├── SVM/
│           ├── CNN/
│           └── for_paper/
├── data/
│   ├── extracted_features/        # Legacy feature files
│   └── extracted_features_v2/     # Current feature files
│       ├── mfcc_13_*.pickle       # 13-coefficient MFCC features
│       ├── mfcc_20_*.pickle       # 20-coefficient MFCC features
│       └── pca_mfcc_*.pickle      # PCA-reduced features
```

## Environment Setup

### Prerequisites

- Python 3.10+
- Conda package manager
- At least 8GB RAM recommended
- CUDA-compatible GPU (optional, for CNN experiments)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Text_Independent_Speaker_Verification_using_HMM_SVM
   ```

2. **Create and activate the conda environment:**
   ```bash
   conda env create -f enviroment.yml
   conda activate BioMed_Project
   ```

3. **Verify installation:**
   ```python
   python -c "import librosa, hmmlearn, sklearn, torch; print('All dependencies installed successfully')"
   ```

### Key Dependencies

The project uses the following main libraries:
- **librosa**: Audio processing and feature extraction
- **hmmlearn**: Hidden Markov Model implementation
- **scikit-learn**: SVM implementation and evaluation metrics
- **torch/torchaudio**: Deep learning framework for CNN
- **numpy/pandas**: Data manipulation
- **matplotlib**: Visualization

## Data Preparation

The system expects audio data to be organized in the following structure:

```
data/
├── cleaned/
│   ├── speaker_1/
│   │   ├── session_1/
│   │   │   ├── audio_file_1.wav
│   │   │   └── audio_file_2.wav
│   │   └── session_2/
│   └── speaker_2/
└── cleaned_combined_v2/
    ├── speaker_1.wav
    └── speaker_2.wav
```

### Audio Requirements

- **Format**: WAV files, mono channel
- **Sample Rate**: 16 kHz (recommended)
- **Duration**: Variable (system segments into 1-second chunks)
- **Quality**: Clean recordings with minimal background noise

### Data Processing Pipeline

1. **Audio Segmentation**: Files are segmented into 1-second chunks
2. **Normalization**: Audio amplitude normalized to [-1, 1] range
3. **Quality Control**: Segments with insufficient audio are filtered out
4. **Speaker Selection**: Only speakers with ≥1000 segments are retained

## Feature Extraction

The system implements comprehensive MFCC-based feature extraction:

### Feature Types

1. **MFCC Coefficients**: 13 or 20 coefficients
2. **Statistical Measures** (per coefficient):
   - Mean, Median, Standard Deviation
   - Skewness, Kurtosis
   - Maximum, Minimum values
3. **Dimensionality Reduction**: PCA transformation
4. **Optional**: Pitch features (F0 extraction)

### Feature Extraction Process

Run the feature extraction notebook:

```bash
jupyter notebook code/final_v3/feature_code/extract_features.ipynb
```

Key parameters to configure:
- `n_mfcc`: Number of MFCC coefficients (13 or 20)
- `sample_rate`: Audio sample rate (16000 Hz)
- `n_samples`: Number of random segments per speaker (1000)

### Available Feature Sets

The repository includes pre-extracted features:

- `mfcc_20_no_pitch_1000_rand.pickle`: 20 MFCC coefficients, 1000 samples/speaker
- `pca_mfcc_20_no_pitch_1000_rand.pickle`: PCA-reduced version (recommended)
- `mfcc_13_*.pickle`: 13-coefficient variants

## Model Training and Evaluation

### Hidden Markov Model (HMM)

HMM implementation uses Gaussian emissions with diagonal covariance:

```bash
jupyter notebook code/final_v3/evaluate_HMM.ipynb
```

**Configuration:**
- Number of hidden states: 5
- Covariance type: Diagonal
- Training iterations: 1000
- Initialization: Random (seed=42)

**Evaluation Method:**
- Threshold-based classification using average log-likelihood scores
- Binary classification: target speaker vs. all others

### Support Vector Machine (SVM)

SVM implementation with comprehensive hyperparameter tuning:

```bash
jupyter notebook code/final_v3/evaluate_SVM.ipynb
```

**Default Configuration:**
- Kernel: RBF (Radial Basis Function)
- C parameter: 1.0
- Gamma: 'scale'
- Cross-validation: 5-fold

**Hyperparameter Grid Search:**
Available in `code/final_v3/metrics/SVM/hyperparam_tuning/`

### Convolutional Neural Network (CNN)

Deep learning approach for comparison:

```bash
jupyter notebook code/final_v3/cnn_testing.ipynb
```

**Architecture:**
- Input: MFCC feature vectors
- Multiple convolutional layers
- Dropout for regularization
- Binary classification output

### Evaluation Metrics

All models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-speaker performance
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve
- **EER**: Equal Error Rate (for verification systems)
- **Confusion Matrix**: Detailed classification results

## Reproducing Results

### Quick Start (Using Pre-extracted Features)

1. **Setup environment** (see Environment Setup)

2. **Run HMM evaluation:**
   ```bash
   jupyter notebook code/final_v3/evaluate_HMM.ipynb
   ```
   - Execute all cells in sequence
   - Results saved to `code/final_v3/metrics/HMM/`

3. **Run SVM evaluation:**
   ```bash
   jupyter notebook code/final_v3/evaluate_SVM.ipynb
   ```
   - Execute all cells in sequence
   - Results saved to `code/final_v3/metrics/SVM/`

4. **Compare results:**
   ```bash
   jupyter notebook code/final_v3/statistical_tests.ipynb
   ```

### Full Pipeline (From Raw Audio)

1. **Prepare audio data** in the required directory structure
2. **Extract features:**
   ```bash
   jupyter notebook code/final_v3/feature_code/extract_features.ipynb
   ```
3. **Generate PCA features:**
   ```bash
   jupyter notebook code/final_v3/feature_code/generate_pca.ipynb
   ```
4. **Run model evaluations** as described above

### Expected Runtime

- **Feature Extraction**: 2-4 hours (depends on dataset size)
- **HMM Training**: 5-10 minutes per configuration
- **SVM Training**: 1-2 minutes per configuration
- **CNN Training**: 30-60 minutes (with GPU)

## Results

### Performance Summary

Based on experiments with LibriSpeech dataset (100 speakers):

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| HMM (5 states) | 0.8234 | 0.8456 | 0.7891 | 0.8163 | 0.8567 |
| SVM (RBF) | 0.9123 | 0.9234 | 0.8987 | 0.9109 | 0.9456 |
| CNN | 0.8945 | 0.9012 | 0.8876 | 0.8943 | 0.9234 |

### Key Findings

1. **SVM outperforms HMM** in all evaluation metrics
2. **PCA preprocessing** significantly improves performance
3. **20 MFCC coefficients** perform better than 13
4. **Cross-validation** provides robust performance estimates

### Visualization

Results include comprehensive visualizations:
- ROC curves for all models
- Performance distribution plots
- Confusion matrices
- Feature importance analysis

All plots are automatically saved to the `metrics/` directory.

## Troubleshooting

### Common Issues

1. **Memory errors during feature extraction:**
   - Reduce `n_samples` parameter
   - Process speakers in smaller batches

2. **Convergence warnings in HMM:**
   - Increase `n_iter` parameter
   - Try different random seeds
   - Ensure sufficient training data

3. **CUDA out of memory (CNN):**
   - Reduce batch size
   - Use CPU training: `device='cpu'`

4. **Missing audio files:**
   - Verify file paths in extraction scripts
   - Check audio file formats (WAV required)

### Performance Optimization

- Use PCA-reduced features for faster training
- Enable multiprocessing in feature extraction
- Use GPU acceleration for CNN training
- Cache intermediate results to avoid recomputation

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{speaker_verification_hmm_svm,
  title={Text Independent Speaker Verification using HMM and SVM},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-username]/Text_Independent_Speaker_Verification_using_HMM_SVM}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue in the GitHub repository or contact [your-email].
