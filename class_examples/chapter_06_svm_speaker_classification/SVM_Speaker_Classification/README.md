# SVM Speaker Classification Performance Analysis

A comprehensive comparison of Support Vector Machine (SVM) performance for speaker identification tasks across different dataset sizes (5, 10, and 20 speakers).

## 🎯 Project Overview

This project demonstrates how SVM classifier performance varies with the complexity of the speaker identification task. It includes synthetic audio data generation, feature extraction using MFCC (Mel-Frequency Cepstral Coefficients), and detailed performance analysis across three different dataset sizes.

### Key Features
- **Synthetic Audio Generation**: Creates realistic speech-like signals with unique speaker characteristics
- **MFCC Feature Extraction**: Extracts comprehensive acoustic features with noise reduction
- **SVM Classification**: Implements RBF kernel SVM with proper scaling and cross-validation
- **Performance Comparison**: Analyzes accuracy, training time, and computational complexity across dataset sizes
- **Comprehensive Visualization**: Generates detailed plots and confusion matrices
- **Automated Reporting**: Creates detailed analysis reports with insights and recommendations

## 📊 Results Summary

| Dataset Size | Test Accuracy | CV Accuracy | Performance Drop |
|-------------|---------------|-------------|------------------|
| 5 Speakers  | 100.0%        | 72.5% ± 9.4% | Baseline        |
| 10 Speakers | 60.0%         | 26.3% ± 14.5% | -40.0%         |
| 20 Speakers | 42.5%         | 18.8% ± 6.8% | -57.5%         |

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Quick Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Yash-Sukhdeve/SVM-Speaker-Classification-Project.git
   cd SVM-Speaker-Classification-Project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation** (optional):
   ```bash
   python -c "import librosa, sklearn, numpy; print('All dependencies installed successfully!')"
   ```

## 🚀 Usage

### Option 1: Run Complete Performance Comparison (Recommended)
This is the main script that runs the entire analysis:

```bash
python svm_performance_comparison.py
```

**What it does:**
- Evaluates SVM performance on 5, 10, and 20-speaker datasets
- Generates comprehensive performance plots
- Creates confusion matrices for all datasets  
- Produces a detailed analysis report
- Saves all results to files

**Output Files:**
- `svm_performance_comparison.png` - Performance comparison charts
- `confusion_matrices_comparison.png` - Confusion matrix visualizations
- `svm_performance_report.txt` - Detailed analysis report

### Option 2: Run Individual Classification (Basic)
To run classification on the current dataset (5 speakers by default):

```bash
python SVM_Classifier.py
```

**What it does:**
- Extracts MFCC features from audio files
- Trains SVM classifier
- Evaluates performance with various metrics
- Saves trained model and preprocessed data

## 📁 Project Structure

```
SVM_Speaker_Classification/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── SVM_Classifier.py                      # Basic SVM implementation
├── svm_performance_comparison.py          # Complete performance analysis
├── datasets/                              # Organized datasets
│   ├── 5_speakers/                        # 5-speaker dataset
│   │   ├── traindata/                     # Training audio files
│   │   ├── testdata/                      # Test audio files
│   │   ├── X_train.csv                    # Training features
│   │   ├── X_test.csv                     # Test features
│   │   ├── Y_train.csv                    # Training labels
│   │   ├── Y_test.csv                     # Test labels
│   │   └── svm_voice_recognition_model.pkl # Trained model
│   ├── 10_speakers/                       # 10-speaker dataset
│   └── 20_speakers/                       # 20-speaker dataset
├── traindata/                             # Current training data (5 speakers)
├── testdata/                              # Current test data (5 speakers)
├── backup_20speakers/                     # Backup of original 20-speaker data
├── svm_performance_comparison.png         # Performance plots
├── confusion_matrices_comparison.png      # Confusion matrices
└── svm_performance_report.txt            # Analysis report
```

## 🔬 Technical Details

### Dataset Characteristics
- **Synthetic Audio**: 3-second WAV files at 48kHz sample rate
- **Speaker Modeling**: Unique fundamental frequencies, formants, and speaking rates
- **Noise Reduction**: Applied using spectral subtraction techniques
- **Data Split**: 80% training, 20% testing per speaker

### Feature Extraction
The system extracts **21 MFCC-based features** per audio file:
- 3 MFCC coefficients (n_mfcc=3)
- 7 statistical measures per coefficient:
  - Mean, Median, Standard Deviation
  - Skewness, Kurtosis  
  - Maximum, Minimum

### SVM Configuration
- **Kernel**: RBF (Radial Basis Function)
- **Decision Function**: One-vs-Rest (OvR)
- **Scaling**: StandardScaler normalization
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Random State**: 42 (for reproducibility)

### Performance Metrics
- **Test Accuracy**: Performance on held-out test set
- **Cross-Validation Accuracy**: 5-fold CV on training data
- **Confusion Matrix**: Classification accuracy per speaker
- **Training Time**: Model training duration
- **Feature Extraction Time**: Audio processing time

## 📈 Understanding the Results

### Key Findings
1. **Accuracy Degradation**: Performance drops significantly as the number of speakers increases
2. **Overfitting**: Large gap between test and CV accuracy in smaller datasets
3. **Computational Scaling**: Feature extraction time scales linearly with dataset size
4. **Classification Challenge**: 20-speaker identification is substantially harder than 5-speaker

### Performance Analysis
- **5 Speakers**: Perfect test accuracy but indicates potential overfitting
- **10 Speakers**: Balanced complexity, good for demonstrating scalability challenges
- **20 Speakers**: Realistic difficulty level, shows practical limitations

### Practical Implications
- **Small datasets**: May suffer from overfitting despite high test accuracy
- **Larger datasets**: Provide more realistic performance estimates
- **Feature engineering**: Current MFCC features may be insufficient for large-scale speaker identification

## 🔄 Reproducibility

### Ensuring Consistent Results
1. **Fixed Random Seeds**: All random operations use `random_state=42`
2. **Deterministic Processing**: Audio files processed in alphabetical order
3. **Consistent Scaling**: Same StandardScaler approach across all datasets
4. **Version Pinning**: Specific package versions in requirements.txt

### Expected Runtime
- **5 speakers**: ~5 seconds
- **10 speakers**: ~10 seconds  
- **20 speakers**: ~15 seconds
- **Full comparison**: ~30 seconds total

### Hardware Requirements
- **Minimum**: 4GB RAM, dual-core processor
- **Recommended**: 8GB RAM, quad-core processor
- **Storage**: ~50MB for datasets and results
- **OS**: Windows, macOS, or Linux

### Verification Steps
To verify your results match the expected output:
1. Check that test accuracies are: 100%, 60%, 42.5%
2. Verify CV accuracies are approximately: 72.5%, 26.3%, 18.8%
3. Confirm all output files are generated
4. Review confusion matrices for diagonal dominance in 5-speaker case

## 🧩 Customization

### Adding More Speakers
To extend the analysis to more speakers:
1. Create additional synthetic data or use real audio samples
2. Update the `dataset_configs` in `svm_performance_comparison.py`
3. Adjust visualization layouts for additional datasets

### Modifying Features  
To experiment with different features:
1. Edit the `preprocess_and_extract_file_level_features()` function
2. Adjust `n_mfcc` parameter for different numbers of MFCC coefficients
3. Add spectral features, chroma, or other audio descriptors

### SVM Parameter Tuning
To optimize SVM parameters:
1. Use GridSearchCV in the training section
2. Experiment with different kernels (linear, polynomial, sigmoid)
3. Tune C and gamma parameters for RBF kernel

## 🐛 Troubleshooting

### Common Issues

**ImportError: Missing packages**
```bash
pip install -r requirements.txt
```

**RuntimeWarning: Audio file issues**  
- Ensure audio files are valid WAV format
- Check sample rate compatibility (48kHz expected)

**Low accuracy results**
- Verify dataset integrity
- Check if random seeds are consistent
- Ensure proper feature scaling

**Memory issues with large datasets**
- Process datasets in smaller batches
- Reduce n_mfcc parameter
- Use feature selection techniques

### Performance Issues
- **Slow feature extraction**: Consider reducing sample rate or duration
- **Long training time**: Try linear kernel for faster training  
- **High memory usage**: Process files individually instead of batch loading

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.21.0 | Numerical computing |
| pandas | >=1.3.0 | Data manipulation |
| scikit-learn | >=1.0.0 | Machine learning |
| librosa | >=0.9.0 | Audio processing |
| scipy | >=1.7.0 | Scientific computing |
| matplotlib | >=3.5.0 | Plotting |
| seaborn | >=0.11.0 | Statistical visualization |
| noisereduce | >=2.0.0 | Audio denoising |
| joblib | >=1.1.0 | Model serialization |

## 🎓 Educational Value

This project is ideal for:
- **Machine Learning Students**: Understanding SVM classification with real-world audio data
- **Signal Processing Courses**: Learning MFCC feature extraction and audio preprocessing
- **Data Science Projects**: Exploring performance scaling and overfitting challenges
- **Research**: Baseline implementation for speaker identification systems

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **LibROSA**: For excellent audio processing capabilities
- **scikit-learn**: For robust machine learning implementations  
- **Synthetic Audio Generation**: Inspired by speech synthesis research
- **MFCC Features**: Based on speech recognition best practices
- **Original work by**: Ansen Herrick

---

**Note**: This project uses synthetic audio data for educational purposes. For production applications, consider using real speaker datasets like VoxCeleb or LibriSpeech.
