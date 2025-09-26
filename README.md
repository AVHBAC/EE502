# 🎵 EE502: Audio Signal Processing and Machine Learning

Welcome to the **EE502 Audio Signal Processing and Machine Learning** repository! This comprehensive collection contains practical implementations, examples, and resources covering the fundamentals of speech signal processing, feature extraction, and modern machine learning techniques applied to audio data.

## 📚 Course Overview

This repository accompanies the EE502 course textbook and provides hands-on implementations of key concepts in audio signal processing and machine learning. Each chapter builds upon previous concepts, creating a complete learning path from basic signal processing to advanced ML applications.

## 📖 Complete Course Materials

### 📄 **Course Textbook**
**[📖 EE502_BOOK_Sep20_version.pdf](./EE502_BOOK_Sep20_version.pdf)** - Complete course textbook covering all theoretical foundations and practical applications.

*Click the link above to view the PDF directly on GitHub, or download it for offline reading.*

### 🎯 **Class Examples**
All practical implementations and code examples are organized in the **[class_examples/](./class_examples/)** directory:

| Chapter | Topic | Description | Key Technologies |
|---------|-------|-------------|------------------|
| **[Chapter 4](./class_examples/chapter_04_audio_classification_mfcc_decision_trees/)** | Audio Classification Using MFCC and Decision Tree Models | Classify spoken digits (0-9) using MFCC features and machine learning | `MFCC`, `Random Forest`, `SVM`, `PCA` |
| **[Chapter 5](./class_examples/chapter_05_keyword_detection_random_forest/)** | Key-Word Detection using Random Forest | Detect keywords ("let", "go", "hold", "general") with 60%+ accuracy | `Random Forest`, `Feature Engineering`, `Data Augmentation` |
| **[Chapter 6](./class_examples/chapter_06_svm_speaker_classification/)** | SVM-based Speaker Classification | Speaker identification using Support Vector Machines | `SVM`, `Speaker Recognition`, `Performance Analysis` |
| **[Chapter 7](./class_examples/chapter_07_speech_emotion_recognition/)** | Model Comparisons for Speech Emotion Recognition | Advanced emotion recognition achieving 85%+ accuracy | `Neural Networks`, `MFCC`, `Ensemble Methods`, `Optuna` |
| **[Chapter 8](./class_examples/chapter_08_deepfake_detection_ml/)** | Deepfake Detection using ML | Detect AI-generated vs. real speech using machine learning | `Deepfake Detection`, `Audio Authentication`, `MFCC Analysis` |
| **[Chapter 9](./class_examples/chapter_09_voice_gender_identification/)** | Voice Based Gender Identification | Gender classification from voice characteristics | `Gender Recognition`, `Voice Analysis`, `Classification` |

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8+ required
pip install numpy pandas matplotlib scikit-learn librosa tensorflow
pip install soundfile scipy seaborn jupyter notebook
```

### Quick Start Guide

1. **Clone the repository:**
```bash
git clone https://github.com/AVHBAC/EE502.git
cd EE502
```

2. **Read the textbook:**
   - Open [EE502_BOOK_Sep20_version.pdf](./EE502_BOOK_Sep20_version.pdf) for theoretical background

3. **Explore class examples:**
```bash
cd class_examples/
ls  # Browse available chapters
```

4. **Run a specific chapter:**
```bash
cd chapter_04_audio_classification_mfcc_decision_trees/
# Follow the chapter's README for specific instructions
```

## 🏗️ Repository Structure

```
EE502/
├── 📄 README.md                           # This file
├── 📖 EE502_BOOK_Sep20_version.pdf        # Complete course textbook
├── 📁 class_examples/                     # All practical implementations
│   ├── 📁 chapter_04_audio_classification_mfcc_decision_trees/
│   ├── 📁 chapter_05_keyword_detection_random_forest/
│   ├── 📁 chapter_06_svm_speaker_classification/
│   ├── 📁 chapter_07_speech_emotion_recognition/
│   ├── 📁 chapter_08_deepfake_detection_ml/
│   └── 📁 chapter_09_voice_gender_identification/
└── 📄 .gitignore                          # Git ignore rules
```

## 🎯 Learning Path

### 🌟 **Beginner Track**
Start here if you're new to audio signal processing:

1. **Chapter 4**: Audio Classification - Learn MFCC basics and classification
2. **Chapter 5**: Keyword Detection - Understand feature engineering and Random Forest
3. **Chapter 6**: Speaker Classification - Master SVM and performance analysis

### 🚀 **Advanced Track**
For those with ML experience:

1. **Chapter 7**: Speech Emotion Recognition - Neural networks and advanced optimization
2. **Chapter 8**: Deepfake Detection - Modern audio authentication techniques
3. **Chapter 9**: Gender Identification - Comprehensive voice analysis

## 🛠️ Core Technologies Used

| Technology | Purpose | Chapters |
|------------|---------|----------|
| **MFCC (Mel-Frequency Cepstral Coefficients)** | Primary audio feature extraction | 4, 5, 6, 7, 8 |
| **scikit-learn** | Machine learning algorithms | 4, 5, 6, 7, 8, 9 |
| **TensorFlow/Keras** | Deep learning models | 7 |
| **librosa** | Audio processing and analysis | All |
| **Random Forest** | Classification algorithm | 4, 5 |
| **Support Vector Machines** | Classification and speaker ID | 4, 6 |
| **Neural Networks** | Advanced emotion recognition | 7 |
| **Optuna** | Hyperparameter optimization | 7 |

## 📊 Performance Highlights

Each chapter includes detailed performance analysis and results:

- **Chapter 4**: 100% accuracy on spoken digit classification
- **Chapter 5**: 60%+ accuracy on keyword detection with class balancing
- **Chapter 6**: Comprehensive speaker classification with performance scaling analysis
- **Chapter 7**: 85%+ accuracy on emotion recognition using ensemble methods
- **Chapter 8**: Effective deepfake detection using ML techniques
- **Chapter 9**: Robust gender identification from voice characteristics

## 🎓 Educational Value

This repository is perfect for:

- **Students** learning audio signal processing and machine learning
- **Researchers** exploring speech processing applications
- **Engineers** implementing audio analysis systems
- **Data Scientists** working with audio data
- **Anyone** interested in the intersection of signal processing and AI

## 📖 Chapter Deep Dive

### Chapter 4: Audio Classification Using MFCC and Decision Trees
Learn fundamental concepts of audio classification by building a system that can recognize spoken digits. This chapter covers:
- MFCC feature extraction
- Random Forest and SVM classifiers
- Principal Component Analysis (PCA)
- Model evaluation and comparison

### Chapter 5: Key-Word Detection using Random Forest
Build a practical keyword detection system with advanced feature engineering:
- 240+ audio features with intelligent selection
- Data augmentation for class balancing
- Ensemble methods for robust performance
- Systematic model improvement pipeline

### Chapter 6: SVM-based Speaker Classification
Master speaker identification techniques with comprehensive analysis:
- Speaker recognition using unique voice characteristics
- Performance scaling with dataset size
- Detailed confusion matrix analysis
- Cross-validation and overfitting detection

### Chapter 7: Speech Emotion Recognition
Implement state-of-the-art emotion recognition systems:
- Multi-dataset support (RAVDESS, CREMA-D, TESS, SAVEE)
- Advanced neural architectures (ResNet-1D, Attention, Transformers)
- Hyperparameter optimization with Optuna
- Data augmentation and ensemble methods

### Chapter 8: Deepfake Detection using ML
Explore modern audio authentication techniques:
- Real vs. AI-generated speech detection
- MFCC-based deepfake analysis
- Machine learning for audio authentication
- Practical security applications

### Chapter 9: Voice Based Gender Identification
Comprehensive gender classification from voice:
- Voice characteristic analysis
- Gender-specific feature extraction
- Classification model development
- Performance evaluation and optimization

## 🤝 Contributing

We welcome contributions to improve the repository:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

Please ensure:
- Code follows existing style conventions
- Documentation is updated accordingly
- All examples run successfully
- Performance metrics are maintained or improved

## 📄 License

This repository is for educational purposes. Please respect licensing terms of individual libraries and datasets used.

## 🆘 Support

### Getting Help
- Check individual chapter READMEs for specific guidance
- Review the course textbook for theoretical background
- Examine code comments and documentation
- Test with smaller datasets if experiencing performance issues

### Common Issues
- **Memory errors**: Reduce batch sizes or feature dimensions
- **Missing dependencies**: Install all required packages using pip
- **Audio loading issues**: Ensure audio files are in correct format (WAV recommended)
- **Performance issues**: Check dataset quality and balance

## 🌟 Acknowledgments

### Course Development
- **Dr. Imtiaz** - Course instructor and content development
- **Course contributors** - Implementation and testing

### Datasets and Libraries
- **RAVDESS, CREMA-D, TESS, SAVEE** - Emotion recognition datasets
- **Free Spoken Digit Dataset** - Digit classification data
- **LibROSA** - Audio processing library
- **scikit-learn** - Machine learning framework
- **TensorFlow** - Deep learning platform

---

## 📈 Repository Statistics

![Repository Views](https://img.shields.io/badge/Repository-EE502-blue)
![Chapters](https://img.shields.io/badge/Chapters-6-green)
![Technologies](https://img.shields.io/badge/Technologies-10+-orange)
![Performance](https://img.shields.io/badge/Best%20Accuracy-85%25+-red)

---

**🎉 Ready to start learning?** Begin with [Chapter 4](./class_examples/chapter_04_audio_classification_mfcc_decision_trees/) or dive into the [complete textbook](./EE502_BOOK_Sep20_version.pdf)!

**⭐ Star this repository** if you find it helpful for your audio signal processing and machine learning journey!