# 🎵 EE502: Audio Signal Processing and Machine Learning

Welcome to the **EE502 Audio Signal Processing and Machine Learning** repository! This comprehensive collection contains practical implementations, examples, and resources covering the fundamentals of speech signal processing, feature extraction, and modern machine learning techniques applied to audio data.

## 📚 Course Overview

This repository accompanies the EE502 course textbook and provides hands-on implementations of key concepts in audio signal processing and machine learning. Each chapter builds upon previous concepts, creating a complete learning path from basic signal processing to advanced ML applications.

## 📖 Complete Course Materials

### 📄 **Course Textbook**
**[📖 EE502_Draft_Booklet_Oct2nd_version.pdf](./EE502_Draft_Booklet_Oct2nd_version.pdf)** - Complete course textbook covering all theoretical foundations and practical applications.

*Click the link above to view the PDF directly on GitHub, or download it for offline reading.*

### 🎓 **Interactive Teaching Notebooks**
The **[class_examples/](./class_examples/)** directory contains three comprehensive Jupyter notebooks designed for classroom teaching using the Mini Speech Commands dataset:

| Notebook | Topic | Description | Key Concepts |
|----------|-------|-------------|--------------|
| **[Audio_DT_RF_MiniSpeechCommands.ipynb](./class_examples/Audio_DT_RF_MiniSpeechCommands%20(1).ipynb)** | Decision Trees & Random Forests | Complete ML pipeline for audio classification with 10 spoken commands | `MFCC`, `Decision Trees`, `Random Forest`, `GridSearchCV`, `ROC`, `Feature Importance` |
| **[Audio_Regression_MiniSpeechCommands.ipynb](./class_examples/Audio_Regression_MiniSpeechCommands%20(1).ipynb)** | Audio Regression Analysis | Predict continuous targets (loudness proxy) from audio features | `Linear Regression`, `Ridge`, `Random Forest Regressor`, `k-NN`, `Residual Analysis` |
| **[EE502_LDA_QDA_PCA_MiniSpeech.ipynb](./class_examples/EE502_LDA_QDA_PCA_MiniSpeech.ipynb)** | Dimensionality Reduction & Classification | Advanced dimensionality reduction and discriminant analysis techniques | `LDA`, `QDA`, `PCA`, `Feature Reduction`, `Statistical Classification` |

### 📚 **Book Scripts - Chapter Implementations**
The **[Book_Scripts/](./Book_Scripts/)** directory contains complete chapter implementations corresponding to the textbook:

| Chapter | Topic | Description | Key Technologies |
|---------|-------|-------------|------------------|
| **[Chapter 4](./Book_Scripts/Chapter4_Audio_Classification_Using_MFCCs_and_Decision_Tree_Models/)** | Audio Classification Using MFCCs and Decision Tree Models | Classify spoken digits using MFCC features and ML classifiers | `MFCC`, `Random Forest`, `SVM`, `PCA` |
| **[Chapter 5](./Book_Scripts/Chapter5_Key-Word_Detection_using_Random_Forest/)** | Key-Word Detection using Random Forest | Keyword detection with advanced feature engineering | `Random Forest`, `Feature Engineering`, `Data Augmentation` |
| **[Chapter 6](./Book_Scripts/Chapter6_SVM-based_Speaker_Classification/)** | SVM-based Speaker Classification | Speaker identification using Support Vector Machines | `SVM`, `Speaker Recognition`, `Performance Analysis` |
| **[Chapter 7](./Book_Scripts/Chapter7_Speech_Emotion_Recognition_using_MFCC_and_Machine_Learning/)** | Speech Emotion Recognition using MFCC and ML | Emotion recognition achieving 85%+ accuracy | `MFCC`, `Neural Networks`, `Ensemble Methods`, `Optuna` |
| **[Chapter 8](./Book_Scripts/Chapter8_Voice-based_Gender_Identification/)** | Voice-based Gender Identification | Gender classification from voice characteristics | `Gender Recognition`, `Voice Analysis`, `Classification` |
| **[Chapter 9](./Book_Scripts/Chapter9_Deepfake_Audio_Detection_via_MFCC_Features_Using_Machine_Learning/)** | Deepfake Audio Detection via MFCC Features | Detect AI-generated vs. real speech using machine learning | `Deepfake Detection`, `Audio Authentication`, `MFCC Analysis` |

### 🔬 **Research Project: Text-Independent Speaker Verification**
**[text_independant_Spkr_vrf_old_chapter5/](./text_independant_Spkr_vrf_old_chapter5/)** - Advanced speaker verification research project comparing HMM and SVM approaches for biometric authentication.

**Paper Reference**: Investigation of Text-independent speaker verification by SVM-based ML approaches submitted to MDPI - Electronics, December 2024

**Key Features:**
- Hidden Markov Models (HMM) with Gaussian emissions
- Support Vector Machines (SVM) with RBF kernel
- Convolutional Neural Networks (CNN) for comparison
- Comprehensive MFCC feature extraction with PCA
- ROC analysis, EER metrics, and statistical validation
- Performance: SVM achieves 91.23% accuracy vs HMM 82.34%

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8+ required
pip install numpy pandas matplotlib scikit-learn librosa tensorflow
pip install soundfile scipy seaborn jupyter notebook hmmlearn tqdm
```

### Quick Start Guide

**Option 1: Interactive Teaching Notebooks (Recommended for Beginners)**
1. **Clone the repository:**
```bash
git clone https://github.com/AVHBAC/EE502.git
cd EE502
```

2. **Start with classroom notebooks:**
```bash
cd class_examples/
jupyter notebook
# Open Audio_DT_RF_MiniSpeechCommands (1).ipynb
```

**Option 2: Complete Chapter Implementations**
1. **Explore book chapters:**
```bash
cd Book_Scripts/
ls  # Browse available chapters (Chapter4-9)
```

2. **Run a specific chapter:**
```bash
cd Chapter4_Audio_Classification_Using_MFCCs_and_Decision_Tree_Models/
cd Audio-Classification-Using-MFCCs-and-Decision-Tree-Models/
# Follow the chapter's README for specific instructions
```

**Option 3: Research Project**
```bash
cd text_independant_Spkr_vrf_old_chapter5/
cd Text_Independent_Speaker_Verification_using_HMM_SVM/
# See README.md for detailed setup and usage
```

## 🏗️ Repository Structure

```
EE502/
├── 📄 README.md                                    # This file
├── 📖 EE502_Draft_Booklet_Oct2nd_version.pdf       # Complete course textbook
│
├── 📁 class_examples/                              # Interactive teaching notebooks
│   ├── 📓 Audio_DT_RF_MiniSpeechCommands (1).ipynb
│   ├── 📓 Audio_Regression_MiniSpeechCommands (1).ipynb
│   └── 📓 EE502_LDA_QDA_PCA_MiniSpeech.ipynb
│
├── 📁 Book_Scripts/                                # Complete chapter implementations
   ├── 📁 Chapter4_Audio_Classification_Using_MFCCs_and_Decision_Tree_Models/
   │   └── Audio-Classification-Using-MFCCs-and-Decision-Tree-Models/
   ├── 📁 Chapter5_Key-Word_Detection_using_Random_Forest/
   │   └── Key-Word-Detection-using-Random-Forest/
   ├── 📁 Chapter6_SVM-based_Speaker_Classification/
   │   └── SVM_Speaker_Classification/
   ├── 📁 Chapter7_Speech_Emotion_Recognition_using_MFCC_and_Machine_Learning/
   │   └── MFCC_SER-main/
   ├── 📁 Chapter8_Voice-based_Gender_Identification/
   │   └── voice_based_gender_identification/
   └── 📁 Chapter9_Deepfake_Audio_Detection_via_MFCC_Features_Using_Machine_Learning/
       └── DeepFake-Audio-Detection-MFCC-main/
```

## 🎯 Learning Path

### 🌟 **Beginner Track: Start with Interactive Notebooks**
Perfect for classroom teaching and hands-on learning:

1. **Audio_DT_RF_MiniSpeechCommands.ipynb** - Learn audio classification fundamentals with Decision Trees and Random Forests
   - Download Mini Speech Commands dataset automatically
   - Extract MFCC and spectral features
   - Train and compare classifiers with GridSearchCV
   - Visualize results with confusion matrices and ROC curves

2. **Audio_Regression_MiniSpeechCommands.ipynb** - Master regression techniques for audio
   - Predict continuous targets from audio features
   - Compare Linear, Ridge, Random Forest, and k-NN regressors
   - Analyze residuals and model performance

3. **EE502_LDA_QDA_PCA_MiniSpeech.ipynb** - Advanced dimensionality reduction
   - Apply PCA for feature reduction
   - Implement LDA and QDA for classification
   - Compare discriminant analysis techniques

### 📚 **Intermediate Track: Book Chapter Implementations**
Deep dive into textbook chapters with complete implementations:

1. **Chapter 4**: Audio Classification Using MFCCs - Spoken digit recognition
2. **Chapter 5**: Keyword Detection - Advanced feature engineering with Random Forest
3. **Chapter 6**: Speaker Classification - SVM-based speaker identification

### 🚀 **Advanced Track: Cutting-Edge Applications**
State-of-the-art audio processing techniques:

1. **Chapter 7**: Speech Emotion Recognition - Neural networks with Optuna optimization
2. **Chapter 8**: Voice-based Gender Identification - Gender classification from voice
3. **Chapter 9**: Deepfake Audio Detection - AI-generated speech detection

### 🔬 **Research Track: Speaker Verification**
For advanced students and researchers:

- **Text-Independent Speaker Verification** - Compare HMM, SVM, and CNN approaches for biometric authentication with comprehensive evaluation metrics and statistical validation

## 🛠️ Core Technologies Used

| Technology | Purpose | Where Used |
|------------|---------|------------|
| **MFCC (Mel-Frequency Cepstral Coefficients)** | Primary audio feature extraction | All chapters, notebooks, research |
| **librosa** | Audio processing and analysis | All implementations |
| **scikit-learn** | ML algorithms (RF, SVM, Linear models) | All chapters and notebooks |
| **Random Forest** | Classification and regression | Ch 4, 5, 7 + teaching notebooks |
| **Support Vector Machines (SVM)** | Classification and speaker verification | Ch 4, 6 + research project |
| **Decision Trees** | Classification and regression | Ch 4 + teaching notebooks |
| **Hidden Markov Models (HMM)** | Speaker verification | Research project |
| **Neural Networks / Deep Learning** | Advanced emotion recognition | Ch 7 + research project (CNN) |
| **TensorFlow/Keras/PyTorch** | Deep learning frameworks | Ch 7, research project |
| **Optuna** | Hyperparameter optimization | Ch 7 |
| **PCA** | Dimensionality reduction | Ch 4, research project, notebooks |
| **LDA/QDA** | Discriminant analysis | Teaching notebooks |
| **GridSearchCV** | Model hyperparameter tuning | Teaching notebooks, chapters |
| **hmmlearn** | HMM implementation | Research project |

## 📊 Performance Highlights

### Teaching Notebooks
- **Audio Classification**: High accuracy on 10-command speech recognition with automatic dataset download
- **Regression Analysis**: Comprehensive comparison of Linear, Ridge, RF, and k-NN regressors
- **Dimensionality Reduction**: Effective feature reduction with PCA, LDA, and QDA

### Book Chapters
- **Chapter 4**: High accuracy on spoken digit classification using MFCC + ML
- **Chapter 5**: 60%+ accuracy on keyword detection with advanced feature engineering
- **Chapter 6**: Robust speaker identification with comprehensive performance analysis
- **Chapter 7**: 85%+ accuracy on speech emotion recognition with neural networks
- **Chapter 8**: Accurate gender identification from voice characteristics
- **Chapter 9**: Effective deepfake audio detection using MFCC features

### Research Project
- **Speaker Verification**: SVM achieves 91.23% accuracy vs HMM 82.34%
- Comprehensive evaluation with ROC curves, EER, and statistical validation
- Paper submitted to MDPI - Electronics, December 2024

## 🎓 Educational Value

This repository is perfect for:

- **Students** learning audio signal processing and machine learning
- **Researchers** exploring speech processing applications
- **Engineers** implementing audio analysis systems
- **Data Scientists** working with audio data
- **Anyone** interested in the intersection of signal processing and AI

## 📖 Deep Dive: Notebooks and Chapters

### 🎓 Teaching Notebooks (class_examples/)

#### Audio_DT_RF_MiniSpeechCommands.ipynb
A complete, production-ready notebook for teaching audio classification:
- **Automatic dataset download**: Mini Speech Commands (10 classes: down, go, left, no, off, on, right, stop, up, yes)
- **Rich feature extraction**: 40 MFCCs + deltas, spectral features, chroma, ZCR, RMS (170+ features)
- **Comprehensive ML pipeline**: Train/test split, stratified k-fold CV, GridSearchCV
- **Model comparison**: Decision Trees vs Random Forests with hyperparameter tuning
- **Advanced visualizations**: Confusion matrices, ROC curves, learning curves, feature importance
- **Bonus regression**: Predict F0 (pitch) from audio features
- **Classroom-ready**: Adjustable parameters for quick demos or deep learning

#### Audio_Regression_MiniSpeechCommands.ipynb
Focused regression tutorial on audio data:
- **54-dimensional feature vectors**: MFCCs, chroma, ZCR, RMS with statistics
- **Multiple regressors**: Linear, Ridge, Random Forest, k-NN comparison
- **Synthetic target**: Loudness proxy (√RMS × 10) for demonstration
- **Complete evaluation**: MAE, MSE, RMSE, R² metrics
- **Professional visualizations**: Predicted vs actual, residual plots, error distributions
- **Export-ready results**: CSV output and styled comparison tables

#### EE502_LDA_QDA_PCA_MiniSpeech.ipynb
Advanced dimensionality reduction and discriminant analysis:
- **PCA**: Principal Component Analysis for feature reduction
- **LDA**: Linear Discriminant Analysis for classification
- **QDA**: Quadratic Discriminant Analysis for non-linear decision boundaries
- **Comparative analysis**: Performance evaluation across techniques
- **Visualization**: Decision boundaries and feature space reduction

### 📚 Book Chapter Implementations (Book_Scripts/)

#### Chapter 4: Audio Classification Using MFCCs and Decision Trees
Fundamental audio classification with spoken digit recognition:
- MFCC feature extraction from audio signals
- Random Forest and SVM classifiers
- Principal Component Analysis (PCA) for dimensionality reduction
- Model evaluation and performance comparison

#### Chapter 5: Key-Word Detection using Random Forest
Advanced keyword detection with sophisticated feature engineering:
- 240+ audio features with intelligent feature selection
- Data augmentation techniques for class balancing
- Ensemble methods for robust performance
- Systematic model improvement pipeline

#### Chapter 6: SVM-based Speaker Classification
Comprehensive speaker identification system:
- Speaker recognition using unique voice characteristics
- Performance scaling analysis with varying dataset sizes
- Detailed confusion matrix analysis
- Cross-validation and overfitting detection strategies

#### Chapter 7: Speech Emotion Recognition using MFCC and Machine Learning
State-of-the-art emotion recognition (85%+ accuracy):
- Multi-dataset support: RAVDESS, CREMA-D, TESS, SAVEE
- Advanced neural architectures: ResNet-1D, Attention mechanisms
- Hyperparameter optimization with Optuna
- Data augmentation and ensemble methods

#### Chapter 8: Voice-based Gender Identification
Gender classification from voice characteristics:
- Voice characteristic analysis and feature extraction
- Gender-specific acoustic features
- Classification model development and tuning
- Performance evaluation and optimization

#### Chapter 9: Deepfake Audio Detection via MFCC Features
Modern audio authentication and deepfake detection:
- Real vs. AI-generated speech detection
- MFCC-based deepfake analysis
- Machine learning for audio authentication
- Practical security applications

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
- **Research team** - Speaker verification project and paper submission

### Datasets and Libraries
- **Mini Speech Commands** - Google's open-access speech dataset for teaching
- **LibriSpeech** - Large-scale corpus for speaker verification research
- **RAVDESS, CREMA-D, TESS, SAVEE** - Emotion recognition datasets
- **Free Spoken Digit Dataset** - Digit classification data
- **LibROSA** - Audio processing library
- **scikit-learn** - Machine learning framework
- **TensorFlow/PyTorch** - Deep learning platforms
- **hmmlearn** - Hidden Markov Model implementation

### Special Thanks
- Open-source community for audio processing tools
- Students and researchers who contributed feedback and improvements
- Academic institutions supporting audio ML research

---

## 📈 Repository Statistics

![Repository](https://img.shields.io/badge/Repository-EE502-blue)
![Book Chapters](https://img.shields.io/badge/Book%20Chapters-6-green)
![Teaching Notebooks](https://img.shields.io/badge/Teaching%20Notebooks-3-brightgreen)
![Research Projects](https://img.shields.io/badge/Research%20Projects-1-purple)
![Technologies](https://img.shields.io/badge/Technologies-14+-orange)
![Performance](https://img.shields.io/badge/Best%20Accuracy-91.23%25-red)

---

## 🎯 Quick Links

- 📖 **Textbook**: [EE502_Draft_Booklet_Oct2nd_version.pdf](./EE502_Draft_Booklet_Oct2nd_version.pdf)
- 🎓 **Start Learning**: [Audio_DT_RF_MiniSpeechCommands.ipynb](./class_examples/Audio_DT_RF_MiniSpeechCommands%20(1).ipynb)
- 📚 **Book Chapters**: [Book_Scripts/](./Book_Scripts/)
- 🔬 **Research**: [Speaker Verification Project](./text_independant_Spkr_vrf_old_chapter5/Text_Independent_Speaker_Verification_using_HMM_SVM/)

**🎉 Ready to start learning?** Begin with the [interactive teaching notebooks](./class_examples/) or dive into [complete chapter implementations](./Book_Scripts/)!


**⭐ Star this repository** if you find it helpful for your audio signal processing and machine learning journey!

