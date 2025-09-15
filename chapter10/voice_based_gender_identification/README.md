# Voice-Based Gender Identification

A machine learning project that classifies gender based on voice acoustic features using multiple algorithms including Random Forest, SVM, Neural Networks, and XGBoost.

## 📖 Overview

This project implements voice-based gender identification using acoustic features extracted from voice recordings. The system analyzes 20 different voice characteristics to determine if a speaker is male or female with high accuracy.

## 🎯 Features

- **Multiple ML Algorithms**: Implements 6 different classification approaches
- **High Accuracy**: Achieves up to 98.58% accuracy with XGBoost
- **Feature Analysis**: Comprehensive analysis of voice acoustic properties
- **Neural Network**: Deep learning implementation using TensorFlow
- **Data Visualization**: Correlation analysis and distribution plots
- **Real-time Prediction**: Test on new voice samples

## 📁 Project Structure

```
voice_based_gender_identification/
├── main.ipynb              # Main Jupyter notebook with all implementations
├── voice_data.csv          # Training dataset (3,168 samples)
├── male_voice.csv          # Male voice sample for testing (extracted from brian.wav)
├── female_voice.csv        # Female voice sample for testing (extracted from amy.wav)
├── amy.wav                 # Female voice audio sample (reference)
├── brian.wav               # Male voice audio sample (reference) 
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔬 Dataset

The dataset contains **3,168 voice samples** (1,584 male, 1,584 female) with 20 acoustic features:

### Feature Description

| Feature | Description |
|---------|-------------|
| `meanfreq` | Mean frequency (kHz) |
| `sd` | Standard deviation of frequency |
| `median` | Median frequency (kHz) |
| `Q25` | First quantile (kHz) |
| `Q75` | Third quantile (kHz) |
| `IQR` | Interquantile range (kHz) |
| `skew` | Skewness (asymmetry of distribution) |
| `kurt` | Kurtosis (tail heaviness) |
| `sp.ent` | Spectral entropy |
| `sfm` | Spectral flatness |
| `mode` | Mode frequency |
| `centroid` | Frequency centroid |
| `meanfun` | Mean fundamental frequency |
| `minfun` | Minimum fundamental frequency |
| `maxfun` | Maximum fundamental frequency |
| `meandom` | Mean of dominant frequency |
| `mindom` | Minimum of dominant frequency |
| `maxdom` | Maximum of dominant frequency |
| `dfrange` | Range of dominant frequency |
| `modindx` | Modulation index |

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required packages (see Installation)

### Hardware Requirements

- **Minimum**: 2GB RAM, CPU-based training
- **Recommended**: 4GB+ RAM for faster processing
- **GPU**: Optional but recommended for TensorFlow neural network training

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd voice_based_gender_identification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow xgboost mglearn
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook main.ipynb
```

4. Run all cells in sequence or execute the entire notebook:
```bash
jupyter nbconvert --to python --execute main.ipynb
```

## 🤖 Machine Learning Models

The project implements and compares 6 different classification algorithms:

### 1. Random Forest Classifier
- **Train Accuracy**: 99.96%
- **Test Accuracy**: 98.11%
- **Best performing traditional ML model**

### 2. Support Vector Machine (RBF)
- **Train Accuracy**: 72.30%
- **Test Accuracy**: 71.45%

### 3. K-Nearest Neighbors
- **Train Accuracy**: 86.50%
- **Test Accuracy**: 75.39%

### 4. Support Vector Machine (Linear)
- **Train Accuracy**: 91.28%
- **Test Accuracy**: 93.85%

### 5. Naive Bayes
- **Train Accuracy**: 93.21%
- **Test Accuracy**: 94.32%

### 6. Decision Tree
- **Train Accuracy**: 100.00%
- **Test Accuracy**: 96.21%

### 7. XGBoost Classifier
- **Train Accuracy**: 100.00%
- **Test Accuracy**: 98.58%
- **Highest test accuracy**

### 8. Neural Network (TensorFlow)
- **Architecture**: 32→64→32→16→1 layers with dropout
- **Test Accuracy**: ~96.37%
- **Optimizer**: Adam
- **Loss**: Binary crossentropy

## 📊 Performance Analysis

### Model Comparison
| Model | Train Acc | Test Acc | Overfitting Risk |
|-------|-----------|----------|------------------|
| XGBoost | 100.00% | 98.58% | Low |
| Random Forest | 99.96% | 98.11% | Low |
| Decision Tree | 100.00% | 96.21% | Medium |
| Neural Network | ~97.98% | 96.37% | Low |
| Naive Bayes | 93.21% | 94.32% | None |
| SVM (Linear) | 91.28% | 93.85% | None |

### Feature Engineering
- **Correlation Analysis**: Identifies highly correlated features
- **Feature Selection**: Removes redundant features (`sfm`, `kurt`, `meandom`, `meanfreq`, `dfrange`, `modindx`)
- **Dimensionality Reduction**: From 20 to 14 features

## 🎵 Usage Examples

### Basic Prediction
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess data
data = pd.read_csv('voice_data.csv')
X = data.drop('label', axis=1)
y = data['label'].map({'male': 1, 'female': 0})

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Predict on new sample
new_sample = pd.read_csv('male_voice.csv')[X.columns]
prediction = rf.predict(new_sample)[0]
gender = 'Male' if prediction == 1 else 'Female'
print(f"Predicted gender: {gender}")
```

### Neural Network Training
```python
import tensorflow as tf

# Normalize features
X_norm = (X - X.min()) / (X.max() - X.min())

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_norm, y, epochs=50, validation_split=0.2)
```

## 📈 Visualizations

The notebook includes comprehensive visualizations:

- **Correlation Heatmap**: Feature relationships
- **KDE Plots**: Distribution differences between genders
- **Pair Plots**: Feature scatter plots by gender
- **Training History**: Neural network learning curves

## 🔍 Key Insights

1. **Gender Differences**: Clear acoustic distinctions between male and female voices
2. **Feature Importance**: Fundamental frequency features are most discriminative
3. **Model Performance**: Ensemble methods (Random Forest, XGBoost) perform best
4. **Overfitting**: Decision Tree shows perfect training but lower test accuracy
5. **Generalization**: Neural networks provide good balance between complexity and performance

## 🛠️ Technical Details

### Environment Specifications
- **Python**: 3.9.7
- **NumPy**: 1.20.3
- **Pandas**: 1.3.4
- **Scikit-learn**: 1.0.2
- **TensorFlow**: 2.16.1+
- **XGBoost**: 3.0.3

### Data Preprocessing
1. **Missing Values**: No missing values in dataset
2. **Label Encoding**: Male=1, Female=0
3. **Feature Scaling**: MinMax normalization for neural networks
4. **Train-Test Split**: 80-20 split with stratification

## 🎯 Results Summary

- **Best Model**: XGBoost with 98.58% test accuracy
- **Fastest Training**: Naive Bayes
- **Most Interpretable**: Decision Tree
- **Most Robust**: Random Forest
- **Deep Learning**: Neural Network with competitive performance

## 🐛 Troubleshooting

### Common Issues

1. **sklearn package error**: If you get "sklearn is deprecated", use:
   ```bash
   pip install scikit-learn
   ```

2. **TensorFlow GPU warnings**: These are normal if you don't have a GPU setup. The code will run on CPU.

3. **Memory issues**: If you run out of memory, try reducing the dataset size in the notebook or close other applications.

4. **Jupyter notebook won't start**: Make sure you have Jupyter installed:
   ```bash
   pip install jupyter
   ```

## 🚀 Future Improvements

1. **Audio Processing**: Direct audio file processing
2. **Real-time Classification**: Live voice classification
3. **Feature Engineering**: Advanced audio features (MFCC, spectrograms)
4. **Model Ensemble**: Combining multiple models
5. **Cross-validation**: K-fold validation for robust evaluation

## 📝 License

This project is for educational purposes. Please ensure proper attribution when using the code.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

## 📞 Contact

For questions or collaborations, please open an issue in the repository.

---

**Note**: This project demonstrates machine learning classification techniques for educational purposes. The acoustic features used are derived from voice analysis research in gender identification.