# SVM Speaker Classification - Comprehensive Experiment Report

**Generated:** 2025-09-30 12:29:44

---

## Executive Summary

This report documents a comprehensive set of experiments on Support Vector Machine (SVM) based speaker classification using MFCC features extracted from audio recordings. The experiments were conducted at three different scales (5, 10, and 20 speakers) to evaluate model performance and scalability.

### Key Findings

- ✅ **Perfect test accuracy (100%)** achieved on 5-speaker dataset
- ⚠️ **Significant performance degradation** with scaling (60% at 10 speakers, 42.5% at 20 speakers)
- 📊 **Cross-validation reveals overfitting**: Test accuracy (100%) >> CV accuracy (68%)
- 🎯 **Top discriminative feature**: MFCC-3 Kurtosis (Feature Index 18)
- ⚡ **Fast inference**: <0.001s prediction time

---

## 1. Experiment Configuration

### Algorithm Parameters
- **Model**: Support Vector Machine (SVM)
- **Kernel**: RBF (Radial Basis Function)
- **Multi-class Strategy**: One-vs-Rest (OvR)
- **Hyperparameters**: C=1, gamma='scale'

### Feature Extraction
- **Method**: MFCC (Mel-Frequency Cepstral Coefficients)
- **Coefficients**: 3
- **Sample Rate**: 48,000 Hz
- **FFT Window**: 960 samples
- **Hop Length**: 480 samples
- **Preprocessing**: Noise reduction (threshold=0.8)

### Statistical Features (per MFCC)
1. Mean
2. Median
3. Standard Deviation
4. Skewness
5. Kurtosis
6. Maximum
7. Minimum

**Total Features**: 3 MFCCs × 7 statistics = 21 features

---

## 2. Dataset Statistics

### Current Experiment (5 Speakers)
- **Training Samples**: 40 (8 per speaker)
- **Testing Samples**: 10 (2 per speaker)
- **Train/Test Split**: 80/20
- **Total Audio Files**: 50
- **Average File Size**: 282 KB
- **Duration per File**: ~5.9 seconds
- **Total Dataset Size**: ~14.1 MB

### Speaker Distribution
All speakers have balanced representation:
- speaker_0000: 8 train, 2 test
- speaker_0001: 8 train, 2 test
- speaker_0002: 8 train, 2 test
- speaker_0003: 8 train, 2 test
- speaker_0004: 8 train, 2 test

---

## 3. Performance Results

### Test Accuracy
| Experiment | Training | Testing | Test Accuracy | CV Accuracy | Accuracy Drop |
|------------|----------|---------|---------------|-------------|---------------|
| 5 speakers | 40       | 10      | **100.0%**    | 68.0% ± 10% | -             |
| 10 speakers| 80       | 20      | 60.0%         | 26.3% ± 14% | ↓ 40.0%       |
| 20 speakers| 160      | 40      | 42.5%         | 18.8% ± 7%  | ↓ 57.5%       |

### 5-Fold Cross-Validation (5 Speakers)
| Fold | Accuracy |
|------|----------|
| 1    | 75.0%    |
| 2    | 75.0%    |
| 3    | 62.5%    |
| 4    | 50.0%    |
| 5    | 75.0%    |
| **Mean** | **68.0%** |
| **Std**  | **10.0%** |

### Classification Report (5 Speakers)
```
              precision    recall  f1-score   support

speaker_0000       1.00      1.00      1.00         2
speaker_0001       1.00      1.00      1.00         2
speaker_0002       1.00      1.00      1.00         2
speaker_0003       1.00      1.00      1.00         2
speaker_0004       1.00      1.00      1.00         2

    accuracy                           1.00        10
   macro avg       1.00      1.00      1.00        10
weighted avg       1.00      1.00      1.00        10
```

### Performance Metrics
- **Precision**: 1.00 (all classes)
- **Recall**: 1.00 (all classes)
- **F1-Score**: 1.00 (all classes)
- **Support**: Balanced across all classes

---

## 4. Statistical Validation

### ANOVA F-Test (Feature Importance)
**Top 5 Most Discriminative Features:**
1. **Feature 18**: MFCC-3 Kurtosis
2. **Feature 7**: MFCC-2 Mean
3. **Feature 6**: MFCC-1 Minimum
4. **Feature 2**: MFCC-1 Std Dev
5. **Feature 8**: MFCC-2 Median

### Chi-Squared Test
- **χ² statistic**: 40.0
- **p-value**: 0.00078 (highly significant)
- **Interpretation**: Model predictions significantly different from random classification

### T-Test (vs Baseline)
- **Baseline F1**: 0.5 (chance level)
- **p-value**: 0.106
- **Interpretation**: Not statistically significant at α=0.05 level

### Confusion Matrix (5 Speakers)
Perfect diagonal matrix with zero misclassifications:
```
[[2 0 0 0 0]
 [0 2 0 0 0]
 [0 0 2 0 0]
 [0 0 0 2 0]
 [0 0 0 0 2]]
```

---

## 5. Computational Performance

### Processing Time Analysis
| Experiment | Feature Extraction | Training | Prediction | Total |
|------------|-------------------|----------|------------|-------|
| 5 speakers | 2.46s             | ~0.00s   | 0.0002s    | 2.46s |
| 10 speakers| 4.94s             | ~0.00s   | 0.0002s    | 4.94s |
| 20 speakers| 9.71s             | ~0.00s   | 0.0004s    | 9.71s |

**Observations:**
- Feature extraction is the computational bottleneck
- Linear scaling with dataset size (~0.06s per audio file)
- Training time negligible for small-scale experiments
- Real-time prediction capability (<1ms)

---

## 6. Analysis and Insights

### Strengths
✅ **Perfect test performance** on 5-speaker task
✅ **Fast inference** suitable for real-time applications
✅ **Well-balanced** dataset with equal class representation
✅ **Robust feature extraction** using established MFCC method
✅ **Comprehensive validation** with multiple statistical tests

### Concerns
⚠️ **Significant overfitting**: 100% test vs 68% cross-validation accuracy
⚠️ **Poor scalability**: 40% accuracy drop when doubling speakers
⚠️ **Small dataset**: Only 2 test samples per speaker
⚠️ **Limited generalization**: High CV standard deviation (10%)

### Comparison with PDF Study
| Aspect | PDF (10 speakers) | Current (5 speakers) |
|--------|-------------------|----------------------|
| Dataset Size | 4,779 files | 50 files |
| Test Accuracy | 97% | 100% |
| CV Accuracy | 92% | 68% |
| Speakers | 10 | 5 |
| Interpretation | Production-ready | Proof-of-concept |

---

## 7. Recommendations

### To Improve Generalization
1. **Increase dataset size**: Minimum 50-100 samples per speaker
2. **Data augmentation**: Pitch shifting, time stretching, noise injection
3. **More test samples**: Current 2 samples per speaker insufficient
4. **Regularization**: Tune C parameter to reduce overfitting

### To Improve Scalability
1. **More MFCC coefficients**: Try n_mfcc=13 or 20
2. **Feature selection**: Use top ANOVA-ranked features only
3. **Ensemble methods**: Combine multiple SVM models
4. **Alternative kernels**: Experiment with polynomial or linear kernels

### For Production Deployment
1. **Collect more diverse data**: Different recording conditions
2. **Speaker enrollment**: Minimum 20-30 samples per new speaker
3. **Confidence thresholding**: Reject low-confidence predictions
4. **Periodic retraining**: Update model with new data

---

## 8. Conclusions

This SVM-based speaker classification system demonstrates:

1. **Proof-of-concept success** for small-scale (5 speaker) classification
2. **Perfect performance** on limited test set, but significant overfitting
3. **Scalability challenges** requiring larger datasets and enhanced features
4. **Solid methodology** aligned with academic literature (PDF chapter)
5. **Need for expansion** to approach production-ready performance

The current implementation serves as an excellent educational example and foundation for more robust systems, but requires substantial dataset expansion and model refinement for real-world deployment.

---

## 9. References

1. VoxForge Open Speech Dataset: http://www.voxforge.org/
2. SVM Speaker Classification Chapter (PDF)
3. GitHub Repository: https://github.com/AVHBAC/SVM_Speaker_Classification

---

## 10. Generated Artifacts

This report includes the following visualizations and data files:

### Visualizations
- `experiment_configuration_table.png`
- `dataset_statistics.png`
- `performance_comparison_comprehensive.png`
- `feature_importance_details.png`
- `statistical_tests_summary.png`
- `experiment_pipeline.png`

### Data Files (CSV)
- `experiment_configuration.csv`
- `dataset_statistics.csv`
- `performance_comparison.csv`
- `cross_validation_details.csv`
- `feature_descriptions.csv`
- `statistical_tests_summary.csv`

---

**Report generated automatically by create_experiment_report.py**
