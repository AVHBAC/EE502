#!/usr/bin/env python3
"""
Comprehensive Experiment Report Generator for SVM Speaker Classification
Generates detailed visualizations and tables for all experiments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import json
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_experiment_configuration_table():
    """Create detailed experiment configuration table"""
    config_data = {
        'Parameter': [
            'Algorithm',
            'Kernel Type',
            'Decision Function',
            'Feature Extraction',
            'MFCC Coefficients',
            'Sample Rate (Hz)',
            'FFT Window (n_fft)',
            'Hop Length',
            'Noise Reduction',
            'Noise Threshold',
            'Statistical Features',
            'Total Features per Sample',
            'Feature Normalization',
            'Cross-Validation Folds',
            'Train/Test Split'
        ],
        'Value': [
            'Support Vector Machine (SVM)',
            'RBF (Radial Basis Function)',
            'One-vs-Rest (OvR)',
            'MFCC (Mel-Frequency Cepstral Coefficients)',
            '3',
            '48000',
            '960',
            '480',
            'Yes (noisereduce library)',
            '0.8 (prop_decrease)',
            'Mean, Median, Std, Skew, Kurtosis, Max, Min',
            '21 (3 MFCCs × 7 statistics)',
            'StandardScaler (zero mean, unit variance)',
            '5-Fold',
            '80% / 20%'
        ],
        'Description': [
            'Supervised learning for classification',
            'Handles non-linear relationships effectively',
            'Multi-class classification strategy',
            'Captures spectral properties of speech',
            'Number of cepstral coefficients extracted',
            'Audio sampling frequency',
            'FFT window size for MFCC calculation',
            'Step size between MFCC windows',
            'Reduces background noise in audio',
            'Proportion of noise to reduce',
            'Global statistics computed per audio file',
            'Feature vector dimensionality',
            'Preprocessing for SVM optimization',
            'Cross-validation strategy',
            'Dataset split ratio'
        ]
    }

    df_config = pd.DataFrame(config_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df_config.values,
                    colLabels=df_config.columns,
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.25, 0.35, 0.4])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(df_config.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(df_config) + 1):
        for j in range(len(df_config.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')

    plt.title('Experiment Configuration Parameters', fontsize=16, weight='bold', pad=20)
    plt.savefig('experiment_configuration_table.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Also save as CSV
    df_config.to_csv('experiment_configuration.csv', index=False)
    print("✓ Created experiment configuration table")
    return df_config


def create_dataset_statistics_table():
    """Create comprehensive dataset statistics"""

    # Current experiment data
    current_data = {
        'Speaker ID': ['speaker_0000', 'speaker_0001', 'speaker_0002',
                       'speaker_0003', 'speaker_0004', 'TOTAL'],
        'Training Files': [8, 8, 8, 8, 8, 40],
        'Testing Files': [2, 2, 2, 2, 2, 10],
        'Total Files': [10, 10, 10, 10, 10, 50],
        'Train/Test Split': ['80/20', '80/20', '80/20', '80/20', '80/20', '80/20'],
        'Avg File Size (KB)': [282, 282, 282, 282, 282, 282],
        'Duration per File (s)': ['~5.9', '~5.9', '~5.9', '~5.9', '~5.9', '~5.9']
    }

    df_dataset = pd.DataFrame(current_data)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Table
    ax1.axis('tight')
    ax1.axis('off')

    table = ax1.table(cellText=df_dataset.values,
                     colLabels=df_dataset.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.13, 0.12, 0.11, 0.14, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Style header
    for i in range(len(df_dataset.columns)):
        table[(0, i)].set_facecolor('#2E75B6')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight total row
    for j in range(len(df_dataset.columns)):
        table[(len(df_dataset), j)].set_facecolor('#FFC000')
        table[(len(df_dataset), j)].set_text_props(weight='bold')

    # Alternate other rows
    for i in range(1, len(df_dataset)):
        for j in range(len(df_dataset.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#DEEAF6')

    ax1.set_title('Dataset Statistics - Current Experiment', fontsize=14, weight='bold', pad=10)

    # Bar chart
    speakers = df_dataset['Speaker ID'][:-1]
    train_counts = df_dataset['Training Files'][:-1]
    test_counts = df_dataset['Testing Files'][:-1]

    x = np.arange(len(speakers))
    width = 0.35

    bars1 = ax2.bar(x - width/2, train_counts, width, label='Training Files', color='#4472C4')
    bars2 = ax2.bar(x + width/2, test_counts, width, label='Testing Files', color='#ED7D31')

    ax2.set_xlabel('Speaker ID', fontsize=12, weight='bold')
    ax2.set_ylabel('Number of Files', fontsize=12, weight='bold')
    ax2.set_title('Train/Test Distribution per Speaker', fontsize=14, weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(speakers, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('dataset_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save CSV
    df_dataset.to_csv('dataset_statistics.csv', index=False)
    print("✓ Created dataset statistics visualization")
    return df_dataset


def create_performance_comparison_table():
    """Create detailed performance comparison across experiments"""

    # Data from svm_performance_report.txt and current run
    performance_data = {
        'Experiment': ['5 Speakers', '10 Speakers', '20 Speakers'],
        'Training Samples': [40, 80, 160],
        'Testing Samples': [10, 20, 40],
        'Test Accuracy': [1.000, 0.600, 0.425],
        'CV Accuracy (Mean)': [0.725, 0.263, 0.188],
        'CV Std Dev': [0.094, 0.145, 0.068],
        'Training Time (s)': [0.00, 0.00, 0.00],
        'Feature Extraction Time (s)': [2.46, 4.94, 9.71],
        'Prediction Time (s)': [0.0002, 0.0002, 0.0004]
    }

    # Add detailed 5-speaker cross-validation scores
    cv_5_speakers = {
        'Metric': ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Mean', 'Std Dev'],
        'Accuracy': [0.75, 0.75, 0.625, 0.50, 0.75, 0.68, 0.10]
    }

    df_perf = pd.DataFrame(performance_data)
    df_cv = pd.DataFrame(cv_5_speakers)

    # Create multi-panel figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Main Performance Table
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('tight')
    ax1.axis('off')

    table1 = ax1.table(cellText=df_perf.values,
                      colLabels=df_perf.columns,
                      cellLoc='center',
                      loc='center')

    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1, 2.5)

    for i in range(len(df_perf.columns)):
        table1[(0, i)].set_facecolor('#70AD47')
        table1[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(df_perf) + 1):
        for j in range(len(df_perf.columns)):
            if i % 2 == 0:
                table1[(i, j)].set_facecolor('#E2EFD9')

    ax1.set_title('Performance Comparison Across Different Scale Experiments',
                  fontsize=14, weight='bold', pad=15)

    # 2. Accuracy Comparison Chart
    ax2 = fig.add_subplot(gs[1, 0])

    experiments = df_perf['Experiment']
    test_acc = df_perf['Test Accuracy']
    cv_acc = df_perf['CV Accuracy (Mean)']

    x = np.arange(len(experiments))
    width = 0.35

    bars1 = ax2.bar(x - width/2, test_acc, width, label='Test Accuracy', color='#4472C4')
    bars2 = ax2.bar(x + width/2, cv_acc, width, label='CV Accuracy', color='#ED7D31')

    ax2.set_xlabel('Experiment Scale', fontsize=11, weight='bold')
    ax2.set_ylabel('Accuracy', fontsize=11, weight='bold')
    ax2.set_title('Test vs Cross-Validation Accuracy', fontsize=12, weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(experiments)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1.1])

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)

    # 3. Processing Time Comparison
    ax3 = fig.add_subplot(gs[1, 1])

    processing_times = df_perf['Feature Extraction Time (s)']

    bars = ax3.bar(experiments, processing_times, color=['#70AD47', '#FFC000', '#C5504E'])
    ax3.set_xlabel('Experiment Scale', fontsize=11, weight='bold')
    ax3.set_ylabel('Time (seconds)', fontsize=11, weight='bold')
    ax3.set_title('Feature Extraction Time Scaling', fontsize=12, weight='bold')
    ax3.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=9)

    # 4. Cross-Validation Details Table (5 speakers)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('tight')
    ax4.axis('off')

    table2 = ax4.table(cellText=df_cv.values,
                      colLabels=df_cv.columns,
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.3, 0.3])

    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2.5)

    table2[(0, 0)].set_facecolor('#5B9BD5')
    table2[(0, 1)].set_facecolor('#5B9BD5')
    table2[(0, 0)].set_text_props(weight='bold', color='white')
    table2[(0, 1)].set_text_props(weight='bold', color='white')

    # Highlight summary rows
    for j in range(len(df_cv.columns)):
        table2[(len(df_cv)-1, j)].set_facecolor('#FFC000')
        table2[(len(df_cv), j)].set_facecolor('#FFC000')
        table2[(len(df_cv)-1, j)].set_text_props(weight='bold')
        table2[(len(df_cv), j)].set_text_props(weight='bold')

    ax4.set_title('5-Fold Cross-Validation Details (5 Speakers)',
                  fontsize=12, weight='bold', pad=10)

    # 5. Accuracy Drop Analysis
    ax5 = fig.add_subplot(gs[2, 1])

    acc_drop_5_to_10 = (1.0 - 0.6) * 100
    acc_drop_5_to_20 = (1.0 - 0.425) * 100
    acc_drop_10_to_20 = (0.6 - 0.425) * 100

    transitions = ['5→10\nSpeakers', '5→20\nSpeakers', '10→20\nSpeakers']
    drops = [acc_drop_5_to_10, acc_drop_5_to_20, acc_drop_10_to_20]

    colors_drop = ['#ED7D31', '#C5504E', '#FFC000']
    bars = ax5.bar(transitions, drops, color=colors_drop)

    ax5.set_ylabel('Accuracy Drop (%)', fontsize=11, weight='bold')
    ax5.set_title('Test Accuracy Degradation with Scale', fontsize=12, weight='bold')
    ax5.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, weight='bold')

    plt.savefig('performance_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save CSVs
    df_perf.to_csv('performance_comparison.csv', index=False)
    df_cv.to_csv('cross_validation_details.csv', index=False)

    print("✓ Created comprehensive performance comparison visualization")
    return df_perf, df_cv


def create_feature_importance_table():
    """Create MFCC feature importance visualization"""

    # MFCC feature descriptions
    features_data = {
        'Index': list(range(21)),
        'MFCC Coefficient': [1]*7 + [2]*7 + [3]*7,
        'Statistic': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis', 'Maximum', 'Minimum'] * 3,
        'Description': [
            'Average of MFCC-1 over time',
            'Median value of MFCC-1',
            'Variability of MFCC-1',
            'Asymmetry of MFCC-1 distribution',
            'Tailedness of MFCC-1 distribution',
            'Peak value of MFCC-1',
            'Minimum value of MFCC-1',

            'Average of MFCC-2 over time',
            'Median value of MFCC-2',
            'Variability of MFCC-2',
            'Asymmetry of MFCC-2 distribution',
            'Tailedness of MFCC-2 distribution',
            'Peak value of MFCC-2',
            'Minimum value of MFCC-2',

            'Average of MFCC-3 over time',
            'Median value of MFCC-3',
            'Variability of MFCC-3',
            'Asymmetry of MFCC-3 distribution',
            'Tailedness of MFCC-3 distribution',
            'Peak value of MFCC-3',
            'Minimum value of MFCC-3'
        ],
        'Feature Type': ['Temporal'] * 21
    }

    df_features = pd.DataFrame(features_data)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))

    # Full feature table
    ax1.axis('tight')
    ax1.axis('off')

    # Split into two tables for better readability
    df_part1 = df_features.iloc[:11]
    df_part2 = df_features.iloc[11:]

    table1 = ax1.table(cellText=df_part1.values,
                      colLabels=df_part1.columns,
                      cellLoc='left',
                      loc='upper center',
                      colWidths=[0.08, 0.15, 0.12, 0.5, 0.15])

    table1.auto_set_font_size(False)
    table1.set_fontsize(8)
    table1.scale(1, 2)

    for i in range(len(df_part1.columns)):
        table1[(0, i)].set_facecolor('#9C27B0')
        table1[(0, i)].set_text_props(weight='bold', color='white')

    # Color by MFCC coefficient
    for i in range(1, len(df_part1) + 1):
        mfcc_coef = df_part1.iloc[i-1]['MFCC Coefficient']
        if mfcc_coef == 1:
            color = '#E1BEE7'
        elif mfcc_coef == 2:
            color = '#F3E5F5'
        else:
            color = '#FFFFFF'

        for j in range(len(df_part1.columns)):
            table1[(i, j)].set_facecolor(color)

    ax1.set_title('MFCC Feature Extraction Details (Part 1: Features 0-10)',
                  fontsize=13, weight='bold', pad=10)

    # Second table
    ax2.axis('tight')
    ax2.axis('off')

    table2 = ax2.table(cellText=df_part2.values,
                      colLabels=df_part2.columns,
                      cellLoc='left',
                      loc='upper center',
                      colWidths=[0.08, 0.15, 0.12, 0.5, 0.15])

    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1, 2)

    for i in range(len(df_part2.columns)):
        table2[(0, i)].set_facecolor('#9C27B0')
        table2[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(df_part2) + 1):
        mfcc_coef = df_part2.iloc[i-1]['MFCC Coefficient']
        if mfcc_coef == 2:
            color = '#F3E5F5'
        elif mfcc_coef == 3:
            color = '#D1C4E9'
        else:
            color = '#FFFFFF'

        for j in range(len(df_part2.columns)):
            table2[(i, j)].set_facecolor(color)

    ax2.set_title('MFCC Feature Extraction Details (Part 2: Features 11-20)',
                  fontsize=13, weight='bold', pad=10)

    plt.tight_layout()
    plt.savefig('feature_importance_details.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save CSV
    df_features.to_csv('feature_descriptions.csv', index=False)

    print("✓ Created feature importance details visualization")
    return df_features


def create_statistical_tests_summary():
    """Create summary of all statistical tests performed"""

    stats_data = {
        'Test Name': [
            'ANOVA F-Test',
            'Chi-Squared Test',
            'T-Test (vs baseline)',
            'K-Fold Cross-Validation',
            'Confusion Matrix Analysis'
        ],
        'Purpose': [
            'Feature importance ranking',
            'Model vs random classification',
            'Performance vs chance (0.5)',
            'Model generalization assessment',
            'Per-class performance analysis'
        ],
        'Result (5 Speakers)': [
            'Top features: 18, 7, 6, 2, 8',
            'χ² = 40.0, p = 0.00078',
            'p = 0.106 (not significant)',
            'Mean: 68%, Std: 10%',
            '100% accuracy, perfect diagonal'
        ],
        'Interpretation': [
            'MFCC-3 Kurtosis most discriminative',
            'Highly significant, non-random',
            'Better than baseline but inconclusive',
            'Moderate generalization, some overfitting',
            'Perfect test performance, all classes'
        ],
        'Statistical Significance': [
            'High (low p-values expected)',
            'Yes (p < 0.001)',
            'No (p > 0.05)',
            'N/A',
            'N/A'
        ]
    }

    df_stats = pd.DataFrame(stats_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df_stats.values,
                    colLabels=df_stats.columns,
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.18, 0.22, 0.22, 0.25, 0.13])

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 3)

    # Style header
    for i in range(len(df_stats.columns)):
        table[(0, i)].set_facecolor('#D32F2F')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(df_stats) + 1):
        for j in range(len(df_stats.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#FFCDD2')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')

    plt.title('Statistical Tests Summary and Results', fontsize=16, weight='bold', pad=20)
    plt.savefig('statistical_tests_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save CSV
    df_stats.to_csv('statistical_tests_summary.csv', index=False)

    print("✓ Created statistical tests summary")
    return df_stats


def create_experiment_timeline():
    """Create experiment workflow and timeline"""

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Define workflow steps
    steps = [
        {'name': '1. Data Collection', 'y': 11, 'color': '#E3F2FD'},
        {'name': '2. Preprocessing & Denoising', 'y': 9.5, 'color': '#BBDEFB'},
        {'name': '3. MFCC Feature Extraction', 'y': 8, 'color': '#90CAF9'},
        {'name': '4. Statistical Feature Computation', 'y': 6.5, 'color': '#64B5F6'},
        {'name': '5. Feature Normalization', 'y': 5, 'color': '#42A5F5'},
        {'name': '6. SVM Model Training', 'y': 3.5, 'color': '#2196F3'},
        {'name': '7. Model Evaluation', 'y': 2, 'color': '#1976D2'},
        {'name': '8. Statistical Validation', 'y': 0.5, 'color': '#1565C0'}
    ]

    details = [
        'VoxForge dataset, 5/10/20 speakers\n48kHz sample rate, ~5.9s per file',
        'Noise reduction (threshold=0.8)\nAudio quality preservation',
        '3 MFCC coefficients\nn_fft=960, hop_length=480',
        'Mean, Median, Std, Skew, Kurtosis\nMax, Min → 21 features total',
        'StandardScaler\nZero mean, unit variance',
        'RBF kernel, One-vs-Rest\nC=1, gamma=scale',
        'Test accuracy, Confusion matrix\nClassification report',
        'ANOVA, Chi-squared, t-test\n5-fold cross-validation'
    ]

    for i, (step, detail) in enumerate(zip(steps, details)):
        # Box
        rect = plt.Rectangle((1, step['y']-0.5), 8, 1.2,
                            facecolor=step['color'],
                            edgecolor='#0D47A1',
                            linewidth=2)
        ax.add_patch(rect)

        # Step name
        ax.text(5, step['y']+0.4, step['name'],
               ha='center', va='center',
               fontsize=12, weight='bold')

        # Details
        ax.text(5, step['y'], detail,
               ha='center', va='center',
               fontsize=9, style='italic')

        # Arrow to next step
        if i < len(steps) - 1:
            ax.annotate('', xy=(5, step['y']-0.6), xytext=(5, step['y']-1.1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='#0D47A1'))

    ax.set_title('SVM Speaker Classification Experiment Pipeline',
                fontsize=18, weight='bold', pad=20)

    plt.savefig('experiment_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created experiment pipeline visualization")


def generate_markdown_report():
    """Generate a comprehensive markdown report"""

    report = f"""# SVM Speaker Classification - Comprehensive Experiment Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

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
"""

    with open('EXPERIMENT_REPORT.md', 'w') as f:
        f.write(report)

    print("✓ Generated comprehensive markdown report")


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("SVM SPEAKER CLASSIFICATION - COMPREHENSIVE EXPERIMENT REPORT GENERATOR")
    print("="*70 + "\n")

    print("Generating visualizations and tables...\n")

    # Generate all visualizations and tables
    create_experiment_configuration_table()
    create_dataset_statistics_table()
    create_performance_comparison_table()
    create_feature_importance_table()
    create_statistical_tests_summary()
    create_experiment_timeline()
    generate_markdown_report()

    print("\n" + "="*70)
    print("✅ REPORT GENERATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  📊 Visualizations: 6 PNG files")
    print("  📄 Data tables: 6 CSV files")
    print("  📝 Report: EXPERIMENT_REPORT.md")
    print("\nAll files saved in current directory.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()