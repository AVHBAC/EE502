#!/usr/bin/env python3
"""
SVM Speaker Classification Performance Comparison
Compares SVM performance on datasets with 5, 10, and 20 speakers
"""

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from noisereduce import reduce_noise
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import time

def preprocess_and_extract_file_level_features(data_path, sample_rate=48000):
    """
    Preprocess audio files by denoising and extracting global statistics
    for the entire file (e.g., mean, median, skew of MFCCs).
    """
    features = []
    labels = []

    # Iterate through all .wav files in the directory
    for file_name in sorted(os.listdir(data_path)):
        if file_name.endswith('.wav'):
            # Load the audio file
            file_path = os.path.join(data_path, file_name)
            audio, sr = librosa.load(file_path, sr=sample_rate)

            # Step 1: Denoise the audio
            audio_denoised = reduce_noise(y=audio, sr=sr, prop_decrease=0.8)

            # Step 2: Calculate MFCC coefficients for the entire audio sequence
            mfcc_data = librosa.feature.mfcc(y=audio_denoised, sr=sr, n_mfcc=3, n_fft=960, hop_length=480)

            # Step 3: Calculate global statistics for MFCCs
            mean_mfcc = np.mean(mfcc_data, axis=1)
            median_mfcc = np.median(mfcc_data, axis=1)
            std_mfcc = np.std(mfcc_data, axis=1)
            skew_mfcc = skew(mfcc_data, axis=1)
            kurt_mfcc = kurtosis(mfcc_data, axis=1)
            maximum_mfcc = np.amax(mfcc_data, axis=1)
            minimum_mfcc = np.amin(mfcc_data, axis=1)

            # Step 4: Combine all statistics into a single feature vector
            feature_vector = np.concatenate(
                (mean_mfcc, median_mfcc, std_mfcc, skew_mfcc, kurt_mfcc, maximum_mfcc, minimum_mfcc)
            )
            features.append(feature_vector)

            # Step 5: Extract the label (username) from the file name
            parts = file_name.split('_')
            if len(parts) >= 2 and parts[0] == 'speaker':
                label = f"{parts[0]}_{parts[1]}"
            else:
                label = parts[0]
            labels.append(label)

    return np.array(features), np.array(labels)

def evaluate_svm_performance(dataset_path, num_speakers):
    """
    Evaluate SVM performance on a given dataset
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING SVM PERFORMANCE ON {num_speakers}-SPEAKER DATASET")
    print(f"{'='*60}")
    
    # Paths
    train_data_path = os.path.join(dataset_path, 'traindata')
    test_data_path = os.path.join(dataset_path, 'testdata')
    
    # Check if paths exist
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        print(f"Error: Dataset paths do not exist for {num_speakers} speakers")
        return None
    
    # Extract features
    print("Extracting features from training data...")
    start_time = time.time()
    X_train, Y_train = preprocess_and_extract_file_level_features(train_data_path, sample_rate=48000)
    train_extraction_time = time.time() - start_time
    
    print("Extracting features from test data...")
    start_time = time.time()
    X_test, Y_test = preprocess_and_extract_file_level_features(test_data_path, sample_rate=48000)
    test_extraction_time = time.time() - start_time
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    print("Training SVM model...")
    start_time = time.time()
    svm_model = svm.SVC(kernel='rbf', decision_function_shape='ovr', random_state=42)
    svm_model.fit(X_train_scaled, Y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    print("Making predictions...")
    start_time = time.time()
    predictions = svm_model.predict(X_test_scaled)
    prediction_time = time.time() - start_time
    
    # Calculate metrics
    test_accuracy = accuracy_score(Y_test, predictions)
    
    # Cross-validation
    print("Performing cross-validation...")
    cv_scores = cross_val_score(svm_model, X_train_scaled, Y_train, cv=5, scoring='accuracy')
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    # Confusion matrix
    cm = confusion_matrix(Y_test, predictions)
    
    # Classification report
    class_report = classification_report(Y_test, predictions, output_dict=True)
    
    # Count unique speakers
    unique_speakers = len(np.unique(Y_train))
    
    print(f"\nResults for {num_speakers}-Speaker Dataset:")
    print(f"- Unique speakers found: {unique_speakers}")
    print(f"- Training samples: {len(X_train)}")
    print(f"- Test samples: {len(X_test)}")
    print(f"- Test accuracy: {test_accuracy:.3f}")
    print(f"- Cross-validation accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
    print(f"- Feature extraction time: {train_extraction_time + test_extraction_time:.2f}s")
    print(f"- Training time: {training_time:.2f}s")
    print(f"- Prediction time: {prediction_time:.2f}s")
    
    return {
        'num_speakers': num_speakers,
        'unique_speakers': unique_speakers,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'test_accuracy': test_accuracy,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'feature_extraction_time': train_extraction_time + test_extraction_time,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'Y_test': Y_test,
        'predictions': predictions
    }

def plot_performance_comparison(results_list):
    """
    Create comprehensive performance comparison plots
    """
    print("\nCreating performance comparison plots...")
    
    # Extract data for plotting
    speaker_counts = [r['num_speakers'] for r in results_list]
    test_accuracies = [r['test_accuracy'] for r in results_list]
    cv_means = [r['cv_mean'] for r in results_list]
    cv_stds = [r['cv_std'] for r in results_list]
    training_times = [r['training_time'] for r in results_list]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SVM Speaker Classification Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy comparison
    axes[0, 0].bar(speaker_counts, test_accuracies, alpha=0.7, label='Test Accuracy', color='skyblue')
    axes[0, 0].errorbar(speaker_counts, cv_means, yerr=cv_stds, fmt='ro-', 
                       capsize=5, label='CV Accuracy ± Std', markersize=8)
    axes[0, 0].set_xlabel('Number of Speakers')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy vs Number of Speakers')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1.1)
    
    # Add accuracy values on bars
    for i, (speakers, acc) in enumerate(zip(speaker_counts, test_accuracies)):
        axes[0, 0].text(speakers, acc + 0.02, f'{acc:.3f}', ha='center', fontweight='bold')
    
    # Plot 2: Training time comparison
    bars = axes[0, 1].bar(speaker_counts, training_times, alpha=0.7, color='lightcoral')
    axes[0, 1].set_xlabel('Number of Speakers')
    axes[0, 1].set_ylabel('Training Time (seconds)')
    axes[0, 1].set_title('Training Time vs Number of Speakers')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add time values on bars
    for bar, time_val in zip(bars, training_times):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{time_val:.2f}s', ha='center', fontweight='bold')
    
    # Plot 3: Sample size comparison
    train_samples = [r['train_samples'] for r in results_list]
    test_samples = [r['test_samples'] for r in results_list]
    
    x = np.arange(len(speaker_counts))
    width = 0.35
    
    bars1 = axes[1, 0].bar(x - width/2, train_samples, width, label='Training Samples', alpha=0.7, color='lightgreen')
    bars2 = axes[1, 0].bar(x + width/2, test_samples, width, label='Test Samples', alpha=0.7, color='orange')
    
    axes[1, 0].set_xlabel('Number of Speakers')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].set_title('Dataset Sizes')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(speaker_counts)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add sample counts on bars
    for bars, values in [(bars1, train_samples), (bars2, test_samples)]:
        for bar, val in zip(bars, values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           str(val), ha='center', fontweight='bold')
    
    # Plot 4: Difficulty analysis (inverse of accuracy as complexity indicator)
    complexity_indicator = [1/acc if acc > 0 else float('inf') for acc in cv_means]
    
    axes[1, 1].plot(speaker_counts, complexity_indicator, 'bo-', markersize=8, linewidth=2)
    axes[1, 1].set_xlabel('Number of Speakers')
    axes[1, 1].set_ylabel('Classification Complexity (1/CV Accuracy)')
    axes[1, 1].set_title('Classification Difficulty vs Number of Speakers')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add complexity values
    for speakers, complexity in zip(speaker_counts, complexity_indicator):
        if complexity != float('inf'):
            axes[1, 1].text(speakers, complexity + 0.05, f'{complexity:.2f}', 
                           ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('svm_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(results_list):
    """
    Plot confusion matrices for all datasets
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
    
    for i, result in enumerate(results_list):
        cm = result['confusion_matrix']
        num_speakers = result['num_speakers']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                   ax=axes[i], square=True, cbar_kws={'shrink': .8})
        axes[i].set_title(f'{num_speakers} Speakers\n(Test Acc: {result["test_accuracy"]:.3f})')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_comparison_report(results_list):
    """
    Generate a comprehensive comparison report
    """
    print(f"\n{'='*80}")
    print("SVM SPEAKER CLASSIFICATION PERFORMANCE COMPARISON REPORT")
    print(f"{'='*80}")
    
    # Create comparison table
    df_data = []
    for result in results_list:
        df_data.append({
            'Speakers': result['num_speakers'],
            'Train Samples': result['train_samples'],
            'Test Samples': result['test_samples'],
            'Test Accuracy': f"{result['test_accuracy']:.3f}",
            'CV Accuracy': f"{result['cv_mean']:.3f} ± {result['cv_std']:.3f}",
            'Training Time (s)': f"{result['training_time']:.2f}",
            'Feature Extraction Time (s)': f"{result['feature_extraction_time']:.2f}",
            'Prediction Time (s)': f"{result['prediction_time']:.4f}"
        })
    
    df = pd.DataFrame(df_data)
    print("\nPerformance Summary Table:")
    print(df.to_string(index=False))
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS & INSIGHTS")
    print(f"{'='*60}")
    
    # Accuracy trend analysis
    test_accs = [r['test_accuracy'] for r in results_list]
    cv_accs = [r['cv_mean'] for r in results_list]
    
    print("\n1. ACCURACY ANALYSIS:")
    print(f"   • As the number of speakers increases, test accuracy tends to decrease")
    print(f"   • 5 speakers:  Test={test_accs[0]:.3f}, CV={cv_accs[0]:.3f}")
    print(f"   • 10 speakers: Test={test_accs[1]:.3f}, CV={cv_accs[1]:.3f}")
    print(f"   • 20 speakers: Test={test_accs[2]:.3f}, CV={cv_accs[2]:.3f}")
    
    # Calculate performance degradation
    acc_drop_10 = (test_accs[0] - test_accs[1]) / test_accs[0] * 100
    acc_drop_20 = (test_accs[0] - test_accs[2]) / test_accs[0] * 100
    
    print(f"   • Performance drop from 5→10 speakers: {acc_drop_10:.1f}%")
    print(f"   • Performance drop from 5→20 speakers: {acc_drop_20:.1f}%")
    
    print("\n2. COMPUTATIONAL COMPLEXITY:")
    training_times = [r['training_time'] for r in results_list]
    print(f"   • Training time scales with dataset size and complexity")
    print(f"   • 5 speakers:  {training_times[0]:.2f}s")
    print(f"   • 10 speakers: {training_times[1]:.2f}s ({training_times[1]/training_times[0]:.1f}x)")
    print(f"   • 20 speakers: {training_times[2]:.2f}s ({training_times[2]/training_times[0]:.1f}x)")
    
    print("\n3. DATASET CHARACTERISTICS:")
    for result in results_list:
        samples_per_speaker = result['train_samples'] / result['num_speakers']
        print(f"   • {result['num_speakers']} speakers: {samples_per_speaker:.0f} train samples per speaker")
    
    print("\n4. PRACTICAL RECOMMENDATIONS:")
    best_balance_idx = 1  # Usually 10 speakers provides good balance
    print(f"   • For balanced accuracy vs complexity: {results_list[best_balance_idx]['num_speakers']} speakers")
    print(f"   • For highest accuracy: {results_list[0]['num_speakers']} speakers")
    print(f"   • For scalability testing: {results_list[2]['num_speakers']} speakers")
    
    # Save report to file
    with open('svm_performance_report.txt', 'w') as f:
        f.write("SVM SPEAKER CLASSIFICATION PERFORMANCE COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        f.write("Performance Summary:\n")
        f.write(df.to_string(index=False))
        f.write(f"\n\nAccuracy Analysis:\n")
        f.write(f"5→10 speakers accuracy drop: {acc_drop_10:.1f}%\n")
        f.write(f"5→20 speakers accuracy drop: {acc_drop_20:.1f}%\n")
    
    print(f"\n✅ Report saved to 'svm_performance_report.txt'")
    print(f"✅ Plots saved to 'svm_performance_comparison.png' and 'confusion_matrices_comparison.png'")

def main():
    """
    Main function to run the complete performance comparison
    """
    print("Starting SVM Speaker Classification Performance Comparison...")
    
    # Dataset configurations
    dataset_configs = [
        ('datasets/5_speakers', 5),
        ('datasets/10_speakers', 10),
        ('datasets/20_speakers', 20)
    ]
    
    results = []
    
    # Evaluate each dataset
    for dataset_path, num_speakers in dataset_configs:
        result = evaluate_svm_performance(dataset_path, num_speakers)
        if result:
            results.append(result)
    
    if len(results) == 3:
        # Generate visualizations
        plot_performance_comparison(results)
        plot_confusion_matrices(results)
        
        # Generate comprehensive report
        generate_comparison_report(results)
    else:
        print("Error: Not all datasets could be evaluated")

if __name__ == "__main__":
    main()