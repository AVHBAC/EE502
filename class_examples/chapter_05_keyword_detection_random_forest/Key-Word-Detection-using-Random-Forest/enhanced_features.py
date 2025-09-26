#!/usr/bin/env python3
"""
Enhanced Feature Extraction for Keyword Detection
Implements comprehensive audio feature engineering to improve model performance
"""

import os
import numpy as np
import pandas as pd
import librosa
from scipy.stats import kurtosis, skew
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import warnings
warnings.filterwarnings("ignore")

def extract_comprehensive_features(audio_file_path, sr=22050):
    """
    Extract comprehensive audio features for keyword detection
    
    Args:
        audio_file_path: Path to audio file
        sr: Sample rate
        
    Returns:
        feature_vector: Numpy array of extracted features
        feature_names: List of feature names
    """
    
    # Load audio
    audio, sr = librosa.load(audio_file_path, sr=sr, mono=True)
    
    features = []
    feature_names = []
    
    # === 1. BASIC ENERGY FEATURES (10 features) ===
    # Root Mean Square Energy
    rmse = librosa.feature.rms(y=audio, frame_length=441)[0]
    features.extend([
        np.mean(rmse), np.median(rmse), np.std(rmse), 
        skew(rmse), kurtosis(rmse)
    ])
    feature_names.extend([
        'rmse_mean', 'rmse_median', 'rmse_std', 'rmse_skew', 'rmse_kurtosis'
    ])
    
    # Zero Crossing Rate  
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=441)[0]
    features.extend([
        np.mean(zcr), np.median(zcr), np.std(zcr),
        skew(zcr), kurtosis(zcr)
    ])
    feature_names.extend([
        'zcr_mean', 'zcr_median', 'zcr_std', 'zcr_skew', 'zcr_kurtosis'
    ])
    
    # === 2. MFCC FEATURES (78 features) ===
    # Most important features for speech recognition
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    for i in range(13):
        mfcc_coeff = mfcc[i]
        features.extend([
            np.mean(mfcc_coeff), np.std(mfcc_coeff), 
            np.min(mfcc_coeff), np.max(mfcc_coeff),
            skew(mfcc_coeff), kurtosis(mfcc_coeff)
        ])
        feature_names.extend([
            f'mfcc_{i}_mean', f'mfcc_{i}_std', f'mfcc_{i}_min',
            f'mfcc_{i}_max', f'mfcc_{i}_skew', f'mfcc_{i}_kurtosis'
        ])
    
    # === 3. SPECTRAL FEATURES (30 features) ===
    # Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features.extend([
        np.mean(spectral_centroids), np.std(spectral_centroids),
        np.min(spectral_centroids), np.max(spectral_centroids),
        skew(spectral_centroids), kurtosis(spectral_centroids)
    ])
    feature_names.extend([
        'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_centroid_min',
        'spectral_centroid_max', 'spectral_centroid_skew', 'spectral_centroid_kurtosis'
    ])
    
    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    features.extend([
        np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
        np.min(spectral_bandwidth), np.max(spectral_bandwidth),
        skew(spectral_bandwidth), kurtosis(spectral_bandwidth)
    ])
    feature_names.extend([
        'spectral_bandwidth_mean', 'spectral_bandwidth_std', 'spectral_bandwidth_min',
        'spectral_bandwidth_max', 'spectral_bandwidth_skew', 'spectral_bandwidth_kurtosis'  
    ])
    
    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features.extend([
        np.mean(spectral_rolloff), np.std(spectral_rolloff),
        np.min(spectral_rolloff), np.max(spectral_rolloff),
        skew(spectral_rolloff), kurtosis(spectral_rolloff)
    ])
    feature_names.extend([
        'spectral_rolloff_mean', 'spectral_rolloff_std', 'spectral_rolloff_min',
        'spectral_rolloff_max', 'spectral_rolloff_skew', 'spectral_rolloff_kurtosis'
    ])
    
    # Spectral Contrast (42 features - 7 bands × 6 statistics)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    for i in range(spectral_contrast.shape[0]):
        contrast_band = spectral_contrast[i]
        features.extend([
            np.mean(contrast_band), np.std(contrast_band),
            np.min(contrast_band), np.max(contrast_band),
            skew(contrast_band), kurtosis(contrast_band)
        ])
        feature_names.extend([
            f'spectral_contrast_{i}_mean', f'spectral_contrast_{i}_std',
            f'spectral_contrast_{i}_min', f'spectral_contrast_{i}_max',
            f'spectral_contrast_{i}_skew', f'spectral_contrast_{i}_kurtosis'
        ])
    
    # === 4. HARMONIC FEATURES (72 features) ===
    # Chroma features (pitch class profiles)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    for i in range(chroma.shape[0]):
        chroma_note = chroma[i]
        features.extend([
            np.mean(chroma_note), np.std(chroma_note),
            np.min(chroma_note), np.max(chroma_note),
            skew(chroma_note), kurtosis(chroma_note)
        ])
        feature_names.extend([
            f'chroma_{i}_mean', f'chroma_{i}_std', f'chroma_{i}_min',
            f'chroma_{i}_max', f'chroma_{i}_skew', f'chroma_{i}_kurtosis'
        ])
    
    # === 5. TEMPORAL FEATURES (8 features) ===
    # Tempo and beat tracking
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    features.extend([
        tempo,
        len(beats) / (len(audio) / sr),  # Beat density
        np.mean(np.diff(beats)) if len(beats) > 1 else 0,  # Beat consistency
        np.std(np.diff(beats)) if len(beats) > 1 else 0    # Beat variation
    ])
    feature_names.extend([
        'tempo', 'beat_density', 'beat_consistency', 'beat_variation'
    ])
    
    # Enhanced ZCR features
    zcr_diff = np.diff(zcr)
    features.extend([
        np.mean(zcr_diff) if len(zcr_diff) > 0 else 0,
        np.std(zcr_diff) if len(zcr_diff) > 0 else 0,
        np.max(zcr) - np.min(zcr),  # ZCR range
        len(audio) / sr  # Audio duration
    ])
    feature_names.extend([
        'zcr_diff_mean', 'zcr_diff_std', 'zcr_range', 'duration'
    ])
    
    return np.array(features), feature_names

def extract_dataset_features(dataset_path, csv_file, audio_folder, output_folder, feature_limit=100):
    """
    Extract comprehensive features for entire dataset with feature selection
    
    Args:
        dataset_path: Path to dataset folder
        csv_file: CSV file with audio filenames and labels
        audio_folder: Folder containing audio files
        output_folder: Folder to save extracted features
        feature_limit: Maximum number of features to keep
    """
    
    # Create output directory
    os.makedirs(os.path.join(dataset_path, output_folder), exist_ok=True)
    
    # Load CSV
    df = pd.read_csv(os.path.join(dataset_path, csv_file))
    audio_files = df['new_id'].values
    labels = df['keyword'].values
    
    print(f"Extracting enhanced features for {len(audio_files)} files...")
    
    # Extract features for all files
    all_features = []
    all_labels = []
    feature_names = None
    
    for i, audio_file in enumerate(audio_files):
        audio_path = os.path.join(dataset_path, audio_folder, audio_file)
        
        if os.path.exists(audio_path):
            try:
                features, names = extract_comprehensive_features(audio_path)
                all_features.append(features)
                all_labels.append(labels[i])
                
                if feature_names is None:
                    feature_names = names
                    
                print(f"Processed {i+1}/{len(audio_files)}: {audio_file}")
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
        else:
            print(f"File not found: {audio_path}")
    
    # Convert to arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"Extracted {X.shape[1]} features from {X.shape[0]} samples")
    
    # Feature selection to avoid overfitting
    if X.shape[1] > feature_limit:
        print(f"Selecting top {feature_limit} features using mutual information...")
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Select top features
        selector = SelectKBest(score_func=lambda X, y: mi_scores, k=feature_limit)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_feature_names = [feature_names[i] for i in selected_indices]
        
        print(f"Selected {X_selected.shape[1]} features")
        print("Top 10 features:")
        for i in range(min(10, len(selected_feature_names))):
            score = mi_scores[selected_indices[i]]
            print(f"  {i+1}. {selected_feature_names[i]}: {score:.4f}")
        
        X = X_selected
        feature_names = selected_feature_names
    
    # Save features
    for i, audio_file in enumerate(audio_files):
        if i < len(X):  # Make sure we have features for this file
            output_path = os.path.join(dataset_path, output_folder, f"{audio_file}.npy")
            np.save(output_path, X[i])
    
    # Save feature names
    feature_names_path = os.path.join(dataset_path, output_folder, "feature_names.txt")
    with open(feature_names_path, 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    print(f"Enhanced features saved to {os.path.join(dataset_path, output_folder)}")
    return X, y, feature_names

if __name__ == "__main__":
    print("=== ENHANCED FEATURE EXTRACTION ===\n")
    
    # Extract features for both datasets
    datasets = [
        ("DatasetV2", "Unfiltered"),
        ("DatasetV2_F", "Filtered")
    ]
    
    for dataset_path, dataset_name in datasets:
        if os.path.exists(dataset_path):
            print(f"Processing {dataset_name} dataset ({dataset_path})...")
            
            # Extract training features
            try:
                X_train, y_train, feature_names = extract_dataset_features(
                    dataset_path, 'train.csv', 'Train', 'train_extracted_enhanced', 
                    feature_limit=80  # Limit to 80 features to avoid overfitting
                )
                print(f"Training features: {X_train.shape}")
                
                # Extract test features  
                X_test, y_test, _ = extract_dataset_features(
                    dataset_path, 'test_idx.csv', 'Test', 'test_extracted_enhanced',
                    feature_limit=80
                )
                print(f"Test features: {X_test.shape}")
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
            
            print()
        else:
            print(f"Dataset {dataset_path} not found\n")
    
    print("Enhanced feature extraction completed!")