#!/usr/bin/env python3
"""
Data Augmentation for Keyword Detection
Implements audio data augmentation to balance classes and increase dataset size
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

def augment_audio_time_stretch(audio, sr, stretch_factors=[0.9, 1.1]):
    """Apply time stretching to audio"""
    augmented = []
    for factor in stretch_factors:
        try:
            stretched = librosa.effects.time_stretch(audio, rate=factor)
            # Ensure same length by padding or truncating
            if len(stretched) > len(audio):
                stretched = stretched[:len(audio)]
            else:
                stretched = np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')
            augmented.append(stretched)
        except Exception as e:
            print(f"Time stretch error: {e}")
            augmented.append(audio)  # Fallback to original
    return augmented

def augment_audio_pitch_shift(audio, sr, n_steps=[-2, -1, 1, 2]):
    """Apply pitch shifting to audio"""
    augmented = []
    for step in n_steps:
        try:
            pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=step)
            augmented.append(pitched)
        except Exception as e:
            print(f"Pitch shift error: {e}")
            augmented.append(audio)  # Fallback to original
    return augmented

def augment_audio_noise(audio, noise_factors=[0.005, 0.01]):
    """Add white noise to audio"""
    augmented = []
    for factor in noise_factors:
        noise = np.random.normal(0, factor * np.std(audio), len(audio))
        noisy = audio + noise
        # Normalize to prevent clipping
        noisy = noisy / np.max(np.abs(noisy)) * np.max(np.abs(audio))
        augmented.append(noisy)
    return augmented

def augment_audio_volume(audio, volume_factors=[0.8, 1.2]):
    """Apply volume scaling to audio"""
    augmented = []
    for factor in volume_factors:
        scaled = audio * factor
        # Prevent clipping
        if np.max(np.abs(scaled)) > 1.0:
            scaled = scaled / np.max(np.abs(scaled)) * 0.95
        augmented.append(scaled)
    return augmented

def augment_audio_time_shift(audio, shift_ratios=[0.1, -0.1]):
    """Apply circular time shifts to audio"""
    augmented = []
    for ratio in shift_ratios:
        shift_samples = int(ratio * len(audio))
        shifted = np.roll(audio, shift_samples)
        augmented.append(shifted)
    return augmented

def create_augmented_dataset(original_dataset_path, augmented_dataset_path, target_samples_per_class=40):
    """
    Create augmented dataset to balance classes
    
    Args:
        original_dataset_path: Path to original dataset
        augmented_dataset_path: Path to save augmented dataset
        target_samples_per_class: Target number of samples per class
    """
    
    print(f"Creating augmented dataset: {original_dataset_path} -> {augmented_dataset_path}")
    
    # Create augmented dataset directories
    os.makedirs(augmented_dataset_path, exist_ok=True)
    os.makedirs(os.path.join(augmented_dataset_path, 'Train'), exist_ok=True)
    os.makedirs(os.path.join(augmented_dataset_path, 'Test'), exist_ok=True)
    
    # Load original training data
    train_df = pd.read_csv(os.path.join(original_dataset_path, 'train.csv'))
    class_counts = Counter(train_df['keyword'])
    
    print("Original class distribution:")
    for keyword, count in class_counts.items():
        print(f"  {keyword}: {count} samples")
    
    # Initialize augmented data tracking
    augmented_data = []
    
    # Process each class
    for keyword in class_counts.keys():
        class_files = train_df[train_df['keyword'] == keyword]['new_id'].values
        current_count = len(class_files)
        needed_samples = max(0, target_samples_per_class - current_count)
        
        print(f"\\nProcessing '{keyword}' class:")
        print(f"  Current: {current_count}, Target: {target_samples_per_class}, Need: {needed_samples}")
        
        # Copy original files
        for i, filename in enumerate(class_files):
            original_path = os.path.join(original_dataset_path, 'Train', filename)
            new_path = os.path.join(augmented_dataset_path, 'Train', filename)
            
            if os.path.exists(original_path):
                # Load and save original
                audio, sr = librosa.load(original_path, sr=22050, mono=True)
                sf.write(new_path, audio, sr)
                augmented_data.append({'new_id': filename, 'keyword': keyword})
                
                # Generate augmented samples if needed
                if i < needed_samples:  # Only augment files we need
                    try:
                        # Choose augmentation technique based on sample index
                        augmentation_type = i % 5  # Cycle through 5 techniques
                        
                        if augmentation_type == 0:
                            # Time stretching
                            augmented_audios = augment_audio_time_stretch(audio, sr, [0.95])
                            suffix = "stretch"
                        elif augmentation_type == 1:
                            # Pitch shifting  
                            augmented_audios = augment_audio_pitch_shift(audio, sr, [1])
                            suffix = "pitch"
                        elif augmentation_type == 2:
                            # Noise addition
                            augmented_audios = augment_audio_noise(audio, [0.005])
                            suffix = "noise"
                        elif augmentation_type == 3:
                            # Volume scaling
                            augmented_audios = augment_audio_volume(audio, [1.15])
                            suffix = "volume"
                        else:
                            # Time shifting
                            augmented_audios = augment_audio_time_shift(audio, [0.1])
                            suffix = "shift"
                        
                        # Save augmented audio
                        for j, aug_audio in enumerate(augmented_audios):
                            aug_filename = f"{filename.split('.')[0]}_{suffix}_{j}.wav"
                            aug_path = os.path.join(augmented_dataset_path, 'Train', aug_filename)
                            sf.write(aug_path, aug_audio, sr)
                            augmented_data.append({'new_id': aug_filename, 'keyword': keyword})
                            
                            print(f"    Created: {aug_filename}")
                            
                            # Stop if we have enough samples
                            if len([d for d in augmented_data if d['keyword'] == keyword]) >= target_samples_per_class:
                                break
                    
                    except Exception as e:
                        print(f"    Error augmenting {filename}: {e}")
                
                # Stop if we have enough samples for this class
                if len([d for d in augmented_data if d['keyword'] == keyword]) >= target_samples_per_class:
                    break
    
    # Copy test files unchanged
    test_df = pd.read_csv(os.path.join(original_dataset_path, 'test_idx.csv'))
    for filename in test_df['new_id'].values:
        original_path = os.path.join(original_dataset_path, 'Test', filename)
        new_path = os.path.join(augmented_dataset_path, 'Test', filename)
        
        if os.path.exists(original_path):
            audio, sr = librosa.load(original_path, sr=22050, mono=True)
            sf.write(new_path, audio, sr)
    
    # Create new CSV files
    augmented_train_df = pd.DataFrame(augmented_data)
    augmented_train_df.to_csv(os.path.join(augmented_dataset_path, 'train.csv'), index=False)
    
    # Copy test CSV
    test_df.to_csv(os.path.join(augmented_dataset_path, 'test_idx.csv'), index=False)
    
    # Print final statistics
    final_counts = Counter(augmented_train_df['keyword'])
    print("\\nAugmented class distribution:")
    total_samples = 0
    for keyword, count in final_counts.items():
        print(f"  {keyword}: {count} samples")
        total_samples += count
    
    print(f"\\nTotal augmented training samples: {total_samples}")
    print(f"Augmented dataset created at: {augmented_dataset_path}")
    
    return augmented_dataset_path

def create_balanced_test_split(dataset_path, train_ratio=0.8):
    """
    Create a more balanced train/test split from the augmented dataset
    
    Args:
        dataset_path: Path to dataset
        train_ratio: Ratio of data to use for training
    """
    
    print(f"Creating balanced train/test split for {dataset_path}")
    
    # Load current data
    train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    
    # Create stratified split
    from sklearn.model_selection import train_test_split
    
    X = train_df['new_id'].values
    y = train_df['keyword'].values
    
    # Split while preserving class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-train_ratio, stratify=y, random_state=42
    )
    
    print(f"New split: {len(X_train)} train, {len(X_test)} test")
    
    # Create new CSV files
    new_train_df = pd.DataFrame({'new_id': X_train, 'keyword': y_train})
    new_test_df = pd.DataFrame({'new_id': X_test, 'keyword': y_test})
    
    # Save new splits
    new_train_df.to_csv(os.path.join(dataset_path, 'train_balanced.csv'), index=False)
    new_test_df.to_csv(os.path.join(dataset_path, 'test_balanced.csv'), index=False)
    
    # Move files to appropriate directories (optional - keep current structure)
    print("Balanced split files created: train_balanced.csv, test_balanced.csv")
    
    return new_train_df, new_test_df

if __name__ == "__main__":
    print("=== DATA AUGMENTATION ===\\n")
    
    # Create augmented version of unfiltered dataset
    original_dataset = "DatasetV2"
    augmented_dataset = "DatasetV2_Augmented"
    
    if os.path.exists(original_dataset):
        # Create augmented dataset with balanced classes
        create_augmented_dataset(
            original_dataset_path=original_dataset,
            augmented_dataset_path=augmented_dataset, 
            target_samples_per_class=35  # Conservative target to avoid overfitting
        )
        
        # Create a more balanced train/test split
        create_balanced_test_split(augmented_dataset, train_ratio=0.75)
        
        print("\\n✅ Data augmentation completed successfully!")
        print(f"New augmented dataset available at: {augmented_dataset}")
        
    else:
        print(f"❌ Original dataset {original_dataset} not found")
        
    print("\\nNext steps:")
    print("1. Extract enhanced features from augmented dataset")
    print("2. Train models with regularization on augmented data") 
    print("3. Use balanced train/test split for evaluation")