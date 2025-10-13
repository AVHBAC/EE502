#!/usr/bin/env python3
"""
Extract features from audio files for keyword detection
"""

import os
import numpy as np
import pandas as pd
import librosa
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings("ignore")

def get_training(original_path):
    """Extract training data and save as numpy arrays"""
    df = pd.read_csv(os.path.join(original_path,'train.csv'))
    
    if not os.path.exists(os.path.join(original_path,'train_extracted')):
        os.makedirs(os.path.join(original_path,'train_extracted'))
    
    audio_files = np.array(df['new_id'])
    
    for i in range(len(audio_files)):    
        audio_file_path = os.path.join(original_path,'Train',str(audio_files[i]))
        if os.path.exists(audio_file_path):
            d, r = librosa.load(audio_file_path, mono=True)
            np.save(os.path.join(original_path, 'train_extracted',str(audio_files[i])+'.npy'), d)
            print(f"Processed: {audio_files[i]}")
        else:
            print(f"Warning: File not found {audio_files[i]}")

def get_testing(original_path):
    """Extract testing data and save as numpy arrays"""
    df = pd.read_csv(os.path.join(original_path,'test_idx.csv'))
    
    if not os.path.exists(os.path.join(original_path,'test_extracted')):
        os.makedirs(os.path.join(original_path,'test_extracted'))
    
    audio_files = np.array(df['new_id'])
    
    for i in range(len(audio_files)):   
        audio_file_path = os.path.join(original_path,'Test',str(audio_files[i]))
        if os.path.exists(audio_file_path):
            d, r = librosa.load(audio_file_path, mono=True)
            np.save(os.path.join(original_path, 'test_extracted',str(audio_files[i])+'.npy'), d)
            print(f"Processed: {audio_files[i]}")
        else:
            print(f"Warning: File not found {audio_files[i]}")

def get_all_features(original_path, csv_file, extracted_folder):
    """Extract comprehensive audio features"""
    df = pd.read_csv(os.path.join(original_path, csv_file))
    df.fillna(0, inplace=True)
    
    audio_extracted = np.array(df['new_id'])
    all_features = []
    
    for i in range(len(audio_extracted)):
        audio_file_data = np.load(os.path.join(original_path, extracted_folder, str(audio_extracted[i])+'.npy'))
        
        # Calculate Root Mean Square Error
        rmse = librosa.feature.rms(y=audio_file_data, frame_length=441)
        
        # Calculate Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio_file_data, frame_length=441)
        
        # Calculate and append statistic features for all the above data features
        addList = np.concatenate((
            np.atleast_1d(np.mean(rmse)),
            np.atleast_1d(np.median(rmse)),
            np.atleast_1d(np.std(rmse)),
            np.atleast_1d(skew(rmse,axis=1)),
            np.atleast_1d(kurtosis(rmse,axis=1)),
            np.atleast_1d(np.mean(zcr)),
            np.atleast_1d(np.median(zcr)),
            np.atleast_1d(np.std(zcr)),
            np.atleast_1d(skew(zcr,axis=1)),
            np.atleast_1d(kurtosis(zcr,axis=1))
        ))
        all_features.append(addList)
        print(f"Extracted features for: {audio_extracted[i]}")
    
    return all_features

if __name__ == "__main__":
    print("Extracting features for DatasetV2...")
    
    # Process unfiltered dataset
    original_path = "DatasetV2"
    print("Extracting training data...")
    get_training(original_path)
    print("Extracting testing data...")
    get_testing(original_path)
    
    print("Feature extraction completed!")