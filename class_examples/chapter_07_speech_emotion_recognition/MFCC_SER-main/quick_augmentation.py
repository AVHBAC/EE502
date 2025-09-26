import pandas as pd
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

class QuickAugmentation:
    """Quick audio augmentation for faster testing"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
    
    def add_noise(self, data, noise_factor=0.005):
        """Add white noise"""
        noise = np.random.randn(len(data))
        return data + noise_factor * noise
    
    def time_stretch(self, data, rate=0.9):
        """Time stretching"""
        return librosa.effects.time_stretch(data, rate=rate)
    
    def pitch_shift(self, data, n_steps=1.5):
        """Pitch shifting"""
        return librosa.effects.pitch_shift(data, sr=self.sample_rate, n_steps=n_steps)

def extract_enhanced_features(data, sample_rate=22050):
    """Extract enhanced audio features"""
    result = np.array([])
    
    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    
    # MFCC (13 coefficients)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13).T, axis=0)
    result = np.hstack((result, mfcc))
    
    # Chroma STFT
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))
    
    # RMS Energy
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    
    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    
    # Spectral features
    spectral_centroids = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate)[0])
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sample_rate)[0])
    result = np.hstack((result, [spectral_centroids, spectral_rolloff]))
    
    return result

def create_quick_enhanced_dataset():
    """Create enhanced dataset with quick augmentation"""
    print("Loading data paths...")
    data_path = pd.read_csv("data_path.csv")
    
    print("Creating enhanced features...")
    augmenter = QuickAugmentation()
    
    X, Y = [], []
    
    # Take only first 200 files for quick testing
    sample_size = min(200, len(data_path))
    
    for idx in range(sample_size):
        path = data_path.Path.iloc[idx]
        emotion = data_path.Emotions.iloc[idx]
        
        if idx % 50 == 0:
            print(f"Processing {idx}/{sample_size} files...")
        
        try:
            # Load audio
            data, _ = librosa.load(path, duration=2.5, offset=0.6, sr=22050)
            
            # Original features
            original_features = extract_enhanced_features(data)
            X.append(original_features)
            Y.append(emotion)
            
            # Add noise augmentation
            noisy_data = augmenter.add_noise(data)
            noise_features = extract_enhanced_features(noisy_data)
            X.append(noise_features)
            Y.append(emotion)
            
            # Add time stretch augmentation
            stretched_data = augmenter.time_stretch(data)
            stretch_features = extract_enhanced_features(stretched_data)
            X.append(stretch_features)
            Y.append(emotion)
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    print(f"Created dataset with {len(X)} samples from {sample_size} original files")
    
    # Create DataFrame
    features_df = pd.DataFrame(X)
    features_df['labels'] = Y
    
    # Save enhanced dataset
    features_df.to_csv("enhanced_features.csv", index=False)
    print(f"Enhanced dataset saved as enhanced_features.csv")
    
    return features_df

if __name__ == "__main__":
    enhanced_df = create_quick_enhanced_dataset()
    print(f"Enhanced dataset shape: {enhanced_df.shape}")
    print(f"Label distribution:")
    print(enhanced_df['labels'].value_counts())