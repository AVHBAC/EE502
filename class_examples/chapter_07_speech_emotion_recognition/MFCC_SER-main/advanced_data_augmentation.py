import pandas as pd
import numpy as np
import librosa
import librosa.effects
from scipy.signal import wiener
import warnings
warnings.filterwarnings("ignore")

class AdvancedAudioAugmentation:
    """Advanced audio augmentation techniques for speech emotion recognition"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
    
    def add_noise(self, data, noise_factor=0.005):
        """Add white Gaussian noise"""
        noise = np.random.randn(len(data))
        augmented_data = data + noise_factor * noise
        return augmented_data.astype(type(data[0]))
    
    def add_colored_noise(self, data, noise_type='pink', noise_factor=0.01):
        """Add colored noise (pink, brown, blue)"""
        noise = np.random.randn(len(data))
        
        if noise_type == 'pink':
            # Pink noise (1/f noise)
            fft_noise = np.fft.fft(noise)
            freqs = np.fft.fftfreq(len(noise), 1/self.sample_rate)
            # Avoid division by zero
            freqs[0] = 1
            pink_filter = 1 / np.sqrt(np.abs(freqs))
            pink_noise = np.fft.ifft(fft_noise * pink_filter).real
            noise = pink_noise
        elif noise_type == 'brown':
            # Brown noise (1/f^2 noise)
            fft_noise = np.fft.fft(noise)
            freqs = np.fft.fftfreq(len(noise), 1/self.sample_rate)
            freqs[0] = 1
            brown_filter = 1 / np.abs(freqs)
            brown_noise = np.fft.ifft(fft_noise * brown_filter).real
            noise = brown_noise
        
        augmented_data = data + noise_factor * noise
        return augmented_data.astype(type(data[0]))
    
    def time_stretch(self, data, stretch_rate=None):
        """Time stretching without pitch change"""
        if stretch_rate is None:
            stretch_rate = np.random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(data, rate=stretch_rate)
    
    def pitch_shift(self, data, n_steps=None):
        """Pitch shifting without tempo change"""
        if n_steps is None:
            n_steps = np.random.uniform(-2, 2)
        return librosa.effects.pitch_shift(data, sr=self.sample_rate, n_steps=n_steps)
    
    def dynamic_range_compression(self, data, threshold=0.5, ratio=4.0):
        """Apply dynamic range compression"""
        data_abs = np.abs(data)
        compressed = np.sign(data) * np.where(
            data_abs > threshold,
            threshold + (data_abs - threshold) / ratio,
            data_abs
        )
        return compressed
    
    def add_reverb(self, data, reverb_factor=0.1):
        """Add simple reverb effect"""
        delay_samples = int(0.05 * self.sample_rate)  # 50ms delay
        reverb = np.zeros_like(data)
        reverb[delay_samples:] = data[:-delay_samples] * reverb_factor
        return data + reverb
    
    def spectral_rolloff_shift(self, data, shift_factor=0.1):
        """Shift spectral characteristics"""
        stft = librosa.stft(data)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply frequency domain modification
        freq_shift = int(magnitude.shape[0] * shift_factor)
        if freq_shift > 0:
            magnitude = np.roll(magnitude, freq_shift, axis=0)
        
        # Reconstruct signal
        modified_stft = magnitude * np.exp(1j * phase)
        return librosa.istft(modified_stft)
    
    def formant_shift(self, data, shift_factor=1.1):
        """Formant shifting using vocal tract length perturbation"""
        # This is a simplified formant shifting
        stft = librosa.stft(data, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Frequency warping
        n_fft = magnitude.shape[0]
        freq_bins = np.arange(n_fft)
        warped_bins = freq_bins * shift_factor
        warped_bins = np.clip(warped_bins, 0, n_fft - 1).astype(int)
        
        warped_magnitude = magnitude[warped_bins, :]
        modified_stft = warped_magnitude * np.exp(1j * phase)
        
        return librosa.istft(modified_stft)
    
    def add_room_impulse(self, data, room_type='small'):
        """Simulate room acoustics"""
        if room_type == 'small':
            # Small room impulse response
            impulse = np.array([1.0, 0.0, 0.3, 0.0, 0.1])
        elif room_type == 'large':
            # Large room impulse response
            impulse = np.array([1.0] + [0.0] * 10 + [0.5] + [0.0] * 20 + [0.2])
        else:
            # Medium room
            impulse = np.array([1.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.15])
        
        return np.convolve(data, impulse, mode='same')
    
    def vocal_tract_length_perturbation(self, data, alpha=0.9):
        """VTLP - commonly used in speech recognition"""
        stft = librosa.stft(data)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Frequency warping
        n_fft, n_frames = magnitude.shape
        freq_warp = np.arange(n_fft) * alpha
        freq_warp = np.clip(freq_warp, 0, n_fft - 1)
        
        # Interpolate to new frequency bins
        warped_magnitude = np.zeros_like(magnitude)
        for frame in range(n_frames):
            warped_magnitude[:, frame] = np.interp(
                np.arange(n_fft), freq_warp, magnitude[:, frame]
            )
        
        modified_stft = warped_magnitude * np.exp(1j * phase)
        return librosa.istft(modified_stft)
    
    def speed_perturbation(self, data, speed_factor=None):
        """Speed perturbation (common in speech processing)"""
        if speed_factor is None:
            speed_factor = np.random.uniform(0.9, 1.1)
        
        # Resample to change speed
        original_length = len(data)
        new_length = int(original_length / speed_factor)
        
        # Use librosa's resample
        resampled = librosa.resample(data, orig_sr=self.sample_rate, target_sr=self.sample_rate*speed_factor)
        
        # Pad or trim to original length
        if len(resampled) > original_length:
            return resampled[:original_length]
        else:
            padded = np.pad(resampled, (0, original_length - len(resampled)), mode='constant')
            return padded
    
    def spec_augment(self, melspec, time_mask_param=10, freq_mask_param=5, num_masks=2):
        """SpecAugment for mel spectrograms"""
        augmented_spec = melspec.copy()
        
        # Time masking
        for _ in range(num_masks):
            t = np.random.uniform(0, time_mask_param)
            t0 = np.random.uniform(0, max(1, melspec.shape[1] - t))
            augmented_spec[:, int(t0):int(t0 + t)] = 0
        
        # Frequency masking
        for _ in range(num_masks):
            f = np.random.uniform(0, freq_mask_param)
            f0 = np.random.uniform(0, max(1, melspec.shape[0] - f))
            augmented_spec[int(f0):int(f0 + f), :] = 0
        
        return augmented_spec

class EnhancedFeatureExtractor:
    """Enhanced feature extraction with augmentation"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.augmenter = AdvancedAudioAugmentation(sample_rate)
    
    def extract_features(self, data):
        """Extract comprehensive audio features"""
        result = np.array([])
        
        # Basic features
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result = np.hstack((result, zcr))
        
        # MFCC (13 coefficients + deltas + delta-deltas)
        mfcc = librosa.feature.mfcc(y=data, sr=self.sample_rate, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        mfcc_features = np.hstack([
            np.mean(mfcc.T, axis=0),
            np.std(mfcc.T, axis=0),
            np.mean(mfcc_delta.T, axis=0),
            np.mean(mfcc_delta2.T, axis=0)
        ])
        result = np.hstack((result, mfcc_features))
        
        # Chroma features
        stft = np.abs(librosa.stft(data))
        chroma = librosa.feature.chroma_stft(S=stft, sr=self.sample_rate)
        result = np.hstack((result, np.mean(chroma.T, axis=0)))
        result = np.hstack((result, np.std(chroma.T, axis=0)))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=data, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=self.sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=data, sr=self.sample_rate)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=data, sr=self.sample_rate)
        spectral_flatness = librosa.feature.spectral_flatness(y=data)
        
        result = np.hstack((result, [np.mean(spectral_centroids), np.std(spectral_centroids)]))
        result = np.hstack((result, [np.mean(spectral_rolloff), np.std(spectral_rolloff)]))
        result = np.hstack((result, [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)]))
        result = np.hstack((result, np.mean(spectral_contrast.T, axis=0)))
        result = np.hstack((result, [np.mean(spectral_flatness), np.std(spectral_flatness)]))
        
        # RMS Energy
        rms = librosa.feature.rms(y=data)
        result = np.hstack((result, [np.mean(rms), np.std(rms)]))
        
        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=data, sr=self.sample_rate, n_mels=40)
        result = np.hstack((result, np.mean(mel.T, axis=0)))
        
        # Tonnetz (harmonic features)
        tonnetz = librosa.feature.tonnetz(y=data, sr=self.sample_rate)
        result = np.hstack((result, np.mean(tonnetz.T, axis=0)))
        
        return result
    
    def get_augmented_features(self, path, augmentation_factor=5):
        """Extract features with multiple augmentations"""
        # Load audio
        data, _ = librosa.load(path, duration=2.5, offset=0.6, sr=self.sample_rate)
        
        # Original features
        original_features = self.extract_features(data)
        features_list = [original_features]
        
        # Augmentation techniques
        augmentations = [
            lambda x: self.augmenter.add_noise(x, noise_factor=0.005),
            lambda x: self.augmenter.add_colored_noise(x, 'pink', 0.01),
            lambda x: self.augmenter.time_stretch(x, np.random.uniform(0.85, 1.15)),
            lambda x: self.augmenter.pitch_shift(x, np.random.uniform(-1.5, 1.5)),
            lambda x: self.augmenter.speed_perturbation(x, np.random.uniform(0.9, 1.1)),
            lambda x: self.augmenter.formant_shift(x, np.random.uniform(0.95, 1.05)),
            lambda x: self.augmenter.add_reverb(x, np.random.uniform(0.05, 0.15)),
            lambda x: self.augmenter.vocal_tract_length_perturbation(x, np.random.uniform(0.88, 1.12)),
            lambda x: self.augmenter.dynamic_range_compression(x, 0.6, 3.0),
            lambda x: self.augmenter.add_room_impulse(x, np.random.choice(['small', 'medium', 'large']))
        ]
        
        # Apply random augmentations
        for _ in range(augmentation_factor - 1):  # -1 because we already have original
            aug_func = np.random.choice(augmentations)
            try:
                augmented_data = aug_func(data.copy())
                # Ensure same length as original
                if len(augmented_data) != len(data):
                    if len(augmented_data) > len(data):
                        augmented_data = augmented_data[:len(data)]
                    else:
                        augmented_data = np.pad(augmented_data, 
                                              (0, len(data) - len(augmented_data)), 
                                              mode='constant')
                
                augmented_features = self.extract_features(augmented_data)
                features_list.append(augmented_features)
            except Exception as e:
                # If augmentation fails, use original with slight noise
                noisy_data = self.augmenter.add_noise(data, 0.002)
                features_list.append(self.extract_features(noisy_data))
        
        return np.array(features_list)

def create_enhanced_dataset(data_path_csv="data_path.csv", 
                          output_csv="enhanced_features.csv",
                          augmentation_factor=5):
    """Create enhanced dataset with advanced augmentation"""
    print("Loading data paths...")
    data_path = pd.read_csv(data_path_csv)
    
    print("Initializing enhanced feature extractor...")
    extractor = EnhancedFeatureExtractor()
    
    print("Extracting enhanced features with augmentation...")
    X, Y = [], []
    
    for idx, (path, emotion) in enumerate(zip(data_path.Path, data_path.Emotions)):
        if idx % 100 == 0:
            print(f"Processing {idx}/{len(data_path)} files...")
        
        try:
            # Get augmented features
            features_list = extractor.get_augmented_features(path, augmentation_factor)
            
            # Add all features and corresponding labels
            for features in features_list:
                X.append(features)
                Y.append(emotion)
                
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    print(f"Created dataset with {len(X)} samples from {len(data_path)} original files")
    
    # Create DataFrame
    features_df = pd.DataFrame(X)
    features_df['labels'] = Y
    
    # Save enhanced dataset
    features_df.to_csv(output_csv, index=False)
    print(f"Enhanced dataset saved as {output_csv}")
    
    return features_df

if __name__ == "__main__":
    # Create enhanced dataset
    enhanced_df = create_enhanced_dataset(
        data_path_csv="data_path.csv",
        output_csv="enhanced_features.csv",
        augmentation_factor=6  # 6x more data
    )
    
    print(f"Enhanced dataset shape: {enhanced_df.shape}")
    print(f"Label distribution:")
    print(enhanced_df['labels'].value_counts())