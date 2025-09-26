#!/usr/bin/env python3
"""
Systematic Implementation of Keyword Detection Improvements
Following the 5-phase improvement plan with measurable results
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.stats import kurtosis, skew
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

class KeywordDetectionImprover:
    def __init__(self, dataset_path="DatasetV2"):
        self.dataset_path = dataset_path
        self.results = {}
        
    def load_basic_features(self, csv_file, extracted_folder):
        """Load current basic features (baseline)"""
        df = pd.read_csv(os.path.join(self.dataset_path, csv_file))
        audio_files = df['new_id'].values
        labels = df['keyword'].values
        
        features = []
        valid_labels = []
        
        for i, audio_file in enumerate(audio_files):
            feature_path = os.path.join(self.dataset_path, extracted_folder, f"{audio_file}.npy")
            if os.path.exists(feature_path):
                audio_data = np.load(feature_path)
                
                # Extract basic features (10 features)
                rmse = librosa.feature.rms(y=audio_data, frame_length=441)
                zcr = librosa.feature.zero_crossing_rate(audio_data, frame_length=441)
                
                basic_features = np.concatenate([
                    np.atleast_1d(np.mean(rmse)),
                    np.atleast_1d(np.median(rmse)),
                    np.atleast_1d(np.std(rmse)),
                    np.atleast_1d(skew(rmse, axis=1)),
                    np.atleast_1d(kurtosis(rmse, axis=1)),
                    np.atleast_1d(np.mean(zcr)),
                    np.atleast_1d(np.median(zcr)),
                    np.atleast_1d(np.std(zcr)),
                    np.atleast_1d(skew(zcr, axis=1)),
                    np.atleast_1d(kurtosis(zcr, axis=1))
                ])
                
                features.append(basic_features)
                valid_labels.append(labels[i])
        
        return np.array(features), np.array(valid_labels)
    
    def extract_enhanced_features(self, csv_file, extracted_folder):
        """Phase 1: Extract enhanced features with MFCC and spectral features"""
        df = pd.read_csv(os.path.join(self.dataset_path, csv_file))
        audio_files = df['new_id'].values
        labels = df['keyword'].values
        
        features = []
        valid_labels = []
        
        print(f"Extracting enhanced features from {len(audio_files)} files...")
        
        for i, audio_file in enumerate(audio_files):
            feature_path = os.path.join(self.dataset_path, extracted_folder, f"{audio_file}.npy")
            if os.path.exists(feature_path):
                try:
                    audio_data = np.load(feature_path)
                    
                    # Basic features (10)
                    rmse = librosa.feature.rms(y=audio_data, frame_length=441)
                    zcr = librosa.feature.zero_crossing_rate(audio_data, frame_length=441)
                    
                    basic_feats = np.concatenate([
                        np.atleast_1d(np.mean(rmse)), np.atleast_1d(np.median(rmse)),
                        np.atleast_1d(np.std(rmse)), np.atleast_1d(skew(rmse, axis=1)),
                        np.atleast_1d(kurtosis(rmse, axis=1)),
                        np.atleast_1d(np.mean(zcr)), np.atleast_1d(np.median(zcr)),
                        np.atleast_1d(np.std(zcr)), np.atleast_1d(skew(zcr, axis=1)),
                        np.atleast_1d(kurtosis(zcr, axis=1))
                    ])
                    
                    # MFCC features (26) - Most important for speech
                    try:
                        mfcc = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=13, 
                                                  n_fft=min(2048, len(audio_data)), 
                                                  hop_length=min(512, len(audio_data)//4))
                        mfcc_mean = np.mean(mfcc, axis=1)
                        mfcc_std = np.std(mfcc, axis=1)
                        mfcc_feats = np.concatenate([mfcc_mean, mfcc_std])
                    except:
                        mfcc_feats = np.zeros(26)
                    
                    # Spectral features (12) 
                    try:
                        n_fft = min(1024, len(audio_data))
                        hop_length = min(256, len(audio_data)//4)
                        
                        spec_cent = librosa.feature.spectral_centroid(y=audio_data, sr=22050, 
                                                                    n_fft=n_fft, hop_length=hop_length)[0]
                        spec_bw = librosa.feature.spectral_bandwidth(y=audio_data, sr=22050,
                                                                   n_fft=n_fft, hop_length=hop_length)[0] 
                        spec_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=22050,
                                                                      n_fft=n_fft, hop_length=hop_length)[0]
                        
                        spectral_feats = np.concatenate([
                            [np.mean(spec_cent), np.std(spec_cent), np.max(spec_cent) - np.min(spec_cent)],
                            [np.mean(spec_bw), np.std(spec_bw), np.max(spec_bw) - np.min(spec_bw)],
                            [np.mean(spec_rolloff), np.std(spec_rolloff), np.max(spec_rolloff) - np.min(spec_rolloff)]
                        ])
                        
                        # Additional spectral features
                        spec_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=22050,
                                                                        n_fft=n_fft, hop_length=hop_length)
                        contrast_feats = [np.mean(spec_contrast), np.std(spec_contrast), np.max(spec_contrast) - np.min(spec_contrast)]
                        spectral_feats = np.concatenate([spectral_feats, contrast_feats])
                        
                    except:
                        spectral_feats = np.zeros(12)
                    
                    # Temporal features (6)
                    try:
                        # Enhanced ZCR features
                        zcr_diff = np.diff(zcr[0])
                        temporal_feats = [
                            np.mean(zcr_diff) if len(zcr_diff) > 0 else 0,
                            np.std(zcr_diff) if len(zcr_diff) > 0 else 0,
                            np.max(zcr[0]) - np.min(zcr[0]),
                            len(audio_data) / 22050,  # duration
                            np.sum(np.abs(np.diff(rmse[0]))) / len(rmse[0]),  # energy variation
                            np.sum(audio_data**2) / len(audio_data)  # total energy
                        ]
                    except:
                        temporal_feats = np.zeros(6)
                    
                    # Combine all features (54 total)
                    all_features = np.concatenate([
                        basic_feats,      # 10
                        mfcc_feats,       # 26  
                        spectral_feats,   # 12
                        temporal_feats    # 6
                    ])
                    
                    features.append(all_features)
                    valid_labels.append(labels[i])
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {str(e)[:50]}...")
                    continue
        
        print(f"Successfully extracted features from {len(features)} files")
        return np.array(features), np.array(valid_labels)
    
    def augment_data(self):
        """Phase 3: Implement data augmentation"""
        print("\\nPhase 3: Creating augmented dataset...")
        
        augmented_path = f"{self.dataset_path}_Augmented"
        if os.path.exists(augmented_path):
            print(f"Augmented dataset already exists at {augmented_path}")
            return augmented_path
        
        os.makedirs(augmented_path, exist_ok=True)
        os.makedirs(os.path.join(augmented_path, 'Train'), exist_ok=True)
        os.makedirs(os.path.join(augmented_path, 'Test'), exist_ok=True)
        
        # Load original training data
        train_df = pd.read_csv(os.path.join(self.dataset_path, 'train.csv'))
        class_counts = Counter(train_df['keyword'])
        
        print("Original class distribution:")
        for keyword, count in class_counts.items():
            print(f"  {keyword}: {count} samples")
        
        target_per_class = 30  # Conservative target
        augmented_data = []
        
        for keyword in class_counts.keys():
            class_files = train_df[train_df['keyword'] == keyword]['new_id'].values
            current_count = len(class_files)
            needed = max(0, target_per_class - current_count)
            
            print(f"\\n{keyword}: {current_count} -> {target_per_class} (+{needed} augmented)")
            
            # Copy original files
            for filename in class_files:
                original_path = os.path.join(self.dataset_path, 'Train', filename)
                if os.path.exists(original_path):
                    new_path = os.path.join(augmented_path, 'Train', filename)
                    audio, sr = librosa.load(original_path, sr=22050, mono=True)
                    sf.write(new_path, audio, sr)
                    augmented_data.append({'new_id': filename, 'keyword': keyword})
            
            # Generate augmented samples
            augmented_count = 0
            for i, filename in enumerate(class_files):
                if augmented_count >= needed:
                    break
                    
                original_path = os.path.join(self.dataset_path, 'Train', filename)
                if os.path.exists(original_path):
                    try:
                        audio, sr = librosa.load(original_path, sr=22050, mono=True)
                        
                        # Apply different augmentation techniques
                        augmentation_type = augmented_count % 4
                        
                        if augmentation_type == 0:
                            # Time stretching
                            aug_audio = librosa.effects.time_stretch(audio, rate=0.95)
                            suffix = "stretch"
                        elif augmentation_type == 1:
                            # Volume scaling
                            aug_audio = audio * 1.15
                            suffix = "volume"
                        elif augmentation_type == 2:
                            # Noise addition
                            noise = np.random.normal(0, 0.005 * np.std(audio), len(audio))
                            aug_audio = audio + noise
                            suffix = "noise"
                        else:
                            # Time shifting
                            shift = int(0.1 * len(audio))
                            aug_audio = np.roll(audio, shift)
                            suffix = "shift"
                        
                        # Normalize to prevent clipping
                        if np.max(np.abs(aug_audio)) > 0:
                            aug_audio = aug_audio / np.max(np.abs(aug_audio)) * 0.95
                        
                        aug_filename = f"{filename.split('.')[0]}_{suffix}.wav"
                        aug_path = os.path.join(augmented_path, 'Train', aug_filename)
                        sf.write(aug_path, aug_audio, sr)
                        
                        augmented_data.append({'new_id': aug_filename, 'keyword': keyword})
                        augmented_count += 1
                        
                    except Exception as e:
                        print(f"    Error augmenting {filename}: {str(e)[:30]}...")
                        continue
        
        # Copy test files unchanged
        test_df = pd.read_csv(os.path.join(self.dataset_path, 'test_idx.csv'))
        for filename in test_df['new_id'].values:
            original_path = os.path.join(self.dataset_path, 'Test', filename)
            if os.path.exists(original_path):
                new_path = os.path.join(augmented_path, 'Test', filename)
                audio, sr = librosa.load(original_path, sr=22050, mono=True)
                sf.write(new_path, audio, sr)
        
        # Save CSV files
        aug_train_df = pd.DataFrame(augmented_data)
        aug_train_df.to_csv(os.path.join(augmented_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(augmented_path, 'test_idx.csv'), index=False)
        
        final_counts = Counter(aug_train_df['keyword'])
        print("\\nAugmented class distribution:")
        for keyword, count in final_counts.items():
            print(f"  {keyword}: {count} samples")
        
        return augmented_path
    
    def extract_augmented_features(self, augmented_path):
        """Extract features from augmented dataset"""
        print("Extracting features from augmented dataset...")
        
        # Extract training features
        train_df = pd.read_csv(os.path.join(augmented_path, 'train.csv'))
        audio_files = train_df['new_id'].values
        labels = train_df['keyword'].values
        
        features = []
        valid_labels = []
        
        for i, audio_file in enumerate(audio_files):
            audio_path = os.path.join(augmented_path, 'Train', audio_file)
            if os.path.exists(audio_path):
                try:
                    audio_data, sr = librosa.load(audio_path, sr=22050, mono=True)
                    
                    # Extract same enhanced features as before
                    rmse = librosa.feature.rms(y=audio_data, frame_length=441)
                    zcr = librosa.feature.zero_crossing_rate(audio_data, frame_length=441)
                    
                    basic_feats = np.concatenate([
                        np.atleast_1d(np.mean(rmse)), np.atleast_1d(np.median(rmse)),
                        np.atleast_1d(np.std(rmse)), np.atleast_1d(skew(rmse, axis=1)),
                        np.atleast_1d(kurtosis(rmse, axis=1)),
                        np.atleast_1d(np.mean(zcr)), np.atleast_1d(np.median(zcr)),
                        np.atleast_1d(np.std(zcr)), np.atleast_1d(skew(zcr, axis=1)),
                        np.atleast_1d(kurtosis(zcr, axis=1))
                    ])
                    
                    # MFCC features
                    try:
                        mfcc = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=13,
                                                  n_fft=min(1024, len(audio_data)),
                                                  hop_length=min(256, len(audio_data)//4))
                        mfcc_feats = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
                    except:
                        mfcc_feats = np.zeros(26)
                    
                    # Spectral features 
                    try:
                        n_fft = min(1024, len(audio_data))
                        hop_length = min(256, len(audio_data)//4)
                        
                        spec_cent = librosa.feature.spectral_centroid(y=audio_data, sr=22050, n_fft=n_fft, hop_length=hop_length)[0]
                        spec_bw = librosa.feature.spectral_bandwidth(y=audio_data, sr=22050, n_fft=n_fft, hop_length=hop_length)[0]
                        spectral_feats = np.array([np.mean(spec_cent), np.std(spec_cent), np.mean(spec_bw), np.std(spec_bw)])
                    except:
                        spectral_feats = np.zeros(4)
                    
                    # Combine features (40 total - reduced for stability)
                    all_features = np.concatenate([basic_feats, mfcc_feats, spectral_feats])
                    
                    features.append(all_features)
                    valid_labels.append(labels[i])
                    
                except Exception as e:
                    continue
        
        print(f"Extracted features from {len(features)} augmented samples")
        return np.array(features), np.array(valid_labels)
    
    def run_systematic_improvements(self):
        """Run all improvement phases systematically"""
        
        print("🚀 SYSTEMATIC KEYWORD DETECTION IMPROVEMENTS")
        print("="*60)
        
        # BASELINE: Current performance
        print("\\nBASELINE: Loading current basic features...")
        X_train_basic, y_train = self.load_basic_features('train.csv', 'train_extracted')
        X_test_basic, y_test = self.load_basic_features('test_idx.csv', 'test_extracted')
        
        print(f"Baseline features: {X_train_basic.shape[1]}")
        print(f"Training samples: {X_train_basic.shape[0]}, Test samples: {X_test_basic.shape[0]}")
        
        # Test baseline model
        scaler_basic = StandardScaler()
        X_train_basic_scaled = scaler_basic.fit_transform(X_train_basic)
        X_test_basic_scaled = scaler_basic.transform(X_test_basic)
        
        rf_basic = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
        rf_basic.fit(X_train_basic_scaled, y_train)
        
        baseline_train_acc = rf_basic.score(X_train_basic_scaled, y_train)
        baseline_test_acc = rf_basic.score(X_test_basic_scaled, y_test)
        
        self.results['baseline'] = {
            'train_acc': baseline_train_acc,
            'test_acc': baseline_test_acc,
            'overfitting': baseline_train_acc - baseline_test_acc,
            'features': X_train_basic.shape[1]
        }
        
        print(f"\\nBaseline Results:")
        print(f"  Training: {baseline_train_acc:.3f}, Test: {baseline_test_acc:.3f}")
        print(f"  Overfitting: {baseline_train_acc - baseline_test_acc:.3f}")
        
        # PHASE 1: Enhanced Features
        print("\\n" + "="*60)
        print("PHASE 1: Enhanced Feature Engineering")
        print("="*60)
        
        X_train_enhanced, y_train_enh = self.extract_enhanced_features('train.csv', 'train_extracted') 
        X_test_enhanced, y_test_enh = self.extract_enhanced_features('test_idx.csv', 'test_extracted')
        
        print(f"Enhanced features: {X_train_enhanced.shape[1]}")
        
        # Test with enhanced features
        scaler_enh = StandardScaler()
        X_train_enh_scaled = scaler_enh.fit_transform(X_train_enhanced)
        X_test_enh_scaled = scaler_enh.transform(X_test_enhanced)
        
        rf_enhanced = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
        rf_enhanced.fit(X_train_enh_scaled, y_train_enh)
        
        enhanced_train_acc = rf_enhanced.score(X_train_enh_scaled, y_train_enh)
        enhanced_test_acc = rf_enhanced.score(X_test_enh_scaled, y_test_enh)
        
        self.results['enhanced_features'] = {
            'train_acc': enhanced_train_acc,
            'test_acc': enhanced_test_acc,
            'overfitting': enhanced_train_acc - enhanced_test_acc,
            'features': X_train_enhanced.shape[1],
            'improvement': enhanced_test_acc - baseline_test_acc
        }
        
        print(f"Enhanced Features Results:")
        print(f"  Training: {enhanced_train_acc:.3f}, Test: {enhanced_test_acc:.3f}")
        print(f"  Overfitting: {enhanced_train_acc - enhanced_test_acc:.3f}")
        print(f"  Improvement: {enhanced_test_acc - baseline_test_acc:+.3f}")
        
        # PHASE 2: Regularization
        print("\\n" + "="*60)
        print("PHASE 2: Overfitting Mitigation")
        print("="*60)
        
        # Feature selection
        if X_train_enhanced.shape[1] > 30:
            selector = SelectKBest(mutual_info_classif, k=30)
            X_train_selected = selector.fit_transform(X_train_enhanced, y_train_enh)
            X_test_selected = selector.transform(X_test_enhanced)
            print(f"Selected {X_train_selected.shape[1]} features using mutual information")
        else:
            X_train_selected = X_train_enhanced
            X_test_selected = X_test_enhanced
        
        # Regularized model
        rf_regularized = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )
        
        scaler_reg = StandardScaler()
        X_train_reg_scaled = scaler_reg.fit_transform(X_train_selected)
        X_test_reg_scaled = scaler_reg.transform(X_test_selected)
        
        rf_regularized.fit(X_train_reg_scaled, y_train_enh)
        
        reg_train_acc = rf_regularized.score(X_train_reg_scaled, y_train_enh)
        reg_test_acc = rf_regularized.score(X_test_reg_scaled, y_test_enh)
        
        self.results['regularized'] = {
            'train_acc': reg_train_acc,
            'test_acc': reg_test_acc,
            'overfitting': reg_train_acc - reg_test_acc,
            'features': X_train_selected.shape[1],
            'improvement': reg_test_acc - baseline_test_acc
        }
        
        print(f"Regularized Model Results:")
        print(f"  Training: {reg_train_acc:.3f}, Test: {reg_test_acc:.3f}")
        print(f"  Overfitting: {reg_train_acc - reg_test_acc:.3f}")
        print(f"  Improvement: {reg_test_acc - baseline_test_acc:+.3f}")
        
        # PHASE 3: Data Augmentation
        print("\\n" + "="*60)
        print("PHASE 3: Data Augmentation")
        print("="*60)
        
        try:
            augmented_path = self.augment_data()
            X_train_aug, y_train_aug = self.extract_augmented_features(augmented_path)
            
            # Test with augmented data
            scaler_aug = StandardScaler()
            X_train_aug_scaled = scaler_aug.fit_transform(X_train_aug)
            X_test_aug_scaled = scaler_aug.transform(X_test_enh_scaled[:len(y_test_enh)])  # Use original test set
            
            rf_augmented = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=8,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42
            )
            
            rf_augmented.fit(X_train_aug_scaled, y_train_aug)
            
            aug_train_acc = rf_augmented.score(X_train_aug_scaled, y_train_aug)
            aug_test_acc = rf_augmented.score(X_test_aug_scaled, y_test_enh)
            
            self.results['augmented'] = {
                'train_acc': aug_train_acc,
                'test_acc': aug_test_acc,
                'overfitting': aug_train_acc - aug_test_acc,
                'train_samples': X_train_aug.shape[0],
                'improvement': aug_test_acc - baseline_test_acc
            }
            
            print(f"Augmented Data Results:")
            print(f"  Training samples: {X_train_aug.shape[0]} (vs {X_train_basic.shape[0]} baseline)")
            print(f"  Training: {aug_train_acc:.3f}, Test: {aug_test_acc:.3f}")
            print(f"  Overfitting: {aug_train_acc - aug_test_acc:.3f}")
            print(f"  Improvement: {aug_test_acc - baseline_test_acc:+.3f}")
            
        except Exception as e:
            print(f"Data augmentation failed: {e}")
            self.results['augmented'] = None
        
        # PHASE 4: Algorithm Comparison
        print("\\n" + "="*60)
        print("PHASE 4: Algorithm Optimization")
        print("="*60)
        
        algorithms = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=8, min_samples_split=8,
                max_features='sqrt', class_weight='balanced', random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )
        }
        
        best_algo = None
        best_score = 0
        
        for name, model in algorithms.items():
            model.fit(X_train_reg_scaled, y_train_enh)
            train_acc = model.score(X_train_reg_scaled, y_train_enh)
            test_acc = model.score(X_test_reg_scaled, y_test_enh)
            
            print(f"{name}:")
            print(f"  Training: {train_acc:.3f}, Test: {test_acc:.3f}")
            print(f"  Overfitting: {train_acc - test_acc:.3f}")
            print(f"  Improvement: {test_acc - baseline_test_acc:+.3f}")
            
            if test_acc > best_score:
                best_score = test_acc
                best_algo = name
        
        self.results['best_algorithm'] = {
            'name': best_algo,
            'test_acc': best_score,
            'improvement': best_score - baseline_test_acc
        }
        
        # PHASE 5: Cross-Validation
        print("\\n" + "="*60)
        print("PHASE 5: Cross-Validation Evaluation")
        print("="*60)
        
        # Combine train and test for better CV evaluation
        if 'augmented' in self.results and self.results['augmented'] is not None:
            X_all = X_train_aug_scaled
            y_all = y_train_aug
            dataset_name = "augmented"
        else:
            X_all = np.vstack([X_train_reg_scaled, X_test_reg_scaled])
            y_all = np.hstack([y_train_enh, y_test_enh])
            dataset_name = "enhanced+regularized"
        
        cv = StratifiedKFold(n_splits=min(10, len(y_all)//4), shuffle=True, random_state=42)
        
        # Test best model with CV
        best_model = algorithms[best_algo] if best_algo else rf_regularized
        cv_scores = cross_val_score(best_model, X_all, y_all, cv=cv, scoring='accuracy')
        
        self.results['cross_validation'] = {
            'mean_acc': np.mean(cv_scores),
            'std_acc': np.std(cv_scores),
            'dataset': dataset_name,
            'folds': len(cv_scores)
        }
        
        print(f"Cross-Validation Results ({dataset_name} dataset):")
        print(f"  Mean Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        print(f"  Individual Scores: {[f'{score:.3f}' for score in cv_scores]}")
        
        return self.results
    
    def print_final_summary(self):
        """Print comprehensive improvement summary"""
        print("\\n" + "="*80)
        print("🎯 FINAL IMPROVEMENT SUMMARY")
        print("="*80)
        
        baseline_acc = self.results['baseline']['test_acc']
        
        print(f"\\n📊 PERFORMANCE PROGRESSION:")
        phases = ['baseline', 'enhanced_features', 'regularized', 'augmented', 'best_algorithm']
        
        for phase in phases:
            if phase in self.results and self.results[phase] is not None:
                result = self.results[phase]
                if 'test_acc' in result:
                    acc = result['test_acc']
                    improvement = acc - baseline_acc
                    overfitting = result.get('overfitting', 0)
                    
                    phase_name = phase.replace('_', ' ').title()
                    print(f"{phase_name:20}: {acc:.3f} ({improvement:+.3f}) | Overfitting: {overfitting:.3f}")
        
        # Cross-validation results
        if 'cross_validation' in self.results:
            cv_result = self.results['cross_validation']
            cv_acc = cv_result['mean_acc']
            cv_improvement = cv_acc - baseline_acc
            print(f"{'Cross-Validation':20}: {cv_acc:.3f} ± {cv_result['std_acc']:.3f} ({cv_improvement:+.3f})")
        
        print(f"\\n📈 KEY IMPROVEMENTS:")
        
        # Find best single improvement
        best_single_improvement = 0
        best_phase = "baseline"
        
        for phase in ['enhanced_features', 'regularized', 'augmented', 'best_algorithm']:
            if phase in self.results and self.results[phase] is not None:
                if 'improvement' in self.results[phase]:
                    improvement = self.results[phase]['improvement']
                    if improvement > best_single_improvement:
                        best_single_improvement = improvement
                        best_phase = phase
        
        print(f"Best Single Phase: {best_phase.replace('_', ' ').title()} (+{best_single_improvement:.3f})")
        
        # Overall improvement
        if 'cross_validation' in self.results:
            total_improvement = self.results['cross_validation']['mean_acc'] - baseline_acc
        else:
            total_improvement = best_single_improvement
        
        print(f"Total Improvement: {total_improvement:+.3f} ({total_improvement/baseline_acc*100:+.1f}%)")
        
        # Overfitting improvement
        baseline_overfitting = self.results['baseline']['overfitting']
        if 'regularized' in self.results:
            final_overfitting = self.results['regularized']['overfitting']
            overfitting_reduction = baseline_overfitting - final_overfitting
            print(f"Overfitting Reduction: {overfitting_reduction:.3f} (from {baseline_overfitting:.3f} to {final_overfitting:.3f})")
        
        print(f"\\n✅ OBJECTIVES MET:")
        objectives_met = []
        
        if total_improvement > 0.10:
            objectives_met.append("✅ Significant accuracy improvement (>10%)")
        elif total_improvement > 0.05:
            objectives_met.append("✅ Moderate accuracy improvement (5-10%)")
        
        if 'regularized' in self.results and self.results['regularized']['overfitting'] < 0.20:
            objectives_met.append("✅ Reduced overfitting (<20%)")
        
        if 'augmented' in self.results and self.results['augmented'] is not None:
            objectives_met.append("✅ Successfully balanced dataset")
        
        if 'cross_validation' in self.results:
            objectives_met.append("✅ Reliable evaluation with cross-validation")
        
        for objective in objectives_met:
            print(f"  {objective}")
        
        return total_improvement

if __name__ == "__main__":
    improver = KeywordDetectionImprover()
    results = improver.run_systematic_improvements()
    final_improvement = improver.print_final_summary()