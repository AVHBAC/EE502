#!/usr/bin/env python3
"""
Improved Keyword Detection Model
Implements enhanced features, regularization, and ensemble methods
"""

import os
import numpy as np
import pandas as pd
import librosa
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

def load_enhanced_features(dataset_path, csv_file, extracted_folder):
    """Load enhanced features from numpy files"""
    df = pd.read_csv(os.path.join(dataset_path, csv_file))
    df.fillna(0, inplace=True)
    
    audio_files = df['new_id'].values
    labels = df['keyword'].values
    
    features = []
    valid_labels = []
    
    for i, audio_file in enumerate(audio_files):
        feature_path = os.path.join(dataset_path, extracted_folder, f"{audio_file}.npy")
        if os.path.exists(feature_path):
            feature_vector = np.load(feature_path)
            features.append(feature_vector)
            valid_labels.append(labels[i])
        else:
            print(f"Warning: Features not found for {audio_file}")
    
    return np.array(features), np.array(valid_labels)

def create_regularized_random_forest(n_estimators=100):
    """Create regularized Random Forest classifier"""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=8,                    # Limit tree depth
        min_samples_split=10,           # Require more samples to split
        min_samples_leaf=5,             # Require more samples in leaves  
        max_features='sqrt',            # Random feature sampling
        class_weight='balanced',        # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )

def create_ensemble_classifier():
    """Create ensemble classifier with multiple algorithms"""
    
    # Individual classifiers
    rf_clf = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_split=10,
        max_features='sqrt', class_weight='balanced', random_state=42
    )
    
    gb_clf = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, 
        random_state=42
    )
    
    svm_clf = SVC(
        kernel='rbf', C=1.0, gamma='scale', 
        class_weight='balanced', probability=True, random_state=42
    )
    
    lr_clf = LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42
    )
    
    # Voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_clf),
            ('gb', gb_clf), 
            ('svm', svm_clf),
            ('lr', lr_clf)
        ],
        voting='soft'  # Use probabilities for voting
    )
    
    return ensemble

def create_processing_pipeline(classifier, n_features=50):
    """Create preprocessing pipeline with feature selection"""
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(mutual_info_classif, k=n_features)),
        ('classifier', classifier)
    ])
    
    return pipeline

def evaluate_model_comprehensive(model, X, y, cv_folds=5):
    """Comprehensive model evaluation with cross-validation"""
    
    print("=== COMPREHENSIVE MODEL EVALUATION ===\\n")
    
    # Stratified cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    print("Cross-Validation Results:")
    print(f"  Mean Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    print(f"  Individual Scores: {[f'{score:.3f}' for score in cv_scores]}")
    
    # Fit model for detailed analysis
    model.fit(X, y)
    predictions = model.predict(X)
    
    # Training accuracy
    train_accuracy = accuracy_score(y, predictions)
    print(f"  Training Accuracy: {train_accuracy:.3f}")
    
    # Overfitting analysis
    overfitting_gap = train_accuracy - np.mean(cv_scores)
    print(f"  Overfitting Gap: {overfitting_gap:.3f}")
    
    if overfitting_gap > 0.15:
        print("  ⚠️  High overfitting detected - consider more regularization")
    elif overfitting_gap > 0.05:
        print("  ⚠️  Moderate overfitting - acceptable but can be improved")
    else:
        print("  ✅ Good generalization")
    
    # Detailed classification report
    print("\\nClassification Report:")
    print(classification_report(y, predictions))
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        top_features = np.argsort(feature_importance)[-10:][::-1]
        print("\\nTop 10 Most Important Features:")
        for i, idx in enumerate(top_features):
            print(f"  {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
    
    return {
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores), 
        'train_accuracy': train_accuracy,
        'overfitting_gap': overfitting_gap
    }

def hyperparameter_tuning(X, y, classifier_type='rf'):
    """Perform hyperparameter tuning for specified classifier"""
    
    print(f"\\n=== HYPERPARAMETER TUNING ({classifier_type.upper()}) ===\\n")
    
    if classifier_type == 'rf':
        classifier = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 8, 12, None],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 'log2', None]
        }
    
    elif classifier_type == 'gb':
        classifier = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 8],
            'min_samples_split': [5, 10, 15]
        }
    
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    # Grid search with cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        classifier, param_grid, cv=cv, scoring='accuracy', 
        n_jobs=-1, verbose=1
    )
    
    print("Running grid search...")
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def compare_models(X, y):
    """Compare different model architectures"""
    
    print("\\n=== MODEL COMPARISON ===\\n")
    
    models = {
        'Baseline RF': RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42),
        'Regularized RF': create_regularized_random_forest(n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'Ensemble': create_ensemble_classifier()
    }
    
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Create pipeline with preprocessing
        pipeline = create_processing_pipeline(model, n_features=min(50, X.shape[1]))
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        
        results[name] = {
            'mean': np.mean(cv_scores),
            'std': np.std(cv_scores),
            'scores': cv_scores
        }
        
        print(f"  CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['mean'])
    print(f"\\n🏆 Best Model: {best_model} ({results[best_model]['mean']:.3f} ± {results[best_model]['std']:.3f})")
    
    return results, models[best_model]

def train_final_model(X_train, y_train, X_test, y_test):
    """Train and evaluate final optimized model"""
    
    print("\\n=== TRAINING FINAL OPTIMIZED MODEL ===\\n")
    
    # Create optimized ensemble model
    best_model = create_ensemble_classifier()
    
    # Create pipeline with preprocessing and feature selection
    pipeline = create_processing_pipeline(best_model, n_features=min(60, X_train.shape[1]))
    
    # Train model
    print("Training final model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on training set
    train_pred = pipeline.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    # Evaluate on test set
    test_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"Final Model Performance:")
    print(f"  Training Accuracy: {train_accuracy:.3f}")
    print(f"  Test Accuracy: {test_accuracy:.3f}")
    print(f"  Overfitting Gap: {train_accuracy - test_accuracy:.3f}")
    
    # Detailed test results
    print("\\nTest Set Classification Report:")
    print(classification_report(y_test, test_pred))
    
    print("\\nTest Set Confusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    labels = sorted(np.unique(y_test))
    print("Actual\\Predicted:", "  ".join(f"{l:>8}" for l in labels))
    for i, label in enumerate(labels):
        print(f"{label:>16}: {"  ".join(f"{cm[i,j]:>8}" for j in range(len(labels)))}")
    
    return pipeline, {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy, 
        'overfitting_gap': train_accuracy - test_accuracy
    }

def main():
    """Main function to run improved keyword detection model"""
    
    print("🚀 IMPROVED KEYWORD DETECTION MODEL\\n")
    
    # Try to load enhanced features first, fall back to basic features if not available
    dataset_paths = ["DatasetV2_Augmented", "DatasetV2"]  # Try augmented first
    
    X_train, y_train, X_test, y_test = None, None, None, None
    
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            print(f"Loading data from {dataset_path}...")
            
            try:
                # Try enhanced features
                if os.path.exists(os.path.join(dataset_path, 'train_extracted_enhanced')):
                    print("  Using enhanced features...")
                    X_train, y_train = load_enhanced_features(
                        dataset_path, 'train.csv', 'train_extracted_enhanced'
                    )
                    X_test, y_test = load_enhanced_features(
                        dataset_path, 'test_idx.csv', 'test_extracted_enhanced'  
                    )
                # Fall back to basic features
                elif os.path.exists(os.path.join(dataset_path, 'train_extracted')):
                    print("  Using basic features...")
                    # Use basic feature extraction function
                    from enhanced_features import get_all_features
                    X_train = get_all_features(dataset_path, 'train.csv', 'train_extracted')
                    X_test = get_all_features(dataset_path, 'test_idx.csv', 'test_extracted')
                    y_train = pd.read_csv(os.path.join(dataset_path, 'train.csv'))['keyword'].values
                    y_test = pd.read_csv(os.path.join(dataset_path, 'test_idx.csv'))['keyword'].values
                    X_train, X_test = np.array(X_train), np.array(X_test)
                else:
                    continue
                    
                print(f"  Training samples: {X_train.shape}")
                print(f"  Test samples: {X_test.shape}")
                break
                
            except Exception as e:
                print(f"  Error loading from {dataset_path}: {e}")
                continue
    
    if X_train is None:
        print("❌ No suitable dataset found. Please run enhanced_features.py first.")
        return
    
    # Model comparison
    print(f"\\nDataset loaded: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} features")
    
    # Compare different models
    results, best_model = compare_models(X_train, y_train)
    
    # Hyperparameter tuning for Random Forest
    tuned_rf, best_params, best_score = hyperparameter_tuning(X_train, y_train, 'rf')
    
    # Train final optimized model
    final_model, final_results = train_final_model(X_train, y_train, X_test, y_test)
    
    # Summary
    print("\\n" + "="*60)
    print("📊 IMPROVEMENT SUMMARY")
    print("="*60)
    print(f"Final Test Accuracy: {final_results['test_accuracy']:.3f}")
    print(f"Overfitting Reduction: {40.0 - final_results['overfitting_gap']*100:.1f} percentage points")
    
    if final_results['test_accuracy'] > 0.70:
        print("🎉 Significant improvement achieved!")
    elif final_results['test_accuracy'] > 0.65:
        print("✅ Good improvement achieved!")
    else:
        print("⚠️  Moderate improvement - consider additional techniques")
    
    # Save final model
    import joblib
    joblib.dump(final_model, 'improved_keyword_model.pkl')
    print("\\n💾 Final model saved as 'improved_keyword_model.pkl'")

if __name__ == "__main__":
    main()