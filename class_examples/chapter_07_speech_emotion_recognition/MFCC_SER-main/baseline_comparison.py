import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

def load_and_prepare_data(use_enhanced=True):
    """Load and prepare data"""
    if use_enhanced:
        try:
            features = pd.read_csv("enhanced_features.csv")
            print("Using enhanced features dataset")
        except:
            features = pd.read_csv("features.csv")
            print("Using original features dataset")
    else:
        features = pd.read_csv("features.csv")
        print("Using original features dataset")
    
    X = features.iloc[:, :-1].values
    y = features['labels'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y_encoded))}")
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, label_encoder

def test_traditional_models(X_train, X_test, y_train, y_test):
    """Test traditional ML models"""

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, kernel='rbf'),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))

    return results


def main():
    """Main comparison function"""

    print("="*60)
    print("TRADITIONAL ML MODEL COMPARISON")
    print("="*60)

    # Test with enhanced features first
    print("\n" + "="*40)
    print("TESTING WITH ENHANCED FEATURES")
    print("="*40)

    try:
        X_train, X_test, y_train, y_test, label_encoder = load_and_prepare_data(use_enhanced=True)
        num_classes = len(label_encoder.classes_)

        # Test traditional models only
        enhanced_results = test_traditional_models(X_train, X_test, y_train, y_test)

    except Exception as e:
        print(f"Error with enhanced features: {e}")
        enhanced_results = {}

    # Test with original features
    print("\n" + "="*40)
    print("TESTING WITH ORIGINAL FEATURES")
    print("="*40)

    try:
        X_train, X_test, y_train, y_test, label_encoder = load_and_prepare_data(use_enhanced=False)
        num_classes = len(label_encoder.classes_)

        # Test traditional models only
        original_results = test_traditional_models(X_train, X_test, y_train, y_test)

    except Exception as e:
        print(f"Error with original features: {e}")
        original_results = {}
    
    # Print comprehensive results
    print("\n" + "="*60)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*60)
    
    if enhanced_results:
        print("\nEnhanced Features Results:")
        for model_name, accuracy in enhanced_results.items():
            print(f"  {model_name}: {accuracy:.4f}")
        best_enhanced = max(enhanced_results.values())
        best_enhanced_model = max(enhanced_results.items(), key=lambda x: x[1])[0]
        print(f"  Best Enhanced Model: {best_enhanced_model} ({best_enhanced:.4f})")
    
    if original_results:
        print("\nOriginal Features Results:")
        for model_name, accuracy in original_results.items():
            print(f"  {model_name}: {accuracy:.4f}")
        best_original = max(original_results.values())
        best_original_model = max(original_results.items(), key=lambda x: x[1])[0]
        print(f"  Best Original Model: {best_original_model} ({best_original:.4f})")
    
    # Calculate improvements
    if enhanced_results and original_results:
        improvement = (best_enhanced - best_original) * 100
        print(f"\nImprovement from enhancements: {improvement:+.2f} percentage points")
        
        print("\nDetailed Comparison:")
        for model_name in original_results.keys():
            if model_name in enhanced_results:
                orig_acc = original_results[model_name]
                enh_acc = enhanced_results[model_name]
                diff = (enh_acc - orig_acc) * 100
                print(f"  {model_name}: {orig_acc:.4f} → {enh_acc:.4f} ({diff:+.2f}pp)")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()