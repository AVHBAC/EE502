import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def test_original_features():
    """Test original features for comparison"""
    
    print("="*50)
    print("ORIGINAL FEATURES BASELINE TEST")
    print("="*50)
    
    # Load original features
    features = pd.read_csv("features.csv")
    print("✓ Using original features dataset")
    
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
    
    print(f"✓ Dataset shape: {X.shape}")
    print(f"✓ Classes: {len(label_encoder.classes_)}")
    print(f"✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    results = {}
    
    # 1. Random Forest (baseline)
    print(f"\n1. Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    results['Random Forest'] = rf_acc
    print(f"   Random Forest Accuracy: {rf_acc:.4f}")
    
    # 2. Logistic Regression
    print(f"\n2. Training Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    results['Logistic Regression'] = lr_acc
    print(f"   Logistic Regression Accuracy: {lr_acc:.4f}")
    
    print(f"\n" + "="*50)
    print("ORIGINAL FEATURES RESULTS")
    print("="*50)
    
    best_original = max(results.values())
    best_model = max(results.items(), key=lambda x: x[1])[0]
    
    print(f"Best original model: {best_model} ({best_original:.4f})")
    
    for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {acc:.4f}")
    
    return results, best_original

if __name__ == "__main__":
    original_results, best_original = test_original_features()
    
    # Compare with enhanced results
    enhanced_best = 0.7417  # From previous test
    improvement = (enhanced_best - best_original) * 100
    
    print(f"\n" + "="*50)
    print("IMPROVEMENT ANALYSIS")
    print("="*50)
    print(f"Original features best: {best_original:.4f}")
    print(f"Enhanced features best: {enhanced_best:.4f}")
    print(f"Improvement: {improvement:+.2f} percentage points")