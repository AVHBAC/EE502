import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

def quick_test():
    """Quick test of different models"""
    
    print("="*50)
    print("QUICK MODEL PERFORMANCE TEST")
    print("="*50)
    
    # Load data
    try:
        features = pd.read_csv("enhanced_features.csv")
        print("✓ Using enhanced features dataset")
        dataset_type = "enhanced"
    except:
        features = pd.read_csv("features.csv") 
        print("✓ Using original features dataset")
        dataset_type = "original"
    
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
    
    # 3. Simple Neural Network
    print(f"\n3. Training Simple Neural Network...")
    
    from keras.utils import to_categorical
    y_train_cat = to_categorical(y_train, len(label_encoder.classes_))
    y_test_cat = to_categorical(y_test, len(label_encoder.classes_))
    
    nn_model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    nn_model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=20,  # Reduced for speed
        batch_size=32,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )
    
    _, nn_acc = nn_model.evaluate(X_test, y_test_cat, verbose=0)
    results['Simple Neural Network'] = nn_acc
    print(f"   Simple Neural Network Accuracy: {nn_acc:.4f}")
    
    # 4. CNN (if we have enough features)
    if X_train.shape[1] > 50:
        print(f"\n4. Training CNN...")
        
        X_train_cnn = np.expand_dims(X_train, axis=2)
        X_test_cnn = np.expand_dims(X_test, axis=2)
        
        cnn_model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(len(label_encoder.classes_), activation='softmax')
        ])
        
        cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        cnn_model.fit(
            X_train_cnn, y_train_cat,
            validation_data=(X_test_cnn, y_test_cat),
            epochs=15,  # Reduced for speed
            batch_size=32,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )
        
        _, cnn_acc = cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)
        results['CNN'] = cnn_acc
        print(f"   CNN Accuracy: {cnn_acc:.4f}")
    
    # Print summary
    print(f"\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Dataset used: {dataset_type} features")
    print(f"Best model performance:")
    
    best_model = max(results.items(), key=lambda x: x[1])
    print(f"✓ {best_model[0]}: {best_model[1]:.4f}")
    
    print(f"\nAll results:")
    for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {acc:.4f}")
    
    # Estimate baseline comparison
    baseline_estimate = 0.65  # Typical baseline for emotion recognition
    improvement = (best_model[1] - baseline_estimate) * 100
    print(f"\nEstimated improvement over baseline: {improvement:+.2f} percentage points")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    quick_test()