import pandas as pd
import numpy as np
import optuna
from optuna_integration import KerasPruningCallback
import keras
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def create_model(trial, input_shape, num_classes):
    """Create model with trial hyperparameters"""
    
    # Suggest hyperparameters
    n_conv_layers = trial.suggest_int('n_conv_layers', 2, 3)
    n_dense_layers = trial.suggest_int('n_dense_layers', 1, 2)
    
    # CNN hyperparameters
    conv1_filters = trial.suggest_categorical('conv1_filters', [64, 128, 256])
    conv2_filters = trial.suggest_categorical('conv2_filters', [64, 128])
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
    
    # Dense layer hyperparameters
    dense1_units = trial.suggest_categorical('dense1_units', [32, 64, 128])
    
    # Regularization
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.5)
    
    # Optimizer hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    
    # Training hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    # Build model
    model = Sequential()
    
    # First conv layer
    model.add(Conv1D(filters=conv1_filters, kernel_size=kernel_size, 
                     activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))
    
    # Second conv layer
    model.add(Conv1D(filters=conv2_filters, kernel_size=kernel_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    # Optional third conv layer
    if n_conv_layers >= 3:
        conv3_filters = trial.suggest_categorical('conv3_filters', [32, 64])
        model.add(Conv1D(filters=conv3_filters, kernel_size=kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
    
    # Flatten
    model.add(Flatten())
    
    # Dense layers
    model.add(Dense(dense1_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    if n_dense_layers >= 2:
        dense2_units = trial.suggest_categorical('dense2_units', [16, 32])
        model.add(Dense(dense2_units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model, batch_size

def objective(trial):
    """Objective function for Optuna optimization"""
    try:
        # Load data
        features = pd.read_csv("enhanced_features.csv")
        X = features.iloc[:, :-1].values
        y = features['labels'].values
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = encoder.fit_transform(y_encoded.reshape(-1, 1))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Reshape for CNN
        X_train = np.expand_dims(X_train, axis=2)
        X_val = np.expand_dims(X_val, axis=2)
        
        # Create model
        model, batch_size = create_model(trial, (X_train.shape[1], 1), y_train.shape[1])
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
            KerasPruningCallback(trial, 'val_accuracy')
        ]
        
        # Train model
        epochs = 20  # Reduced for speed
        
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        # Get best validation accuracy
        best_val_accuracy = max(history.history['val_accuracy'])
        return best_val_accuracy
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return 0.0

def run_quick_optimization():
    """Run quick Optuna optimization"""
    print("Starting Quick Optuna optimization...")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
    )
    
    # Optimize
    n_trials = 10  # Reduced for speed
    study.optimize(objective, n_trials=n_trials, timeout=600)  # 10 minutes max
    
    print("\n" + "="*60)
    print("OPTUNA OPTIMIZATION RESULTS")
    print("="*60)
    
    # Best trial info
    best_trial = study.best_trial
    print(f"Best validation accuracy: {best_trial.value:.4f}")
    print(f"Number of finished trials: {len(study.trials)}")
    
    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    return study, best_trial

def train_final_model(best_params):
    """Train final model with best parameters"""
    print("\nTraining final model with best parameters...")
    
    # Load and prepare data
    features = pd.read_csv("enhanced_features.csv")
    X = features.iloc[:, :-1].values
    y = features['labels'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y_encoded.reshape(-1, 1))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reshape for CNN
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    # Create model with best parameters (simplified)
    model = Sequential([
        Conv1D(filters=best_params.get('conv1_filters', 128), 
               kernel_size=best_params.get('kernel_size', 5), 
               activation='relu', input_shape=(X_train.shape[1], 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(best_params.get('dropout_rate', 0.3)),
        
        Conv1D(filters=best_params.get('conv2_filters', 64), 
               kernel_size=best_params.get('kernel_size', 5), 
               activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Flatten(),
        Dense(best_params.get('dense1_units', 64), activation='relu'),
        Dropout(best_params.get('dropout_rate', 0.3)),
        Dense(y_train.shape[1], activation='softmax')
    ])
    
    # Compile
    optimizer = Adam(learning_rate=best_params.get('learning_rate', 0.001))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=best_params.get('batch_size', 32),
        epochs=40,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    model.save('quick_optimized_model.h5')
    print("Model saved as 'quick_optimized_model.h5'")
    
    return model, test_accuracy, history

if __name__ == "__main__":
    # Run optimization
    study, best_trial = run_quick_optimization()
    
    # Train final model
    final_model, final_accuracy, history = train_final_model(best_trial.params)
    
    print("\n" + "="*60)
    print("QUICK OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"Best validation accuracy from search: {best_trial.value:.4f}")
    print(f"Final test accuracy: {final_accuracy:.4f}")
    print(f"Total trials completed: {len(study.trials)}")
    
    # Compare with baseline (assuming ~70% baseline)
    baseline_accuracy = 0.70
    improvement = (final_accuracy - baseline_accuracy) * 100
    print(f"Estimated improvement over baseline: {improvement:+.2f} percentage points")