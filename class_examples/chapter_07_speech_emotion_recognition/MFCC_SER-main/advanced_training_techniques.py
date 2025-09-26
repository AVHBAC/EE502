import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise
from keras.callbacks import (Callback, ReduceLROnPlateau, EarlyStopping, 
                           ModelCheckpoint, LearningRateScheduler)
from keras.optimizers import Adam, AdamW, RMSprop
from keras.regularizers import l1, l2, l1_l2
import keras.backend as K
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class AdvancedRegularizationTechniques:
    """Advanced regularization and training techniques"""
    
    @staticmethod
    def mixup_generator(X, y, alpha=0.2, batch_size=32):
        """Mixup data augmentation generator"""
        while True:
            indices = np.random.permutation(X.shape[0])
            
            for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
                end_idx = min(start_idx + batch_size, X.shape[0])
                batch_indices = indices[start_idx:end_idx]
                
                if len(batch_indices) < batch_size:
                    continue
                
                # Get batch
                batch_x = X[batch_indices]
                batch_y = y[batch_indices]
                
                # Apply mixup
                lam = np.random.beta(alpha, alpha, batch_size)
                lam = np.maximum(lam, 1 - lam)  # Ensure lambda >= 0.5
                
                # Shuffle for mixing
                shuffle_indices = np.random.permutation(batch_size)
                
                # Mix samples
                mixed_x = np.zeros_like(batch_x)
                mixed_y = np.zeros_like(batch_y)
                
                for i in range(batch_size):
                    mixed_x[i] = lam[i] * batch_x[i] + (1 - lam[i]) * batch_x[shuffle_indices[i]]
                    mixed_y[i] = lam[i] * batch_y[i] + (1 - lam[i]) * batch_y[shuffle_indices[i]]
                
                yield mixed_x, mixed_y
    
    @staticmethod
    def cutmix_generator(X, y, alpha=1.0, batch_size=32):
        """CutMix data augmentation for 1D audio features"""
        while True:
            indices = np.random.permutation(X.shape[0])
            
            for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
                end_idx = min(start_idx + batch_size, X.shape[0])
                batch_indices = indices[start_idx:end_idx]
                
                if len(batch_indices) < batch_size:
                    continue
                
                batch_x = X[batch_indices].copy()
                batch_y = y[batch_indices].copy()
                
                # Apply CutMix
                shuffle_indices = np.random.permutation(batch_size)
                lam = np.random.beta(alpha, alpha)
                
                # Choose random region to cut
                seq_len = batch_x.shape[1]
                cut_len = int(seq_len * np.sqrt(1 - lam))
                cut_start = np.random.randint(0, seq_len - cut_len + 1)
                cut_end = cut_start + cut_len
                
                # Mix features
                for i in range(batch_size):
                    batch_x[i, cut_start:cut_end] = batch_x[shuffle_indices[i], cut_start:cut_end]
                    batch_y[i] = lam * batch_y[i] + (1 - lam) * batch_y[shuffle_indices[i]]
                
                yield batch_x, batch_y
    
    @staticmethod
    def label_smoothing_loss(y_true, y_pred, smoothing=0.1):
        """Label smoothing for regularization"""
        num_classes = tf.shape(y_pred)[-1]
        smoothed_labels = y_true * (1 - smoothing) + smoothing / tf.cast(num_classes, tf.float32)
        return tf.keras.losses.categorical_crossentropy(smoothed_labels, y_pred)

class CustomCallbacks:
    """Custom callbacks for advanced training"""
    
    class CosineAnnealingScheduler(Callback):
        """Cosine annealing learning rate scheduler"""
        
        def __init__(self, min_lr=1e-7, max_lr=1e-2, steps_per_epoch=None, epochs=None, cycle_length=None):
            super().__init__()
            self.min_lr = min_lr
            self.max_lr = max_lr
            self.steps_per_epoch = steps_per_epoch
            self.epochs = epochs
            self.cycle_length = cycle_length or epochs
            
        def on_batch_end(self, batch, logs=None):
            if not hasattr(self.model.optimizer, 'learning_rate'):
                raise ValueError('Optimizer must have a "learning_rate" attribute.')
            
            step = self.steps_per_epoch * self.epoch + batch
            cycle = np.floor(1 + step / (2 * self.steps_per_epoch * self.cycle_length))
            x = np.abs(step / (self.steps_per_epoch * self.cycle_length) - 2 * cycle + 1)
            lr = self.min_lr + (self.max_lr - self.min_lr) * max(0, (1 - x)) * 0.5
            
            K.set_value(self.model.optimizer.learning_rate, lr)
    
    class WarmUpScheduler(Callback):
        """Learning rate warm-up scheduler"""
        
        def __init__(self, warmup_steps, target_lr, steps_per_epoch):
            super().__init__()
            self.warmup_steps = warmup_steps
            self.target_lr = target_lr
            self.steps_per_epoch = steps_per_epoch
            self.step_count = 0
            
        def on_batch_end(self, batch, logs=None):
            self.step_count += 1
            if self.step_count <= self.warmup_steps:
                lr = self.target_lr * (self.step_count / self.warmup_steps)
                K.set_value(self.model.optimizer.learning_rate, lr)
    
    class GradientClipping(Callback):
        """Gradient clipping callback"""
        
        def __init__(self, clip_norm=1.0):
            super().__init__()
            self.clip_norm = clip_norm
            
        def on_batch_end(self, batch, logs=None):
            # This is handled by the optimizer in modern Keras/TensorFlow
            pass

class AdvancedModelTrainer:
    """Advanced model trainer with regularization techniques"""
    
    def __init__(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        
        # Calculate class weights for imbalanced data
        y_integers = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_integers), y=y_integers
        )
        self.class_weight_dict = dict(enumerate(class_weights))
    
    def compile_with_advanced_optimizer(self, optimizer_type='adamw', learning_rate=0.001):
        """Compile model with advanced optimizer"""
        
        if optimizer_type == 'adamw':
            optimizer = AdamW(
                learning_rate=learning_rate,
                weight_decay=0.01,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                clipnorm=1.0  # Gradient clipping
            )
        elif optimizer_type == 'adam':
            optimizer = Adam(
                learning_rate=learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                clipnorm=1.0
            )
        else:
            optimizer = RMSprop(
                learning_rate=learning_rate,
                momentum=0.9,
                clipnorm=1.0
            )
        
        # Compile with label smoothing
        self.model.compile(
            optimizer=optimizer,
            loss=lambda y_true, y_pred: AdvancedRegularizationTechniques.label_smoothing_loss(y_true, y_pred, 0.1),
            metrics=['accuracy']
        )
    
    def create_advanced_callbacks(self, monitor='val_accuracy', patience=15):
        """Create advanced callbacks for training"""
        
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                mode='max' if 'acc' in monitor else 'min',
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                'best_advanced_model.h5',
                monitor=monitor,
                save_best_only=True,
                mode='max' if 'acc' in monitor else 'min',
                verbose=1
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Cosine annealing (if available)
            CustomCallbacks.CosineAnnealingScheduler(
                min_lr=1e-7,
                max_lr=0.001,
                steps_per_epoch=len(self.X_train) // 32,
                epochs=100,
                cycle_length=30
            )
        ]
        
        return callbacks
    
    def train_with_mixup(self, epochs=100, batch_size=32, mixup_alpha=0.2):
        """Train model with MixUp augmentation"""
        
        print("Training with MixUp augmentation...")
        
        # Compile model
        self.compile_with_advanced_optimizer('adamw', 0.001)
        
        # Create callbacks
        callbacks = self.create_advanced_callbacks()
        
        # Create MixUp generator
        train_generator = AdvancedRegularizationTechniques.mixup_generator(
            self.X_train, self.y_train, mixup_alpha, batch_size
        )
        
        steps_per_epoch = len(self.X_train) // batch_size
        
        # Train model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            class_weight=self.class_weight_dict,
            verbose=1
        )
        
        return history
    
    def train_with_cutmix(self, epochs=100, batch_size=32, cutmix_alpha=1.0):
        """Train model with CutMix augmentation"""
        
        print("Training with CutMix augmentation...")
        
        # Compile model
        self.compile_with_advanced_optimizer('adamw', 0.001)
        
        # Create callbacks
        callbacks = self.create_advanced_callbacks()
        
        # Create CutMix generator
        train_generator = AdvancedRegularizationTechniques.cutmix_generator(
            self.X_train, self.y_train, cutmix_alpha, batch_size
        )
        
        steps_per_epoch = len(self.X_train) // batch_size
        
        # Train model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            class_weight=self.class_weight_dict,
            verbose=1
        )
        
        return history
    
    def train_with_cross_validation(self, n_splits=5, epochs=80):
        """Train with stratified cross-validation"""
        
        print(f"Training with {n_splits}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        fold_histories = []
        
        # Combine train and validation for CV
        X_full = np.vstack([self.X_train, self.X_val])
        y_full = np.vstack([self.y_train, self.y_val])
        y_integers = np.argmax(y_full, axis=1)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_integers)):
            print(f"\nTraining fold {fold + 1}/{n_splits}...")
            
            # Split data
            X_train_fold = X_full[train_idx]
            y_train_fold = y_full[train_idx]
            X_val_fold = X_full[val_idx]
            y_val_fold = y_full[val_idx]
            
            # Clone model for this fold
            fold_model = keras.models.clone_model(self.model)
            fold_model.set_weights(self.model.get_weights())
            
            # Compile
            fold_model.compile(
                optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
            ]
            
            history = fold_model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate fold
            val_accuracy = max(history.history['val_accuracy'])
            cv_scores.append(val_accuracy)
            fold_histories.append(history)
            
            print(f"Fold {fold + 1} validation accuracy: {val_accuracy:.4f}")
        
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        print(f"\nCross-validation results:")
        print(f"Mean accuracy: {mean_cv_score:.4f} (+/- {std_cv_score * 2:.4f})")
        
        return cv_scores, fold_histories
    
    def progressive_resizing_training(self, size_schedule=[(0.5, 30), (0.75, 30), (1.0, 40)]):
        """Progressive resizing training strategy"""
        
        print("Training with progressive resizing...")
        
        histories = []
        
        for size_factor, epochs in size_schedule:
            print(f"\nTraining with {size_factor*100:.0f}% data size for {epochs} epochs...")
            
            # Sample data
            n_samples = int(len(self.X_train) * size_factor)
            indices = np.random.choice(len(self.X_train), n_samples, replace=False)
            
            X_train_subset = self.X_train[indices]
            y_train_subset = self.y_train[indices]
            
            # Compile model
            self.compile_with_advanced_optimizer('adamw', 0.001 * size_factor)
            
            # Train
            callbacks = self.create_advanced_callbacks(patience=int(10/size_factor))
            
            history = self.model.fit(
                X_train_subset, y_train_subset,
                validation_data=(self.X_val, self.y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
                class_weight=self.class_weight_dict,
                verbose=1
            )
            
            histories.append(history)
        
        return histories
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        
        print("\nEvaluating model...")
        
        # Test accuracy
        test_loss, test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        # Predictions
        y_pred = self.model.predict(self.X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(self.y_test, axis=1)
        
        # Classification report
        from sklearn.metrics import classification_report, confusion_matrix
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test_classes, y_pred_classes))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        
        return {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'predictions': y_pred,
            'confusion_matrix': cm
        }

def create_regularized_model(input_shape, num_classes=8):
    """Create a well-regularized model"""
    
    model = Sequential([
        # Input noise for regularization
        GaussianNoise(0.1, input_shape=input_shape),
        
        # First block
        Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        BatchNormalization(),
        Dropout(0.4),
        
        # Second block
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        
        # Third block
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.4),
        
        # Fourth block
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    """Main function to demonstrate advanced training techniques"""
    
    # Note: This assumes you have already created enhanced_features.csv
    try:
        features = pd.read_csv("enhanced_features.csv")
    except FileNotFoundError:
        print("Please run advanced_data_augmentation.py first to create enhanced_features.csv")
        return
    
    # Prepare data (similar to previous scripts)
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
    from sklearn.model_selection import train_test_split
    
    X = features.iloc[:, :-1].values
    y = features['labels'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y_encoded.reshape(-1, 1))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.15, random_state=42, stratify=y_onehot
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Reshape for models
    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    print("="*60)
    print("ADVANCED TRAINING TECHNIQUES")
    print("="*60)
    
    # Create regularized model
    model = create_regularized_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=y_train.shape[1]
    )
    
    # Initialize trainer
    trainer = AdvancedModelTrainer(model, X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Train with different techniques
    print("\n1. Training with MixUp augmentation...")
    mixup_history = trainer.train_with_mixup(epochs=80, mixup_alpha=0.2)
    mixup_results = trainer.evaluate_model()
    
    # Reset model for next experiment
    model = create_regularized_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])
    trainer.model = model
    
    print("\n2. Training with Cross-Validation...")
    cv_scores, cv_histories = trainer.train_with_cross_validation(n_splits=5, epochs=60)
    
    print("\nFinal Results:")
    print(f"MixUp Training Test Accuracy: {mixup_results['test_accuracy']:.4f}")
    print(f"Cross-Validation Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")

if __name__ == "__main__":
    main()