import pandas as pd
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import (Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, 
                         GlobalAveragePooling1D, Flatten, Dropout, 
                         BatchNormalization, LSTM, GRU, Bidirectional,
                         Attention, MultiHeadAttention, LayerNormalization,
                         Add, Concatenate, Input, Reshape)
from keras.optimizers import Adam, AdamW
from keras.callbacks import (ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, 
                           CosineAnnealingScheduler)
from keras.regularizers import l1, l2, l1_l2
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

class AdvancedModelArchitectures:
    """Advanced model architectures for speech emotion recognition"""
    
    def __init__(self, input_shape, num_classes=8):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def create_resnet_1d(self, filters_list=[64, 128, 256], kernel_size=3, dropout_rate=0.3):
        """1D ResNet architecture for audio features"""
        
        def residual_block(x, filters, kernel_size=3, dropout_rate=0.3):
            # Main path
            y = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
            y = BatchNormalization()(y)
            y = Dropout(dropout_rate)(y)
            y = Conv1D(filters, kernel_size, padding='same', activation='relu')(y)
            y = BatchNormalization()(y)
            
            # Skip connection
            if x.shape[-1] != filters:
                x = Conv1D(filters, 1, padding='same')(x)
            
            # Add skip connection
            output = Add()([x, y])
            output = keras.activations.relu(output)
            return output
        
        input_layer = Input(shape=self.input_shape)
        
        # Initial conv layer
        x = Conv1D(64, 7, padding='same', activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = MaxPooling1D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        for filters in filters_list:
            x = residual_block(x, filters, kernel_size, dropout_rate)
            x = MaxPooling1D(2)(x)
        
        # Global pooling and classification
        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        return model
    
    def create_attention_cnn(self, filters=[128, 256, 512], dropout_rate=0.4):
        """CNN with self-attention mechanism"""
        
        input_layer = Input(shape=self.input_shape)
        
        # CNN feature extraction
        x = input_layer
        for i, f in enumerate(filters):
            x = Conv1D(f, 5, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(2)(x)
            if i % 2 == 1:
                x = Dropout(dropout_rate * 0.5)(x)
        
        # Self-attention mechanism
        attention_output = MultiHeadAttention(
            num_heads=8, key_dim=x.shape[-1]//8
        )(x, x)
        attention_output = LayerNormalization()(attention_output)
        
        # Combine CNN and attention features
        combined = Add()([x, attention_output])
        
        # Global pooling
        gap = GlobalAveragePooling1D()(combined)
        gmp = GlobalMaxPooling1D()(combined)
        global_pool = Concatenate()([gap, gmp])
        
        # Classification head
        x = Dense(256, activation='relu')(global_pool)
        x = Dropout(dropout_rate)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        return model
    
    def create_cnn_lstm_attention(self, cnn_filters=[256, 512], lstm_units=128, dropout_rate=0.4):
        """Hybrid CNN-LSTM with attention"""
        
        input_layer = Input(shape=self.input_shape)
        
        # CNN feature extraction
        x = input_layer
        for f in cnn_filters:
            x = Conv1D(f, 5, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling1D(2)(x)
            x = Dropout(dropout_rate * 0.5)(x)
        
        # Bidirectional LSTM
        x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
        x = Dropout(dropout_rate)(x)
        
        # Attention mechanism
        attention = MultiHeadAttention(num_heads=4, key_dim=lstm_units)(x, x)
        attention = LayerNormalization()(attention)
        
        # Combine LSTM and attention
        x = Add()([x, attention])
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Classification layers
        x = Dense(256, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        return model
    
    def create_transformer_model(self, d_model=256, num_heads=8, num_layers=4, dropout_rate=0.3):
        """Transformer-based model for sequence modeling"""
        
        def transformer_encoder(inputs, d_model, num_heads, dropout_rate):
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model//num_heads
            )(inputs, inputs)
            attention_output = Dropout(dropout_rate)(attention_output)
            attention_output = LayerNormalization()(Add()([inputs, attention_output]))
            
            # Feed forward network
            ffn_output = Dense(d_model * 2, activation='relu')(attention_output)
            ffn_output = Dense(d_model)(ffn_output)
            ffn_output = Dropout(dropout_rate)(ffn_output)
            ffn_output = LayerNormalization()(Add()([attention_output, ffn_output]))
            
            return ffn_output
        
        input_layer = Input(shape=self.input_shape)
        
        # Project to model dimension
        x = Dense(d_model)(input_layer)
        
        # Stack transformer encoder layers
        for _ in range(num_layers):
            x = transformer_encoder(x, d_model, num_heads, dropout_rate)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Classification head
        x = Dense(256, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        return model
    
    def create_inception_1d(self, dropout_rate=0.4):
        """1D Inception-like architecture"""
        
        def inception_block(x, filters_1x1, filters_3x3, filters_5x5, filters_pool):
            # 1x1 conv
            conv_1x1 = Conv1D(filters_1x1, 1, padding='same', activation='relu')(x)
            
            # 3x3 conv
            conv_3x3 = Conv1D(filters_3x3, 3, padding='same', activation='relu')(x)
            
            # 5x5 conv
            conv_5x5 = Conv1D(filters_5x5, 5, padding='same', activation='relu')(x)
            
            # Max pooling
            pool = MaxPooling1D(3, strides=1, padding='same')(x)
            pool = Conv1D(filters_pool, 1, padding='same', activation='relu')(pool)
            
            # Concatenate all paths
            output = Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5, pool])
            return output
        
        input_layer = Input(shape=self.input_shape)
        
        # Initial layers
        x = Conv1D(64, 7, padding='same', activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = MaxPooling1D(3, strides=2, padding='same')(x)
        
        # Inception blocks
        x = inception_block(x, 64, 96, 16, 32)
        x = BatchNormalization()(x)
        x = MaxPooling1D(3, strides=2, padding='same')(x)
        
        x = inception_block(x, 128, 128, 32, 64)
        x = BatchNormalization()(x)
        x = MaxPooling1D(3, strides=2, padding='same')(x)
        
        x = inception_block(x, 192, 192, 48, 64)
        x = BatchNormalization()(x)
        
        # Global pooling and classification
        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        return model

class EnsembleModels:
    """Ensemble methods for improved performance"""
    
    def __init__(self, input_shape, num_classes=8):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = []
        self.model_weights = []
    
    def create_diverse_models(self):
        """Create diverse models for ensemble"""
        arch = AdvancedModelArchitectures(self.input_shape, self.num_classes)
        
        models = {
            'resnet_1d': arch.create_resnet_1d(),
            'attention_cnn': arch.create_attention_cnn(),
            'cnn_lstm_attention': arch.create_cnn_lstm_attention(),
            'transformer': arch.create_transformer_model(),
            'inception_1d': arch.create_inception_1d()
        }
        
        return models
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, epochs=80):
        """Train ensemble of models"""
        models = self.create_diverse_models()
        trained_models = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Compile model
            model.compile(
                optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7),
                ModelCheckpoint(f'best_{name}_model.h5', save_best_only=True, monitor='val_accuracy')
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate and store
            val_acc = max(history.history['val_accuracy'])
            print(f"{name} best validation accuracy: {val_acc:.4f}")
            
            trained_models[name] = {
                'model': model,
                'val_accuracy': val_acc,
                'history': history
            }
        
        return trained_models
    
    def create_stacking_ensemble(self, base_models, X_train, y_train, X_val, y_val):
        """Create stacking ensemble with meta-learner"""
        
        # Get base model predictions
        train_predictions = []
        val_predictions = []
        
        for name, model_info in base_models.items():
            model = model_info['model']
            
            # Get predictions for training data (using cross-validation to avoid overfitting)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            train_pred = np.zeros((X_train.shape[0], self.num_classes))
            
            for train_idx, test_idx in skf.split(X_train, np.argmax(y_train, axis=1)):
                fold_model = keras.models.clone_model(model)
                fold_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                fold_model.fit(X_train[train_idx], y_train[train_idx], 
                             epochs=20, batch_size=32, verbose=0)
                train_pred[test_idx] = fold_model.predict(X_train[test_idx], verbose=0)
            
            train_predictions.append(train_pred)
            
            # Get predictions for validation data
            val_pred = model.predict(X_val, verbose=0)
            val_predictions.append(val_pred)
        
        # Stack predictions
        X_train_stack = np.concatenate(train_predictions, axis=1)
        X_val_stack = np.concatenate(val_predictions, axis=1)
        
        # Create meta-learner
        meta_model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_stack.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        meta_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train meta-learner
        meta_model.fit(
            X_train_stack, y_train,
            validation_data=(X_val_stack, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=1
        )
        
        return meta_model, base_models
    
    def ensemble_predict(self, base_models, meta_model, X_test):
        """Make ensemble predictions"""
        # Get base model predictions
        test_predictions = []
        for name, model_info in base_models.items():
            pred = model_info['model'].predict(X_test, verbose=0)
            test_predictions.append(pred)
        
        # Stack predictions
        X_test_stack = np.concatenate(test_predictions, axis=1)
        
        # Meta-learner prediction
        ensemble_pred = meta_model.predict(X_test_stack, verbose=0)
        
        return ensemble_pred
    
    def weighted_average_ensemble(self, base_models, X_test):
        """Weighted average ensemble based on validation performance"""
        predictions = []
        weights = []
        
        for name, model_info in base_models.items():
            pred = model_info['model'].predict(X_test, verbose=0)
            predictions.append(pred)
            weights.append(model_info['val_accuracy'])
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred

class ModelTrainer:
    """Main class for training advanced models"""
    
    def __init__(self, features_path="enhanced_features.csv"):
        self.features_path = features_path
        self.load_and_prepare_data()
    
    def load_and_prepare_data(self):
        """Load and prepare data"""
        print("Loading enhanced features...")
        features = pd.read_csv(self.features_path)
        
        self.X = features.iloc[:, :-1].values
        self.y = features['labels'].values
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(self.y)
        self.encoder = OneHotEncoder(sparse_output=False)
        self.y_onehot = self.encoder.fit_transform(y_encoded.reshape(-1, 1))
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_onehot, test_size=0.15, random_state=42, 
            shuffle=True, stratify=self.y_onehot
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.15, random_state=42,
            shuffle=True, stratify=self.y_train
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Reshape for models
        self.X_train = np.expand_dims(self.X_train, axis=2)
        self.X_val = np.expand_dims(self.X_val, axis=2)
        self.X_test = np.expand_dims(self.X_test, axis=2)
        
        print(f"Data shapes - Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}")
        print(f"Number of classes: {self.y_train.shape[1]}")
    
    def train_single_model(self, model_type='resnet_1d'):
        """Train a single advanced model"""
        arch = AdvancedModelArchitectures(
            input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
            num_classes=self.y_train.shape[1]
        )
        
        if model_type == 'resnet_1d':
            model = arch.create_resnet_1d()
        elif model_type == 'attention_cnn':
            model = arch.create_attention_cnn()
        elif model_type == 'cnn_lstm_attention':
            model = arch.create_cnn_lstm_attention()
        elif model_type == 'transformer':
            model = arch.create_transformer_model()
        else:
            model = arch.create_inception_1d()
        
        model.compile(
            optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True, monitor='val_accuracy'),
            ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-7),
            ModelCheckpoint(f'best_{model_type}_advanced.h5', save_best_only=True, monitor='val_accuracy')
        ]
        
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        
        return model, history, test_acc
    
    def train_ensemble(self):
        """Train ensemble of models"""
        ensemble = EnsembleModels(
            input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
            num_classes=self.y_train.shape[1]
        )
        
        # Train base models
        base_models = ensemble.train_ensemble(
            self.X_train, self.y_train, self.X_val, self.y_val
        )
        
        # Create stacking ensemble
        meta_model, base_models = ensemble.create_stacking_ensemble(
            base_models, self.X_train, self.y_train, self.X_val, self.y_val
        )
        
        # Evaluate ensembles
        print("\nEvaluating ensemble methods...")
        
        # Stacking ensemble
        stacking_pred = ensemble.ensemble_predict(base_models, meta_model, self.X_test)
        stacking_acc = np.mean(np.argmax(stacking_pred, axis=1) == np.argmax(self.y_test, axis=1))
        print(f"Stacking Ensemble Test Accuracy: {stacking_acc:.4f}")
        
        # Weighted average ensemble
        weighted_pred = ensemble.weighted_average_ensemble(base_models, self.X_test)
        weighted_acc = np.mean(np.argmax(weighted_pred, axis=1) == np.argmax(self.y_test, axis=1))
        print(f"Weighted Average Ensemble Test Accuracy: {weighted_acc:.4f}")
        
        return base_models, meta_model, stacking_acc, weighted_acc

def main():
    """Main function to run advanced model training"""
    
    print("="*60)
    print("ADVANCED MODEL TRAINING FOR SPEECH EMOTION RECOGNITION")
    print("="*60)
    
    trainer = ModelTrainer("enhanced_features.csv")
    
    # Train individual advanced models
    model_types = ['resnet_1d', 'attention_cnn', 'cnn_lstm_attention', 'transformer', 'inception_1d']
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*40}")
        print(f"Training {model_type.upper()}")
        print('='*40)
        
        model, history, test_acc = trainer.train_single_model(model_type)
        results[model_type] = test_acc
        
        print(f"{model_type} Test Accuracy: {test_acc:.4f}")
    
    # Train ensemble
    print(f"\n{'='*40}")
    print("TRAINING ENSEMBLE MODELS")
    print('='*40)
    
    base_models, meta_model, stacking_acc, weighted_acc = trainer.train_ensemble()
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    print("\nIndividual Model Results:")
    for model_type, acc in results.items():
        print(f"  {model_type}: {acc:.4f}")
    
    print(f"\nEnsemble Results:")
    print(f"  Stacking Ensemble: {stacking_acc:.4f}")
    print(f"  Weighted Average Ensemble: {weighted_acc:.4f}")
    
    best_single = max(results.values())
    best_ensemble = max(stacking_acc, weighted_acc)
    
    print(f"\nBest Single Model: {best_single:.4f}")
    print(f"Best Ensemble: {best_ensemble:.4f}")
    print(f"Improvement: {(best_ensemble - best_single) * 100:.2f}%")

if __name__ == "__main__":
    main()