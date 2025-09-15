import pandas as pd
import numpy as np
import optuna
from optuna.integration import KerasPruningCallback
import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
import joblib

warnings.filterwarnings("ignore")

class OptunaHyperparameterTuner:
    def __init__(self, features_path="features.csv"):
        """Initialize the Optuna hyperparameter tuner"""
        print("Loading and preparing data...")
        self.features = pd.read_csv(features_path)
        self.X = self.features.iloc[:, :-1].values
        self.Y = self.features['labels'].values
        
        # Encode labels
        self.encoder = OneHotEncoder()
        self.Y = self.encoder.fit_transform(np.array(self.Y).reshape(-1, 1)).toarray()
        
        # Split data with stratification
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=42, shuffle=True, stratify=self.Y
        )
        
        # Further split training data for validation
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train, self.y_train, test_size=0.2, random_state=42, shuffle=True, stratify=self.y_train
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_val = self.scaler.transform(self.x_val)
        self.x_test = self.scaler.transform(self.x_test)
        
        # Reshape for CNN
        self.x_train = np.expand_dims(self.x_train, axis=2)
        self.x_val = np.expand_dims(self.x_val, axis=2)
        self.x_test = np.expand_dims(self.x_test, axis=2)
        
        print(f"Data shapes - Train: {self.x_train.shape}, Val: {self.x_val.shape}, Test: {self.x_test.shape}")
        print(f"Number of classes: {self.y_train.shape[1]}")
        
    def create_model(self, trial):
        """Create model with hyperparameters suggested by Optuna"""
        
        # Suggest hyperparameters
        n_conv_layers = trial.suggest_int('n_conv_layers', 2, 4)
        n_dense_layers = trial.suggest_int('n_dense_layers', 1, 3)
        
        # CNN hyperparameters
        conv_filters = []
        for i in range(n_conv_layers):
            filters = trial.suggest_categorical(f'conv_{i}_filters', [64, 128, 256, 512])
            conv_filters.append(filters)
        
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
        pool_size = trial.suggest_categorical('pool_size', [2, 3, 4])
        
        # Dense layer hyperparameters
        dense_units = []
        for i in range(n_dense_layers):
            units = trial.suggest_categorical(f'dense_{i}_units', [16, 32, 64, 128, 256])
            dense_units.append(units)
        
        # Regularization
        conv_dropout = trial.suggest_uniform('conv_dropout', 0.0, 0.5)
        dense_dropout = trial.suggest_uniform('dense_dropout', 0.1, 0.6)
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        
        # Optimizer hyperparameters
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        
        # Training hyperparameters
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Build model
        model = Sequential()
        
        # First conv layer
        model.add(Conv1D(
            filters=conv_filters[0],
            kernel_size=kernel_size,
            activation='relu',
            input_shape=(self.x_train.shape[1], 1)
        ))
        
        if use_batch_norm:
            model.add(BatchNormalization())
        
        model.add(MaxPooling1D(pool_size=pool_size))
        
        if conv_dropout > 0:
            model.add(Dropout(conv_dropout))
        
        # Additional conv layers
        for i in range(1, n_conv_layers):
            model.add(Conv1D(
                filters=conv_filters[i],
                kernel_size=kernel_size,
                activation='relu'
            ))
            
            if use_batch_norm:
                model.add(BatchNormalization())
            
            model.add(MaxPooling1D(pool_size=2))
            
            # Add dropout every other layer
            if i % 2 == 1 and conv_dropout > 0:
                model.add(Dropout(conv_dropout))
        
        # Flatten
        model.add(Flatten())
        
        # Dense layers
        for i in range(n_dense_layers):
            model.add(Dense(dense_units[i], activation='relu'))
            
            if dense_dropout > 0:
                model.add(Dropout(dense_dropout))
        
        # Output layer
        model.add(Dense(8, activation='softmax'))  # 8 emotion classes
        
        # Compile model with suggested optimizer
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model, batch_size
    
    def objective(self, trial):
        """Objective function for Optuna optimization"""
        try:
            # Create model with trial hyperparameters
            model, batch_size = self.create_model(trial)
            
            # Early stopping and learning rate reduction
            early_stop_patience = trial.suggest_int('early_stop_patience', 5, 15)
            lr_patience = trial.suggest_int('lr_patience', 2, 8)
            lr_factor = trial.suggest_uniform('lr_factor', 0.1, 0.8)
            
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=early_stop_patience,
                    restore_best_weights=True,
                    mode='max'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=lr_factor,
                    patience=lr_patience,
                    min_lr=1e-7,
                    verbose=0
                ),
                KerasPruningCallback(trial, 'val_accuracy')  # Optuna pruning
            ]
            
            # Train model
            epochs = trial.suggest_int('epochs', 20, 100)
            
            history = model.fit(
                self.x_train, self.y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(self.x_val, self.y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            # Get best validation accuracy
            best_val_accuracy = max(history.history['val_accuracy'])
            
            return best_val_accuracy
            
        except Exception as e:
            print(f"Trial failed with error: {e}")
            return 0.0
    
    def optimize(self, n_trials=50, timeout=7200):  # 2 hours timeout
        """Run Optuna optimization"""
        print(f"Starting Optuna optimization with {n_trials} trials...")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=5
            )
        )
        
        # Optimize
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[self.print_callback]
        )
        
        return study
    
    def print_callback(self, study, trial):
        """Callback to print trial results"""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"Trial {trial.number}: Validation Accuracy = {trial.value:.4f}")
            if study.best_trial.number == trial.number:
                print(f"New best trial! Accuracy: {trial.value:.4f}")
    
    def train_best_model(self, best_params):
        """Train final model with best parameters"""
        print("Training final model with best parameters...")
        
        # Create model with best parameters
        trial = optuna.trial.create_trial(
            params=best_params,
            distributions={},
            value=0.0
        )
        model, batch_size = self.create_model(trial)
        
        # Train with more epochs for final model
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=best_params.get('early_stop_patience', 10),
                restore_best_weights=True,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=best_params.get('lr_factor', 0.5),
                patience=best_params.get('lr_patience', 5),
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Combine train and validation for final training
        x_train_final = np.vstack([self.x_train, self.x_val])
        y_train_final = np.vstack([self.y_train, self.y_val])
        
        history = model.fit(
            x_train_final, y_train_final,
            batch_size=batch_size,
            epochs=best_params.get('epochs', 80),
            validation_data=(self.x_test, self.y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate final model
        test_loss, test_accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
        print(f"Final Test Loss: {test_loss:.4f}")
        
        return model, history, test_accuracy
    
    def analyze_study(self, study):
        """Analyze and visualize study results"""
        print("\n" + "="*60)
        print("OPTUNA OPTIMIZATION RESULTS")
        print("="*60)
        
        # Best trial info
        best_trial = study.best_trial
        print(f"Best validation accuracy: {best_trial.value:.4f}")
        print(f"Best trial number: {best_trial.number}")
        
        print("\nBest hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        
        # Study statistics
        print(f"\nStudy statistics:")
        print(f"  Number of finished trials: {len(study.trials)}")
        print(f"  Number of pruned trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED]))}")
        print(f"  Number of complete trials: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))}")
        
        # Save study
        joblib.dump(study, 'optuna_study.pkl')
        print("\nStudy saved as 'optuna_study.pkl'")
        
        # Save results to CSV
        df = study.trials_dataframe()
        df.to_csv('optuna_trials.csv', index=False)
        print("Trial results saved as 'optuna_trials.csv'")
        
        return best_trial.params
    
    def plot_optimization_history(self, study):
        """Plot optimization history"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Optimization history
            optuna.visualization.matplotlib.plot_optimization_history(study, ax=axes[0])
            axes[0].set_title('Optimization History')
            
            # Parameter importances
            optuna.visualization.matplotlib.plot_param_importances(study, ax=axes[1])
            axes[1].set_title('Hyperparameter Importances')
            
            plt.tight_layout()
            plt.savefig('optuna_optimization_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Parallel coordinate plot
            fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            fig.write_image('optuna_parallel_coordinate.png')
            
        except ImportError:
            print("Install plotly for visualization: pip install plotly")

def main():
    """Main function to run Optuna hyperparameter optimization"""
    
    # Initialize tuner
    tuner = OptunaHyperparameterTuner()
    
    # Run optimization
    study = tuner.optimize(n_trials=30, timeout=3600)  # 1 hour timeout, 30 trials
    
    # Analyze results
    best_params = tuner.analyze_study(study)
    
    # Train final model with best parameters
    best_model, history, final_accuracy = tuner.train_best_model(best_params)
    
    # Save best model
    best_model.save('optuna_best_emotion_model.h5')
    print(f"\nBest model saved as 'optuna_best_emotion_model.h5'")
    
    # Plot results
    tuner.plot_optimization_history(study)
    
    print("\n" + "="*60)
    print(f"FINAL RESULTS")
    print("="*60)
    print(f"Best validation accuracy from optimization: {study.best_value:.4f}")
    print(f"Final test accuracy: {final_accuracy:.4f}")
    print(f"Improvement potential: {(final_accuracy - study.best_value) * 100:.2f}%")
    
    return tuner, study, best_model, final_accuracy

if __name__ == "__main__":
    # Install optuna if not available
    try:
        import optuna
    except ImportError:
        print("Installing Optuna...")
        import subprocess
        subprocess.check_call(["pip", "install", "optuna"])
        import optuna
    
    tuner, study, best_model, final_accuracy = main()