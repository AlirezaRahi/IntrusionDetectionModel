# 7. main.py
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project path
project_root = r'C:\Alex\Projects\ai_security_env\Intrusion Detection Model'
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model import IntrusionDetectionModel
from src.train import Trainer
from src.evaluate import Evaluator

import numpy as np
import tensorflow as tf

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    print("=" * 60)
    print("Starting Multi-class Intrusion Detection Project with CNN + LSTM")
    print("=" * 60)
    
    # Paths
    data_path = r'C:\Users\alire\Downloads\datasets'
    processed_path = r'C:\Alex\Projects\ai_security_env\Intrusion Detection Model\data\processed'
    checkpoint_path = r'C:\Alex\Projects\ai_security_env\Intrusion Detection Model\data\checkpoints'
    
    # Create required folders
    os.makedirs(processed_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    try:
        # 1. Data Preprocessing
        print("\n[1/6] Data Preprocessing...")
        preprocessor = DataPreprocessor(data_path)
        
        # Load and combine datasets
        df = preprocessor.load_and_combine_datasets()
        
        # Preprocess
        X, y = preprocessor.preprocess_data(df)
        
        # If no data exists, raise error
        if len(X) == 0 or len(y) == 0:
            raise Exception("After preprocessing, no data remains!")
        
        # Split and scale
        (X_train, X_val, X_test), (y_train, y_val, y_test) = preprocessor.split_and_scale(X, y)
        
        # 2. Feature Engineering
        print("\n[2/6] Feature Engineering...")
        feature_engineer = FeatureEngineer()
        
        # Save preprocessors
        preprocessor.save_artifacts(processed_path)
        
        # 3. Prepare Sequences for CNN + LSTM
        print("\n[3/6] Preparing Time Sequences...")
        sequence_length = min(10, len(X_train) // 10)  # Adjust sequence length based on data
        if sequence_length < 2:
            sequence_length = 2
        
        print(f"Selected sequence length: {sequence_length}")
        
        # Create sequences
        trainer = Trainer(None, preprocessor, checkpoint_path)
        
        # Check if we have enough data for sequences
        if len(X_train) < sequence_length:
            raise Exception(f"Insufficient data to create sequences of length {sequence_length}!")
        
        X_train_seq = trainer.prepare_sequences(X_train, sequence_length)
        X_val_seq = trainer.prepare_sequences(X_val, sequence_length)
        X_test_seq = trainer.prepare_sequences(X_test, sequence_length)
        
        # Adjust label lengths
        y_train_seq = y_train[sequence_length-1:]
        y_val_seq = y_val[sequence_length-1:]
        y_test_seq = y_test[sequence_length-1:]
        
        print(f"Training data shape: {X_train_seq.shape}")
        print(f"Validation data shape: {X_val_seq.shape}")
        print(f"Test data shape: {X_test_seq.shape}")
        
        # 4. Build Model
        print("\n[4/6] Building CNN + LSTM Model...")
        num_classes = len(np.unique(y))
        input_shape = X_train.shape[1]
        
        print(f"Number of classes: {num_classes}")
        print(f"Number of features: {input_shape}")
        
        model = IntrusionDetectionModel(
            input_shape=input_shape,
            num_classes=num_classes,
            sequence_length=sequence_length
        )
        model.build_model()
        
        # 5. Train Model
        print("\n[5/6] Training Model...")
        trainer.model = model
        history = trainer.train(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            epochs=30,  # You can increase this number
            batch_size=32
        )
        
        # Plot training graphs
        trainer.plot_training_history()
        
        # 6. Evaluate Model
        print("\n[6/6] Evaluating Model...")
        test_accuracy, y_pred = trainer.evaluate(X_test_seq, y_test_seq)
        
        # In-depth analysis
        evaluator = Evaluator(model, preprocessor)
        
        # Plot ROC curves
        evaluator.plot_roc_curves(X_test_seq, y_test_seq)
        
        # Analyze predictions
        misclassified, class_accuracy = evaluator.analyze_predictions(
            X_test_seq, y_test_seq, y_pred
        )
        
        print("\n" + "=" * 60)
        print(f"Project Completed! Best Test Accuracy: {test_accuracy:.4f}")
        print("=" * 60)
        
        # Save final results
        print("\nSaving final results...")
        with open(os.path.join(checkpoint_path, 'final_results.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            f.write("\nClass-wise Accuracy:\n")
            for class_name, acc in class_accuracy.items():
                f.write(f"{class_name}: {acc:.4f}\n")
        
        print("Project completed successfully!")
        
    except Exception as e:
        print(f"\nError in program execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()