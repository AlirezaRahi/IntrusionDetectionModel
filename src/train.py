# 5. src/train.py
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class Trainer:
    def __init__(self, model, data_preprocessor, checkpoint_dir):
        """
        Model training class
        """
        self.model = model
        self.data_preprocessor = data_preprocessor
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = None
        
    def prepare_sequences(self, X_scaled, sequence_length):
        """
        Prepare sequences for model input
        """
        n_samples = len(X_scaled) - sequence_length + 1
        X_sequence = np.zeros((n_samples, sequence_length, X_scaled.shape[1]))
        
        for i in range(n_samples):
            X_sequence[i] = X_scaled[i:i+sequence_length]
        
        return X_sequence
    
    def train(self, X_train_seq, y_train, X_val_seq, y_val, epochs=50, batch_size=32):
        """
        Train the model
        """
        print("Starting model training...")
        
        # تغییر پسوند به .h5 برای سازگاری
        checkpoint_path = self.checkpoint_dir / 'best_model.h5'
        callbacks = self.model.get_callbacks(str(checkpoint_path))
        
        self.history = self.model.model.fit(
            X_train_seq, y_train,
            validation_data=(X_val_seq, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best model
        if checkpoint_path.exists():
            self.model.model = tf.keras.models.load_model(checkpoint_path)
        
        return self.history
    
    def plot_training_history(self):
        """
        Plot training graphs
        """
        if self.history is None:
            print("No model has been trained yet!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_history.png')
        plt.show()
        
    def evaluate(self, X_test_seq, y_test):
        """
        Evaluate the model on test data
        """
        print("\nEvaluating model on test data...")
        
        # Evaluation
        test_loss, test_accuracy = self.model.model.evaluate(X_test_seq, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Prediction
        y_pred_prob = self.model.model.predict(X_test_seq)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        target_names = self.data_preprocessor.label_encoder.classes_
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'confusion_matrix.png')
        plt.show()
        
        return test_accuracy, y_pred