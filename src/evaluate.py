# 6. src/evaluate.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, model, data_preprocessor):
        self.model = model
        self.data_preprocessor = data_preprocessor
    
    def plot_roc_curves(self, X_test_seq, y_test):
        """
        Plot ROC curves for each class
        """
        # Convert to one-hot format
        y_test_bin = label_binarize(y_test, classes=range(self.model.num_classes))
        
        # Predict probabilities
        y_score = self.model.model.predict(X_test_seq)
        
        # Calculate ROC and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.model.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot curves
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.tab20(np.linspace(0, 1, self.model.num_classes))
        target_names = self.data_preprocessor.label_encoder.classes_
        
        for i, color in zip(range(self.model.num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{target_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Multi-class Classification')
        plt.legend(loc="lower right", bbox_to_anchor=(1.4, 0))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('roc_curves.png')
        plt.show()
    
    def analyze_predictions(self, X_test_seq, y_test, y_pred):
        """
        In-depth analysis of predictions
        """
        # Identify misclassified samples
        misclassified = np.where(y_test != y_pred)[0]
        
        if len(misclassified) > 0:
            print(f"\nNumber of misclassified samples: {len(misclassified)}")
            
            # Check error distribution
            error_distribution = pd.crosstab(
                pd.Series(y_test[misclassified], name='Actual'),
                pd.Series(y_pred[misclassified], name='Predicted')
            )
            
            print("\nError distribution:")
            print(error_distribution)
        
        # Calculate per-class accuracy
        class_accuracy = {}
        target_names = self.data_preprocessor.label_encoder.classes_
        
        for i in range(self.model.num_classes):
            class_indices = np.where(y_test == i)[0]
            if len(class_indices) > 0:
                correct = np.sum(y_test[class_indices] == y_pred[class_indices])
                class_accuracy[target_names[i]] = correct / len(class_indices)
        
        print("\nPer-class accuracy:")
        for class_name, acc in class_accuracy.items():
            print(f"{class_name}: {acc:.4f}")
        
        return misclassified, class_accuracy