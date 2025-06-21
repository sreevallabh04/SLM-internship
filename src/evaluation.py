"""
Evaluation utilities for the text classification pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    classification_report,
    roc_auc_score,
    roc_curve
)
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation utilities"""
    
    def __init__(self, class_names=['Negative', 'Positive']):
        self.class_names = class_names
    
    def compute_metrics(self, y_true, y_pred, y_prob=None):
        """
        Compute comprehensive evaluation metrics
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_prob (array, optional): Prediction probabilities
            
        Returns:
            dict: Evaluation metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'class_metrics': {
                'precision': [float(p) for p in precision],
                'recall': [float(r) for r in recall],
                'f1': [float(f) for f in f1],
                'support': [int(s) for s in support]
            }
        }
        
        # Add AUC if probabilities provided
        if y_prob is not None:
            try:
                auc = roc_auc_score(y_true, y_prob[:, 1])
                metrics['auc'] = float(auc)
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")
        
        return metrics
    
    def create_confusion_matrix(self, y_true, y_pred, save_path=None):
        """
        Create and optionally save confusion matrix visualization
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            save_path (str, optional): Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Sentiment Classification', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
        
        return cm
    
    def create_roc_curve(self, y_true, y_prob, save_path=None):
        """
        Create ROC curve visualization
        
        Args:
            y_true (array): True labels
            y_prob (array): Prediction probabilities
            save_path (str, optional): Path to save the plot
        """
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            auc = roc_auc_score(y_true, y_prob[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"ROC curve saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating ROC curve: {e}")
    
    def generate_classification_report(self, y_true, y_pred, save_path=None):
        """
        Generate detailed classification report
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            save_path (str, optional): Path to save the report
            
        Returns:
            str: Classification report
        """
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names, 
            digits=4
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Classification report saved to {save_path}")
        
        return report
    
    def analyze_predictions(self, texts, y_true, y_pred, y_prob=None, num_examples=10):
        """
        Analyze model predictions with examples
        
        Args:
            texts (list): Input texts
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_prob (array, optional): Prediction probabilities
            num_examples (int): Number of examples to show
            
        Returns:
            dict: Analysis results
        """
        # Find correct and incorrect predictions
        correct_mask = y_true == y_pred
        incorrect_mask = ~correct_mask
        
        analysis = {
            'total_samples': len(y_true),
            'correct_predictions': int(correct_mask.sum()),
            'incorrect_predictions': int(incorrect_mask.sum()),
            'accuracy': float(correct_mask.mean())
        }
        
        # Get examples of correct predictions
        if correct_mask.sum() > 0:
            correct_indices = np.where(correct_mask)[0]
            sample_correct = np.random.choice(correct_indices, 
                                            min(num_examples, len(correct_indices)), 
                                            replace=False)
            
            analysis['correct_examples'] = []
            for idx in sample_correct:
                example = {
                    'text': texts[idx][:100] + "..." if len(texts[idx]) > 100 else texts[idx],
                    'true_label': self.class_names[y_true[idx]],
                    'predicted_label': self.class_names[y_pred[idx]]
                }
                if y_prob is not None:
                    example['confidence'] = float(y_prob[idx].max())
                analysis['correct_examples'].append(example)
        
        # Get examples of incorrect predictions
        if incorrect_mask.sum() > 0:
            incorrect_indices = np.where(incorrect_mask)[0]
            sample_incorrect = np.random.choice(incorrect_indices, 
                                              min(num_examples, len(incorrect_indices)), 
                                              replace=False)
            
            analysis['incorrect_examples'] = []
            for idx in sample_incorrect:
                example = {
                    'text': texts[idx][:100] + "..." if len(texts[idx]) > 100 else texts[idx],
                    'true_label': self.class_names[y_true[idx]],
                    'predicted_label': self.class_names[y_pred[idx]]
                }
                if y_prob is not None:
                    example['confidence'] = float(y_prob[idx].max())
                analysis['incorrect_examples'].append(example)
        
        return analysis
    
    def create_performance_summary(self, metrics, analysis=None):
        """
        Create a comprehensive performance summary
        
        Args:
            metrics (dict): Evaluation metrics
            analysis (dict, optional): Prediction analysis
            
        Returns:
            str: Performance summary
        """
        summary = f"""
# Model Performance Summary

## Overall Metrics
- **Accuracy**: {metrics['accuracy']:.4f}
- **Weighted Precision**: {metrics['precision_weighted']:.4f}
- **Weighted Recall**: {metrics['recall_weighted']:.4f}
- **Weighted F1-Score**: {metrics['f1_weighted']:.4f}
"""
        
        if 'auc' in metrics:
            summary += f"- **AUC-ROC**: {metrics['auc']:.4f}\n"
        
        summary += "\n## Class-wise Performance\n"
        summary += "| Class | Precision | Recall | F1-Score | Support |\n"
        summary += "|-------|-----------|--------|----------|---------|\n"
        
        for i, class_name in enumerate(self.class_names):
            precision = metrics['class_metrics']['precision'][i]
            recall = metrics['class_metrics']['recall'][i]
            f1 = metrics['class_metrics']['f1'][i]
            support = metrics['class_metrics']['support'][i]
            summary += f"| {class_name} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support} |\n"
        
        if analysis:
            summary += f"\n## Prediction Analysis\n"
            summary += f"- **Total Samples**: {analysis['total_samples']}\n"
            summary += f"- **Correct Predictions**: {analysis['correct_predictions']}\n"
            summary += f"- **Incorrect Predictions**: {analysis['incorrect_predictions']}\n"
        
        return summary 