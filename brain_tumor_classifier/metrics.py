"""
Metrics and visualization utilities for Brain Tumor Classification.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, recall_score, precision_score
)


def auc_macro(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate macro-averaged AUC for multi-class problems.
    
    Args:
        y_true: Ground truth labels (shape: n_samples)
        y_prob: Predicted probabilities (shape: n_samples x n_classes)
        
    Returns:
        Macro-averaged AUC score
    """
    n_classes = y_prob.shape[1]
    
    # Convert ground truth labels to one-hot encoding
    y_true_bin = np.eye(n_classes)[y_true.astype(int)]
    
    # Calculate AUC for each class
    auc_scores = []
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            auc_scores.append(auc(fpr, tpr))
        except ValueError:
            # Skip classes with only one label
            continue
    
    # Calculate macro average
    if not auc_scores:
        raise ValueError("Could not compute AUC for any class.")
    
    return np.mean(auc_scores)


class SciMetrics:
    """
    Scientific metrics and visualization utility class.
    
    This class provides methods for calculating and visualizing various
    performance metrics for classification models.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize the SciMetrics class.
        
        Args:
            class_names: List of class names for plotting
        """
        self.class_names = class_names
        
    def set_class_names(self, class_names: List[str]) -> None:
        """
        Set the class names.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        
    def calculate_all_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate all classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_per_class': precision_score(y_true, y_pred, average=None),
            'recall_per_class': recall_score(y_true, y_pred, average=None),
            'f1_per_class': f1_score(y_true, y_pred, average=None),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'report': classification_report(y_true, y_pred, output_dict=True),
        }
        
        # Add AUC metrics if probabilities are provided
        if y_prob is not None:
            n_classes = y_prob.shape[1]
            
            # Convert ground truth to one-hot encoding
            y_true_bin = np.eye(n_classes)[y_true.astype(int)]
            
            # Calculate AUC for each class
            class_auc = []
            for i in range(n_classes):
                try:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    class_auc.append(auc(fpr, tpr))
                except ValueError:
                    class_auc.append(np.nan)
            
            metrics['auc_per_class'] = np.array(class_auc)
            
            # Calculate macro AUC
            try:
                metrics['auc_macro'] = auc_macro(y_true, y_prob)
            except ValueError:
                metrics['auc_macro'] = np.nan
                
        return metrics
    
    def print_metrics_summary(self, metrics: Dict[str, Any]) -> None:
        """
        Print a summary of the metrics.
        
        Args:
            metrics: Dictionary of metrics as returned by calculate_all_metrics
        """
        print("\n=== CLASSIFICATION METRICS ===")
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print(f"Precision:     {metrics['precision_macro']:.4f} (macro avg)")
        print(f"Recall:        {metrics['recall_macro']:.4f} (macro avg)")
        print(f"F1 Score:      {metrics['f1_macro']:.4f} (macro avg)")
        
        if 'auc_macro' in metrics:
            print(f"AUC:           {metrics['auc_macro']:.4f} (macro avg)")
            
        print("\n=== PER-CLASS METRICS ===")
        for i, class_name in enumerate(self.class_names or [f"Class {i}" for i in range(len(metrics['precision_per_class']))]):
            print(f"{class_name}:")
            print(f"  Precision:  {metrics['precision_per_class'][i]:.4f}")
            print(f"  Recall:     {metrics['recall_per_class'][i]:.4f}")
            print(f"  F1 Score:   {metrics['f1_per_class'][i]:.4f}")
            
            if 'auc_per_class' in metrics:
                print(f"  AUC:        {metrics['auc_per_class'][i]:.4f}")
    
    def plot_training_history(
        self, 
        history: Dict[str, List[float]], 
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Plot training history curves.
        
        Args:
            history: Dictionary with keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'val_auc'
            save_path: Path to save the figure (optional)
            figsize: Figure size as (width, height)
        """
        epochs = range(1, len(history['train_loss']) + 1)
        
        plt.figure(figsize=figsize)
        
        # Set up a 2x2 grid of subplots
        plt.subplot(2, 2, 1)
        plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
        plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
        plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot AUC if available
        if 'val_auc' in history:
            plt.subplot(2, 2, 3)
            plt.plot(epochs, history['val_auc'], 'go-', label='Validation AUC')
            plt.title('Validation AUC')
            plt.xlabel('Epochs')
            plt.ylabel('AUC')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # Plot Learning Rate if available
        if 'lr' in history:
            plt.subplot(2, 2, 4)
            plt.plot(epochs, history['lr'], 'mo-', label='Learning Rate')
            plt.title('Learning Rate')
            plt.xlabel('Epochs')
            plt.ylabel('Learning Rate')
            plt.grid(True, alpha=0.3)
            plt.legend()
            # Use log scale for learning rate
            plt.yscale('log')
            
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            
        plt.show()
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        normalize: bool = False,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'Blues'
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            normalize: Whether to normalize the confusion matrix
            save_path: Path to save the figure (optional)
            figsize: Figure size as (width, height)
            cmap: Colormap for the plot
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        # Get class names
        class_names = self.class_names or [f"Class {i}" for i in range(cm.shape[0])]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot confusion matrix
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.2f' if normalize else 'd', 
            cmap=cmap,
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            
        plt.show()
    
    def plot_roc_curves(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities
            save_path: Path to save the figure (optional)
            figsize: Figure size as (width, height)
        """
        n_classes = y_prob.shape[1]
        
        # Get class names
        class_names = self.class_names or [f"Class {i}" for i in range(n_classes)]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Convert ground truth to one-hot encoding
        y_true_bin = np.eye(n_classes)[y_true.astype(int)]
        
        # Compute ROC curve and AUC for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.2f})'
            )
            
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            
        plt.show()
    
    def plot_precision_recall_curves(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot Precision-Recall curves for multi-class classification.
        
        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities
            save_path: Path to save the figure (optional)
            figsize: Figure size as (width, height)
        """
        n_classes = y_prob.shape[1]
        
        # Get class names
        class_names = self.class_names or [f"Class {i}" for i in range(n_classes)]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Convert ground truth to one-hot encoding
        y_true_bin = np.eye(n_classes)[y_true.astype(int)]
        
        # Compute Precision-Recall curve and average precision for each class
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            avg_precision = average_precision_score(y_true_bin[:, i], y_prob[:, i])
            plt.plot(
                recall, precision, lw=2,
                label=f'{class_names[i]} (AP = {avg_precision:.2f})'
            )
            
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            
        plt.show()
        
    def create_performance_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        output_dir: str = 'plots',
        prefix: str = '',
        show_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Create a complete performance report with metrics and visualizations.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            output_dir: Directory to save plots
            prefix: Prefix for saved files
            show_plots: Whether to display plots (set to False for headless environments)
        
        Returns:
            Dictionary with computed metrics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate metrics
        metrics = self.calculate_all_metrics(y_true, y_pred, y_prob)
        
        # Print metrics summary
        self.print_metrics_summary(metrics)
        
        # Plot confusion matrix
        if show_plots:
            # Regular confusion matrix
            self.plot_confusion_matrix(
                y_true, y_pred,
                normalize=False,
                save_path=f"{output_dir}/{prefix}confusion_matrix.png"
            )
            
            # Normalized confusion matrix
            self.plot_confusion_matrix(
                y_true, y_pred, 
                normalize=True,
                save_path=f"{output_dir}/{prefix}confusion_matrix_normalized.png"
            )
        
        # Plot ROC curves
        if show_plots:
            self.plot_roc_curves(
                y_true, y_prob,
                save_path=f"{output_dir}/{prefix}roc_curves.png"
            )
        
        # Plot Precision-Recall curves
        if show_plots:
            self.plot_precision_recall_curves(
                y_true, y_prob,
                save_path=f"{output_dir}/{prefix}precision_recall_curves.png"
            )
            
        return metrics 