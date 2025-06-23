"""
Utility functions for the sentiment analysis pipeline.

Contains helper functions for logging, preprocessing, metrics calculation,
data handling, and other common operations.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Use absolute imports when running as script, relative when as package
try:
    from .config import LOGGING_CONFIG, PATHS, LABEL_MAPPING
except ImportError:
    import sys
    from pathlib import Path
    # Add current directory to path to import our local config
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    import config as local_config
    LOGGING_CONFIG = local_config.LOGGING_CONFIG
    PATHS = local_config.PATHS
    LABEL_MAPPING = local_config.LABEL_MAPPING


def setup_logging(
    log_level: str = LOGGING_CONFIG["level"],
    log_file: Optional[Path] = LOGGING_CONFIG["log_file"],
    console_output: bool = LOGGING_CONFIG["console_output"]
) -> logging.Logger:
    """
    Setup logging configuration for the pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (None for no file logging)
        console_output: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("sentiment_pipeline")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(LOGGING_CONFIG["format"])
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def clean_text(text: str) -> str:
    """
    Clean and normalize text data.
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text
    """
    import re
    
    # Convert to string and handle None values
    text = str(text) if text is not None else ""
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def calculate_text_statistics(texts: List[str]) -> Dict[str, float]:
    """
    Calculate statistics for a collection of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary with text statistics
    """
    word_counts = [len(text.split()) for text in texts]
    char_counts = [len(text) for text in texts]
    
    return {
        "avg_words": np.mean(word_counts),
        "median_words": np.median(word_counts),
        "max_words": np.max(word_counts),
        "min_words": np.min(word_counts),
        "avg_chars": np.mean(char_counts),
        "total_samples": len(texts),
        "vocab_size": len(set(' '.join(texts).lower().split()))
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with all metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1]
    )
    
    # Weighted averages
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': float(accuracy),
        'precision_weighted': float(precision_w),
        'recall_weighted': float(recall_w),
        'f1_weighted': float(f1_w),
        'class_precision': [float(p) for p in precision],
        'class_recall': [float(r) for r in recall],
        'class_f1': [float(f) for f in f1],
        'class_support': [int(s) for s in support],
        'confusion_matrix': cm.tolist(),
        'total_samples': len(y_true)
    }


def save_results(results: Dict[str, Any], filepath: Path) -> None:
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        filepath: Path to save file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results(filepath: Path) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Results dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_performance_plots(
    results: Dict[str, Any],
    save_path: Optional[Path] = None
) -> None:
    """
    Create comprehensive performance visualization plots.
    
    Args:
        results: Results dictionary from pipeline
        save_path: Path to save the plot (optional)
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Performance metrics bar chart
    if 'evaluation' in results:
        eval_results = results['evaluation']
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            eval_results.get('accuracy', 0),
            eval_results.get('precision_weighted', 0),
            eval_results.get('recall_weighted', 0),
            eval_results.get('f1_weighted', 0)
        ]
        
        bars = ax1.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax1.set_ylim(0, 1)
        ax1.set_title('Model Performance Metrics', fontweight='bold')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{value:.3f}', ha='center', fontweight='bold')
    
    # 2. Training progress (if available)
    if 'training' in results and 'progress' in results['training']:
        progress = results['training']['progress']
        epochs = [p['epoch'] for p in progress]
        
        if 'eval_accuracy' in progress[0]:
            accuracies = [p['eval_accuracy'] for p in progress]
            ax2.plot(epochs, accuracies, 'b-o', linewidth=2, label='Accuracy')
        
        if 'eval_loss' in progress[0]:
            losses = [p['eval_loss'] for p in progress]
            ax2_twin = ax2.twinx()
            ax2_twin.plot(epochs, losses, 'r-s', linewidth=2, label='Loss')
            ax2_twin.set_ylabel('Loss', color='r')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy', color='b')
        ax2.set_title('Training Progress', fontweight='bold')
        ax2.legend(loc='upper left')
    
    # 3. Confusion matrix
    if 'evaluation' in results and 'confusion_matrix' in results['evaluation']:
        cm = np.array(results['evaluation']['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        ax3.set_title('Confusion Matrix', fontweight='bold')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
    
    # 4. Class-wise performance
    if 'evaluation' in results and 'class_precision' in results['evaluation']:
        eval_results = results['evaluation']
        classes = ['Negative', 'Positive']
        precision = eval_results.get('class_precision', [0, 0])
        recall = eval_results.get('class_recall', [0, 0])
        f1 = eval_results.get('class_f1', [0, 0])
        
        x = np.arange(len(classes))
        width = 0.25
        
        ax4.bar(x - width, precision, width, label='Precision', color='lightblue')
        ax4.bar(x, recall, width, label='Recall', color='lightgreen')
        ax4.bar(x + width, f1, width, label='F1-Score', color='lightcoral')
        
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Score')
        ax4.set_title('Class-wise Performance', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(classes)
        ax4.legend()
        ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Performance plots saved to {save_path}")
    
    plt.close()


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def get_model_size(model_path: Path) -> str:
    """
    Get the size of a model directory.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Formatted size string
    """
    if not model_path.exists():
        return "N/A"
    
    total_size = 0
    for file_path in model_path.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    # Convert to human readable format
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024:
            return f"{total_size:.1f} {unit}"
        total_size /= 1024
    
    return f"{total_size:.1f} TB"


def validate_dataset_balance(labels: List[int], tolerance: float = 0.1) -> Tuple[bool, Dict[str, int]]:
    """
    Validate that dataset is reasonably balanced.
    
    Args:
        labels: List of labels
        tolerance: Acceptable imbalance ratio (0.1 = 10%)
        
    Returns:
        Tuple of (is_balanced, class_counts)
    """
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    if len(unique) != 2:
        return False, class_counts
    
    ratio = min(counts) / max(counts)
    is_balanced = ratio >= (1 - tolerance)
    
    return is_balanced, class_counts


def print_system_info():
    """Print system and environment information."""
    import torch
    import platform
    
    print("🖥️  System Information:")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Python: {platform.python_version()}")
    
    if torch is not None:
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    print()


def create_directory_structure():
    """Create the project directory structure if it doesn't exist."""
    try:
        from .config import DATA_DIR, CACHE_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR
    except ImportError:
        import sys
        from pathlib import Path
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        import config as local_config
        DATA_DIR = local_config.DATA_DIR
        CACHE_DIR = local_config.CACHE_DIR
        MODELS_DIR = local_config.MODELS_DIR
        REPORTS_DIR = local_config.REPORTS_DIR
        LOGS_DIR = local_config.LOGS_DIR
    
    directories = [DATA_DIR, CACHE_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
    print("📁 Directory structure created/verified") 