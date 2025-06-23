"""
Configuration settings for the sentiment analysis pipeline.

Contains hyperparameters, file paths, and model configurations.

Author: Sreevallabh Kakarala
Note: These settings were fine-tuned through many experiments and 
      late-night debugging sessions. Change at your own risk! 😅
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model configuration
MODEL_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "num_labels": 2,
    "max_length": 512,
    "problem_type": "single_label_classification"
}

# Training hyperparameters
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-5,
    "weight_decay": 0.1,
    "num_epochs": 5,
    "warmup_steps": 500,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "logging_steps": 100,
    "dataloader_num_workers": 0,  # Windows compatibility
    "report_to": None,  # Disable wandb
}

# Dataset configuration
DATASET_CONFIG = {
    "dataset_name": "imdb",
    "train_split": "train",
    "test_split": "test",
    "text_column": "text",
    "label_column": "label",
    "use_full_dataset": True,  # Set to False for smaller samples
    "train_sample_size": 25000,  # Used when use_full_dataset = False
    "test_sample_size": 25000,   # Used when use_full_dataset = False
}

# Inference configuration
INFERENCE_CONFIG = {
    "max_length": 512,
    "return_probabilities": True,
    "batch_size": 16,
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": LOGS_DIR / "pipeline.log",
    "console_output": True,
}

# Hardware configuration
DEVICE_CONFIG = {
    "use_cuda": True,  # Will auto-detect and fallback to CPU
    "mixed_precision": True,  # FP16 training if available
    "device": None,  # Will be auto-detected
}

# File paths
PATHS = {
    "model_output": MODELS_DIR / "distilbert-imdb-sentiment",
    "final_model": MODELS_DIR / "distilbert-imdb-sentiment" / "final",
    "training_logs": LOGS_DIR / "training_results.json",
    "performance_dashboard": REPORTS_DIR / "performance_dashboard.png",
    "pipeline_report": REPORTS_DIR / "clean_pipeline_report.md",
    "pipeline_results": REPORTS_DIR / "clean_pipeline_results.json",
}

# Reproducibility
RANDOM_SEED = 42

# Environment variables
def setup_environment():
    """Setup environment variables for optimal performance"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings
    
    # Create directories if they don't exist
    for directory in [DATA_DIR, CACHE_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Test samples for inference testing
TEST_SAMPLES = [
    "This movie is absolutely amazing! Outstanding performances throughout.",
    "Terrible film with poor acting and boring plot. Complete waste of time.",
    "Decent movie with good moments. Worth watching but not exceptional.",
    "Disappointing sequel that fails to live up to the original's quality.",
    "Brilliant cinematography and exceptional storytelling make this a masterpiece.",
    "Boring and predictable with terrible dialogue and poor direction."
]

# Class labels
LABEL_MAPPING = {
    0: "NEGATIVE",
    1: "POSITIVE"
}

REVERSE_LABEL_MAPPING = {
    "NEGATIVE": 0,
    "POSITIVE": 1
} 