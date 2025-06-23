"""
PyTorch-only sentiment analysis model training.

This module provides a clean, TensorFlow-free interface for training 
the DistilBERT sentiment analysis model using only PyTorch.
"""

import os
import sys
import warnings
from pathlib import Path

# 🚫 BLOCK TENSORFLOW BEFORE ANY IMPORTS
os.environ['USE_TF'] = 'None'
os.environ['USE_TORCH'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Disable TensorFlow import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

# Add current directory to path for local imports
sys.path.append(str(Path(__file__).parent))

# Core imports - PyTorch only
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json

# Specific PyTorch-only Hugging Face imports
try:
    # Import only PyTorch components explicitly
    from transformers.models.distilbert.tokenization_distilbert import DistilBertTokenizer
    from transformers.models.distilbert.modeling_distilbert import DistilBertForSequenceClassification
    from transformers.trainer import Trainer
    from transformers.training_args import TrainingArguments
    from datasets import load_dataset
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Try: pip install transformers[torch] --no-deps --force-reinstall")
    sys.exit(1)

# Local imports (no relative imports)
try:
    from config import MODEL_CONFIG, TRAINING_CONFIG, PATHS
except ImportError:
    # Fallback configuration if config.py has issues
    MODEL_CONFIG = {
        "model_name": "distilbert-base-uncased",
        "num_labels": 2,
        "max_length": 512,
    }
    TRAINING_CONFIG = {
        "batch_size": 16,
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "warmup_steps": 500,
        "weight_decay": 0.01,
    }
    PATHS = {
        "model_output": Path("../models/distilbert-imdb-sentiment"),
    }


class SentimentClassifier:
    """PyTorch-only DistilBERT sentiment classifier."""
    
    def __init__(self):
        """Initialize the sentiment classifier."""
        self.model_name = MODEL_CONFIG['model_name']
        self.max_length = MODEL_CONFIG['max_length']
        self.num_labels = MODEL_CONFIG['num_labels']
        
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        self.trainer = None
        
        # Ensure PyTorch device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
    def load_data(self):
        """Load and cache the IMDb dataset."""
        print("📥 Loading IMDb dataset...")
        try:
            # Force PyTorch format only
            self.dataset = load_dataset("imdb", cache_dir="../data/cache")
            print(f"✅ Dataset loaded: {len(self.dataset['train']):,} train, {len(self.dataset['test']):,} test")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            raise
            
    def _tokenize_function(self, examples):
        """Tokenize text examples for PyTorch."""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors=None  # Let datasets handle tensor conversion
        )
    
    def preprocess_data(self):
        """Preprocess the dataset for training."""
        print("🔤 Loading tokenizer and preprocessing data...")
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        
        # Tokenize datasets
        self.train_dataset = self.dataset['train'].map(
            self._tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        self.test_dataset = self.dataset['test'].map(
            self._tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        
        # Set format to PyTorch tensors
        self.train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        self.test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        
        print(f"✅ Preprocessing completed")
        print(f"   Train samples: {len(self.train_dataset):,}")
        print(f"   Test samples: {len(self.test_dataset):,}")
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self):
        """Train the DistilBERT model."""
        print("🤖 Initializing DistilBERT model...")
        
        # Load model - force PyTorch only
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            torch_dtype=torch.float32,  # Explicit PyTorch dtype
            device_map=None  # Manual device management
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup training arguments
        output_dir = Path("../models/distilbert-imdb-sentiment")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=TRAINING_CONFIG['num_epochs'],
            per_device_train_batch_size=TRAINING_CONFIG['batch_size'],
            per_device_eval_batch_size=TRAINING_CONFIG['batch_size'],
            learning_rate=TRAINING_CONFIG['learning_rate'],
            warmup_steps=TRAINING_CONFIG['warmup_steps'],
            weight_decay=TRAINING_CONFIG['weight_decay'],
            logging_dir=str(output_dir / "logs"),
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            save_total_limit=2,
            dataloader_num_workers=0,  # Windows compatibility
            use_cpu=False if torch.cuda.is_available() else True,
            fp16=False,  # Disable FP16 for Windows compatibility
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        print("🏋️ Starting training...")
        try:
            # Train the model
            train_result = self.trainer.train()
            print(f"✅ Training completed!")
            print(f"   Final loss: {train_result.training_loss:.4f}")
            
            # Save model and tokenizer
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"💾 Model saved to: {output_dir}")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
    
    def evaluate(self):
        """Evaluate the trained model."""
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation")
        
        print("📊 Evaluating model...")
        eval_result = self.trainer.evaluate()
        
        # Save results
        results = {
            "model_name": self.model_name,
            "accuracy": eval_result['eval_accuracy'],
            "f1": eval_result['eval_f1'],
            "precision": eval_result['eval_precision'],
            "recall": eval_result['eval_recall'],
            "eval_loss": eval_result['eval_loss'],
        }
        
        # Save to JSON
        results_file = Path("../reports/evaluation_metrics.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({"model_performance": results}, f, indent=2)
        
        print(f"📊 Evaluation results saved to: {results_file}")
        return results


def train_sentiment_model():
    """Main training function - PyTorch only."""
    try:
        classifier = SentimentClassifier()
        classifier.load_data()
        classifier.preprocess_data()
        classifier.train()
        results = classifier.evaluate()
        return classifier, results
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    print("🚀 PyTorch-only DistilBERT Training")
    print("=" * 50)
    classifier, results = train_sentiment_model()
    print("🎉 Training completed successfully!") 