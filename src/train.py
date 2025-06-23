#!/usr/bin/env python3
"""
🚀 PyTorch-only DistilBERT Training Script

This script provides a TensorFlow-free training interface for the 
DistilBERT sentiment classifier on the IMDb dataset.

Features:
- 100% PyTorch implementation
- No TensorFlow dependencies
- Windows-compatible DLL loading
- Clean, production-ready code

Usage:
    cd src && python train.py
"""

import os
import sys
from pathlib import Path

# 🚫 BLOCK TENSORFLOW BEFORE ANY IMPORTS
os.environ['USE_TF'] = 'None'
os.environ['USE_TORCH'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Add paths for imports
sys.path.append(str(Path(__file__).parent))

# Import our PyTorch-only modules
try:
    from train_model import SentimentClassifier, train_sentiment_model
    from config import MODEL_CONFIG, TRAINING_CONFIG
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running from the src directory")
    print("💡 Try: cd src && python train.py")
    sys.exit(1)


def main():
    """Main PyTorch-only training function."""
    print("🚀 PyTorch-only DistilBERT Sentiment Analysis")
    print("=" * 60)
    
    print(f"📝 Configuration:")
    print(f"   Model: {MODEL_CONFIG['model_name']}")
    print(f"   Max Length: {MODEL_CONFIG['max_length']}")
    print(f"   Batch Size: {TRAINING_CONFIG['batch_size']}")
    print(f"   Learning Rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"   Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"   TensorFlow: ❌ DISABLED")
    print(f"   PyTorch: ✅ ENABLED")
    print("=" * 60)
    
    try:
        # Run the complete training pipeline
        print("🏋️ Starting PyTorch-only training pipeline...")
        classifier, results = train_sentiment_model()
        
        # Print final results
        print("\n" + "=" * 60)
        print("🎉 PYTORCH TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"✅ Final Accuracy: {results['accuracy']:.4f} ({results['accuracy']:.2%})")
        print(f"🏆 Final F1-Score: {results['f1']:.4f}")
        print(f"🎯 Final Precision: {results['precision']:.4f}")
        print(f"🔄 Final Recall: {results['recall']:.4f}")
        print(f"📉 Final Loss: {results['eval_loss']:.4f}")
        print("=" * 60)
        print(f"📁 Model saved to: {Path('../models/distilbert-imdb-sentiment').resolve()}")
        print(f"📊 Results saved to: {Path('../reports').resolve()}")
        print("=" * 60)
        print("✅ NO TENSORFLOW ISSUES - PYTORCH ONLY! 🎯")
        print("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ PyTorch training failed: {e}")
        print("🔍 Error details:")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure you're in the src/ directory")
        print("   2. Check that PyTorch is installed: pip list | grep torch")
        print("   3. Verify transformers[torch] is installed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 