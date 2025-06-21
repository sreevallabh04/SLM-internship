#!/usr/bin/env python3
"""
Standalone training script for sentiment analysis model.

This script provides a focused training interface for the RoBERTa sentiment classifier
on the IMDb dataset with configurable hyperparameters.

Usage:
    python src/train.py [--epochs 5] [--batch-size 32] [--learning-rate 1e-5]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Use absolute imports when running as script, relative when as package
try:
    from .config import (
        TRAINING_CONFIG, MODEL_CONFIG, DATASET_CONFIG, PATHS, RANDOM_SEED,
        setup_environment
    )
    from .utils import setup_logging, print_system_info, create_directory_structure
    from .clean_pipeline import CleanNLPPipeline
except ImportError:
    from config import (
        TRAINING_CONFIG, MODEL_CONFIG, DATASET_CONFIG, PATHS, RANDOM_SEED,
        setup_environment
    )
    from utils import setup_logging, print_system_info, create_directory_structure
    from clean_pipeline import CleanNLPPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RoBERTa sentiment analysis model on IMDb dataset"
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', type=int, default=TRAINING_CONFIG['num_epochs'],
        help=f"Number of training epochs (default: {TRAINING_CONFIG['num_epochs']})"
    )
    parser.add_argument(
        '--batch-size', type=int, default=TRAINING_CONFIG['batch_size'],
        help=f"Training batch size (default: {TRAINING_CONFIG['batch_size']})"
    )
    parser.add_argument(
        '--learning-rate', type=float, default=TRAINING_CONFIG['learning_rate'],
        help=f"Learning rate (default: {TRAINING_CONFIG['learning_rate']})"
    )
    parser.add_argument(
        '--weight-decay', type=float, default=TRAINING_CONFIG['weight_decay'],
        help=f"Weight decay (default: {TRAINING_CONFIG['weight_decay']})"
    )
    parser.add_argument(
        '--warmup-steps', type=int, default=TRAINING_CONFIG['warmup_steps'],
        help=f"Warmup steps (default: {TRAINING_CONFIG['warmup_steps']})"
    )
    
    # Dataset parameters  
    parser.add_argument(
        '--use-full-dataset', action='store_true', default=DATASET_CONFIG['use_full_dataset'],
        help="Use full IMDb dataset (25K train + 25K test)"
    )
    parser.add_argument(
        '--train-samples', type=int, default=DATASET_CONFIG['train_sample_size'],
        help=f"Number of training samples if not using full dataset (default: {DATASET_CONFIG['train_sample_size']})"
    )
    parser.add_argument(
        '--test-samples', type=int, default=DATASET_CONFIG['test_sample_size'],
        help=f"Number of test samples if not using full dataset (default: {DATASET_CONFIG['test_sample_size']})"
    )
    
    # Model parameters
    parser.add_argument(
        '--model-name', type=str, default=MODEL_CONFIG['model_name'],
        help=f"Base model name (default: {MODEL_CONFIG['model_name']})"
    )
    parser.add_argument(
        '--max-length', type=int, default=MODEL_CONFIG['max_length'],
        help=f"Maximum sequence length (default: {MODEL_CONFIG['max_length']})"
    )
    
    # Output parameters
    parser.add_argument(
        '--output-dir', type=str, default=str(PATHS['model_output']),
        help=f"Output directory for model (default: {PATHS['model_output']})"
    )
    parser.add_argument(
        '--experiment-name', type=str, default=None,
        help="Experiment name for output folder (default: timestamp)"
    )
    
    # Training options
    parser.add_argument(
        '--no-mixed-precision', action='store_true',
        help="Disable mixed precision training"
    )
    parser.add_argument(
        '--no-early-stopping', action='store_true',
        help="Disable early stopping"
    )
    parser.add_argument(
        '--seed', type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )
    
    # Logging
    parser.add_argument(
        '--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help="Suppress progress output"
    )
    
    return parser.parse_args()


def setup_training_config(args):
    """Setup training configuration based on arguments."""
    # Update training config with command line arguments
    training_config = TRAINING_CONFIG.copy()
    training_config.update({
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'fp16': not args.no_mixed_precision,
    })
    
    # Update model config
    model_config = MODEL_CONFIG.copy()
    model_config.update({
        'model_name': args.model_name,
        'max_length': args.max_length,
    })
    
    # Update dataset config
    dataset_config = DATASET_CONFIG.copy()
    dataset_config.update({
        'use_full_dataset': args.use_full_dataset,
        'train_sample_size': args.train_samples,
        'test_sample_size': args.test_samples,
    })
    
    return training_config, model_config, dataset_config


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment and directories
    setup_environment()
    create_directory_structure()
    
    # Setup logging
    log_file = PATHS['training_logs'].parent / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(
        log_level=args.log_level,
        log_file=log_file,
        console_output=not args.quiet
    )
    
    # Print system info
    if not args.quiet:
        print("🚀 RoBERTa Sentiment Analysis Training")
        print("=" * 50)
        print_system_info()
    
    # Setup configurations
    training_config, model_config, dataset_config = setup_training_config(args)
    
    # Determine output directory
    if args.experiment_name:
        output_dir = Path(args.output_dir) / args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"experiment_{timestamp}"
    
    logger.info(f"Starting training with the following configuration:")
    logger.info(f"  Epochs: {training_config['num_epochs']}")
    logger.info(f"  Batch size: {training_config['batch_size']}")
    logger.info(f"  Learning rate: {training_config['learning_rate']}")
    logger.info(f"  Weight decay: {training_config['weight_decay']}")
    logger.info(f"  Warmup steps: {training_config['warmup_steps']}")
    logger.info(f"  Model: {model_config['model_name']}")
    logger.info(f"  Max length: {model_config['max_length']}")
    logger.info(f"  Full dataset: {dataset_config['use_full_dataset']}")
    logger.info(f"  Output directory: {output_dir}")
    
    try:
        # Initialize pipeline with custom configurations
        pipeline = CleanNLPPipeline(seed=args.seed)
        
        # Override configurations (if we had a more modular pipeline)
        # For now, the pipeline uses its built-in configurations
        
        # Run training pipeline
        start_time = datetime.now()
        
        logger.info("Loading dataset...")
        pipeline.load_data()
        
        logger.info("Preprocessing data...")
        pipeline.preprocess_data()
        
        logger.info("Starting model training...")
        success = pipeline.train_model()
        
        if not success:
            logger.error("Training failed!")
            return 1
        
        logger.info("Evaluating model...")
        pipeline.evaluate_model()
        
        logger.info("Creating visualizations...")
        pipeline.create_visualizations()
        
        logger.info("Testing inference...")
        pipeline.test_inference()
        
        logger.info("Generating report...")
        pipeline.generate_report()
        
        # Calculate total time
        end_time = datetime.now()
        total_time = end_time - start_time
        
        # Final results
        if hasattr(pipeline, 'results') and 'evaluation' in pipeline.results:
            accuracy = pipeline.results['evaluation']['accuracy']
            f1_score = pipeline.results['evaluation']['f1_weighted']
            
            logger.info("🎉 Training completed successfully!")
            logger.info(f"  Final accuracy: {accuracy:.4f}")
            logger.info(f"  Final F1-score: {f1_score:.4f}")
            logger.info(f"  Total time: {total_time}")
            logger.info(f"  Results saved to: {PATHS['reports']}")
            
            if not args.quiet:
                print("\n" + "=" * 50)
                print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
                print("=" * 50)
                print(f"✅ Final Accuracy: {accuracy:.4f}")
                print(f"🏆 Final F1-Score: {f1_score:.4f}")
                print(f"⏱️  Total Time: {total_time}")
                print(f"📁 Results: {PATHS['reports']}")
                print("=" * 50)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        print("\n⚠️  Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        print(f"\n❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 