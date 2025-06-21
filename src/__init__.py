"""
Sentiment Analysis Pipeline Package

A production-ready NLP text classification pipeline for movie sentiment analysis
using RoBERTa transformer model on the IMDb dataset.

Built after many late nights of debugging dependency conflicts and wrestling
with transformer models. The result: a system that actually works in production!

Key Components:
- clean_pipeline.py: Main end-to-end pipeline
- train.py: Full training script  
- inference.py: Standalone prediction script
- utils.py: Helper functions and utilities
- config.py: Configuration and hyperparameters
"""

__version__ = "1.0.0"
__author__ = "Sreevallabh Kakarala"

from .clean_pipeline import CleanNLPPipeline

__all__ = ["CleanNLPPipeline"] 