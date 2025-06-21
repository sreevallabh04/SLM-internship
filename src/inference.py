#!/usr/bin/env python3
"""
Standalone inference script for sentiment analysis.

This script provides a simple interface for making sentiment predictions
on new text using a trained RoBERTa model.

Usage:
    python src/inference.py --text "This movie is amazing!"
    python src/inference.py --file input.txt
    python src/inference.py --interactive
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Use absolute imports when running as script, relative when as package
try:
    from .config import PATHS, MODEL_CONFIG, INFERENCE_CONFIG, TEST_SAMPLES, LABEL_MAPPING
    from .utils import setup_logging, print_system_info
except ImportError:
    from config import PATHS, MODEL_CONFIG, INFERENCE_CONFIG, TEST_SAMPLES, LABEL_MAPPING
    from utils import setup_logging, print_system_info


class SentimentPredictor:
    """Standalone sentiment prediction class."""
    
    def __init__(self, model_path: str = None):
        """Initialize the predictor."""
        self.model_path = model_path or str(PATHS['final_model'])
        self.logger = setup_logging(console_output=True)
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # Try to load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and tokenizer."""
        try:
            # Try importing transformers
            import torch
            from transformers import RobertaTokenizer, RobertaForSequenceClassification
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Check if model exists
            model_path = Path(self.model_path)
            if not model_path.exists():
                self.logger.warning(f"Model not found at {model_path}")
                self.logger.info("Will use simulation mode for predictions")
                return
            
            # Load tokenizer and model
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path)
            self.model = RobertaForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"✅ Model loaded successfully from {model_path}")
            self.logger.info(f"   Device: {self.device}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load model: {e}")
            self.logger.info("Will use simulation mode for predictions")
    
    def predict_text(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is not None and self.tokenizer is not None:
            return self._predict_with_model(text)
        else:
            return self._predict_simulation(text)
    
    def _predict_with_model(self, text: str) -> Dict[str, Any]:
        """Make prediction using the trained model."""
        import torch
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=INFERENCE_CONFIG['max_length']
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = torch.max(probabilities).item()
            
            return {
                'text': text,
                'prediction': LABEL_MAPPING[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    'negative': probabilities[0][0].item(),
                    'positive': probabilities[0][1].item()
                },
                'model_used': 'roberta-base'
            }
    
    def _predict_simulation(self, text: str) -> Dict[str, Any]:
        """Make prediction using keyword-based simulation."""
        import numpy as np
        
        # Enhanced keyword-based prediction
        positive_words = [
            'amazing', 'outstanding', 'excellent', 'brilliant', 'wonderful',
            'fantastic', 'great', 'good', 'love', 'perfect', 'best',
            'incredible', 'awesome', 'beautiful', 'impressive', 'superb'
        ]
        negative_words = [
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate',
            'disappointing', 'boring', 'poor', 'waste', 'stupid',
            'annoying', 'ridiculous', 'pathetic', 'useless', 'garbage'
        ]
        
        text_lower = text.lower()
        pos_score = sum(1 for word in positive_words if word in text_lower)
        neg_score = sum(1 for word in negative_words if word in text_lower)
        
        if pos_score > neg_score:
            prediction = "POSITIVE"
            confidence = 0.75 + np.random.random() * 0.20
            neg_prob = 1 - confidence
            pos_prob = confidence
        elif neg_score > pos_score:
            prediction = "NEGATIVE"
            confidence = 0.75 + np.random.random() * 0.20
            pos_prob = 1 - confidence
            neg_prob = confidence
        else:
            prediction = "NEUTRAL"
            confidence = 0.5 + np.random.random() * 0.2
            pos_prob = 0.5
            neg_prob = 0.5
        
        return {
            'text': text,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'negative': neg_prob,
                'positive': pos_prob
            },
            'model_used': 'simulation'
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            result = self.predict_text(text)
            results.append(result)
        
        return results
    
    def predict_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Predict sentiment for texts in a file (one per line).
        
        Args:
            file_path: Path to input file
            
        Returns:
            List of prediction dictionaries
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        return self.predict_batch(texts)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Make sentiment predictions using trained RoBERTa model"
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--text', type=str,
        help="Single text to analyze"
    )
    input_group.add_argument(
        '--file', type=str,
        help="File containing texts to analyze (one per line)"
    )
    input_group.add_argument(
        '--interactive', action='store_true',
        help="Interactive mode for multiple predictions"
    )
    input_group.add_argument(
        '--examples', action='store_true',
        help="Run predictions on example texts"
    )
    
    # Model options
    parser.add_argument(
        '--model-path', type=str, default=str(PATHS['final_model']),
        help=f"Path to trained model (default: {PATHS['final_model']})"
    )
    
    # Output options
    parser.add_argument(
        '--output', type=str, default=None,
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        '--format', type=str, choices=['json', 'text'], default='text',
        help="Output format (default: text)"
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help="Suppress informational output"
    )
    
    return parser.parse_args()


def format_prediction(prediction: Dict[str, Any], format_type: str = 'text') -> str:
    """Format prediction result for display."""
    if format_type == 'json':
        return json.dumps(prediction, indent=2)
    else:
        text_preview = prediction['text'][:60] + "..." if len(prediction['text']) > 60 else prediction['text']
        return (
            f"Text: {text_preview}\n"
            f"Prediction: {prediction['prediction']}\n"
            f"Confidence: {prediction['confidence']:.3f}\n"
            f"Probabilities: Neg={prediction['probabilities']['negative']:.3f}, "
            f"Pos={prediction['probabilities']['positive']:.3f}\n"
            f"Model: {prediction['model_used']}\n"
        )


def interactive_mode(predictor: SentimentPredictor):
    """Run interactive prediction mode."""
    print("\n🔮 Interactive Sentiment Analysis")
    print("=" * 40)
    print("Enter text to analyze (type 'quit' to exit):")
    print()
    
    while True:
        try:
            text = input("➤ ")
            
            if text.lower().strip() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not text.strip():
                continue
            
            result = predictor.predict_text(text)
            print(f"   {result['prediction']} (confidence: {result['confidence']:.3f})")
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main inference function."""
    args = parse_arguments()
    
    # Setup predictor
    if not args.quiet:
        print("🔮 Sentiment Analysis Inference")
        print("=" * 40)
        print_system_info()
    
    predictor = SentimentPredictor(model_path=args.model_path)
    
    try:
        results = []
        
        if args.text:
            # Single text prediction
            result = predictor.predict_text(args.text)
            results = [result]
            
        elif args.file:
            # File prediction
            results = predictor.predict_file(args.file)
            
        elif args.interactive:
            # Interactive mode
            interactive_mode(predictor)
            return 0
            
        elif args.examples:
            # Example predictions
            results = predictor.predict_batch(TEST_SAMPLES)
        
        # Display results
        if not args.quiet:
            print(f"\n📊 Results ({len(results)} predictions):")
            print("-" * 40)
        
        for i, result in enumerate(results, 1):
            if not args.quiet:
                print(f"{i}. {format_prediction(result, args.format)}")
            else:
                print(format_prediction(result, args.format))
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            if not args.quiet:
                print(f"💾 Results saved to {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 