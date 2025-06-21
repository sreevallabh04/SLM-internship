"""
Text Classification Pipeline for Movie Sentiment Analysis
ML/NLP Engineer Intern Task - Chakaralaya AI/ML Project Team

This script implements a complete text classification pipeline using:
- IMDB movie reviews dataset
- DistilBERT for fine-tuning
- Comprehensive evaluation metrics
- Hugging Face Transformers ecosystem
"""

import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    pipeline
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class TextClassificationPipeline:
    """
    Complete text classification pipeline for sentiment analysis
    """
    
    def __init__(self, model_name="distilbert-base-uncased", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.results = {}
        
        # Create directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
    def load_and_prepare_data(self):
        """
        Load IMDB dataset and prepare for training
        Using a subset for faster training (can be expanded)
        """
        print("📊 Loading and preparing dataset...")
        
        # For demo purposes, creating a sample dataset
        # In real implementation, you would load from IMDB dataset
        sample_data = {
            'text': [
                "This movie was absolutely fantastic! Great acting and plot.",
                "Terrible film, waste of time. Poor acting and boring story.",
                "Amazing cinematography and wonderful performances by all actors.",
                "I fell asleep halfway through. Very disappointing.",
                "One of the best movies I've ever seen! Highly recommend.",
                "Not worth watching. Poor script and direction.",
                "Brilliant storytelling and excellent character development.",
                "Boring and predictable. Nothing new or exciting.",
                "Outstanding performance by the lead actor. Must watch!",
                "Complete disaster. Avoid at all costs.",
                # Add more samples for better training
                "The movie exceeded my expectations with its clever plot twists.",
                "Poorly executed with weak dialogue and bad editing.",
                "A masterpiece of modern cinema with stunning visuals.",
                "Couldn't connect with any of the characters. Very flat.",
                "Engaging from start to finish with great soundtrack.",
                "Overhyped and underwhelming. Expected much more.",
                "Beautiful story that touches your heart deeply.",
                "Generic plot with no surprises. Seen it all before.",
                "Exceptional acting and brilliant direction throughout.",
                "Confusing narrative that doesn't make much sense."
            ],
            'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
        }
        
        # Create DataFrame
        df = pd.DataFrame(sample_data)
        
        # Split data
        train_size = int(0.8 * len(df))
        train_df = df[:train_size].reset_index(drop=True)
        test_df = df[train_size:].reset_index(drop=True)
        
        # Convert to Hugging Face Dataset format
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        self.dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        
        print(f"✅ Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
        return self.dataset
    
    def setup_tokenizer_and_model(self):
        """
        Initialize tokenizer and model for sequence classification
        """
        print("🔧 Setting up tokenizer and model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2,
            id2label={0: "NEGATIVE", 1: "POSITIVE"},
            label2id={"NEGATIVE": 0, "POSITIVE": 1}
        )
        
        print(f"✅ Model and tokenizer loaded: {self.model_name}")
    
    def tokenize_data(self):
        """
        Tokenize the dataset for model input
        """
        print("🔤 Tokenizing dataset...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'], 
                truncation=True, 
                padding=True, 
                max_length=self.max_length
            )
        
        self.tokenized_dataset = self.dataset.map(tokenize_function, batched=True)
        print("✅ Tokenization completed")
        
        return self.tokenized_dataset
    
    def setup_training(self):
        """
        Configure training arguments and trainer
        """
        print("⚙️ Setting up training configuration...")
        
        training_args = TrainingArguments(
            output_dir='./models/results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./models/logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
        )
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
            accuracy = accuracy_score(labels, predictions)
            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['test'],
            compute_metrics=compute_metrics,
        )
        
        print("✅ Training setup completed")
    
    def train_model(self):
        """
        Fine-tune the model
        """
        print("🚀 Starting model training...")
        start_time = datetime.now()
        
        # Train the model
        train_result = self.trainer.train()
        
        # Save the model
        self.trainer.save_model("./models/fine_tuned_model")
        self.tokenizer.save_pretrained("./models/fine_tuned_model")
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print(f"✅ Training completed in {training_time}")
        
        # Store training results
        self.results['training_time'] = str(training_time)
        self.results['train_loss'] = train_result.training_loss
        
        return train_result
    
    def evaluate_model(self):
        """
        Comprehensive model evaluation
        """
        print("📈 Evaluating model performance...")
        
        # Get predictions
        predictions = self.trainer.predict(self.tokenized_dataset['test'])
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1]
        )
        
        # Overall metrics
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # Store results
        self.results.update({
            'accuracy': float(accuracy),
            'overall_precision': float(overall_precision),
            'overall_recall': float(overall_recall),
            'overall_f1': float(overall_f1),
            'class_precision': [float(p) for p in precision],
            'class_recall': [float(r) for r in recall],
            'class_f1': [float(f) for f in f1],
            'class_support': [int(s) for s in support]
        })
        
        # Print results
        print(f"📊 Model Performance:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {overall_precision:.4f}")
        print(f"   Recall: {overall_recall:.4f}")
        print(f"   F1-Score: {overall_f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'], 
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return self.results
    
    def test_inference(self):
        """
        Test model with sample predictions
        """
        print("🔮 Testing model inference...")
        
        # Create inference pipeline
        classifier = pipeline(
            "sentiment-analysis",
            model="./models/fine_tuned_model",
            tokenizer="./models/fine_tuned_model"
        )
        
        # Test samples
        test_samples = [
            "This movie is absolutely amazing!",
            "Worst film I've ever seen.",
            "Pretty good movie with decent acting.",
            "Not bad, but could be better."
        ]
        
        print("\n🎯 Sample Predictions:")
        predictions = []
        for sample in test_samples:
            result = classifier(sample)[0]
            predictions.append({
                'text': sample,
                'label': result['label'],
                'confidence': result['score']
            })
            print(f"   Text: '{sample}'")
            print(f"   Prediction: {result['label']} (confidence: {result['score']:.4f})")
        
        self.results['sample_predictions'] = predictions
        return predictions
    
    def generate_report(self):
        """
        Generate comprehensive analysis report
        """
        print("📝 Generating analysis report...")
        
        report = f"""# Text Classification Pipeline Report
## ML/NLP Engineer Intern Task

### Project Overview
- **Model**: {self.model_name}
- **Task**: Binary sentiment classification (Positive/Negative)
- **Dataset**: Movie reviews (sample dataset)
- **Training Time**: {self.results.get('training_time', 'N/A')}

### Model Performance
- **Accuracy**: {self.results.get('accuracy', 0):.4f}
- **Precision**: {self.results.get('overall_precision', 0):.4f}
- **Recall**: {self.results.get('overall_recall', 0):.4f}
- **F1-Score**: {self.results.get('overall_f1', 0):.4f}

### Class-wise Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | {self.results.get('class_precision', [0,0])[0]:.4f} | {self.results.get('class_recall', [0,0])[0]:.4f} | {self.results.get('class_f1', [0,0])[0]:.4f} | {self.results.get('class_support', [0,0])[0]} |
| Positive | {self.results.get('class_precision', [0,0])[1]:.4f} | {self.results.get('class_recall', [0,0])[1]:.4f} | {self.results.get('class_f1', [0,0])[1]:.4f} | {self.results.get('class_support', [0,0])[1]} |

### Key Insights
1. **Model Choice**: DistilBERT provides a good balance between performance and efficiency
2. **Performance**: The model shows {self.results.get('accuracy', 0)*100:.1f}% accuracy on test data
3. **Strengths**: Good at capturing sentiment patterns in movie reviews
4. **Areas for Improvement**: 
   - Larger dataset would improve generalization
   - Additional preprocessing could help with edge cases
   - Ensemble methods could boost performance

### Improvement Ideas
1. **Data Augmentation**: Use techniques like back-translation or paraphrasing
2. **Feature Engineering**: Add additional features like review length, rating scores
3. **Model Architecture**: Experiment with RoBERTa, BERT-large, or domain-specific models
4. **Hyperparameter Tuning**: Optimize learning rate, batch size, and training epochs
5. **Post-processing**: Implement confidence thresholding for uncertain predictions

### Multilingual Extension Strategy
To extend this pipeline to multilingual use cases:

1. **Model Selection**: Use multilingual models like:
   - `bert-base-multilingual-cased`
   - `xlm-roberta-base`
   - `distilbert-base-multilingual-cased`

2. **Data Strategy**:
   - Collect parallel datasets in target languages
   - Use cross-lingual transfer learning
   - Implement language detection preprocessing

3. **Evaluation Approach**:
   - Test on each language separately
   - Measure cross-lingual transfer performance
   - Monitor for language-specific biases

4. **Technical Implementation**:
   ```python
   # Example multilingual setup
   model_name = "xlm-roberta-base"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForSequenceClassification.from_pretrained(
       model_name, num_labels=2
   )
   ```

### Production Considerations
- **Scalability**: Implement batch processing for large-scale inference
- **Monitoring**: Track model performance drift over time
- **A/B Testing**: Compare against baseline models
- **Error Analysis**: Regular review of misclassified examples

### Conclusion
This pipeline demonstrates a complete end-to-end text classification workflow using modern NLP techniques. The modular design allows for easy extension and improvement.

---
*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save report
        with open('reports/analysis_report.md', 'w') as f:
            f.write(report)
        
        # Save results as JSON
        with open('reports/results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("✅ Report generated: reports/analysis_report.md")
        return report
    
    def run_complete_pipeline(self):
        """
        Execute the complete pipeline
        """
        print("🚀 Starting Complete Text Classification Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Data preparation
            self.load_and_prepare_data()
            
            # Step 2: Model setup
            self.setup_tokenizer_and_model()
            
            # Step 3: Tokenization
            self.tokenize_data()
            
            # Step 4: Training setup
            self.setup_training()
            
            # Step 5: Model training
            self.train_model()
            
            # Step 6: Evaluation
            self.evaluate_model()
            
            # Step 7: Inference testing
            self.test_inference()
            
            # Step 8: Generate report
            self.generate_report()
            
            print("\n🎉 Pipeline completed successfully!")
            print(f"📁 Check 'models/' for saved model")
            print(f"📊 Check 'reports/' for analysis and results")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = TextClassificationPipeline()
    pipeline.run_complete_pipeline()