"""
Unit tests for the text classification pipeline
"""

import unittest
import sys
import os
sys.path.append('../src')

from data_preprocessing import DataPreprocessor
from evaluation import ModelEvaluator
import numpy as np

class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class"""
    
    def setUp(self):
        self.preprocessor = DataPreprocessor(seed=42)
    
    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        # Test HTML tag removal
        text_with_html = "This is <b>bold</b> text with <br/> tags"
        cleaned = self.preprocessor.clean_text(text_with_html)
        self.assertNotIn('<', cleaned)
        self.assertNotIn('>', cleaned)
        
        # Test whitespace normalization
        text_with_spaces = "This   has   multiple    spaces"
        cleaned = self.preprocessor.clean_text(text_with_spaces)
        self.assertNotIn('  ', cleaned)  # No double spaces
        
        # Test empty input
        self.assertEqual(self.preprocessor.clean_text(""), "")
        self.assertEqual(self.preprocessor.clean_text(None), "")
    
    def test_sample_dataset_creation(self):
        """Test sample dataset creation"""
        dataset = self.preprocessor.create_sample_dataset()
        
        # Check dataset structure
        self.assertIn('train', dataset)
        self.assertIn('test', dataset)
        
        # Check data integrity
        self.assertGreater(len(dataset['train']), 0)
        self.assertGreater(len(dataset['test']), 0)
        
        # Check label distribution
        train_labels = dataset['train']['label']
        self.assertTrue(all(label in [0, 1] for label in train_labels))

class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class"""
    
    def setUp(self):
        self.evaluator = ModelEvaluator()
    
    def test_compute_metrics(self):
        """Test metrics computation"""
        # Create sample predictions
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        metrics = self.evaluator.compute_metrics(y_true, y_pred)
        
        # Check metric structure
        self.assertIn('accuracy', metrics)
        self.assertIn('precision_weighted', metrics)
        self.assertIn('recall_weighted', metrics)
        self.assertIn('f1_weighted', metrics)
        self.assertIn('class_metrics', metrics)
        
        # Check metric ranges
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Mock metrics
        metrics = {
            'accuracy': 0.85,
            'precision_weighted': 0.83,
            'recall_weighted': 0.85,
            'f1_weighted': 0.84,
            'class_metrics': {
                'precision': [0.80, 0.86],
                'recall': [0.82, 0.88],
                'f1': [0.81, 0.87],
                'support': [50, 50]
            }
        }
        
        summary = self.evaluator.create_performance_summary(metrics)
        
        # Check summary content
        self.assertIn('accuracy', summary.lower())
        self.assertIn('precision', summary.lower())
        self.assertIn('recall', summary.lower())
        self.assertIn('f1', summary.lower())

class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized"""
        try:
            from main import TextClassificationPipeline
            pipeline = TextClassificationPipeline(model_name="distilbert-base-uncased")
            self.assertIsNotNone(pipeline)
            self.assertEqual(pipeline.model_name, "distilbert-base-uncased")
        except ImportError:
            self.skipTest("Main pipeline module not available")
    
    def test_sample_data_flow(self):
        """Test data can flow through preprocessing"""
        preprocessor = DataPreprocessor()
        dataset = preprocessor.create_sample_dataset()
        
        # Check we can get dataset statistics
        stats = preprocessor.get_dataset_statistics(dataset)
        self.assertIn('train', stats)
        self.assertIn('test', stats)
        
        for split in ['train', 'test']:
            self.assertIn('num_samples', stats[split])
            self.assertIn('avg_length', stats[split])
            self.assertIn('label_distribution', stats[split])

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataPreprocessor))
    test_suite.addTest(unittest.makeSuite(TestModelEvaluator))
    test_suite.addTest(unittest.makeSuite(TestPipelineIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}") 