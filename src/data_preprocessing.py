"""
Data preprocessing utilities for the text classification pipeline
"""

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
import re
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Utilities for data loading and preprocessing"""
    
    def __init__(self, seed=42):
        self.seed = seed
    
    def load_imdb_dataset(self, dataset_size="small"):
        """
        Load IMDB dataset from Hugging Face
        
        Args:
            dataset_size (str): Size of dataset ("small", "medium", "full")
            
        Returns:
            DatasetDict: Processed dataset
        """
        try:
            dataset = load_dataset("imdb")
            
            # Determine subset size
            if dataset_size == "small":
                train_size, test_size = 1000, 200
            elif dataset_size == "medium":
                train_size, test_size = 5000, 1000
            else:  # full
                train_size = len(dataset['train'])
                test_size = len(dataset['test'])
            
            # Create subsets
            train_dataset = dataset['train'].shuffle(seed=self.seed).select(range(min(train_size, len(dataset['train']))))
            test_dataset = dataset['test'].shuffle(seed=self.seed).select(range(min(test_size, len(dataset['test']))))
            
            return DatasetDict({
                'train': train_dataset,
                'test': test_dataset
            })
            
        except Exception as e:
            logger.error(f"Error loading IMDB dataset: {e}")
            return None
    
    def clean_text(self, text):
        """
        Clean and normalize text data
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def create_sample_dataset(self):
        """Create a sample dataset for testing"""
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
                "The movie exceeded my expectations with its clever plot twists.",
                "Poorly executed with weak dialogue and bad editing.",
                "A masterpiece of modern cinema with stunning visuals.",
                "Couldn't connect with any of the characters. Very flat.",
                "Engaging from start to finish with great soundtrack.",
                "Overhyped and underwhelming. Expected much more.",
                "Beautiful story that touches your heart deeply.",
                "Generic plot with no surprises. Seen it all before.",
                "Exceptional acting and brilliant direction throughout.",
                "Confusing narrative that doesn't make much sense.",
                "Incredible visual effects and compelling storyline throughout.",
                "Disappointing sequel that fails to live up to the original.",
                "Heartwarming tale with excellent character development.",
                "Predictable plot with wooden performances from the cast.",
                "Stunning cinematography makes up for the weak script.",
                "Poorly paced with too many unnecessary subplots.",
                "Brilliant performances by the entire ensemble cast.",
                "Generic action movie with no emotional depth whatsoever.",
                "Thought-provoking film that stays with you long after.",
                "Complete waste of talent and budget. Utterly forgettable."
            ],
            'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Apply text cleaning
        df['text'] = df['text'].apply(self.clean_text)
        
        # Split data
        train_size = int(0.8 * len(df))
        train_df = df[:train_size].reset_index(drop=True)
        test_df = df[train_size:].reset_index(drop=True)
        
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        return DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
    
    def get_dataset_statistics(self, dataset):
        """
        Get statistics about the dataset
        
        Args:
            dataset (DatasetDict): Input dataset
            
        Returns:
            dict: Dataset statistics
        """
        stats = {}
        
        for split in dataset.keys():
            texts = dataset[split]['text']
            labels = dataset[split]['label']
            
            stats[split] = {
                'num_samples': len(texts),
                'avg_length': np.mean([len(text.split()) for text in texts]),
                'max_length': max([len(text.split()) for text in texts]),
                'min_length': min([len(text.split()) for text in texts]),
                'label_distribution': {
                    'positive': sum(labels),
                    'negative': len(labels) - sum(labels)
                }
            }
        
        return stats 