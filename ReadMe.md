# 🎬 IMDb Sentiment Analysis with DistilBERT
*SLM ML/NLP Engineer Internship Project by Sreevallabh Kakarala*

A production-ready sentiment analysis pipeline using DistilBERT to classify IMDb movie reviews as positive or negative sentiment.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![Transformers](https://img.shields.io/badge/Transformers-4.21%2B-green)
![DistilBERT](https://img.shields.io/badge/Model-DistilBERT-yellow)

## 🎯 Project Overview

This project implements a complete machine learning pipeline for sentiment analysis using the DistilBERT transformer model on the IMDb movie reviews dataset. The system achieves **93.48% accuracy** on 25,000 test samples.

### Key Features
- **🤖 DistilBERT Model**: Efficient transformer architecture for text classification
- **📊 Full IMDb Dataset**: 50K movie reviews (25K train + 25K test)
- **🔧 Production Ready**: Clean, modular, and well-documented code
- **📈 High Performance**: 93.48% accuracy with balanced precision/recall
- **🎨 Analysis Tools**: Comprehensive evaluation and visualization notebooks

## 📂 Project Structure

```
SLM-internship/
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── submission.md                 # Project submission details
├── deliverables.md              # Project requirements specification
├── train.py                     # Main training script
├── models/
│   └── distilbert-imdb-sentiment/  # Trained model artifacts
├── notebooks/
│   ├── data_exploration.ipynb   # Dataset analysis and visualization
│   ├── model_training.ipynb     # Training process and metrics
│   └── evaluation_analysis.ipynb # Model evaluation and results
├── reports/
│   ├── evaluation_metrics.json  # Performance metrics
│   └── model_report.md         # Detailed model analysis
└── src/
    ├── __init__.py              # Package initialization
    ├── config.py                # Configuration settings
    ├── data_preprocessing.py    # Data handling utilities
    ├── model_utils.py          # Model utility functions
    └── train_model.py          # Training implementation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ available RAM for training

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SLM-internship
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

#### Option 1: Main Training Script
```bash
python train.py
```

#### Option 2: Using the Training Module
```bash
cd src
python train_model.py
```

#### Option 3: Interactive Notebooks
```bash
jupyter notebook
```
Then open and run:
- `notebooks/data_exploration.ipynb` - Dataset analysis
- `notebooks/model_training.ipynb` - Training process
- `notebooks/evaluation_analysis.ipynb` - Results analysis

## ⚙️ Configuration

Modify training parameters in `src/config.py`:

```python
# Model Configuration
MODEL_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "num_labels": 2,
    "max_length": 512,
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-5,
    "num_epochs": 5,
    "warmup_steps": 500,
}

# Dataset Configuration
DATASET_CONFIG = {
    "dataset_name": "imdb",
    "use_full_dataset": True,  # Use full 50K samples
}
```

## 📊 Results

### Performance Metrics
- **🎯 Accuracy**: 93.48%
- **📈 Precision**: 93.51%
- **📈 Recall**: 93.48%
- **🏆 F1-Score**: 93.49%

### Class-wise Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 92.85% | 94.12% | 93.48% | 12,500 |
| Positive | 94.13% | 92.84% | 93.48% | 12,500 |

### Training Details
- **Model**: DistilBERT-base-uncased
- **Dataset**: IMDb movie reviews (50K samples)
- **Training Time**: ~1 hour 42 minutes
- **Epochs**: 3
- **Batch Size**: 16
- **Learning Rate**: 2e-5

## 📈 Model Architecture

**DistilBERT** (Distilled BERT) was chosen for this project because:

- **Efficiency**: 60% smaller than BERT-base while retaining 97% of performance
- **Speed**: 60% faster inference than BERT
- **Memory**: Lower memory requirements for training and deployment
- **Performance**: Excellent results on text classification tasks

## 🔧 Advanced Usage

### Custom Training Parameters
```bash
python train.py --help  # View available options
```

### Programmatic Usage
```python
from src.train_model import train_model
from src.config import MODEL_CONFIG, TRAINING_CONFIG

# Train with custom parameters
model, tokenizer = train_model(
    model_name="distilbert-base-uncased",
    num_epochs=3,
    batch_size=16,
    learning_rate=2e-5
)
```

### Making Predictions
```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load trained model
model_path = "models/distilbert-imdb-sentiment"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Make prediction
text = "This movie is absolutely amazing!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1)
    
print(f"Sentiment: {'Positive' if predicted_class == 1 else 'Negative'}")
print(f"Confidence: {predictions.max().item():.4f}")
```

## 📊 Data Analysis

The IMDb dataset contains:
- **50,000 total reviews** (balanced dataset)
- **25,000 training samples** (12,500 positive + 12,500 negative)
- **25,000 test samples** (12,500 positive + 12,500 negative)
- **Average review length**: ~230 words
- **Vocabulary size**: ~196K unique tokens

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**
   ```python
   # Reduce batch size in src/config.py
   TRAINING_CONFIG["batch_size"] = 8
   ```

2. **Import Errors**
   ```bash
   pip install --upgrade transformers torch datasets
   ```

3. **CUDA Issues**
   ```bash
   # Force CPU usage
   export CUDA_VISIBLE_DEVICES=""
   ```

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for model and dataset cache
- **GPU**: Optional (CUDA-compatible), CPU training supported

## 📚 Documentation

- **`reports/model_report.md`**: Comprehensive model analysis and methodology
- **`reports/evaluation_metrics.json`**: Detailed performance metrics
- **`submission.md`**: Project submission summary
- **`notebooks/`**: Interactive analysis and visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make changes and test thoroughly
4. Commit with clear messages (`git commit -m 'Add feature X'`)
5. Push and create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For the transformers library and dataset access
- **Google Research**: For the DistilBERT model architecture
- **IMDb**: For providing the movie review dataset
- **PyTorch Team**: For the deep learning framework

## 📞 Contact & Support

**Sreevallabh Kakarala**
- Project Repository: [GitHub Link]
- Issues: Use GitHub Issues for bug reports
- Documentation: Check `reports/` directory for detailed analysis

---

## 🎯 Project Highlights

This sentiment analysis pipeline demonstrates:

✅ **Professional ML Engineering**: Clean, modular, production-ready code  
✅ **Strong Performance**: 93.48% accuracy on real-world data  
✅ **Comprehensive Analysis**: Detailed evaluation and visualization  
✅ **Best Practices**: Proper project structure, documentation, and testing  
✅ **Reproducibility**: Clear setup instructions and configuration management  

**Built for the SLM ML/NLP Engineer Internship Program**

---

*Last Updated: June 2025*