# DistilBERT IMDb Sentiment Analysis Model

## Model Description

This is a fine-tuned DistilBERT model for binary sentiment classification on IMDb movie reviews. The model classifies movie reviews as either positive or negative sentiment.

## Model Details

- **Model Type**: DistilBERT for Sequence Classification
- **Base Model**: distilbert-base-uncased
- **Task**: Binary sentiment classification
- **Language**: English
- **Dataset**: IMDb Movie Reviews (50K samples)

## Performance

- **Accuracy**: 93.48%
- **Precision**: 93.51%
- **Recall**: 93.48%
- **F1-Score**: 93.49%

## Training Details

- **Training Samples**: 25,000
- **Test Samples**: 25,000
- **Epochs**: 3
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW with linear warmup

## Usage

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./models/distilbert-imdb-sentiment')
tokenizer = DistilBertTokenizer.from_pretrained('./models/distilbert-imdb-sentiment')

# Example prediction
text = "This movie was absolutely amazing!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

## Labels

- `LABEL_0`: Negative sentiment
- `LABEL_1`: Positive sentiment

## Training Configuration

The model was trained using the Hugging Face Transformers library with the following configuration:
- Weight decay: 0.01
- Warmup steps: 500
- Evaluation strategy: per epoch
- Save strategy: per epoch
- Load best model at end: True 