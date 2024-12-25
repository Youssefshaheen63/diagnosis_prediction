# Diagnosis Prediction using LLM and Transformers

## Overview
This project leverages the power of Transformer-based architectures, specifically BERT, to predict medical diagnoses from clinical notes. It provides an end-to-end pipeline including data preprocessing, model training, evaluation, and deployment, along with an interactive interface for real-time predictions.

## Key Features
- **Data Preprocessing**:
  - Cleans clinical notes by removing noise, special characters, and stopwords.
  - Encodes diagnoses into numerical labels for model compatibility.
- **Transformer-Based Model**:
  - Fine-tunes a pre-trained BERT model for multi-class classification.
  - Supports scalability and adaptability to various medical datasets.
- **Evaluation Metrics**:
  - Generates classification reports with precision, recall, and F1-scores.
  - Visualizes performance using confusion matrices.
- **Interactive Prediction Tool**:
  - Gradio-based interface allows users to input clinical notes and receive predicted diagnoses in real time.

## Quick Start
1. **Install Dependencies**:
   Install all required Python packages using:
   ```bash
   pip install -r requirements.txt
   ```


## Evaluation Metrics
The project uses the following metrics for evaluation:
- **Classification Report**: Detailed metrics for precision, recall, and F1-score for each diagnosis class.
- **Confusion Matrix**: Visual representation of prediction accuracy and errors across all classes.

## Deployment
The trained model, tokenizer, and label encoder are saved for future use. Reload the saved components for production environments:
```python
from transformers import BertForSequenceClassification, BertTokenizer
import pickle

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('./patient_model')
tokenizer = BertTokenizer.from_pretrained('./patient_model')

# Load the label encoder
with open("label_encoder.pkl", 'rb') as f:
    label_encoder = pickle.load(f)
```




