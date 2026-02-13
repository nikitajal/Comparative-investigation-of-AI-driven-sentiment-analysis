# Comparative Investigation of AI-Driven Sentiment Analysis Models
## A Study of BERT, GPT-3, and LSTM on the Amazon Reviews Dataset

---

## Project Overview

This project presents a comparative analysis of three sentiment analysis models:

- BERT (Bidirectional Encoder Representations from Transformers)
- GPT-3 (Generative Pre-trained Transformer-3)
- LSTM (Long Short-Term Memory Network)

The objective is to evaluate and compare deep learning and transformer-based architectures for sentiment classification using the Amazon Reviews dataset. The study focuses on model accuracy, robustness, scalability, and performance consistency across different dataset batch sizes.

---

## Objectives

- Perform sentiment classification on real-world customer review data.
- Compare LSTM, BERT, and GPT-3 performance.
- Evaluate model stability across varying dataset batch sizes.
- Analyze generalization using a multi-batch evaluation strategy.
- Visualize model performance using industry-standard evaluation metrics.

---

## Cross-Batch Evaluation Strategy

To evaluate robustness and performance stability, the dataset was divided into three batches:

- **Small Batch:** 500 rows  
- **Medium Batch:** 1,000 rows  
- **Large Batch:** 2,566 rows  

Each model (LSTM, BERT, GPT-3) was trained and evaluated on all three batches. This approach allows comparison of model behavior across increasing dataset sizes and ensures reproducibility.

---

## Dataset Information

Only the **Medium Batch dataset** is included. Models were run on a personal laptop (Ryzen processor), so smaller batches were used to ensure reasonable runtime.

The original Amazon Reviews dataset can be downloaded from Kaggle:  
*(Enhancing-Product-Design-through-AI-Driven-Sentiment-Analysis-of-Amazon-Reviews-using-BERT/synthetic_data.csv)*  

---

## Dataset Files Included

dataset/
│── amazon_reviews_dataset.csv
│── processed_amazon_reviews.csv
│── train_amazon_reviews.csv
│── val_amazon_reviews.csv
│── test_amazon_reviews.csv


---

## Models Implemented

### 1.LSTM
- Embedding layer with LSTM architecture
- Implemented using PyTorch
- Tokenization and padding applied
- Evaluated using validation loss and ROC-AUC

### 2.BERT
- Fine-tuned using Hugging Face Transformers
- Implemented with Trainer API and custom training arguments
- Context-aware embeddings
- Evaluated across all dataset batches

### 3.GPT-3
- Prompt-based sentiment classification
- Zero-shot / few-shot evaluation strategy
- Compared against supervised models

---

## Evaluation Metrics

Models were evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  
- ROC Curve  
- ROC-AUC Score  
- Precision-Recall Curve  
- Validation Loss  

All visualizations are stored in the `/graphs` directory.

---

## Technologies & Libraries Used

**Programming Language:** Python  

**Data Processing:** pandas, numpy, json, os, time  

**Deep Learning & NLP:** PyTorch, torch.nn, torch.optim, torch.utils.data (Dataset, DataLoader), transformers (BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW)  

**Model Evaluation:** scikit-learn (accuracy_score, precision_recall_fscore_support, confusion_matrix, LabelEncoder)  

**Visualization:** matplotlib, seaborn  

**Utilities:** tqdm, openai (for GPT-3 evaluation)

---

## Project Structure

Project Structure
├── dataset/
├── graphs/
│   ├── confusionMatrix.png
│   ├── models_accuracy.png
│   ├── multi-line_graph.png
│   ├── precision-recall-curve.png
│   ├── roc curves.png
│   ├── validation_loss_bert.png
│   └── validation_loss_lstm.png
├── python/
│   ├── ROC-AUC.py
│   ├── Train_Evaluate_BERT.py
│   ├── Train_Evaluate_GPT-3.py
│   ├── Train_Evaluate_LSTM.py
│   ├── batches_accuracy.py
│   ├── heat_map.py
│   ├── multi-line.py
│   └── precision_recall.py
├── screenshots/
│   ├── label_distribution_large_dataset.png
│   ├── label_distribution_medium_dataset.png
│   ├── label_distribution_small_dataset.png
│   ├── sample_distribution_large_dataset.png
│   ├── sample_distribution_medium_dataset.png
│   └── sample_distribution_small_dataset.png
└── README.md

---

## Key Findings

- **BERT** consistently achieved the highest accuracy, precision, recall, and F1-score across all dataset batches, making it the most robust and reliable model for sentiment classification.  
- **GPT-3** performed slightly below BERT but still offered strong accuracy, making it a viable alternative when high performance is required and computational resources are available.  
- **LSTM** trained extremely quickly but lagged in accuracy and precision, resulting in more misclassifications. Its speed makes it suitable for applications where rapid deployment is more important than peak accuracy.  
- **Trade-offs**: BERT and GPT-3 provide high accuracy but require substantial computation time, while LSTM offers low latency at the cost of reduced performance.  
- **Practical Implications**:  
  - Use **BERT** for applications needing high accuracy and deep understanding, such as detailed customer feedback analysis.  
  - Use **GPT-3** when strong performance is needed and longer processing time is acceptable.  
  - Use **LSTM** for real-time or resource-limited scenarios where speed is critical.  

---

## Future Improvements

- Implement **k-fold cross-validation** on each batch to further validate model stability.  
- Hyperparameter tuning for all models.  
- Deployment using Streamlit or Flask.  
- Experiment with additional transformer architectures for performance benchmarking.

---

## Author

Nikita
Aspiring Data Analyst | NLP & Machine Learning Enthusiast



Project Structure
├── dataset/
├── graphs/
│   ├── confusionMatrix.png
│   ├── models_accuracy.png
│   ├── multi-line_graph.png
│   ├── precision-recall-curve.png
│   ├── roc curves.png
│   ├── validation_loss_bert.png
│   └── validation_loss_lstm.png
├── python/
│   ├── ROC-AUC.py
│   ├── Train_Evaluate_BERT.py
│   ├── Train_Evaluate_GPT-3.py
│   ├── Train_Evaluate_LSTM.py
│   ├── batches_accuracy.py
│   ├── heat_map.py
│   ├── multi-line.py
│   └── precision_recall.py
├── screenshots/
│   ├── label_distribution_large_dataset.png
│   ├── label_distribution_medium_dataset.png
│   ├── label_distribution_small_dataset.png
│   ├── sample_distribution_large_dataset.png
│   ├── sample_distribution_medium_dataset.png
│   └── sample_distribution_small_dataset.png
└── README.md

