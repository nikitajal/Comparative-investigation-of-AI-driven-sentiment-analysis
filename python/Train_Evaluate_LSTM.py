import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
import json
import time
import os

# Ensure the LSTM directory exists
os.makedirs('LSTM', exist_ok=True)

# Load the split datasets
train_data = pd.read_csv('train_amazon_reviews.csv')
val_data = pd.read_csv('val_amazon_reviews.csv')
test_data = pd.read_csv('test_amazon_reviews.csv')

# Encode the labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['sentiment_label'])
y_val = label_encoder.transform(val_data['sentiment_label'])
y_test = label_encoder.transform(test_data['sentiment_label'])

# Tokenize the text data
class AmazonReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(text)
        tokens = tokens[:self.max_length]  # Truncate to max_length
        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens, label

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts_padded, labels

# Simple tokenizer function
def tokenizer(text):
    return [ord(c) for c in text]  # Replace with a proper tokenizer in practice

max_seq_length = 128
train_dataset = AmazonReviewsDataset(train_data['text'].tolist(), y_train, tokenizer, max_seq_length)
val_dataset = AmazonReviewsDataset(val_data['text'].tolist(), y_val, tokenizer, max_seq_length)
test_dataset = AmazonReviewsDataset(test_data['text'].tolist(), y_test, tokenizer, max_seq_length)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Build the LSTM model using PyTorch
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Use the last hidden state
        out = self.fc(lstm_out)
        return out

# Define hyperparameters
vocab_size = 256  # Adjust this based on tokenizer
embedding_dim = 100
hidden_dim = 100
output_dim = 3
n_layers = 2
dropout = 0.2

model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs):
    train_losses = []
    val_losses = []
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for texts, labels in val_loader:
                outputs = model(texts)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_losses.append(val_loss / len(val_loader))

        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}, Val Accuracy: {val_accuracy}")

    return train_losses, val_losses

# Train the model
n_epochs = 5
start_time = time.time()
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs)
end_time = time.time()
training_time = end_time - start_time

# Plot validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Validation Loss over Epochs')
plt.savefig('LSTM/validation_loss.png')
plt.close()

# Evaluate on test set
model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts)
        test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# Compute metrics for the test set
accuracy = accuracy_score(test_labels, test_preds)
precision, recall, f1_score, support = precision_recall_fscore_support(labels, test_preds, average='weighted')

# Print metrics
print(f"LSTM Test Accuracy: {accuracy}\n")
print(f"LSTM Test Precision: {precision}\n")
print(f"LSTM Test Recall: {recall}\n")
print(f"LSTM Test F1-score: {f1_score}\n")
print(f"LSTM Training Time: {training_time} seconds")

# Compute confusion matrix
test_conf_matrix = confusion_matrix(test_labels, test_preds)

# Ensure that lengths match
assert len(test_data['text']) == len(test_labels) == len(test_preds), "Lengths of text, labels, and predictions must match"

# Construct DataFrame
test_data_with_preds = pd.DataFrame({
    'text': test_data['text'],
    'true_label': test_labels,
    'predicted_label': test_preds
})

# Save to CSV
test_data_with_preds.to_csv('LSTM/lstm_test_predictions.csv', index=False)

# Plot confusion matrices
plt.figure(figsize=(8, 6))
sns.heatmap(test_conf_matrix, annot=True, fmt='d')
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('LSTM/test_confusion_matrix.png')
plt.close()

# Calculate True Positives, True Negatives, False Positives, False Negatives
tn, fp, fn, tp = test_conf_matrix.ravel()
metrics_text = (
    f"True Positives (TP): {tp}\n"
    f"True Negatives (TN): {tn}\n"
    f"False Positives (FP): {fp}\n"
    f"False Negatives (FN): {fn}\n"
)

# Print metrics
print(metrics_text)

# Save the metrics to a text file
with open('LSTM/lstm_confusion_metrics.txt', 'w') as f:
    f.write(metrics_text)

# Save predictions and metrics to files
metrics = {
    "test_accuracy": test_accuracy,
    "test_precision": test_precision,
    "test_recall": test_recall,
    "test_f1_score": test_f1,
    "training_time": training_time
}

with open('LSTM/lstm_computational_metrics.json', 'w') as f:
    json.dump(metrics, f)
