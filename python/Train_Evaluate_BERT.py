import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure the BERT directory exists
os.makedirs('BERT', exist_ok=True)

# Load the split datasets
train_data = pd.read_csv('train_amazon_reviews.csv')
val_data = pd.read_csv('val_amazon_reviews.csv')
test_data = pd.read_csv('test_amazon_reviews.csv')

# Display the first few rows of the training dataset to understand its structure
print(train_data.head())

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Tokenize the data
def tokenize_function(texts):
    return tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128)

train_encodings = tokenize_function(train_data['text'])
val_encodings = tokenize_function(val_data['text'])
test_encodings = tokenize_function(test_data['text'])

# Create Dataset class
class AmazonReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = AmazonReviewsDataset(train_encodings, train_data['sentiment_label'].tolist())
val_dataset = AmazonReviewsDataset(val_encodings, val_data['sentiment_label'].tolist())
test_dataset = AmazonReviewsDataset(test_encodings, test_data['sentiment_label'].tolist())

# Define DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Define metric computation function
def compute_metrics(preds, labels):
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return accuracy, precision, recall, f1

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model and record training time
start_time = time.time()
train_result = trainer.train()
end_time = time.time()
training_time = end_time - start_time

# Save the trained model
# model.save_pretrained('./saved_model')
# tokenizer.save_pretrained('./saved_model')

# Predict on the test set
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

# Compute metrics
accuracy = accuracy_score(labels, preds)
precision, recall, f1 = precision_recall_fscore_support(labels, preds, average='weighted')

# Print metrics
print(f"BERT Test Accuracy: {accuracy}")
print(f"BERT Test Precision: {precision}")
print(f"BERT Test Recall: {recall}")
print(f"BERT Test F1-score: {f1}")
print(f"BERT Training Time: {training_time} seconds")

# Save predictions and metrics to files
test_data_with_preds = pd.DataFrame({
    'text': test_data['text'],
    'true_label': labels,
    'predicted_label': preds
})
test_data_with_preds.to_csv('BERT/bert_test_predictions.csv', index=False)

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "training_time": training_time
}

with open('BERT/bert_computational_metrics.json', 'w') as f:
    json.dump(metrics, f)


# Compute and plot confusion matrix
conf_matrix = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('BERT/test_confusion_matrix.png')
plt.close()

# Calculate True Positives, True Negatives, False Positives, False Negatives
tn, fp, fn, tp = conf_matrix.ravel()
conf_matrix_text = (
    f"True Positives (TP): {tp}\n"
    f"True Negatives (TN): {tn}\n"
    f"False Positives (FP): {fp}\n"
    f"False Negatives (FN): {fn}\n"
)

# Print confusion matrix metrics
print(conf_matrix_text)

# Save the confusion matrix metrics to a text file
with open('BERT/confusion_matrix.txt', 'w') as f:
    f.write(conf_matrix_text)

# Extract and plot validation loss from logs
try:
    log_history = trainer.state.log_history
    val_loss = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_loss)+1), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Validation Loss over Epochs')
    plt.savefig('BERT/validation_loss.png')
    plt.close()
except:
    print("Could not extract validation loss from log history.")

    # Attempt to plot validation loss directly from train_result
    try:
        training_loss = train_result.metrics['train_loss']  # Adjust if this is not correct
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(training_loss)+1), training_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Validation Loss over Epochs')
        plt.savefig('BERT/validation_loss.png')
        plt.close()
    except:
        print("Could not extract training loss from train_result metrics.")

        # If both attempts fail, manually track losses in the training loop
        num_epochs = 3
        training_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
                optimizer.zero_grad()
                inputs = {key: val.to('cuda') for key, val in batch.items() if key != 'labels'}
                labels = batch['labels'].to('cuda')
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            training_losses.append(avg_train_loss)

            model.eval()
            total_val_loss = 0
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    inputs = {key: val.to('cuda') for key, val in batch.items() if key != 'labels'}
                    labels = batch['labels'].to('cuda')
                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss
                    total_val_loss += loss.item()
                    val_preds.extend(outputs.logits.argmax(dim=1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            val_accuracy, val_precision, val_recall, val_f1 = compute_metrics(val_preds, val_labels)
           # Store the formatted string in a variable
            output_str = f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"

            # Print the stored string
            print(output_str)

