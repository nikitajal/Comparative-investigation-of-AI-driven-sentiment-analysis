import openai
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Set API key
openai.api_key = 'sk-proj-SExiq15iM6Ng------------------------------'

# Load the dataset
df = pd.read_csv('test_amazon_reviews.csv')  # Replace with the path to your dataset

# Ensure the dataset has 'text' and 'label' columns
# If not, adjust the column names accordingly

# Function to get sentiment using GPT-3 with rate limit handling
def get_sentiment(text):
    try:
        response = openai.Completion.create(
            engine="davinci-002",
            prompt=f"Text: {text}\nSentiment (Positive, Negative, Neutral):",
            max_tokens=10
        )
        sentiment = response.choices[0].text.strip()
        return sentiment
    except openai.error.RateLimitError:
        print(f"Rate limit exceeded. Waiting for 60 seconds.")
        time.sleep(60)
        return get_sentiment(text)
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Apply the function to dataset and handle rate limits
predictions = []
for i, text in enumerate(df['text']):
    sentiment = get_sentiment(text)
    if sentiment is not None:
        predictions.append(sentiment)

    if (i + 1) % 3 == 0:  # Wait after every 3 requests
        time.sleep(60)

# Map predictions to numerical values
label_mapping = {'Positive': 1, 'Negative': 0, 'Neutral': 2}
predicted_labels = [label_mapping.get(pred, 2) for pred in predictions]

# Compute metrics
true_labels = df['label'][:len(predicted_labels)]  # Ensure true labels match predicted labels length
accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

# Save metrics to a file
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

with open('metrics.txt', 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")

# Compute and plot confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Save predictions to a CSV file
df_predicted = df[:len(predicted_labels)]  # Use only rows for which predictions were made
df_predicted['predicted_label'] = predicted_labels
df_predicted['predicted_sentiment'] = predictions
df_predicted.to_csv('predictions.csv', index=False)
