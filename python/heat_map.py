import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data for confusion matrices
data_large = {
    'Model': ['BERT', 'GPT-3', 'LSTM'],
    'TP': [234, 228, 228],
    'TN': [206, 201, 169],
    'FP': [17, 22, 54],
    'FN': [16, 22, 22]
}

data_medium = {
    'Model': ['BERT', 'GPT-3', 'LSTM'],
    'TP': [99, 94, 95],
    'TN': [86, 85, 70],
    'FP': [7, 10, 24],
    'FN': [8, 11, 11]
}

data_small = {
    'Model': ['BERT', 'GPT-3', 'LSTM'],
    'TP': [49, 48, 48],
    'TN': [44, 42, 36],
    'FP': [4, 5, 11],
    'FN': [3, 5, 5]
}

# Convert to DataFrames
df_large = pd.DataFrame(data_large).set_index('Model')
df_medium = pd.DataFrame(data_medium).set_index('Model')
df_small = pd.DataFrame(data_small).set_index('Model')

# Plot heatmaps
fig, axes = plt.subplots(1, 3, figsize=(9, 3))

sns.heatmap(df_large, annot=True, fmt='d', cmap='Blues', ax=axes[0], annot_kws={"size": 9})
axes[0].set_title('Large Dataset', fontsize=9)

sns.heatmap(df_medium, annot=True, fmt='d', cmap='Blues', ax=axes[1], annot_kws={"size": 9})
axes[1].set_title('Medium Dataset', fontsize=9)

sns.heatmap(df_small, annot=True, fmt='d', cmap='Blues', ax=axes[2], annot_kws={"size": 9})
axes[2].set_title('Small Dataset', fontsize=9)

# Add main title
plt.suptitle('Confusion Matrix', fontsize=12)

# Improve layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Show plot
plt.show()
