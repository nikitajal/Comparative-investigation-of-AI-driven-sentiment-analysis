import matplotlib.pyplot as plt
import seaborn as sns

# Define the datasets and their corresponding performance metrics for each model
datasets = ['Large Dataset', 'Medium Dataset', 'Small Dataset']

# Metrics for each dataset and model
precision = {
    'Large Dataset': [93, 91, 80],
    'Medium Dataset': [93, 90, 79],
    'Small Dataset': [92, 90, 81]
}

recall = {
    'Large Dataset': [93, 91, 91],
    'Medium Dataset': [92, 89, 89],
    'Small Dataset': [94, 90, 90]
}

f1_score = {
    'Large Dataset': [93, 90, 85],
    'Medium Dataset': [92, 89, 83],
    'Small Dataset': [93, 90, 84]
}

# Define models
models = ['BERT', 'GPT-3', 'LSTM']

# Set a color palette with distinct and beautiful colors
palette = sns.color_palette("tab10")  # "tab10" provides distinct colors for up to 10 items

# Plot precision, recall, and F1-score for each dataset
plt.figure(figsize=(6, 4))

for i, dataset in enumerate(datasets):
    # Use distinct colors for each dataset
    color = palette[i]  # Use one color per dataset
    
    plt.plot(models, precision[dataset], marker='o', markersize=3, label=f'Precision ({dataset})', linestyle='-', color=color)
    plt.plot(models, recall[dataset], marker='s', markersize=3, label=f'Recall ({dataset})', linestyle='--', color=color)
    plt.plot(models, f1_score[dataset], marker='D', markersize=3, label=f'F1-Score ({dataset})', linestyle='-.', color=color)

# Adding labels and title
plt.xlabel('Models', fontsize=10)
plt.ylabel('Scores (%)', fontsize=10)
plt.title('Precision, Recall, and F1-Score Comparison Across Datasets', fontsize=10)

# Display the legend with appropriate font size and icon size
plt.legend(loc='best', fontsize=6, handlelength=2.5, handletextpad=1.5)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Customize tick parameters for better readability
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

# Display the plot
plt.tight_layout()
plt.show()
