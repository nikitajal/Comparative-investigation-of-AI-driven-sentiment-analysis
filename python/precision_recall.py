import matplotlib.pyplot as plt

# Data for datasets
datasets = {
    "Large Dataset": {
        "BERT": {"Precision": [0.93, 0.91, 0.80], "Recall": [0.93, 0.91, 0.91]},
        "GPT-3": {"Precision": [0.93, 0.91, 0.80], "Recall": [0.93, 0.91, 0.91]},
        "LSTM": {"Precision": [0.93, 0.91, 0.80], "Recall": [0.93, 0.91, 0.91]}
    },
    "Medium Dataset": {
        "BERT": {"Precision": [0.93, 0.92, 0.79], "Recall": [0.92, 0.89, 0.89]},
        "GPT-3": {"Precision": [0.93, 0.92, 0.79], "Recall": [0.92, 0.89, 0.89]},
        "LSTM": {"Precision": [0.93, 0.92, 0.79], "Recall": [0.92, 0.89, 0.89]}
    },
    "Small Dataset": {
        "BERT": {"Precision": [0.92, 0.90, 0.81], "Recall": [0.94, 0.90, 0.90]},
        "GPT-3": {"Precision": [0.92, 0.90, 0.81], "Recall": [0.94, 0.90, 0.90]},
        "LSTM": {"Precision": [0.92, 0.90, 0.81], "Recall": [0.94, 0.90, 0.90]}
    }
}

# Colors for models
colors = {
    "BERT": "#1f77b4",  # blue
    "GPT-3": "#2ca02c", # green
    "LSTM": "#d62728"   # red
}

# Function to plot Precision-Recall curves
def plot_pr_curves(ax, dataset, title):
    for model, color in colors.items():
        precision = dataset[model]["Precision"]
        recall = dataset[model]["Recall"]
        ax.plot(recall, precision, color=color, lw=2, label=model)
    
    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True)

fig, axs = plt.subplots(1, 3, figsize=(9, 4))  # 1 row, 3 columns

# Plot Precision-Recall curves for each dataset
plot_pr_curves(axs[0], datasets["Large Dataset"], 'Large Dataset')
plot_pr_curves(axs[1], datasets["Medium Dataset"], 'Medium Dataset')
plot_pr_curves(axs[2], datasets["Small Dataset"], 'Small Dataset')

# Overall title for the figure
plt.suptitle('Precision-Recall Curves', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for the suptitle
plt.show()
