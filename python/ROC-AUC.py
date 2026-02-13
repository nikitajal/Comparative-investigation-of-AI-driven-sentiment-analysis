import matplotlib.pyplot as plt
from sklearn.metrics import auc

# Data for datasets
datasets = {
    "Large Dataset": {
        "BERT": {"TP": 234, "TN": 206, "FP": 17, "FN": 16},
        "GPT-3": {"TP": 228, "TN": 201, "FP": 22, "FN": 22},
        "LSTM": {"TP": 228, "TN": 169, "FP": 54, "FN": 22}
    },
    "Medium Dataset": {
        "BERT": {"TP": 99, "TN": 86, "FP": 7, "FN": 8},
        "GPT-3": {"TP": 94, "TN": 85, "FP": 10, "FN": 11},
        "LSTM": {"TP": 95, "TN": 70, "FP": 24, "FN": 11}
    },
    "Small Dataset": {
        "BERT": {"TP": 49, "TN": 44, "FP": 4, "FN": 3},
        "GPT-3": {"TP": 48, "TN": 42, "FP": 5, "FN": 5},
        "LSTM": {"TP": 48, "TN": 36, "FP": 11, "FN": 5}
    }
}

# Function to calculate the ROC curve
def get_roc_curve(data):
    tpr = data["TP"] / (data["TP"] + data["FN"])
    fpr = data["FP"] / (data["FP"] + data["TN"])
    return tpr, fpr

# Function to plot ROC curves
def plot_roc_curves(ax, dataset, title, colors):
    for model, color in colors.items():
        tpr, fpr = get_roc_curve(dataset[model])
        roc_auc = auc([0, fpr, 1], [0, tpr, 1])
        linestyle = '-' if model == 'BERT' else ('--' if model == 'GPT-3' else ':')
        ax.plot([0, fpr, 1], [0, tpr, 1], linestyle=linestyle, color=color, lw=2, label=f'{model} (AUC = {roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True)

fig, axs = plt.subplots(1, 3, figsize=(9, 4))

# Define colors for models
colors = {
    "BERT": "#1f77b4",  # blue
    "GPT-3": "#2ca02c", # green
    "LSTM": "#d62728"   # red
}

# Plot ROC curves for each dataset
plot_roc_curves(axs[0], datasets["Large Dataset"], 'Large Dataset', colors)
plot_roc_curves(axs[1], datasets["Medium Dataset"], 'Medium Dataset', colors)
plot_roc_curves(axs[2], datasets["Small Dataset"], 'Small Dataset', colors)

# Overall title for the figure
plt.suptitle('ROC Curves', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for the suptitle
plt.show()
