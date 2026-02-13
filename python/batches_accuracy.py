import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['BERT', 'GPT-3', 'LSTM']
large_dataset = [93, 91, 83]
medium_dataset = [92, 89, 82]
small_dataset = [93, 90, 84]

# Bar width
bar_height = 0.2

# Positions of the bars on the y-axis
r1 = np.arange(len(models))
r2 = [x + bar_height for x in r1]
r3 = [x + bar_height for x in r2]

# Create horizontal bar plot
plt.figure(figsize=(8, 4))
plt.barh(r1, large_dataset, color='#1f77b4', height=bar_height, edgecolor='grey', label='Large Dataset')
plt.barh(r2, medium_dataset, color='#2ca02c', height=bar_height, edgecolor='grey', label='Medium Dataset')
plt.barh(r3, small_dataset, color='#d62728', height=bar_height, edgecolor='grey', label='Small Dataset')

# Add labels and title
plt.ylabel('Models', fontweight='bold')
plt.xlabel('Accuracy (%)', fontweight='bold')
plt.title('Model Accuracy Across Different Dataset Sizes', fontweight='bold')
plt.yticks([r + bar_height for r in range(len(models))], models)
plt.xlim(0, 100)

# Add legend
plt.legend()

# Add accuracy values on bars
for i in range(len(models)):
    plt.text(large_dataset[i] + 1, r1[i] - bar_height / 3, str(large_dataset[i]) + '%', color='#1f77b4', va='center', fontweight='bold')
    plt.text(medium_dataset[i] + 1, r2[i] - bar_height / 3, str(medium_dataset[i]) + '%', color='#2ca02c', va='center', fontweight='bold')
    plt.text(small_dataset[i] + 1, r3[i] - bar_height / 3, str(small_dataset[i]) + '%', color='#d62728', va='center', fontweight='bold')

# Show plot
plt.tight_layout()
plt.show()
