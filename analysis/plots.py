import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data preparation
data = {
    "Model": ["Roberta", "Roberta", "Albert", "Albert", "DistilBERT", "DistilBERT", "BERT", "BERT", 
            "Electra Small Discriminator", "Electra Small Discriminator", "T5", "T5", 
            "Roberta", "Roberta", "Albert", "Albert", "DistilBERT", "DistilBERT"],
    "Baseline": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    "Epochs": [5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10],
    "Train Loss": [0.3796, 0.3467, 0.2810, 0.2703, 0.3504, 0.3452, 0.3680, 0.3595, 0.5617, 0.5560, 0.6048, 0.5895,
                0.4352, 0.4172, 0.4469, 0.4327, 0.4191, 0.4070],
    "Test Loss": [0.3113, 0.2675, 0.2589, 0.2577, 0.3192, 0.3110, 0.3233, 0.3156, 0.5337, 0.5018, 0.5848, 0.5672,
                0.3918, 0.3736, 0.4316, 0.4223, 0.3982, 0.3902],
    "Accuracy": [0.8799, 0.8961, 0.8934, 0.8932, 0.8642, 0.8676, 0.8606, 0.8640, 0.7252, 0.7547, 0.6883, 0.7027,
                0.8304, 0.8367, 0.8007, 0.8057, 0.8183, 0.8218],
    "Precision": [0.8816, 0.8961, 0.8935, 0.8943, 0.8644, 0.8677, 0.8607, 0.8644, 0.7477, 0.7591, 0.6899, 0.7048,
                0.8310, 0.8367, 0.8018, 0.8063, 0.8190, 0.8219],
    "Recall": [0.8799, 0.8961, 0.8934, 0.8932, 0.8642, 0.8676, 0.8606, 0.8640, 0.7252, 0.7547, 0.6883, 0.7027,
            0.8304, 0.8367, 0.8007, 0.8057, 0.8183, 0.8218],
    "F1": [0.8797, 0.8961, 0.8934, 0.8931, 0.8642, 0.8676, 0.8606, 0.8640, 0.7188, 0.7537, 0.6876, 0.7020,
        0.8303, 0.8367, 0.8005, 0.8056, 0.8182, 0.8218]
}

df = pd.DataFrame(data)

# Plotting
metrics = ["Train Loss", "Test Loss", "Accuracy", "Precision", "Recall", "F1"]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Model Performance Comparison", fontsize=16)

# Create a line plot for each metric
for metric, ax in zip(metrics, axes.flatten()):
    sns.lineplot(data=df, x="Epochs", y=metric, hue="Model", style="Baseline", markers=True, ax=ax)
    ax.set_title(f"{metric} Over Epochs")
    ax.set_ylabel(metric)
    ax.set_xlabel("Epochs")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.show()
