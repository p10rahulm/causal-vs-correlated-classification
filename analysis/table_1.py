import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import sys

filename = "../outputs/imdb_sentiment_classifier_naive_baseline/results_20240902_000819.csv"
df = pd.read_csv(filename)
df.columns = ["Model", "Epochs", "Train Loss", "Test Loss", "Accuracy", "Precision", "Recall", "F1"]
metrics = ["Train Loss", "Test Loss", "Accuracy", "Precision", "Recall", "F1"]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Model Performance Comparison", fontsize=16)
for metric, ax in zip(metrics, axes.flatten()):
        sns.lineplot(data=df, x="Epochs", y=metric, hue="Model", markers=True, ax=ax)
        ax.set_title(f"{metric} Over Epochs")
        ax.set_ylabel(metric)
        ax.set_xlabel("Epochs")
        ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.savefig('images/table_1.png')
plt.show()