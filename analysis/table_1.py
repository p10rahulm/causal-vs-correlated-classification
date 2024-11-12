# Methods : Baseline 1, Baseline 2
# Dataset : IMDB Regular Test Set.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import sys

filename = "../outputs/imdb_sentiment_classifier_naive_baseline/results_20240902_000819.csv"
df = pd.read_csv(filename)
df = df.replace(to_replace="electra_small_discriminator", value="electra")
df.columns = ["Model", "Epochs", "Train Loss", "Test Loss", "Accuracy", "Precision", "Recall", "F1"]
metrics = ["Train Loss", "Test Loss", "Accuracy", "Precision", "Recall", "F1"]
fig, axes = plt.subplots(3, 2, figsize=(18, 10))
fig.suptitle("Initial Model Performances | Baselines 1 and 2 | IMDB Dataset", fontsize=10)
for metric, ax in zip(metrics, axes.flatten()):
        # sns.lineplot(data=df, x="Model", y=metric, hue="Epochs", style= "Epochs", markers=True, ax=ax)
        sns.barplot(data=df, x="Model", y=metric, hue="Epochs",  ax=ax, width=0.35)
        ax.set_title(f"{metric} Over Epochs")
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=40)
        ax.set_xlabel("Models")
        ax.legend(title="Epochs", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.savefig('images/table_1.png')
plt.show()