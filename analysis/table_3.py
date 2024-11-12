# Methods : Baseline 1, Baseline 2, Regularized
# Dataset : CF-LTD.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import sys

filename = "../outputs/ood_sentiment_test/results_20240916_160122.csv"
df = pd.read_csv(filename)
df.columns = ["Dataset", "Model", "Method", "Epochs", "Test Loss", "Accuracy", "Precision", "Recall", "F1"]
df = df.replace(to_replace="electra_small_discriminator", value="electra")
metrics = ["Test Loss", "Accuracy", "Precision", "Recall", "F1"]
for epoch in [5,10]:
        fig, axes = plt.subplots(2, 3, figsize=(20, 5))
        fig.suptitle("Performance Comparisons | Baseline 1, Baseline 2, Regularized | CF-LTD Dataset", fontsize=10)
        for metric, ax in zip(metrics, axes.flatten()):
                # sns.lineplot(data=df, x="Epochs", y=metric, hue="Model", markers=True, ax=ax)
                sns.barplot(data=df[df['Epochs'] == epoch], x="Model", y=metric, hue="Method",  ax=ax, width=0.35)
                ax.set_title(f"{metric} for " + str(epoch) + " Epochs")
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=40)
                ax.set_xlabel("Model")
                ax.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.savefig('images/table_3_' + str(int(epoch / 5))+ '.png')
        plt.show()