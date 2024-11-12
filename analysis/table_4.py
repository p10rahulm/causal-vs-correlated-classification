# Methods : Baseline 1, Baseline 2, Regularized
# Dataset : OOD Sentiment

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import sys

filename = "../outputs/ood_sentiment_test/results_20240916_150508.csv"
df = pd.read_csv(filename)
df.columns = ["Model", "Method", "Epochs", "Test Loss", "Accuracy", "Precision", "Recall", "F1"]
df = df.replace(to_replace="electra_small_discriminator", value="electra")
metrics = ["Test Loss", "Accuracy", "Precision", "Recall", "F1"]
for model in ['roberta', 'albert', 'distilbert', 'bert', 'electra', 't5']:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Performance Comparisons | Baseline 1, Baseline 2, Regularized | CF-LTD Dataset", fontsize=10)
        for metric, ax in zip(metrics, axes.flatten()):
                sns.barplot(data=df[df['Model'] == model], x="Method", y=metric, hue="Epochs",  ax=ax, width=0.35, palette='dark:green')
                ax.set_title(f"{metric} for " + model)
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=40)
                ax.set_xlabel("Method")
                ax.legend(title="Epochs", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.savefig('images/table_4_' + model+ '.png')
        plt.show()
