# Methods : Regularized
# Dataset : OOD Sentiment

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import sys

filename = "../outputs/ood_sentiment_test_lambda/results_20240923_080818.csv"
df = pd.read_csv(filename)
df.columns = ["Dataset", "Model", "Epochs", "Lambda", "Test Loss", "Accuracy", "Precision", "Recall", "F1"]
df = df.replace(to_replace="electra_small_discriminator", value="electra")
metrics = ["Test Loss", "Accuracy", "Precision", "Recall", "F1"]
for dataset in set(df['Dataset']):
    temp_df = df[df['Dataset'] == dataset]
    for model in ['roberta', 'albert', 'distilbert', 'bert', 'electra']:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Performance Comparisons | Regularized | " + dataset +" Dataset", fontsize=10)
        for metric, ax in zip(metrics, axes.flatten()):
                sns.barplot(data=temp_df[temp_df['Model'] == model], x="Lambda", y=metric, hue="Epochs",  ax=ax, width=0.35)
                ax.set_title(f"{metric} for " + model)
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=40)
                ax.set_xlabel("Lambda")
                ax.legend(title="Epochs", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.savefig('images/table_6' +'_' + model + '_' + dataset + '.png')
        plt.show()