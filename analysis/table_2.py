# Methods : Baseline 1, Baseline 2, Regularized
# Dataset : IMDB Regular Test Set.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import sys

filenames = ["../outputs/imdb_sentiment_classifier_naive_baseline/results_20240902_000819.csv",
            "../outputs/imdb_sentiment_causal_phrase_baseline/results_20240903_101058.csv",
            "../outputs/imdb_sentiment_regularized/results_20240911_094201.csv"]

final_df = pd.DataFrame()
for i in range(len(filenames)):
    df = pd.read_csv(filenames[i])
    if i == 0:
        df.columns = ["Model", "Epochs", "Train Loss", "Test Loss", "Accuracy", "Precision", "Recall", "F1"]
        df['Method'] = 'Naive Classifier'
    elif i==1:
        df.columns = ["Model", "Epochs", "Train Loss", "Test Loss", "Accuracy", "Precision", "Recall", "F1"]
        df['Method'] = 'Z Masked'
    else:
        df.columns = ["Model", "Epochs", "Reg_Epochs", "Train Loss", "Test Loss", "Accuracy", "Precision", "Recall", "F1"]
        df['Method'] = 'Causal Regularized (Lambda = 0.1)'
        df = df.drop(columns=['Reg_Epochs'])
    final_df = final_df._append(df, ignore_index=True)
final_df = final_df.replace(to_replace="electra_small_discriminator", value="electra")
final_df = final_df[final_df["Model"] != 't5_batch_4']
metrics = ["Train Loss", "Test Loss", "Accuracy", "Precision", "Recall", "F1"]
for epoch in [5,10]:
    fig, axes = plt.subplots(3, 2, figsize=(18, 10))
    fig.suptitle("Initial Performance Comparisons | Baseline 1, Baseline 2, Regularized | IMDB Dataset", fontsize=10)
    for metric, ax in zip(metrics, axes.flatten()):
        sns.barplot(data=final_df[final_df['Epochs'] == epoch], x="Model", y=metric, hue="Method", ax=ax, width=0.35, palette='dark:skyblue')
        ax.set_title(f"{metric} for " + str(epoch) + " Epochs")
        ax.tick_params(axis='x', rotation=40)
        ax.set_ylabel(metric)
        ax.set_xlabel("Model")
        ax.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig('images/table_2_' + str(int(epoch / 5))+ '.png')
    plt.show()