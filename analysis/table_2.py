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

metrics = ["Train Loss", "Test Loss", "Accuracy", "Precision", "Recall", "F1"]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Model Performance Comparison", fontsize=16)
for metric, ax in zip(metrics, axes.flatten()):
    sns.lineplot(data=final_df, x="Epochs", y=metric, hue="Model", size="Method", markers=True, ax=ax)
    ax.set_title(f"{metric} Over Epochs")
    ax.set_ylabel(metric)
    ax.set_xlabel("Epochs")
    ax.legend_.remove()  
handles, labels = axes[0, 0].get_legend_handles_labels() 
fig.legend(handles, labels, title="Model", loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))
plt.tight_layout(rect=[0, 0, 0.9, 0.95])
plt.savefig('images/table_2.png')
plt.show()