# Methods : Baseline 1, Baseline 2, Regularized
# Dataset : OOD Sentiment

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import sys

filename = "../outputs/ood_sentiment_test/results_20240919_141957.csv"
df = pd.read_csv(filename)
print(df.head())
# df.columns = ["Dataset", "Model", "Method", "Epochs", "Test Loss", "Accuracy", "Precision", "Recall", "F1"]
# metrics = ["Test Loss", "Accuracy", "Precision", "Recall", "F1"]
# fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# fig.suptitle("Model Performance Comparison", fontsize=16)
# for metric, ax in zip(metrics, axes.flatten()):
#         sns.lineplot(data=df, x="Epochs", y=metric, hue="Model", markers=True, ax=ax)
#         ax.set_title(f"{metric} Over Epochs")
#         ax.set_ylabel(metric)
#         ax.set_xlabel("Epochs")
#         ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.tight_layout(rect=[0, 0, 0.9, 0.95])
# plt.savefig('images/table_5-1.png')
# plt.show()

filename = "../outputs/ood_sentiment_test/results_20240919_141957.csv"
df = pd.read_csv(filename)
print(df.head())
# df.columns = ["Dataset", "Model", "Method", "Epochs", "Test Loss", "Accuracy", "Precision", "Recall", "F1"]
# metrics = ["Test Loss", "Accuracy", "Precision", "Recall", "F1"]
# fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# fig.suptitle("Model Performance Comparison", fontsize=16)
# for metric, ax in zip(metrics, axes.flatten()):
#         sns.lineplot(data=df, x="Epochs", y=metric, hue="Model", markers=True, ax=ax)
#         ax.set_title(f"{metric} Over Epochs")
#         ax.set_ylabel(metric)
#         ax.set_xlabel("Epochs")
#         ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.tight_layout(rect=[0, 0, 0.9, 0.95])
# plt.savefig('images/table_5-2.png')
# plt.show()

filename = "../outputs/ood_sentiment_test/results_20240919_141957.csv"
df = pd.read_csv(filename)
print(df.head())
# df.columns = ["Dataset", "Model", "Method", "Epochs", "Test Loss", "Accuracy", "Precision", "Recall", "F1"]
# metrics = ["Test Loss", "Accuracy", "Precision", "Recall", "F1"]
# fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# fig.suptitle("Model Performance Comparison", fontsize=16)
# for metric, ax in zip(metrics, axes.flatten()):
#         sns.lineplot(data=df, x="Epochs", y=metric, hue="Model", markers=True, ax=ax)
#         ax.set_title(f"{metric} Over Epochs")
#         ax.set_ylabel(metric)
#         ax.set_xlabel("Epochs")
#         ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.tight_layout(rect=[0, 0, 0.9, 0.95])
# plt.savefig('images/table_5-3.png')
# plt.show()