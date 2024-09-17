import os
from pathlib import Path
import sys

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))


import torch
import csv
from datetime import datetime
import logging
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from models.causal_neutral_model_variations import model_variations
from models.model_utilities import load_trained_model, find_model_file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))


class SentimentDataset(Dataset):
    def __init__(self, csv_file, tokenizer, text_column, sentiment_column, max_length=512):
        self.data = pd.read_csv(csv_file, sep='|', skiprows=1, names=[text_column, sentiment_column])
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.sentiment_column = sentiment_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx][self.text_column])
        sentiment = self.data.iloc[idx][self.sentiment_column]
        
        # Convert sentiment to string and handle potential NaN values
        if pd.isna(sentiment):
            logging.warning(f"NaN sentiment value found at index {idx}. Defaulting to Negative.")
            label = 0
        else:
            sentiment = str(sentiment).strip()
            if sentiment.lower() == 'positive':
                label = 1
            elif sentiment.lower() == 'negative':
                label = 0
            else:
                logging.warning(f"Unexpected sentiment value: {sentiment} at index {idx}. Defaulting to Negative.")
                label = 0

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def run_ood_sentiment_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Experiment parameters
    models = ["roberta", "albert", "distilbert", "bert", "electra_small_discriminator", "t5"]
    original_epochs = [5, 10]
    run_types = ['sentiment_naive_baseline', 'regularized', 'causal_phrases']
    classification_word = "Sentiment"
    batch_size = 32
    hidden_layer = "1_hidden"

    # Dataset configurations
    datasets = [
        {
            'name': 'CF Test Ltd Paper',
            'file': 'data/cf_test_ltd_paper.csv',
            'text_column': 'Text',
            'sentiment_column': 'Sentiment'
        },
        {
            'name': 'OOD Sentiment',
            'file': 'data/ood_sentiments.csv',
            'text_column': 'CF_Rev_Sentiment',
            'sentiment_column': 'CF_Sentiment'
        },
        
    ]

    # Prepare results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/ood_sentiment_test"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{timestamp}.csv")

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['dataset', 'model', 'run_type', 'original_epochs', 'test_loss', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_config in datasets:
            for model_name in models:
                for run_type in run_types:
                    for epochs in original_epochs:
                        # try:
                        logging.info(f"Testing model: {model_name}, run_type={run_type}, original_epochs={epochs}, dataset={dataset_config['name']}")

                        # Construct the model path based on the run type
                        if run_type == 'sentiment_naive_baseline':
                            model_path = find_model_file(f"trained_models/imdb_sentiment_naive_baseline_{model_name}_{epochs}epochs/sentiment")
                        elif run_type == 'regularized':
                            model_path = find_model_file(f"trained_models/imdb_regularized_{model_name}_{epochs}_5epochs/sentiment")
                        elif run_type == 'causal_phrases':
                            model_path = find_model_file(f"trained_models/imdb_causal_phrases_{model_name}_{epochs}epochs/sentiment")
                        else:
                            raise ValueError(f"Unknown run_type: {run_type}")

                        if model_path is None:
                            logging.warning(f"Model file not found for {model_name} with run_type={run_type} and epochs={epochs}. Skipping...")
                            continue

                        model = load_trained_model(model_path, model_variations[model_name][hidden_layer](classification_word, freeze_encoder=False)).to(device)
                        tokenizer = AutoTokenizer.from_pretrained(model.model_name)

                        # Create dataset and dataloader
                        dataset = SentimentDataset(
                            dataset_config['file'],
                            tokenizer,
                            dataset_config['text_column'],
                            dataset_config['sentiment_column']
                        )
                        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                        # Test the model
                        model.eval()
                        total_loss = 0
                        correct_predictions = 0
                        total_predictions = 0
                        all_predictions = []
                        all_labels = []

                        with torch.no_grad():
                            for batch in dataloader:
                                input_ids = batch['input_ids'].to(device)
                                attention_mask = batch['attention_mask'].to(device)
                                labels = batch['labels'].to(device)

                                logits = model(input_ids, attention_mask=attention_mask)
                                loss = F.cross_entropy(logits, labels)

                                total_loss += loss.item()
                                predictions = torch.argmax(logits, dim=1)
                                correct_predictions += (predictions == labels).sum().item()
                                total_predictions += labels.shape[0]

                                all_predictions.extend(predictions.cpu().numpy())
                                all_labels.extend(labels.cpu().numpy())

                        avg_loss = total_loss / len(dataloader)
                        accuracy = correct_predictions / total_predictions

                        # Calculate precision, recall, and F1 score
                        from sklearn.metrics import precision_recall_fscore_support
                        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')

                        # Write results
                        writer.writerow({
                            'dataset': dataset_config['name'],
                            'model': model_name,
                            'run_type': run_type,
                            'original_epochs': epochs,
                            'test_loss': avg_loss,
                            'test_accuracy': accuracy,
                            'test_precision': precision,
                            'test_recall': recall,
                            'test_f1': f1
                        })
                        csvfile.flush()  # Ensure data is written immediately

                        logging.info(f"Completed testing {model_name}, run_type={run_type}, original_epochs={epochs}, dataset={dataset_config['name']}")
                        logging.info(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

                        # except Exception as e:
                        #     logging.error(f"Error testing {model_name} with run_type={run_type}, epochs={epochs}, and dataset={dataset_config['name']}: {e}")

                        # finally:
                        #     # Clear CUDA cache
                        #     if torch.cuda.is_available():
                        #         torch.cuda.empty_cache()

    logging.info(f"Testing completed. Results saved to {results_file}")

if __name__ == "__main__":
    run_ood_sentiment_test()