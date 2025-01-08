import os
from pathlib import Path
import sys
import torch
import csv
from datetime import datetime
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm


# Set CUDA DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from models.causal_neutral_model_variations import model_variations
from models.model_utilities import load_trained_model, find_model_file
from data_loaders.sentiment_dataset_test import SentimentDataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_model(model, dataloader, device):
    """Unified testing function similar to the original test method"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Testing")
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return avg_loss, accuracy, precision, recall, f1

def run_comprehensive_ood_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Experiment parameters
    models = ["electra_small_discriminator", "distilbert", "roberta", "bert", "albert", "deberta", "modern_bert"]
    original_epochs = [5, 10]
    lambda_values = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    classification_word = "Sentiment"
    batch_size = 256
    hidden_layer = "1_hidden"

    # Model configurations
    model_configs = [
        {
            'type': 'naive_baseline',
            'path_template': "trained_models/imdb_sentiment_naive_baseline_{model}_{epochs}epochs/sentiment",
            'needs_epochs': True,
            'lambda_reg': None
        },
        {
            'type': 'causal_phrases',
            'path_template': "trained_models/imdb_causal_only_{model}_{epochs}epochs/sentiment",
            'needs_epochs': True,
            'lambda_reg': None
        },
        {
            'type': 'regularized',
            'path_template': "trained_models/imdb_causal_mediation_{model}_lambda{lambda_reg}_{epochs}epochs/sentiment",  # Updated path template
            'needs_epochs': True,  # Changed to True since epochs are needed in path
            'lambda_values': lambda_values
        }
    ]

    # Dataset configurations
    datasets = [
        {
            'name': 'OOD Genres',
            'file': 'data/ood_genres.csv',
            'text_column': 'CF_Rev_Genres',
            'sentiment_column': 'CF_Sentiment'
        },
        {
            'name': 'OOD Sentiment',
            'file': 'data/ood_sentiments_test.csv',
            'text_column': 'CF_Rev_Sentiment',
            'sentiment_column': 'CF_Sentiment'
        },
        {
            'name': 'CF Test Ltd Paper',
            'file': 'data/cf_test_ltd_paper.csv',
            'text_column': 'Text',
            'sentiment_column': 'Sentiment'
        },
    ]

    # Prepare results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/ood_sentiment_comprehensive_test"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{timestamp}.csv")

    fieldnames = ['dataset', 'model', 'model_type', 'original_epochs', 'lambda_reg', 
                'test_loss', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']

    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_config in datasets:
            logging.info(f"Processing dataset: {dataset_config['name']}")
            
            for model_name in models:
                logging.info(f"Testing model: {model_name}")
                
                for config in model_configs:
                    if config['needs_epochs']:
                        epochs_list = original_epochs
                    else:
                        epochs_list = [None]

                    lambda_list = config['lambda_values'] if 'lambda_values' in config else [None]
                    
                    for epochs in epochs_list:
                        for lambda_reg in lambda_list:
                            try:
                                # Construct model path
                                if config['type'] == 'regularized':
                                    model_path = config['path_template'].format(
                                        model=model_name, 
                                        lambda_reg=lambda_reg,
                                        epochs=epochs
                                    )
                                elif epochs is not None:
                                    model_path = config['path_template'].format(
                                        model=model_name, 
                                        epochs=epochs
                                    )
                                else:
                                    model_path = config['path_template'].format(model=model_name)

                                model_path = find_model_file(model_path)
                                
                                if model_path is None:
                                    logging.warning(f"Model file not found: {model_path}. Skipping...")
                                    continue

                                # Load model and tokenizer
                                model = load_trained_model(
                                    model_path, 
                                    model_variations[model_name][hidden_layer](
                                        classification_word, 
                                        freeze_encoder=False
                                    )
                                ).to(device)
                                
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
                                avg_loss, accuracy, precision, recall, f1 = test_model(model, dataloader, device)

                                # Write results
                                result = {
                                    'dataset': dataset_config['name'],
                                    'model': model_name,
                                    'model_type': config['type'],
                                    'original_epochs': epochs,
                                    'lambda_reg': lambda_reg,
                                    'test_loss': avg_loss,
                                    'test_accuracy': accuracy,
                                    'test_precision': precision,
                                    'test_recall': recall,
                                    'test_f1': f1
                                }
                                writer.writerow(result)
                                csvfile.flush()

                                logging.info(f"Completed testing {model_name} ({config['type']})")
                                logging.info(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

                            except Exception as e:
                                logging.error(f"Error testing {model_name}: {str(e)}", exc_info=True)

                            finally:
                                # Clear CUDA cache
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

    logging.info(f"Testing completed. Results saved to {results_file}")

if __name__ == "__main__":
    run_comprehensive_ood_test()