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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from models.causal_neutral_model_variations import model_variations
from models.model_utilities import load_trained_model, find_model_file
from data_loaders.sentiment_dataset_test import SentimentDataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_model(model, dataloader, device):
    """Unified testing function similar to the original test method"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Testing", leave=False)
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


def discover_model_folders(base_dir: str):
    """
    Walks through your trained_models folder structure and returns
    a list of subdirectories that contain a .pth file in the "sentiment" subfolder.

    Example: it will return entries like:
      [
        {
          'directory': 'trained_models/imdb_sentiment_naive_baseline_albert_10epochs/sentiment',
          'type': 'naive_baseline',
          'model': 'albert',
          'epochs': 10,
          'lambda_reg': None,
          'lr_schedule': None,
          'lambda_schedule_mode': None,
          'lr_value': None,
        },
        {
          'directory': 'trained_models/imdb_causal_mediation_albert_cosine_warm_restarts_exponential_lr0.0001/sentiment',
          'type': 'regularized',
          'model': 'albert',
          'epochs': None,
          'lambda_reg': None,
          'lr_schedule': 'cosine_warm_restarts',
          'lambda_schedule_mode': 'exponential',
          'lr_value': 0.0001,
        },
        ...
      ]
    """
    results = []

    base_path = Path(base_dir)
    if not base_path.exists():
        return results

    for subdir, dirs, files in os.walk(base_path):
        # We only care about subdirs named 'sentiment' that actually have a .pth file
        if Path(subdir).name == "sentiment":
            pth_files = list(Path(subdir).glob("*.pth"))
            if not pth_files:
                continue
            
            # Figure out what this directory represents by name
            path_str = str(Path(subdir).parent.name)  # e.g. 'imdb_causal_mediation_albert_cosine_warm_restarts_exponential_lr0.0001'
            
            info_dict = {
                'directory': subdir,
                'type': None,
                'model': None,
                'epochs': None,
                'lambda_reg': None,
                'lr_schedule': None,
                'lambda_schedule_mode': None,
                'lr_value': None,
            }

            # 1) Check if it's naive baseline or causal only or older 'lambdaX' style or the gridsearch style
            if "naive_baseline" in path_str:
                info_dict['type'] = 'naive_baseline'
            elif "causal_only" in path_str:
                info_dict['type'] = 'causal_only'
            elif "causal_mediation" in path_str:
                info_dict['type'] = 'regularized'
            else:
                # fallback
                info_dict['type'] = 'unknown'

            # 2) Extract the model name.  Often it's like "imdb_sentiment_naive_baseline_albert_10epochs"
            #    or "imdb_causal_mediation_albert_lambda0.01_5epochs" etc.
            #    We'll do a simple guess by splitting on underscores
            parts = path_str.split("_")

            # A quick approach: find which part is your model name
            # (You already know your possible model names: [albert, bert, roberta, etc.])
            # We'll store them for reference:
            # Note the order of these is important as subparts may be matched.
            possible_models = ["electra_small_discriminator", "distilbert", "roberta", 
                               "albert", "deberta", "modern_bert", "bert"]
            

            # If your directory name is big, you can do a loop:
            for pm in possible_models:
                if pm in path_str:
                    info_dict['model'] = pm
                    break
            
            # 3) If there's an `_epochs` somewhere, try to parse it
            #    e.g. 'imdb_sentiment_naive_baseline_albert_10epochs'
            #    we see "10epochs" in there
            for part in parts:
                if part.endswith("epochs"):
                    try:
                        # remove "epochs", parse int
                        epoch_str = part.replace("epochs", "")
                        info_dict['epochs'] = int(epoch_str)
                    except:
                        pass
                
                # Also check if part looks like "lambda0.025"
                if part.startswith("lambda"):
                    # e.g. "lambda0.025" -> 0.025
                    maybe_val = part.replace("lambda", "")
                    try:
                        info_dict['lambda_reg'] = float(maybe_val)
                    except:
                        pass

                # For the LR schedule approach: e.g. 'cyclic_triangular2', 'cosine_warm_restarts', etc.
                # We'll do a naive check:
                if part in ["cosine", "cosine_warm_restarts", "one", "cycle", "cyclic", "cyclic_triangular", "cyclic_triangular2"]:
                    info_dict['lr_schedule'] = part  # or we might need to join "one" + "cycle" -> "one_cycle"
                # For the "exponential", "linear", "piecewise":
                if part in ["exponential", "linear", "piecewise"]:
                    info_dict['lambda_schedule_mode'] = part
                # For lr0.0001
                if part.startswith("lr"):
                    # e.g. "lr0.0001"
                    try:
                        lr_val_str = part.replace("lr", "")
                        info_dict['lr_value'] = float(lr_val_str)
                    except:
                        pass

            results.append(info_dict)

    return results


def run_comprehensive_ood_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # These are the 3 OOD-labeled test sets you want to evaluate on
    datasets = [
        # {
        #     'name': 'OOD Genres',
        #     'file': 'data/ood_genres.csv',
        #     'text_column': 'CF_Rev_Genres',
        #     'sentiment_column': 'CF_Sentiment'
        # },
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

    # Discover all subdirectories with a .pth in "sentiment" subfolder
    all_model_dirs = discover_model_folders("trained_models")

    # Prepare results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/ood_sentiment_comprehensive_test"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{timestamp}.csv")

    fieldnames = [
        'dataset',
        'model',
        'model_type',
        'epochs',
        'lambda_reg',
        'lr_schedule',
        'lambda_schedule_mode',
        'lr_value',
        'test_loss',
        'test_accuracy',
        'test_precision',
        'test_recall',
        'test_f1',
        'model_dir'
    ]

    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Evaluate on each OOD dataset
        for ds_conf in datasets:
            dataset_name = ds_conf['name']
            dataset_file = ds_conf['file']
            text_col = ds_conf['text_column']
            label_col = ds_conf['sentiment_column']
            # Sort all_model_dirs by multiple keys
            all_model_dirs = sorted(all_model_dirs, 
                key=lambda x: (
                    x['model'] or '',  # Use empty string as fallback if None
                    x['type'] or '',
                    x['epochs'] or float('inf'),  # Use infinity for None values to put them last
                    x['lr_schedule'] or '',
                    x['lambda_schedule_mode'] or '',
                    x['lr_value'] or float('inf')
                )
            )
            # Loop over each discovered directory
            for info in all_model_dirs:
                model_dir = info['directory']
                model_type = info['type']
                model_name = info['model']
                
                # If you discovered a directory that doesn't match or no model found, skip
                if model_name is None:
                    logging.warning(f"Cannot parse the model name from: {model_dir}")
                    continue

                # The classification word
                classification_word = "Sentiment"

                # Try to find the actual .pth file inside that directory
                # (You already found them in discover_model_folders, but let's do a final check)
                model_pth = find_model_file(model_dir)
                if not model_pth:
                    logging.warning(f"No .pth file found in {model_dir}, skipping.")
                    continue
                
                # If you used "1_hidden" or "2_hidden", you can check the directory or just pick 
                # the correct hidden-layers approach:
                # For naive baseline or causal-only, you might have used 1 hidden, 
                # for the regularized, you used 2 hidden, etc.
                # If you want to unify, you can guess or store it in info. For now, let's just do 2-hidden:
                hidden_layer = "2_hidden"
                if "1hidden" in model_pth:
                    hidden_layer = "1_hidden"               
                elif model_type in ["naive_baseline", "causal_only"]:
                    # Maybe your checkpoint used "1_hidden". Adjust as needed.
                    hidden_layer = "1_hidden"  
                logging.info(f"model_dir = {model_dir}\nmodel_type = {model_type}\n" + 
                             f"model_name = {model_name}\n model_pth = {model_pth}\n" + 
                             f"hidden_layer = {hidden_layer}")
                # Construct the model object
                try:
                    net = model_variations[model_name][hidden_layer](
                        classification_word, freeze_encoder=False
                    ).to(device)

                    # Load the checkpoint weights
                    model = load_trained_model(model_pth, net)
                    model.to(device)
                except Exception as ex:
                    logging.warning(f"First attempt failed with {hidden_layer}: {ex}")
                    try:
                        # Define fallback mapping
                        fallback_layers = {
                            "1_hidden": "2_hidden",
                            "2_hidden": "1_hidden"
                        }
                        # Get the fallback layer configuration
                        hidden_layer = fallback_layers.get(hidden_layer)
                        if hidden_layer is None:
                            raise ValueError(f"No fallback defined for layer: {hidden_layer}")
                            
                        net = model_variations[model_name][hidden_layer](
                            classification_word, freeze_encoder=False
                        ).to(device)

                        # Load the checkpoint weights
                        model = load_trained_model(model_pth, net)
                        model.to(device)
                    except Exception as ex:
                        logging.error(f"Error loading model from {model_dir}: {ex}", exc_info=True)
                        continue

                # If the checkpoint stored the huggingface model name inside, you might do:
                # tokenizer = AutoTokenizer.from_pretrained(model.model_name)
                # 
                # If not, you might just guess e.g. "albert-base-v2" if it's ALBERT:
                # (Make sure your model_variations call returns a .model_name or something.)
                # For simplicity, let's do:
                tokenizer_name = model.model_name if hasattr(model, "model_name") else "albert-base-v2"
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            
                # Create the dataset and dataloader
                dataset = SentimentDataset(
                    dataset_file,
                    tokenizer,
                    text_col,
                    label_col
                )
                dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

                # Test the model
                try:
                    avg_loss, accuracy, precision, recall, f1 = test_model(model, dataloader, device)
                except Exception as e:
                    logging.error(f"Error while testing on {dataset_name}: {e}", exc_info=True)
                    continue

                # Save the result
                row = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'model_type': model_type,
                    'epochs': info['epochs'],
                    'lambda_reg': info['lambda_reg'],
                    'lr_schedule': info['lr_schedule'],
                    'lambda_schedule_mode': info['lambda_schedule_mode'],
                    'lr_value': info['lr_value'],
                    'test_loss': avg_loss,
                    'test_accuracy': accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1,
                    'model_dir': info['directory'],
                }
                writer.writerow(row)
                csvfile.flush()

                logging.info(f"Done testing {model_dir} on {dataset_name} with results: "
                             f"Loss={avg_loss:.4f} Acc={accuracy:.3f} P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

    logging.info(f"Finished all evaluations. Results are in {results_file}")

if __name__ == "__main__":
    run_comprehensive_ood_test()
