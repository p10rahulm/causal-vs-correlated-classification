import os
from pathlib import Path
import sys
import json
from datetime import datetime
import re


# Set CUDA DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from models.causal_neutral_model_variations import model_variations
from utilities.phrase_classifier_helpers import (
    load_causal_neutral_classifier, 
    extract_causal_and_neutral_phrases
)

def get_model_type_from_path(checkpoint_path):
    """Extract model type (bert or distilbert) from checkpoint path."""
    if 'distilbert' in checkpoint_path.lower():
        return 'distilbert'
    elif 'bert' in checkpoint_path.lower():
        return 'bert'
    else:
        raise ValueError(f"Cannot determine model type from path: {checkpoint_path}")

def get_epochs_from_path(checkpoint_path):
    """Extract number of epochs from checkpoint path."""
    match = re.search(r'(\d+)epochs', checkpoint_path.lower())
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot determine number of epochs from path: {checkpoint_path}")
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # List of checkpoint paths
    checkpoint_paths = [
        "trained_models/imdb_sentiment_wz_distilbert_40epochs/sentiment/causalneutralclassifier_1hidden_2025-01-08_04-59-01.pth",
        "trained_models/imdb_sentiment_wz_distilbert_10epochs/sentiment/causalneutralclassifier_1hidden_2025-01-08_03-28-51.pth",
        "trained_models/imdb_sentiment_wz_bert_10epochs/sentiment/causalneutralclassifier_2hidden_2025-01-08_08-32-40.pth",
        "trained_models/imdb_sentiment_wz_bert_40epochs/sentiment/causalneutralclassifier_2hidden_2025-01-08_11-18-01.pth"
    ]

    # 1) Load the IMDB dataset from Hugging Face
    imdb = load_dataset("imdb")
    train_data = imdb["train"]
    test_data = imdb["test"]

    # 2) Create output folder
    output_dir = Path("data/imdb_sentiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3) Process each checkpoint
    for checkpoint_path in checkpoint_paths:
        # Determine model type from path
        model_name = get_model_type_from_path(checkpoint_path)
        epochs = get_epochs_from_path(checkpoint_path)
        
        print(f"\n==== Precomputing splits for {model_name} ({epochs} epochs) ====")
        print(f"Using checkpoint: {checkpoint_path}")
        
        # Load the classifier model and corresponding tokenizer
        classifier_model, used_checkpoint = load_causal_neutral_classifier(
            model_name, 
            device, 
            checkpoint_path
        )
        tokenizer = classifier_model.tokenizer

        # Process train set
        print(f"Processing train set: {len(train_data)} examples")
        train_records = []
        for example in tqdm(train_data, desc=f"Processing train set for {model_name}_{epochs}"):
            text = example["text"]
            label = example["label"]
            causal_phrases, neutral_phrases = extract_causal_and_neutral_phrases(
                text, classifier_model, tokenizer
            )
            record = {
                "text": text,
                "label": label,
                "causal_phrases": causal_phrases,
                "neutral_phrases": neutral_phrases,
            }
            train_records.append(record)

        # Process test set
        print(f"Processing test set: {len(test_data)} examples")
        test_records = []
        for example in tqdm(test_data, desc=f"Processing test set for {model_name}_{epochs}"):
            text = example["text"]
            label = example["label"]
            causal_phrases, neutral_phrases = extract_causal_and_neutral_phrases(
                text, classifier_model, tokenizer
            )
            record = {
                "text": text,
                "label": label,
                "causal_phrases": causal_phrases,
                "neutral_phrases": neutral_phrases,
            }
            test_records.append(record)

        # Write JSON files with epoch information in filename
        train_out_path = output_dir / f"train_with_causal_neutral_splits_{model_name}_{epochs}.json"
        test_out_path = output_dir / f"test_with_causal_neutral_splits_{model_name}_{epochs}.json"

        print(f"Writing train output to: {train_out_path}")
        with open(train_out_path, "w", encoding="utf-8") as f:
            json.dump(train_records, f, indent=2, ensure_ascii=False)

        print(f"Writing test output to: {test_out_path}")
        with open(test_out_path, "w", encoding="utf-8") as f:
            json.dump(test_records, f, indent=2, ensure_ascii=False)

        print(f"Done for {model_name} ({epochs} epochs)!\n")

if __name__ == "__main__":
    main()