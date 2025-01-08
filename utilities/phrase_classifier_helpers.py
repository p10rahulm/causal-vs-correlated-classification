
from utilities.phrase_extraction import extract_phrases, remove_punctuation_phrases
import os
import glob

import torch
from models.causal_neutral_model_variations import model_variations
from pathlib import Path
import sys
import json
from datetime import datetime


# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

################################################################################
# Configuration: Checkpoint paths and model references
################################################################################


def get_checkpoint_paths():
    """Get the latest checkpoint paths for DistilBERT and BERT models."""
    
    distilbert_dir = "trained_models/imdb_sentiment_wz_distilbert_10epochs/sentiment"
    bert_dir = "trained_models/imdb_sentiment_wz_bert_10epochs/sentiment"
    
    # Get latest DistilBERT checkpoint
    distilbert_pattern = os.path.join(distilbert_dir, "causalneutralclassifier_1hidden_*.pth")
    distilbert_files = glob.glob(distilbert_pattern)
    DISTILBERT_CKPT_PATH = max(distilbert_files, key=os.path.getmtime) if distilbert_files else None
    
    # Get latest BERT checkpoint
    bert_pattern = os.path.join(bert_dir, "causalneutralclassifier_2hidden_*.pth")
    bert_files = glob.glob(bert_pattern)
    BERT_CKPT_PATH = max(bert_files, key=os.path.getmtime) if bert_files else None
    
    return DISTILBERT_CKPT_PATH, BERT_CKPT_PATH


try:
    DISTILBERT_CKPT_PATH, BERT_CKPT_PATH = get_checkpoint_paths()
except FileNotFoundError as e:
    print(f"Error: {e}")
    raise


# Map "distilbert" and "bert" to (model-creation-fn, checkpoint-path, hidden-layers)
CAUSAL_NEUTRAL_MODELS = {
    "distilbert": {
        "model_fn": model_variations["distilbert"]["1_hidden"],  # e.g., same config as you used for training
        "ckpt_path": DISTILBERT_CKPT_PATH,
    },
    "bert": {
        "model_fn": model_variations["bert"]["2_hidden"],
        "ckpt_path": BERT_CKPT_PATH,
    },
}


################################################################################
# Helper functions
################################################################################

def load_causal_neutral_classifier(model_name: str, device: torch.device, checkpoint_path: str = None):
    """
    Loads a causal–neutral classifier from a checkpoint.
    
    Args:
        model_name: str, either "bert" or "distilbert"
        device: torch.device
        checkpoint_path: str, optional custom checkpoint path
    """
    if checkpoint_path:
        # Use custom checkpoint
        if model_name == "distilbert":
            model_create_fn = model_variations["distilbert"]["1_hidden"]
        else:  # bert
            model_create_fn = model_variations["bert"]["2_hidden"]
        ckpt_path = checkpoint_path
    else:
        # Use default checkpoint from CAUSAL_NEUTRAL_MODELS
        info = CAUSAL_NEUTRAL_MODELS[model_name]
        model_create_fn = info["model_fn"]
        ckpt_path = info["ckpt_path"]
        if ckpt_path is None:
            raise ValueError(f"No default checkpoint found for {model_name}")
    
    print(f"Loading checkpoint from: {ckpt_path}")
    
    # Create model instance
    model = model_create_fn("Sentiment", freeze_encoder=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval().to(device)
    model.device = device
    
    return model, ckpt_path

def classify_phrases(phrases, classifier_model, tokenizer):
    """
    Classify each phrase as 0=neutral or 1=causal with the given classifier.
    Returns a list of integer labels [0 or 1].
    """   
    # Batching the phrases can help if you have many phrases per review.
    inputs = tokenizer(phrases, truncation=True, padding=True, return_tensors="pt")
    # Move to same device as the classifier
    inputs = {
        k: v.to(classifier_model.device)
        for k, v in inputs.items()
        if k != "token_type_ids"
    }


    with torch.no_grad():
        outputs = classifier_model(**inputs)
        # For some model definitions, outputs = model(**inputs) might directly be the logits,
        # or you might need outputs.logits depending on how it's implemented.
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        preds = torch.argmax(logits, dim=1)

    return preds.cpu().tolist()


def extract_causal_and_neutral_phrases(review_text, classifier_model, tokenizer):
    """
    1) Split the review_text into phrases.
    2) Classify each phrase as causal or neutral.
    3) Return (list_of_causal_phrases, list_of_neutral_phrases).
    """
    all_phrases = extract_phrases(review_text)                 # e.g., user-defined logic
    clean_phrases = remove_punctuation_phrases(all_phrases)    # user-defined

    if not clean_phrases:
        # If no phrases at all, fallback:
        return [], []

    labels = classify_phrases(clean_phrases, classifier_model, tokenizer)

    causal_phrases = []
    neutral_phrases = []
    for phrase, lbl in zip(clean_phrases, labels):
        if lbl == 1:
            causal_phrases.append(phrase)
        else:
            neutral_phrases.append(phrase)

    return causal_phrases, neutral_phrases
