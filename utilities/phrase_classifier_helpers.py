
from utilities.phrase_extraction import extract_phrases, remove_punctuation_phrases

import torch
from models.causal_neutral_model_variations import model_variations



################################################################################
# Configuration: Checkpoint paths and model references
################################################################################

DISTILBERT_CKPT_PATH = (
    "trained_models/imdb_sentiment_wz_distilbert_10epochs/sentiment/"
    "CausalNeutralClassifier_1hidden_2024-12-30_19-26-16.pth"
)
BERT_CKPT_PATH = (
    "trained_models/imdb_sentiment_wz_bert_10epochs/sentiment/"
    "CausalNeutralClassifier_2hidden_2024-12-30_19-34-20.pth"
)


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

def load_causal_neutral_classifier(model_name: str, device: torch.device):
    """
    Loads a causal–neutral classifier for BERT or DistilBERT from a checkpoint.
    """
    info = CAUSAL_NEUTRAL_MODELS[model_name]
    model_create_fn = info["model_fn"]
    ckpt_path = info["ckpt_path"]

    # Create the model (freeze encoder or not, but typically it was trained with freeze_encoder=True)
    # model = model_create_fn("Sentiment", freeze_encoder=True)
    model = model_create_fn("Sentiment")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval().to(device)

    # Some versions of your model add `.device = device`. If not, do it explicitly:
    model.device = device
    return model


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
