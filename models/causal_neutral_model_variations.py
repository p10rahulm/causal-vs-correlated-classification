# models/causal_neutral_model_variations.py
from transformers import T5EncoderModel, T5Tokenizer

from models.causal_neutral_model_template import create_model
import torch
import torch.nn as nn
from models.t5_for_classification import T5ForClassification


def create_model_with_device(model_name, classification_word, hidden_layers, freeze_encoder=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return create_model(model_name, classification_word, hidden_layers, device, freeze_encoder)


def create_distilbert_model(classification_word, hidden_layers, freeze_encoder=True):
    return create_model_with_device("distilbert-base-uncased", classification_word, hidden_layers, freeze_encoder)


def create_roberta_model(classification_word, hidden_layers, freeze_encoder=True):
    return create_model_with_device("roberta-base", classification_word, hidden_layers, freeze_encoder)


def create_bert_model(classification_word, hidden_layers, freeze_encoder=True):
    return create_model_with_device("bert-base-uncased", classification_word, hidden_layers, freeze_encoder)


def create_t5_model(classification_word, hidden_layers, freeze_encoder=True):
    try:
        model_name = "t5-small"  # You can change this to other T5 variants if needed
        t5_encoder = T5EncoderModel.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)

        model = T5ForClassification(t5_encoder, hidden_layers=hidden_layers,
                                    tokenizer=tokenizer, classification_word=classification_word)

        if freeze_encoder:
            model.freeze_encoder()

        return model.to_device('cuda' if torch.cuda.is_available() else 'cpu')

    except ImportError as e:
        print(f"Error creating T5 model: {e}")
        print("Please install SentencePiece by running: pip install sentencepiece")
        return None  # or return a default model


def create_deberta_small_model(classification_word, hidden_layers, freeze_encoder=True):
    return create_model_with_device("microsoft/deberta-v3-small", classification_word, hidden_layers, freeze_encoder)


# Create variations for each model type
model_variations = {
    "distilbert": {
        "0_hidden": lambda cw, freeze_encoder=True: create_distilbert_model(cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_distilbert_model(cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_distilbert_model(cw, [256, 128], freeze_encoder)
    },
    "roberta": {
        "0_hidden": lambda cw, freeze_encoder=True: create_roberta_model(cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_roberta_model(cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_roberta_model(cw, [256, 128], freeze_encoder)
    },
    "bert": {
        "0_hidden": lambda cw, freeze_encoder=True: create_bert_model(cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_bert_model(cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_bert_model(cw, [256, 128], freeze_encoder)
    },
    "t5": {
        "0_hidden": lambda cw, freeze_encoder=True: create_t5_model(cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_t5_model(cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_t5_model(cw, [256, 128], freeze_encoder)
    },
    "deberta": {
        "0_hidden": lambda cw, freeze_encoder=True: create_deberta_small_model(cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_deberta_small_model(cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_deberta_small_model(cw, [256, 128], freeze_encoder)
    }
}
# Usage example:
# classification_word = "Sentiment"
# model = model_variations["distilbert"]["1_hidden"](classification_word)
# To create a model with unfrozen encoder:
# model = model_variations["distilbert"]["1_hidden"](classification_word, False)
