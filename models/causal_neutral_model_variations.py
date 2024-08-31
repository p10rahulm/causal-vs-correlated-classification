# models/causal_neutral_model_variations.py

from models.causal_neutral_model_template import create_model
import torch


def create_model_with_device(model_name, classification_word, hidden_layers, freeze_encoder=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return create_model(model_name, classification_word, hidden_layers, device, freeze_encoder)


def create_distilbert_model(classification_word, hidden_layers, freeze_encoder=True):
    return create_model_with_device("distilbert-base-uncased", classification_word, hidden_layers, freeze_encoder)


def create_roberta_model(classification_word, hidden_layers, freeze_encoder=True):
    return create_model_with_device("roberta-base", classification_word, hidden_layers, freeze_encoder)


def create_bert_model(classification_word, hidden_layers, freeze_encoder=True):
    return create_model_with_device("bert-base-uncased", classification_word, hidden_layers, freeze_encoder)


def create_albert_model(classification_word, hidden_layers, freeze_encoder=True):
    return create_model_with_device("albert-base-v2", classification_word, hidden_layers, freeze_encoder)


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
    "albert": {
        "0_hidden": lambda cw, fe=True: create_albert_model(cw, [], fe),
        "1_hidden": lambda cw, fe=True: create_albert_model(cw, [256], fe),
        "2_hidden": lambda cw, fe=True: create_albert_model(cw, [256, 128], fe)
    }
}

# Usage example:
# classification_word = "Sentiment"
# model = model_variations["distilbert"]["1_hidden"](classification_word)
# To create a model with unfrozen encoder:
# model = model_variations["distilbert"]["1_hidden"](classification_word, False)
