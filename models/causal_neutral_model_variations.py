# models/causal_neutral_model_variations.py

from transformers import T5EncoderModel, T5Tokenizer
from models.causal_neutral_model_template import create_model
import torch
from models.t5_for_classification import T5ForClassification


def create_model_with_device(model_name, classification_word, hidden_layers, freeze_encoder=True):
    """
    Creates a model with the specified device (GPU or CPU).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return create_model(model_name, classification_word, hidden_layers, device, freeze_encoder)


def create_custom_model(model_name, classification_word, hidden_layers, freeze_encoder=True):
    """
    General-purpose function to create models using create_model_with_device.
    """
    return create_model_with_device(model_name, classification_word, hidden_layers, freeze_encoder)


def create_custom_t5_model(model_name, classification_word, hidden_layers, freeze_encoder=True):
    """
    Specialized function to create T5 models.
    """
    try:
        t5_encoder = T5EncoderModel.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)

        model = T5ForClassification(
            t5_encoder,
            hidden_layers=hidden_layers,
            tokenizer=tokenizer,
            classification_word=classification_word
        )

        if freeze_encoder:
            model.freeze_encoder()

        return model.to_device('cuda' if torch.cuda.is_available() else 'cpu')

    except ImportError as e:
        print(f"Error creating T5 model: {e}")
        print("Please install SentencePiece by running: pip install sentencepiece")
        return None


# Create variations for each model type
model_variations = {
    # Base Models
    "distilbert": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("distilbert-base-uncased", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("distilbert-base-uncased", cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("distilbert-base-uncased", cw, [256, 128], freeze_encoder)
    },
    "roberta": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("roberta-base", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("roberta-base", cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("roberta-base", cw, [256, 128], freeze_encoder)
    },
    "bert": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("bert-base-uncased", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("bert-base-uncased", cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("bert-base-uncased", cw, [256, 128], freeze_encoder)
    },
    "deberta": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("microsoft/deberta-base", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("microsoft/deberta-base", cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("microsoft/deberta-base", cw, [256, 128], freeze_encoder)
    },
    "albert": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("albert-base-v2", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("albert-base-v2", cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("albert-base-v2", cw, [256, 128], freeze_encoder)
    },
    "electra_small_discriminator": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("google/electra-small-discriminator", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("google/electra-small-discriminator", cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("google/electra-small-discriminator", cw, [256, 128], freeze_encoder)
    },
    "deberta_small": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("microsoft/deberta-v3-small", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("microsoft/deberta-v3-small", cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("microsoft/deberta-v3-small", cw, [256, 128], freeze_encoder)
    },
    "bart": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("facebook/bart-base", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("facebook/bart-base", cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("facebook/bart-base", cw, [256, 128], freeze_encoder)
    },
    "xlnet": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("xlnet-base-cased", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("xlnet-base-cased", cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("xlnet-base-cased", cw, [256, 128], freeze_encoder)
    },
    
    # Additional Small Models
    "deberta_v3": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("microsoft/deberta-v3-base", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("microsoft/deberta-v3-base", cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("microsoft/deberta-v3-base", cw, [256, 128], freeze_encoder)
    },
    "albert_tiny": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("albert-tiny-v2", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("albert-tiny-v2", cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("albert-tiny-v2", cw, [256, 128], freeze_encoder)
    },
    "minilm": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("microsoft/MiniLM-L12-H384-uncased", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("microsoft/MiniLM-L12-H384-uncased", cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("microsoft/MiniLM-L12-H384-uncased", cw, [256, 128], freeze_encoder)
    },
    "tinybert": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("huawei-noah/TinyBERT_General_4L_312D", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("huawei-noah/TinyBERT_General_4L_312D", cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("huawei-noah/TinyBERT_General_4L_312D", cw, [256, 128], freeze_encoder)
    },
    "distilroberta": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("distilroberta-base", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("distilroberta-base", cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("distilroberta-base", cw, [256, 128], freeze_encoder)
    },

    # Large Models
    "roberta_large": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("roberta-large", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("roberta-large", cw, [512], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("roberta-large", cw, [512, 256], freeze_encoder)
    },
    "bert_large": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("bert-large-uncased", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("bert-large-uncased", cw, [512], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("bert-large-uncased", cw, [512, 256], freeze_encoder)
    },
    "deberta_v3_large": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("microsoft/deberta-v3-large", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("microsoft/deberta-v3-large", cw, [512], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("microsoft/deberta-v3-large", cw, [512, 256], freeze_encoder)
    },
    "albert_xxlarge": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("albert-xxlarge-v2", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("albert-xxlarge-v2", cw, [512], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("albert-xxlarge-v2", cw, [512, 256], freeze_encoder)
    },
    "electra_large_discriminator": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("google/electra-large-discriminator", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("google/electra-large-discriminator", cw, [512], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("google/electra-large-discriminator", cw, [512, 256], freeze_encoder)
    },
    "xlnet_large": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_model("xlnet-large-cased", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_model("xlnet-large-cased", cw, [512], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_model("xlnet-large-cased", cw, [512, 256], freeze_encoder)
    },

    # T5 Models (Special Case)
    "t5": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_t5_model("t5-small", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_t5_model("t5-small", cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_t5_model("t5-small", cw, [256, 128], freeze_encoder)
    },
    "t5_base": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_t5_model("t5-base", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_t5_model("t5-base", cw, [256], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_t5_model("t5-base", cw, [256, 128], freeze_encoder)
    },
    "t5_large": {
        "0_hidden": lambda cw, freeze_encoder=True: create_custom_t5_model("t5-large", cw, [], freeze_encoder),
        "1_hidden": lambda cw, freeze_encoder=True: create_custom_t5_model("t5-large", cw, [512], freeze_encoder),
        "2_hidden": lambda cw, freeze_encoder=True: create_custom_t5_model("t5-large", cw, [512, 256], freeze_encoder)
    },
}

# Usage example:
# classification_word = "Sentiment"
# model = model_variations["distilbert"]["1_hidden"](classification_word)
# To create a model with unfrozen encoder:
# model = model_variations["distilbert"]["1_hidden"](classification_word, False)
