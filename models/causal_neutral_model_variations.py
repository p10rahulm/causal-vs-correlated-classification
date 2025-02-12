from models.causal_neutral_model_template import create_model
import torch
from typing import Dict, Any, List, Callable
from functools import partial

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device

def create_custom_model(model_name, classification_word, hidden_layers, freeze_encoder=True, cuda_device=0, max_length=1024):
    device = get_device()
    return create_model(model_name, classification_word, hidden_layers, device, freeze_encoder,max_length)

def create_model_factory(model_path: str, cuda_device: int = 0) -> Callable:
    """Creates a partial function for model creation with a specific model path and device."""
    return partial(create_custom_model, model_path, cuda_device=cuda_device)

def generate_hidden_layer_configs() -> Dict[str, List[int]]:
    """Generate standard hidden layer configurations."""
    return {
        "0_hidden": [],
        "1_hidden": [256],
        "2_hidden": [256, 128],
        "2_hidden_wide": [512, 256],
        "3_hidden": [512, 256, 128]
    }

def create_model_variations() -> Dict[str, Dict[str, Callable]]:
    """Create model variations with standard configurations and specified CUDA device."""
    
    base_models = {
        "distilbert": "distilbert-base-uncased",
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "deberta": "microsoft/deberta-base",
        "albert": "albert-base-v2",
        "electra_small_discriminator": "google/electra-small-discriminator",
        "bart": "facebook/bart-base",
        # "xlnet": "xlnet-base-cased",
        "minilm": "microsoft/MiniLM-L12-H384-uncased",
        "tinybert": "huawei-noah/TinyBERT_General_4L_312D",
        "distilroberta": "distilroberta-base",
        "modern_bert": "answerdotai/ModernBERT-base",
        "xlnet": "xlnet-large-cased",  # Updated to large
        "deberta_large": "microsoft/deberta-v3-large",
        "roberta_large": "roberta-large",
        "albert_xxlarge": "albert-xxlarge-v2",
        "electra_large": "google/electra-large-discriminator",
        "xlm_roberta": "xlm-roberta-large",
    }
    
    # Get standard hidden layer configurations
    hidden_configs = generate_hidden_layer_configs()
    
    # Generate model variations
    variations = {}
    
    for model_name, model_path in base_models.items():
        variations[model_name] = {}
        for config_name, hidden_layers in hidden_configs.items():
            def create_model_fn(cw, model_path=model_path, hidden_layers=hidden_layers,freeze_encoder=True,max_length=1024):
                return create_custom_model(
                    model_name=model_path,
                    classification_word=cw,
                    hidden_layers=hidden_layers,
                    freeze_encoder=freeze_encoder,
                    max_length=max_length
                )
            variations[model_name][config_name] = create_model_fn
    return variations

# Create the model variations with default CUDA:0
model_variations = create_model_variations()

