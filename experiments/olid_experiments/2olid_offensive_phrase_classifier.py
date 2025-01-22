import os
import json
from pathlib import Path
import sys
import csv
from datetime import datetime

# Set CUDA DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

import torch
from models.causal_neutral_model_variations import model_variations
from data_loaders.imdb_sentiment.phrase_classification_dataloader import CausalNeutralDataModule

from trainers.trainer import Trainer, save_trained_model
from optimizers.optimizer_params import optimizer_configs

def load_hyperparameters(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_model_specific_lr(model_name):
    """Get the optimal learning rate for each model type."""
    lr_mapping = {
        'deberta': 8e-6,
        'roberta': 8e-6,
        'electra_small_discriminator': 1e-5,
        'distilbert': 1.5e-5,
        'bert': 1e-5,
        'albert': 2e-5,
        'modern_bert': 8e-6
    }
    return lr_mapping.get(model_name, 1e-5)  # Default to 1e-5 if model not found.


def get_hyperparameters(model_name, hyperparams):
    """Get hyperparameters with model-specific learning rates."""
    if model_name in hyperparams:
        return {
            'optimizer_name': hyperparams[model_name]['optimizer'],
            'hidden_layer': hyperparams[model_name]['hidden_layers'],
            'learning_rate': get_model_specific_lr(model_name)  # Use model-specific LR
        }
    else:
        return {
            'optimizer_name': 'adamw',
            'hidden_layer': '2_hidden',
            'learning_rate': get_model_specific_lr(model_name)  # Use model-specific LR even for fallback
        }

def run_olid_sentiment_experiment():
    # Experiment parameters
    models = ["deberta", "electra_small_discriminator", "distilbert", "roberta", "bert", "albert", "modern_bert"]
    classification_word = "Offensive"
    epochs = [10, 20, 30, 40]
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load hyperparameters
    hyperparams = load_hyperparameters('../../models/optimal_wz_classifier_validation_hyperparams.json')


    # Prepare data loader
    file_path = "outputs/olid_phrase_dataset/olid_offensive_phrases_20250122_145455.json"
    data_module = CausalNeutralDataModule(file_path, classification_word)

    # Training parameters for the new trainer
    training_params = {
        'layer_wise_lr_decay': 0.95,
        'max_grad_norm': 5.0,
        'warmup_ratio': 0.1,
        'cosine_decay': True,
        'drop_lr_on_plateau': False,
        'patience': 3
    }
    training_params = {
        'layer_wise_lr_decay': 0.8,        # Steeper decay (was 0.95) since lower layers are more important for phrase-level features
        'max_grad_norm': 1.0,              # Tighter gradient clipping (was 5.0) for more stable training
        'warmup_ratio': 0.06,              # Shorter warmup (was 0.1) since we're fine-tuning for a specific task
        'cosine_decay': True,
        'drop_lr_on_plateau': False,
        'patience': 2                       # Reduced patience (was 3) for faster adaptation
    }



    # Prepare results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/olid_phrase_classification"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"wz_training_results_{timestamp}.csv")

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['model', 'epochs', 'optimizer', 'hidden_layers', 'learning_rate', 'train_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_name in models:
            # Get hyperparameters for the current model
            model_hyperparams = get_hyperparameters(model_name, hyperparams)
            optimizer_name = model_hyperparams['optimizer_name']
            hidden_layer = model_hyperparams['hidden_layer']
            learning_rate = model_hyperparams['learning_rate']

            for num_epochs in epochs:
                print(f"Running experiment: {model_name}, epochs={num_epochs}")

                # Create model
                model = model_variations[model_name][hidden_layer](classification_word, freeze_encoder=True)
                # model = model_variations[model_name][hidden_layer](classification_word)
                # Setup optimizer parameters
                # optimizer_params = {
                #     "lr": learning_rate,
                #     "betas": (0.9, 0.999),
                #     "eps": 1e-8,
                #     "weight_decay": 0.01
                # }
                optimizer_params = {
                    "lr": learning_rate,
                    "betas": (0.9, 0.98),              # Modified beta2 for better adaptation to phrase-level patterns
                    "eps": 1e-6,                       # Slightly larger epsilon for stability
                    "weight_decay": 0.1                # Increased weight decay (was 0.01) to prevent overfitting on phrases
                }

                # Update optimizer config
                optimizer_config = optimizer_configs[optimizer_name].copy()
                optimizer_config['params'] = optimizer_config['params'].copy()
                optimizer_config['params']['lr'] = learning_rate

                # Create Trainer with parameters
                trainer = Trainer(
                    model=model,
                    data_module=data_module,
                    optimizer_name=optimizer_name,
                    optimizer_params=optimizer_params,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    device=device,
                    dataset_name="olid",
                    **training_params
                )


                # Train on full dataset
                print(f"Training {model_name} for {num_epochs} epochs...")
                epoch_losses = trainer.train_on_full_dataset(num_epochs)

                # Write results
                writer.writerow({
                    'model': model_name,
                    'epochs': num_epochs,
                    'optimizer': optimizer_name,
                    'hidden_layers': hidden_layer,
                    'learning_rate': learning_rate,
                    'train_loss': epoch_losses[-1]  # Last epoch's training loss
                })
                csvfile.flush()  # Ensure data is written immediately

                # Save the trained model
                save_trained_model(trainer, f"olid_offensive_wz_{model_name}_{num_epochs}epochs",
                                int(hidden_layer[0]))

        print(f"Experiments completed. Results saved to {results_file}")

if __name__ == "__main__":
    run_olid_sentiment_experiment()