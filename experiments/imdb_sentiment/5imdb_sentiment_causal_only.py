# causal_only_experiment.py

import os
import torch
from pathlib import Path
import sys
import csv

# Set CUDA DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from datetime import datetime

# Add project root to system path as before
project_root = Path(__file__).resolve().parent
while not (project_root / ".git").exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from models.causal_neutral_model_variations import model_variations
from trainers.trainer import Trainer, save_trained_model
from optimizers.optimizer_params import optimizer_configs

from data_loaders.imdb_sentiment.causal_only import CausalOnlyIMDBDataModule


def get_model_specific_params(model_name: str):
    """Get model-specific learning rates and parameters."""
    
    params = {
        "electra_small_discriminator": {
            "lr": 1e-4,  # Higher LR due to discriminative pretraining
            "weight_decay": 0.1,
            "layer_wise_lr_decay": 0.9,  # Steeper decay as discriminator layers are more specialized
            "batch_size": 32  # Bit bigger since it's a smaller model          
        },
        "distilbert": {
            "lr": 2e-5,  # Standard LR as it's well-balanced
            "weight_decay": 0.01,
            "layer_wise_lr_decay": 0.95,
            "batch_size": 16
        },
        "roberta": {
            "lr": 1e-5,  # Lower LR as it's sensitive to training dynamics
            "weight_decay": 0.01,
            "layer_wise_lr_decay": 0.9,  # Steeper decay for better fine-tuning
            "batch_size": 16
        },
        "bert": {
            "lr": 2e-5,  # Standard BERT learning rate
            "weight_decay": 0.01,
            "layer_wise_lr_decay": 0.95,
            "batch_size": 16
        },
        "albert": {
            "lr": 5e-5,  # Higher LR due to parameter sharing
            "weight_decay": 0.02,
            "layer_wise_lr_decay": 0.98,  # Less decay due to shared parameters
            "batch_size": 16
        },
        "deberta": {
            "lr": 8e-6,  # Lower LR as it's a more sophisticated model
            "weight_decay": 0.05,
            "layer_wise_lr_decay": 0.8,  # Steeper decay for disentangled attention
            "batch_size": 4 # smaller as larger size model
        },
        "modern_bert": {
            "lr": 1e-5,  # Similar to roberta
            "weight_decay": 0.05,
            "layer_wise_lr_decay": 0.9,
            "batch_size": 16
        }
    }
    
    # Default values if model not found
    default_params = {
        "lr": 2e-5,
        "weight_decay": 0.01,
        "layer_wise_lr_decay": 0.95
    }
    
    return params.get(model_name, default_params)

def run_causal_only_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to your precomputed JSON files
    train_json = "data/imdb_sentiment/train_with_causal_neutral_splits_bert_40.json"
    test_json  = "data/imdb_sentiment/test_with_causal_neutral_splits_bert_40.json"

    # 1) Create the data module
    data_module = CausalOnlyIMDBDataModule(
        train_json=train_json,
        test_json=test_json,
        val_split=0.1  # e.g., 90/10 split for train/val (if your data loader uses val_split)
    )

    # 2) Some experiment settings
    models = ["electra_small_discriminator", "distilbert", "roberta", "bert", "albert", "deberta", "modern_bert"]
    # models = ["electra_small_discriminator", "modern_bert"]
    models = ["roberta", "distilbert"]
    # models = ["bert", "albert"]
    # models = ["deberta"]
    classification_word = "Sentiment"
    hidden_layer = "2_hidden"   # or "1_hidden"
    epochs = [10, 20, 40, 60, 80, 100]

    # 3) Optimizer & training hyperparameters
    base_training_params = {
        "layer_wise_lr_decay": 0.95,
        "max_grad_norm": 5.0,
        "warmup_ratio": 0.1,
        "cosine_decay": True,
        "drop_lr_on_plateau": False
    }

    # 4) Prepare results CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/imdb_causal_only_precomputed"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{timestamp}.csv")

    # 5) Run experiments, writing stats for each (model_name, num_epochs) pair
    with open(results_file, "w", newline="") as csvfile:
        fieldnames = [
            "model", "epochs", "final_loss", "test_loss",
            "test_accuracy", "test_precision", "test_recall", "test_f1"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_name in models:           
            # Get model-specific parameters
            model_params = get_model_specific_params(model_name)
            
            # Update optimizer and training params with model-specific values
            optimizer_params = {
                "lr": model_params["lr"],
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": model_params["weight_decay"]
            }
            
            training_params = {
                **base_training_params,
                "layer_wise_lr_decay": model_params["layer_wise_lr_decay"]
            }
            batch_size = model_params.get("batch_size", 16)           
            
            for num_epochs in epochs:
                print(f"Training {model_name} for {num_epochs} epochs...")
                
                # (A) Create the model
                model = model_variations[model_name][hidden_layer](
                    classification_word,
                    freeze_encoder=False
                ).to(device)
                
                # (B) Build the trainer
                trainer = Trainer(
                    model=model,
                    data_module=data_module,
                    optimizer_name="adamw",
                    optimizer_params=optimizer_params,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    device=device,
                    dataset_name="imdb_causal_only_precomputed",
                    **training_params  # include the extra training params
                )               

                # (C) Train on the full dataset (train + val combined)
                epoch_losses = trainer.train_on_full_dataset(num_epochs)
                final_loss   = epoch_losses[-1] if epoch_losses else None

                # (D) Evaluate on test
                test_loss, test_acc, test_prec, test_rec, test_f1 = trainer.test()

                # (E) Write results to CSV
                writer.writerow({
                    "model": model_name,
                    "epochs": num_epochs,
                    "final_loss": final_loss,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "test_precision": test_prec,
                    "test_recall": test_rec,
                    "test_f1": test_f1
                })
                csvfile.flush()

                # (F) Save the model with the helper
                #    Note the last arg is the hidden layers used
                save_trained_model(
                    trainer,
                    f"imdb_causal_only_{model_name}_{num_epochs}epochs",
                    num_hidden_layers=2
                )

                print(
                    f"Done: {model_name}, epochs={num_epochs}, "
                    f"test_acc={test_acc:.4f}, test_f1={test_f1:.4f}"
                )
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_causal_only_experiment()
