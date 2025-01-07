# causal_only_experiment.py

import os
import torch
from pathlib import Path
import sys
import csv

# Set CUDA DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from datetime import datetime
from transformers import AutoTokenizer

# Add project root to system path as before
project_root = Path(__file__).resolve().parent
while not (project_root / ".git").exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from models.causal_neutral_model_variations import model_variations
from trainers.trainer import Trainer, save_trained_model
from optimizers.optimizer_params import optimizer_configs

from data_loaders.imdb_sentiment.causal_only import CausalOnlyIMDBDataModule


def run_causal_only_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to your precomputed JSON files
    train_json = "data/imdb_sentiment/train_with_causal_neutral_splits_bert.json"
    test_json  = "data/imdb_sentiment/test_with_causal_neutral_splits_bert.json"

    # Create the data module
    data_module = CausalOnlyIMDBDataModule(
        train_json=train_json,
        test_json=test_json,
        val_split=0.1
    )

    # Some experiment settings
    models = ["electra_small_discriminator", "distilbert", "roberta", "bert", "albert", "deberta", "modern_bert"]
    # models = ["electra_small_discriminator", "modern_bert"]
    # models = ["roberta", "distilbert"]
    # models = ["bert", "albert"]
    # models = ["deberta"]
    classification_word = "Sentiment"
    epochs = [5, 10]
    batch_size = 16

    # Improved optimizer settings
    optimizer_params = {
        "lr": 2e-5,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01
    }

    # Additional training parameters
    training_params = {
        "layer_wise_lr_decay": 0.95,
        "max_grad_norm": 5.0,
        "warmup_ratio": 0.1,
        "cosine_decay": True,
        "drop_lr_on_plateau": False
    }

    # Prepare results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/imdb_causal_only_precomputed"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{timestamp}.csv")

    with open(results_file, "w", newline="") as csvfile:
        fieldnames = [
            "model", "epochs", "final_loss", "test_loss",
            "test_accuracy", "test_precision", "test_recall", "test_f1"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_name in models:
            for num_epochs in epochs:
                print(f"Training {model_name} for {num_epochs} epochs...")

                # Create model, freeze_encoder=False so you can fine-tune the entire thing
                # model = model_variations[model_name][hidden_layer](classification_word, freeze_encoder=False).to(device)
                model = model_variations[model_name][hidden_layer](classification_word).to(device)

                # Build trainer
                trainer = Trainer(
                    model=model,
                    data_module=data_module,
                    optimizer_name="adamw",
                    optimizer_params=optimizer_params,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    device=device,
                    dataset_name="imdb_causal_only_precomputed",
                    **training_params  # Include the additional training parameters
                )

                # Train on the full dataset (train+val)
                epoch_losses = trainer.train_on_full_dataset(num_epochs)

                # Evaluate on test
                test_loss, test_acc, test_prec, test_rec, test_f1 = trainer.test()

                # Save stats
                writer.writerow({
                    "model": model_name,
                    "epochs": num_epochs,
                    "final_loss": epoch_losses[-1],
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "test_precision": test_prec,
                    "test_recall": test_rec,
                    "test_f1": test_f1
                })
                csvfile.flush()

                # Optionally save the model
                save_trained_model(trainer, f"imdb_causal_only_{model_name}_{num_epochs}epochs", 1)

                print(f"Done: {model_name}, epochs={num_epochs}, test_acc={test_acc:.4f}")


if __name__ == "__main__":
    run_causal_only_experiment()
