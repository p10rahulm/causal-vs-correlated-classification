# causal_only_experiment.py

import os
import torch
from pathlib import Path
import sys
import csv
from datetime import datetime

# Set CUDA DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / ".git").exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from models.causal_neutral_model_variations import model_variations
from trainers.trainer import Trainer, save_trained_model

# Data module that loads only the "causal" portion
from data_loaders.imdb_sentiment.causal_only import CausalOnlyIMDBDataModule

def run_causal_only_experiment():
    """
    Example experiment that trains on causal-only data from IMDB,
    using the updated Trainer with cyclical triangular schedule.
    Trains on the entire dataset (train+val) with mid-epoch validation
    every 4 epochs, logs each epoch to CSV, then does a final test.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # JSON data paths (adapt to your actual file paths)
    train_json = "data/imdb_sentiment/train_with_causal_neutral_splits_bert_40.json"
    test_json  = "data/imdb_sentiment/test_with_causal_neutral_splits_bert_40.json"

    # Build the data module (train + test).
    # If your data loader internally splits out 10% for validation, so be it.
    data_module = CausalOnlyIMDBDataModule(
        train_json=train_json,
        test_json=test_json,
        val_split=0.1
    )

    # We can run multiple models if we like:
    models = [
        "albert",
        "bert",
        "distilbert",
        "roberta",
        "electra_small_discriminator",
        "modern_bert",
        "deberta",
    ]

    # We'll do only 20 epochs for each, as an example
    num_epochs = 20

    # Classification word in your head
    classification_word = "Sentiment"
    # We'll use 2 hidden layers
    hidden_layer = "2_hidden"

    # Baseline hyperparams matching your previous usage:
    base_lr = 5e-4
    batch_size = 16

    # We'll store everything in one CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/imdb_causal_only_precomputed"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{timestamp}.csv")

    # Prepare the CSV columns
    with open(results_file, "w", newline="") as csvfile:
        fieldnames = [
            "model",
            "epoch",
            "train_loss",
            "val_loss",
            "test_loss",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_name in models:
            print(f"\n===== Starting {model_name} for {num_epochs} epochs =====")

            # (1) Create model with 2 hidden layers, freeze_encoder=False
            model = model_variations[model_name][hidden_layer](
                classification_word,
                freeze_encoder=False
            ).to(device)

            # (2) Build the Trainer with cyclical triangular LR schedule
            #     and your chosen baseline hyperparams
            trainer = Trainer(
                model=model,
                data_module=data_module,
                optimizer_name="adamw",
                optimizer_params={
                    "lr": base_lr,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.01
                },
                batch_size=batch_size,
                num_epochs=num_epochs,
                device=device,
                dataset_name="imdb_causal_only_precomputed",
                # Baseline training params
                layer_wise_lr_decay=0.95,
                max_grad_norm=5.0,
                warmup_ratio=0.1,
                cosine_decay=False,            # Turn off default cos decay
                drop_lr_on_plateau=False,
                lr_schedule="cyclic_triangular",
                cycle_length_epochs=num_epochs // 10,  # e.g. 20//10=2
                # We'll do mid-epoch validation every 4 epochs *inside*
                # trainer.train_on_full_dataset, as you coded.
            )

            # (3) Train on the entire dataset with mid-epoch validations
            #     returning a list of epoch-level stats
            training_history = trainer.train_on_full_dataset(num_epochs)
            # Write each epoch's stats to CSV
            for epoch_stats in training_history:
                row_dict = {
                    "model": model_name,
                    "epoch": epoch_stats["epoch"],
                    "train_loss": epoch_stats["train_loss"],
                    "val_loss": epoch_stats["val_loss"],
                    # test metrics will only be filled after we do final test
                    "test_loss": None,
                    "test_accuracy": None,
                    "test_precision": None,
                    "test_recall": None,
                    "test_f1": None
                }
                writer.writerow(row_dict)
                csvfile.flush()

            # (4) Final Test
            test_loss, test_acc, test_prec, test_rec, test_f1 = trainer.test()

            # Write a final row for the test result
            row_dict = {
                "model": model_name,
                "epoch": "final_test",
                "train_loss": None,
                "val_loss": None,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "test_precision": test_prec,
                "test_recall": test_rec,
                "test_f1": test_f1
            }
            writer.writerow(row_dict)
            csvfile.flush()

            # (5) Save trained model
            num_hidden_layers = int(hidden_layer[0])
            trainer.save_trained_model_with_path(
                dataset_name=f"imdb_causal_only_{model_name}_{num_epochs}epochs",
                num_hidden_layers=num_hidden_layers
            )

            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\nAll experiments done. Results saved to {results_file}")


if __name__ == "__main__":
    run_causal_only_experiment()
