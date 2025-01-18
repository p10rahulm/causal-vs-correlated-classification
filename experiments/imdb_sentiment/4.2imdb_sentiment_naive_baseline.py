import os
import torch
from pathlib import Path
import sys
import csv
from datetime import datetime

# Set CUDA DEVICE (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

# Imports
from models.causal_neutral_model_variations import model_variations
from data_loaders.imdb_sentiment.naive import IMDBDataModule
from trainers.trainer import Trainer, save_trained_model
# (Make sure your 'trainer.py' is the updated version you showed above!)

def run_imdb_sentiment_experiment():
    """
    Run experiments for IMDB sentiment classification using various models.
    Each model:
      - uses lr=5e-4
      - cycic_triangular LR schedule with cycle_length=num_epochs//10
      - batch_size=16
      - 20 epochs total
      - validation every 4 epochs
      - 2 hidden layers, freeze_encoder=False
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # List of models to run
    models = [
        "albert",
        "albert_xxlarge",
        "bert",
        "roberta",
        "roberta_large"
        "electra_large",
    ]


    # Classification setup
    classification_word = "Sentiment"
    hidden_layer = "2_hidden"  # You want 2 hidden layers
    num_epochs = 20
    test_interval = 4   # Validate every 4 epochs
    base_lr = 5e-4
    batch_size = 8
    max_length = 1024
    # Common training params for ALL models
    base_training_params = {
        "layer_wise_lr_decay": 0.95,
        "max_grad_norm": 5.0,
        "warmup_ratio": 0.1,
        "cosine_decay": False,                # turn off cosine; we'll use cycic triangular
        "drop_lr_on_plateau": False,
        "lr_schedule": "cyclic_triangular2",
        "cycle_length_epochs": num_epochs // 10,  # e.g. 20//10 = 2
        "num_epochs_eval": test_interval,
    }


    # Prepare output CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/imdb_sentiment_classifier_naive_baseline"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{timestamp}.csv")

    # CSV headers
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = [
            'model',         # Which model
            'epoch',         # Which epoch
            'train_loss',    # Training loss
            'val_loss',      # Validation loss (0 if not done that epoch)
            'test_loss',     # For final test row
            'test_accuracy',
            'test_precision',
            'test_recall',
            'test_f1'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Loop over each model
        for model_name in models:
            print(f"Running experiment: {model_name}, epochs={num_epochs}")

            # Create model with 2 hidden layers, freeze_encoder=False
            model = model_variations[model_name][hidden_layer](
                classification_word,
                freeze_encoder=False,
                max_length=max_length
            )
            
            # Build data module
            data_module = IMDBDataModule(classification_word, max_length=model.max_length)


            # Build optimizer settings
            optimizer_name = "adamw"
            optimizer_params = {
                "lr": base_lr,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01
            }

            # Create trainer
            trainer = Trainer(
                model=model,
                data_module=data_module,
                optimizer_name=optimizer_name,
                optimizer_params=optimizer_params,
                batch_size=batch_size,
                num_epochs=num_epochs,
                device=device,
                dataset_name="imdb_naive_baseline",
                csv_writer=writer,
                csv_file=csvfile,
                **base_training_params
            )

            # ----------------- Train -----------------
            training_history = trainer.train_on_full_dataset(num_epochs)

            # Log each epoch's results to the CSV
            for epoch_stats in training_history:
                row_dict = {
                    'model': model_name,
                    'epoch': epoch_stats['epoch'],
                    'train_loss': epoch_stats['train_loss'],
                    'val_loss': epoch_stats['val_loss'],
                    # We won't have test metrics except in final row
                    'test_loss': None,
                    'test_accuracy': None,
                    'test_precision': None,
                    'test_recall': None,
                    'test_f1': None
                }
                writer.writerow(row_dict)
                csvfile.flush()


            # ----------------- Test ------------------
            test_loss, test_accuracy, test_precision, test_recall, test_f1 = trainer.test()

            # Write a "final test" row
            row_dict = {
                'model': model_name,
                'epoch': f"naive_baseline",
                'train_loss': None,
                'val_loss': None,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1
            }
            writer.writerow(row_dict)
            csvfile.flush()

            # ----------------- Save model -------------
            # For "2_hidden" we parse out '2' from the string:
            num_hidden_layers = int(hidden_layer[0])
            trainer.save_trained_model_with_path(
                dataset_name=f"imdb_sentiment_naive_baseline_{model_name}_{num_epochs}epochs",
                num_hidden_layers=num_hidden_layers
            )
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Experiments completed. Results saved to {results_file}")


if __name__ == "__main__":
    run_imdb_sentiment_experiment()
