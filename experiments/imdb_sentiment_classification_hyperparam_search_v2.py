

import os
from pathlib import Path
import sys
from datetime import datetime

# Set CUDA DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))


# Try to find the project root
current_file = Path(__file__).resolve()
project_root = current_file.parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

import csv
from itertools import product
from models.causal_neutral_model_variations import model_variations
from data_loaders.imdb_sentiment.naive import IMDBDataModule
from trainers.trainer import Trainer, save_trained_model
from optimizers.optimizer_params import optimizer_configs


def run_experiments():
    # Experiment parameters
    models = ["albert"]
    optimizers = ["adamw"]
    hidden_layers = ["1_hidden", "2_hidden"]
    learning_rates = [5e-5, 1e-4]
    classification_word = "Sentiment"
    num_epochs = 20
    batch_size = 16

    # Prepare data loader
    data_module = IMDBDataModule(classification_word)

    # Prepare results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/imdb_sentiment_classifier_experiments"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"experiment_results_{timestamp}.csv")

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['model', 'epochs', 'train_loss', 'test_loss', 'accuracy', 'precision', 'recall', 'f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Run experiments (Hyperparameter search)
        for model_name, optimizer_name, hidden_layer, lr in product(models, optimizers, hidden_layers, learning_rates):
            print(f"Running experiment: {model_name}, {optimizer_name}, {hidden_layer}, lr={lr}")

            # Create model
            model = model_variations[model_name][hidden_layer](classification_word, freeze_encoder=False)

            # Update optimizer config with current learning rate
            optimizer_config = optimizer_configs[optimizer_name].copy()
            optimizer_config['params'] = optimizer_config['params'].copy()
            optimizer_config['params']['lr'] = lr

            # Create and run trainer
            trainer = Trainer(model, data_module, optimizer_name=optimizer_name,
                            optimizer_params=optimizer_config['params'],
                            batch_size=batch_size, num_epochs=num_epochs)

            # Train on full dataset
            print(f"Training {model_name} for {num_epochs} epochs...")
            epoch_losses = trainer.train_on_full_dataset(num_epochs)

            # Evaluate on test set
            test_loss, test_accuracy, test_precision, test_recall, test_f1 = trainer.test()
            
            # Write results
            writer.writerow({
                'model': model_name,
                'epochs': num_epochs,
                'train_loss': epoch_losses[-1],  # Last epoch's training loss
                'test_loss': test_loss,
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1': test_f1
            })
            csvfile.flush()  # Ensure data is written immediately



if __name__ == "__main__":
    run_experiments()