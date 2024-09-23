import os
from pathlib import Path
import sys
import csv
from itertools import product
import datetime

# Try to find the project root
current_file = Path(__file__).resolve()
project_root = current_file.parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

from models.causal_neutral_model_variations import model_variations
from data_loaders.wz_data_loaders.sentiment_imdb import CausalNeutralDataModule
from trainers.trainer import Trainer, save_trained_model
from optimizers.optimizer_params import optimizer_configs


def run_experiments():
    # Experiment parameters
    models = ["electra_small_discriminator", "distilbert", "t5", "roberta", "bert"]
    optimizers = ["adam", "adamw", "nadam"]
    hidden_layers = ["0_hidden", "1_hidden", "2_hidden"]
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    classification_word = "Sentiment"
    num_epochs = 5
    batch_size = 32

    # Prepare data loader
    file_path = "outputs/imdb_train_sentiment_analysis.json"


    # Prepare results file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/wz_classifier_experiments"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"experiment_results_{timestamp}.csv")

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['model', 'optimizer', 'hidden_layers', 'learning_rate', 'epoch', 'train_loss', 'val_loss',
                    'accuracy', 'precision', 'recall', 'f1', 'best_val_loss', 'best_accuracy', 'best_f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Run experiments
        for model_name, optimizer_name, hidden_layer, lr in product(models, optimizers, hidden_layers, learning_rates):
            print(f"Running experiment: {model_name}, {optimizer_name}, {hidden_layer}, lr={lr}")

            # Create model
            model = model_variations[model_name][hidden_layer](classification_word, freeze_encoder=True)

            # Update optimizer config with current learning rate
            optimizer_config = optimizer_configs[optimizer_name].copy()
            optimizer_config['params'] = optimizer_config['params'].copy()
            optimizer_config['params']['lr'] = lr

            # Create a fresh instance of the data module for each experiment
            data_module = CausalNeutralDataModule(file_path, classification_word)

            # Create and run trainer
            trainer = Trainer(model, data_module, optimizer_name=optimizer_name,
                            optimizer_params=optimizer_config['params'],
                            batch_size=batch_size, num_epochs=num_epochs)

            # Prepare data before training loop
            trainer.prepare_data()

            best_val_loss = float('inf')
            best_accuracy = 0
            best_f1 = 0

            for epoch in range(num_epochs):
                train_loss = trainer.train_epoch()
                val_loss, accuracy, precision, recall, f1 = trainer.validate()

                best_val_loss = min(best_val_loss, val_loss)
                best_accuracy = max(best_accuracy, accuracy)
                best_f1 = max(best_f1, f1)

                # Write results
                writer.writerow({
                    'model': model_name,
                    'optimizer': optimizer_name,
                    'hidden_layers': hidden_layer,
                    'learning_rate': lr,
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'best_val_loss': best_val_loss,
                    'best_accuracy': best_accuracy,
                    'best_f1': best_f1
                })
                csvfile.flush()  # Ensure data is written immediately

            # Save the trained model
            save_trained_model(trainer, "imdb", int(hidden_layer[0]))

    print(f"Experiments completed. Results saved to {results_file}")


if __name__ == "__main__":
    run_experiments()
