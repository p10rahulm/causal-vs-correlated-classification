import os
import json
from pathlib import Path
import sys
import csv
from datetime import datetime

# Set CUDA DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from models.causal_neutral_model_variations import model_variations
from data_loaders.imdb_sentiment.phrase_classification_dataloader import CausalNeutralDataModule

from trainers.trainer import Trainer, save_trained_model
from optimizers.optimizer_params import optimizer_configs

def load_hyperparameters(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_hyperparameters(model_name, hyperparams):
    if model_name in hyperparams:
        return {
            'optimizer_name': hyperparams[model_name]['optimizer'],
            'hidden_layer': hyperparams[model_name]['hidden_layers'],
            'learning_rate': hyperparams[model_name]['learning_rate']
        }
    else:
        return {
            'optimizer_name': 'adamw',
            'hidden_layer': '2_hidden',
            'learning_rate': 0.0005
        }

def run_imdb_sentiment_experiment():
    # Experiment parameters
    models = ["deberta", "electra_small_discriminator", "distilbert", "roberta", "bert", "albert", "modern_bert"]
    classification_word = "Sentiment"
    epochs = [5, 10]
    batch_size = 16

    # Load hyperparameters
    hyperparams = load_hyperparameters('models/optimal_wz_classifier_validation_hyperparams.json')

    # Prepare data loader
    file_path = "outputs/imdb_train_sentiment_analysis.json"  # Update this path if necessary
    data_module = CausalNeutralDataModule(file_path, classification_word)

    # Prepare results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/imdb_sentiment_classifier_experiment"
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
                # model = model_variations[model_name][hidden_layer](classification_word, freeze_encoder=True)
                model = model_variations[model_name][hidden_layer](classification_word)

                # Update optimizer config
                optimizer_config = optimizer_configs[optimizer_name].copy()
                optimizer_config['params'] = optimizer_config['params'].copy()
                optimizer_config['params']['lr'] = learning_rate

                # Create trainer
                trainer = Trainer(model, data_module, optimizer_name=optimizer_name,
                                  optimizer_params=optimizer_config['params'],
                                  batch_size=batch_size, num_epochs=num_epochs)

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
                save_trained_model(trainer, f"imdb_sentiment_wz_{model_name}_{num_epochs}epochs",
                                   int(hidden_layer[0]))

        print(f"Experiments completed. Results saved to {results_file}")

if __name__ == "__main__":
    run_imdb_sentiment_experiment()