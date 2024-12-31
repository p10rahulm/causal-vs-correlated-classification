import os
from pathlib import Path
import sys
import csv
from datetime import datetime

# Set CUDA DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from models.causal_neutral_model_variations import model_variations
from data_loaders.imdb_sentiment.naive import IMDBDataModule
from trainers.trainer import Trainer, save_trained_model
from optimizers.optimizer_params import optimizer_configs


def run_imdb_sentiment_experiment():
    # Experiment parameters
    models = ["electra_small_discriminator", "distilbert", "roberta", "bert", "albert", "deberta", "modern_bert"]
    # models = ["electra_small_discriminator", "distilbert"]
    # models = ["roberta"]
    # models = ["bert", "albert"]
    # models = ["albert"]
    # models = ["deberta"]
    # models = ["modern_bert"]
    classification_word = "Sentiment"
    epochs = [5, 10]
    batch_size = 16

    # Hyperparameters (using DistilBERT's optimal parameters for all models)
    optimizer_name = "adamw"
    hidden_layer = "1_hidden"
    learning_rate = 0.0001

    # Prepare data loader
    data_module = IMDBDataModule(classification_word)

    # Prepare results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/imdb_sentiment_classifier_naive_baseline"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{timestamp}.csv")

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['model', 'epochs', 'train_loss', 'test_loss', 'accuracy', 'precision', 'recall', 'f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_name in models:
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

                # Save the trained model
                save_trained_model(trainer, f"imdb_sentiment_naive_baseline_{model_name}_{num_epochs}epochs",
                                   int(hidden_layer[0]))

        print(f"Experiments completed. Results saved to {results_file}")


if __name__ == "__main__":
    run_imdb_sentiment_experiment()
