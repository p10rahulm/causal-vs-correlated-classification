import os
from pathlib import Path
import sys

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer

from data_loaders.classification_data_loaders.causal_sentiment_imdb import CausalPhraseIMDBDataModule
from models.causal_neutral_model_variations import model_variations
from trainers.trainer import Trainer, save_trained_model
from optimizers.optimizer_params import optimizer_configs
import csv
from datetime import datetime


def load_causal_neutral_classifier(model_path):
    model = model_variations["bert"]["2_hidden"]("Sentiment", freeze_encoder=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def run_causal_phrase_sentiment_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Causal Neutral classifier
    causal_neutral_model_path = "trained_models/imdb_sentiment_wz_bert_10epochs/sentiment/CausalNeutralClassifier_2hidden_2024-09-02_19-50-51.pth"
    causal_neutral_model = load_causal_neutral_classifier(causal_neutral_model_path).to(device)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Initialize the CausalPhraseIMDBDataModule
    data_module = CausalPhraseIMDBDataModule()
    data_module.set_causal_neutral_model(causal_neutral_model, tokenizer)

    # Experiment parameters
    models = ["roberta", "albert", "distilbert", "bert", "electra_small_discriminator", "t5"]

    classification_word = "Sentiment"
    epochs = [5, 10]
    batch_size = 32

    # Hyperparameters
    optimizer_name = "adamw"
    hidden_layer = "1_hidden"
    learning_rate = 0.0001

    # Prepare results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/imdb_sentiment_causal_phrase_baseline"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{timestamp}.csv")

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['model', 'epochs', 'final_loss', 'test_loss', 'test_accuracy', 'test_precision', 'test_recall',
                      'test_f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_name in models:
            for num_epochs in epochs:
                print(f"Running experiment: {model_name}, epochs={num_epochs}")

                # Initialize sentiment classifier model
                model = model_variations[model_name][hidden_layer](classification_word, freeze_encoder=False).to(device)

                # Update optimizer config
                optimizer_config = optimizer_configs[optimizer_name].copy()
                optimizer_config['params'] = optimizer_config['params'].copy()
                optimizer_config['params']['lr'] = learning_rate

                # Create trainer
                trainer = Trainer(model, data_module, optimizer_name=optimizer_name,
                                  dataset_name='imdb_causal_phrases',
                                  optimizer_params=optimizer_config['params'],
                                  batch_size=batch_size, num_epochs=num_epochs, device=device)

                # Train on full dataset
                print(f"Training {model_name} for {num_epochs} epochs...")
                epoch_losses = trainer.train_on_full_dataset(num_epochs)

                # Evaluate on test set
                test_loss, test_accuracy, test_precision, test_recall, test_f1 = trainer.test()

                # Write results
                writer.writerow({
                    'model': model_name,
                    'epochs': num_epochs,
                    'final_loss': epoch_losses[-1],
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1
                })
                csvfile.flush()  # Ensure data is written immediately

                # Save the trained model
                save_trained_model(trainer, f"imdb_causal_phrases_{model_name}_{num_epochs}epochs", 1)

                print(f"Completed {model_name}, epochs={num_epochs}")
                print(f"Final Loss: {epoch_losses[-1]:.4f}")
                print(f"Test Accuracy: {test_accuracy:.4f}")
                print("------------------------------------------------------")

    print(f"Experiment completed. Results saved to {results_file}")


if __name__ == "__main__":
    run_causal_phrase_sentiment_experiment()