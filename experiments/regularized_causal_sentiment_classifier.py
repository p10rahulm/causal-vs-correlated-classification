import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
from data_loaders.classification_data_loaders.sentiment_imdb import IMDBDataModule
from models.causal_neutral_model_variations import model_variations
from trainers.regularized_trainer import RegularizedTrainer
from trainers.trainer import save_trained_model
from optimizers.optimizer_params import optimizer_configs
import csv
from datetime import datetime
from utilities import get_project_root

# Add project root to system path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def load_trained_model(model_path, model_class):
    model = model_class
    model.load_state_dict(torch.load(model_path))
    return model

def find_model_file(directory):
    for file in os.listdir(directory):
        if file.endswith(".pth"):
            return os.path.join(directory, file)
    return None

def run_regularized_imdb_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Causal Neutral classifier (for causal phrase extraction)
    causal_neutral_model_path = "trained_models/imdb_sentiment_wz_bert_10epochs/sentiment/CausalNeutralClassifier_2hidden_2024-09-02_19-50-51.pth"
    causal_neutral_model = load_trained_model(causal_neutral_model_path, model_variations["bert"]["2_hidden"]("Sentiment", freeze_encoder=True)).to(device)

    # Initialize the IMDBDataModule
    data_module = IMDBDataModule()

    # Experiment parameters
    models = ["roberta", "albert", "distilbert", "bert", "electra_small_discriminator", "t5"]
    epochs = [5, 10]
    classification_word = "Sentiment"
    batch_size = 32
    num_epochs = 5  # for regularization training
    lambda_reg = 0.1

    # Hyperparameters
    optimizer_name = "adamw"
    hidden_layer = "1_hidden"
    learning_rate = 0.0001

    # Prepare results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/imdb_sentiment_regularized"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{timestamp}.csv")

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['model', 'original_epochs', 'regularized_epochs', 'final_loss', 'test_loss', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_name in models:
            for original_epochs in epochs:
                print(f"Running experiment: {model_name}, original_epochs={original_epochs}")

                # Load the trained model from Baseline 1 (P_η)
                model_eta_path = find_model_file(f"trained_models/imdb_sentiment_naive_baseline_{model_name}_{original_epochs}epochs/sentiment")
                if model_eta_path is None:
                    print(f"Model file not found for {model_name} with {original_epochs} epochs. Skipping...")
                    continue

                model_eta = load_trained_model(model_eta_path, model_variations[model_name][hidden_layer](classification_word, freeze_encoder=True)).to(device)

                # Create a copy of the model (P_θ)
                model_theta = model_variations[model_name][hidden_layer](classification_word, freeze_encoder=False).to(device)
                model_theta.load_state_dict(model_eta.state_dict())

                # Update optimizer config
                optimizer_config = optimizer_configs[optimizer_name].copy()
                optimizer_config['params'] = optimizer_config['params'].copy()
                optimizer_config['params']['lr'] = learning_rate

                # Create RegularizedTrainer
                trainer = RegularizedTrainer(
                    model_eta=model_eta,
                    model_theta=model_theta,
                    data_module=data_module,
                    causal_phrase_model=causal_neutral_model,
                    optimizer_name=optimizer_name,
                    dataset_name='imdb_regularized',
                    optimizer_params=optimizer_config['params'],
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    device=device,
                    lambda_reg=lambda_reg
                )

                # Run the regularized training
                print(f"Training regularized {model_name} for {num_epochs} epochs...")
                epoch_losses = trainer.train_with_regularization()

                # Evaluate on test set
                test_loss, test_accuracy, test_precision, test_recall, test_f1 = trainer.test()

                # Write results
                writer.writerow({
                    'model': model_name,
                    'original_epochs': original_epochs,
                    'regularized_epochs': num_epochs,
                    'final_loss': epoch_losses[-1],
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1
                })
                csvfile.flush()  # Ensure data is written immediately

                # Save the regularized model
                save_trained_model(trainer, f"imdb_regularized_{model_name}_{original_epochs}_{num_epochs}epochs", 1)

                print(f"Completed {model_name}, original_epochs={original_epochs}, regularized_epochs={num_epochs}")
                print(f"Final Loss: {epoch_losses[-1]:.4f}")
                print(f"Test Accuracy: {test_accuracy:.4f}")
                print("------------------------------------------------------")

    print(f"Experiment completed. Results saved to {results_file}")


if __name__ == "__main__":
    run_regularized_imdb_experiment()