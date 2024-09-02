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
    causal_neutral_model_path = "models/imdb_sentiment_wz_bert_10epochs/sentiment/CausalNeutralClassifier_2hidden_2024-09-02_19-50-51.pt"
    causal_neutral_model = load_causal_neutral_classifier(causal_neutral_model_path).to(device)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Initialize the CausalPhraseIMDBDataModule
    data_module = CausalPhraseIMDBDataModule()
    data_module.set_causal_neutral_model(causal_neutral_model, tokenizer)

    # Experiment parameters
    models = ["electra_small_discriminator", "distilbert", "t5", "roberta", "bert", "albert"]
    classification_word = "Sentiment"
    epochs = [5, 10]
    batch_size = 4

    # Hyperparameters
    optimizer_name = "adamw"
    hidden_layer = "1_hidden"
    learning_rate = 0.0001

    # Prepare results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"causal_phrase_sentiment_results_{timestamp}.csv"

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['model', 'epochs', 'train_loss', 'val_loss', 'accuracy', 'precision', 'recall', 'f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_name in models:
            for num_epochs in epochs:
                print(f"Running experiment: {model_name}, epochs={num_epochs}")

                # Initialize sentiment classifier model
                sentiment_model = model_variations[model_name][hidden_layer](classification_word,
                                                                             freeze_encoder=False).to(device)

                # Set up optimizer
                optimizer_config = optimizer_configs[optimizer_name].copy()
                optimizer_config['params'] = optimizer_config['params'].copy()
                optimizer_config['params']['lr'] = learning_rate

                # Create trainer
                trainer = Trainer(sentiment_model, data_module, optimizer_name=optimizer_name,
                                  dataset_name='imdb_causal_phrases',
                                  optimizer_params=optimizer_config['params'],
                                  batch_size=batch_size, num_epochs=num_epochs, device=device)

                # Train the model
                training_history = trainer.train()

                # Test the model
                test_loss, accuracy, precision, recall, f1 = trainer.test()

                # Write results
                writer.writerow({
                    'model': model_name,
                    'epochs': num_epochs,
                    'train_loss': training_history[-1]['train_loss'],
                    'val_loss': training_history[-1]['val_loss'],
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
                csvfile.flush()  # Ensure data is written immediately

                # Save the trained model
                save_trained_model(trainer, f"imdb_causal_phrases_{model_name}_{num_epochs}epochs", 1)

                print(f"Completed {model_name}, epochs={num_epochs}")
                print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                print("------------------------------------------------------")

    print(f"Experiment completed. Results saved to {results_file}")


if __name__ == "__main__":
    run_causal_phrase_sentiment_experiment()