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

from models.causal_neutral_model_variations import model_variations
from data_loaders.imdb_sentiment.naive import IMDBDataModule
from trainers.trainer import Trainer, save_trained_model
from optimizers.optimizer_params import optimizer_configs

###############################
# 1) Model-specific params
###############################
def get_model_specific_params(model_name: str):
    params = {
        "electra_small_discriminator": {
            "lr": 1e-4,
            "weight_decay": 0.1,
            "layer_wise_lr_decay": 0.9,
            "batch_size": 32
        },
        "distilbert": {
            "lr": 2e-5,
            "weight_decay": 0.01,
            "layer_wise_lr_decay": 0.95,
            "batch_size": 16
        },
        "roberta": {
            "lr": 1e-5,
            "weight_decay": 0.01,
            "layer_wise_lr_decay": 0.9,
            "batch_size": 16
        },
        "bert": {
            "lr": 2e-5,
            "weight_decay": 0.01,
            "layer_wise_lr_decay": 0.95,
            "batch_size": 16
        },
        "albert": {
            "lr": 5e-5,
            "weight_decay": 0.02,
            "layer_wise_lr_decay": 0.98,
            "batch_size": 16
        },
        "deberta": {
            "lr": 8e-6,
            "weight_decay": 0.05,
            "layer_wise_lr_decay": 0.8,
            "batch_size": 4
        },
        "modern_bert": {
            "lr": 1e-5,
            "weight_decay": 0.05,
            "layer_wise_lr_decay": 0.9,
            "batch_size": 16
        }
    }
    default_params = {
        "lr": 2e-5,
        "weight_decay": 0.01,
        "layer_wise_lr_decay": 0.95,
        "batch_size": 16
    }
    return params.get(model_name, default_params)

def run_imdb_sentiment_experiment():
    """
    Run naive baseline experiments for IMDB sentiment classification using various models.
    Tests different epoch counts and model architectures, saving results and models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Experiment parameters
    
    models = ["electra_small_discriminator", "distilbert", "roberta", "bert", "albert", "deberta", ""]
    models = ["electra_small_discriminator", "modern_bert"]
    # models = ["roberta","distilbert"]
    # models = ["albert", "bert"]
    # models = ["deberta"]
    classification_word = "Sentiment"
    hidden_layer = "2_hidden"  # or "1_hidden"
    epochs = [10, 20, 40, 60, 80, 100]

    ##########################
    # 2) Base training params
    ##########################
    base_training_params = {
        "layer_wise_lr_decay": 0.95,
        "max_grad_norm": 5.0,
        "warmup_ratio": 0.1,
        "cosine_decay": True,
        "drop_lr_on_plateau": False
    }

    # Data
    data_module = IMDBDataModule(classification_word)

    # Prepare results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/imdb_sentiment_classifier_naive_baseline"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{timestamp}.csv")

    # 3) Update CSV fieldnames
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = [
            'model', 'epochs', 'final_loss', 'test_loss',
            'test_accuracy', 'test_precision', 'test_recall', 'test_f1'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_name in models:
            # Get model-specific hyperparams
            model_params = get_model_specific_params(model_name)

            # Merge with base training params
            training_params = {
                **base_training_params,
                "layer_wise_lr_decay": model_params["layer_wise_lr_decay"]
            }

            # Construct optimizer params
            optimizer_name = "adamw"
            optimizer_params = {
                "lr": model_params["lr"],
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": model_params["weight_decay"]
            }

            for num_epochs in epochs:
                print(f"Running experiment: {model_name}, epochs={num_epochs}")

                # Create model
                model = model_variations[model_name][hidden_layer](classification_word).to(device)

                # Create trainer
                trainer = Trainer(
                    model=model,
                    data_module=data_module,
                    optimizer_name=optimizer_name,
                    optimizer_params=optimizer_params,
                    batch_size=model_params["batch_size"],
                    num_epochs=num_epochs,
                    device=device,
                    dataset_name="imdb_naive_baseline",
                    **training_params
                )

                # Train
                epoch_losses = trainer.train_on_full_dataset(num_epochs)
                final_loss = epoch_losses[-1] if epoch_losses else None

                # Test
                test_loss, test_accuracy, test_precision, test_recall, test_f1 = trainer.test()

                # Write CSV row
                writer.writerow({
                    'model': model_name,
                    'epochs': num_epochs,
                    'final_loss': final_loss,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1
                })
                csvfile.flush()

                # Save model
                num_hidden_layers = int(hidden_layer[0])  # quick parse for 1_hidden or 2_hidden
                save_trained_model(trainer,
                                   f"imdb_sentiment_naive_baseline_{model_name}_{num_epochs}epochs",
                                   num_hidden_layers)

                # (Optional) clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"Experiments completed. Results saved to {results_file}")


if __name__ == "__main__":
    run_imdb_sentiment_experiment()
