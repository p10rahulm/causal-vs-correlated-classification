import os
import sys
from pathlib import Path
import csv
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import glob

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from models.causal_neutral_model_variations import model_variations
from models.model_utilities import load_model, load_best_causal_neutral_model
from data_loaders.imdb_sentiment.core import IMDBDataModule
from data_loaders.common.causal_neutral import CausalNeutralDataModule
from data_loaders.common.causal_classification import CausalPhraseDataModule
from data_loaders.common.regularized import RegularizedDataModule

from trainers.trainer import Trainer
from trainers.regularized_classification_trainer import RegularizedTrainer
from optimizers.optimizer_params import optimizer_configs


def log_model_location(model_info):
    log_file = "trained_models/imdb_sentiment/trained_model_locations.csv"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fieldnames = ['model_type', 'model_name', 'epochs', 'lambda', 'model_path']
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({
            'model_type': model_info['model_type'],
            'model_name': model_info['model'],
            'epochs': model_info['epochs'],
            'lambda': model_info['lambda'],
            'model_path': model_info['model_path']
        })


def get_model_path(model_type, model_name, epochs, lambda_value=None):
    base_path = "trained_models/imdb_sentiment"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if model_type == "naive":
        return f"{base_path}/naive/{model_name}_{epochs}epochs/{timestamp}.pth"
    elif model_type == "causal_neutral":
        return f"{base_path}/causal_neutral/{model_name}_{epochs}epochs/{timestamp}.pth"
    elif model_type == "causal_classifier":
        return f"{base_path}/causal_classifier/{model_name}_{epochs}epochs/{timestamp}.pth"
    elif model_type == "regularized":
        return f"{base_path}/regularized/{model_name}_{epochs}_10epochs_lambda{lambda_value}/{timestamp}.pth"
    else:
        raise ValueError("Invalid model type")


def get_latest_model(model_type, model_name, epochs):
    base_path = f"trained_models/imdb_sentiment/{model_type}/{model_name}_{epochs}epochs"
    model_files = glob.glob(f"{base_path}/*.pth")
    if not model_files:
        return None
    return max(model_files, key=os.path.getctime)


def setup_results_file(model_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"outputs/imdb_sentiment_{model_type}"
    if model_type=='regularized':
        results_dir = f"outputs/imdb_sentiment_{model_type}_lambda"
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, f"results_{timestamp}.csv")


def train_and_evaluate(model, trainer, model_type, model_name, epochs, lambda_value=None, evaluation_flag=True):
    print(f"Training {model_type} model: {model_name}, epochs={epochs}")
    
    epoch_losses = trainer.train_on_full_dataset(epochs)
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = 0, 0, 0, 0, 0
    if evaluation_flag:
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = trainer.test()
    
    # Save model
    save_path = get_model_path(model_type, model_name, epochs, lambda_value)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
    return {
        'model_type': model_type,
        'model': model_name,
        'epochs': epochs,
        'lambda': lambda_value,
        'final_loss': epoch_losses[-1],
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'model_path': save_path
    }


def train_naive_baseline(config, imdb_data_module, model_name, epochs):
    model = load_model('naive', model_name, config['hidden_layer'], epochs, config['device'], 
                       classification_word=config['classification_word'], dataset_name=config['dataset_name'])
    if model is None:
        model = model_variations[model_name][config['hidden_layer']]("Sentiment", freeze_encoder=True).to(config['device'])
        optimizer_config = get_optimizer_config(config)
        trainer = Trainer(model, imdb_data_module, optimizer_name=config['optimizer_name'],
                        optimizer_params=optimizer_config['params'],
                        batch_size=config['batch_size'], num_epochs=epochs, device=config['device'])
        results = train_and_evaluate(model, trainer, 'naive', model_name, epochs)
        log_results(results, config['result_files']['naive'])
        log_model_location(results)
        return model, results
    return model, None


def train_causal_neutral(config, model_name, causal_neutral_data_module):
    model = load_model('causal_neutral', model_name, config['hidden_layer'], 10, config['device'], 
                       classification_word=config['classification_word'], dataset_name=config['dataset_name'])
    if model is None:
        print(f"Training new causal neutral model: {model_name}")
        model = model_variations[model_name][config['hidden_layer']]("Sentiment", freeze_encoder=True).to(config['device'])
        optimizer_config = get_optimizer_config(config)
        trainer = Trainer(model, causal_neutral_data_module, optimizer_name=config['optimizer_name'],
                        optimizer_params=optimizer_config['params'],
                        batch_size=config['batch_size'], num_epochs=10, device=config['device'])
        results = train_and_evaluate(model, trainer, 'causal_neutral', model_name, 10, evaluation_flag=False)
        log_results(results, config['result_files']['causal_neutral'])
        log_model_location(results)
        return model, results
    return model, None


def train_causal_classifier(config, causal_phrase_data_module, model_name, epochs):
    model = load_model('causal_classifier', model_name, config['hidden_layer'], epochs, config['device'], 
                       classification_word=config['classification_word'], dataset_name=config['dataset_name'])
    if model is None:
        print(f"Training new causal neutral model: {model_name}, epochs: {epochs}, \
            dataset_name={config['dataset_name']}, hidden_layer:{config['hidden_layer']}")
        model = model_variations[model_name][config['hidden_layer']]("Sentiment", freeze_encoder=False).to(config['device'])
        optimizer_config = get_optimizer_config(config)
        trainer = Trainer(model, causal_phrase_data_module, optimizer_name=config['optimizer_name'],
                        optimizer_params=optimizer_config['params'],
                        batch_size=config['batch_size'], num_epochs=epochs, device=config['device'])
        results = train_and_evaluate(model, trainer, 'causal_classifier', model_name, epochs)
        log_results(results, config['result_files']['causal_classifier'])
        log_model_location(results)
        return model, results
    return model, None

def train_regularized(config, regularized_data_module, model_name, epochs, lambda_value, naive_model):
    model = load_model('regularized', f"{model_name}_{lambda_value}", config['hidden_layer'], epochs, config['device'], 
                       classification_word=config['classification_word'], dataset_name=config['dataset_name'])
    if model is None:
        print(f"Training new regularized model: {model_name}, epochs: {epochs}, lambda: {lambda_value}, \
              dataset_name={config['dataset_name']}, hidden_layer:{config['hidden_layer']}")
        model = model_variations[model_name][config['hidden_layer']]("Sentiment", freeze_encoder=False).to(config['device'])
        model.load_state_dict(naive_model.state_dict())  # Initialize with naive model weights
        optimizer_config = get_optimizer_config(config)
        trainer = RegularizedTrainer(
            model_eta=naive_model,
            model_theta=model,
            data_module=regularized_data_module,
            optimizer_name=config['optimizer_name'],
            optimizer_params=optimizer_config['params'],
            batch_size=config['batch_size'],
            num_epochs=10,
            device=config['device'],
            lambda_reg=lambda_value
        )
        results = train_and_evaluate(model, trainer, 'regularized', model_name, epochs, lambda_value)
        log_results(results, config['result_files']['regularized'])
        log_model_location(results)
        return model, results
    return model, None

def get_optimizer_config(config):
    optimizer_config = optimizer_configs[config['optimizer_name']].copy()
    optimizer_config['params'] = optimizer_config['params'].copy()
    optimizer_config['params']['lr'] = config['learning_rate']
    return optimizer_config

def log_results(results, file_path):
    with open(file_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(results)

def main():
    config = {
        'models': ["bert", "electra_small_discriminator", "distilbert", "t5", "roberta", "albert"],
        'epochs': [5, 10],
        'batch_size': 256,
        'optimizer_name': "adamw",
        'hidden_layer': "1_hidden",
        'learning_rate': 0.0001,
        'lambda_values': [0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'result_files': {
            'naive': setup_results_file('naive'),
            'causal_neutral': setup_results_file('causal_neutral'),
            'causal_classifier': setup_results_file('causal_classifier'),
            'regularized': setup_results_file('regularized')
        },
        'classification_word': "Sentiment",
        'dataset_name': "imdb_sentiment",
        'fixed_causal_neutral_model': 'bert',
        'use_fixed_causal_neutral_model': False,
    }
    # temporarily modifying this file
    config['models'] = ['distilroberta']

    imdb_data_module = IMDBDataModule("Sentiment")
    
    for model_name in config['models']:
        for epoch in config['epochs']:
            # Model 1: Naive Baseline
            naive_model, _ = train_naive_baseline(config, imdb_data_module, model_name, epoch)
            
            # Model 2: Causal Neutral Classifier
            causal_neutral_model = None
            if config['use_fixed_causal_neutral_model']:
                try:
                    causal_neutral_model = load_best_causal_neutral_model(config)
                    print("Using fixed causal neutral model for Models 3 and 4")
                except Exception as e:
                    print(f"Failed to load fixed causal neutral model: {str(e)}\nTraining a new model.")
            
            if causal_neutral_model is None:
                input_file_path = "outputs/imdb_train_sentiment_analysis.json"
                causal_neutral_data_module = CausalNeutralDataModule(input_file_path , "Sentiment")
                causal_neutral_model, _ = train_causal_neutral(config, model_name, causal_neutral_data_module)                
                       
            # Model 3: Causal Phrase Sentiment Classifier
            causal_phrase_data_module = CausalPhraseDataModule(base_data_module=imdb_data_module, 
                                                               classification_word="Sentiment", 
                                                               text_column='text', label_column='label')
            # causal_phrase_data_module = CausalPhraseIMDBDataModule()
            causal_phrase_data_module.set_causal_neutral_model(causal_neutral_model, causal_neutral_model.tokenizer)
            train_causal_classifier(config, causal_phrase_data_module, model_name, epoch)
            
            # Model 4: Regularized Model
            regularized_data_module = RegularizedDataModule(base_data_module=imdb_data_module, classification_word="Sentiment", 
                                                            text_column='text', label_column='label', batch_size=config['batch_size'])
            if config['fixed_causal_neutral_model'] is not None:
                try:
                    fixed_causal_neutral_model = load_best_causal_neutral_model(config=config)
                    regularized_data_module.set_models(fixed_causal_neutral_model, naive_model)
                except:
                    regularized_data_module.set_models(causal_neutral_model, naive_model)
            
            
            for lambda_value in config['lambda_values']:
                train_regularized(config, regularized_data_module, model_name, epoch, lambda_value, naive_model)

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("All experiments completed.")

if __name__ == "__main__":
    main()