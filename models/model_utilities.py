import os
import torch
from datetime import datetime
from pathlib import Path
import glob
from models.causal_neutral_model_variations import model_variations


def load_trained_model(model_path, model_class):
    model = model_class
    model.load_state_dict(torch.load(model_path))
    return model

def find_model_file(directory):
    latest_file = None
    latest_time = 0
    
    for file in os.listdir(directory):
        if file.endswith(".pth"):
            file_path = os.path.join(directory, file)
            file_mtime = os.path.getmtime(file_path)
            if file_mtime > latest_time:
                latest_time = file_mtime
                latest_file = file_path

    return latest_file


def get_latest_model_path(model_type, model_name, epochs, dataset_name="imdb_sentiment"):
    base_path = f"trained_models/{dataset_name}/{model_type}/{model_name}_{epochs}epochs"
    model_files = glob.glob(f"{base_path}/*.pth")
    return max(model_files, key=os.path.getctime) if model_files else None


def load_model(model_type, model_name, hidden_layer, epochs, device, classification_word="Sentiment", dataset_name="imdb_sentiment"):
    model_path = get_latest_model_path(model_type, model_name, epochs, dataset_name=dataset_name)
    if model_path:
        print(f"Loading saved {model_type} model from {model_path}")
        model = model_variations[model_name][hidden_layer](classification_word, freeze_encoder=True).to(device)
        model.load_state_dict(torch.load(model_path))
        return model
    return None

def save_model(model, model_type, model_name, epochs, dataset_name="imdb_sentiment"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"trained_models/{dataset_name}/{model_type}/{model_name}_{epochs}epochs"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/model_{timestamp}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved {model_type} model to {save_path}")