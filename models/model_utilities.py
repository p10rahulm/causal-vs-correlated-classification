import os
import torch


def load_trained_model(model_path, model_class):
    model = model_class
    model.load_state_dict(torch.load(model_path))
    return model

def find_model_file(directory):
    for file in os.listdir(directory):
        if file.endswith(".pth"):
            return os.path.join(directory, file)
    return None
