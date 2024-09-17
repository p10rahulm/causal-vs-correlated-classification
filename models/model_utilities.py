import os
import torch


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
