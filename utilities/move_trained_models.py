import os
import sys
from pathlib import Path
import shutil

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

# Define the models, epochs, and lambda values
models = ["roberta", "albert", "distilbert", "bert", "electra_small_discriminator", "t5"]
epochs = [5, 10]
lambda_values = [0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]

# Base directories
src_base = "trained_models"
dest_base_regularized = os.path.join("trained_models", "imdb_sentiment", "regularized")
dest_base_causal_neutral = os.path.join("trained_models", "imdb_sentiment", "causal_neutral")
dest_base_causal_classifier = os.path.join("trained_models", "imdb_sentiment", "causal_classifier")


def move_files(src_dir, dest_dir):
    # Check if the source directory exists
    if os.path.exists(src_dir):
        # Create the destination directory
        os.makedirs(dest_dir, exist_ok=True)
        
        # Move all files from source to destination
        for file in os.listdir(src_dir):
            src_file = os.path.join(src_dir, file)
            dest_file = os.path.join(dest_dir, file)
            shutil.move(src_file, dest_file)
        
        print(f"Moved files from {src_dir} to {dest_dir}")
    else:
        print(f"Source directory not found: {src_dir}")


def move_regularized_files():
    # Ensure the destination base directory exists
    os.makedirs(dest_base_regularized, exist_ok=True)

    # Iterate through all combinations
    for model in models:
        for epoch in epochs:
            for lambda_value in lambda_values:
                # Construct the source directory path
                src_dir = os.path.join(src_base, f"imdb_regularized_{model}_{epoch}_10epochs_lambda{lambda_value}", "sentiment")
                
                # Construct the destination directory path
                dest_dir = os.path.join(dest_base_regularized, f"{model}_{epoch}_10epochs_lambda{lambda_value}")
                
                move_files(src_dir, dest_dir)


def move_causal_neutral_files():
    # Ensure the destination base directory exists
    os.makedirs(dest_base_causal_neutral, exist_ok=True)

    # Iterate through all combinations
    for model in models:
        for epoch in epochs:
            # Construct the source directory path
            src_dir = os.path.join(src_base, f"imdb_sentiment_wz_{model}_{epoch}epochs", "sentiment")
            
            # Construct the destination directory path
            dest_dir = os.path.join(dest_base_causal_neutral, f"{model}_{epoch}epochs")
            
            move_files(src_dir, dest_dir)


def move_causal_classifier_files():
    # Ensure the destination base directory exists
    os.makedirs(dest_base_causal_classifier, exist_ok=True)

    # Iterate through all combinations
    for model in models:
        for epoch in epochs:
            # Construct the source directory path
            src_dir = os.path.join(src_base, f"imdb_causal_phrases_{model}_{epoch}epochs", "sentiment")
            
            # Construct the destination directory path
            dest_dir = os.path.join(dest_base_causal_classifier, f"{model}_{epoch}epochs")
            
            move_files(src_dir, dest_dir)


if __name__ == "__main__":
    print("Starting file moving process...")
    print("\nMoving regularized files...")
    move_regularized_files()
    print("\nMoving causal neutral files...")
    move_causal_neutral_files()
    print("\nMoving causal classifier files...")
    move_causal_classifier_files()
    print("\nFile moving process completed.")