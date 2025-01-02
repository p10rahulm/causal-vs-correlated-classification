#!/usr/bin/env python3
"""
experiments/imdb_sentiment/causal_mediation.py

Example experiment script for "Stage 2" regularized training with
causal mediation. It:
  1) Loads precomputed JSON data (with full text and z-only text).
  2) Loads a baseline model checkpoint as reference (P_ref).
  3) Copies that into a new model as policy (P_theta).
  4) Trains for 5 more epochs with a RegularizedTrainer that implements
     the ExpSE penalty or similar.
  5) Logs per-epoch validation losses to a CSV file.
"""

import os
import sys
import csv
import logging
from datetime import datetime
from pathlib import Path

# Set CUDA DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer

# Adjust these imports to match your project's structure
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from data_loaders.imdb_sentiment.causal_reg_data_loader import CausalRegDataModule
from models.causal_neutral_model_variations import model_variations
from optimizers.optimizer_params import optimizer_configs
from trainers.regularized_classification_trainer import RegularizedTrainer
from trainers.trainer import save_trained_model
from models.model_utilities import load_trained_model, find_model_file


def run_causal_mediation_experiment():
    """
    Main entry point for the causal mediation experiment on IMDB.
    Loads a set of baseline checkpoints, does 5 epochs of "ExpSE" or
    similar penalty training with a RegularizedTrainer, logs results.
    """
    # 1) Basic setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Using device: {device}")

    # Data paths: These JSONs should have "text", "neutral_phrases", "label", etc.
    train_json = "data/imdb_sentiment/train_with_causal_neutral_splits_bert.json"
    val_json   = "data/imdb_sentiment/test_with_causal_neutral_splits_bert.json"  
    # or if you prefer a separate "val" set, change accordingly

    # Create results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("outputs/imdb_sentiment_causal_mediation")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv = results_dir / f"results_{timestamp}.csv"

    # We'll test a few baseline models. Map them to the checkpoint paths you gave:
    baseline_checkpoints = {
        "albert":  "trained_models/imdb_sentiment_naive_baseline_albert_5epochs/sentiment/CausalNeutralClassifier_1hidden_2024-12-30_21-28-06.pth",
        "bert":    "trained_models/imdb_sentiment_naive_baseline_bert_5epochs/sentiment/CausalNeutralClassifier_1hidden_2024-12-30_20-22-48.pth",
        "deberta": "trained_models/imdb_sentiment_naive_baseline_deberta_5epochs/sentiment/CausalNeutralClassifier_1hidden_2024-12-31_05-23-05.pth",
        "distilbert": "trained_models/imdb_sentiment_naive_baseline_distilbert_5epochs/sentiment/CausalNeutralClassifier_1hidden_2024-12-30_20-28-40.pth",
        "electra_small_discriminator": "trained_models/imdb_sentiment_naive_baseline_electra_small_discriminator_5epochs/sentiment/CausalNeutralClassifier_1hidden_2024-12-30_20-11-26.pth",
        "modern_bert": "trained_models/imdb_sentiment_naive_baseline_modern_bert_5epochs/sentiment/CausalNeutralClassifier_1hidden_2024-12-31_00-43-29.pth",
        "roberta": "trained_models/imdb_sentiment_naive_baseline_roberta_5epochs/sentiment/CausalNeutralClassifier_1hidden_2024-12-30_20-25-53.pth",
    }

    # Which subset of models to run?
    model_list = ["electra_small_discriminator", "distilbert", "roberta", "bert", "albert", "deberta", "modern_bert"]
    model_list = ["electra_small_discriminator", "modern_bert"]
    # model_list = ["roberta", "distilbert"]
    # model_list = ["bert", "albert"]
    # model_list = ["deberta"]
    # or just list(baseline_checkpoints.keys())
    batch_size = 16
    # Lambda runs
    lambda_values = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2]
    # We'll do 5 epochs of regularized training
    # regularized_epochs = 5
    regularized_epochs_list = [5, 10]

    # 2) Prepare a CSV to log each run
    with open(results_csv, "w", newline="") as csvfile:
        fieldnames = [
            "model",
            "lambda_reg",
            "regularized_epochs",
            "epoch",
            "train_loss",
            "test_loss", 
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 3) Outer loop: for each model in baseline
        for model_name in model_list:
            ckpt_path = baseline_checkpoints[model_name]
            logging.info(f"=== Model: {model_name}, Baseline checkpoint: {ckpt_path} ===")

            # 3a) Load the baseline as "reference" model
            #     We assume 1_hidden is correct for all, but if your config differs, adjust accordingly
            ref_model = model_variations[model_name]["1_hidden"]("Sentiment", freeze_encoder=True).to(device)
            ref_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            ref_model.eval()

            # 3b) Create a "policy" model that we'll fine-tune
            policy_model = model_variations[model_name]["1_hidden"]("Sentiment", freeze_encoder=False).to(device)
            policy_model.load_state_dict(ref_model.state_dict())  # Copy baseline weights


            # 4) Prepare the data module with stratified sampling, etc.
            #    We'll pick a tokenizer. Usually match the *policy* model (or we can just pick 'bert-base-uncased' for all).
            tokenizer = policy_model.tokenizer

            data_module = CausalRegDataModule(
                train_json_path=train_json,
                val_json_path=val_json,
                tokenizer=tokenizer,
                batch_size=batch_size,
                drop_last=False,
                shuffle=True
            )

            # 5) For each lambda
            for lambda_reg in lambda_values:
                logging.info(f"--- Starting Regularized Training: {model_name}, lambda={lambda_reg} ---")
                for regularized_epochs in regularized_epochs_list:
                    logging.info(f"==== Starting runs with {regularized_epochs} epochs of regularized training ====")

                    # 5a) Build the trainer
                    #     This depends on your RegularizedTrainer signature.
                    #     We'll pass in the "ref_model", "policy_model", data_module, etc.

                    trainer = RegularizedTrainer(
                        model_ref=ref_model,
                        model_theta=policy_model,
                        data_module=data_module,
                        # standard Trainer params:
                        optimizer_name="adamw",
                        dataset_name=f"imdb_causal_mediation_{model_name}",
                        optimizer_params={
                            "lr": 5e-5,  # or 1e-4, etc.
                            "betas": (0.9, 0.999),
                            "eps": 1e-8,
                            "weight_decay": 0.01
                        },
                        batch_size=batch_size,
                        num_epochs=regularized_epochs,
                        device=device,
                        # your custom param:
                        lambda_reg=lambda_reg,
                        classification_word = "Sentiment",
                        model_name = model_name
                    )

                    # 5b) Train for the specified number of epochs
                    #     If your trainer has a loop that returns the per-epoch train/val stats, great.
                    #     We'll store them in 'epoch_records'.
                    epoch_records = trainer.train_on_full_dataset(num_epochs=regularized_epochs)

                    # 5c) Each entry in 'epoch_records' might have something like:
                    #    {"epoch":..., "train_loss":..., "val_loss":..., "val_acc":..., etc.}
                    #    So we can log them to CSV.
                    for record in epoch_records:
                        writer.writerow({
                            "model": model_name,
                            "lambda_reg": lambda_reg,
                            "regularized_epochs": regularized_epochs,
                            "epoch": record.get("epoch", -1),
                            "train_loss": record.get("train_loss", None),
                            "test_loss": record.get("val_loss", None),
                            "test_accuracy": record.get("accuracy", None),
                            "test_precision": record.get("precision", None),
                            "test_recall": record.get("recall", None),
                            "test_f1": record.get("f1", None),
                        })
                    csvfile.flush()

                    # 5d) Now do final test
                    test_loss, test_acc, test_prec, test_rec, test_f1 = trainer.test()
                    logging.info(
                        f"Final test: {model_name}, λ={lambda_reg}, "
                        f"epochs={regularized_epochs}, loss={test_loss:.4f}, acc={test_acc:.4f}"
                    )
                    

                    writer.writerow({
                        "model": model_name,
                        "lambda_reg": lambda_reg,
                        "regularized_epochs": regularized_epochs,
                        "epoch": "final_test",
                        "train_loss": None,
                        "test_loss": test_loss,
                        "test_accuracy": test_acc,
                        "test_precision": test_prec,
                        "test_recall": test_rec,
                        "test_f1": test_f1
                    })
                    csvfile.flush()

                    # 5e) Optionally, save the final policy model after these 5 epochs
                    save_trained_model(
                        trainer,
                        dataset_name=f"imdb_causal_mediation_{model_name}_lambda{lambda_reg}_{regularized_epochs}epochs",
                        num_hidden_layers=1
                    )

                    logging.info(f"--- Done: {model_name}, lambda={lambda_reg} ---")

    logging.info(f"All experiments done! Results in {results_csv}")


if __name__ == "__main__":
    run_causal_mediation_experiment()
