#!/usr/bin/env python3
"""
experiments/imdb_sentiment/causal_mediation_gridsearch.py

Loop over:
  - LR schedules: 
      ["cosine_warm_restarts", "one_cycle", "cyclic_triangular", "cyclic_triangular2"]
  - Lambda schedule modes: ["piecewise", "exponential", "linear"]
  - Base LRs: [5e-5, 1e-4, 2.5e-4]
  
Trains for 100 epochs, testing every 20 epochs.
Outputs results to a single CSV or multiple CSVs.
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

# Insert project root if needed
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from data_loaders.imdb_sentiment.causal_reg_data_loader import CausalRegDataModule
from models.causal_neutral_model_variations import model_variations
from trainers.advanced_regularized_trainer import RegularizedTrainer 
from trainers.trainer import save_trained_model
from models.model_utilities import find_model_file, load_trained_model

def run_causal_mediation_gridsearch():
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data paths
    train_json = "data/imdb_sentiment/train_with_causal_neutral_splits_bert_40.json"
    val_json   = "data/imdb_sentiment/test_with_causal_neutral_splits_bert_40.json"

    # Results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("outputs/imdb_causal_mediation")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv = results_dir / f"results_{timestamp}.csv"

    # We'll do only e.g. "albert" or whichever model(s) you prefer to test
    models = [
        "albert",
        "bert",
        "distilbert",
        "roberta",
        "electra_small_discriminator",
        "modern_bert",
        "deberta",
    ]

    # Dictionary mapping each model to its checkpoint
    model_checkpoint_paths = {
        "albert": "trained_models/imdb_sentiment_naive_baseline_albert_20epochs/sentiment/CausalNeutralClassifier_2hidden_2025-01-12_22-41-21.pth",
        "bert": "trained_models/imdb_sentiment_naive_baseline_bert_20epochs/sentiment/CausalNeutralClassifier_2hidden_2025-01-12_23-49-29.pth",
        "deberta": "trained_models/imdb_sentiment_naive_baseline_deberta_20epochs/sentiment/CausalNeutralClassifier_2hidden_2025-01-14_05-56-12.pth",
        "distilbert": "trained_models/imdb_sentiment_naive_baseline_distilbert_20epochs/sentiment/CausalNeutralClassifier_2hidden_2025-01-12_22-00-29.pth",
        "electra_small_discriminator": "trained_models/imdb_sentiment_naive_baseline_electra_small_discriminator_20epochs/sentiment/CausalNeutralClassifier_2hidden_2025-01-12_21-44-37.pth",
        "modern_bert": "trained_models/imdb_sentiment_naive_baseline_modern_bert_20epochs/sentiment/CausalNeutralClassifier_2hidden_2025-01-13_14-15-31.pth",
        "roberta": "trained_models/imdb_sentiment_naive_baseline_roberta_20epochs/sentiment/CausalNeutralClassifier_2hidden_2025-01-13_00-00-15.pth",
    }
    
    # 10 epochs, test every 2 epochs
    num_epochs = 20
    test_interval = 4

    # Define the lists for looping
    lr_schedules = ["cyclic_triangular"]
    lambda_schedule_modes = ["piecewise"]
    learning_rates = [5e-4]


    
    # We'll store results from all runs in one CSV
    with open(results_csv, "w", newline="") as csvfile:
        fieldnames = [
            "model_name",
            "lr_schedule",
            "lambda_schedule_mode",
            "base_lr",
            "epoch",
            "train_loss",
            "lambda_value",
            "test_loss",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Outer loops
        for model_name in models:
            logging.info(f"Starting grid search for model: {model_name}")
            
            baseline_ckpt_path = model_checkpoint_paths[model_name]
            # Load reference & policy from baseline
            # For "2 hidden layers" in the classifier, ensure you reference the correct variation
            ref_model = model_variations[model_name]["2_hidden"]("Sentiment", freeze_encoder=True).to(device)
            ref_model.load_state_dict(torch.load(baseline_ckpt_path, map_location=device))


            ref_model.eval()
            
            for lr_schedule_choice in lr_schedules:
                for lambda_schedule_mode in lambda_schedule_modes:
                    for base_lr in learning_rates:
                        # Make a fresh copy for the policy model each time
                        policy_model = model_variations[model_name]["2_hidden"]("Sentiment", freeze_encoder=False).to(device)
                        policy_model.load_state_dict(ref_model.state_dict())  # copy baseline weights

                        # Data module (tokenizer from the policy model, or you can pick "albert-base-v2" etc.)
                        tokenizer = policy_model.tokenizer
                        data_module = CausalRegDataModule(
                            train_json_path=train_json,
                            val_json_path=val_json,
                            tokenizer=tokenizer,
                            batch_size=16,
                            drop_last=False,
                            shuffle=True
                        )

                        # Create the trainer
                        trainer = RegularizedTrainer(
                            model_ref=ref_model,
                            model_theta=policy_model,
                            data_module=data_module,
                            optimizer_name="adamw",
                            optimizer_params={
                                "lr": base_lr,
                                "betas": (0.9, 0.999),
                                "eps": 1e-8,
                                "weight_decay": 0.01
                            },
                            num_epochs=num_epochs,
                            batch_size=16,
                            device=device,
                            lr_schedule=lr_schedule_choice,
                            lambda_schedule_mode=lambda_schedule_mode,
                            lambda_start=0.5,
                            lambda_end=0.005,
                            test_interval=test_interval,
                            cycle_length_epochs=num_epochs//10,  # if you want 10-epoch cycles for restarts
                            classification_word="Sentiment",
                            model_name=model_name
                        )

                        # Train (optionally on full dataset or partial)
                        # We'll do "train" so we have separate train & val sets and test at intervals
                        epoch_records = trainer.train(full_dataset=False)

                        # Save each epoch's metrics to CSV
                        for record in epoch_records:
                            row_dict = {
                                "model_name": model_name,
                                "lr_schedule": lr_schedule_choice,
                                "lambda_schedule_mode": lambda_schedule_mode,
                                "base_lr": base_lr,
                                "epoch": record["epoch"],
                                "train_loss": record["train_loss"],
                                "lambda_value": record.get("lambda", None),
                                "test_loss": record.get("test_loss", None),
                                "accuracy": record.get("accuracy", None),
                                "precision": record.get("precision", None),
                                "recall": record.get("recall", None),
                                "f1": record.get("f1", None)
                            }
                            writer.writerow(row_dict)
                            csvfile.flush()

                        # Optionally do final test, log final row
                        final_test_loss, final_test_acc, final_test_prec, final_test_rec, final_test_f1 = trainer.test()
                        writer.writerow({
                            "model_name": model_name,
                            "lr_schedule": lr_schedule_choice,
                            "lambda_schedule_mode": lambda_schedule_mode,
                            "base_lr": base_lr,
                            "epoch": "causal_mediation",
                            "train_loss": None,
                            "lambda_value": None,
                            "test_loss": final_test_loss,
                            "accuracy": final_test_acc,
                            "precision": final_test_prec,
                            "recall": final_test_rec,
                            "f1": final_test_f1
                        })
                        csvfile.flush()

                        # If you want to save each final model:
                        trainer.save_trained_model_with_path(
                            dataset_name=f"imdb_causal_mediation_{model_name}_{lr_schedule_choice}_{lambda_schedule_mode}_lr{base_lr}",
                            num_hidden_layers=2
                        )

    logging.info(f"All runs complete. Consolidated results in {results_csv}")

if __name__ == "__main__":
    run_causal_mediation_gridsearch()
