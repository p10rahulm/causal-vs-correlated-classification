import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from trainers.trainer import Trainer
from transformers import get_cosine_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_

from utilities.general_utilities import get_project_root

class RegularizedTrainer(Trainer):
    """
    A trainer that implements the ExpSE penalty for binary classification:
        R_{B} = sum_{x_i in {0,1}} [
                    (1/|B|)  sum_{s in B} [ P_theta(y=x_i|s) * ( P_ref(y=x_i|s) / P_ref(y=x_i|s_{z-only}) ) ]
                -  (1/|B_{x_i}|) sum_{s in B_{x_i}} [ P_theta(y=x_i|s) ]
                ]

    We add:  total_loss = cross_entropy_loss + lambda_reg * R_B

    Usage:
        trainer = RegularizedTrainer(
            model_ref=...,   # reference (frozen or semi-updated)
            model_theta=..., # policy
            data_module=..., # must yield "full_input_ids", "z_input_ids", "labels", etc.
            ...
            lambda_reg=0.1
        )
        trainer.train_on_full_dataset(num_epochs=5)   # Or trainer.train()
    """

    def __init__(
        self,
        model_ref: nn.Module,
        model_theta: nn.Module,
        data_module: Any,
        optimizer_name: str = "adamw",
        dataset_name: str = "imdb_causal_mediation",
        optimizer_params: Dict[str, Any] = None,
        batch_size: int = 32,
        num_epochs: int = 5,
        device: Any = None,
        lambda_reg: float = 0.1,
        drop_lr_on_plateau: bool = False,  # Default changed to False since we prefer cosine
        cosine_decay: bool = True,
        warmup_ratio: float = 0.1,  # Increased default warmup
        layer_wise_lr_decay: float = 0.95,  # Layer-wise learning rate decay factor
        max_grad_norm: float = 5.0,
        classification_word = "Sentiment",
        model_name = "bert",
        **kwargs
    ):
        """
        :param model_ref: Reference model P_ref. Typically loaded from a naive baseline checkpoint.
        :param model_theta: Policy model P_theta, initialized (copied) from model_ref weights.
        :param data_module: Something that yields train/val DataLoaders with
            batch = {
               "full_input_ids":      ...,
               "full_attention_mask": ...,
               "z_input_ids":         ...,
               "z_attention_mask":    ...,
               "labels":              ...
            }
        :param optimizer_name: "adamw", "adam", etc. (must exist in your optimizer_configs).
        :param dataset_name: Just a string label for saving/logging.
        :param optimizer_params: Dict of optimizer hyperparams.
        :param batch_size: Not strictly necessary if your data_module already sets that.
        :param num_epochs: Number of training epochs.
        :param device: e.g. torch.device("cuda") or "cpu".
        :param lambda_reg: Weight of the ExpSE penalty.
        :param drop_lr_on_plateau: If True, use a scheduler that reduces LR on plateau.
        :param kwargs: Extra placeholders if needed.
        """
        self.model_ref = model_ref
        self.model_theta = model_theta
        self.model = self.model_theta
        self.data_module = data_module
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device if device is not None else torch.device("cpu")
        self.lambda_reg = lambda_reg
        self.classification_word = classification_word
        self.model_name = model_name
        self.layer_wise_lr_decay = layer_wise_lr_decay
        self.max_grad_norm = max_grad_norm
        # Move models to device
        self.model_ref.to(self.device)
        self.model_theta.to(self.device)

        # Possibly freeze reference model parameters
        for p in self.model_ref.parameters():
            p.requires_grad = False

        # Set up optimizer params with defaults
        if optimizer_params is None:
            optimizer_params = {
                "lr": 2e-5,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        self.optimizer_params = optimizer_params

        # If you have a global optimizer config somewhere, retrieve that class
        if optimizer_name.lower() == "adamw":
            from torch.optim import AdamW
            self.optimizer_class = AdamW
        elif optimizer_name.lower() == "adam":
            from torch.optim import Adam
            self.optimizer_class = Adam
        else:
            raise ValueError(f"Unknown optimizer name: {optimizer_name}")

        # self.optimizer = self.optimizer_class(self.model_theta.parameters(), **self.optimizer_params)
        # Create optimizer with layer-wise decay
        self.optimizer = self._create_layer_wise_optimizer()

        # Optional LR scheduler
        if drop_lr_on_plateau:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3)
        else:
            self.scheduler = None

        # Build your data loader once to figure out how many steps (or do it later)
        train_loader = data_module.get_full_train_dataloader()
        steps_per_epoch = len(train_loader)
        total_training_steps = steps_per_epoch * self.num_epochs
        # Ensure minimum warmup steps
        warmup_steps = max(
            min(int(warmup_ratio * total_training_steps), total_training_steps // 10),
            100  # Minimum warmup steps
        )

        # Set up scheduler
        if cosine_decay:
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_training_steps
            )
        elif drop_lr_on_plateau:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min',
                factor=0.1,
                patience=3,
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Loss function for classification
        self.cls_loss_fn = nn.CrossEntropyLoss()


    ###########################################################################
    # Setup the optimizer
    ###########################################################################
    def _create_layer_wise_optimizer(self):
        """Creates optimizer with layer-wise learning rate decay and proper weight decay settings."""
        no_decay = ["bias", "LayerNorm.weight"]
        
        # Get the number of transformer layers
        num_layers = len([name for name, _ in self.model_theta.named_parameters() 
                         if "encoder.layer" in name and ".0." in name])
        
        optimizer_grouped_parameters = []
        
        # Handle encoder layers with decay
        for layer_idx in range(num_layers, -1, -1):
            layer_decay_rate = self.layer_wise_lr_decay ** (num_layers - layer_idx)
            
            # Parameters with weight decay
            layer_params_decay = {
                "params": [p for n, p in self.model_theta.named_parameters() 
                          if f"encoder.layer.{layer_idx}." in n 
                          and not any(nd in n for nd in no_decay)],
                "weight_decay": self.optimizer_params["weight_decay"],
                "lr": self.optimizer_params["lr"] * layer_decay_rate,
            }
            
            # Parameters without weight decay
            layer_params_no_decay = {
                "params": [p for n, p in self.model_theta.named_parameters() 
                          if f"encoder.layer.{layer_idx}." in n 
                          and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.optimizer_params["lr"] * layer_decay_rate,
            }
            
            optimizer_grouped_parameters.extend([layer_params_decay, layer_params_no_decay])
        
        # Handle non-encoder parameters (e.g., classifier, pooler)
        other_params_decay = {
            "params": [p for n, p in self.model_theta.named_parameters() 
                      if not any(f"encoder.layer.{i}." in n for i in range(num_layers + 1))
                      and not any(nd in n for nd in no_decay)],
            "weight_decay": self.optimizer_params["weight_decay"],
            "lr": self.optimizer_params["lr"],
        }
        
        other_params_no_decay = {
            "params": [p for n, p in self.model_theta.named_parameters() 
                      if not any(f"encoder.layer.{i}." in n for i in range(num_layers + 1))
                      and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": self.optimizer_params["lr"],
        }
        
        optimizer_grouped_parameters.extend([other_params_decay, other_params_no_decay])
        
        return self.optimizer_class(
            optimizer_grouped_parameters,
            betas=self.optimizer_params.get("betas", (0.9, 0.999)),
            eps=self.optimizer_params.get("eps", 1e-8)
        )
    
    ###########################################################################
    # 1) Compute ExpSE Penalty
    ###########################################################################
    def compute_expse_penalty(self,
                              theta_probs: torch.Tensor,
                              ref_probs_full: torch.Tensor,
                              ref_probs_z: torch.Tensor,
                              labels: torch.Tensor) -> torch.Tensor:
        """
        Compute R_{B} for a batch B (binary classification).
        Inputs:
         - theta_probs: shape (batch_size, 2). P_theta(y=0|s) and P_theta(y=1|s).
         - ref_probs_full: shape (batch_size, 2). P_ref(y=0|s), P_ref(y=1|s).
         - ref_probs_z: shape (batch_size, 2).   P_ref(y=0|s_z), P_ref(y=1|s_z).
         - labels: shape (batch_size,). 0 or 1.

        Returns a scalar penalty: R_B
        """

        # Avoid division by zero by clamping
        ref_probs_z = torch.clamp(ref_probs_z, min=1e-7, max=1 - 1e-7)

        # We gather the policy's probabilities for x=0 or x=1
        # and the ratio from reference model for x=0 or x=1
        # Then sum them up as the formula says.

        # We'll do it in two parts: unconditional term (U) and conditional term (C).

        # unconditional: sum_{i in B} [ pTheta(x=0|s_i)* (pRef(0|s_i)/pRef(0|sZ_i)) ]  for x=0,
        #                 and similarly x=1
        # We'll accumulate them, then multiply by 1/|B|.

        batch_size = theta_probs.shape[0]
        # P_theta(0|s), P_theta(1|s)
        ptheta_0 = theta_probs[:, 0]
        ptheta_1 = theta_probs[:, 1]

        pref_ratio_0 = ref_probs_full[:, 0] / ref_probs_z[:, 0]  # for x=0
        pref_ratio_1 = ref_probs_full[:, 1] / ref_probs_z[:, 1]  # for x=1

        # unconditional sums (over entire batch)
        # sum_{i in B} ptheta_0_i * pref_ratio_0_i
        sum_uncond_0 = torch.sum(ptheta_0 * pref_ratio_0)
        sum_uncond_1 = torch.sum(ptheta_1 * pref_ratio_1)

        # unconditional term = ( sum_uncond_0 + sum_uncond_1 ) / |B|
        # but note we have separate x=0,1 blocks in the original formula
        # We'll compute them separately for clarity, then sum.
        uncond_0 = sum_uncond_0 / float(batch_size)
        uncond_1 = sum_uncond_1 / float(batch_size)

        # conditional: for x=0, gather subset B_0, sum ptheta(0|s) => average
        # similarly for x=1
        # => sum_{s in B_{x=0}} ptheta(0|s) / |B_{x=0}|
        # We'll do that by boolean indexing

        # Indices for x=0 or x=1 ground truth
        mask_0 = (labels == 0)
        mask_1 = (labels == 1)

        # size of B_0
        b0_size = max(int(torch.sum(mask_0).item()), 1)  # avoid div by 0
        b1_size = max(int(torch.sum(mask_1).item()), 1)

        cond_0 = torch.sum(ptheta_0[mask_0]) / float(b0_size)
        cond_1 = torch.sum(ptheta_1[mask_1]) / float(b1_size)

        # Summation over x in {0,1} of ( uncond_x - cond_x ).
        # That is: R_B = [ uncond_0 - cond_0 ] + [ uncond_1 - cond_1 ]
        # The sign is + because the eqn has a minus between them.
        #  R_B = sum_{x in {0,1}} [uncond_x - cond_x]
        penalty = torch.abs(uncond_0 - cond_0) + torch.abs(uncond_1 - cond_1)

        return penalty

    ###########################################################################
    # 2) Training / Validation
    ###########################################################################
    def train(self, full_dataset: bool = False) -> List[Dict[str, float]]:
        """
        Standard train method across self.num_epochs.
        Returns a list of per-epoch stats:
            [{'epoch': 1, 'train_loss': x, 'val_loss': y, 'accuracy': z, ...}, ...]
        """
        if full_dataset:
            train_loader = self.data_module.get_full_train_dataloader()
        else:
            train_loader, val_loader = self.data_module.get_dataloaders()

        # If we do want a separate val loader, we can do:
        _, val_loader = self.data_module.get_dataloaders()

        # We'll store epoch stats
        history = []
        for epoch in range(self.num_epochs):
            epoch_stat = self._train_one_epoch(train_loader, epoch)
            self.model_ref.load_state_dict(self.model_theta.state_dict())
            for param in self.model_ref.parameters():
                param.requires_grad = False

            # Evaluate on val set if you want per-epoch logging:
            val_loss, acc, prec, rec, f1 = self.validate(val_loader)

            epoch_stat.update({
                "val_loss": val_loss,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1
            })
            history.append(epoch_stat)

            # Step LR scheduler if in use
            if self.scheduler:
                self.scheduler.step(val_loss)

        return history

    def train_on_full_dataset(self, num_epochs: int = None) -> List[Dict[str, float]]:
        """
        Shortcut to train on the concatenation of train+val sets for final training
        if desired. We'll skip separate val checks each epoch, or we can still do them
        if you prefer. We'll demonstrate skipping them for brevity.
        """
        if num_epochs is not None:
            self.num_epochs = num_epochs

        history = []
        for epoch in range(self.num_epochs):
            train_loader = self.data_module.get_full_train_dataloader()
            epoch_stat = self._train_one_epoch(train_loader, epoch)
            self.model_ref.load_state_dict(self.model_theta.state_dict())
            for param in self.model_ref.parameters():
                param.requires_grad = False
            # Optionally no val set here, or we can do a test on real test set
            # We'll omit for brevity.
            history.append(epoch_stat)

        return history

    def _train_one_epoch(self, loader, epoch_idx: int) -> Dict[str, float]:
        self.model_ref.eval()     # Typically keep reference model fixed
        self.model_theta.train()  # We update the policy model

        total_loss = 0.0
        total_steps = 0
        total_loss = 0.0
        running_loss = 0.0
        progress_bar = tqdm(
            loader, 
            desc=f"Epoch {epoch_idx+1}/{self.num_epochs}",
            mininterval=10.0,
            miniters=max(len(loader)//1000, 1), 
            smoothing=0,  # Disable smoothing
            leave=False
        )
        for batch in progress_bar:
            # 1) Zero out gradients
            self.optimizer.zero_grad()

            # 2) Forward pass for policy model (theta) on full text
            full_ids = batch["full_input_ids"].to(self.device)
            full_mask = batch["full_attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            policy_out = self.model_theta(full_ids, full_mask)
            # For huggingface style models, the actual logits might be in policy_out.logits
            # For a custom model, might be direct. We'll handle both:
            if hasattr(policy_out, "logits"):
                logits_theta = policy_out.logits
            else:
                logits_theta = policy_out
            # Classification loss
            cls_loss = self.cls_loss_fn(logits_theta, labels)

            # 3) Probability predictions for policy
            probs_theta = F.softmax(logits_theta, dim=1)

            # 4) Forward pass for reference model: 
            #    (a) on full text
            with torch.no_grad():
                ref_out_full = self.model_ref(full_ids, full_mask)
                if hasattr(ref_out_full, "logits"):
                    logits_ref_full = ref_out_full.logits
                else:
                    logits_ref_full = ref_out_full
                ref_probs_full = F.softmax(logits_ref_full, dim=1)

            # (b) On z-only text
            z_ids = batch["z_input_ids"].to(self.device)
            z_mask = batch["z_attention_mask"].to(self.device)
            
            with torch.no_grad():
                ref_out_z = self.model_ref(z_ids, z_mask)
                if hasattr(ref_out_z, "logits"):
                    logits_ref_z = ref_out_z.logits
                else:
                    logits_ref_z = ref_out_z
                ref_probs_z = F.softmax(logits_ref_z, dim=1)

            # 5) Compute ExpSE penalty for this batch
            expse_penalty = self.compute_expse_penalty(
                theta_probs=probs_theta,
                ref_probs_full=ref_probs_full,
                ref_probs_z=ref_probs_z,
                labels=labels
            )

            # 6) Combine losses
            loss = cls_loss + self.lambda_reg * expse_penalty

            loss.backward()
            clip_grad_norm_(self.model_theta.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

            # Step scheduler if using cosine decay
            if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step()

            total_loss += loss.item()
            total_steps += 1
            running_loss = (running_loss * (total_steps - 1) + loss.item()) / total_steps
            
            # Only update progress bar when tqdm updates (every ~5% of epoch)
            if total_steps % max(len(loader)//20, 1) == 0:
                progress_bar.set_postfix({
                    "avg_loss": f"{running_loss:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
                }, refresh=False)

        avg_loss = total_loss / max(total_steps, 1)
        return {"epoch": epoch_idx + 1, "train_loss": avg_loss}

    ###########################################################################
    # 3) Validation / Test
    ###########################################################################
    def validate(self, loader) -> Tuple[float, float, float, float, float]:
        self.model_theta.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                full_ids = batch["full_input_ids"].to(self.device)
                full_mask = batch["full_attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model_theta(full_ids, full_mask)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs
                loss = self.cls_loss_fn(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary"
        )

        return avg_loss, accuracy, precision, recall, f1

    def test(self) -> Tuple[float, float, float, float, float]:
        """
        If your data_module has a dedicated 'get_test_dataloader()',
        we can call that here.
        """
        loader = self.data_module.get_val_dataloader()  # or get_test_dataloader()
        return self.validate(loader)

    def save_model(self, path: str) -> None:
        self.model = self.model_theta
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

    def save_trained_model_with_path(self, dataset_name, num_hidden_layers, filename=None):
        # Get the project root directory
        project_root = get_project_root()

        # Create the directory structure
        save_dir = os.path.join(project_root, "trained_models", dataset_name, self.classification_word.lower())
        os.makedirs(save_dir, exist_ok=True)

        # Create the filename with current date and time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if filename is None:
            filename = f"{self.model_name}_{num_hidden_layers}hidden_{current_time}.pth"

        # Full path
        full_path = os.path.join(save_dir, filename)

        # Use the trainer's save_model method
        self.save_model(full_path)