import os
import datetime
import logging
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import math
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from trainers.trainer import Trainer
from transformers import get_cosine_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    CyclicLR
)


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
    
    Advanced Trainer includes:
      - Multiple LR schedule options (cosine warm restarts, one-cycle, cyclic)
      - A flexible lambda schedule (piecewise or smooth)
      - Optional test every N epochs
      - 2 hidden layers in your classification head (controlled by how you instantiate the model).
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
        num_epochs: int = 100,
        device: Any = None,       
        lambda_schedule_mode: str = "piecewise",  # "piecewise", "linear", "exponential", ...
        lambda_start: float = 1.0,
        lambda_end: float = 0.005,
        lr_schedule: str = "cosine_warm_restarts",
        test_interval: int = 20,  # how often to run test() in epochs
        warmup_ratio: float = 0.1,  # For OneCycle / partial warmups
        cycle_length_epochs: int = 10,  # For CyclicLR
        layer_wise_lr_decay: float = 0.95,
        max_grad_norm: float = 5.0,
        classification_word="Sentiment",
        model_name="albert",
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
        :param kwargs: Extra placeholders if needed.
        """
        super().__init__()
        self.model_ref = model_ref
        self.model_theta = model_theta
        self.model = self.model_theta # for consistency
        self.data_module = data_module
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device if device is not None else torch.device("cpu")
        
        self.classification_word = classification_word
        self.model_name = model_name
        self.test_interval = test_interval
        
        # Move models to device
        self.model_ref.to(self.device)
        self.model_theta.to(self.device)
        # freeze reference model
        self.model_ref.to(self.device)
        for p in self.model_ref.parameters():
            p.requires_grad = False

        # Default optimizer_params
        if optimizer_params is None:
            optimizer_params = {
                "lr": 5e-5,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        self.optimizer_params = optimizer_params
        # --- Set up the optimizer class ---
        if optimizer_name.lower() == "adamw":
            from torch.optim import AdamW
            self.optimizer_class = AdamW
        elif optimizer_name.lower() == "adam":
            from torch.optim import Adam
            self.optimizer_class = Adam
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        
        # Create optimizer with layer-wise LR decay
        self.layer_wise_lr_decay = layer_wise_lr_decay
        self.max_grad_norm = max_grad_norm
        self.optimizer = self._create_layer_wise_optimizer()

        # Build your data loader once to figure out how many steps (or do it later)
        train_loader = data_module.get_full_train_dataloader()
        self.steps_per_epoch = len(train_loader)
        self.total_steps = self.steps_per_epoch * self.num_epochs
        
        # --- LR Scheduler ---
        self.lr_schedule = lr_schedule
        self.warmup_ratio = warmup_ratio
        self.cycle_length_epochs = cycle_length_epochs
        self._setup_scheduler()

        # --- Lambda schedule setup ---
        valid_lambda_modes = ["piecewise", "exponential", "linear"]
        if lambda_schedule_mode not in valid_lambda_modes:
            raise ValueError(f"lambda_schedule_mode must be one of {valid_lambda_modes}")
        
        self.lambda_schedule_mode = lambda_schedule_mode
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self._lambda_piecewise_values = [
            1.0, 0.75, 0.5, 0.25, 0.1,
            0.075, 0.05, 0.025, 0.01, 0.005
        ]  # Each for 10 epochs => total 100
        # If you want exactly your chunked approach, each block is 10 epochs.

        # We'll store the current lambda each step.
        self.current_lambda = lambda_start

        
        # Loss function for classification
        self.cls_loss_fn = nn.CrossEntropyLoss()


    ###########################################################################
    # Setup the optimizer
    ###########################################################################
    
    def _get_current_lambda(self, epoch_idx: int, global_step: int) -> float:
        """
        Returns the penalty weight (lambda) given the chosen schedule.
        We'll consider the entire training as 100 epochs total, so
        if self.num_epochs=100, each epoch is 1 chunk in piecewise, or
        we do a smooth approach for "linear" / "exponential".
        """
        if self.lambda_schedule_mode == "piecewise":
            # We assume 100 epochs total, each chunk of 10 => next piece.
            chunk_size = max(self.num_epochs // 10, 1)
            chunk_idx = epoch_idx // chunk_size
            if chunk_idx >= len(self._lambda_piecewise_values):
                chunk_idx = len(self._lambda_piecewise_values) - 1
            return self._lambda_piecewise_values[chunk_idx]

        elif self.lambda_schedule_mode == "linear":
            # linear from self.lambda_start -> self.lambda_end across total_steps
            progress = float(global_step) / float(self.total_steps)
            lam = self.lambda_start + (self.lambda_end - self.lambda_start) * progress
            return lam

        elif self.lambda_schedule_mode == "exponential":
            # exponential from start->end
            # We solve for alpha so that: end = start * exp(-alpha * total_steps)
            # alpha = -ln(end/start) / total_steps
            if self.lambda_start <= 0 or self.lambda_end <= 0:
                raise ValueError("For exponential decay, need positive start/end!")
            alpha = -math.log(self.lambda_end / self.lambda_start) / float(self.total_steps)
            lam = self.lambda_start * math.exp(-alpha * global_step)
            return lam

        else:
            # fallback: just return a constant
            return self.lambda_start
    
    def _setup_scheduler(self):
        # We'll define a convenience routine that wraps the
        # Cosine/OneCycle/Cyclic approaches.

        if self.lr_schedule in ["cosine_warm_restarts", "one_cycle", 
                                "cyclic_triangular", "cyclic_triangular2"]:

            # For all approaches, define total_steps
            total_steps = self.steps_per_epoch * self.num_epochs
            base_lr = self.optimizer_params["lr"]

            if self.lr_schedule == "cosine_warm_restarts":
                # We'll do something like T_0 = (cycle_length_epochs * steps_per_epoch)
                # Then each T_0 epochs, we do a warm restart.
                # The warmup can be done by just letting the model start at a lower lr or
                # a short manual warmup step. For a quick approach, we let the scheduler
                # handle everything but you can do a manual warmup if you want.
                
                self.scheduler = CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=self.cycle_length_epochs * self.steps_per_epoch,
                    T_mult=1,
                    eta_min=5e-7  # end LR
                )

            elif self.lr_schedule == "one_cycle":
                # OneCycleLR allows specifying max_lr, total_steps, etc.
                # We'll do 10% warmup via pct_start=0.1
                self.scheduler = OneCycleLR(
                    self.optimizer,
                    max_lr=base_lr,
                    total_steps=total_steps,
                    pct_start=self.warmup_ratio,  # e.g. 0.1 => 10% of cycle is warmup
                    anneal_strategy='cos',
                    final_div_factor=base_lr / (5e-7)  # or something so we end near 5e-7
                )

            elif self.lr_schedule in ["cyclic_triangular", "cyclic_triangular2"]:
                if self.optimizer_params["lr"] < 1e-10:
                    raise ValueError("base_lr must be >= 1e-10 for cyclic schedules")
                # cycle_length_epochs = 10 means we want 10 epochs per cycle
                steps_up = self.cycle_length_epochs * self.steps_per_epoch // 2
                mode = "triangular2" if self.lr_schedule == "cyclic_triangular2" else "triangular"
                
                self.scheduler = CyclicLR(
                    self.optimizer,
                    base_lr=5e-7,
                    max_lr=base_lr,
                    step_size_up=steps_up,
                    mode=mode,
                    cycle_momentum=False
                )

        else:
            # If none, you can also do a fallback or no scheduler
            self.scheduler = None      
    

    def _create_layer_wise_optimizer(self):
        """ Creates an optimizer that applies layer-wise LR decay. """
        no_decay = ["bias", "LayerNorm.weight"]
        
        # Attempt to detect how many "encoder.layer" blocks exist
        # (works for BERT/RoBERTa/etc. but for ALBERT might differ).
        # If needed, adapt to your own naming scheme.
        # We'll do something generic: find the largest X in 'encoder.layer.X.'.
        layer_indices = []
        for n, _ in self.model_theta.named_parameters():
            if "encoder.layer." in n:
                # parse out the layer number
                try:
                    layer_num = int(n.split("encoder.layer.")[1].split(".")[0])
                    layer_indices.append(layer_num)
                except:
                    pass
        num_layers = max(layer_indices) + 1 if layer_indices else 0

        grouped_params = []
        base_lr = self.optimizer_params["lr"]

        # from last layer to first layer
        for layer_idx in range(num_layers - 1, -1, -1):
            decay_rate = self.layer_wise_lr_decay ** (num_layers - 1 - layer_idx)
            layer_decay = base_lr * decay_rate

            # decay
            grouped_params.append({
                "params": [
                    p for n, p in self.model_theta.named_parameters()
                    if f"encoder.layer.{layer_idx}." in n and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.optimizer_params["weight_decay"],
                "lr": layer_decay
            })
            # no decay
            grouped_params.append({
                "params": [
                    p for n, p in self.model_theta.named_parameters()
                    if f"encoder.layer.{layer_idx}." in n and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
                "lr": layer_decay
            })

        # Finally, handle embeddings & pooler & classification heads if not in "encoder.layer"
        # with base lr or no decay
        grouped_params.append({
            "params": [
                p for n, p in self.model_theta.named_parameters()
                if "encoder.layer." not in n and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": self.optimizer_params["weight_decay"],
            "lr": base_lr
        })
        grouped_params.append({
            "params": [
                p for n, p in self.model_theta.named_parameters()
                if "encoder.layer." not in n and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": base_lr
        })

        optimizer = self.optimizer_class(grouped_params,
                                         betas=self.optimizer_params.get("betas", (0.9, 0.999)),
                                         eps=self.optimizer_params.get("eps", 1e-8))
        return optimizer

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
        # We'll store epoch stats
        history = []
        for epoch in range(self.num_epochs):
            if full_dataset:
                train_loader = self.data_module.get_full_train_dataloader()
            else:
                train_loader, val_loader = self.data_module.get_dataloaders()
            epoch_stat = self._train_one_epoch(train_loader, epoch)
            self.model_ref.load_state_dict(self.model_theta.state_dict())
            for param in self.model_ref.parameters():
                param.requires_grad = False
            
            metrics = {"epoch": epoch_stat['epoch'], 
                       "train_loss": epoch_stat['train_loss'], 
                       "lambda": epoch_stat['lambda']}
            if (epoch + 1) % self.test_interval == 0:
                # run test
                logging.info(f"Running validation at epoch {epoch + 1}")
                val_loss, val_acc, val_prec, val_rec, val_f1 = self.test()
                logging.info(f"Val metrics: loss={val_loss:.4f}, acc={val_acc:.4f}, f1={val_f1:.4f}")

                metrics.update({
                    "test_loss": val_loss,
                    "accuracy": val_acc,
                    "precision": val_prec,
                    "recall": val_rec,
                    "f1": val_f1
                })
            history.append(metrics)

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
            
            metrics = {"epoch": epoch_stat['epoch'], 
                       "train_loss": epoch_stat['train_loss'], 
                       "lambda": epoch_stat['lambda']}
            if (epoch + 1) % self.test_interval == 0:
                # run test
                val_loss, val_acc, val_prec, val_rec, val_f1 = self.test()
                metrics.update({
                    "test_loss": val_loss,
                    "accuracy": val_acc,
                    "precision": val_prec,
                    "recall": val_rec,
                    "f1": val_f1
                })
            history.append(metrics)

        return history

    def _train_one_epoch(self, loader, epoch_idx: int) -> Dict[str, float]:
        """
        Train the policy model for one epoch. Return as many metrics as we can:
          {
            "epoch": int,
            "train_loss": float
          }
        """
        self.model_ref.eval()     # Keep reference model fixed
        self.model_theta.train()  # We update the policy model

        total_loss = 0.0
        total_steps = 0
        running_loss = 0.0

        # For training metrics
        all_preds = []
        all_labels_list = []


        progress_bar = tqdm(
            loader, 
            desc=f"Epoch {epoch_idx+1}/{self.num_epochs}",
            mininterval=10.0,
            miniters=max(len(loader)//1000, 1),
            smoothing=0,  # Disable smoothing
            leave=False
        )

        for batch_idx, batch in enumerate(progress_bar):
            # 0) Possibly compute a "global_step" if you have a lambda schedule
            global_step = epoch_idx * len(loader) + batch_idx
            current_lambda = self._get_current_lambda(epoch_idx=epoch_idx,global_step=global_step)

            # 1) Zero out grads
            self.optimizer.zero_grad()

            # 2) Forward pass (policy model)
            full_ids = batch["full_input_ids"].to(self.device)
            full_mask = batch["full_attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            policy_out = self.model_theta(full_ids, full_mask)
            logits_theta = policy_out.logits if hasattr(policy_out, "logits") else policy_out

            # Classification loss
            cls_loss = self.cls_loss_fn(logits_theta, labels)

            # 3) Probability predictions for policy
            probs_theta = F.softmax(logits_theta, dim=1)

            # 4) Reference forward passes
            with torch.no_grad():
                ref_out_full = self.model_ref(full_ids, full_mask)
                logits_ref_full = ref_out_full.logits if hasattr(ref_out_full, "logits") else ref_out_full
                ref_probs_full = F.softmax(logits_ref_full, dim=1)

                z_ids = batch["z_input_ids"].to(self.device)
                z_mask = batch["z_attention_mask"].to(self.device)
                ref_out_z = self.model_ref(z_ids, z_mask)
                logits_ref_z = ref_out_z.logits if hasattr(ref_out_z, "logits") else ref_out_z
                ref_probs_z = F.softmax(logits_ref_z, dim=1)

            # 5) Compute ExpSE penalty (user-defined function)
            expse_penalty = self.compute_expse_penalty(
                theta_probs=probs_theta,
                ref_probs_full=ref_probs_full,
                ref_probs_z=ref_probs_z,
                labels=labels
            )

            # 6) Combine losses
            loss = cls_loss + current_lambda * expse_penalty
            loss.backward()
            clip_grad_norm_(self.model_theta.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()

            # If scheduler is not ReduceLROnPlateau, step each iteration
            if (self.scheduler is not None) and (not isinstance(self.scheduler, ReduceLROnPlateau)):
                self.scheduler.step()

            total_loss += loss.item()
            total_steps += 1
            running_loss = (running_loss * (total_steps - 1) + loss.item()) / total_steps

            # For training metrics, collect predictions
            preds = torch.argmax(logits_theta, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels_list.extend(labels.detach().cpu().numpy())

            # Update progress bar
            if total_steps % max(len(loader)//20, 1) == 0:
                progress_bar.set_postfix({
                    "avg_loss": f"{running_loss:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    "lambda": f"{current_lambda:.3f}"
                }, refresh=False)

        # Compute final average loss
        avg_loss = total_loss / max(total_steps, 1)

        return {"epoch": epoch_idx + 1, "train_loss": avg_loss, 'lambda': current_lambda}

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