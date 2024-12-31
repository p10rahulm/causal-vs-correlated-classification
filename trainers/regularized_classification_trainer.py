import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class RegularizedTrainer:
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
        drop_lr_on_plateau: bool = True,
        classification_word = "Sentiment",
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
        self.data_module = data_module
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device if device is not None else torch.device("cpu")
        self.lambda_reg = lambda_reg
        self.classification_word = classification_word

        # Move models to device
        self.model_ref.to(self.device)
        self.model_theta.to(self.device)

        # Possibly freeze reference model parameters
        for p in self.model_ref.parameters():
            p.requires_grad = False

        # Build optimizer
        if optimizer_params is None:
            optimizer_params = {}
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

        self.optimizer = self.optimizer_class(self.model_theta.parameters(), **self.optimizer_params)

        # Optional LR scheduler
        if drop_lr_on_plateau:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3)
        else:
            self.scheduler = None

        # Loss function for classification
        self.cls_loss_fn = nn.CrossEntropyLoss()

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
        penalty = (uncond_0 - cond_0) + (uncond_1 - cond_1)

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

        train_loader = self.data_module.get_full_train_dataloader()

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

        progress_bar = tqdm(loader, desc=f"Epoch {epoch_idx+1}/{self.num_epochs}")
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
            self.optimizer.step()

            total_loss += loss.item()
            total_steps += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

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
