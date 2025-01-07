import os
from pathlib import Path
import sys

# Try to find the project root
current_file = Path(__file__).resolve()
project_root = current_file.parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from optimizers.optimizer_params import get_optimizer_config
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utilities.general_utilities import get_project_root
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any, Tuple, List, Union
import logging
from data_loaders.common.base_data_module import BaseDataModule


import torch
from torch.optim import AdamW, Adam
from transformers import get_cosine_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_

class Trainer:
    def __init__(
        self,
        model,
        data_module,
        optimizer_name='adam',
        dataset_name='imdb',
        optimizer_params=None,
        batch_size=32,
        num_epochs=3,
        device=None,
        patience=3,
        layer_wise_lr_decay=0.95,  # New parameter
        max_grad_norm=5.0,         # New parameter
        warmup_ratio=0.1,          # New parameter
        cosine_decay=True,         # New parameter
        drop_lr_on_plateau=False   # Modified default
    ):
        # Existing initialization code remains the same
        self.model = model
        self.data_module = data_module
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.patience = patience
        
        # Validation checks
        if not hasattr(model, 'classification_word'):
            raise AttributeError("Model must have 'classification_word' attribute")
        
        if not hasattr(model, 'num_hidden_layers'):
            raise AttributeError("Model must have 'num_hidden_layers' attribute")
            
        if not hasattr(data_module, 'get_full_train_dataloader'):
            raise AttributeError("data_module must have 'get_full_train_dataloader' method")
        
        self.model_name = type(model).__name__
        self.classification_word = model.classification_word
        self.dataset_name = dataset_name
        self.num_hidden_layers = self.model.num_hidden_layers

        # New parameters
        self.layer_wise_lr_decay = layer_wise_lr_decay
        self.max_grad_norm = max_grad_norm
        
        # Set up optimizer params with better defaults
        if optimizer_params is None:
            optimizer_params = {
                "lr": 2e-5,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        self.optimizer_params = optimizer_params
        
        # Set up optimizer with layer-wise decay
        if optimizer_name.lower() == "adamw":
            self.optimizer_class = AdamW
        elif optimizer_name.lower() == "adam":
            self.optimizer_class = Adam
        else:
            raise ValueError(f"Unknown optimizer name: {optimizer_name}")
            
        self.optimizer = self._create_layer_wise_optimizer()

        # Set up data loaders
        try:
            # Try without tokenizer first
            self.train_loader = data_module.get_full_train_dataloader(batch_size=self.batch_size)
        except TypeError:
            # Fall back to including tokenizer if needed
            self.train_loader = data_module.get_full_train_dataloader(
                self.model.tokenizer, 
                batch_size=self.batch_size
            )
        

        # Set up scheduler with warmup and cosine decay

        steps_per_epoch = len(self.train_loader)
        total_training_steps = steps_per_epoch * self.num_epochs
        warmup_steps = max(
            min(int(warmup_ratio * total_training_steps), total_training_steps // 10),
            100  # Minimum warmup steps
        )
        
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
                patience=patience,
                verbose=True
            )
        else:
            self.scheduler = None
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_loader = None

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


    def _create_layer_wise_optimizer(self):
        """Creates optimizer with layer-wise learning rate decay and proper weight decay settings."""
        no_decay = ["bias", "LayerNorm.weight"]
        
        # Get the number of transformer layers
        num_layers = len([name for name, _ in self.model.named_parameters() 
                         if "encoder.layer" in name and ".0." in name])
        
        optimizer_grouped_parameters = []
        
        # Handle encoder layers with decay
        for layer_idx in range(num_layers, -1, -1):
            layer_decay_rate = self.layer_wise_lr_decay ** (num_layers - layer_idx)
            
            # Parameters with weight decay
            layer_params_decay = {
                "params": [p for n, p in self.model.named_parameters() 
                          if f"encoder.layer.{layer_idx}." in n 
                          and not any(nd in n for nd in no_decay)],
                "weight_decay": self.optimizer_params["weight_decay"],
                "lr": self.optimizer_params["lr"] * layer_decay_rate,
            }
            
            # Parameters without weight decay
            layer_params_no_decay = {
                "params": [p for n, p in self.model.named_parameters() 
                          if f"encoder.layer.{layer_idx}." in n 
                          and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.optimizer_params["lr"] * layer_decay_rate,
            }
            
            optimizer_grouped_parameters.extend([layer_params_decay, layer_params_no_decay])
        
        # Handle non-encoder parameters (e.g., classifier, pooler)
        other_params_decay = {
            "params": [p for n, p in self.model.named_parameters() 
                      if not any(f"encoder.layer.{i}." in n for i in range(num_layers + 1))
                      and not any(nd in n for nd in no_decay)],
            "weight_decay": self.optimizer_params["weight_decay"],
            "lr": self.optimizer_params["lr"],
        }
        
        other_params_no_decay = {
            "params": [p for n, p in self.model.named_parameters() 
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

    def prepare_data(self) -> None:
        if self.train_loader is None or self.val_loader is None:
            if hasattr(self.data_module, 'get_dataloaders'):
                # self.train_loader, self.val_loader = self.data_module.get_dataloaders(self.model.tokenizer, self.batch_size)
                self.train_loader, self.val_loader = self.data_module.get_dataloaders(self.batch_size)
            else:
                texts, labels = self.data_module.preprocess()

                train_texts, val_texts, train_labels, val_labels = train_test_split(
                    texts, labels, test_size=0.2, random_state=42
                )

                train_encodings = self.model.tokenizer(train_texts, truncation=True, padding=True)
                val_encodings = self.model.tokenizer(val_texts, truncation=True, padding=True)

                train_dataset = TensorDataset(
                    torch.tensor(train_encodings['input_ids']),
                    torch.tensor(train_encodings['attention_mask']),
                    torch.tensor(train_labels)
                )
                val_dataset = TensorDataset(
                    torch.tensor(val_encodings['input_ids']),
                    torch.tensor(val_encodings['attention_mask']),
                    torch.tensor(val_labels)
                )

                self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

    def train_epoch(self):
        """Modified train_epoch with gradient clipping and better progress reporting"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(
            self.train_loader,
            desc="Training",
            mininterval=10.0,
            miniters=int(len(self.train_loader)/1000),
            smoothing=0,
            leave=False
        )
        running_loss = 0
        batch_count = 0

        for batch in progress_bar:
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
            else:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            
            # Add gradient clipping
            clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            
            self.optimizer.step()
            
            # Step scheduler if using cosine decay
            if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step()
                
            total_loss += loss.item()
            # Update running averages
            batch_count += 1
            running_loss = (running_loss * (batch_count - 1) + loss.item()) / batch_count
            
            
            # Only update progress bar description when tqdm updates
            if batch_count % max(len(self.train_loader)//20, 1) == 0:
                progress_bar.set_postfix({
                    'avg_loss': f'{running_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                }, refresh=False)
                
            
        return total_loss / len(self.train_loader)

    def validate(self) -> Tuple[float, float, float, float, float]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        progress_bar = tqdm(self.val_loader, desc="Validating", mininterval=10)
        with torch.no_grad():
            for batch in progress_bar:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids, attention_mask)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        return total_loss / len(self.val_loader), accuracy, precision, recall, f1

    def train(self) -> List[Dict[str, float]]:
        self.prepare_data()
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        training_history = []

        for epoch in range(self.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            train_loss = self.train_epoch()
            val_loss, accuracy, precision, recall, f1 = self.validate()
            self.scheduler.step(val_loss)

            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            training_history.append(epoch_stats)

            self.logger.info(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
            self.logger.info(
                f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_trained_model_with_path(self.dataset_name, self.num_hidden_layers, filename='best_model.pth')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        return training_history

    def train_on_full_dataset(self, num_epochs: int) -> List[float]:
        """
        Train the model on the full dataset (training + validation).

        Args:
        num_epochs (int): Number of epochs to train for.

        Returns:
        List[float]: List of average losses for each epoch.
        """
        try:
            # Try without tokenizer first
            full_train_loader = self.data_module.get_full_train_dataloader(batch_size=self.batch_size)
        except TypeError:
            # Fall back to including tokenizer if needed
            full_train_loader = self.data_module.get_full_train_dataloader(self.model.tokenizer, self.batch_size)
            
        epoch_losses = []

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(
                full_train_loader, 
                desc=f"Full Training Epoch {epoch + 1}/{num_epochs}", 
                mininterval=10.0, 
                miniters=int(len(full_train_loader)/1000),
                position=0,
                leave=False
            )
            batch_count = 0
            running_loss = 0
            for batch in progress_bar:
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                else:
                    input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                    
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                
                # Add gradient clipping
                clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
                self.optimizer.step()
                
                # Step scheduler if using cosine decay
                if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step()
                    
                total_loss += loss.item()
                batch_count += 1
                running_loss = (running_loss * (batch_count - 1) + loss.item()) / batch_count

                # Update progress bar with loss and learning rate
                if batch_count % max(len(full_train_loader)//20, 1) == 0:
                    progress_bar.set_postfix({
                        'avg_loss': f'{running_loss:.4f}',
                        'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                    }, refresh=False)
            avg_loss = total_loss / len(full_train_loader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Step scheduler if using ReduceLROnPlateau
            if self.scheduler is not None and isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(avg_loss)

        return epoch_losses

    def test(self) -> Tuple[float, float, float, float, float]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        # Get the test dataloader
        test_loader = self.data_module.get_test_dataloader(self.model.tokenizer, self.batch_size)

        progress_bar = tqdm(test_loader, desc="Testing", mininterval=10)
        with torch.no_grad():
            for batch in progress_bar:
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                else:
                    input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids, attention_mask)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        return total_loss / len(test_loader), accuracy, precision, recall, f1

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
        self.logger.info(f"Model loaded from {path}")

    def get_best_model(self) -> nn.Module:
        self.load_model('best_model.pth')
        return self.model

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


def save_trained_model(trainer, dataset_name, num_hidden_layers):
    # Get the project root directory
    project_root = get_project_root()

    # Create the directory structure
    classification_word = trainer.classification_word.lower() if trainer.classification_word else "sentiment"
    save_dir = os.path.join(project_root, "trained_models", dataset_name, classification_word)
    os.makedirs(save_dir, exist_ok=True)

    model_name = trainer.model_name.lower() if trainer.model_name else "bert"
    # Create the filename with current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{model_name}_{num_hidden_layers}hidden_{current_time}.pth"

    # Full path
    full_path = os.path.join(save_dir, filename)

    # Use the trainer's save_model method
    trainer.save_model(full_path)


# Usage example
if __name__ == "__main__":
    from models.causal_neutral_model_variations import model_variations

    # Create model and data loader
    classification_word = "Sentiment"
    model = model_variations["distilbert"]["1_hidden"](classification_word, freeze_encoder=True)

    from data_loaders.imdb_sentiment.phrase_classification_dataloader import CausalNeutralDataModule

    file_path = "outputs/imdb_train_sentiment_analysis.json"
    sentiment_imdb_loader = CausalNeutralDataModule(file_path, classification_word)

    # Create and run trainer
    trainer = Trainer(model, sentiment_imdb_loader, optimizer_name='adamw', batch_size=32, num_epochs=5)
    trainer.train()
    save_trained_model(trainer, "imdb", 1)
