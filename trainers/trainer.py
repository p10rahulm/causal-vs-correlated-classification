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
from utilities import get_project_root
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any, Tuple, List, Union
import logging
from data_loaders.base_data_module import BaseDataModule


class Trainer:
    def __init__(self, model: nn.Module, data_module: Union[BaseDataModule, Any], optimizer_name: str = 'adam',
                 dataset_name: str = 'imdb',
                 optimizer_params: Dict[str, Any] = None, batch_size: int = 32,
                 num_epochs: int = 3, device: Union[str, None] = None, patience: int = 3):
        self.model = model
        self.data_module = data_module
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_name = type(model).__name__
        self.classification_word = model.classification_word
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.patience = patience

        self.dataset_name = dataset_name
        self.num_hidden_layers = self.model.num_hidden_layers

        # Set up optimizer
        optimizer_config = get_optimizer_config(optimizer_name)
        if not optimizer_config:
            raise ValueError(f"Optimizer {optimizer_name} not found in config")
        if optimizer_params:
            optimizer_config['params'].update(optimizer_params)
        self.optimizer = optimizer_config['class'](self.model.parameters(), **optimizer_config['params'])

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=patience)
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_loader = None
        self.train_loader = None

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def prepare_data(self) -> None:
        if self.train_loader is None or self.val_loader is None:
            if hasattr(self.data_module, 'get_dataloaders'):
                self.train_loader, self.val_loader = self.data_module.get_dataloaders(self.model.tokenizer,
                                                                                      self.batch_size)
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

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
            input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        return total_loss / len(self.train_loader)

    def validate(self) -> Tuple[float, float, float, float, float]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        progress_bar = tqdm(self.val_loader, desc="Validating")
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
        full_train_loader = self.data_module.get_full_train_dataloader(self.model.tokenizer, self.batch_size)
        epoch_losses = []

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(full_train_loader, desc=f"Full Training Epoch {epoch + 1}/{num_epochs}")

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
                self.optimizer.step()
                total_loss += loss.item()

                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / len(full_train_loader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

            # You might want to add learning rate scheduling here if you're using it
            if self.scheduler:
                self.scheduler.step(avg_loss)

        return epoch_losses

    def test(self) -> Tuple[float, float, float, float, float]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        # Get the test dataloader
        test_loader = self.data_module.get_test_dataloader(self.model.tokenizer, self.batch_size)

        progress_bar = tqdm(test_loader, desc="Testing")
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
    save_dir = os.path.join(project_root, "trained_models", dataset_name, trainer.classification_word.lower())
    os.makedirs(save_dir, exist_ok=True)

    # Create the filename with current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{trainer.model_name}_{num_hidden_layers}hidden_{current_time}.pth"

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

    from data_loaders.wz_data_loaders.sentiment_imdb import CausalNeutralDataModule

    file_path = "outputs/imdb_train_sentiment_analysis.json"
    sentiment_imdb_loader = CausalNeutralDataModule(file_path, classification_word)

    # Create and run trainer
    trainer = Trainer(model, sentiment_imdb_loader, optimizer_name='adamw', batch_size=32, num_epochs=5)
    trainer.train()
    save_trained_model(trainer, "imdb", 1)
