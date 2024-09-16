import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from trainers.trainer import Trainer
from typing import Dict, List, Tuple


class RegularizedTrainer(Trainer):
    def __init__(self, model_eta, model_theta, data_module,
                 optimizer_name='adamw', dataset_name='imdb', optimizer_params=None,
                 batch_size=32, num_epochs=3, device=None, lambda_reg=0.1, **kwargs):
        super().__init__(model_theta, data_module, optimizer_name, dataset_name,
                         optimizer_params, batch_size, num_epochs, device, **kwargs)
        self.model_eta = model_eta.to(self.device)
        self.model_theta = model_theta.to(self.device)
        self.lambda_reg = lambda_reg
        self.optimizer_params = optimizer_params or {}

    def compute_regularizer(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        original_input_ids = batch['original_input_ids'].to(self.device)
        original_attention_mask = batch['original_attention_mask'].to(self.device)
        causal_probs = batch['causal_probs'].to(self.device)
        eta_probs = batch['eta_probs'].to(self.device)

        # Compute P_θ(y|s) using original reviews
        outputs_theta = self.model_theta(original_input_ids, original_attention_mask)
        probs_theta = F.softmax(outputs_theta.logits if hasattr(outputs_theta, 'logits') else outputs_theta, dim=1)

        # Calculate overall expectation (mean over all classes)
        overall_expectation = probs_theta.mean()

        # Number of classes
        num_classes = probs_theta.shape[1]

        # Compute regularization terms for each class
        reg_terms = []
        for i in range(num_classes):
            second_term = probs_theta[:, i] * (eta_probs[:, i] / causal_probs[:, i])
            second_term_mean = second_term.mean()
            reg_term = overall_expectation - second_term_mean
            reg_terms.append(reg_term)

        # Compute the 2-norm of the regularization terms
        regularizer = torch.norm(torch.tensor(reg_terms), p=2)

        return regularizer

    def compute_regularizer_old(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        original_input_ids = batch['original_input_ids'].to(self.device)
        original_attention_mask = batch['original_attention_mask'].to(self.device)
        causal_probs = batch['causal_probs'].to(self.device)
        eta_probs = batch['eta_probs'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Compute P_θ(y|s) using original reviews
        outputs_theta = self.model_theta(original_input_ids, original_attention_mask)
        probs_theta = F.softmax(outputs_theta.logits if hasattr(outputs_theta, 'logits') else outputs_theta, dim=1)

        # Use precomputed P_η(x|s) and P_η(x|causal phrase in s)
        probs_eta = eta_probs
        probs_causal = causal_probs

        # Compute regularizer
        reg_pos = torch.abs(probs_theta[:, 1] - probs_theta[:, 1] * (probs_eta[:, 1] / probs_causal[:, 1]))
        reg_neg = torch.abs(probs_theta[:, 0] - probs_theta[:, 0] * (probs_eta[:, 0] / probs_causal[:, 0]))

        regularizer_vector = torch.where(labels == 1, reg_pos, reg_neg)
        regularizer = torch.norm(regularizer_vector, p=2)

        return regularizer

    def compute_kl_divergence(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        original_input_ids = batch['original_input_ids'].to(self.device)
        original_attention_mask = batch['original_attention_mask'].to(self.device)
        eta_probs = batch['eta_probs'].to(self.device)

        outputs_theta = self.model_theta(original_input_ids, original_attention_mask)
        probs_theta = F.softmax(outputs_theta.logits if hasattr(outputs_theta, 'logits') else outputs_theta, dim=1)

        kl_div = F.kl_div(probs_theta.log(), eta_probs, reduction='batchmean')

        return kl_div

    def train_with_regularization(self, full_dataset=True) -> List[float]:
        if full_dataset:
            self.train_loader = self.data_module.get_full_train_dataloader(self.model.tokenizer, self.batch_size)
        else:
            self.prepare_data()
        self.model_eta.eval()  # Freeze P_η
        self.model_theta.train()

        epoch_losses = []
        for epoch in range(self.num_epochs):
            total_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for batch in progress_bar:
                self.optimizer.zero_grad()

                kl_div = self.compute_kl_divergence(batch)
                regularizer = self.compute_regularizer(batch)

                loss = kl_div + self.lambda_reg * regularizer

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / len(self.train_loader)
            epoch_losses.append(avg_loss)
            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, Avg Loss: {avg_loss:.4f}")

            if self.scheduler:
                self.scheduler.step(avg_loss)

        return epoch_losses

    def train(self, full_dataset=False) -> List[dict]:
        # Override the train method to use train_with_regularization
        return self.train_with_regularization(full_dataset=full_dataset)
  
    def test(self) -> Tuple[float, float, float, float, float]:
        self.model_theta.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        test_loader = self.data_module.get_test_dataloader(self.batch_size)

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model_theta(input_ids, attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss = total_loss / len(test_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

        return test_loss, accuracy, precision, recall, f1
