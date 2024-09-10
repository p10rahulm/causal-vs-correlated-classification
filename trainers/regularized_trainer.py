import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from trainers.trainer import Trainer
from phrase_extraction import extract_phrases, remove_punctuation_phrases


class RegularizedTrainer(Trainer):
    def __init__(self, model_eta, model_theta, data_module, causal_phrase_model,
                 optimizer_name='adamw', dataset_name='imdb', optimizer_params=None,
                 batch_size=32, num_epochs=3, device=None, lambda_reg=0.1):
        super().__init__(model_theta, data_module, optimizer_name, dataset_name,
                         optimizer_params, batch_size, num_epochs, device)
        self.model_eta = model_eta.to(self.device)
        self.model_theta = model_theta.to(self.device)
        self.causal_phrase_model = causal_phrase_model.to(self.device)
        self.lambda_reg = lambda_reg

    def compute_regularizer(self, batch):
        input_ids, attention_mask, labels = [b.to(self.device) for b in batch]

        # Compute P_θ(y|s)
        outputs_theta = self.model_theta(input_ids, attention_mask)
        probs_theta = F.softmax(outputs_theta, dim=1)

        # Compute P_η(x|s) and P_η(x|causal phrase in s)
        outputs_eta = self.model_eta(input_ids, attention_mask)
        probs_eta = F.softmax(outputs_eta, dim=1)

        # Extract causal phrases
        causal_phrases = self.extract_causal_phrases(input_ids, attention_mask)

        # Compute probabilities for causal phrases
        causal_inputs = self.model_theta.tokenizer(causal_phrases, truncation=True, padding=True, return_tensors="pt")
        causal_inputs = {k: v.to(self.device) for k, v in causal_inputs.items()}

        with torch.no_grad():
            outputs_causal = self.causal_phrase_model(**causal_inputs)
        probs_causal = F.softmax(outputs_causal.logits, dim=1)

        # Compute regularizer
        reg_pos = probs_theta[:, 1] - probs_theta[:, 1] * (probs_eta[:, 1] / probs_causal[:, 1])
        reg_neg = probs_theta[:, 0] - probs_theta[:, 0] * (probs_eta[:, 0] / probs_causal[:, 0])

        regularizer = torch.where(labels == 1, reg_pos, reg_neg).mean()

        return regularizer

    def compute_kl_divergence(self, batch):
        input_ids, attention_mask, _ = [b.to(self.device) for b in batch]

        outputs_eta = self.model_eta(input_ids, attention_mask)
        outputs_theta = self.model_theta(input_ids, attention_mask)

        probs_eta = F.softmax(outputs_eta, dim=1)
        probs_theta = F.softmax(outputs_theta, dim=1)

        kl_div = F.kl_div(probs_theta.log(), probs_eta, reduction='batchmean')

        return kl_div

    def train_with_regularization(self):
        self.model_eta.eval()  # Freeze P_η
        self.model_theta.train()

        optimizer = AdamW(self.model_theta.parameters(), **self.optimizer_params)

        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
                optimizer.zero_grad()

                kl_div = self.compute_kl_divergence(batch)
                regularizer = self.compute_regularizer(batch)

                loss = kl_div + self.lambda_reg * regularizer

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Avg Loss: {avg_loss:.4f}")

    def extract_causal_phrases(self, input_ids, attention_mask):
        # Decode the input_ids to get the original text
        texts = self.model_theta.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        all_causal_phrases = []
        for text in texts:
            phrases = remove_punctuation_phrases(extract_phrases(text))

            # Classify phrases
            inputs = self.model_theta.tokenizer(phrases, truncation=True, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.causal_phrase_model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1)

            causal_phrases = [phrase for phrase, pred in zip(phrases, predictions) if pred == 1]

            if not causal_phrases:
                # Fallback strategy: use the first sentence or first 50 words
                sentences = text.split('.')
                if sentences:
                    causal_phrases = [sentences[0].strip()]
                else:
                    words = text.split()
                    causal_phrases = [' '.join(words[:50])]

            all_causal_phrases.append(' '.join(causal_phrases))

        return all_causal_phrases