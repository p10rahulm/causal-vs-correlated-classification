import os
from pathlib import Path
import sys

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))



import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from data_loaders.classification_data_loaders.sentiment_imdb import IMDBDataModule
from phrase_extraction import extract_phrases, remove_punctuation_phrases
import torch.nn.functional as F
from tqdm import tqdm

class PrecomputedCausalPhraseDataset(Dataset):
    def __init__(self, dataset, causal_neutral_model, eta_model, causal_tokenizer, eta_tokenizer):
        self.dataset = dataset
        self.causal_neutral_model = causal_neutral_model
        self.eta_model = eta_model
        self.causal_tokenizer = causal_tokenizer
        self.eta_tokenizer = eta_tokenizer
        self.device = next(causal_neutral_model.parameters()).device

        self.causal_phrases = []
        self.causal_probs = []
        self.eta_probs = []

        self._precompute_all()

    def _precompute_all(self):
        print("Precomputing causal phrases and probabilities...")
        for idx in tqdm(range(len(self.dataset)), desc="Processing reviews", unit="review"):
            item = self.dataset[idx]
            review, label = item['text'], item['label']

            causal_phrase = self._extract_causal_phrase(review)
            causal_prob = self._compute_causal_prob(causal_phrase)
            eta_prob = self._compute_eta_prob(review)

            self.causal_phrases.append(causal_phrase)
            self.causal_probs.append(causal_prob)
            self.eta_probs.append(eta_prob)

    def _extract_causal_phrase(self, review):
        phrases = remove_punctuation_phrases(extract_phrases(review))
        classifications = self.classify_phrases(phrases)
        causal_phrases = [phrase for phrase, cls in zip(phrases, classifications) if cls == 1]

        if not causal_phrases:
            sentences = review.split('.')
            if sentences:
                causal_phrases = [sentences[0].strip()]
            else:
                words = review.split()
                causal_phrases = [' '.join(words[:50])]

        return ' '.join(causal_phrases)

    def classify_phrases(self, phrases):
        inputs = self.causal_tokenizer(phrases, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k != 'token_type_ids'}

        with torch.no_grad():
            outputs = self.causal_neutral_model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            predictions = torch.argmax(logits, dim=1)

        return predictions.cpu().numpy()

    def _compute_causal_prob(self, causal_phrase):
        inputs = self.eta_tokenizer(causal_phrase, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k != 'token_type_ids'}

        with torch.no_grad():
            outputs = self.eta_model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probs = F.softmax(logits, dim=1)

        return probs.squeeze().cpu()

    def _compute_eta_prob(self, review):
        inputs = self.eta_tokenizer(review, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k != 'token_type_ids'}

        with torch.no_grad():
            outputs = self.eta_model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probs = F.softmax(logits, dim=1)

        return probs.squeeze().cpu()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'text': item['text'],
            'causal_phrase': self.causal_phrases[idx],
            'causal_prob': self.causal_probs[idx],
            'eta_prob': self.eta_probs[idx],
            'label': item['label']
        }

    def set_debug_mode(self, mode):
        self.debug_mode = mode


import torch
from torch.utils.data import Dataset, DataLoader
from phrase_extraction import extract_phrases, remove_punctuation_phrases
import torch.nn.functional as F
from tqdm import tqdm


class PrecomputedCausalPhraseDataset(Dataset):
    def __init__(self, dataset, causal_neutral_model, eta_model, causal_tokenizer, eta_tokenizer, batch_size=16):
        self.dataset = dataset
        self.causal_neutral_model = causal_neutral_model
        self.eta_model = eta_model
        self.causal_tokenizer = causal_tokenizer
        self.eta_tokenizer = eta_tokenizer
        self.device = next(causal_neutral_model.parameters()).device
        self.batch_size = batch_size

        self.causal_phrases = []
        self.causal_probs = []
        self.eta_probs = []

        self._precompute_all()

    def _precompute_all(self):
        print("Precomputing causal phrases and probabilities...")
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
            reviews = batch['text']

            # Extract causal phrases
            causal_phrases = [self._extract_causal_phrase(review) for review in reviews]
            self.causal_phrases.extend(causal_phrases)

            # Compute causal probabilities
            causal_probs = self._compute_probs_batch(causal_phrases, self.eta_model, self.eta_tokenizer)
            self.causal_probs.extend(causal_probs)

            # Compute eta probabilities
            eta_probs = self._compute_probs_batch(reviews, self.eta_model, self.eta_tokenizer)
            self.eta_probs.extend(eta_probs)

    def _extract_causal_phrase(self, review):
        phrases = remove_punctuation_phrases(extract_phrases(review))
        classifications = self._classify_phrases_batch([phrases])
        causal_phrases = [phrase for phrase, cls in zip(phrases, classifications[0]) if cls == 1]

        if not causal_phrases:
            sentences = review.split('.')
            if sentences:
                causal_phrases = [sentences[0].strip()]
            else:
                words = review.split()
                causal_phrases = [' '.join(words[:50])]

        return ' '.join(causal_phrases)

    def _classify_phrases_batch(self, phrase_batches):
        all_predictions = []
        for phrases in phrase_batches:
            inputs = self.causal_tokenizer(phrases, truncation=True, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k != 'token_type_ids'}

            with torch.no_grad():
                outputs = self.causal_neutral_model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                predictions = torch.argmax(logits, dim=1)
            all_predictions.append(predictions.cpu().numpy())
        return all_predictions

    def _compute_probs_batch(self, texts, model, tokenizer):
        inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k != 'token_type_ids'}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probs = F.softmax(logits, dim=1)

        return probs.cpu().tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'text': item['text'],
            'causal_phrase': self.causal_phrases[idx],
            'causal_prob': torch.tensor(self.causal_probs[idx]),
            'eta_prob': torch.tensor(self.eta_probs[idx]),
            'label': item['label']
        }



class CausalPhraseWithOriginalIMDBDataModule(IMDBDataModule):
    def __init__(self, classification_word="Sentiment", val_split=0.1):
        super().__init__(classification_word, val_split)
        self.causal_neutral_model = None
        self.eta_model = None
        self.causal_tokenizer = None
        self.eta_tokenizer = None

    def set_models(self, causal_model, eta_model):
        self.causal_neutral_model = causal_model
        self.eta_model = eta_model
        self.causal_tokenizer = causal_model.tokenizer
        self.eta_tokenizer = eta_model.tokenizer

    def get_dataloaders(self, batch_size):
        if self.causal_neutral_model is None or self.eta_model is None:
            raise ValueError("Models not set. Call set_models first.")

        train_dataset = PrecomputedCausalPhraseDataset(self.train_dataset, self.causal_neutral_model, self.eta_model,
                                                       self.causal_tokenizer, self.eta_tokenizer)
        val_dataset = PrecomputedCausalPhraseDataset(self.val_dataset, self.causal_neutral_model, self.eta_model,
                                                     self.causal_tokenizer, self.eta_tokenizer)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=self.collate_fn)

        return train_loader, val_loader

    def get_test_dataloader(self, batch_size):
        if self.causal_neutral_model is None or self.eta_model is None:
            raise ValueError("Models not set. Call set_models first.")

        test_dataset = PrecomputedCausalPhraseDataset(self.test_dataset, self.causal_neutral_model, self.eta_model,
                                                      self.causal_tokenizer, self.eta_tokenizer)

        return DataLoader(test_dataset, batch_size=batch_size, collate_fn=self.collate_fn)

    def get_full_train_dataloader(self, tokenizer, batch_size):
        if self.causal_neutral_model is None:
            raise ValueError("Causal Neutral model not set. Call set_causal_neutral_model first.")

        full_dataset = ConcatDataset([
            PrecomputedCausalPhraseDataset(self.train_dataset, self.causal_neutral_model, self.eta_model,
                                           self.causal_tokenizer, self.eta_tokenizer),
            PrecomputedCausalPhraseDataset(self.val_dataset, self.causal_neutral_model, self.eta_model,
                                           self.causal_tokenizer, self.eta_tokenizer)
        ])

        return DataLoader(full_dataset, batch_size=batch_size, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        original_reviews = [item['text'] for item in batch]
        causal_phrases = [item['causal_phrase'] for item in batch]
        causal_probs = torch.stack([item['causal_prob'] for item in batch])
        eta_probs = torch.stack([item['eta_prob'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])

        original_encodings = self.eta_tokenizer(original_reviews, truncation=True, padding=True)
        causal_encodings = self.causal_tokenizer(causal_phrases, truncation=True, padding=True)

        return {
            'original_input_ids': torch.tensor(original_encodings['input_ids']),
            'original_attention_mask': torch.tensor(original_encodings['attention_mask']),
            'causal_input_ids': torch.tensor(causal_encodings['input_ids']),
            'causal_attention_mask': torch.tensor(causal_encodings['attention_mask']),
            'causal_probs': causal_probs,
            'eta_probs': eta_probs,
            'labels': labels
        }


#
# if __name__ == "__main__":
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
#     # Initialize the data module
#     data_module = CausalPhraseWithOriginalIMDBDataModule()
#
#     # Load a pre-trained model and tokenizer for testing
#     model_name = "distilbert-base-uncased"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     causal_neutral_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
#
#     # Set the causal neutral model
#     data_module.set_causal_neutral_model(causal_neutral_model, tokenizer)
#
#     # Get dataloaders
#     train_loader, val_loader = data_module.get_dataloaders(tokenizer, batch_size=4)
#     test_loader = data_module.get_test_dataloader(tokenizer, batch_size=4)
#     full_train_loader = data_module.get_full_train_dataloader(tokenizer, batch_size=4)
#
#     # Test train_loader
#     print("Testing train_loader:")
#     batch = next(iter(train_loader))
#     print(f"Batch keys: {batch.keys()}")
#     print(f"Causal input shape: {batch['causal_input_ids'].shape}")
#     print(f"Original input shape: {batch['original_input_ids'].shape}")
#     print(f"Labels shape: {batch['labels'].shape}")
#
#     # Test val_loader
#     print("\nTesting val_loader:")
#     batch = next(iter(val_loader))
#     print(f"Batch keys: {batch.keys()}")
#     print(f"Causal input shape: {batch['causal_input_ids'].shape}")
#     print(f"Original input shape: {batch['original_input_ids'].shape}")
#     print(f"Labels shape: {batch['labels'].shape}")
#
#     # Test test_loader
#     print("\nTesting test_loader:")
#     batch = next(iter(test_loader))
#     print(f"Batch keys: {batch.keys()}")
#     print(f"Causal input shape: {batch['causal_input_ids'].shape}")
#     print(f"Original input shape: {batch['original_input_ids'].shape}")
#     print(f"Labels shape: {batch['labels'].shape}")
#
#     # Test full_train_loader
#     print("\nTesting full_train_loader:")
#     batch = next(iter(full_train_loader))
#     print(f"Batch keys: {batch.keys()}")
#     print(f"Causal input shape: {batch['causal_input_ids'].shape}")
#     print(f"Original input shape: {batch['original_input_ids'].shape}")
#     print(f"Labels shape: {batch['labels'].shape}")
#
#     print("\nAll tests completed successfully!")
