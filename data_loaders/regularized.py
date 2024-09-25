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
from utilities.phrase_extraction import extract_phrases, remove_punctuation_phrases
import torch.nn.functional as F
from tqdm import tqdm

class SimpleDataset(Dataset):
    def __init__(self, dataset, text_column, label_column):
        self.dataset = dataset
        self.text_column = text_column
        self.label_column = label_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'text': item[self.text_column],
            'label': item[self.label_column]
        }

class PrecomputedCausalPhraseDataset(Dataset):
    def __init__(self, dataset, causal_neutral_model, eta_model, causal_tokenizer, eta_tokenizer, 
                 text_column, label_column, batch_size=16):
        self.dataset = dataset
        self.causal_neutral_model = causal_neutral_model
        self.eta_model = eta_model
        self.causal_tokenizer = causal_tokenizer
        self.eta_tokenizer = eta_tokenizer
        self.text_column = text_column
        self.label_column = label_column
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
            texts = batch[self.text_column]

            # Extract causal phrases
            causal_phrases = [self._extract_causal_phrase(text) for text in texts]
            self.causal_phrases.extend(causal_phrases)

            # Compute causal probabilities
            causal_probs = self._compute_probs_batch(causal_phrases, self.eta_model, self.eta_tokenizer)
            self.causal_probs.extend(causal_probs)

            # Compute eta probabilities
            eta_probs = self._compute_probs_batch(texts, self.eta_model, self.eta_tokenizer)
            self.eta_probs.extend(eta_probs)

    def _extract_causal_phrase(self, text):
        phrases = remove_punctuation_phrases(extract_phrases(text))
        classifications = self._classify_phrases_batch([phrases])
        causal_phrases = [phrase for phrase, cls in zip(phrases, classifications[0]) if cls == 1]

        if not causal_phrases:
            sentences = text.split('.')
            if sentences:
                causal_phrases = [sentences[0].strip()]
            else:
                words = text.split()
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
            'text': item[self.text_column],
            'causal_phrase': self.causal_phrases[idx],
            'causal_prob': torch.tensor(self.causal_probs[idx]),
            'eta_prob': torch.tensor(self.eta_probs[idx]),
            'label': item[self.label_column]
        }

class RegularizedDataModule:
    def __init__(self, base_data_module, classification_word, text_column, label_column, val_split=0.1, batch_size=16):
        self.base_data_module = base_data_module
        self.classification_word = classification_word
        self.text_column = text_column
        self.label_column = label_column
        self.val_split = val_split
        self.batch_size = batch_size
        self.causal_neutral_model = None
        self.eta_model = None
        self.causal_tokenizer = None
        self.eta_tokenizer = None

    def set_models(self, causal_model, eta_model):
        self.causal_neutral_model = causal_model
        self.eta_model = eta_model
        self.causal_tokenizer = causal_model.tokenizer
        self.eta_tokenizer = eta_model.tokenizer

    def get_dataloaders(self, batch_size=None):
        if self.causal_neutral_model is None or self.eta_model is None:
            raise ValueError("Models not set. Call set_models first.")

        batch_size = batch_size or self.batch_size
        train_dataset = PrecomputedCausalPhraseDataset(
            self.base_data_module.train_dataset, 
            self.causal_neutral_model, 
            self.eta_model,
            self.causal_tokenizer, 
            self.eta_tokenizer, 
            self.text_column, 
            self.label_column,
            batch_size=batch_size
        )
        val_dataset = PrecomputedCausalPhraseDataset(
            self.base_data_module.val_dataset, 
            self.causal_neutral_model, 
            self.eta_model,
            self.causal_tokenizer, 
            self.eta_tokenizer, 
            self.text_column, 
            self.label_column,
            batch_size=batch_size
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.train_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=self.train_collate_fn)

        return train_loader, val_loader

    def get_test_dataloader(self, batch_size=None):
        if self.causal_neutral_model is None or self.eta_model is None:
            raise ValueError("Models not set. Call set_models first.")

        batch_size = batch_size or self.batch_size
        test_dataset = SimpleDataset(self.base_data_module.test_dataset, self.text_column, self.label_column)
        return DataLoader(test_dataset, batch_size=batch_size, collate_fn=self.test_collate_fn)

    def get_full_train_dataloader(self, batch_size=None):
        if self.causal_neutral_model is None:
            raise ValueError("Causal Neutral model not set. Call set_models first.")

        batch_size = batch_size or self.batch_size

        full_dataset = ConcatDataset([
            PrecomputedCausalPhraseDataset(
                self.base_data_module.train_dataset, 
                self.causal_neutral_model, 
                self.eta_model,
                self.causal_tokenizer, 
                self.eta_tokenizer, 
                self.text_column, 
                self.label_column,
                batch_size=batch_size
            ),
            PrecomputedCausalPhraseDataset(
                self.base_data_module.val_dataset, 
                self.causal_neutral_model, 
                self.eta_model,
                self.causal_tokenizer, 
                self.eta_tokenizer, 
                self.text_column, 
                self.label_column,
                batch_size=batch_size
            )
        ])

        return DataLoader(full_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.train_collate_fn)

    def train_collate_fn(self, batch):
        original_texts = [item['text'] for item in batch]
        causal_phrases = [item['causal_phrase'] for item in batch]
        causal_probs = torch.stack([item['causal_prob'] for item in batch])
        eta_probs = torch.stack([item['eta_prob'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])

        original_encodings = self.eta_tokenizer(original_texts, truncation=True, padding=True)
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
    
    def test_collate_fn(self, batch):
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch])

        encodings = self.eta_tokenizer(texts, truncation=True, padding=True)

        return {
            'input_ids': torch.tensor(encodings['input_ids']),
            'attention_mask': torch.tensor(encodings['attention_mask']),
            'labels': labels
        }

    def get_class_names(self):
        return self.base_data_module.get_class_names()

# Example usage
if __name__ == "__main__":
    from data_loaders.imdb_sentiment.core import IMDBDataModule
    from data_loaders.jigsaw_toxicity.core import JigsawToxicityDataModule

    # For IMDB
    imdb_base_module = IMDBDataModule()
    imdb_causal_module = RegularizedDataModule(
        base_data_module=imdb_base_module,
        classification_word="Sentiment",
        text_column='text',
        label_column='label'
    )

    # For Jigsaw
    jigsaw_base_module = JigsawToxicityDataModule()
    jigsaw_causal_module = RegularizedDataModule(
        base_data_module=jigsaw_base_module,
        classification_word="toxic",
        text_column='comment_text',
        label_column='toxic'
    )

    # Use these modules in your pipeline...