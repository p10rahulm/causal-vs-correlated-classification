import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from data_loaders.classification_data_loaders.sentiment_imdb import IMDBDataModule
from phrase_extraction import extract_phrases, remove_punctuation_phrases


class CausalPhraseDataset(Dataset):
    def __init__(self, dataset, causal_neutral_model, tokenizer):
        self.dataset = dataset
        self.causal_neutral_model = causal_neutral_model
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        review, label = item['text'], item['label']
        phrases = remove_punctuation_phrases(extract_phrases(review))
        classifications = self.classify_phrases(phrases)
        causal_phrases = [phrase for phrase, cls in zip(phrases, classifications) if cls == 1]
        return ' '.join(causal_phrases), label

    def classify_phrases(self, phrases):
        inputs = self.tokenizer(phrases, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.causal_neutral_model.device) for k, v in inputs.items() if k != 'token_type_ids'}

        with torch.no_grad():
            outputs = self.causal_neutral_model(**inputs)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            predictions = torch.argmax(logits, dim=1)

        return predictions.cpu().numpy()


class CausalPhraseIMDBDataModule(IMDBDataModule):
    def __init__(self, classification_word="Sentiment", val_split=0.1):
        super().__init__(classification_word, val_split)
        self.causal_neutral_model = None
        self.tokenizer = None

    def set_causal_neutral_model(self, model, tokenizer):
        self.causal_neutral_model = model
        self.tokenizer = tokenizer

    def get_dataloaders(self, tokenizer, batch_size):
        if self.causal_neutral_model is None:
            raise ValueError("Causal Neutral model not set. Call set_causal_neutral_model first.")

        train_dataset = CausalPhraseDataset(self.train_dataset, self.causal_neutral_model, self.tokenizer)
        val_dataset = CausalPhraseDataset(self.val_dataset, self.causal_neutral_model, self.tokenizer)

        def collate_fn(batch):
            texts, labels = zip(*batch)
            encodings = tokenizer(texts, truncation=True, padding=True)
            return {
                'input_ids': torch.tensor(encodings['input_ids']),
                'attention_mask': torch.tensor(encodings['attention_mask']),
                'labels': torch.tensor(labels)
            }

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

        return train_loader, val_loader

    def get_test_dataloader(self, tokenizer, batch_size):
        if self.causal_neutral_model is None:
            raise ValueError("Causal Neutral model not set. Call set_causal_neutral_model first.")

        test_dataset = CausalPhraseDataset(self.test_dataset, self.causal_neutral_model, self.tokenizer)

        def collate_fn(batch):
            texts, labels = zip(*batch)
            encodings = tokenizer(texts, truncation=True, padding=True)
            return {
                'input_ids': torch.tensor(encodings['input_ids']),
                'attention_mask': torch.tensor(encodings['attention_mask']),
                'labels': torch.tensor(labels)
            }

        return DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    def get_full_train_dataloader(self, tokenizer, batch_size):
        if self.causal_neutral_model is None:
            raise ValueError("Causal Neutral model not set. Call set_causal_neutral_model first.")

        full_dataset = ConcatDataset([
            CausalPhraseDataset(self.train_dataset, self.causal_neutral_model, self.tokenizer),
            CausalPhraseDataset(self.val_dataset, self.causal_neutral_model, self.tokenizer)
        ])

        def collate_fn(batch):
            texts, labels = zip(*batch)
            encodings = tokenizer(texts, truncation=True, padding=True)
            return {
                'input_ids': torch.tensor(encodings['input_ids']),
                'attention_mask': torch.tensor(encodings['attention_mask']),
                'labels': torch.tensor(labels)
            }

        return DataLoader(full_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
