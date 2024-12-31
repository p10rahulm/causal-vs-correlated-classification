import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from data_loaders.common.base_data_module import BaseDataModule
from sklearn.model_selection import train_test_split

class CausalNeutralDataModule(BaseDataModule):
    def __init__(self, file_path, classification_word, val_split=0.2):
        self.file_path = Path(file_path)
        self.classification_word = classification_word
        self.val_split = val_split
        self.neutral_phrases = []
        self.causal_phrases = []
        self.train_dataset = None
        self.val_dataset = None
        self.load_data()

    def load_data(self):
        with self.file_path.open('r') as file:
            data = json.load(file)

        for item in data:
            self.neutral_phrases.extend(item.get('neutral_phrases', []))
            causal_key = f"{self.classification_word.lower()}_phrases"
            self.causal_phrases.extend(item.get(causal_key, []))

        # Create full dataset
        full_texts = self.neutral_phrases + self.causal_phrases
        full_labels = [0] * len(self.neutral_phrases) + [1] * len(self.causal_phrases)

        # Split into train and validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            full_texts, full_labels, test_size=self.val_split, random_state=42
        )

        self.train_dataset = list(zip(train_texts, train_labels))
        self.val_dataset = list(zip(val_texts, val_labels))

    def preprocess(self):
        texts = [item[0] for item in self.train_dataset + self.val_dataset]
        labels = [item[1] for item in self.train_dataset + self.val_dataset]
        return texts, labels

    def get_dataloaders(self, tokenizer, batch_size):
        train_texts = [item[0] for item in self.train_dataset]
        train_labels = [item[1] for item in self.train_dataset]
        val_texts = [item[0] for item in self.val_dataset]
        val_labels = [item[1] for item in self.val_dataset]

        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)

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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader

    def get_full_train_dataloader(self, tokenizer, batch_size):
        full_texts = [item[0] for item in self.train_dataset + self.val_dataset]
        full_labels = [item[1] for item in self.train_dataset + self.val_dataset]

        full_encodings = tokenizer(full_texts, truncation=True, padding=True)

        full_dataset = TensorDataset(
            torch.tensor(full_encodings['input_ids']),
            torch.tensor(full_encodings['attention_mask']),
            torch.tensor(full_labels)
        )

        return DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    def get_class_names(self):
        return ["neutral", self.classification_word.lower()]

# Example usage
if __name__ == "__main__":
    file_path = "outputs/imdb_train_sentiment_analysis.json"
    classification_word = "Sentiment"

    data_module = CausalNeutralDataModule(file_path, classification_word)
    processed_data, labels = data_module.preprocess()

    print(f"Loaded {len(processed_data)} phrases")
    print(f"Class names: {data_module.get_class_names()}")