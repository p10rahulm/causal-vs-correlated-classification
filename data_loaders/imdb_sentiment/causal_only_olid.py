import json
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset


class CausalOnlyOLIDDataset(Dataset):
    """
    Reads a precomputed JSON file where each record has:
      {
        "text": "...",
        "label": 0 or 1,
        "causal_phrases": [...],
        "neutral_phrases": [...]
      }
    Then returns ' '.join(causal_phrases) as the input text, and label as the class.
    """

    def __init__(self, json_path, fallback_to_original=False):
        super().__init__()
        self.fallback_to_original = fallback_to_original
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        label = record["label"]  # 0=negative, 1=positive
        # Join all causal phrases into one string
        causal_phrases = record.get("causal_phrases", [])
        text = " ".join(causal_phrases)

        # If there are no causal phrases and you want a fallback:
        if not text.strip() and self.fallback_to_original:
            # e.g., fallback to the original text or first 50 words
            text = record["text"][:200]  # or some fallback

        return text, label


class CausalOnlyOLIDDataModule:
    def __init__(self,
                 train_json="data/olid_offensive/train_with_causal_neutral_splits_bert.json",
                 test_json="data/olid_offensive/test_with_causal_neutral_splits_bert.json",
                 val_split=0.1):
        """
        :param train_json: Path to precomputed JSON for training
        :param test_json:  Path to precomputed JSON for testing
        :param val_split:  Fraction of train used for validation
        """
        self.train_json = train_json
        self.test_json = test_json
        self.val_split = val_split

        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None

        self.load_data()

    def load_data(self):
        # 1) Read the train JSON
        full_dataset = CausalOnlyOLIDDataset(self.train_json)

        # 2) Split into train & val
        total_len = len(full_dataset)
        val_size  = int(self.val_split * total_len)
        train_size = total_len - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # or remove for random each time
        )

        # 3) Read the test JSON
        self.test_dataset = CausalOnlyOLIDDataset(self.test_json)

    def get_dataloaders(self, tokenizer, batch_size=32):
        """
        Return separate train & val DataLoaders.
        """
        def collate_fn(batch):
            # batch is list of (text, label)
            texts, labels = zip(*batch)
            encodings = tokenizer(list(texts), truncation=True, padding=True)
            return {
                "input_ids":      torch.tensor(encodings["input_ids"]),
                "attention_mask": torch.tensor(encodings["attention_mask"]),
                "labels":         torch.tensor(labels),
            }

        train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_fn)
        val_loader   = DataLoader(self.val_dataset, batch_size=batch_size,
                                  shuffle=False, collate_fn=collate_fn)
        return train_loader, val_loader

    def get_test_dataloader(self, tokenizer, batch_size=32):
        """
        Return test DataLoader.
        """
        def collate_fn(batch):
            texts, labels = zip(*batch)
            encodings = tokenizer(list(texts), truncation=True, padding=True)
            return {
                "input_ids":      torch.tensor(encodings["input_ids"]),
                "attention_mask": torch.tensor(encodings["attention_mask"]),
                "labels":         torch.tensor(labels),
            }

        return DataLoader(self.test_dataset, batch_size=batch_size,
                          shuffle=False, collate_fn=collate_fn)

    def get_full_train_dataloader(self, tokenizer, batch_size=32):
        """
        Return a DataLoader that merges train + val for final training.
        """
        full_data = ConcatDataset([self.train_dataset, self.val_dataset])

        def collate_fn(batch):
            texts, labels = zip(*batch)
            encodings = tokenizer(list(texts), truncation=True, padding=True)
            return {
                "input_ids":      torch.tensor(encodings["input_ids"]),
                "attention_mask": torch.tensor(encodings["attention_mask"]),
                "labels":         torch.tensor(labels),
            }

        return DataLoader(full_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
