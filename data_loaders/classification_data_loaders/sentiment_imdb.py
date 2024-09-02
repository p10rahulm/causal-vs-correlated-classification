from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from data_loaders.base_data_module import BaseDataModule


class IMDBDataModule(BaseDataModule):
    def __init__(self, classification_word="Sentiment", val_split=0.1):
        self.classification_word = classification_word
        self.val_split = val_split
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.load_data()  # Load data upon initialization

    def load_data(self):
        dataset = load_dataset("imdb")
        self.test_dataset = dataset['test']

        # Split the original training set into train and validation
        train_val = dataset['train']
        val_size = int(len(train_val) * self.val_split)
        train_size = len(train_val) - val_size
        self.train_dataset, self.val_dataset = random_split(train_val, [train_size, val_size])

    def preprocess(self):
        train_texts = [self.train_dataset[i]['text'] for i in range(len(self.train_dataset))]
        train_labels = [self.train_dataset[i]['label'] for i in range(len(self.train_dataset))]
        val_texts = [self.val_dataset[i]['text'] for i in range(len(self.val_dataset))]
        val_labels = [self.val_dataset[i]['label'] for i in range(len(self.val_dataset))]

        return (train_texts + val_texts, train_labels + val_labels)

    def get_dataloaders(self, tokenizer, batch_size):
        train_texts = [self.train_dataset[i]['text'] for i in range(len(self.train_dataset))]
        train_labels = [self.train_dataset[i]['label'] for i in range(len(self.train_dataset))]
        val_texts = [self.val_dataset[i]['text'] for i in range(len(self.val_dataset))]
        val_labels = [self.val_dataset[i]['label'] for i in range(len(self.val_dataset))]

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

    def get_test_dataloader(self, tokenizer, batch_size):
        test_texts = self.test_dataset['text']
        test_labels = self.test_dataset['label']

        test_encodings = tokenizer(test_texts, truncation=True, padding=True)

        test_dataset = TensorDataset(
            torch.tensor(test_encodings['input_ids']),
            torch.tensor(test_encodings['attention_mask']),
            torch.tensor(test_labels)
        )

        return DataLoader(test_dataset, batch_size=batch_size)

    def get_full_train_dataloader(self, tokenizer, batch_size):
        # Combine train and validation datasets
        full_train_texts = [self.train_dataset[i]['text'] for i in range(len(self.train_dataset))] + \
                           [self.val_dataset[i]['text'] for i in range(len(self.val_dataset))]
        full_train_labels = [self.train_dataset[i]['label'] for i in range(len(self.train_dataset))] + \
                            [self.val_dataset[i]['label'] for i in range(len(self.val_dataset))]

        # Tokenize the full training set
        full_train_encodings = tokenizer(full_train_texts, truncation=True, padding=True)

        # Create a TensorDataset
        full_train_dataset = TensorDataset(
            torch.tensor(full_train_encodings['input_ids']),
            torch.tensor(full_train_encodings['attention_mask']),
            torch.tensor(full_train_labels)
        )

        # Create and return DataLoader
        return DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)

    def get_class_names(self):
        return ["negative", "positive"]