import torch
from data_loaders.classification_data_loaders.sentiment_imdb import IMDBDataModule
from phrase_extraction import extract_phrases, remove_punctuation_phrases


class CausalPhraseIMDBDataModule(IMDBDataModule):
    def __init__(self, classification_word="Sentiment", val_split=0.1):
        super().__init__(classification_word, val_split)
        self.causal_neutral_model = None
        self.tokenizer = None

    def set_causal_neutral_model(self, model, tokenizer):
        self.causal_neutral_model = model
        self.tokenizer = tokenizer

    def process_dataset(self, dataset):
        processed_data = []
        for item in dataset:
            review, label = item['text'], item['label']
            phrases = remove_punctuation_phrases(extract_phrases(review))
            classifications = self.classify_phrases(phrases)
            causal_phrases = [phrase for phrase, cls in zip(phrases, classifications) if
                              cls == 1]  # Assuming 1 is Causal
            processed_data.append((' '.join(causal_phrases), label))
        return processed_data

    def classify_phrases(self, phrases):
        inputs = self.tokenizer(phrases, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.causal_neutral_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.causal_neutral_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)

        return predictions.cpu().numpy()

    def get_dataloaders(self, tokenizer, batch_size):
        if self.causal_neutral_model is None:
            raise ValueError("Causal Neutral model not set. Call set_causal_neutral_model first.")

        train_data = self.process_dataset(self.train_dataset)
        val_data = self.process_dataset(self.val_dataset)

        train_texts, train_labels = zip(*train_data)
        val_texts, val_labels = zip(*val_data)

        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)

        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_encodings['input_ids']),
            torch.tensor(train_encodings['attention_mask']),
            torch.tensor(train_labels)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(val_encodings['input_ids']),
            torch.tensor(val_encodings['attention_mask']),
            torch.tensor(val_labels)
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader

    def get_test_dataloader(self, tokenizer, batch_size):
        if self.causal_neutral_model is None:
            raise ValueError("Causal Neutral model not set. Call set_causal_neutral_model first.")

        test_data = self.process_dataset(self.test_dataset)
        test_texts, test_labels = zip(*test_data)

        test_encodings = tokenizer(test_texts, truncation=True, padding=True)

        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(test_encodings['input_ids']),
            torch.tensor(test_encodings['attention_mask']),
            torch.tensor(test_labels)
        )

        return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
