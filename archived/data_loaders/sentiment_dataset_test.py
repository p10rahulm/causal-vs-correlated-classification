import torch
import logging
import pandas as pd
import re
import unicodedata
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, csv_file, tokenizer, text_column, sentiment_column, column_names=None, max_length=512):
        if column_names is None:
            column_names = [text_column, sentiment_column]
        self.data = pd.read_csv(csv_file, sep='|', skiprows=1, names=column_names)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.sentiment_column = sentiment_column

    def __len__(self):
        return len(self.data)

    def clean_text(self, text):
        # Remove non-ASCII characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx][self.text_column])
        text = self.clean_text(text)
        sentiment = self.data.iloc[idx][self.sentiment_column]
        
        # Convert sentiment to string and handle potential NaN values
        if pd.isna(sentiment):
            logging.warning(f"NaN sentiment value found at index {idx}. Defaulting to Negative.")
            label = 0
        else:
            sentiment = str(sentiment).strip()
            if sentiment.lower() == 'positive':
                label = 1
            elif sentiment.lower() == 'negative':
                label = 0
            else:
                logging.warning(f"Unexpected sentiment value: {sentiment} at index {idx}. Defaulting to Negative.")
                label = 0

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

