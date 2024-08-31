import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


class DistilBERTCausalNeutralClassifier(nn.Module):
    def __init__(self, classification_word, num_classes=2, dropout_rate=0.1):
        super().__init__()
        self.model_name = "distilbert-base-uncased"
        self.classification_word = classification_word
        self.config = DistilBertConfig.from_pretrained(self.model_name)
        self.encoder = DistilBertModel.from_pretrained(self.model_name, config=self.config)

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_class_names(self):
        return ["neutral", f"causal_{self.classification_word.lower()}"]
