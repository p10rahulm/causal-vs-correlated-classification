import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer

class CausalNeutralClassifier(nn.Module):
    def __init__(self, model_name, classification_word, num_classes=2, dropout_rate=0.1, hidden_layers=[], freeze_encoder=True, max_length=1024):
        super().__init__()
        self.model_name = model_name
        self.classification_word = classification_word
        # Get original config to check max position embeddings
        original_config = AutoConfig.from_pretrained(model_name)
        original_max_length = original_config.max_position_embeddings
        
        # If original length is smaller than desired, keep original
        self.max_length = min(original_max_length, max_length)
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.max_position_embeddings = self.max_length
        self.config.max_length = self.max_length
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Initialize tokenizer with max length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=self.max_length,
            padding_side='right',  # Ensure consistent padding
            truncation_side='right'  # Truncate from the right side
        )
        self.num_hidden_layers = len(hidden_layers)
        self.dropout = nn.Dropout(dropout_rate)

        # Dynamically create classifier based on hidden_layers
        layers = []
        in_features = self.config.hidden_size
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_features = hidden_size
        layers.append(nn.Linear(in_features, num_classes))

        self.classifier = nn.Sequential(*layers)

        if freeze_encoder:
            self.freeze_encoder()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def preprocess_text(self, text):
        """Helper method to preprocess text with correct truncation"""
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_class_names(self):
        return ["neutral", f"causal_{self.classification_word.lower()}"]

    def to_device(self, device):
        self.to(device)
        self.device = device
        return self

def create_model(model_name, classification_word, hidden_layers=[],
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 freeze_encoder=True, max_length=1024):
    model = CausalNeutralClassifier(model_name, classification_word, hidden_layers=hidden_layers, freeze_encoder=freeze_encoder, max_length=max_length)
    return model.to_device(device)