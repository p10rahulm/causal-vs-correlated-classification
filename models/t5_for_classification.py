import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer

class T5ForClassification(nn.Module):
    def __init__(self, t5_encoder, num_classes=2, hidden_layers=[], classification_word="Sentiment",
                 tokenizer=T5Tokenizer.from_pretrained("t5-base")):
        super().__init__()
        self.t5_encoder = t5_encoder
        self.classification_word = classification_word
        self.tokenizer = tokenizer
        self.num_hidden_layers = len(hidden_layers)  # Add this line
        self.model_name = "t5-small"
        # Create classifier
        layers = []
        in_features = t5_encoder.config.d_model
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_features = hidden_size
        layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):
        outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use the first token's representation
        logits = self.classifier(pooled_output)
        return logits

    def freeze_encoder(self):
        for param in self.t5_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.t5_encoder.parameters():
            param.requires_grad = True

    def to_device(self, device):
        self.to(device)
        return self

    def get_class_names(self):
        return ["neutral", f"causal_{self.classification_word.lower()}"]