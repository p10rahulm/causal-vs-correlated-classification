import json
import os
import sys
from pathlib import Path

# Try to find the project root
current_file = Path(__file__).resolve()
project_root = current_file.parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

from data_loaders.base_data_module import BaseDataModule


class CausalNeutralDataModule(BaseDataModule):
    def __init__(self, file_path, classification_word):
        self.file_path = Path(file_path)
        self.classification_word = classification_word
        self.neutral_phrases = []
        self.causal_phrases = []  # Previously comparison_phrases
        self.load_data()  # Load data upon initialization

    def load_data(self):
        with self.file_path.open('r') as file:
            data = json.load(file)

        for item in data:
            self.neutral_phrases.extend(item.get('neutral_phrases', []))
            causal_key = f"{self.classification_word.lower()}_phrases"
            self.causal_phrases.extend(item.get(causal_key, []))

    def preprocess(self):
        # Placeholder implementation
        processed_data = []
        labels = []

        for phrase in self.neutral_phrases:
            processed_data.append(phrase)
            labels.append(0)  # 0 for neutral

        for phrase in self.causal_phrases:
            processed_data.append(phrase)
            labels.append(1)  # 1 for causal

        return processed_data, labels

    def get_class_names(self):
        return ["neutral", self.classification_word.lower()]


# Example usage
if __name__ == "__main__":
    file_path = "outputs/imdb_train_sentiment_analysis.json"
    classification_word = "Sentiment"

    data_module = CausalNeutralDataModule(file_path, classification_word)
    data_module.load_data()
    processed_data, labels = data_module.preprocess()

    print(f"Loaded {len(processed_data)} phrases")
    print(f"Class names: {data_module.get_class_names()}")
