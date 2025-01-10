import pandas as pd
import os
from pathlib import Path
import sys
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))


from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from models.causal_neutral_model_variations import model_variations
from utilities.phrase_classifier_helpers import load_causal_neutral_classifier, extract_causal_and_neutral_phrases

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv('../../data/toxicity_data/train.csv')
    X_train, X_test, y_train, y_test = train_test_split(data['comment_text'], data['toxic'], test_size=0.33, random_state=42)

    output_dir = Path("../../data/jigsaw_toxicity")
    output_dir.mkdir(parents=True, exist_ok=True)


    for model_name in ["distilbert", "bert"]:
        print(f"\n==== Precomputing splits for {model_name} ====")
        classifier_model = load_causal_neutral_classifier(model_name, device)
        tokenizer = classifier_model.tokenizer

        print(f"Processing train set: {len(X_train)} examples")
        train_records = []
        for i in tqdm(range(len(X_train)), desc=f"Processing train set for {model_name}"):
            text  = X_train.loc[i]
            label = y_train.loc[i]
            print(text, label)
            causal_phrases, neutral_phrases = extract_causal_and_neutral_phrases(
                text, classifier_model, tokenizer
            )

            record = {
                "text": text,
                "label": label,
                "causal_phrases": causal_phrases,
                "neutral_phrases": neutral_phrases,
            }
            train_records.append(record)

        print(f"Processing test set: {len(X_test)} examples")
        test_records = []
        for example in tqdm(range(len(X_test)), desc=f"Processing test set for {model_name}"):
            text  = X_test.loc[i]
            label = y_test.loc[i]
            causal_phrases, neutral_phrases = extract_causal_and_neutral_phrases(
                text, classifier_model, tokenizer
            )

            record = {
                "text": text,
                "label": label,
                "causal_phrases": causal_phrases,
                "neutral_phrases": neutral_phrases,
            }
            test_records.append(record)

        # d) Write JSON files
        train_out_path = output_dir / f"train_with_causal_neutral_splits_{model_name}.json"
        test_out_path  = output_dir / f"test_with_causal_neutral_splits_{model_name}.json"

        print(f"Writing train output to: {train_out_path}")
        with open(train_out_path, "w", encoding="utf-8") as f:
            json.dump(train_records, f, indent=2, ensure_ascii=False)

        print(f"Writing test output to: {test_out_path}")
        with open(test_out_path, "w", encoding="utf-8") as f:
            json.dump(test_records, f, indent=2, ensure_ascii=False)

        print(f"Done for {model_name}!\n")


if __name__ == "__main__":
    main()
