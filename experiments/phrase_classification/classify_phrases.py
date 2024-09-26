import os
import sys
from pathlib import Path

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from utilities import get_claude_response, save_results, set_seed, append_to_json
import json
import textwrap
from data_loaders.imdb import get_imdb_train_samples
from utilities.phrase_extraction import remove_punctuation_phrases, extract_phrases
import pandas as pd
import re
import argparse
from tqdm import tqdm 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from prompts import prompt_builder


def remove_non_ascii(text):
    if isinstance(text, str):
        return re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def classify_phrases(phrases, full_text, classification_word="Sentiment", dataset = 'imdb'):
    user_prompt = prompt_builder(phrases, full_text, classification_word, dataset=dataset)
    print(user_prompt)
    sys.exit()
    return get_claude_response(user_prompt)

def process_texts(texts, labels, classification_word="Sentiment", num_samples=None, output_file=None, dataset = 'imdb'):
    for i, (text, label) in tqdm(enumerate(zip(texts, labels))):
        if num_samples is not None and i >= num_samples:
            break
        text = text.replace('"', '')
        extracted_phrases = remove_punctuation_phrases(extract_phrases(text))
        analysis = classify_phrases(extracted_phrases, text, classification_word, dataset)

        if analysis:
            result = {
                'text': text,
                f'{classification_word.lower()}_phrases': analysis.get(f'{classification_word.lower()}_phrases', []),
                'neutral_phrases': analysis.get('neutral_phrases', []),
                'label': label
            }
            if output_file:
                append_to_json(result, output_file)
        else:
            print(f"Failed to get a valid response for text {i}")

def parse_args():
    parser = argparse.ArgumentParser(description="Script for phrase classification")
    parser.add_argument('--dataset', type=str, required=False, help='imdb, jailbreak, toxicity, olid', default='imdb')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    set_seed(42)
    num_examples, random_seed = 500, 42
    
    dataset = args.dataset 
    if dataset == 'imdb':
        train_texts, train_labels = get_imdb_train_samples(n=num_examples)
        classification_word = "Sentiment" 
        output_file = f'outputs/imdb_train_{classification_word.lower()}_analysis.json'
        process_texts(train_texts, train_labels, classification_word, output_file=output_file)
        print(f"Processed {num_examples} training texts "
            f"and saved results to imdb_train_{classification_word.lower()}_analysis.json")

    elif dataset == 'jailbreak':
        splits = {'train': 'data/0124/toxic-chat_annotation_train.csv', 'test': 'data/0124/toxic-chat_annotation_test.csv'}
        data = pd.read_csv("hf://datasets/lmsys/toxic-chat/" + splits["train"])
        data = data.sample(n=(num_examples / 100), random_state=random_seed)
        col = ['user_input', 'jailbreaking']        
        classification_word = 'jailbreak'
    
    elif dataset == 'toxicity':
        data = pd.read_csv('data/toxicity_data/train.csv')
        data = data.sample(n=num_examples, random_state=random_seed)
        col = ['comment_text', 'toxic']
        classification_word = 'toxic'
        
    elif dataset == 'olid':
        data = pd.read_csv('data/olid_data/olid-training-v1.0.tsv', sep='\t')
        data = data.sample(n=num_examples, random_state=random_seed)
        lb = LabelEncoder() 
        col = ['tweet', 'subtask_a']
        data[col[1]] = lb.fit_transform(data[col[1]])
        classification_word = 'offensive'


    data = data.map(lambda x: x.replace('"', "'") if isinstance(x, str) else x)
    data[col[0]] = data[col[0]].apply(lambda x: f'"{x}"' if isinstance(x, str) else x)
    data = data.map(remove_non_ascii)
    data[col[0]] = data[col[0]].str.replace(r'<.*?>', '', regex=True)
    data = data.replace('\n','', regex=True)
    
    try:
        os.mkdir(f'outputs/{dataset}')
    except:
        pass
    
    output_file = f'outputs/{dataset}/{dataset}_train_{classification_word.lower()}_analysis.json'
    process_texts(data[col[0]], data[col[1]], classification_word, output_file=output_file, dataset = dataset)
    print(f"Processed {num_examples} training texts "
        f"and saved results to {dataset}_train_{classification_word.lower()}_analysis.json")

if __name__ == "__main__":
    main()
