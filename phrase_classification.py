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



def remove_non_ascii(text):
    if isinstance(text, str):
        # Remove non-ASCII characters using regex
        return re.sub(r'[^\x00-\x7F]+', '', text)
    return text


def read_examples_from_file(classification_word):
    file_path = f"prompt_templates/wz_classification/{classification_word.lower()}.txt"
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Warning: Example file for {classification_word} not found. Using default examples.")
        file_path = f"prompt_templates/wz_classification/sentiment.txt"
        with open(file_path, 'r') as file:
            return file.read().strip()


def prompt_builder(phrases, full_text, classification_word="Sentiment", dataset = 'imdb'):
    phrases_str = ", ".join(f'"{phrase}"' for phrase in phrases)
    examples = read_examples_from_file(classification_word)
    if dataset == 'imdb':
        return textwrap.dedent(f"""
            You are given the following full text:
            "{full_text}"
            From this text, the following phrases have been extracted:
            {phrases_str}
            Classify each phrase into one of two categories:
            "{classification_word}_phrases": Those phrases that are directly related to or express {classification_word.lower()}.
            "neutral_phrases": Those phrases that are not directly related to {classification_word.lower()}.
            {examples}
            Now, classify the extracted phrases from the given text based on the classification word "{classification_word}":
            Output your classification as a JSON object with two keys: "{classification_word.lower()}_phrases" and "neutral_phrases", each containing a list of the corresponding phrases.
            IMPORTANT: Your response must be ONLY valid JSON that matches the structure described. Do not include any explanatory text before or after the JSON. Begin your response with the opening curly brace '{{' and end with the closing curly brace '}}'.
        """).strip()

    else:
        return textwrap.dedent(f"""
            You are given the following full text:
            "{full_text}"
            From this text, the following phrases have been extracted:
            {phrases_str}
            Classify each phrase into one of two categories:
            "{classification_word}_phrases": Those phrases that are directly related to or express {classification_word.lower()} behaviour.
            "neutral_phrases": Those phrases that are not directly related to {classification_word.lower()} behaviour.
            {examples}
            Now, classify the extracted phrases from the given text based on the classification word "{classification_word}":
            Output your classification as a JSON object with two keys: "{classification_word.lower()}_phrases" and "neutral_phrases", each containing a list of the corresponding phrases.
            IMPORTANT: Your response must be ONLY valid JSON that matches the structure described. Do not include any explanatory text before or after the JSON. Begin your response with the opening curly brace '{{' and end with the closing curly brace '}}'.
        """).strip()


def classify_phrases(phrases, full_text, classification_word="Sentiment", dataset = 'imdb'):
    user_prompt = prompt_builder(phrases, full_text, classification_word, dataset=dataset)
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
    num_examples, random_seed = 1000, 42
    
    dataset = args.dataset 
    if dataset == 'imdb':
        train_texts, train_labels = get_imdb_train_samples(n=num_examples)
        classification_word = "Sentiment"  # or Genre
        
        output_file = f'outputs/imdb_train_{classification_word.lower()}_analysis.json'
        process_texts(train_texts, train_labels, classification_word, output_file=output_file)
        print(f"Processed {num_examples} training texts "
            f"and saved results to imdb_train_{classification_word.lower()}_analysis.json")

    elif dataset == 'jailbreak':
        splits = {'train': 'data/0124/toxic-chat_annotation_train.csv', 'test': 'data/0124/toxic-chat_annotation_test.csv'}
        data = pd.read_csv("hf://datasets/lmsys/toxic-chat/" + splits["train"])
        data = data.sample(n=int(len(data)/10), random_state=random_seed)
        col = ['user_input', 'toxicity']
        classification_word = 'toxic'
    
    elif dataset == 'jigsaw_toxicity':
        data = pd.read_csv('data/toxicity_data/train.csv')
        data = data.sample(n=1000, random_state=random_seed)
        col = ['comment_text', 'toxic']
        classification_word = 'toxic'
        
    elif dataset == 'olid':
        data = pd.read_csv('data/olid_data/olid-training-v1.0.tsv', sep='\t')
        data = data.sample(n=1000, random_state=random_seed)
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
