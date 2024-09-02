from utilities import get_claude_response, get_claude_pre_prompt, save_results, set_seed, append_to_json
import json
import textwrap
from data_loaders.imdb import get_imdb_train_samples, get_imdb_test_samples
from phrase_extraction import remove_punctuation_phrases, extract_phrases
import sys
from phrase_classification import prompt_builder

def read_examples_from_file(classification_word):
    file_path = f"prompt_templates/wz_classification/{classification_word.lower()}_cf.txt"
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Warning: Example file for {classification_word}_cf not found. Using default examples.")
        file_path = f"prompt_templates/wz_classification/sentiment.txt"
        with open(file_path, 'r') as file:
            return file.read().strip()


def pre_prompt_builder(full_text, label, classification_word="Sentiment"):
    return f"""
            You are given the following text:
            "{full_text}"
            And the label:
            "{label}"
            
            IMPORTANT : If label is 1, it indicates positive sentiment, and 0 implies negative sentiment.
            Change the text in such a way that the context remains the same but the "{classification_word}" is reversed.
            If the text given is positive, then the output must be negative, and vice versa.
            Provide only the revised text as a string and nothing else.
        """

def genre_prompt_builder(phrases, full_text, classification_word="Sentiment"):
    phrases_str = ", ".join(f'"{phrase}"' for phrase in phrases)
    # examples = read_examples_from_file(classification_word)
    
    return textwrap.dedent(f"""
            You are given the following full text:
            "{full_text}"
            From this text, the following phrases have been extracted:
            {phrases_str}
            Classify each phrase into one of two categories:
            "{classification_word}_phrases": Those phrases that are directly related to or express {classification_word.lower()}.
            "neutral_phrases": Those phrases that are not directly related to {classification_word.lower()}.
            Now, classify the extracted phrases from the given text based on the classification word "{classification_word}":
            Output your classification as a JSON object with two keys: "{classification_word.lower()}_phrases" and "neutral_phrases", each containing a list of the corresponding phrases.
            IMPORTANT: Your response must be ONLY valid JSON that matches the structure described. Do not include any explanatory text before or after the JSON. Begin your response with the opening curly brace '{{' and end with the closing curly brace '}}'.
        """).strip()


def classify_phrases(full_text, label, classification_word="Sentiment"):
    pre_prompt = get_claude_pre_prompt(pre_prompt_builder(full_text, label, classification_word))
    phrases = remove_punctuation_phrases(extract_phrases(pre_prompt))
    user_prompt = prompt_builder(phrases, pre_prompt, classification_word)
    return pre_prompt, get_claude_response(user_prompt)


def process_texts(texts, labels, classification_word="Sentiment", num_samples=None, output_file=None):
    for i, (text, label) in enumerate(zip(texts, labels)):
        if num_samples is not None and i >= num_samples:
            break
        text = text.replace('"', '')
        cf_text, analysis = classify_phrases(text, label, classification_word)
        if analysis:
            result = {
                'text': text,
                'cf_text': cf_text,
                f'{classification_word.lower()}_phrases': analysis.get(f'{classification_word.lower()}_phrases', []),
                'neutral_phrases': analysis.get('neutral_phrases', []),
                'label': label
            }
            if output_file:
                append_to_json(result, output_file)
        else:
            print(f"Failed to get a valid response for text {i}")


def main():
    # Process a subset of the training dataset
    set_seed(42)
    num_examples = 2
    train_texts, train_labels = get_imdb_train_samples(n=num_examples)
    classification_word = "Sentiment"  # This can be changed to any other word
    output_file = f'outputs/ood-data/imdb_ood_{classification_word.lower()}_cf_analysis.json'
    process_texts(train_texts, train_labels, classification_word, output_file=output_file)
    print(f"Processed {num_examples} training texts "
        f"and saved results to imdb_ood_{classification_word.lower()}_cf_analysis.json")


if __name__ == "__main__":
    main()