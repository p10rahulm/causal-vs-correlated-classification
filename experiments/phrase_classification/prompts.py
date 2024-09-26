import textwrap
def read_examples_from_file(classification_word):
    file_path = f"../../prompt_templates/wz_classification/{classification_word.lower()}.txt"
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Warning: Example file for {classification_word} not found. Using default examples.")
        file_path = f"../../prompt_templates/wz_classification/sentiment.txt"
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

    elif dataset == 'jailbreak':
        return textwrap.dedent(f"""
            You are given the following full text:
            "{full_text}"
            From this text, the following phrases have been extracted:
            {phrases_str}
            Your task is to classify the phrases into two sets, one which contains phrases causing the {classification_word}ing to occur
            (i.e., phrases where there is a deliberate attempt to trick or manipulate the output) and the other containing the neutral ones.

            1. Identify the key phrases or sentences that are most responsible for determining {classification_word}ing.
            2. Add these phrases to {classification_word}ing_phrases.
            3. Add the remaining phrases to neutral_phrases.
            4. In case of lack of clarity, add the phrase to neutral_phrases.
            Output your classification as a JSON object with two keys: "{classification_word.lower()}_phrases" and "neutral_phrases", each containing a list of the corresponding phrases.
            IMPORTANT: Your response must be ONLY valid JSON that matches the structure described. Do not include any explanatory text before or after the JSON. Begin your response with the opening curly brace '{{' and end with the closing curly brace '}}'.
        """).strip()
        
    elif dataset == 'jigsaw_toxicity':
        return textwrap.dedent(f"""
            You are given the following full text:
            "{full_text}"
            From this text, the following phrases have been extracted:
            {phrases_str}
            
            Your task is to classify the phrases into two sets, one which contains phrases causing the {classification_word}ity to occur
            and the other containing the neutral ones.

            1. Identify the key phrases or sentences that are most responsible for determining {classification_word}ity.
            2. Add these phrases to {classification_word}_phrases.
            3. Add the remaining phrases to neutral_phrases.
            4. In case of lack of clarity, add the phrase to neutral_phrases.
            Output your classification as a JSON object with two keys: "{classification_word.lower()}_phrases" and "neutral_phrases", each containing a list of the corresponding phrases.
            IMPORTANT: Your response must be ONLY valid JSON that matches the structure described. Do not include any explanatory text before or after the JSON. Begin your response with the opening curly brace '{{' and end with the closing curly brace '}}'.
        """).strip()

    elif dataset == 'olid':
        return textwrap.dedent(f"""
            You are given the following full text:
            "{full_text}"
            From this text, the following phrases have been extracted:
            {phrases_str}
            
            Your task is to classify the phrases into two sets, one which contains phrases which cause the sentence to be {classification_word}.
            The other set must contain the neutral ones.

            1. Identify the key phrases or sentences that are most responsible for being {classification_word}.
            2. Add these phrases to {classification_word}_phrases.
            3. Add the remaining phrases to neutral_phrases.
            4. In case of lack of clarity, add the phrase to neutral_phrases.
            Output your classification as a JSON object with two keys: "{classification_word.lower()}_phrases" and "neutral_phrases", each containing a list of the corresponding phrases.
            IMPORTANT: Your response must be ONLY valid JSON that matches the structure described. Do not include any explanatory text before or after the JSON. Begin your response with the opening curly brace '{{' and end with the closing curly brace '}}'.
        """).strip()