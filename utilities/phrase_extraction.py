import spacy
import string
from utilities import load_spacy_model
# Load the English model
nlp = load_spacy_model("en_core_web_sm")



def extract_phrases(sentence):
    """
    Extracts phrases from a given sentence using spaCy.
    Includes noun chunks, verb phrases, and individual words,
    ensuring all words are captured without duplication or omission.
    Phrases are returned in the order they appear in the original sentence.
    """
    doc = nlp(sentence)

    phrases = []
    current_phrase = []
    last_end = 0

    def add_phrase():
        if current_phrase:
            phrases.append(' '.join(current_phrase))
            current_phrase.clear()

    for token in doc:
        if token.i < last_end:
            continue

        if token.i > last_end:
            add_phrase()
            phrases.extend(t.text for t in doc[last_end:token.i] if not t.is_punct)

        if token.i == token.head.i:  # Root or isolated token
            add_phrase()
            if not token.is_punct:
                phrases.append(token.text)
            last_end = token.i + 1
        elif token.dep_ in ['compound', 'amod', 'det', 'nummod']:  # Part of a noun phrase
            current_phrase.append(token.text)
            last_end = token.i + 1
        elif token.head.i < token.i:  # End of a phrase
            current_phrase.append(token.text)
            add_phrase()
            last_end = token.i + 1
        else:  # Start of a new phrase
            add_phrase()
            current_phrase.append(token.text)
            last_end = token.i + 1

    add_phrase()
    if last_end < len(doc):
        phrases.extend(t.text for t in doc[last_end:] if not t.is_punct)

    return phrases


def remove_punctuation_phrases(phrases):
    """
    Removes standalone punctuation marks from a list of phrases.

    Args:
    phrases (list): A list of strings (phrases).

    Returns:
    list: A new list of phrases with standalone punctuation marks removed.
    """
    # Define a set of punctuation marks
    punctuation_marks = set(string.punctuation)

    # Filter out standalone punctuation marks
    filtered_phrases = [phrase for phrase in phrases if phrase not in punctuation_marks]

    return filtered_phrases


if __name__=="__main__":
    # Example usage:


    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "The tall buildings in the city center are very impressive.",
        "She quickly solved the complex math problem during the exam.",
        "The cat sat on the mat and looked out the window.",
        "John and Mary went to the market to buy fresh vegetables.",
        "The new smartphone features a high-resolution camera and a fast processor.",
        "A beautiful garden with colorful flowers was in the backyard.",
        "The scientist conducted an experiment to test the new hypothesis.",
        "The movie was entertaining, but the ending was quite predictable.",
        "During the hike, they saw a variety of wildlife and stunning landscapes.",
        "She wore a red dress to the party and received many compliments.",
        "The movie was great. The main actors were Tom Cruise and Nicole Kidman. I wish to see it again."
    ]

    for sentence in test_sentences:
        result = remove_punctuation_phrases(extract_phrases(sentence))
        print(f"Sentence: {sentence}")
        print(f"Extracted Phrases: {result}\n")