import spacy
import string
from utilities import load_spacy_model
# Load the English model
nlp = load_spacy_model("en_core_web_sm")



def extract_phrases(sentence):
    """
    Extracts phrases from a given sentence using spaCy.
    Includes noun chunks, verb phrases, and individual words,
    ensuring all words are captured without unwanted merging.
    Phrases are returned in the order they appear in the original sentence.
    """
    doc = nlp(sentence)

    phrases = []
    included_words = set()

    # Helper function to add a phrase
    def add_phrase(start, end, phrase_text):
        phrases.append((start, end, phrase_text))
        included_words.update(range(start, end))

    # Extract noun chunks
    for chunk in doc.noun_chunks:
        add_phrase(chunk.start, chunk.end, chunk.text)

    # Process verbs and their direct dependents
    for token in doc:
        if token.pos_ == "VERB" and token.i not in included_words:
            verb_phrase = [token.text]
            verb_start = token.i
            verb_end = token.i + 1
            for child in token.children:
                if child.dep_ in ["dobj", "pobj", "acomp", "xcomp", "advmod", "aux", "prep"] and child.i not in included_words:
                    verb_phrase.append(child.text)
                    verb_end = max(verb_end, child.i + 1)
            add_phrase(verb_start, verb_end, " ".join(verb_phrase))

    # Add remaining words as individual phrases
    for token in doc:
        # if token.i not in included_words and not token.is_punct:
        if token.i not in included_words:
            add_phrase(token.i, token.i + 1, token.text)

    # Sort phrases by their start position and return only the text
    return [phrase for _, _, phrase in sorted(phrases)]


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
    sentence = "The quick brown fox jumps over the lazy dog."
    result = remove_punctuation_phrases(extract_phrases(sentence))
    print(f"sentence = {sentence}\nresult = {result}")
    sentence = "The movie was great. The main actors were Tom Cruise and Nicole Kidman. I wish to see it again."
    result = remove_punctuation_phrases(extract_phrases(sentence))
    print(f"sentence = {sentence}\nresult = {result}")

    test_sentences = [
        "The tall buildings in the city center are very impressive.",
        "She quickly solved the complex math problem during the exam.",
        "The cat sat on the mat and looked out the window.",
        "John and Mary went to the market to buy fresh vegetables.",
        "The new smartphone features a high-resolution camera and a fast processor.",
        "A beautiful garden with colorful flowers was in the backyard.",
        "The scientist conducted an experiment to test the new hypothesis.",
        "The movie was entertaining, but the ending was quite predictable.",
        "During the hike, they saw a variety of wildlife and stunning landscapes.",
        "She wore a red dress to the party and received many compliments."
    ]

    for sentence in test_sentences:
        result = remove_punctuation_phrases(extract_phrases(sentence))
        print(f"Sentence: {sentence}")
        print(f"Extracted Phrases: {result}\n")