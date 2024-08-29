import spacy
from utilities import load_spacy_model
# Load the English model
nlp = load_spacy_model("en_core_web_sm")


def extract_phrases(sentence):
    """
    Extracts phrases from a given sentence using spaCy.
    Includes noun chunks, verb phrases, and ensures all words are captured.
    Merges adjacent single-word phrases to reduce fragmentation.
    """
    doc = nlp(sentence)

    # Initialize a list to keep track of words we've included in phrases
    included_words = set()
    phrases = []

    # Extract noun chunks
    for chunk in doc.noun_chunks:
        phrases.append(chunk.text)
        included_words.update(token.i for token in chunk)

    # Extract verb phrases
    for token in doc:
        if token.pos_ == "VERB":
            phrase = [token.text]
            phrase_tokens = {token.i}
            for child in token.subtree:
                if child.dep_ in ["dobj", "pobj", "acomp", "xcomp", "advmod", "aux", "prep"]:
                    phrase.append(child.text)
                    phrase_tokens.update(t.i for t in child.subtree)
            phrases.append(" ".join(phrase).strip())
            included_words.update(phrase_tokens)

    # Check for any words not included and add them as individual phrases
    for token in doc:
        if token.i not in included_words:
            phrases.append(token.text)
            included_words.add(token.i)

    # Merge adjacent single-word phrases
    merged_phrases = []
    current_phrase = []
    for phrase in phrases:
        if len(phrase.split()) == 1 and current_phrase:
            current_phrase.append(phrase)
        else:
            if current_phrase:
                merged_phrases.append(" ".join(current_phrase))
                current_phrase = []
            if len(phrase.split()) == 1:
                current_phrase.append(phrase)
            else:
                merged_phrases.append(phrase)

    if current_phrase:
        merged_phrases.append(" ".join(current_phrase))

    return merged_phrases


if __name__=="__main__":
    # Example usage:
    sentence = "The quick brown fox jumps over the lazy dog."
    result = extract_phrases(sentence)
    print(result)

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
        result = extract_phrases(sentence)
        print(f"Sentence: {sentence}")
        print(f"Extracted Phrases: {result}\n")