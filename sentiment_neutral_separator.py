from utilities import get_claude_response, save_results
import json
import textwrap
from data_loaders.imdb import get_imdb_train_samples, get_imdb_test_samples

def prompt_builder(sentence):
    return textwrap.dedent(f"""
        You are given the following movie review:

        {sentence.strip()}

        Our goal is to separate this into two sets:
        "sentiment_phrases": Those words in the movie review that express or are directly related to sentiment.
        "neutral_phrases": Those words in the movie review that are factual or not directly related to sentiment.
        

        For example:
        Sentence: "The movie was great. The main actors were Tom Cruise and Nicole Kidman. I wish to see it again."
        Output:
        {{
            "sentiment_phrases": ["The movie was great.", "I wish to see it again."],
            "neutral_phrases": ["The main actors were Tom Cruise and Nicole Kidman."]
        }}

        Task: Analyze the given movie review and separate it into sentiment and neutral phrases in the given JSON format.

        IMPORTANT: Your response must be ONLY valid JSON that exactly matches the structure of the example output. Do not include any explanatory text before or after the JSON. Begin your response with the opening curly brace '{{' and end with the closing curly brace '}}'.
    """).strip()

    # Check the same for genre.


def analyze_sentiment(sentence):
    user_prompt = prompt_builder(sentence)
    return get_claude_response(user_prompt)


def process_reviews(reviews, labels, num_samples=None):
    results = []
    for i, (review, label) in enumerate(zip(reviews, labels)):
        if num_samples is not None and i >= num_samples:
            break
        analysis = analyze_sentiment(review)
        if analysis:
            try:
                parsed_analysis = json.loads(analysis)
                result = {
                    'review': review,
                    'sentiment_phrases': parsed_analysis.get('sentiment_phrases', []),
                    'neutral_phrases': parsed_analysis.get('neutral_phrases', []),
                    'label': 'positive' if label == 1 else 'negative'
                }
                results.append(result)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for review {i}")
        else:
            print(f"Failed to get a valid response for review {i}")
    return results


def main():
    # Process a subset of the training dataset
    train_reviews, train_labels = get_imdb_train_samples(n=10)
    train_results = process_reviews(train_reviews, train_labels)
    save_results(train_results, 'output/imdb_train_sentiment_analysis.json')
    print(f"Processed {len(train_results)} training reviews and saved results to imdb_train_sentiment_analysis.json")



if __name__ == "__main__":
    main()
