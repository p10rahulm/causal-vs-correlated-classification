import sys
import torch
import pandas as pd 
import numpy as np
import json
import textwrap
from random import randrange
import math

from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from experiments.baseline_method.baseline import tokenize, get_embeddings
from utilities import get_claude_response, get_claude_pre_prompt, save_results, set_seed, append_to_json
from phrase_extraction import remove_punctuation_phrases, extract_phrases
from phrase_classification import prompt_builder

def get_cf_data(texts, labels, cf_labels, output_file=None):
    for i in range(len(texts)):        
        prompt =  textwrap.dedent(f"""
                You are given the following full film review:
                "{texts[i]}"
                with the following original genres:
                {labels[i]}
                Change the given film review in a manner such that its genre is changed to the following counterfactual instead:
                {cf_labels[i]}
                Provide the output in a JSON file with two keys, one for the counterfactual text, and the other with the counterfactual genres.
                IMPORTANT: Your response must be ONLY valid JSON that matches the structure described. Do not include any explanatory text before or after the JSON. Begin your response with the opening curly brace '{{' and end with the closing curly brace '}}'.
            """).strip()
        response = get_claude_response(prompt)
        print(response)
        sys.exit()

def furthest_genre(genre_list, genre_embeddings):
    genres = genre_list.split(',')
    for i in range(len(genres)):
        genres[i] = genres[i][1:-1]
    
    label_embed = []
    cf_ret = len(genres)
    for genre in genres:
        label_embed.append(genre_embeddings[genre])
    label_embed = torch.stack(label_embed, dim=0)
    label_embed = torch.mean(label_embed, dim=0)

    distances = {genre: cosine_distances([label_embed.clone().detach().numpy()], [genre_embedding.clone().detach().numpy()])[0][0]
                for genre, genre_embedding in genre_embeddings.items()} 
    
    temperature = 1.0
    scaled_distances= np.array([value / temperature for value in list(distances.values())])

    exp_logits = np.exp(scaled_distances - np.max(scaled_distances))  
    probabilities = exp_logits / np.sum(exp_logits)
    print(probabilities)
    sys.exit()
    furthest_genres = sorted(distances, key=distances.get, reverse=True)[:cf_ret]
    return furthest_genres

def main():
    device = "cpu"
    data = pd.read_csv('data/originalTrainDataWithGenres.csv', sep='|')

    unique_labels = data['imdbGenres'].str.split(',').explode().unique()
    for i in range(len(unique_labels)):
        unique_labels[i] = unique_labels[i][1:-1]

    col = 'Text'
    tok_for_embed = tokenize(data, col=col)
    
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    tokenEmbeddingFunction = model.encoder.embed_tokens
    
    embeddings = get_embeddings(tok_for_embed, tokenEmbeddingFunction)
    stacked_embeds = torch.stack(embeddings, dim=0)
    sentence_stacked_embeds = torch.mean(stacked_embeds, dim=1)
    genre_embeds = {}
    for label in unique_labels:
        label_sent_embeds = []
        label_cnt = 0
        for i in range(len(data)):
            if label in data['imdbGenres'][i]:
                label_cnt += 1
                label_sent_embeds.append(sentence_stacked_embeds[i])
        label_sent_embeds = torch.stack(label_sent_embeds, dim=0)
        genre_embed = torch.mean(label_sent_embeds, dim=0)
        genre_embeds[label] = genre_embed
    cf_genre_list = []
    for i in range(len(data)):
        cf_genre = furthest_genre(data['imdbGenres'][i], genre_embeds)        
        cf_genre_list.append(cf_genre)
    
    get_cf_data(data['Text'], data['imdbGenres'], cf_genre_list)
if __name__ == "__main__":
    main()