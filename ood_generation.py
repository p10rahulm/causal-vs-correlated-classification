from experiments.baseline_method.baseline import tokenize, get_embeddings
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_distances

import torch
import pandas as pd 
import numpy as np

def counterfactual_by_genre(sentence_embedding, genre_embeddings):
    distances = {genre: cosine_distances([sentence_embedding], [genre_embedding])[0][0]
                for genre, genre_embedding in genre_embeddings.items()}
    
    furthest_genre = max(distances, key=distances.get)
    furthest_embedding = genre_embeddings[furthest_genre]

    alpha = 0.5
    counterfactual_embedding = sentence_embedding + alpha * (furthest_embedding - sentence_embedding)
    
    return furthest_genre, counterfactual_embedding

def main():
    device = "cpu"
    data = pd.read_csv('data/originalTrainDataWithGenres.csv', sep='|')
    
    for col in ['Text', 'imdbGenres']:
        tok_for_embed = tokenize(data, col=col)
        model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
        tokenEmbeddingFunction = model.encoder.embed_tokens
        
        embeddings = get_embeddings(tok_for_embed, tokenEmbeddingFunction)
        stacked_embeds = torch.stack(embeddings, dim=0)
        
        if col == 'Text':
            sentence_stacked_embeds = torch.mean(stacked_embeds, dim=1)
        else:
            genre_stacked_embeds = torch.mean(stacked_embeds, dim=1)
    
    genre_dict = {}
    for i in  range(len(data['imdbGenres'])):
        genre_dict[data['imdbGenres'][i]] = genre_stacked_embeds[i]

    print(sentence_stacked_embeds.shape, genre_stacked_embeds.shape)
    
if __name__ == "__main__":
    main()