from experiments.baseline_method.baseline import tokenize, get_embeddings
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_distances

import sys
import torch
import pandas as pd 
import numpy as np

def furthest_genre(genre_list, genre_embeddings):
    genres = genre_list.split(',')
    for i in range(len(genres)):
        genres[i] = genres[i][1:-1]
    
    label_embed = []
    for genre in genres:
        label_embed.append(genre_embeddings[genre])
    label_embed = torch.stack(label_embed, dim=0)
    label_embed = torch.mean(label_embed, dim=0)

    distances = {genre: cosine_distances([label_embed.clone().detach().numpy()], [genre_embedding.clone().detach().numpy()])[0][0]
                for genre, genre_embedding in genre_embeddings.items()}
    
    furthest_genre = max(distances, key=distances.get)
    print(genre_list, furthest_genre)
    return furthest_genre

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
        for i in range(len(data)):
            if label in data['imdbGenres'][i]:
                label_sent_embeds.append(sentence_stacked_embeds[i])
        label_sent_embeds = torch.stack(label_sent_embeds, dim=0)
        genre_embed = torch.mean(label_sent_embeds, dim=0)
        genre_embeds[label] = genre_embed

    cf_genre_list = []
    for i in range(len(data)):
        cf_genre = furthest_genre(data['imdbGenres'][i], genre_embeds)
        cf_genre_list.append(cf_genre)
    
if __name__ == "__main__":
    main()