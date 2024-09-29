import sys
import torch
import pandas as pd 
import numpy as np
import json
import textwrap
from random import randrange
import math
from tqdm import tqdm
import re
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
import argparse


from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

from baseline_method.baseline import tokenize, get_embeddings
from utilities.general_utilities import get_claude_response, get_claude_pre_prompt, save_results, set_seed, append_to_json
from utilities.phrase_extraction import remove_punctuation_phrases, extract_phrases
from phrase_classification import prompt_builder

def remove_non_ascii(text):
    if isinstance(text, str):
        # Remove non-ASCII characters using regex
        return re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def get_cf_data(texts, labels, cf_labels = None, ood_mode = 'genre', dataset = 'imdb', classification_word = 'review'):
    for i in tqdm(range(len(texts))): 
        if ood_mode == 'genre':
            prompt =  textwrap.dedent(f"""
            You are given the following full film review:
            "{texts[i]}"
            with the following original genres:
            {labels[i]}            
            Your task is to create a counterfactual version of this review by following these steps:

            1. Identify the key phrases or sentences that are most responsible for determining the review's genre.
            2. Modify ONLY these genre-determining parts to change the genres to be {' , '.join(cf_labels[i])}
            3. Keep all other parts of the review, including the sentences invariant of the genre, as similar as possible.
            4. Ensure that the modifications are minimal and targeted, focusing solely on changing the genres.
            5. The overall structure, length, and style of the review should remain as close to the original as possible.

            Provide the output as a CSV row with the modified film review in the first column, the new genres in the second, 
            and the sentiment of the review generated in the third.
            The columns must be separated using '|'. 
            Do not print anything else.

            IMPORTANT: 
            - Only change what is absolutely necessary to change the genres.
            - Preserve all context, details about the film, and non-genre related content.
            - The counterfactual review should read naturally and maintain the original's tone and style. 
            For example, if the original tone is sarcastic, or funny, or formal, please maintain the same tone in your answer"""
            ).strip()
            response = get_claude_response(prompt, mode='ood')
            with open('data/ood_genres_modified.csv', 'a') as file:
                if i == 0:
                    file.write('CF_Rev_Genres | CF_Genres | CF_Sentiment\n')
                file.write(response + '\n')
        
        elif ood_mode == 'sentiment':
            prompt = textwrap.dedent(f"""
            You are given the following full {classification_word}:
            "{texts.iloc[i]}"
            with the following sentiment:
            {labels.iloc[i]}
            Here, 0 denotes negative and 1 denotes positive sentiment.
            Your task is to create a counterfactual version of this {classification_word} by following these steps:

            1. Identify the key phrases or sentences that are most responsible for determining the {classification_word}'s sentiment.
            2. Modify ONLY these sentiment-determining parts to reverse the overall sentiment of the {classification_word}.
            3. Keep all other parts of the {classification_word}, including the sentences invariant of the sentiment, as similar as possible.
            4. Ensure that the modifications are minimal and targeted, focusing solely on reversing the sentiment.
            5. The overall structure, length, and style of the {classification_word} should remain as close to the original as possible.

            Provide the output as a CSV row with the modified {classification_word} in the first column and the reversed sentiment in the other 
            (Write 'Negative' if 0, 'Positive' if 1).
            
            The columns must be separated using '|'. 
            Do not print anything else.

            IMPORTANT: 
            - Only change what is absolutely necessary to reverse the sentiment.
            - Preserve all context, details about the {classification_word}, and non-sentiment related content.
            - The counterfactual {classification_word} should read naturally and maintain the original's tone and style. 
            For example, if the original tone is sarcastic, or funny, or formal, please maintain the same tone in your answer"
        """).strip()
            response = get_claude_response(prompt, mode='ood')
            with open('data/new.csv', 'a') as file:
                if i == 0:
                    file.write('CF_Rev_Sentiment | CF_Sentiment \n')
                file.write(response + '\n')
        
        else:
            prompt = textwrap.dedent(f"""
            You are given the following full {classification_word}:
            "{texts.iloc[i]}"
            with the following {ood_mode} behaviour:
            {labels.iloc[i]}
            Here, 0 denotes negative and 1 denotes positive {ood_mode} behaviour.
            Your task is to create a counterfactual version of this {classification_word} by following these steps:

            1. Identify the key phrases or sentences that are most responsible for determining the {classification_word}'s {ood_mode} behaviour.
            2. Modify ONLY these {ood_mode} behaviour-determining parts to reverse the overall {ood_mode} behaviour of the {classification_word}.
            3. Keep all other parts of the {classification_word}, including the sentences invariant of the {ood_mode} behaviour, as similar as possible.
            4. Ensure that the modifications are minimal and targeted, focusing solely on reversing the {ood_mode} behaviour.
            5. The overall structure, length, and style of the {classification_word} should remain as close to the original as possible.

            Provide the output as a CSV row with the modified {classification_word} in the first column and the reversed {ood_mode} behaviour in the other 
            (Write 'Negative' if 0, 'Positive' if 1).
            
            The columns must be separated using '|'. 
            Do not print anything else.

            IMPORTANT: 
            - Only change what is absolutely necessary to reverse the {ood_mode} behaviour.
            - Preserve all context, details about the {classification_word}, and non-{ood_mode} behaviour related content.
            - The counterfactual {classification_word} should read naturally and maintain the original's tone and style. 
            For example, if the original tone is sarcastic, or funny, or formal, please maintain the same tone in your answer"
        """).strip()
            if dataset == 'jailbreak':
                prompt = f"Jailbreaking means {classification_word} where there is a deliberate attempt to trick or manipulate the output. " + prompt
            response = get_claude_response(prompt, mode='ood')
            with open(f"../data/ood_{dataset}.csv", 'a') as file:
                if i == 0:
                    file.write(f"CF_Comment_{ood_mode} | CF_{ood_mode} \n")
                file.write(response + '\n')
        
        
def furthest_genre(genre_list, genre_embeddings):
    genres = genre_list.split(',')
    for i in range(len(genres)):
        genres[i] = genres[i][1:-1]
    
    label_embed = []
    cf_ret = 2
    for genre in genres:
        label_embed.append(genre_embeddings[genre])
    label_embed = torch.stack(label_embed, dim=0)
    label_embed = torch.mean(label_embed, dim=0)

    distances = {genre: cosine_distances([label_embed.clone().detach().numpy()], [genre_embedding.clone().detach().numpy()])[0][0]
                for genre, genre_embedding in genre_embeddings.items()} 
    
    temperature = 1.0
        
    scaled_distances= np.array([value / temperature for value in list(distances.values())])

    exp_logits = np.exp(scaled_distances) 
    probabilities = exp_logits / np.sum(exp_logits)
    
    probs, ind = {}, 0
    for genre in distances.keys():
        probs[genre] = probabilities[ind]
        ind += 1

    furthest_genres = []
    fg = np.random.choice(len(list(probs.keys())), cf_ret, replace=False, p=list(probs.values()))
    for elem in fg:
        furthest_genres.append(list(probs.keys())[elem])
    return furthest_genres

def parse_args():
    parser = argparse.ArgumentParser(description="Script for phrase classification")
    parser.add_argument('--dataset', type=str, required=False, help='imdb, jailbreak, jigsaw_toxicity, olid', default='imdb')
    parser.add_argument('--ood_mode', type=str, required=False, help='genre, sentiment, jailbreak, toxic, offensive', default='sentiment')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = "cpu"
    ood_mode = args.ood_mode
    dataset = args.dataset
    random_seed = 42
    
    if ood_mode == 'genre':
        data = pd.read_csv('data/originalTrainDataWithGenres.csv', sep='|')
        col = ['Text', 'label']

    else:
        if dataset == 'imdb':
            splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'test': 'plain_text/test-00000-of-00001.parquet', 'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'}
            data = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["test"])
            data = data.sample(n=1000, random_state=random_seed)
            data = data.map(lambda x: x.replace('"', "'") if isinstance(x, str) else x)
            data['text'] = data['text'].apply(lambda x: f'"{x}"' if isinstance(x, str) else x)
            data = data.map(remove_non_ascii)
            data['text'] = data['text'].str.replace(r'<.*?>', '', regex=True)
            data = data.replace('\n','', regex=True)
            col = ['text', 'label']
            classification_word = 'review'

        elif dataset == 'jailbreak':
            splits = {'train': 'data/0124/toxic-chat_annotation_train.csv', 'test': 'data/0124/toxic-chat_annotation_test.csv'}
            data = pd.read_csv("hf://datasets/lmsys/toxic-chat/" + splits["test"])
            col = ['user_input', 'jailbreaking']        
            data = data.loc[(data['jailbreaking'] == 1) | (data['toxicity'] == 1)]
            data = data.sample(n=5, random_state=random_seed)
            data = data.map(lambda x: x.replace('"', "'") if isinstance(x, str) else x)
            data[col[0]] = data[col[0]].apply(lambda x: f'"{x}"' if isinstance(x, str) else x)
            data = data.map(remove_non_ascii)
            data[col[0]] = data[col[0]].str.replace(r'<.*?>', '', regex=True)
            data = data.replace('\n','', regex=True)
            classification_word = 'input'

        elif dataset == 'jigsaw_toxicity':
            data = pd.read_csv('../data/toxicity_data/train.csv')
            data = data.sample(n=1000, random_state=random_seed)
            col = ['comment_text', 'toxic']
            data = data.map(lambda x: x.replace('"', "'") if isinstance(x, str) else x)
            data[col[0]] = data[col[0]].apply(lambda x: f'"{x}"' if isinstance(x, str) else x)
            data = data.map(remove_non_ascii)
            data[col[0]] = data[col[0]].str.replace(r'<.*?>', '', regex=True)
            data = data.replace('\n','', regex=True)
            classification_word = 'comment'

        elif dataset == 'olid':
            data = pd.read_csv('../data/olid_data/olid-training-v1.0.tsv', sep='\t')
            data = data.sample(n=5, random_state=random_seed)
            data = data.replace('@USER','', regex=True)
            lb = LabelEncoder() 
            col = ['tweet', 'subtask_a']
            data[col[1]] = lb.fit_transform(data[col[1]])
            data = data.map(lambda x: x.replace('"', "'") if isinstance(x, str) else x)
            data[col[0]] = data[col[0]].apply(lambda x: f'"{x}"' if isinstance(x, str) else x)
            data = data.map(remove_non_ascii)
            data[col[0]] = data[col[0]].str.replace(r'<.*?>', '', regex=True)
            data = data.replace('\n','', regex=True)
            classification_word = 'comment'
    
    tok_for_embed = tokenize(data, col=col[0])
    
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    tokenEmbeddingFunction = model.encoder.embed_tokens
    
    embeddings = get_embeddings(tok_for_embed, tokenEmbeddingFunction)
    stacked_embeds = torch.stack(embeddings, dim=0)
    sentence_stacked_embeds = torch.mean(stacked_embeds, dim=1)
    
    
    if ood_mode == 'genre':
        unique_labels = data['imdbGenres'].str.split(',').explode().unique()
        for i in range(len(unique_labels)):
            unique_labels[i] = unique_labels[i][1:-1]
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
        
    else:
        get_cf_data(data[col[0]], data[col[1]], ood_mode=ood_mode, dataset=dataset, classification_word = classification_word)
        
if __name__ == "__main__":
    main()