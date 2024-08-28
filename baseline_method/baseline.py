import sys
import os
import json

import pandas as pd 
import numpy as np
import base64 
from tqdm import tqdm

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.metrics import confusion_matrix as cm

from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F


class nnClassifier(nn.Module):
    def __init__(self, input_dim = 548, hidden_dim_1 = 1000, hidden_dim_2 = 10, *args, **kwargs,):
        super(nnClassifier, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.out_dim = 0
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim_1) 
        self.fc2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = torch.softmax(self.fc2(x), self.out_dim)
        x = self.fc2(x)
        return x
    

def fetch_data(preprocess = True):
    data = pd.read_csv('../data_dir/originalTrainDataWithGenres.csv', sep='|')
    with open('../data_dir/setOfWords.txt') as f:
        w_z = f.read()
    w_z = json.loads(w_z)
    return data, w_z

def base_64_encoding(text):
    return str(base64.b64encode(text.encode("utf-8")).decode("utf-8"))

def tokenize(data, max_source_length=512):
    tokenizer = T5Tokenizer.from_pretrained("t5-base", max_source_length=512)
    
    TokenizerEncoding = tokenizer(
        data['Text'].tolist(),
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt")
    
    return TokenizerEncoding, TokenizerEncoding['input_ids'], TokenizerEncoding['attention_mask']
    
def get_T5_embeddings(tok_for_embed):
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    tokenEmbeddingFunction = model.encoder.embed_tokens
    embed = []
    for i in range(len(tok_for_embed)):
        embed.append(tokenEmbeddingFunction(tok_for_embed[i]).clone().detach().requires_grad_(True)) 
    return embed

if __name__ == "__main__":
    device = "cpu"
    data, w_z = fetch_data()
    w = []
    z = []
    for key, val in w_z.items():
        if val == -1:
            w.append(key)
        else:
            z.append(key)
    
    tok_encoder, tok_for_embed, tok_attention = tokenize(data)
    embeddings = get_T5_embeddings(tok_for_embed)
    stacked_embeds = torch.stack(embeddings, dim=0)

    label_encoder = preprocessing.LabelEncoder() 
    y = label_encoder.fit_transform(data['imdbGenres']) 

    X_train, X_test, y_train, y_test = train_test_split(
    stacked_embeds, y, test_size=0.33, random_state=42)
    
    X_train = torch.reshape(X_train, (len(y_train), -1))
    y_train = torch.LongTensor(y_train).clone().detach()    
    
    X_test = torch.reshape(X_test, (len(y_test), -1))
    y_test = torch.LongTensor(y_test).clone().detach()
    
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32)
    
    input_dim = X_train.size()[1]
    
    model = nnClassifier(input_dim=input_dim, hidden_dim_2= len(np.unique(y)))
    model.train()
    
    n_iter = 10 # 250

    loss_list = []
    
    optimizer = optim.AdamW(model.parameters(),
                            lr=1e-4,
                            betas=(0.9, 0.999),
                            eps=1e-08,
                            weight_decay=0, #weight decay causes L2 regularization. We implement L1 regularization below
                            amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                step_size = 20, # Period of learning rate decay
                gamma = 0.8) # Multiplicative factor of learning rate decay

    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(n_iter)):
        for x, y in tqdm(dataloader):
            optimizer.zero_grad()
            y_pred = model(x)

            loss = criterion(y_pred, y)
            loss_list.append(loss)

            loss.backward(retain_graph = True)
            optimizer.step()
            scheduler.step()

    model.eval()
    predictions = []
    with torch.no_grad():
        for x in X_test:
            prediction = model(x)
            predictions.append(prediction)
    predictions = np.array(predictions)
    print(np.array(loss_list))
    print('test_loss = ', criterion(torch.from_numpy(predictions).clone().detach(), y_test))