import json

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

# import matplotlib

# matplotlib.style.use('ggplot')

from models.binary_head import MultiHeadBinaryModel

class BinaryDataset(Dataset):
    def __init__(self, x, y):
        super(BinaryDataset).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        features = self.x[index, :]
        labels = self.y[index, :]

        features = features.clone().detach()
        for i in range(1, self.y.size()[1] + 1):
            setattr(self, f"label{i}", labels[i - 1].clone().detach())

        ret_dict = {}
        ret_dict['features'] = features
        for i in range(1, self.y.size()[1] + 1):
            ret_dict['label' + str(i)] = getattr(self, f"label{i}")

        return ret_dict


class MultiHeadBinaryModel(nn.Module):
    def __init__(self, input_dim=393216, hidden_dim_1=1000, hidden_dim_2=10, l=37):
        super(MultiHeadBinaryModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.out_dim = 1
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim_1)
        self.fc2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.l = l

        for i in range(1, self.l + 1):
            setattr(self, f"out{i}", nn.Linear(self.hidden_dim_2, self.out_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        out_list = []
        for i in range(1, self.l + 1):
            var = getattr(self, f"out{i}")
            out_list.append(F.sigmoid(var(x)))

        return [out for out in out_list]


def binary_loss_fn(outputs, targets):
    losses = []

    for i in range(len(outputs)):
        loss = nn.BCELoss()(outputs[i].squeeze().to(torch.float32), targets[i].to(torch.float32))
        losses.append(loss)
    return torch.mean(torch.stack(losses))


def fetch_data(preprocess=True):
    data = pd.read_csv('../../data/originalTrainDataWithGenres.csv', sep='|')
    with open('../../data/setOfWords.txt') as f:
        w_z = f.read()
    w_z = json.loads(w_z)
    return data, w_z


def tokenize(data, max_source_length=512, ret_choice='input_ids', col = 'Text'):
    tokenizer = T5Tokenizer.from_pretrained("t5-base", max_source_length=512)

    TokenizerEncoding = tokenizer(
        data[col].tolist(),
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt")
    if ret_choice == 'input_ids':
        return TokenizerEncoding['input_ids']
    elif ret_choice == 'attention_mask':
        return TokenizerEncoding['attention_mask']
    else:
        return TokenizerEncoding


def get_embeddings(tok_for_embed, embedding_function):
    embed = []
    for i in range(len(tok_for_embed)):
        embed.append(embedding_function(tok_for_embed[i]).clone().detach().requires_grad_(True))
    return embed


def train(model, dataloader, optimizer, train_dataset, device='cpu', l=37):
    model.train()
    counter = 0
    train_running_loss = 0.0

    for i, data in tqdm(enumerate(dataloader), total=int(len(train_dataset) / dataloader.batch_size)):
        counter += 1
        features = data['features'].to(device)
        targets = []
        for j in range(1, l + 1):
            targets.append(data['label' + str(j)])
        optimizer.zero_grad()
        outputs = model(features)
        loss = binary_loss_fn(outputs, targets)
        train_running_loss += loss
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / counter
    return train_loss


if __name__ == "__main__":
    device = "cpu"
    data, w_z = fetch_data()
    # w = []
    # z = []
    # for key, val in w_z.items():
    #     if val == -1:
    #         w.append(key)
    #     else:
    #         z.append(key)
    #

    # Obtaining the Embeddings
    tok_for_embed = tokenize(data)
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    tokenEmbeddingFunction = model.encoder.embed_tokens
    embeddings = get_embeddings(tok_for_embed, tokenEmbeddingFunction)
    stacked_embeds = torch.stack(embeddings, dim=0)
    stacked_embeds = torch.mean(stacked_embeds, dim=1)

    # Encoding the multi-label outputs
    encoder = MultiLabelBinarizer()
    y = encoder.fit_transform(data['imdbGenres'])

    # Creating the Training and Testing Data
    X_train, X_test, y_train, y_test = train_test_split(
        stacked_embeds, y, test_size=0.33, random_state=42)
    X_train = torch.reshape(X_train, (len(y_train), -1))
    y_train = torch.LongTensor(y_train).clone().detach()
    X_test = torch.reshape(X_test, (len(y_test), -1))
    y_test = torch.LongTensor(y_test).clone().detach()

    # Putting the data in the format required for multi-label classification
    train_dataset = BinaryDataset(X_train, y_train)
    dataloader = DataLoader(train_dataset, batch_size=64)
    model = MultiHeadBinaryModel()

    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    epochs = 100
    model.to(device)

    train_loss = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss = train(model, dataloader, optimizer, train_dataset, device)
        train_loss.append(train_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
    # torch.save(model.state_dict(), 'models/multi_head_binary.pth')
