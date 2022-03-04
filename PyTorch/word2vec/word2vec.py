from doctest import Example
import numpy as np
import torch
from torch import nn
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

class encoder(nn.Module):
    def __init__(self, in_size: int, embed_size: int):
        super().__init__()

        self.in_size = self.out_size = in_size
        self.h_size = embed_size
        
        self.l1 = nn.Linear(self.in_size, self.h_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(self.h_size, self.out_size)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # out = nn.Softmax(out)

        return out

def pre_process(file_name):
    f = open(file_name, 'r')
    lines = f.readlines()

    stopWords = {'a', 'an', 'the', 'it', 'will', 'be', 'is', 'the', 'and'}
    allWords = set()
    sentences = []
    for line in lines:
        # line = re.sub(r"[^\S\w]", "", line) # Remove Punctuation
        line = line.lower()
        words = line.split()
        sentence = [word for word in words if word not in stopWords]
        sentences.append(sentence)
        for word in words:
            if word not in stopWords:
                allWords.add(word)
    
    indices = {word:i for i,word in enumerate(allWords)}
    OHE = torch.eye(len(indices))

    return sentences, indices, OHE


def make_pairs(sentences):
    # Defining the window for context
    window = 3

    # Creating a placeholder for the scanning of the word list
    word_lists = []

    for text in sentences:

        # Creating a context dictionary
        for i, word in enumerate(text):
            for w in range(window):
                # Getting the context that is ahead by *window* words
                if i + 1 + w < len(text): 
                    word_lists.append([word] + [text[(i + 1 + w)]])
                # Getting the context that is behind by *window* words    
                if i - w - 1 >= 0:
                    word_lists.append([word] + [text[(i - w - 1)]])
    
    return word_lists

def make_pair_OHE(pairs, OHE, indices):
    X = []
    # Y = torch.zeros((len(pairs), OHE.shape[1]))
    for i,pair in enumerate(pairs):
        [w1, w2] = pair
        v1 = OHE[indices[w1]]
        v2 = OHE[indices[w2]]
        # print(v1, v2)
        X.append([v1,v2])
    
    return X

def main():

    file_name = "/Users/abu/Desktop/CMU/CodingPractice/PyTorch/word2vec/data.txt"
    sentences, indices, OHE = pre_process(file_name)
    # print(sentences)
    pairs = make_pairs(sentences)
    X = make_pair_OHE(pairs, OHE, indices)

    train_loader = torch.utils.data.DataLoader(dataset = X, batch_size=256,
    shuffle= True)

    epochs = 1000
    model = encoder(len(indices), 2)
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs)):
        for i, (x,y) in enumerate(train_loader):
            # x, y = torch.from_numpy(x), torch.from_numpy(y)
            out = model(x)

            loss = criterion(out, y)

            # backward
            loss.backward()

            # update
            optimizer.step()

            optimizer.zero_grad()
    
    embeddings = model.l1.weight
    embeddings = embeddings.T
    embedding_dict = {}
    for word in indices:
        embedding_dict[word] = embeddings[indices[word]]
    
    with torch.no_grad():
        plt.figure(figsize=(10, 10))
        for word in list(indices.keys()):
            coord = embedding_dict.get(word)
            plt.scatter(coord[0], coord[1])
            plt.annotate(word, (coord[0], coord[1]))
        
        plt.show()
    
main()