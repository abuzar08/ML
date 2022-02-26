from turtle import forward
import torch
from torch import nn
from torch import optim
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
from const import *
from tqdm import tqdm

spacy_ger = spacy.load('de_core_news_sm')
spacy_eng = spacy.load('en_core_web_sm')

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenize(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenize(text)]

german = Field(tokenize_ger, lower = True, init_token = '<sos>', eos_token = '<eos>')
english = Field(tokenize_eng, lower = True, init_token = '<sos>', eos_token = '<eos>')

# dataset.splits(exts=(src, dst), fields =(src_field, dst_field))
train_data, validation_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (german, english))


german.build_vocab(train_data, max_size = 1e4, min_freq = 2)
english.build_vocab(train_data, max_size = 1e4, min_freq = 2)

class Encoder(nn.Module):
    def __init__ (self, input_size: int, embed_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, dropout = dropout)

    def forward(self, x):

        embedding = self.dropout(self.embedding(x))
        # print(x.shape)
        # print(embedding.shape)

        outputs, (hidden, cell) = self.rnn(embedding)
        # print(hidden.shape)
        # print(cell.shape)
        # exit()

        return hidden, cell


class Decoder(nn.Module):
    def __init__ (self, input_size: int, embed_size: int, hidden_size: int, output_size: int,  num_layers: int, dropout: float):
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, num_layers, dropout = dropout)

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    
    def forward(self, x, hidden, cell):

        # from (B) to (1,B)
        # print(x.shape)
        # print(hidden.shape)
        # exit()
        x = x.unsqueeze(0)

        # (1,B,embed_size)
        embedding = self.dropout(self.embedding(x))
        # print(embedding.shape)
        # exit()

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs = (1, B, hidden_size)
        # print(outputs.shape)
        # exit()

        predictions = self.fc(outputs) 
        # shape: (1,B,vocab_length) -> crossEntropyLoss

        prediction = predictions.squeeze(0)

        return prediction, (hidden, cell)

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    def forward(self, source, target, teacher_force_ratio = 0.5):
        # Sometimes use target words, sometimes use predicted words
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size)

        hidden, cell = self.encoder(source)
        # print(hidden.shape)
        # exit()
        
        # Grab start token
        x = target[0]

        for t in range(1,target_len):
            output, (hidden, cell) = self.decoder(x, hidden, cell)
            outputs[t] = output

            best_guess = output.argmax(1)

            x = best_guess if random.random() < teacher_force_ratio else target[t]
        
        return outputs


# Training!

num_epochs = 20
lr = 1e-2
batch_size = 64

load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
output_size = len(english.vocab)

num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

# tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size = batch_size,
    sort_within_batch = True,
    sort_key = lambda x: len(x.src)
)

encoder_net = Encoder(input_size=input_size_encoder,
        embed_size=encoder_embedding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout = encoder_dropout)

# encoder_net = Encoder(
#     input_size_encoder, encoder_embedding_size, hidden_size, num_layers, encoder_dropout
# )

decoder_net = Decoder(
    input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, decoder_dropout
)

model = seq2seq(
    encoder_net, decoder_net
)

pad_index = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index = pad_index)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# if load_model:
#     load_checkpoint()

for epoch in range(num_epochs):
    # print()
    batch_bar = tqdm(total=len(train_iterator), dynamic_ncols=True, leave=False, position=0, desc='Train') 
    for idx, batch in enumerate(train_iterator):
        input, target = batch.src, batch.trg

        output = model(input, target)
        # Shape: target_len, batch_size, output_dim
        #  to work with CEL - need N x output_shape , N

        output = output[1:].reshape(-1, output.shape[2]) # Rmeoving start token

        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
        # prevent exploding gradients

        optimizer.step()

        writer.add_scalar('Training loss', loss, global_step = step)
        step += 1
        batch_bar.update()
    batch_bar.close()

