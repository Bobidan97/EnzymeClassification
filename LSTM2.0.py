#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sequence):
        embeds = self.word_embeddings(sequence)
        lstm_out, _ = self.lstm(embeds.view(len(sequence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sequence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores


model = LSTMTClassifier(EMBEDDING_DIM, HIDDEN_DIM, len(w2i), len(i2w))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], w2i)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(3):
    for sentence, tags in training_data:

        model.zero_grad()

        sentence_in = prepare_sequence(sentence, w2i)
        targets = prepare_sequence(tags, i2w)

        tag_scores = model(sentence_in)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], w2i)
    tag_scores = model(inputs)

    print(tag_scores)