#!/usr/bin/env python3

from EnzymeDataset import ProteinSequencesDataset
from EnzymeDataset import default_w2i_i2w
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader


#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size = 24
sequence_length = 28
num_layers = 1
hidden_size = 100
num_classes = 10
vocab_size = 24
learning_rate = 0.001
batch_size = 3
num_epochs = 2


#create LSTM neural network
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size, num_layers, num_classes, device):
        super(LSTMClassifier, self).__init__()
        #variables
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = device

        #Layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, bidirectional = False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, batch_of_sequences, input_sequences_lengths):
        # sort by sequence length
        batch_size = batch_of_sequences.size(0)
        sorted_lengths, sorted_idx = torch.sort(input_sequences_lengths, descending=True)
        X = batch_of_sequences[sorted_idx]

        # Convert X to one_hot
        X = self.one_hot(X)

        # packing for efficient passing through RNN
        X_packed = pack_padded_sequence(X, sorted_lengths.data.tolist(), batch_first=True)

        #Initialise hidden state and initial cell state to zero
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(device)

        #pass through LSTM
        output, _ = self.lstm(X_packed, (h0,c0))

        #Pass through linear layer to project to num classes
        output = self.fc(output[:,-1,:]) #-1 takes the last element of 2nd dimension

        return output

    def one_hot(self, batch_of_sequences):
        # Take target and convert to one-hot.
        # convert batch of sequences to one hot representation
        batch_size, items_in_seq = batch_of_sequences.size()
        one_hot = torch.zeros((batch_size, items_in_seq, self.vocab_size), dtype=torch.float, device=device)

        for i in range(batch_size):
            for j, element in enumerate(batch_of_sequences[i, :]):
                one_hot[i, j, element] = 1

        return one_hot


#Initialise Network
model = LSTMClassifier(input_size, vocab_size, hidden_size, num_layers, num_classes, device)
model.to(device)

#loss and optimiser
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)

print(model)

#define input
fastafile = "fasta_file"
max_seq_length = 237

#obtain dictionaries
w2i, i2w = default_w2i_i2w()

#construct dataset
sequence_dataset = ProteinSequencesDataset(fastafile,
                                           w2i,
                                           i2w,
                                           device,
                                           max_seq_length = max_seq_length)

#define dataloader
train_loader = DataLoader(sequence_dataset, batch_size, shuffle = False)

#Train Network
for epoch in range(num_epochs):
    #number of batch updates per epoch
    total_steps = len(train_loader)

    for batch_idx, data in enumerate(train_loader):
        input = data["input"].to(device = device)
        length = data["length"].to(device = device)
        targets = data["target"].to(device = device)

        #BEFORE RESHAPING
        #print("Before reshaping...")
        #print("Data size:", data.size())        #[batch_size, 1, 28, 28]
        #print("Targets size:", targets.size()) #[batch_size]

        #AFTER RESHAPING
        #print("After reshaping...")
        #print("Data size:", data.size())    #[batch_size, 28, 28#

        #Forward
        scores = model(input,length)  #scores of size

        #compute loss
        loss = criterion(scores, targets)

        #backward
        optimiser.zero_grad()
        loss.backward()

        #gradient descent or adam step
        optimiser.step()

        if (batch_idx+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss {:.4f}'
                  .format(epoch + 1, num_epochs, batch_idx + 1, total_steps, loss.item()))



