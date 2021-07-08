#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from EnzymeDataset import device, train_loader, test_loader, to_one_hot, w2i
from EarlyStopClass import EarlyStopping
import time
import argparse
import matplotlib

from sklearn.metrics import confusion_matrix, classification_report

#Define Hyperparameters
input_size = len(w2i)
num_layers = 1
hidden_size = 16
vocab_size = len(w2i)
learning_rate = 0.001


#create binary classification neural network
class BinaryClassifier(nn.Module):
    '''
        Decoder for OneHot representation of amino-acids.
        INPUT:
            input_size   : number of features in the input (in one-hot representation is equal to vocab-size), int
            vocab_size   : size of vocabulary, int
            hidden_size  : number of features in hidden dimension in RNN, int
            num_layers   : number of stacked RNNs, int, default: 1
            bidirectional: whether to use bidirectional RNNs, boolean, default: False

        '''

    def __init__(self, input_size, vocab_size, hidden_size, num_layers, device, bidirectional = False):
        super().__init__()

        #variables
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device


        #Layers
        self.encoder_rnn = nn.GRU(input_size,
                                   hidden_size,
                                   num_layers,
                                   batch_first = True,
                                   bidirectional = bidirectional)

        # linear layer to move from hidden_size to 1
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, batch_of_input_sequences, input_sequences_lengths, h0 = None):
        '''
            Forward pass for NN. Requires a batch of input sequences (as a list of indices!) and a batch of sequence lengths.
            One-hot conversion happens inside forward pass.
            Sequence lengths are needed for  efficient packing.
        '''

        # sort by sequence length
        batch_size = batch_of_input_sequences.size(0)
        sorted_lengths, sorted_idx = torch.sort(input_sequences_lengths, descending = True)
        X = batch_of_input_sequences[sorted_idx]

        # Convert X to one_hot
        X = self.transform(X)

        # packing for efficient passing through LSTM
        X_packed = pack_padded_sequence(X, sorted_lengths.data.tolist(), batch_first = True)

        if h0 is None:
            _, hidden = self.encoder_rnn(X_packed)
        else:
            _, hidden = self.encoder_rnn(X_packed, h0)

        # hidden is [1,batch_size,hidden_size], we don't need first dimension, so we squeeze it
        # print("hidden size before squeezing:",hidden.size())
        hidden = hidden.squeeze(0)
        # print("hidden size after squeezing:",hidden.size())

        # apply linear layer -> move from 16 dimensions to 1
        out = self.fc(hidden)

        return out

        # define transform for one hot
    def transform(self, X):
        return to_one_hot(X, self.vocab_size, self.device)

        #Initialise hidden state and initial cell state to zero
        #h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(device)
        #c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(device)

        #pass through LSTM
        #output, _ = self.encoder_rnn(X_packed, (h0, c0))

        #Pass through linear layer to project to num classes
        #output = self.fc(output[:, -1, :])  # -1 takes the last element of 2nd dimension

        #return output

# construct the argument parser e.g. while executing BinaryClassifierGRU.py use --early-stopping as command line argument
parser = argparse.ArgumentParser()
parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
args = vars(parser.parse_args())

#Initialise Network
model = BinaryClassifier(input_size,
                         vocab_size,
                         hidden_size,
                         num_layers,
                         bidirectional = False,
                         device = device)

#print(model)
model.to(device)

#loss and optimiser
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.0)


#Define classification accuracy
def binary_acc(predicted,test):
    predicted_tag = torch.round(torch.sigmoid(predicted))

    correct_results_sum = (predicted_tag == test).sum().float()
    acc = correct_results_sum/test.shape[0]
    acc = torch.round(acc*100)

    return acc

#Train Network
num_epochs = 5
torch.manual_seed(0)

#track minimum train loss
min_loss = 100000000.0
min_loss_epoch = 0

#training function
def train(model,train_loader, criterion, optimizer, binary_acc):
    model.train()
    # number of batch updates per epoch
    n_batches_per_epoch = len(train_loader)

    #Define loss
    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch_idx, data_batch in enumerate(train_loader):

        #get X and Y
        sequences = data_batch["input"]
        target_labels = data_batch["target"]
        sequences_lengths = data_batch["length"]

        #forward pass through NN
        out = model(sequences, sequences_lengths)
        out = torch.round(torch.sigmoid(out))

        #compute loss
        loss = criterion(out, target_labels)
        acc = binary_acc(out, target_labels)

        #do backpropagation
        optimizer.zero_grad()  # clean old gradients
        loss.backward()  # compute all derivatives
        optimizer.step()  # apply derivatives

        #accumulate epoch loss
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    #average epoch loss
    avg_epoch_loss = (epoch_loss / n_batches_per_epoch)

    return epoch_loss, epoch_acc, avg_epoch_loss


#validation function
def validate(model, test_loader, criterion, binary_acc):

    # evaluation mode
    model.eval()
    epoch_val_loss = 0.0
    epoch_val_acc = 0.0

    # number of batch updates per epoch
    n_batches_per_epoch = len(test_loader)

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(test_loader):

            #get X and Y
            sequences = data_batch["input"]
            target_labels = data_batch["target"]
            sequences_lengths = data_batch["length"]

            # forward pass through NN
            out = model(sequences, sequences_lengths)
            out = torch.round(torch.sigmoid(out))

            #compute loss
            loss = criterion(out, target_labels)
            acc = binary_acc(out, target_labels)

            #accumulate epoch validation loss
            epoch_val_loss += loss.item()
            epoch_val_acc += acc.item()

        #average epoch loss
        avg_val_epoch_loss = (epoch_val_loss / n_batches_per_epoch)

        return epoch_val_loss, epoch_val_acc, avg_val_epoch_loss

#lists to store per epoch loss and accuracy values
epoch_loss, epoch_acc, avg_epoch_loss = [], [], []
epoch_val_loss, epoch_val_acc, avg_val_epoch_loss = [], [], []

#start timer
start = time.time()

#training and validation loop
for epoch in range(num_epochs):

    #define outputs from train and validation functions
    train_epoch_loss, train_epoch_accuracy, train_avg_epoch_loss = train(model, train_loader, criterion, optimizer, binary_acc)
    val_epoch_loss, val_epoch_accuracy, val_avg_epoch_loss = validate(model, test_loader, criterion, binary_acc)

    #append train outputs to lists
    epoch_loss.append(train_epoch_loss)
    epoch_acc.append(train_epoch_accuracy)
    avg_epoch_loss.append(train_avg_epoch_loss)

    #append validation outputs to lists
    epoch_val_loss.append(val_epoch_loss)
    epoch_val_acc.append(val_epoch_accuracy)
    avg_val_epoch_loss.append(val_avg_epoch_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}] | Average training loss: {train_avg_epoch_loss:0.4f} | Training Accuracy: {train_epoch_accuracy:0.3f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}] | Average validation loss: {val_avg_epoch_loss:0.4f} | Validation Accuracy: {val_epoch_accuracy:0.3f}")

end = time.time()
print(f"Training Time: {(end-start)/60:.3f} minutes")
print(f"Minimum training loss is achieved at epoch: {min_loss_epoch}. Loss value: {min_loss:0.4f}")





#model_predicted_list = []
#target_labels_list = []
#out = torch.sigmoid(out)
#model_predicted = torch.round(out)
#model_predicted_list.append(model_predicted.cpu().numpy())
#target_labels_list.append(target_labels.cpu().numpy())

#append to list
#model_predicted_list = [predicted_values.squeeze().tolist() for predicted_values in model_predicted_list]
#print(model_predicted_list)

#target_labels_list = [target_values.squeeze().tolist() for target_values in target_labels_list]

#convert list of lists into one list for confusion matrix
#model_predicted_list = [item for sublist in model_predicted_list for item in sublist]
#target_labels_list = [item for sublist in target_labels_list for item in sublist]



#confusion matrix and classification report
#confusion_matrix = confusion_matrix(target_labels_list, model_predicted_list)
#print(confusion_matrix)

#print(classification_report(target_labels_list, model_predicted_list))