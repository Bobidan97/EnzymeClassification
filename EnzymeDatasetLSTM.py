#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset
from Bio import SeqIO
from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

###############################ENZYME DATASET CONSTRUCTION####################################

#Vocab list used later in script
#amino_acids = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
#special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']


def default_w2i_i2w():
    '''Constructs default maps that can be passed to ProteinSequencesDataset.'''
    w2i = dict() #maps word i.e amino acids into index
    i2w = dict() #maps index into word i.e amino acids

    amino_acids = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V'] #Vocabulary of the 20 amino acids
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']

    for w in amino_acids:
        i2w[len(w2i)] = w
        w2i[w] = len(w2i)

    for st in special_tokens:
        i2w[len(w2i)] = st
        w2i[st] = len(w2i)

    return w2i, i2w

class ProteinSequencesDataset(Dataset):
    '''
        Custom dataset class that works with protein sequences.
        INPUT:
            fasta_file         : FASTA file from Uniprot with protein sequences (needs to be prepared separately), string
            w2i                : map word-to-index, dictionary
            i2w                : map index-to-word, dictionary
            max_sequence_length: maximum length of protein sequence to be considered for VAE,
                                 whatever is beyond is ignored, int'''

    def __init__(self, fasta_file, w2i, i2w, device, max_seq_length=500):
        super().__init__()

        self.device = device
        self.max_seq_length = max_seq_length + 1 # to account for <eos>/<sos>

        # need to create w2i and i2w dictionaries
        self.w2i = w2i
        self.i2w = i2w
        #print(len(w2i)) # Just a check of dictionary length

        # need to construct data object
        self.data = self.__construct_data(fasta_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __sym2num_conversion(self, input_, target_):

        '''
        Conversion of string array into numeric array. Needed if we use embedding layers.
        Conversion is the SAME for input_ and target_
        EX.: ['<sos>','M','Q','H'] -> [2,4,7,10]

        INPUT:
        input_ : Input array of strings, list of strings
        target_: Next element predictions for the input_, list of strings

        OUTPUT:
        input_num, target_num
        '''

        input_num = torch.tensor(
            [self.w2i.get(element, self.w2i['<unk>']) for element in input_],
            dtype=torch.long,
            device=self.device
        )

        target_num = torch.tensor(
            [self.w2i.get(element, self.w2i['<unk>']) for element in target_],
            dtype=torch.long,
            device=self.device
        )
        return input_num,target_num

    def __construct_data(self,fasta_file):
        data = defaultdict(dict)

        records = [record for record in SeqIO.parse(fasta_file, "fasta") if self.__passed_filter(record) == True]

        for i, record in enumerate(records):

            # get reference id
            reference_ = record.id

            # convert to a list
            sequence = list(record.seq)
            sequence_plus_sos = ['<sos>'] + sequence

            # obtain input and target as character arrays
            input_ = sequence_plus_sos[:self.max_seq_length]
            target_ = sequence[:self.max_seq_length - 1] + ['<eos>']
            assert len(input_) == len(target_), "Length mismatch"
            len_ = len(input_)

            # cast to tensor
            len_ = torch.tensor(len_,
                                dtype=torch.long,
                                device=self.device
                                )

            # need to append <pad> tokens if necessary
            input_.extend(['<pad>'] * (self.max_seq_length - len_))
            target_.extend(['<pad>'] * (self.max_seq_length - len_))

            #print(f"Working with sequences #{i}")
            #print("BEFORE CONVERTING TO NUMBERS")
            #print("Original sequence:", record.seq)
            #print("Length of sequence:", len_)
            #print("Input: ", input_)
            #print("Target: ", target_)

            input_, target_ = self.__sym2num_conversion(input_, target_)

            #print("AFTER CONVERTING TO NUMBERS")
            #print("Input: ", input_)
            #print("Target: ", target_)

            #print("\n")

            # Save to data
            data[i]["input"] = input_
            data[i]["target"] = target_
            data[i]["length"] = len_
            data[i]["reference"] = reference_

        return data

        # filter function
    def __passed_filter(self, record):

        set_amino_acids = set(self.w2i.keys())
        unique_set_of_amino_acids_in_ID = set(list(str(record.seq)))
        set_diff = unique_set_of_amino_acids_in_ID - set_amino_acids

        if len(set_diff) == 0:
            return True
        else:
            return False

if __name__ == '__main__':
    #torch.set_printoptions(profile="full")
    w2i,i2w=default_w2i_i2w()
    dataset = ProteinSequencesDataset("fasta_file",w2i,i2w,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    data = dataset.data
    #print(dataset.data[0]["input"])



###################################NEURAL NETWORK CONSTRUCTION AND TRAINING####################################

#define input
fastafile = "fasta_file"
max_seq_length = 237
#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#obtain dictionaries
w2i, i2w = default_w2i_i2w()

#construct dataset
sequence_dataset = ProteinSequencesDataset(fastafile,
                                           w2i,
                                           i2w,
                                           device,
                                           max_seq_length = max_seq_length)

print(sequence_dataset[0])

#define dataloader
batch_size = 3 # 8 sequences in example fasta set so should get 3 batches of sizes 3, 3, and 2.
train_loader = DataLoader(sequence_dataset, batch_size, shuffle = False)

for batch in train_loader:
    print(batch)

#Define Hyperparameters
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
        '''
            Convert list of indices to one-hot representation
            INPUT:
                batch_of_sequences : list of indices for sequences in  a batch, tensor
                vocab_size         : vocabular size, int
            OUTPUT:
                one-hot encoding of the batch_of_sequences input
        '''

        # Take target and convert to one-hot.
        # convert batch of sequences to one hot representation
        batch_size, items_in_seq = batch_of_sequences.size()
        one_hot = torch.zeros((batch_size, items_in_seq, self.vocab_size), dtype=torch.float, device=device)

        for i in range(batch_size):
            for j, element in enumerate(batch_of_sequences[i, :]):
                one_hot[i, j, element] = 1

        return one_hot

# now loop over one more time
for batch in train_loader:
    X = batch["target"]
    X_one_hot = one_hot(X,len(w2i),device)
    print("Original target       :",X)
    print("Target in one-hot form:",X_one_hot)
    print("\n")

#Initialise Network
model = LSTMClassifier(input_size, vocab_size, hidden_size, num_layers, num_classes, device)
model.to(device)

#loss and optimiser
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)

#Train Network
for epoch in range(num_epochs):
    #number of batch updates per epoch
    total_steps = len(train_loader)

    for batch_idx, data in enumerate(train_loader):
        input = data["input"].to(device = device)
        length = data["length"].to(device = device)
        targets = data["target"].to(device = device)

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
