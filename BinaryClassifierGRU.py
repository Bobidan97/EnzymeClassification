#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset
from Bio import SeqIO
from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


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

#Create one-hot representation
def to_one_hot(batch_of_sequences, vocab_size, device):
    '''
    Convert list of indices to one-hot representation
    INPUT:
        batch_of_sequences : list of indices for sequences in  a batch, tensor
        vocab_size         : vocabulary size, int
    OUTPUT:
        one-hot encoding of the batch_of_sequences input
    '''

    # Take target and convert to one-hot.
    # convert batch of sequences to one hot representation
    batch_size, items_in_seq = batch_of_sequences.size()
    one_hot = torch.zeros((batch_size, items_in_seq, vocab_size), dtype = torch.float, device = device)

    for i in range(batch_size):
        for j, element in enumerate(batch_of_sequences[i, :]):
            one_hot[i, j, element] = 1

    return one_hot


class ProteinSequencesDataset(Dataset):
    '''
        Custom dataset class that works with protein sequences.
        INPUT:
            fasta_file         : FASTA file from Uniprot with protein sequences (needs to be prepared separately), string
            w2i                : map word-to-index, dictionary
            i2w                : map index-to-word, dictionary
            max_sequence_length: maximum length of protein sequence to be considered for VAE,
                                 whatever is beyond is ignored, int'''

    def __init__(self, positive_set, negative_set, w2i, i2w, device, max_sequence_length, debug):
        super().__init__()

        self.debug = debug
        self.device = device # Device
        self.max_sequence_length = max_sequence_length + 2 # to account for <eos>/<sos>

        # need to create w2i and i2w dictionaries
        self.w2i,self.i2w = w2i, i2w

        # need to construct data object
        self.data = self.__construct_data(positive_set, negative_set)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __sym2num_conversion(self, input_):

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
        return input_num

    def __construct_data(self,positive_fasta_file, negative_fasta_file):
        '''
        Explicit construction of data object that is used in __getitem__ method.
            INPUT:
                fasta_files : FASTA file from Uniprot with protein sequences (needs to be prepared separately), string
            OUTPUT:
                data       :defaultdict that has a following format:
                            data[i] = {"input"     : input array for element i,
                                       "target"    : target array for element i (0/1 label),
                                       "length"    : length of input (for sorting)
                                       "reference" : id of a sequence
                                           }
        '''
        # create a nested dictionary with default dictionary
        data = defaultdict(dict)

        #get list of sequences
        positive_records = [record for record in SeqIO.parse(positive_fasta_file, "fasta") if self.__passed_filter(record) == True]
        negative_records = [record for record in SeqIO.parse(negative_fasta_file, "fasta") if self.__passed_filter(record) == True]


        #start counter
        i = 0

        #positive sequences first -> y=1
        for record in positive_records:

            # get reference id
            reference_ = record.id

            # convert to a list
            sequence = list(record.seq)
            sequence_plus_sos = ['<sos>'] + sequence


            # obtain input and target as character arrays
            input_ = sequence_plus_sos[:self.max_sequence_length] + ['<eos>']
            target_ = torch.tensor(
                                    [1],
                                    dtype=torch.float32,  # this set to float32! Otherwise, BCEwithLogitsLoss complains!
                                    device=self.device
                                    )

            # get length
            len_ = len(input_)

            # cast to tensor
            len_ = torch.tensor(len_,
                                dtype=torch.long,
                                device=self.device
                                )

            # need to append <pad> tokens if necessary
            input_.extend(['<pad>'] * (self.max_sequence_length - len_))

            if self.debug:
                print(f"Working with sequence #{i}")
                print("BEFORE CONVERTING TO NUMBERS")
                print("Original sequence :", record.seq)
                print("Length of sequence:", len_)
                print("Input             :", input_)
                print("Target            :", target_)

            # need to convert into numerical format
            input_ = self.__sym2num_conversion(input_)

            if self.debug:
                print("AFTER CONVERTING TO NUMBERS")
                print("Input            :", input_)
                print("Target           :", target_)
                print("\n")

            # save to data: everything but reference_ is torch tensor (pushed to cpu or gpu, if available)
            data[i]["input"] = input_
            data[i]["target"] = target_
            data[i]["length"] = len_
            data[i]["reference"] = reference_

            # increment counter
            i += 1

            # negative sequences first -> y = 0
        for record in negative_records:

            # get reference id
            reference_ = record.id

            # convert to a list
            sequence = list(record.seq)
            sequence_plus_sos = ['<sos>'] + sequence

            # obtain input and target as character arrays
            input_ = sequence_plus_sos[:self.max_sequence_length] + ['<eos>']
            target_ = torch.tensor(
                                    [0],
                                    dtype=torch.float32,  # this set to float32! Otherwise, BCEwithLogitsLoss complains!
                                    device=self.device
                                    )

            # get length
            len_ = len(input_)

            # cast to tensor
            len_ = torch.tensor(len_,
                                dtype=torch.long,
                                device=self.device
                                )

            # need to append <pad> tokens if necessary
            input_.extend(['<pad>'] * (self.max_sequence_length - len_))

            if self.debug:
                print(f"Working with sequence #{i}")
                print("BEFORE CONVERTING TO NUMBERS")
                print("Original sequence :", record.seq)
                print("Length of sequence:", len_)
                print("Input             :", input_)
                print("Target            :", target_)

            # need to convert into numerical format
            input_ = self.__sym2num_conversion(input_)

            if self.debug:
                print("AFTER CONVERTING TO NUMBERS")
                print("Input            :", input_)
                print("Target           :", target_)
                print("\n")

            # save to data: everything but reference_ is torch tensor (pushed to cpu or gpu, if available)
            data[i]["input"] = input_
            data[i]["target"] = target_
            data[i]["length"] = len_
            data[i]["reference"] = reference_

            # increment counter
            i += 1

        return data


        # filter function
    def __passed_filter(self, record):
        '''
        INPUT:
            Record: Record object of BioPython module
        OUTPUT:
            True if no weird amino acids are found. else, False.
        '''
        #Obtain amino acids as a set
        set_amino_acids = set(self.w2i.keys())

        #obtain set of amino acids in a given record
        unique_set_of_amino_acids_in_ID = set(list(str(record.seq)))

        #do set difference
        set_diff = unique_set_of_amino_acids_in_ID - set_amino_acids

        #if set is emptyy, filtering criteria is passed, else not.
        if len(set_diff) == 0:
            return True
        else:
            return False

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def max_seq_len(self):
        return self.max_sequence_length

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

#################################################################################################################
################################### NEURAL NETWORK CONSTRUCTION AND TRAINING ####################################
#################################################################################################################

#define input

positive_set = "methyltransferaseEC_2.1.1.fasta" #"positive-set.fasta"
negative_set = "EC_2.3andEC_2.1.4.fasta" #"negative-set.fasta"
max_sequence_length = 6

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#obtain dictionaries
w2i, i2w = default_w2i_i2w()

#construct dataset

sequence_dataset = ProteinSequencesDataset(positive_set, negative_set,
                                           w2i,
                                           i2w,
                                           device,
                                           max_sequence_length = max_sequence_length,
                                           debug = True)

train_size = int(0.8 * len(sequence_dataset))
test_size = len(sequence_dataset) - train_size
train_dataset, test_dataset = random_split(sequence_dataset, [train_size, test_size], generator= torch.Generator().manual_seed(0))


#define dataloader and load data
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size, shuffle = True)

#print("Length:",len(dataloader))

#for batch in dataloader:
    #print(batch)

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

        #Initialise hidden state and initial cell state to zero
        #h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(device)
        #c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(device)

        #pass through LSTM
        #output, _ = self.encoder_rnn(X_packed, (h0, c0))

        #Pass through linear layer to project to num classes
        #output = self.fc(output[:, -1, :])  # -1 takes the last element of 2nd dimension

        #return output

        #define transform for one hot
    def transform(self, X):
        return to_one_hot(X, self.vocab_size, self.device)


#Initialise Network
model = BinaryClassifier(input_size,
                         vocab_size,
                         hidden_size,
                         num_layers,
                         bidirectional = False,
                         device = device)

print(model)
model.to(device)

#loss and optimiser
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.0)

#Train Network
num_epochs = 5
torch.manual_seed(0)

# track minimum train loss
min_loss       = 100000000.0
min_loss_epoch = 0

model.train()

for epoch in range(num_epochs):
    #number of batch updates per epoch
    n_batches_per_epoch = len(train_loader)
    epoch_loss = 0.0

    for batch_idx, data_batch in enumerate(train_loader):

        #get X and Y
        sequences = data_batch["input"]
        target_labels = data_batch["target"]
        sequences_lengths = data_batch["length"]

        # check sizes
        #print("X:",sequences.size())
        #print("y:",target_labels.size())

        # forward pass through NN
        out = model(sequences, sequences_lengths)
        # compute loss
        #print(out.shape)
        #print(target_labels)
        loss = criterion(out, target_labels)

        # do backpropagation
        optimizer.zero_grad()  # clean old gradients
        loss.backward()  # compute all derivatives
        optimizer.step()  # apply derivatives

        # accumulate epoch loss
        epoch_loss += loss.item()

        # print average epoch loss
    avg_epoch_loss = epoch_loss / n_batches_per_epoch
    print(f"Epoch [{epoch}/{num_epochs}], Average loss: {avg_epoch_loss:0.4f}")

    # track where minimum is achieved
    if avg_epoch_loss < min_loss:
        min_loss = avg_epoch_loss
        min_loss_epoch = epoch

print(f"Minimum training loss is achieved at epoch: {min_loss_epoch}. Loss value: {min_loss:0.4f}")

#evaluate model

def check_accuracy(Dataloader, model):
    if Dataloader.train_loader:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0.0
    num_samples = 0.0
    model.eval()

    with torch.no_grad():
        for batch_idx, data_batch in enumerate(test_loader):

            #get X and Y
            sequences = data_batch["input"]
            target_labels = data_batch["target"]
            sequences_lengths = data_batch["length"]

            # forward pass through NN
            out = model(sequences, sequences_lengths)
            _, predictions = out.max(1)
            num_correct += (predictions == target_labels).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

