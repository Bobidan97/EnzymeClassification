#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
# here I import everything I need from utilities.py
from utilities import (
                        default_w2i_i2w,
                        ProteinSequencesDataset,
                        BinaryClassifier,
                        train_nn,
                        validate_nn
                      )

def main():
    parser = argparse.ArgumentParser(description='Neural network parameters')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--learning_rate', type=int, default=0.001,
                        help='learning rate of neural network (default: 0.001')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in neural network')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='hidden size within a single layer (default: 128)')
    parser.add_argument('--weight_decay', type=int, default=0.0,
                        help=' L2-normalisation constant (default: 0.0)')
    parser.add_argument('--positive_set', required=True, type=str,
                        help='Positive enzyme sequence set for learning')
    parser.add_argument('--negative_set', required=True, type=str,
                        help='Negative enzyme sequence set for learning')
    parser.add_argument('--max_seq_length', required=True, type=int,
                        help='Maximum sequence length from dataset')
    args = parser.parse_args()
    train(args)

def train(args):
    # set seed
    torch.manual_seed(0)

    # STEP 1: create train/validation dataset
    positive_set        = args.positive_set #"methyltransferaseEC_2.1.1.fasta"
    negative_set        = args.negative_set #"EC_2.3andEC_2.1.4.fasta"

    #positive_set        = "positive-set.fasta"
    #negative_set        = "negative-set.fasta"

    max_sequence_length = args.max_seq_length
    device              = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    w2i, i2w            = default_w2i_i2w()

    # here I instantiate object. Note that my class is written elsewhere, here I am just using it.
    # it saves space and make code easier to read (i.e. there is a mistake, I will go to utilities and make changes in ProteinSequencesDataset).
    sequence_dataset = ProteinSequencesDataset(
                                                positive_set, 
                                                negative_set,
                                                w2i,
                                                i2w,
                                                device,
                                                max_sequence_length = max_sequence_length,
                                                debug               = False
                                              )
    # split the dataset into train/validation
    train_size                  = int(0.8 * len(sequence_dataset))
    test_size                   = len(sequence_dataset) - train_size
    train_dataset, test_dataset = random_split(sequence_dataset, [train_size, test_size], generator= torch.Generator().manual_seed(0))

    # wrap datasets into Dataloaders
    batch_size   = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
    test_loader  = DataLoader(test_dataset, batch_size, shuffle = True)

    # STEP 2: define model
    input_size    = len(w2i)
    num_layers    = args.num_layers
    hidden_size   = args.hidden_size
    vocab_size    = len(w2i)
    learning_rate = args.learning_rate
    
    # again I am creating a BinaryClassifier object, 
    # the definition of BinaryClassifier happens in utilities
    model = BinaryClassifier(
                                input_size,
                                vocab_size,
                                hidden_size,
                                num_layers,
                                bidirectional = False,
                                device        = device
                            )
    model.to(device)

    #loss and optimiser
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = args.weight_decay)
    
    num_epochs     = args.epochs
    min_loss       = 100000000.0
    min_loss_epoch = 0

    #lists to store per epoch loss and accuracy values
    epoch_loss, epoch_acc          = [], []
    epoch_val_loss, epoch_val_acc  = [], []

    #start timer
    start = time.time()

    # loop over epochs
    for epoch in range(num_epochs):
        
        train_epoch_loss, train_epoch_accuracy = train_nn(model, train_loader, criterion, optimizer)
        val_epoch_loss, val_epoch_accuracy     = validate_nn(model, test_loader, criterion)

        #append train outputs to lists
        epoch_loss.append(train_epoch_loss)
        epoch_acc.append(train_epoch_accuracy)

        #append validation outputs to lists
        epoch_val_loss.append(val_epoch_loss)
        epoch_val_acc.append(val_epoch_accuracy)

        print(f"Epoch [{epoch}/{num_epochs}]    |    Average train loss: {train_epoch_loss:0.4f}    |    Average val loss: {val_epoch_loss:0.4f}    |    Average train accuracy: {train_epoch_accuracy:.2f}    |    Average val accuracy: {val_epoch_accuracy:.2f}")

        # compute where min loss happens -> for train loss!
        if train_epoch_loss < min_loss:
            min_loss       = train_epoch_loss
            min_loss_epoch = epoch

    end = time.time()

    print(f"Training Time: {(end-start)/60:.3f} minutes")
    print(f"Minimum training loss is achieved at epoch: {min_loss_epoch}. Loss value: {min_loss:0.4f}")

if __name__ == '__main__':
    main()
    train()


    # here I call train(): this means that when in the command line you type: python3 main.py -> train() is called.
    '''
    Your tasks:
    1. Determine which values need to be parameterized (for example, number of epochs) and make those values as parameters.
        I.E. in case of number of epochs you would call a function like: train(num_epochs = 5).
        You also need to think how to provide parameters to train() function. You can try argparse (I saw you tried using it) or typer (https://typer.tiangolo.com/tutorial/first-steps/).
    2. You need to implement how to determine when to stop training.
        You can try EarlyStopping (as you tried but did not finish) or you can train until very end and then determine which epoch gave you the best desired result.
        Obviously, for the latter you would need to save model parameters after each epoch.
    3. Create nice plots.
    4. Save model parameters (check https://pytorch.org/tutorials/beginner/saving_loading_models.html)

    Please, check the code and its comments. You should find some useful information in them.
    Also, I ran the model on toy dataset to show that I can achieve 100% accuracy on validation data.
    If you run the code as provided you should be able to see similar results.
    '''

