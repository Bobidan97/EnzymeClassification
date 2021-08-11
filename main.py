#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader, random_split
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, roc_auc_score
from utilities import (
                        default_w2i_i2w,
                        ProteinSequencesDataset,
                        BinaryClassifier,
                        load_checkpoint,
                        save_checkpoint,
                        save_list,
                        loadlist,
                        train_nn,
                        validate_nn,
                        EarlyStopping
                       )

matplotlib.style.use('ggplot')


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
    load_model = False
    

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

    if load_model:
        load_checkpoint(torch.load("checkpoint.pth"), model, optimizer)

    num_epochs     = args.epochs
    min_loss       = 100000000.0
    min_loss_epoch = 0

    #lists to store per epoch loss and accuracy values
    #lists to store ROC inputs
    epoch_loss, epoch_acc          = [], []
    epoch_val_loss, epoch_val_acc  = [], []
    y_pred_outputs, y_true_labels = [], []

    # if not using `--early_stopping`, then use simple names
    loss_plot_name = 'loss'
    acc_plot_name = 'accuracy'
    model_name = 'model'

    if args.early_stopping:
        print('INFO: Initializing early stopping')
        early_stopping = EarlyStopping()
        #change the accuracy, loss plot names and model name
        loss_plot_name = 'es_loss'
        acc_plot_name = 'es_accuracy'

    #start timer
    start = time.time()
    # loop over epochs
    for epoch in range(num_epochs):

        if epoch == 99:
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint)

        train_epoch_loss, train_epoch_accuracy = train_nn(model, train_loader, criterion, optimizer)
        val_epoch_loss, val_epoch_accuracy, confusion, report, y_pred, y_true = validate_nn(model, test_loader, criterion)

        #append train outputs to lists
        epoch_loss.append(train_epoch_loss)
        epoch_acc.append(train_epoch_accuracy)

        #append validation outputs to lists
        epoch_val_loss.append(val_epoch_loss)
        epoch_val_acc.append(val_epoch_accuracy)

        #append ROC arguments to lists
        y_pred_outputs.append(y_pred)
        y_true_labels.append(y_true)

        if args.early_stopping:
            early_stopping(val_epoch_loss, val_epoch_accuracy)
            if early_stopping.early_stop:
                break
                
        print(f"Epoch [{epoch}/{num_epochs}]    |    Average train loss: {train_epoch_loss:0.4f}    |    Average val loss: {val_epoch_loss:0.4f}    |    Average train accuracy: {train_epoch_accuracy:.2f}    |    Average val accuracy: {val_epoch_accuracy:.2f}")
        print(confusion)
        print(report)

        # compute where min loss happens -> for train loss.
        if val_epoch_loss < min_loss:
            min_loss       = val_epoch_loss
            min_loss_epoch = epoch

        #if val_epoch_accuracy > 80:
            #print(confusion)
            #print(report)

    end = time.time()

    print(f"Training Time: {(end-start)/60:.3f} minutes")
    print(f"Minimum validation loss is achieved at epoch: {min_loss_epoch}. Loss value: {min_loss:0.4f}")

    #save loss and accuracy lists
    loss_acc_list = (epoch_loss, epoch_acc, epoch_val_loss, epoch_val_acc)
    #save_list(loss_acc_list, "C:/Users/alex_/PycharmProjects/EnzymeClassification/outputs/loss_acc.npy")

    # ROC and AUC creation
    y_pred_array = y_pred_outputs.cpu().numpy().flatten()
    y_true_array = y_true_labels.cpu().numpy().flatten()
    fpr, tpr, threshold = roc_curve(y_true_array, y_pred_array)
    auc = roc_auc_score(y_true_labels, y_pred_outputs)
    print(auc)

    print('Saving loss and accuracy plots...')
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(epoch_acc, color='green', label='train accuracy')
    plt.plot(epoch_val_acc, color='blue', label='validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.savefig(f"C:/Users/alex_/PycharmProjects/EnzymeClassification/outputs/{acc_plot_name}.png")
    plt.show()

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(epoch_loss, color='orange', label='train loss')
    plt.plot(epoch_val_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.savefig(f"C:/Users/alex_/PycharmProjects/EnzymeClassification/outputs/{loss_plot_name}.png")
    plt.show()

    #ROC curve
    plt.figure(figsize=(10,7))
    plt.plot([0,1],[0,1], 'r--')
    plt.plot(fpr, tpr, label="AUC="+str(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()

    # serialize the model to disk
    print('Saving model...')
    #save = {"model_state": model.state_dict(),
            #"optim_state": optimizer.state_dict()}
    #FILE = "C:/Users/alex_/PycharmProjects/EnzymeClassification/outputs/GRU_model.pth"
    #torch.save(save, FILE)

    print('TRAINING COMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural network parameters')
    parser.add_argument('--batch_size', type=int, dest="batch_size", default=32,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, dest="epochs", default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--learning_rate', type=int, dest="learning_rate", default=0.001,
                        help='learning rate of neural network (default: 0.001')
    parser.add_argument('--num_layers', type=int, dest="num_layers", default=1,
                        help='number of layers in neural network')
    parser.add_argument('--hidden_size', type=int, dest="hidden_size", default=128,
                        help='hidden size within a single layer (default: 128)')
    parser.add_argument('--weight_decay', type=int, dest="weight_decay", default=0.0,
                        help=' L2-normalisation constant (default: 0.0)')
    parser.add_argument('--positive_set', required=True, type=str, dest="positive_set",
                        help='Positive enzyme sequence set for learning')
    parser.add_argument('--negative_set', required=True, type=str, dest="negative_set",
                        help='Negative enzyme sequence set for learning')
    parser.add_argument('--max_seq_length', required=True, type=int, dest="max_seq_length",
                        help='Maximum sequence length from dataset')
    parser.add_argument('--early_stopping', dest='early_stopping', action='store_true')
    args = parser.parse_args()
    train(args)

