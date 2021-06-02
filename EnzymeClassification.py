#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset
import numpy as np

def default_w2i_i2w():
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

class protein_sequences_dataset(Dataset):

        def __init__(self, sequences_file, w2i, i2w, max_seq_length = 500, one_hot = False):
            super().__init__()

            self.one_hot = one_hot_encoding
            self.max_seq_length = max_seq_length +1

            self.w2i, self.i2w = w2i_dict, i2w_dict

            self.device = device

            self.data = self.__construct_data(sequences_file)

        def __len__(self):
            return len(self.data)


        def __getitem__(self, idx):
            return self.data[idx]

        def __sym2one_hot_conversion(self,input_,target_):

            input_num  = torch.zeros((len(input_)),
                                 dtype  = torch.long,
                                 device = self.device
                                     )
            for i, element in enumerate(input_):
                input_num[i, self.w2i[element]] = 1

            target_num  = torch.tensor(
                                        [self.w2i.get(element,self.w2i['<unk>']) for element in target_],
                                        dtype  = torch.long,
                                        device = self.device
                                    )

            return input_num,target_num

        def __construct_data(self, sequences_file):