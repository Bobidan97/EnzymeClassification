#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset
from Bio import SeqIO
from collections import defaultdict

amino_acids = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']

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


class ProteinSequencesDataset(Dataset):

    def __init__(self, fasta_file, w2i, i2w, device, max_seq_length=500, one_hot=True):
        super().__init__()

        self.device = device
        self.max_seq_length = max_seq_length + 1
        self.one_hot = one_hot

        self.w2i = w2i
        self.i2w = i2w

        self.data = self.__construct_data(fasta_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __sym2num_conversion(self, input_, target_):

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

    def __sym2one_hot_conversion(self,input_,target_):

        input_num  = torch.zeros((len(input_),len(amino_acids)+len(special_tokens)),
                             dtype  = torch.long,
                             device = self.device
                                 )
        for i, element in enumerate(input_):
            input_num[i][self.w2i[element]] = 1

        target_num = torch.tensor(
                                    [self.w2i.get(element, self.w2i['<unk>']) for element in target_],
                                    dtype = torch.long,
                                    device = self.device
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

            print(f"Working with sequences #{i}")
            print("BEFORE CONVERTING TO NUMBERS")
            print("Original sequence:", record.seq)
            print("Length of sequence:", len_)
            print("Input: ", input_)
            print("Target: ", target_)

            # need to convert into numerical format
            if self.one_hot:
                input_, target_ = self.__sym2one_hot_conversion(input_, target_)
            else:
                input_, target_ = self.__sym2num_conversion(input_, target_)

            print("AFTER CONVERTING TO NUMBERS")
            print("Input: ", input_)
            print("Target: ", target_)

            print("\n")

            #Save to data
            data[i]["input"] = input_
            data[i]["target"] = target_
            data[i]["length"] = len_
            data[i]["reference"] = reference_

        return data

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def max_seq_len(self):
        return self.max_seq_length

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

    #filter function
    def __passed_filter(self, record):

        set_amino_acids = set(self.w2i.keys())

        unique_set_of_amino_acids_in_ID = set(list(str(record.seq)))

        set_diff = unique_set_of_amino_acids_in_ID - set_amino_acids

        if len(set_diff) == 0:
            return True
        else:
            return False

if __name__ == '__main__':
    torch.set_printoptions(profile="full")
    w2i,i2w=default_w2i_i2w()
    dataset = ProteinSequencesDataset("fasta_file",w2i,i2w,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    data = dataset.data
    print(dataset.data[0]["input"])
