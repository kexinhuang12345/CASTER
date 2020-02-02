import numpy as np
import pandas as pd
import torch
from torch.utils import data

from subword_nmt.apply_bpe import BPE
import codecs

'''
DataFolder = './data'

unsup_train_file = 'food_smiles.csv' & 'drug_smiles.csv' & 'ddi_smiles.csv'
 
unsupervised pair dataset to pretrain the network
SMILES string as input

DDI supervised data files:

    train = 'train.csv'
    val = 'val.csv'
    test = 'test.csv' 
    

build a UnsupData which returns v_d, v_f for a batch

supTrainData which return v_d, v_f, label for DDI only 
supTrainData.num_of_iter_in_a_epoch contains iteration in an epoch

ValData which return v_d, v_f, label for DDI only 

'''

dataFolder = './data'

vocab_path = dataFolder + '/codes.txt'
bpe_codes_fin = codecs.open(vocab_path)
bpe = BPE(bpe_codes_fin, merges=-1, separator='')

vocab_map = pd.read_csv(dataFolder + '/subword_units_map.csv')
idx2word = vocab_map['index'].values
words2idx = dict(zip(idx2word, range(0, len(idx2word))))
max_set = 30

def smiles2index(s1, s2):
    t1 = bpe.process_line(s1).split() #split
    t2 = bpe.process_line(s2).split() #split
    i1 = [words2idx[i] for i in t1] # index
    i2 = [words2idx[i] for i in t2] # index
    return i1, i2

def index2multi_hot(i1, i2):
    v1 = np.zeros(len(idx2word),)
    v2 = np.zeros(len(idx2word),)
    v1[i1] = 1
    v2[i2] = 1
    v_d = np.maximum(v1, v2)
    return v_d

def index2single_hot(i1, i2):
    comb_index  = set(i1 + i2)
    v_f = np.zeros((max_set*2, len(idx2word)))
    for i, j in enumerate(comb_index):
        if i < max_set*2:
            v_f[i][j] = 1
        else:
            break
    return v_f

def smiles2vector(s1, s2):
    i1, i2 = smiles2index(s1, s2)
    v_d = index2multi_hot(i1, i2)
    #v_f = index2single_hot(i1, i2)
    return v_d


class supData(data.Dataset):

    def __init__(self, list_IDs, labels, df_ddi):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_ddi
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        s1 = self.df.iloc[index].Drug1_SMILES
        s2 = self.df.iloc[index].Drug2_SMILES
        v_d = smiles2vector(s1, s2)
        y = self.labels[index]
        return v_d, y
    
class supData_index(data.Dataset):

    def __init__(self, list_IDs, labels, df_ddi):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_ddi
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        s1 = self.df.iloc[index].Drug1_SMILES
        s2 = self.df.iloc[index].Drug2_SMILES
        v_d = smiles2vector(s1, s2)
        y = self.labels[index]
        return v_d, y, index

class unsupData(data.Dataset):

    def __init__(self, list_IDs, df):
        'Initialization'
        self.list_IDs = list_IDs
        self.df = df
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        index = self.list_IDs[index]
        s1 = self.df.iloc[index].input1_SMILES
        s2 = self.df.iloc[index].input2_SMILES
        v_d = smiles2vector(s1, s2)
        return v_d