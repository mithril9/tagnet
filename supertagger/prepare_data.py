"""Preprocesses data for the supertagger"""

import torch
from torchtext.data import Field, BucketIterator, Iterator, TabularDataset
import pdb
import os
from typing import List,  Tuple
#from sklearn.model_selection import train_test_split
import pandas as pd
from config import *

def create_datasets(data_path):
    sent_field = Field(lower=True)
    tag_field = Field()
    data_fields = [('sentence', sent_field), ('tags', tag_field)]
    for data_set in ["train", "val"]:
        create_csv(data_path+"/"+data_set)
    train_dataset, val_dataset = TabularDataset.splits(path=data_path, train='train.csv',
                                                            validation='val.csv', format='csv',
                                                            fields=data_fields, skip_header=True)
    #build the vocab over the train set only
    sent_field.build_vocab(train_dataset)
    tag_field.build_vocab(train_dataset)
    char_to_ix = get_char_to_ix(train_dataset)
    train_iter = to_iter(train_dataset)
    val_iter = to_iter(val_dataset)
    return train_iter, val_iter, sent_field.vocab.stoi, sent_field.vocab.itos, \
           tag_field.vocab.stoi, tag_field.vocab.itos, char_to_ix

def get_char_to_ix(dataset):
    #we will create a dummy csv and dataset so that we end up with a Vocab object that we can use as our char_to_ix map
    char_field = Field(lower=True)
    field_list = [('chars', char_field)]
    char_set = set({})
    for example in dataset:
        for word in example.sentence:
            for char in word:
                if char not in char_set:
                    char_set.add(char)
    char_list = sorted(list(char_set))
    chars = " ".join(char_list)
    df = pd.DataFrame({'chars':[chars]}, columns=["chars"])
    df.to_csv('chars.csv', index=False)
    dummy_dataset = TabularDataset(path="./chars.csv", format='csv', fields=field_list, skip_header=True)
    char_field.build_vocab(dummy_dataset)
    os.remove('chars.csv')
    return char_field.vocab.stoi


def to_iter(dataset, bucket=True):
    if bucket:
        #sort_within_batch is used for when you want to "pack_padded_sequence with the padded sequence data and \
        #convert the padded sequence tensor to a PackedSequence object" (A Comprehesive Introduction to Torchtext)
        data_iter = BucketIterator(dataset, sort=True, batch_size=batch_size, device=-1, sort_within_batch=False, sort_key=lambda x: len(x.sentence), shuffle=False)
    else:
        data_iter = Iterator(dataset, batch_size=batch_size, device=-1, sort=False, sort_within_batch=False, repeat=False)
    data_iter.sent_lengths = []
    for batch in data_iter:
        batch_sent_len = batch.sentence[:,0].shape[0]
        lengths = []
        for i in range(batch.sentence.shape[1]):
            #must subtract the number of pads from the length
            lengths.append((batch_sent_len - (batch.sentence[:,i] == 1).sum(dim=0)).item())
        data_iter.sent_lengths.append(lengths)
    return data_iter


def create_csv(data_path):
    raw_sentences = open(data_path+".words").readlines()
    raw_tag_sequences = open(data_path+".tags").readlines()
    raw_data = {'sentence': [sent.strip() for sent in raw_sentences if sent != ''],
                'tags': [tag_seq.strip() for tag_seq in raw_tag_sequences if tag_seq != '']}
    df = pd.DataFrame(raw_data, columns=["sentence", "tags"])
    df.to_csv(data_path+".csv", index=False)

def create_char_ix_mappings(train_dataset):
    char_to_ix = {}
    for example in train_dataset:
        for word in example.sentence:
            for char in  word:
                if char not in char_to_ix:
                    char_to_ix[char] = len(char_to_ix)
    return char_to_ix


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def get_words_in(sentences_in, char_to_ix, ix_to_word):
    words_in = []
    for i in range(sentences_in.shape[0]):
        words_in.append([prepare_sequence(word, char_to_ix) for word in [ix_to_word[ix] for ix in sentences_in[i, :]]])
        words_in[-1] = batchify_sent(words_in[-1])
    return words_in

def batchify_sent(sent):
    #we need to pad the words in the sentence so that they all have the same number of characters
    max_len = 0
    for word in sent:
        if word.shape[0] > max_len:
            max_len = word.shape[0]
    bm = False
    for i in range(len(sent)):
        #create a tensor of ones (1 is the pad index) and then fill it in with values from the source tensor
        sent_tensor = torch.ones(1, max_len, dtype=torch.long)
        sent_tensor[0,:sent[i].shape[0]] = sent[i]
        if bm:
            batch_matrix = torch.cat((batch_matrix, sent_tensor), dim=0)
        else:
            bm = True
            batch_matrix = sent_tensor
    return batch_matrix


