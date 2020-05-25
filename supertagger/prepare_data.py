"""Preprocesses data for the supertagger"""

import torch
from torchtext.data import Field, BucketIterator, TabularDataset
import pdb
from typing import List,  Tuple
#from sklearn.model_selection import train_test_split
import pandas as pd
from config import *

def create_datasets(data_path):
    sent_field = Field(lower=True)
    tag_field = Field()
    data_fields = [('sentences', sent_field), ('tags', tag_field)]
    for data_set in ["train", "val"]:
        create_csv(data_path+"/"+data_set)
    train_dataset, val_dataset = TabularDataset.splits(path=data_path, train='train.csv',
                                                            validation='val.csv', format='csv',
                                                            fields=data_fields, skip_header=True)
    sent_field.build_vocab(train_dataset, val_dataset)
    tag_field.build_vocab(train_dataset, val_dataset)

    return to_iter(train_dataset), to_iter(val_dataset), sent_field.vocab.stoi, sent_field.vocab.itos, \
           tag_field.vocab.stoi, tag_field.vocab.itos

def to_iter(dataset):
    return BucketIterator(dataset, batch_size=batch_size, sort_key=lambda x: len(x.sentences), shuffle=False)

def create_csv(data_path):
    raw_sentences = open(data_path+".words").readlines()
    raw_tag_sequences = open(data_path+".tags").readlines()
    raw_data = {'sentences': [sent.strip() for sent in raw_sentences if sent != ''],
                'tags': [tag_seq.strip() for tag_seq in raw_tag_sequences if tag_seq != '']}
    df = pd.DataFrame(raw_data, columns=["sentences", "tags"])
    df.to_csv(data_path+".csv", index=False)

def create_ix_mappings(data: List[Tuple[List, List]]):
    """data is a tuple of 2 lists and looks as follows
    (POS tags are  shown but these could be any tags including CCG supertags):
    [(["The", "dog", "ate", "the", "apple"], ["DET", "NN", "V", "DET", "NN"]),
    (["Everybody", "read", "that", "book"], ["NN", "V", "DET", "NN"])]"""
    word_to_ix = {}
    char_to_ix = {}
    tag_to_ix = {}
    for sent, tags in data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            for char in word:
                if char not in char_to_ix:
                    char_to_ix[char] = len(char_to_ix)
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    return word_to_ix, tag_to_ix, char_to_ix


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

