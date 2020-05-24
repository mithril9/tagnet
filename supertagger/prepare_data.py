"""Preprocesses data for the supertagger"""

import torch
from torchtext.data import Field, BucketIterator, TabularDataset
import pdb
from typing import List,  Tuple
from sklearn.model_selection import train_test_split
import pandas as pd

def data_to_tuples(word_file_name, tag_file_name):
    sent_field = Field(lower=True)
    tag_field = Field()
    raw_sentences = open(word_file_name).readlines()
    raw_tag_sequences = open(tag_file_name).readlines()
    raw_data = {'sentences': [sent.strip() for sent in raw_sentences if sent != ''],
                'tags': [tag_seq.strip() for tag_seq in raw_tag_sequences if tag_seq != '']}
    #assert len(raw_data['sentences']) == len(raw_data['tags'])
    #for i in range(len(raw_data['sentences'])):
        #assert len(raw_data['sentences'][i]) == len(raw_data['tags'][i])
    df = pd.DataFrame(raw_data, columns=["sentences", "tags"])
    train, val = train_test_split(df, train_size=0.75)
    train.to_csv("train.csv", index=False)
    val.to_csv("val.csv", index=False)
    data_fields = [('sentences', sent_field), ('tags', tag_field)]
    train, val = TabularDataset.splits(path='./', train='train.csv', validation='val.csv', format='csv',
                                            fields=data_fields)
    sent_field.build_vocab(train, val)
    tag_field.build_vocab(train, val)
    pdb.set_trace()
    #return data

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

