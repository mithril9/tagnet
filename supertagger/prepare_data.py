"""Preprocesses data for the supertagger"""

import torch
import pdb
from typing import List,  Tuple

def data_to_tuples(word_file_name, tag_file_name):

    sentences = open(word_file_name).readlines()
    tag_sequences = open(tag_file_name).readlines()
    while '' in sentences:
        sentences.remove('')
    while '' in tag_sequences:
        tag_sequences.remove('')
    data = [([word for word in sentences[i].strip().split()], [tag for tag in tag_sequences[i].strip().split()]) for i in range(len(sentences))]
    for sentence, tag_sequence in data:
        if len(sentence) != len(tag_sequence):
            raise Exception("sentence and tag sequence lengths do not line up!!")
    return data

def create_ix_mappings(data: List[Tuple[List, List]]):

    """data is a tuple of 2 lists and looks as follows
    (POS tags are  shown but these could be any tags including CCG supertags):
    [(["The", "dog", "ate", "the", "apple"], ["DET", "NN", "V", "DET", "NN"]),
    (["Everybody", "read", "that", "book"], ["NN", "V", "DET", "NN"])]"""

    word_to_ix = {}
    char_to_ix = {}
    tag_to_ix = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            for char in word:
                if char not in char_to_ix:
                    char_to_ix[char] = len(char_to_ix)
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    return word_to_ix, char_to_ix, tag_to_ix


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

