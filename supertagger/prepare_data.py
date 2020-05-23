"""Preprocesses data for the supertagger"""

import torch

def create_ix_mappings(training_data: tuple[list, list]):
    """training_data is a tuple of 2 lists and looks as follows
    (POS tags are  shown but these could be any tags including CCG supertags):
    [
    (["The", "dog", "ate", "the", "apple"], ["DET", "NN", "V", "DET", "NN"]),
    (["Everybody", "read", "that", "book"], ["NN", "V", "DET", "NN"])
    ]"""

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

