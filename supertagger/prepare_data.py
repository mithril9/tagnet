"""Preprocesses data for the supertagger"""

import torch
import numpy as np
from torchtext.data import Field, BucketIterator, Iterator, TabularDataset
from torchtext.data.dataset import TabularDataset
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import pdb
import os
from typing import List,  Tuple
#from sklearn.model_selection import train_test_split
import pandas as pd
from config import *
from utils import *
from typing import DefaultDict, Union
from transformers import BertTokenizer
from constants import *
from collections import Counter

createDatasetsReturnType = Union[Tuple[BucketIterator, BucketIterator, Vocab, Vocab, DefaultDict[str, int]],
                                 Tuple[BucketIterator]]

if use_bert_uncased:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
elif use_bert_cased:
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class BertTokenizedDataset(Dataset):

    def __init__(self, data_path_words: str, data_path_tags: str, tag_to_ix: Vocab):
        self.original_sentences = open(data_path_words).readlines()
        self.tags = open(data_path_tags).readlines()
        input_ids = []
        self.tag_ids = [[tag_to_ix[tag] for tag in tags.strip().split()] for tags in self.tags]
        self.token_start_idx = self.get_token_start_idxs()
        self.attention_masks = []
        for sent in self.original_sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=bert_max_seq_len,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
            self.attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        self.input_ids = torch.cat(input_ids, dim=0)
        self.attention_masks = torch.cat(self.attention_masks, dim=0)

    def get_token_start_idxs(self):
        token_start_idxs = []
        for sent in self.original_sentences:
            words = sent.split()
            subwords = list(map(tokenizer.tokenize, words))
            subword_lengths = list(map(len, subwords))
            subwords = [CLS] + list(self.flatten(subwords)) + [SEP]
            token_start_idxs.append(list(1 + np.cumsum([0] + subword_lengths[:-1])))
        return token_start_idxs

    def flatten(self, list_of_lists):
        flattened_list = []
        for LIST in list_of_lists:
            flattened_list += LIST
        return flattened_list

    def __len__(self):
        return len(self.original_sentences)

    def __getitem__(self, idx):
        sample = {
            'original_sentence':    self.original_sentences[idx],
            'tags':                 self.tags[idx],
            'input_ids':            self.input_ids[idx],
            'tag_ids':              self.tag_ids[idx],
            'attention_mask':       self.attention_masks[idx],
            'token_start_idx':      self.token_start_idx[idx]
        }
        return sample

class BertTokenToIx:

    def __init__(self):
        pass

    def __getitem__(self, token):
        return tokenizer.convert_tokens_to_ids(token)

class BertIxToWord:

    def __init__(self):
        pass

    def __getitem__(self, idx):
        return tokenizer.convert_ids_to_tokens(token)


def create_bert_datasets(data_path: str, mode: str):
    if mode == TRAIN:
        tag_vocab = get_tag_vocab(os.path.join(data_path))
        tag_to_ix, ix_to_tag = tag_vocab.stoi, tag_vocab.itos
        word_to_ix, ix_to_word = BertTokenToIx(), BertIxToWord()
        train_data_path = os.path.join(data_path, TRAIN)
        val_data_path = os.path.join(data_path, VAL)
        train_dataset = BertTokenizedDataset(
            data_path_words=train_data_path+".words",
            data_path_tags=train_data_path+".tags",
            tag_to_ix=tag_to_ix
        )
        val_dataset = BertTokenizedDataset(
            data_path_words=val_data_path+".words",
            data_path_tags=val_data_path+".tags",
            tag_to_ix=tag_to_ix
        )

        train_iter = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset),  # Select batches randomly
            batch_size=batch_size,  # Trains with this batch size.
            collate_fn=collate_fn
        )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        val_iter = DataLoader(
            val_dataset,  # The validation samples.
            sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
            batch_size=batch_size,  # Evaluate with this batch size.
            collate_fn=collate_fn
        )

        char_vocab = get_char_vocab(train_dataset)
        char_to_ix = char_vocab.stoi

        return train_iter, val_iter, word_to_ix, ix_to_word, tag_vocab, char_to_ix

def get_char_vocab(dataset: BertTokenizedDataset):
    charCounter = Counter()
    for entry in dataset:
        for word in entry['original_sentence'].strip().split():
            for char in word:
                charCounter[char] += 1
    return Vocab(charCounter)

def collate_fn(sentences_batch):
    input_ids = torch.cat(tuple([torch.unsqueeze(features['input_ids'], dim=0) for features in sentences_batch]), dim=0)
    attention_masks = torch.cat(tuple([torch.unsqueeze(features['attention_mask'], dim=0) for features in sentences_batch]), dim=0)
    token_start_idx = [features['token_start_idx'] for features in sentences_batch]
    tag_ids = [features['tag_ids'] for features in sentences_batch]
    batch = [features[key] for features in sentences_batch for key in ('input_ids', 'attention_mask', 'token_start_idx', 'tag_ids')]
    return batch


def get_tag_vocab(data_path: str):
    tagCounter = Counter()
    for line in open(os.path.join(data_path, 'train.tags')):
        for tag in line.strip().split():
            tagCounter[tag] += 1
    return Vocab(tagCounter)


def create_datasets(data_path: str, mode: str, word_to_ix=None, word_vocab=None, tag_vocab=None) -> createDatasetsReturnType:
    sent_field = Field(lower=True)
    tag_field = Field()
    data_fields = [('sentence', sent_field), ('tags', tag_field)]
    if mode == TRAIN:
        dataSetNames = [TRAIN, VAL]
    elif mode == TEST:
        dataSetNames = [TEST]
    for data_set in dataSetNames:
        create_csv(os.path.join(data_path, data_set))
    if mode == TRAIN:
        train_dataset, val_dataset = TabularDataset.splits(path=data_path,
                                                           train='train.csv',
                                                           validation='val.csv',
                                                           format='csv',
                                                           fields=data_fields,
                                                           skip_header=True)
        # build the vocab over the train set only
        sent_field.build_vocab(train_dataset)
        tag_field.build_vocab(train_dataset)
        char_to_ix = get_char_to_ix(train_dataset)
        train_iter = to_iter(train_dataset, sent_field.vocab.stoi['<pad>'], batch_size)
        val_iter = to_iter(val_dataset, sent_field.vocab.stoi['<pad>'], 1)
        return train_iter, val_iter, sent_field.vocab, tag_field.vocab, char_to_ix
    elif mode == TEST:
        sent_field.vocab = word_vocab
        tag_field.vocab = tag_vocab
        test_dataset = TabularDataset(path=os.path.join(data_path, 'test.csv'),
                                      format='csv',
                                      fields=data_fields,
                                      skip_header=True)
        test_iter = to_iter(test_dataset, word_to_ix['<pad>'], 1)
        return test_iter


def get_char_to_ix(dataset: TabularDataset) -> DefaultDict[str, int]:
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


def to_iter(dataset: TabularDataset, pad_ix: int, batch_size: int) -> BucketIterator:
    #sort_within_batch = True is need for packing the padded sequence
    data_iter = BucketIterator(dataset, sort=True, batch_size=batch_size, device=-1, sort_within_batch=True, sort_key=lambda x: len(x.sentence), shuffle=False)
    data_iter.sent_lengths = []
    for batch in data_iter:
        batch_sent_len = batch.sentence[:,0].shape[0]
        lengths = []
        for i in range(batch.sentence.shape[1]):
            #must subtract the number of pads from the length
            lengths.append((batch_sent_len - (batch.sentence[:,i] == pad_ix).sum(dim=0)).item())
        data_iter.sent_lengths.append(lengths)
    return data_iter


def create_csv(data_path: str) -> None:
    raw_sentences = open(data_path + ".words").readlines()
    raw_tag_sequences = open(data_path+".tags").readlines()
    raw_data = {'sentence': [sent.strip() for sent in raw_sentences if sent != ''],
                'tags': [tag_seq.strip() for tag_seq in raw_tag_sequences if tag_seq != '']}
    df = pd.DataFrame(raw_data, columns=["sentence", "tags"])
    df.to_csv(data_path+".csv", index=False)


def prepare_untagged_data(
        data_path: str,
        word_to_ix: DefaultDict[str, int],
        device: torch.device
) -> Tuple[List[List[str]], List[torch.Tensor]]:
    sentences = [tokenize(line) for line in open(data_path).readlines()]
    sent_tensors = [prepare_sequence(sent, word_to_ix).view(1,-1).to(device) for sent in sentences]
    return sentences, sent_tensors


def prepare_sequence(seq: List[str], to_ix: DefaultDict[str, int]) -> torch.Tensor:
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def get_words_in(
        sentences_in: torch.Tensor,
        char_to_ix: DefaultDict[str, int],
        ix_to_word: List[str],
        device: torch.device = 'cpu'
) -> List[torch.Tensor]:
    words_in = []
    for i in range(sentences_in.shape[0]):
        words_in.append([prepare_sequence(word, char_to_ix) for word in [ix_to_word[ix] for ix in sentences_in[i, :]]])
        words_in[-1] = batchify_sent(words_in[-1]).to(device)
    return words_in


def batchify_sent(sent: List[torch.Tensor]) -> torch.Tensor:
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


