"""Preprocesses data for the supertagger"""

#standard library imports
import os
import pandas as pd
from collections import Counter, defaultdict
from constants import *
from typing import List, Tuple, DefaultDict, Union, Optional, Dict

#third party imports
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torchtext.data import Field, BucketIterator, Iterator, TabularDataset
from torchtext.data.dataset import TabularDataset
from torchtext.vocab import Vocab
from transformers import BertTokenizer

#local imports
from config import *
from utils import *


createDatasetsReturnType = Union[Tuple[BucketIterator, BucketIterator, Vocab, Vocab, DefaultDict[str, int]],
                                 Tuple[BucketIterator]]

class BertTokenizedDataset(Dataset):
    """
    The main class containing the Bert-tokenized dataset.  The data is subsequently transferred into a DataLoader object
    so that it can be iterated over without having to be loaded into memory all at once (see create_bert_datasets() below).
    """

    def __init__(self, data_path_words: str, data_path_tags: str, tag_to_ix: Vocab, tokenizer: BertTokenizer) -> None:
        self.tokenizer = tokenizer
        self.original_sentences = [sent.strip() for sent in open(data_path_words).readlines()]
        self.tags = open(data_path_tags).readlines()
        input_ids = []
        self.tag_ids = [[tag_to_ix[tag] for tag in tags.strip().split()] for tags in self.tags]
        self.attention_masks = []
        for sent in self.original_sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=512,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            self.attention_masks.append(encoded_dict['attention_mask'])
        self.token_start_idx = self.get_token_start_idxs()
        self.input_ids = torch.cat(input_ids, dim=0)
        self.attention_masks = torch.cat(self.attention_masks, dim=0)

    def get_token_start_idxs(self) -> List[int]:
        """
        Returns a list of token start indexes.  Needed because the bert tokenization breaks many (unseen) words up.
        For example "isolationist" -> ['isolation', '##ist'].   For the final supertagging we only want one supertag per
        word so we will just use the hidden state of the first subtoken for each word and token_start_idxs enables this.
        """
        token_start_idxs = []
        for sent in self.original_sentences:
            words = sent.split()
            subwords = list(map(self.tokenizer.tokenize, words))
            subword_lengths = list(map(len, subwords))
            token_start_idxs.append(list(np.cumsum([0] + subword_lengths))[1:])
        return token_start_idxs

    def __len__(self) -> int:
        return len(self.original_sentences)

    def __getitem__(self, idx) -> Dict:
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
    """
    A wrapper class for the BertTokenizer method convert_tokens_to_ids()
    allowing thesame syntax as standard python dictioaries.
    """

    def __init__(self, tokenizer: BertTokenizer):
        self.tokenizer = tokenizer

    def __getitem__(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

class BertIxToToken:
    """
    A wrapper class for the BertTokenizer method convert_ids_to_token()
    allowing thesame syntax as standard python dictionaries.
    """

    def __init__(self, tokenizer: BertTokenizer):
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        return self.tokenizer.convert_ids_to_tokens(idx)


createBertDatasetsReturnType = Union[Tuple[DataLoader, DataLoader, BertTokenToIx, BertIxToToken, Vocab, defaultdict], DataLoader]


def create_bert_datasets(
        data_path: str,
        mode: str,
        use_bert_cased: bool,
        use_bert_uncased: bool,
        use_bert_large: bool,
        tag_to_ix=None
) -> createBertDatasetsReturnType:
    """
    The main function which compiles the data first into a BertTokenizedDataset object and the into a DataLoader object.
    """
    if use_bert_uncased:
        if use_bert_large:
            tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    elif use_bert_cased:
        if use_bert_large:
            tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    if mode == TRAIN:
        tag_vocab = get_tag_vocab(os.path.join(data_path))
        tag_to_ix, ix_to_tag = tag_vocab.stoi, tag_vocab.itos
        word_to_ix, ix_to_word = BertTokenToIx(tokenizer), BertIxToToken(tokenizer)
        train_data_path = os.path.join(data_path, TRAIN)
        val_data_path = os.path.join(data_path, VAL)
        train_dataset = BertTokenizedDataset(
            data_path_words=train_data_path+".words",
            data_path_tags=train_data_path+".tags",
            tag_to_ix=tag_to_ix,
            tokenizer=tokenizer
        )
        val_dataset = BertTokenizedDataset(
            data_path_words=val_data_path+".words",
            data_path_tags=val_data_path+".tags",
            tag_to_ix=tag_to_ix,
            tokenizer=tokenizer
        )

        train_iter = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=batch_size,
            collate_fn=collate_fn
        )
        # For validation the order doesn't matter, so we'll just read them sequentially.
        val_iter = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=batch_size,
            collate_fn=collate_fn
        )
        char_vocab = get_char_vocab(train_dataset)
        char_to_ix = char_vocab.stoi

        return train_iter, val_iter, word_to_ix, ix_to_word, tag_vocab, char_to_ix

    else:
        test_data_path = os.path.join(data_path, TEST)
        test_dataset = BertTokenizedDataset(
            data_path_words=test_data_path + ".words",
            data_path_tags=test_data_path + ".tags",
            tag_to_ix=tag_to_ix,
            tokenizer=tokenizer
        )
        test_iter = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=batch_size,
            collate_fn=collate_fn
        )

        return test_iter


def collate_fn(sentences_batch: List[Dict]) -> Tuple[torch.Tensor,
                                                     torch.Tensor,
                                                     List[List[int]],
                                                     torch.Tensor,
                                                     List[str]]:
    """
    Determines how the data is returned when DataLoader object is iterated over.
    The dictionaries inside the sentences_batch list each have the following structure:

    {'original_sentence': 'Someone discovered a secret .', 'tags': 'NN V DET NN PUNC',
    'input_ids': tensor([ 101, 2619, 3603, 1037, 3595, 1012,  102,    0,    0,    0,    0,    0,
    0,    0,   ]), 'tag_ids': [2, 5, 3, 2, 4], 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, ]), 'token_start_idx': [1, 2, 3, 4, 5]}
    """
    input_ids = torch.cat(tuple([torch.unsqueeze(features['input_ids'], dim=0) for features in sentences_batch]), dim=0)
    attention_masks = torch.cat(tuple([torch.unsqueeze(features['attention_mask'], dim=0) for features in sentences_batch]), dim=0)
    token_start_idx = [features['token_start_idx'] for features in sentences_batch]
    original_sentences = [features['original_sentence'] for features in sentences_batch]
    max_sent_len = max([len(starts) for starts in token_start_idx])
    tag_ids = pad_sequence([torch.tensor(features['tag_ids'][:max_sent_len]) for features in sentences_batch], batch_first=True, padding_value=1).view(-1)
    return input_ids, attention_masks, token_start_idx, tag_ids, original_sentences


def get_char_vocab(dataset: BertTokenizedDataset) -> Vocab:
    charCounter = Counter()
    for entry in dataset:
        for word in entry['original_sentence'].strip().split():
            for char in word:
                charCounter[char] += 1
    return Vocab(charCounter)


def get_tag_vocab(data_path: str) -> Vocab:
    tagCounter = Counter()
    for line in open(os.path.join(data_path, 'train.tags')):
        for tag in line.strip().split():
            tagCounter[tag] += 1
    return Vocab(tagCounter)


def create_datasets(
        data_path: str,
        mode: str,
        word_to_ix=None,
        word_vocab=None,
        tag_vocab=None
) -> Union[createDatasetsReturnType, BucketIterator]:
    """
    Used when bert embeddings are switched off (i.e. we just used randomly initialized embeddings.
    Compiles the data first into a TabularDataset object and then into a BucketIterator
    (similar to a DataLoader) object via the function to_iter().
    """
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
        device: torch.device,
        tokenizer: Optional[BertTokenizer],
        use_bert: bool
) -> Tuple[List[List[str]], List[torch.Tensor]]:
    """
    Compiles unseen, untagged sentences into a tuple of tokenized sentences and tensors ready to be input to the model.
    """
    if not tokenizer:
        from nltk.tokenize import word_tokenize
        sentences = [word_tokenize(line) for line in open(data_path).readlines()]
        sent_tensors = [prepare_sequence(sent, word_to_ix).view(1, -1).to(device) for sent in sentences]
    else:
        sentences = [tokenizer.tokenize(line) for line in open(data_path).readlines()]
        sent_tensors = [torch.tensor(tokenizer.encode(sent, add_special_tokens=True)).unsqueeze(0).to(device) for sent in sentences]
    return sentences, sent_tensors


def prepare_sequence(seq: List[str], to_ix: DefaultDict[str, int]) -> torch.Tensor:
    """
    Converts a sequence (used here for characters) into a tensor of integers.
    """
    idxs = [to_ix[w] for w in seq]

    return torch.tensor(idxs, dtype=torch.long)


def get_words_in(
        sentences_in: torch.Tensor,
        char_to_ix: DefaultDict[str, int],
        ix_to_word: List[str],
        device: torch.device = 'cpu',
        original_sentences_split: Optional[List[List[str]]] = None
) -> List[torch.Tensor]:
    """
    Converts sentences into a list of (padded) 2d torch tensors in which each row vector represents a word, which each
    integer value of the row vector representing a character.
    """
    words_in = []
    if original_sentences_split:
        max_sent_len = max([len(sent) for sent in original_sentences_split])
    for i in range(sentences_in.shape[0]):
        if original_sentences_split:
            sentence_words = original_sentences_split[i]
            while len(sentence_words) < max_sent_len:
                sentence_words.append("<pad>")
        else:
            sentence_words = [ix_to_word[ix.item()] for ix in sentences_in[i, :]]
        words_in.append([prepare_sequence(word, char_to_ix) for word in sentence_words])
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


