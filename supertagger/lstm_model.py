"""The neural network model.  Contains LSTMs for both words and characters"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb
from typing import List, Optional
from config import *
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence

class LSTMTagger(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: Optional[int],
                 tagset_size: int,
                 char_embedding_dim: int,
                 char_hidden_dim: int,
                 char_vocab_size: int,
                 lstm_dropout: float = 0) -> None:
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_hidden_dim = char_hidden_dim
        if use_bert_uncased or use_bert_uncased:
            self.use_bert = True
        else:
            self.use_bert = False
        if self.use_bert:
            self.bert = self.get_bert_model()
            #we are not going to fine-tune the bert model itself
            for param in self.bert.parameters():
                param.requires_grad = False
            bert_dim = self.bert.embeddings.word_embeddings.weight.size()[1]
            #we use an affine transformation to compress the bert embeddings down to embedding_dim
            self.compressBertLinear = nn.Linear(bert_dim, embedding_dim)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.char_embeddings = nn.Embedding(char_vocab_size, char_embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(
            embedding_dim + (char_hidden_dim*2),
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout
        )
        self.char_lstm = nn.LSTM(
            char_embedding_dim,
            char_hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        # The linear layer that maps from hidden state space to tag space
        self.linear1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def get_bert_model(self):
        if use_bert_uncased:
            if use_bert_large:
                cased_uncased = "bert-large-uncased"
            else:
                cased_uncased = "bert-base-uncased"
        elif use_bert_cased:
            if use_bert_large:
                cased_uncased = "bert-large-uncased"
            else:
                cased_uncased = "bert-base-cased"
        device = torch.device("cuda:0" if (torch.cuda.is_available() and use_cuda_if_available) else "cpu")
        bert = BertModel.from_pretrained(cased_uncased).to(device=torch.device(device))
        if data_parallel:
            bert = torch.nn.DataParallel(bert)
        return bert

    def init_hidden(self, sent_batch_size: int, device: torch.device) ->  None:
        # The axes semantics are (num_layers*2, minibatch_size, hidden_dim)
        self.hidden = (torch.zeros(4, sent_batch_size, self.hidden_dim).to(device),
                torch.zeros(4, sent_batch_size, self.hidden_dim).to(device))

    def init_char_hidden(self, word_batch_size: int, device: torch.device) -> None:
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.char_hidden = (torch.zeros(2, word_batch_size, self.char_hidden_dim).to(device),
                torch.zeros(2, word_batch_size, self.char_hidden_dim).to(device))

    def forward(self,
                sentences: torch.Tensor,
                words: List[torch.Tensor],
                char_hidden_dim: int,
                sent_lengths: List[int],
                word_batch_size: int,
                device: torch.device,
                attention_masks:torch.Tensor,
                token_start_idx: List[int]
                ) -> torch.Tensor:
        sent_batch_size = sentences.shape[0]
        if self.use_bert:
            #segment_ids = torch.zeros_like(attention_masks)
            bert_last_layer = self.bert(sentences, attention_masks)[0]
            bert_token_reprs = []
            for layer, starts in zip(bert_last_layer, token_start_idx):
                bert_token_reprs.append(layer[torch.tensor(starts)])
            #bert_token_reprs = [layer[torch.tensor(starts).nonzero().squeeze(1)] for layer, starts in zip(bert_last_layer, token_start_idx)]
            padded_bert_token_reprs = pad_sequence(bert_token_reprs, batch_first=True, padding_value=0)
            embeds = self.compressBertLinear(padded_bert_token_reprs[0]).unsqueeze(0)
            for i in range(sent_batch_size-1):
                embeds = torch.cat((embeds,self.compressBertLinear(padded_bert_token_reprs[i+1]).unsqueeze(0)), dim=0)
        else:
            embeds = self.word_embeddings(sentences)
        sent_len = words[0].shape[0]
        char_final_hiddens = torch.zeros(sent_batch_size, sent_len, char_hidden_dim*2)#, requires_grad=False)
        for sent in range(sent_batch_size):
            self.init_char_hidden(word_batch_size=word_batch_size, device=device)
            char_embeds = self.char_embeddings(words[sent])
            word_lengths = (words[sent] != 1).float().sum(dim=1)
            # we treat each sentence as a batch of words for the char LSTM, hence batch size = sent_len
            char_embeds = pack_padded_sequence(char_embeds, word_lengths, enforce_sorted=False, batch_first=True)
            _, self.char_hidden = self.char_lstm(char_embeds, self.char_hidden)
            char_final_hiddens[sent,:,:] = torch.cat((self.char_hidden[0][0], self.char_hidden[0][1]), dim=1)
        embeds = torch.cat((embeds, char_final_hiddens), dim=2)
        embeds = pack_padded_sequence(embeds, sent_lengths, enforce_sorted=False, batch_first=True)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = lstm_out.contiguous()
        #flatten out batches and pass all sentences to Linear layer as one big sequence (i.e. each word is a batch)
        lstm_out = lstm_out.view(sent_batch_size*sent_len, lstm_out.shape[2])
        linear1_out = F.relu(self.linear1(lstm_out))
        tag_logits = self.hidden2tag(linear1_out)
        tag_logits = tag_logits.view(sent_batch_size*sent_len, -1)
        return tag_logits