#The neural network model.  Uses LSTMs for both words and characters.  Optionally uses (frozen) bert word embeddings
#(final layer) otherwise uses randomly initialized trainable word embeddings.

# WRITTEN BY:
#   John Torr (john.torr@cantab.net)

#standard library imports
from typing import List, Optional

#third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from transformers import BertModel

#local imports
from config import data_parallel


class LSTMTagger(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: Optional[int],
                 tagset_size: int,
                 char_embedding_dim: int,
                 char_hidden_dim: int,
                 char_vocab_size: int,
                 use_bert_cased: bool,
                 use_bert_uncased: bool,
                 use_bert_large: bool
                 ) -> None:
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_hidden_dim = char_hidden_dim
        if use_bert_uncased or use_bert_uncased:
            self.use_bert = True
        else:
            self.use_bert = False
        if self.use_bert:
            self.bert = self.get_bert_model(use_bert_cased, use_bert_uncased, use_bert_large)
            #we are not going to fine-tune the bert model itself
            for param in self.bert.parameters():
                param.requires_grad = False
            bert_dim = self.bert.embeddings.word_embeddings.weight.size()[1]
            #we use an affine transformation to compress the bert embeddings down to embedding_dim
            self.compressBertLinear = nn.Linear(bert_dim, embedding_dim)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.char_embeddings = nn.Embedding(char_vocab_size, char_embedding_dim)
        # The LSTM takes word embeddings (either bert or just randomly initialized) concateated with character-based
        # LSTM final hidden state representations of each word as inputs, and outputs hidden states with dimensionality
        # hidden_dim.
        self.lstm = nn.LSTM(
            embedding_dim + (char_hidden_dim*2),
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
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

    def get_bert_model(self, use_bert_cased: bool, use_bert_uncased: bool, use_bert_large: bool) -> BertModel:
        if use_bert_uncased:
            if use_bert_large:
                which_bert = "bert-large-uncased"
            else:
                which_bert = "bert-base-uncased"
        elif use_bert_cased:
            if use_bert_large:
                which_bert = "bert-large-uncased"
            else:
                which_bert = "bert-base-cased"
        print("Loading model: "+which_bert+"...")
        device = torch.device("cuda:0" if (torch.cuda.is_available() and use_cuda_if_available) else "cpu")
        bert = BertModel.from_pretrained(which_bert).to(device=torch.device(device))
        if data_parallel:
            bert = torch.nn.DataParallel(bert)
        return bert

    def init_hidden(self, sent_batch_size: int, device: torch.device) ->  None:
        # The axes semantics are (num_layers*2, minibatch_size, hidden_dim)
        # the *2 is because this is a bi-LSTM
        self.hidden = (torch.zeros(4, sent_batch_size, self.hidden_dim).to(device),
                torch.zeros(4, sent_batch_size, self.hidden_dim).to(device))

    def init_char_hidden(self, word_batch_size: int, device: torch.device) -> None:
        # The axes semantics are (num_layers*2, minibatch_size, hidden_dim)
        # the *2 is because this is a bi-LSTM
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
            bert_last_layer = self.bert(sentences, attention_masks)[0]
            bert_token_reprs = []
            #we only use the hidden state representations for the first subword token for each word because we only
            #want to predict one supertag per word, not one per bert token.
            for layer, starts in zip(bert_last_layer, token_start_idx):
                bert_token_reprs.append(layer[torch.tensor(starts)])
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