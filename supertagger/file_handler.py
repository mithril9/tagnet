#Code for training the model

# WRITTEN BY:
#   John Torr (john.torr@cantab.net)

#standard library imports
import os
from numpy import float64
from typing import DefaultDict, Union, Tuple, Optional, List
from collections import defaultdict

#third party imports
import torch
from torch.optim.adam import Adam
from torchtext.vocab import Vocab

#local imports
from lstm_model import LSTMTagger
from prepare_data import BertTokenToIx, BertIxToToken


loadModelReturn = Tuple[List[float],
                        List[float],
                        int,
                        float64,
                        float,
                        float64,
                        float64,
                        float64,
                        float64,
                        float64,
                        float64]

def load_vocab_and_char_to_ix(saved_model_path: str) -> Tuple[Vocab, Vocab, DefaultDict[str, int]]:
    print("Attempting to load saved ix mappings from: " + saved_model_path)
    checkpoint = torch.load(saved_model_path)
    return checkpoint['word_to_ix'], \
           checkpoint['ix_to_word'], \
           checkpoint['word_vocab'], \
           checkpoint['tag_vocab'], \
           checkpoint['char_to_ix']


def load_model(model: LSTMTagger,
               saved_model_path: str,
               optimizer: Optional[Adam] = None
               ) -> Optional[loadModelReturn]:
    print("Attempting to load saved model checkpoint from: " + saved_model_path)
    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Successfully loaded model..")
    if not optimizer:
        return None
    else:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['av_train_losses'], \
               checkpoint['av_eval_losses'], \
               checkpoint['checkpoint_epoch'], \
               checkpoint['accuracy'], \
               checkpoint['av_eval_loss'], \
               checkpoint['micro_precision'], \
               checkpoint['micro_recall'], \
               checkpoint['micro_f1'], \
               checkpoint['weighted_macro_precision'], \
               checkpoint['weighted_macro_recall'], \
               checkpoint['weighted_macro_f1']

def load_hyper_params(saved_model_path: str) -> Tuple[int]:
    print("Attempting to load saved hyperparameters from: " + saved_model_path)
    checkpoint = torch.load(saved_model_path)
    return checkpoint['embedding_dim'], \
           checkpoint['char_embedding_dim'], \
           checkpoint['hidden_dim'], \
           checkpoint['char_hidden_dim'], \
           checkpoint['use_bert_cased'], \
           checkpoint['use_bert_uncased'], \
           checkpoint['use_bert_large']

def save_model(epoch: int,
               model: LSTMTagger,
               optimizer: Adam,
               av_train_losses: List[float],
               av_eval_losses: List[float],
               model_file_name: str,
               word_to_ix: Union[BertTokenToIx, defaultdict],
               ix_to_word: Union[BertIxToToken, defaultdict],
               word_vocab: Optional[Vocab],
               tag_vocab: Vocab,
               char_to_ix: DefaultDict[str, int],
               models_folder: str,
               embedding_dim: int,
               char_embedding_dim: int,
               hidden_dim: int,
               char_hidden_dim: int,
               accuracy: float64,
               av_eval_loss: float,
               micro_precision: float64,
               micro_recall: float64,
               micro_f1: float64,
               weighted_macro_precision: float64,
               weighted_macro_recall: float64,
               weighted_macro_f1: float64,
               use_bert_cased: bool,
               use_bert_uncased: bool,
               use_bert_large: bool
               ) -> None:
    try:
        os.remove(os.path.join("..", models_folder, model_file_name))
    except FileNotFoundError:
        pass
    torch.save({
            'checkpoint_epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'av_train_losses': av_train_losses,
            'av_eval_losses': av_eval_losses,
            'word_to_ix': word_to_ix,
            'ix_to_word': ix_to_word,
            'word_vocab': word_vocab,
            'tag_vocab': tag_vocab,
            'char_to_ix': char_to_ix,
            'embedding_dim': embedding_dim,
            'char_embedding_dim': char_embedding_dim,
            'hidden_dim': hidden_dim,
            'char_hidden_dim': char_hidden_dim,
            'accuracy': accuracy,
            'av_eval_loss': av_eval_loss,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'weighted_macro_precision': weighted_macro_precision,
            'weighted_macro_recall': weighted_macro_recall,
            'weighted_macro_f1': weighted_macro_f1,
            'use_bert_cased': use_bert_cased,
            'use_bert_uncased': use_bert_uncased,
            'use_bert_large': use_bert_large
    }, os.path.join("..", models_folder, model_file_name))
    print("Model with lowest average eval loss successfully saved as: "+os.path.join("..", models_folder, model_file_name))
