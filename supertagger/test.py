"""Code for testing a saved model"""

from lstm_model import LSTMTagger
from prepare_data import *
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import pdb, json
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from time import gmtime, strftime
from file_handler import *
from utils import *

models_folder = 'models'


def main(data_path, saved_model_path):
    EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, CHAR_HIDDEN_DIM = load_hyper_params(saved_model_path)
    word_vocab, tag_vocab, char_to_ix = load_vocab_and_char_to_ix(saved_model_path)
    word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = word_vocab.stoi, word_vocab.itos, tag_vocab.stoi, tag_vocab.itos
    test_iter = create_datasets(data_path, mode='test', word_to_ix=word_to_ix, word_vocab=word_vocab, tag_vocab=tag_vocab)
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM,\
                       len(char_to_ix))
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=tag_to_ix['<pad>'])
    _, _, _, _ = load_model(model, optimizer, saved_model_path)
    #torch.autograd.set_detect_anomaly(True)
    print("testing model: "+saved_model_path+'\n')
    model.eval()
    y_pred = []
    y_true = []
    av_test_losses = []
    with torch.no_grad():
        batch_num = 0
        test_losses = []
        for batch in test_iter:
            batch_num += 1
            word_batch_size = batch.sentence.shape[0]
            sent_batch_size = batch.sentence.shape[1]
            model.init_hidden(sent_batch_size)
            sentences_in = batch.sentence.permute(1, 0)
            targets = batch.tags.permute(1,0).reshape(sent_batch_size*word_batch_size)
            y_true += [ix_to_tag[ix.item()] for ix in targets]
            words_in = get_words_in(sentences_in, char_to_ix, ix_to_word)
            tag_logits = model(sentences_in, words_in, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, test_iter.sent_lengths[batch_num-1], word_batch_size)
            test_loss = loss_function(tag_logits, targets)
            test_losses.append(round(test_loss.item(), 2))
            pred = categoriesFromOutput(tag_logits, ix_to_tag)
            y_pred += pred
        av_test_losses.append(sum(test_losses)/len(test_losses))
        accuracy = accuracy_score(y_true, y_pred)
        micro_precision, micro_recall, micro_f1, support = precision_recall_fscore_support(y_true, y_pred, average='micro')
        weighted_macro_precision, weighted_macro_recall, weighted_macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        av_epoch_test_loss = sum(test_losses)/len(test_losses)
        print("Test accuracy: {:.2f}%".format(accuracy*100))
        print("Average Test loss: {}".format(str(av_epoch_test_loss)))
        print("Micro Precision: {}".format(micro_precision))
        print("Micro Recall: {}".format(micro_recall))
        print("Micro F1: {}".format(micro_f1))
        print("Weighted Macro Precision: {}".format(weighted_macro_precision))
        print("Weighted Macro Recall: {}".format(weighted_macro_recall))
        print("Weighted Macro F1: {}".format(weighted_macro_f1))


if __name__ == '__main__':
    cmd_parser = argparse.ArgumentParser(description='command line arguments.')
    cmd_parser.add_argument('--data-path', dest='data_path', type=str, nargs=1, help='the path to the folder containing the data files')
    cmd_parser.add_argument('--model-path', dest='saved_model_path', type=str, nargs=1,
                            help='the relative path to the model you wish to resume training from')
    args = cmd_parser.parse_args()

    main(args.data_path[0], args.saved_model_path[0])
