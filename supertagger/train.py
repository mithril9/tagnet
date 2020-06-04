"""Code for training the model"""

from lstm_model import LSTMTagger
from config import *
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
import copy

models_folder = 'models'


def main(data_path, saved_model_path):
    if saved_model_path:
        global EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, CHAR_HIDDEN_DIM
        EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, CHAR_HIDDEN_DIM = load_hyper_params(saved_model_path)
    train_iter, \
    val_iter, \
    word_vocab, \
    tag_vocab, \
    char_to_ix = create_datasets(data_path, mode='train')
    #char_to_ix gets added to automatically with any characters (e.g. < >) encountered during evaluation, but we want to
    #save the original copy so that the char embeddings para can be computed, hence we create a copy here.
    char_to_ix_copy = copy.deepcopy(char_to_ix)
    word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = word_vocab.stoi, word_vocab.itos, tag_vocab.stoi, tag_vocab.itos
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM,\
                       len(char_to_ix))
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=tag_to_ix['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    if models_folder not in os.listdir("../"):
        os.mkdir("../"+models_folder)
    if saved_model_path:
        av_train_losses, av_eval_losses, checkpoint_epoch, loss = load_model(model, optimizer, saved_model_path)
        lowest_av_eval_loss  = min(av_eval_losses)
        model_file_name = os.path.split(saved_model_path)[1]
    else:
        checkpoint_epoch = 0
        av_train_losses = []
        av_eval_losses = []
        lowest_av_eval_loss = 999999
        model_file_name = strftime("%Y_%m_%d_%H_%M_%S.pt")
    #torch.autograd.set_detect_anomaly(True)
    print("training..\n")
    model.train()
    for epoch in range(num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        print('===============================')
        print('\n======== Epoch {} / {} ========'.format(epoch + 1 + checkpoint_epoch, num_epochs+checkpoint_epoch))
        batch_num = 0
        train_losses = []
        for batch in train_iter:
            batch_num += 1
            if batch_num % 20 == 0 or batch_num == 1:
                if batch_num != 1:
                    print("\nAverage Training loss for epoch {} at end of batch {}: {}".format(epoch + checkpoint_epoch, str(batch_num-1),sum(train_losses)/len(train_losses)))
                print('\n======== Batch {} / {} ========'.format(batch_num, len(train_iter)))
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            word_batch_size = batch.sentence.shape[0]
            sent_batch_size = batch.sentence.shape[1]
            model.init_hidden(sent_batch_size)
            #we want batch to be the first dimension
            sentences_in = batch.sentence.permute(1,0)
            targets = batch.tags.permute(1,0).reshape(sent_batch_size*word_batch_size)
            words_in = get_words_in(sentences_in, char_to_ix, ix_to_word)
            # Step 3. Run our forward pass.
            tag_logits = model(sentences_in, words_in, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, train_iter.sent_lengths[batch_num-1], word_batch_size)
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_logits, targets)
            train_losses.append(round(loss.item(), 2))
            loss.backward()
            optimizer.step()
        av_train_losses.append(sum(train_losses)/len(train_losses))
        # Evaluate the model
        model.eval()
        y_pred = []
        y_true = []
        print("\nEvaluating model...")
        with torch.no_grad():
            batch_num = 0
            eval_losses = []
            for batch in val_iter:
                batch_num += 1
                word_batch_size = batch.sentence.shape[0]
                sent_batch_size = batch.sentence.shape[1]
                model.init_hidden(sent_batch_size)
                sentences_in = batch.sentence.permute(1, 0)
                targets = batch.tags.permute(1,0).reshape(sent_batch_size*word_batch_size)
                y_true += [ix_to_tag[ix.item()] for ix in targets]
                words_in = get_words_in(sentences_in, char_to_ix, ix_to_word)
                tag_logits = model(sentences_in, words_in, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, val_iter.sent_lengths[batch_num-1], word_batch_size)
                eval_loss = loss_function(tag_logits, targets)
                eval_losses.append(round(eval_loss.item(), 2))
                pred = categoriesFromOutput(tag_logits, ix_to_tag)
                y_pred += pred
            av_eval_losses.append(sum(eval_losses)/len(eval_losses))
            if av_eval_losses[-1] < lowest_av_eval_loss:
                lowest_av_eval_loss = av_eval_losses[-1]
                save_model(epoch + 1 + checkpoint_epoch, model, optimizer, loss, av_train_losses, av_eval_losses,
                           model_file_name, word_vocab, tag_vocab, char_to_ix_copy, models_folder,
                           EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, CHAR_HIDDEN_DIM)
            accuracy = accuracy_score(y_true, y_pred)
            micro_precision, micro_recall, micro_f1, support = precision_recall_fscore_support(y_true, y_pred,
                                                                                                        average='micro')
            weighted_macro_precision, weighted_macro_recall, weighted_macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            av_epoch_eval_loss = sum(eval_losses)/len(eval_losses)
            print("Eval accuracy at end of epoch {}: {:.2f}%".format(epoch + 1 + checkpoint_epoch, accuracy*100))
            print("Average Eval loss for epoch {}: {}".format(epoch + 1 + checkpoint_epoch, str(av_epoch_eval_loss)))
            print("Micro Precision: {}".format(micro_precision))
            print("Micro Recall: {}".format(micro_recall))
            print("Micro F1: {}".format(micro_f1))
            print("Weighted Macro Precision: {}".format(weighted_macro_precision))
            print("Weighted Macro Recall: {}".format(weighted_macro_recall))
            print("Weighted Macro F1: {}".format(weighted_macro_f1))
    plt.xlabel("n epochs")
    plt.ylabel("loss")
    plt.plot(av_train_losses, label='train')
    plt.plot(av_eval_losses, label='eval')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    cmd_parser = argparse.ArgumentParser(description='command line arguments.')
    cmd_parser.add_argument('--data-path', dest='data_path', type=str, nargs=1, help='the path to the folder containing the data files')
    cmd_parser.add_argument('--model-path', dest='saved_model_path', type=str, nargs=1,
                            help='the relative path to a model you wish to resume training from')
    args = cmd_parser.parse_args()
    if args.saved_model_path:
        saved_model_path = args.saved_model_path[0]
    else:
        saved_model_path = None

    main(args.data_path[0], saved_model_path)
