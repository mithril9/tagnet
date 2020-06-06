"""Code for training the model"""

from lstm_model import LSTMTagger
from config import *
from prepare_data import *
import torch.optim as optim
import argparse
import os
import pdb, json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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
                       len(char_to_ix), dropout=dropout)
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=tag_to_ix['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if models_folder not in os.listdir("../"):
        os.mkdir("../"+models_folder)
    if saved_model_path:
        av_train_losses, av_eval_losses, checkpoint_epoch, loss, best_accuracy, lowest_av_eval_loss, \
        best_micro_precision, best_micro_recall, best_micro_f1, best_weighted_macro_precision, best_weighted_macro_recall, \
        best_weighted_macro_f1 = load_model(model=model, saved_model_path=saved_model_path, optimizer=optimizer)
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
    start_epoch = checkpoint_epoch+1
    end_epoch = checkpoint_epoch+num_epochs
    for epoch in range(start_epoch, end_epoch+1):  # again, normally you would NOT do 300 epochs, it is toy data
        print('===============================')
        print('\n======== Epoch {} / {} ========'.format(epoch, end_epoch))
        batch_num = 0
        train_losses = []
        for batch in train_iter:
            batch_num += 1
            if batch_num % 20 == 0 or batch_num == 1:
                if batch_num != 1:
                    print("\nAverage Training loss for epoch {} at end of batch {}: {}".format(epoch-1, str(batch_num-1),sum(train_losses)/len(train_losses)))
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
        av_train_losses.append(sum(train_losses) / len(train_losses))
        accuracy, av_epoch_eval_loss, micro_precision, micro_recall, micro_f1, weighted_macro_precision, \
        weighted_macro_recall, weighted_macro_f1 = eval_model(model, loss_function, val_iter, char_to_ix, ix_to_word,
                                                              ix_to_tag, av_eval_losses, lowest_av_eval_loss, epoch)
        print_results(epoch, accuracy, av_epoch_eval_loss, micro_precision, micro_recall, micro_f1, weighted_macro_precision, weighted_macro_recall, weighted_macro_f1)
        if av_eval_losses[-1] < lowest_av_eval_loss:
            lowest_av_eval_loss = av_eval_losses[-1]
            best_accuracy, \
            best_micro_precision, \
            best_micro_recall, \
            best_micro_f1, \
            best_weighted_macro_precision, \
            best_weighted_macro_recall, \
            best_weighted_macro_f1 = accuracy, \
                                     micro_precision, \
                                     micro_recall, \
                                     micro_f1, \
                                     weighted_macro_precision, \
                                     weighted_macro_recall, \
                                     weighted_macro_f1
            checkpoint_epoch = epoch
            save_model(checkpoint_epoch, model, optimizer, loss, av_train_losses, av_eval_losses, model_file_name, word_vocab,
                       tag_vocab, char_to_ix_copy, models_folder, EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM,
                       CHAR_HIDDEN_DIM, best_accuracy, lowest_av_eval_loss, best_micro_precision, best_micro_recall,
                       best_micro_f1, best_weighted_macro_precision, best_weighted_macro_recall, best_weighted_macro_f1)
    print_results(checkpoint_epoch, best_accuracy, lowest_av_eval_loss, best_micro_precision, best_micro_recall,
                  best_micro_f1, best_weighted_macro_precision, best_weighted_macro_recall, best_weighted_macro_f1, final=True)
    plot_train_eval_loss(av_train_losses, av_eval_losses)

def eval_model(model, loss_function, val_iter, char_to_ix, ix_to_word, ix_to_tag, av_eval_losses, lowest_av_eval_loss, epoch):
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
            targets = batch.tags.permute(1, 0).reshape(sent_batch_size * word_batch_size)
            y_true += [ix_to_tag[ix.item()] for ix in targets]
            words_in = get_words_in(sentences_in, char_to_ix, ix_to_word)
            tag_logits = model(sentences_in, words_in, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM,
                               val_iter.sent_lengths[batch_num - 1], word_batch_size)
            eval_loss = loss_function(tag_logits, targets)
            eval_losses.append(round(eval_loss.item(), 2))
            pred = categoriesFromOutput(tag_logits, ix_to_tag)
            y_pred += pred
        av_eval_losses.append(sum(eval_losses) / len(eval_losses))
        accuracy = accuracy_score(y_true, y_pred)
        micro_precision, micro_recall, micro_f1, support = precision_recall_fscore_support(y_true, y_pred,
                                                                                           average='micro')
        weighted_macro_precision, weighted_macro_recall, weighted_macro_f1, _ = precision_recall_fscore_support(y_true,
                                                                                                                y_pred,
                                                                                                                average='weighted')
        av_epoch_eval_loss = sum(eval_losses) / len(eval_losses)

    return accuracy, av_epoch_eval_loss, micro_precision, micro_recall, micro_f1, weighted_macro_precision, \
           weighted_macro_recall, weighted_macro_f1

def print_results(epoch, accuracy, av_epoch_eval_loss, micro_precision, micro_recall, micro_f1,
                  weighted_macro_precision, weighted_macro_recall, weighted_macro_f1, final=False):
    if not final:
        print("\nEval results at end of epoch {}".format(epoch)+":\n")
    else:
        print("\nBest eval results were obtained on epoch {} and are shown below:\n".format(epoch))
    try:
        print("Eval accuracy at end of epoch {}: {:.2f}%".format(epoch, accuracy * 100))
    except Exception:
        pdb.set_trace()
    print("Average Eval loss for epoch {}: {}".format(epoch, str(av_epoch_eval_loss)))
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
                            help='the relative path to a model you wish to resume training from')
    args = cmd_parser.parse_args()
    if args.saved_model_path:
        saved_model_path = args.saved_model_path[0]
    else:
        saved_model_path = None

    main(args.data_path[0], saved_model_path)
