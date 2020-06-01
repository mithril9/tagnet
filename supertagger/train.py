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
from sklearn.metrics import accuracy_score


def main(data_path):
    train_iter, val_iter, word_to_ix, ix_to_word, tag_to_ix, ix_to_tag, char_to_ix = create_datasets(data_path)
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM,\
                       len(char_to_ix))
    #loss_function = nn.NLLLoss()
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=tag_to_ix['<pad>'])
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    #torch.autograd.set_detect_anomaly(True)
    print("training..\n")
    model.train()
    for epoch in range(num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        print('\n======== Epoch {} / {} ========'.format(epoch + 1, num_epochs)+'\n')
        if epoch:
            print("Current training loss: " + str(loss.item()))
        batch_num = 0
        for batch in train_iter:
            batch_num += 1
            if batch_num % 20 == 0 or batch_num == 1:
                print('======== Batch {} / {} ========'.format(batch_num, len(train_iter)))
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
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        batch_num = 0
        for batch in val_iter:
            batch_num += 1
            word_batch_size = batch.sentence.shape[0]
            sent_batch_size = batch.sentence.shape[1]
            model.init_hidden(sent_batch_size)
            sentences_in = batch.sentence.permute(1, 0)
            targets = batch.tags.permute(1,0).reshape(sent_batch_size*word_batch_size)
            y_true += [ix_to_tag[ix.item()] for ix in targets]
            words_in = get_words_in(sentences_in, char_to_ix, ix_to_word)
            tag_logits = model(sentences_in, words_in, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, val_iter.sent_lengths[batch_num-1], word_batch_size, eval=True)
            loss = loss_function(tag_logits, targets)
            print("Eval loss: " + str(loss.item()))
            pred = categoriesFromOutput(tag_logits, ix_to_tag)
            print(pred)
            y_pred += pred
            #for target in targets:
                #y_true += [ix_to_tag[y.item()] for y in target]
        accuracy = accuracy_score(y_true, y_pred)
        print("Eval accuracy: {:.2f}%".format(accuracy*100))

def categoriesFromOutput(tag_scores, ix_to_tag):
    predictions = []
    top_n, top_i = tag_scores.topk(1, dim=1)
    #unroll all batches into one long sequence for convenience
    top_i = top_i.view(top_i.shape[0]*top_i.shape[1])
    for prediction in top_i:
        pred = ix_to_tag[prediction.item()]
        predictions.append(pred)
    return predictions


if __name__ == '__main__':
    cmd_parser = argparse.ArgumentParser(description='command line arguments.')
    cmd_parser.add_argument('--data-path', dest='data_path', type=str, nargs=1, help='the path to the folder containing the data files')

    args = cmd_parser.parse_args()

    main(args.data_path[0])
