"""Code for training the model"""

from lstm_model import LSTMTagger
from config import *
from prepare_data import *
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import pdb, json


def main(data_path):
    train_iter, val_iter, word_to_ix, ix_to_word, tag_to_ix, ix_to_tag, char_to_ix = create_datasets(data_path)
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM,\
                       len(char_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    torch.autograd.set_detect_anomaly(True)
    print("training..\n")
    for epoch in range(num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        if epoch == 0 or (epoch+1) % 20 == 0:
            print('======== Epoch {} / {} ========'.format(epoch + 1, num_epochs))
        for batch in train_iter:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            word_batch_size = batch.sentence.shape[0]
            sent_batch_size = batch.sentence.shape[1]
            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden(sent_batch_size)
            model.char_hidden = model.init_char_hidden(word_batch_size)
            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentences_in = batch.sentence
            words_in = get_words_in(sentences_in, char_to_ix, ix_to_word)
            targets = batch.tags
            # Step 3. Run our forward pass.
            tag_scores = model(sentences_in, words_in, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

    # See what the scores are after training
    with torch.no_grad():
        sentence_in = prepare_sequence(data[0][0], word_to_ix)
        words_in = [prepare_sequence(word, char_to_ix) for word in data[0][0]]
        tag_scores = model(sentence_in, words_in, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM)

        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
        print(tag_scores)


if __name__ == '__main__':
    cmd_parser = argparse.ArgumentParser(description='command line arguments.')
    cmd_parser.add_argument('--data-path', dest='data_path', type=str, nargs=1, help='the path to the folder containing the data files')

    args = cmd_parser.parse_args()

    main(args.data_path[0])
