"""Code for testing a saved model"""

from lstm_model import LSTMTagger
from prepare_data import *
import argparse
import pdb
from file_handler import *
from utils import *
import copy

models_folder = 'models'


def main(data_path, dest_path, saved_model_path):
    EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM, CHAR_HIDDEN_DIM = load_hyper_params(saved_model_path)
    word_vocab, tag_vocab, char_to_ix = load_vocab_and_char_to_ix(saved_model_path)
    word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = word_vocab.stoi, word_vocab.itos, tag_vocab.stoi, tag_vocab.itos
    sent_iter, raw_sentences = create_datasets(data_path, mode='tag', word_to_ix=copy.deepcopy(word_to_ix), word_vocab=copy.deepcopy(word_vocab), tag_vocab=copy.deepcopy(tag_vocab))
    dest_file = open(dest_path, 'w')
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM,\
                       len(char_to_ix))
    load_model(model=model, saved_model_path=saved_model_path)
    #torch.autograd.set_detect_anomaly(True)
    print("\ntagging sentences using model: "+saved_model_path+'\n')
    model.eval()
    y_pred = []
    with torch.no_grad():
        batch_num = 0
        for batch in sent_iter:
            batch_num += 1
            word_batch_size = batch.sentence.shape[0]
            sent_batch_size = batch.sentence.shape[1]
            model.init_hidden(sent_batch_size)
            sentences_in = batch.sentence.permute(1, 0)
            words_in = get_words_in(sentences_in, char_to_ix, ix_to_word)
            tag_logits = model(sentences_in, words_in, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, sent_iter.sent_lengths[batch_num-1], word_batch_size)
            pred = categoriesFromOutput(tag_logits, ix_to_tag)
            dest_file.write(" ".join([ix_to_word[ix.item()] for ix in sentences_in[0]])+'\t'+" ".join(pred)+'\n')
    dest_file.close()
    print("tagging complete")


if __name__ == '__main__':
    cmd_parser = argparse.ArgumentParser(description='command line arguments.')
    cmd_parser.add_argument('--data-path', dest='data_path', type=str, nargs=1, help='the path to the file containing the sentences to be tagged')
    cmd_parser.add_argument('--dest-path', dest='dest_path', type=str, nargs=1, help='the path to the file you want the tagged sentences to be saved in (file will be created/overwritten)')
    cmd_parser.add_argument('--model-path', dest='saved_model_path', type=str, nargs=1,
                            help='the relative path to the model you wish to resume training from')
    args = cmd_parser.parse_args()

    main(args.data_path[0], args.dest_path[0], args.saved_model_path[0])
