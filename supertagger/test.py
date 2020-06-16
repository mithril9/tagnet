"""Code for testing a saved model"""

from lstm_model import LSTMTagger
from prepare_data import *
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from file_handler import *
from utils import *
import copy
from config import *
from constants import *

models_folder = 'models'
device = torch.device("cuda:0" if (torch.cuda.is_available() and use_cuda_if_available) else "cpu")


def main(data_path: str, saved_model_path: str) -> None:
    embedding_dim, char_embedding_dim, hidden_dim, char_hidden_dim = load_hyper_params(saved_model_path)
    word_to_ix, ix_to_word, tag_vocab, char_to_ix = load_vocab_and_char_to_ix(saved_model_path)
    tag_to_ix, ix_to_tag = tag_vocab.stoi, tag_vocab.itos
    test_iter = create_datasets(
        data_path=data_path,
        mode=TEST,
        word_to_ix=copy.deepcopy(word_to_ix),
        word_vocab=copy.deepcopy(word_vocab),
        tag_vocab=copy.deepcopy(tag_vocab)
    )
    model = LSTMTagger(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        vocab_size=len(word_to_ix),
        tagset_size=len(tag_to_ix),
        char_embedding_dim=char_embedding_dim,
        char_hidden_dim=char_hidden_dim,
        char_vocab_size=len(char_to_ix)
    )
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=tag_to_ix['<pad>'])
    load_model(model=model, saved_model_path=saved_model_path)
    mode.to(device)
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
            sentences_in = batch.sentence.permute(1, 0).to(device)
            targets = batch.tags.permute(1,0).reshape(sent_batch_size*word_batch_size).to(device)
            y_true += [ix_to_tag[ix.item()] for ix in targets]
            words_in = get_words_in(
                sentences_in=sentences_in,
                char_to_ix=char_to_ix,
                ix_to_word=ix_to_word,
                device=device
            )
            tag_logits = model(
                sentences=sentences_in,
                words=words_in,
                char_hidden_dim=char_hidden_dim,
                sent_lengths=test_iter.sent_lengths[batch_num-1],
                word_batch_size=word_batch_size,
                device=device
            )
            test_loss = loss_function(tag_logits, targets)
            test_losses.append(round(test_loss.item(), 2))
            pred = categoriesFromOutput(tag_logits, ix_to_tag)
            y_pred += pred
        av_test_losses.append(sum(test_losses)/len(test_losses))
        accuracy = accuracy_score(y_true, y_pred)
        micro_precision, micro_recall, micro_f1, support = precision_recall_fscore_support(y_true, y_pred, average='micro')
        weighted_macro_precision, weighted_macro_recall, weighted_macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        av_test_loss = sum(test_losses)/len(test_losses)
        print("Test accuracy: {:.2f}%".format(accuracy*100))
        print("Average Test loss: {}".format(str(av_test_loss)))
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
