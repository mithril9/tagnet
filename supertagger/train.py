#Code for training the model

# WRITTEN BY:
#   John Torr (john.torr@cantab.net)

#standard library imports
import argparse
import copy
import os
from numpy import float64
from time import strftime
from typing import DefaultDict, List, Tuple

#third party imports
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.nn import CrossEntropyLoss
from torchtext.data.iterator import BucketIterator, Iterator

#local imports
from config import *
from constants import *
from file_handler import *
from lstm_model import LSTMTagger
from prepare_data import *
from utils import *


evalModelReturn = Tuple[float64, float, float64, float64, float64, float64, float64, float64]

models_folder = 'models'
device = torch.device("cuda:0" if (torch.cuda.is_available() and use_cuda_if_available) else "cpu")


def main(data_path: str, saved_model_path: str) -> None:
    """The main training function"""
    if saved_model_path:
        global embedding_dim, char_embedding_dim, hidden_dim, char_hidden_dim, use_bert_cased, \
            use_bert_uncased, use_bert_large
        embedding_dim, char_embedding_dim, hidden_dim, char_hidden_dim, use_bert_cased, use_bert_uncased, \
        use_bert_large = load_hyper_params(saved_model_path)
    if use_bert_uncased or use_bert_cased:
        use_bert = True
    else:
        use_bert = False
    if use_bert:
        train_iter, \
        val_iter, \
        word_to_ix, \
        ix_to_word, \
        tag_vocab, \
        char_to_ix = create_bert_datasets(
            data_path=data_path,
            mode=TRAIN,
            use_bert_cased=use_bert_cased,
            use_bert_uncased=use_bert_uncased,
            use_bert_large=use_bert_large
        )
        vocab_size = None
        word_vocab = None
    else:
        train_iter, \
        val_iter, \
        word_vocab, \
        tag_vocab, \
        char_to_ix = create_datasets(data_path=data_path, mode=TRAIN)
        #char_to_ix gets added to automatically with any characters (e.g. < >) encountered during evaluation, but we want to
        #save the original copy so that the char embeddings para can be computed, hence we create a copy here.
        word_to_ix, ix_to_word = word_vocab.stoi, word_vocab.itos
        vocab_size = len(word_to_ix)
    tag_to_ix, ix_to_tag = tag_vocab.stoi, tag_vocab.itos
    char_to_ix_original = copy.deepcopy(char_to_ix)
    model = LSTMTagger(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        tagset_size=len(tag_to_ix),
        char_embedding_dim=char_embedding_dim,
        char_hidden_dim=char_hidden_dim,
        char_vocab_size=len(char_to_ix),
        use_bert_cased=use_bert_cased,
        use_bert_uncased=use_bert_uncased,
        use_bert_large=use_bert_large
    )
    loss_function = CrossEntropyLoss(ignore_index=tag_to_ix['<pad>'])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if models_folder not in os.listdir(".."):
        os.mkdir(os.path.join("..", models_folder))
    if saved_model_path:
        av_train_losses, \
        av_eval_losses, \
        checkpoint_epoch, \
        best_accuracy, \
        lowest_av_eval_loss, \
        best_micro_precision, \
        best_micro_recall, \
        best_micro_f1, \
        best_weighted_macro_precision, \
        best_weighted_macro_recall, \
        best_weighted_macro_f1 = load_model(model=model,
                                            saved_model_path=saved_model_path,
                                            optimizer=optimizer)
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
        model.train()
        print('===============================')
        print('\n======== Epoch {} / {} ========'.format(epoch, end_epoch))
        batch_num = 0
        train_losses = []
        for batch in train_iter:
            batch_num += 1
            if batch_num % 20 == 0 or batch_num == 1:
                if batch_num != 1:
                    print("\nAverage Training loss for epoch {} at end of batch {}: {}".format(epoch, str(batch_num-1),sum(train_losses)/len(train_losses),4))
                print('\n======== at batch {} / {} ========'.format(batch_num, len(train_iter)))
            model.zero_grad()
            if use_bert:
                sentences_in, attention_masks, token_start_idx, targets, original_sentences = batch
                sentences_in = sentences_in.to(device)
                attention_masks = attention_masks.to(device)
                targets = targets.to(device)
                max_length = (attention_masks != 0).max(0)[0].nonzero()[-1].item()+1
                if max_length < sentences_in.shape[1]:
                    sentences_in = sentences_in[:, :max_length]
                    attention_masks = attention_masks[:, :max_length]
                sent_batch_size = sentences_in.shape[0]
                original_sentences_split = [sent.split() for sent in original_sentences]
                word_batch_size = max([len(sent) for sent in original_sentences_split])
                sent_lengths = [item for item in map(len, token_start_idx)]
            else:
                word_batch_size = batch.sentence.shape[0]
                sent_batch_size = batch.sentence.shape[1]
                sentences_in = batch.sentence.permute(1, 0).to(device)
                targets = batch.tags.permute(1, 0).reshape(sent_batch_size * word_batch_size).to(device)
                attention_masks = None
                token_start_idx = None
                original_sentences_split = None
                sent_lengths = train_iter.sent_lengths[batch_num - 1]
            words_in = get_words_in(
                sentences_in=sentences_in,
                char_to_ix=char_to_ix,
                ix_to_word=ix_to_word,
                device=device,
                original_sentences_split=original_sentences_split
            )
            model.init_hidden(sent_batch_size=sent_batch_size, device=device)
            tag_logits = model(
                sentences=sentences_in,
                words=words_in,
                char_hidden_dim=char_hidden_dim,
                sent_lengths=sent_lengths,
                word_batch_size=word_batch_size,
                device=device,
                attention_masks=attention_masks,
                token_start_idx=token_start_idx
            )
            mask = targets != 1
            loss = loss_function(tag_logits, targets)
            loss /= mask.float().sum()
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        av_train_losses.append(sum(train_losses) / len(train_losses))
        accuracy, av_eval_loss, micro_precision, micro_recall, micro_f1, weighted_macro_precision, \
        weighted_macro_recall, weighted_macro_f1 = eval_model(
            model=model,
            loss_function=loss_function,
            val_iter=val_iter,
            char_to_ix=char_to_ix,
            ix_to_word=ix_to_word,
            ix_to_tag=ix_to_tag,
            av_eval_losses=av_eval_losses,
            use_bert=use_bert
        )
        print_results(epoch, accuracy, av_eval_loss, micro_precision, micro_recall, micro_f1, weighted_macro_precision, weighted_macro_recall, weighted_macro_f1)
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
            save_model(
                epoch=checkpoint_epoch,
                model=model,
                optimizer=optimizer,
                av_train_losses=av_train_losses,
                av_eval_losses=av_eval_losses,
                model_file_name=model_file_name,
                word_to_ix=word_to_ix,
                ix_to_word=ix_to_word,
                word_vocab=word_vocab,
                tag_vocab=tag_vocab,
                char_to_ix=char_to_ix_original,
                models_folder=models_folder,
                embedding_dim=embedding_dim,
                char_embedding_dim=char_embedding_dim,
                hidden_dim=hidden_dim,
                char_hidden_dim=char_hidden_dim,
                accuracy=best_accuracy,
                av_eval_loss=lowest_av_eval_loss,
                micro_precision=best_micro_precision,
                micro_recall=best_micro_recall,
                micro_f1=best_micro_f1,
                weighted_macro_precision=best_weighted_macro_precision,
                weighted_macro_recall=best_weighted_macro_recall,
                weighted_macro_f1=best_weighted_macro_f1,
                use_bert_cased=use_bert_cased,
                use_bert_uncased=use_bert_uncased,
                use_bert_large=use_bert_large
                )
    print_results(
        epoch=checkpoint_epoch,
        accuracy=best_accuracy,
        av_eval_loss=lowest_av_eval_loss,
        micro_precision=best_micro_precision,
        micro_recall=best_micro_recall,
        micro_f1=best_micro_f1,
        weighted_macro_precision=best_weighted_macro_precision,
        weighted_macro_recall=best_weighted_macro_recall,
        weighted_macro_f1=best_weighted_macro_f1,
        final=True
        )
    plot_train_eval_loss(av_train_losses, av_eval_losses)

def eval_model(
        model: LSTMTagger,
        loss_function: CrossEntropyLoss,
        val_iter: BucketIterator,
        char_to_ix: DefaultDict[str, int],
        ix_to_word: List[str],
        ix_to_tag: List[str],
        av_eval_losses: List[str],
        use_bert: bool
) -> evalModelReturn:
    """
    Function for evaluating the model being trained.
    """
    model.eval()
    y_pred = []
    y_true = []
    print("\nEvaluating model...")
    with torch.no_grad():
        batch_num = 0
        eval_losses = []
        for batch in val_iter:
            batch_num += 1
            if use_bert:
                sentences_in, attention_masks, token_start_idx, targets, original_sentences = batch
                sentences_in = sentences_in.to(device)
                attention_masks = attention_masks.to(device)
                targets = targets.to(device)
                max_length = (attention_masks != 0).max(0)[0].nonzero()[-1].item() + 1
                if max_length < sentences_in.shape[1]:
                    sentences_in = sentences_in[:, :max_length]
                    attention_masks = attention_masks[:, :max_length]
                sent_batch_size = sentences_in.shape[0]
                original_sentences_split = [sent.split() for sent in original_sentences]
                word_batch_size = max([len(sent) for sent in original_sentences_split])
                sent_lengths = [item for item in map(len, token_start_idx)]
            else:
                word_batch_size = batch.sentence.shape[0]
                sent_batch_size = batch.sentence.shape[1]
                sentences_in = batch.sentence.permute(1, 0).to(device)
                targets = batch.tags.permute(1, 0).reshape(sent_batch_size * word_batch_size).to(device)
                attention_masks = None
                token_start_idx = None
                original_sentences_split = None
                sent_lengths = val_iter.sent_lengths[batch_num - 1]
            y_true += [ix_to_tag[ix.item()] for ix in targets]
            words_in = get_words_in(
                sentences_in=sentences_in,
                char_to_ix=char_to_ix,
                ix_to_word=ix_to_word,
                device=device,
                original_sentences_split=original_sentences_split
            )
            model.init_hidden(sent_batch_size=sent_batch_size, device=device)
            tag_logits = model(
                sentences=sentences_in,
                words=words_in,
                char_hidden_dim=char_hidden_dim,
                sent_lengths=sent_lengths,
                word_batch_size=word_batch_size,
                device=device,
                attention_masks=attention_masks,
                token_start_idx=token_start_idx
            )
            eval_loss = loss_function(tag_logits, targets)
            mask = targets != 1
            eval_loss /= mask.float().sum()
            eval_losses.append(eval_loss.item())
            pred = categoriesFromOutput(tag_logits, ix_to_tag)
            y_pred += pred
        av_eval_losses.append(sum(eval_losses) / len(eval_losses))
        y_true, y_pred = remove_pads(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        micro_precision, micro_recall, micro_f1, support = precision_recall_fscore_support(y_true, y_pred,
                                                                                           average='micro')
        weighted_macro_precision, weighted_macro_recall, weighted_macro_f1, _ = precision_recall_fscore_support(y_true,
                                                                                                                y_pred,
                                                                                                                average='weighted')
        av_eval_loss = sum(eval_losses) / len(eval_losses)

    return accuracy, av_eval_loss, micro_precision, micro_recall, micro_f1, weighted_macro_precision, \
           weighted_macro_recall, weighted_macro_f1


def print_results(
        epoch: int,
        accuracy: float64,
        av_eval_loss: float,
        micro_precision: float64,
        micro_recall: float64,
        micro_f1: float64,
        weighted_macro_precision: float64,
        weighted_macro_recall: float64,
        weighted_macro_f1: float64,
        final=False
) -> None:
    if not final:
        print("\nEval results at end of epoch {}".format(epoch)+":\n")
    else:
        print("\nBest eval results were obtained on epoch {} and are shown below:\n".format(epoch))
    print("Eval accuracy: {:.2f}%".format(accuracy * 100))
    print("Average Eval loss: {}".format(str(av_eval_loss,4)))
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
    if use_bert_cased and use_bert_uncased:
        raise Exception("Both use_bert_cased and use_bert_uncased are set to True in config.py!! \
        Please edit the file so that at most one of these is set to true.")
    if args.saved_model_path:
        saved_model_path = args.saved_model_path[0]
    else:
        saved_model_path = None
    main(args.data_path[0], saved_model_path)
