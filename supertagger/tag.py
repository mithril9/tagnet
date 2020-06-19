"""Code for testing a saved model"""


#standard library imports
import argparse
import copy

#third party imports
from transformers import BertTokenizer

#local imports
from constants import *
from file_handler import *
from lstm_model import LSTMTagger
from prepare_data import *
from utils import *


models_folder = 'models'
device = torch.device("cuda:0" if (torch.cuda.is_available() and use_cuda_if_available) else "cpu")


def main(data_path: str, dest_path:str, saved_model_path: str) -> None:
    embedding_dim, char_embedding_dim, hidden_dim, char_hidden_dim, \
    use_bert_cased, use_bert_uncased, use_bert_large = load_hyper_params(saved_model_path)
    if use_bert_uncased:
        if use_bert_large:
            tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    elif use_bert_cased:
        if use_bert_large:
            tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    else:
        tokenizer = None
    if use_bert_uncased or use_bert_cased:
        use_bert = True
    else:
        use_bert = False
    word_to_ix, ix_to_word, word_vocab, tag_vocab, char_to_ix = load_vocab_and_char_to_ix(saved_model_path)
    tag_to_ix, ix_to_tag = tag_vocab.stoi, tag_vocab.itos
    if use_bert:
        vocab_size = None
    else:
        vocab_size = len(word_to_ix)
    sentences, sent_tensors = prepare_untagged_data(
        data_path=data_path,
        word_to_ix=copy.deepcopy(word_to_ix),
        device=device,
        tokenizer=tokenizer,
        use_bert=use_bert
    )
    dest_file = open(dest_path, 'w')
    model = LSTMTagger(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        tagset_size=len(tag_to_ix),
        char_embedding_dim=char_embedding_dim,
        char_hidden_dim=char_hidden_dim,\
        char_vocab_size=len(char_to_ix),
        use_bert_cased=use_bert_cased,
        use_bert_uncased=use_bert_uncased,
        use_bert_large=use_bert_large
    )
    load_model(model=model, saved_model_path=saved_model_path)
    #torch.autograd.set_detect_anomaly(True)
    model.to(device)
    print("\ntagging sentences using model: "+saved_model_path+'\n')
    model.eval()
    with torch.no_grad():
        sent_num = 0
        for sent_tensor in sent_tensors:
            sentence = sentences[sent_num]
            sent_num += 1
            word_batch_size = len(sentence)
            sent_batch_size = 1
            if use_bert:
                #ADD CLS and SEP before HERE!!!
                subwords = list(map(tokenizer.tokenize, sentence))
                subword_lengths = list(map(len, subwords))
                token_start_idx = [list(np.cumsum([0] + subword_lengths))[1:]]
                sent_lengths = [len(token_start_idx[0])]
                attention_masks = torch.tensor([1, 1]+[1 for x in token_start_idx[0]]).unsqueeze(0)
                original_sentences_split = [sentence]
            else:
                attention_masks = None
                token_start_idx = None
                original_sentences_split = None
                sent_lengths = [len(sentence)]
            words_in = get_words_in(
                sentences_in=sent_tensor,
                char_to_ix=char_to_ix,
                ix_to_word=ix_to_word,
                device=device,
                original_sentences_split=original_sentences_split
            )
            model.init_hidden(sent_batch_size=sent_batch_size, device=device)
            tag_logits = model(
                sentences=sent_tensor,
                words=words_in,
                char_hidden_dim=char_hidden_dim,
                sent_lengths=sent_lengths,
                word_batch_size=word_batch_size,
                device=device,
                attention_masks=attention_masks,
                token_start_idx=token_start_idx
            )
            pred = categoriesFromOutput(tag_logits, ix_to_tag)
            dest_file.write(" ".join(sentence)+'\t'+" ".join(pred)+'\n')
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
