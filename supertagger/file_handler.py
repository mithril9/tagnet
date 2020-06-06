"""Code for training the model"""

import torch
import os, pdb

def load_vocab_and_char_to_ix(saved_model_path):
    print("Attempting to load saved ix mappings from: " + saved_model_path)
    checkpoint = torch.load(saved_model_path)
    return checkpoint['word_vocab'], checkpoint['tag_vocab'], checkpoint['char_to_ix']


def load_model(model, saved_model_path, optimizer=None):
    print("Attempting to load saved model checkpoint from: " + saved_model_path)
    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Successfully loaded model..")
    if not optimizer:
        return
    else:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['av_train_losses'], checkpoint['av_eval_losses'], checkpoint['checkpoint_epoch'], \
               checkpoint['loss'], checkpoint['accuracy'], checkpoint['av_epoch_eval_loss'], \
               checkpoint['micro_precision'], checkpoint['micro_recall'], checkpoint['micro_f1'], \
               checkpoint['weighted_macro_precision'], checkpoint['weighted_macro_recall'], \
               checkpoint['weighted_macro_f1']

def load_hyper_params(saved_model_path):
    print("Attempting to load saved hyperparameters from: " + saved_model_path)
    checkpoint = torch.load(saved_model_path)
    return checkpoint['EMBEDDING_DIM'], checkpoint['CHAR_EMBEDDING_DIM'], checkpoint['HIDDEN_DIM'], \
           checkpoint['CHAR_HIDDEN_DIM']

def save_model(epoch, model, optimizer, loss, av_train_losses, av_eval_losses, model_file_name,
               word_vocab, tag_vocab, char_to_ix, models_folder, EMBEDDING_DIM, CHAR_EMBEDDING_DIM, HIDDEN_DIM,
               CHAR_HIDDEN_DIM, accuracy, av_epoch_eval_loss, micro_precision, micro_recall, micro_f1,
               weighted_macro_precision, weighted_macro_recall, weighted_macro_f1):
    try:
        os.remove("../"+os.path.join(models_folder, model_file_name))
    except FileNotFoundError:
        pass
    torch.save({
            'checkpoint_epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'av_train_losses': av_train_losses,
            'av_eval_losses': av_eval_losses,
            'word_vocab': word_vocab,
            'tag_vocab': tag_vocab,
            'char_to_ix': char_to_ix,
            'EMBEDDING_DIM': EMBEDDING_DIM,
            'CHAR_EMBEDDING_DIM': CHAR_EMBEDDING_DIM,
            'HIDDEN_DIM': HIDDEN_DIM,
            'CHAR_HIDDEN_DIM': CHAR_HIDDEN_DIM,
            'accuracy': accuracy,
            'av_epoch_eval_loss': av_epoch_eval_loss,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'weighted_macro_precision': weighted_macro_precision,
            'weighted_macro_recall': weighted_macro_recall,
            'weighted_macro_f1': weighted_macro_f1
    }, "../"+os.path.join(models_folder, model_file_name))
    print("Model with lowest average eval loss successfully saved as: "+"../"+os.path.join(models_folder, model_file_name))
