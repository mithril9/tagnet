#Useful utility functions

# WRITTEN BY:
#   John Torr (john.torr@cantab.net)

#standard library imports
from typing import List, Tuple

#third party imports
import matplotlib.pyplot as plt
from torch import Tensor

#local imports


def categoriesFromOutput(tag_scores: Tensor, ix_to_tag: List[str]) -> List[str]:
    """
    Takes a torch.Tensor of scores for each tag for each word in each sentence
    and returns a flattened list of strings, where each string corresponds to a tag
    in some sentence.
    """
    predictions = []
    top_n, top_i = tag_scores.topk(1, dim=1)
    #unroll all batches into one long sequence for convenience
    top_i = top_i.view(top_i.shape[0]*top_i.shape[1])
    for prediction in top_i:
        pred = ix_to_tag[prediction.item()]
        predictions.append(pred)
    return predictions

def plot_train_eval_loss(av_train_losses: List[float], av_eval_losses: List[float]) -> None:
    plt.xlabel("n epochs")
    plt.ylabel("loss")
    plt.plot(av_train_losses, label='train')
    plt.plot(av_eval_losses, label='eval')
    plt.legend(loc='upper left')
    plt.show()

def remove_pads(y_true: List[str], y_pred: List[str]) -> Tuple[List[str], List[str]]:
    """
    Takes as input two lists of strings corresponding to the predicted and actual tags
    and returns the same lists except that any <pad> tags in y_true are removed and the tags
    corresponding to the same index position in y_pred are also removed.
    """
    new_y_true = []
    new_y_pred = []
    for i in range(len(y_true)):
        if y_true[i] != "<pad>":
            new_y_true.append(y_true[i])
            new_y_pred.append(y_pred[i])
    return new_y_true, new_y_pred
