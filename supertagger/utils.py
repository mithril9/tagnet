"""Useful utility functions"""


def categoriesFromOutput(tag_scores, ix_to_tag):
    predictions = []
    top_n, top_i = tag_scores.topk(1, dim=1)
    #unroll all batches into one long sequence for convenience
    top_i = top_i.view(top_i.shape[0]*top_i.shape[1])
    for prediction in top_i:
        pred = ix_to_tag[prediction.item()]
        predictions.append(pred)
    return predictions
