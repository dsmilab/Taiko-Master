import numpy as np
from sklearn import metrics

__all__ = ['get_processing_score',
           'my_f1_score']


def get_processing_score(pred_score_list):
    score = 0

    broken = False
    leading_space = True

    for d in pred_score_list:
        if d == 10:
            d = 0
            if not leading_space:
                broken = True
        else:
            leading_space = False
        score = score * 10 + d

    if broken or leading_space:
        return None

    return score


def my_f1_score(y_pred, y):
    y = y.get_label()
    y_pred = y_pred.reshape(len(np.unique(y)), -1).argmax(axis=0)
    return "f1-score", metrics.f1_score(y, y_pred, average="macro"), True
