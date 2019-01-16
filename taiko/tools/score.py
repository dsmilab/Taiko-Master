import numpy as np
from sklearn import metrics

__all__ = ['get_processing_score',
           'my_f1_score',
           'get_gained_score_multiplier']


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


def get_gained_score_multiplier(y_pred, y_test):
    if y_test == 0:
        return 0
    if y_test in [1, 2] and y_test == y_pred:
        return 1
    if y_test == 3:
        if y_pred == 1:
            return 1
        elif y_pred == 3:
            return 2
    if y_test == 4:
        if y_pred == 2:
            return 1
        elif y_pred == 4:
            return 2
    return 0
