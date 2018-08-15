__all__ = ['get_processing_score']


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
