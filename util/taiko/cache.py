from .preprocessing.performance import *

import pandas as pd
import re

ACC_REGEX = '^\w*_A[XYZ]{0,2}_\w*$'
GYR_REGEX = '^\w*_G[XYZ]{0,2}_\w*$'
NEAR_REGEX = '^[LR]\d+$'
HIT_TYPE_REGEX = '^(hit_type|[A-Z]+\d)$'

__all__ = ['get_event_primitive_df']


def get_event_primitive_df(who_id, song_id, order_id, scaling=True, resampling=True, acc=True, gyr=True, near=True):
    if who_id == 4 and song_id == 4 and order_id in [3, 4]:
        raise ValueError('Corrupted EP')

    boolean_dict = {True: 'y', False: 'n'}
    resampling_boolean = {True: '0.02S', False: None}

    filename = '%d-%d-%d-%s%s' % (who_id, song_id, order_id, boolean_dict[scaling], boolean_dict[resampling])

    try:
        event_primitive_df = pd.read_csv('CSV/event_primitive/' + filename + '.csv')
    except FileNotFoundError:
        event_primitive_df = get_performance(who_id,
                                             song_id,
                                             order_id,
                                             scaling,
                                             resampling_boolean[resampling]).event_primitive_df
        filename = '%d-%d-%d-%s%s' % (who_id, song_id, order_id, boolean_dict[scaling], boolean_dict[resampling])
        event_primitive_df.to_csv('CSV/event_primitive/' + filename + '.csv', index=False, float_format='%.4g')

    if not acc:
        columns = [col for col in event_primitive_df.columns if re.match(ACC_REGEX, col)]
        event_primitive_df.drop(columns, axis=1, inplace=True)

    if not gyr:
        columns = [col for col in event_primitive_df.columns if re.match(GYR_REGEX, col)]
        event_primitive_df.drop(columns, axis=1, inplace=True)

    if not near:
        columns = [col for col in event_primitive_df.columns if re.match(NEAR_REGEX, col)]
        event_primitive_df.drop(columns, axis=1, inplace=True)

    columns = event_primitive_df.columns
    columns = [col for col in columns if re.match(HIT_TYPE_REGEX, col)]
    for col in columns:
        event_primitive_df.loc[:, col] = event_primitive_df[col].apply(transform_hit_type_label_big_small)
    return event_primitive_df


def transform_hit_type_label(label):
    """
    Relabel the column.

    :param label: original label
    :return: transformed label
    """

    if label in [1, 2, 3, 4]:
        return 1
    elif label in [5, 6, 7]:
        return 2
    return 0


def transform_hit_type_label_dong_ka(label):
    """
    Relabel the column.

    :param label: original label
    :return: transformed label
    """

    if label in [1, 3]:
        return 1
    elif label in [2, 4]:
        return 2
    elif label in [5, 6, 7]:
        return 3
    return 0


def transform_hit_type_label_ship(label):
    """
    Relabel the column.

    :param label: original label
    :return: transformed label
    """

    if label in [1, 2, 3, 4]:
        return 1
    elif label in [5, 6]:
        return 2
    elif label in [7]:
        return 3
    return 0


def transform_hit_type_label_big_small(label):
    """
    Relabel the column.

    :param label: original label
    :return: transformed label
    """

    if label in [1, 2]:
        return 1
    elif label in [3, 4]:
        return 2
    elif label in [5, 6, 7]:
        return 3
    return 0
