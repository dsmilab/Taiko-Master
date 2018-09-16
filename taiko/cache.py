from .preprocessing.performance import *
from .config import *

import pandas as pd
import re


__all__ = ['get_event_primitive_df']


def get_event_primitive_df(who_id, song_id, order_id,
                           scaling=True,
                           resampling=True,
                           acc=True,
                           gyr=True,
                           near=True,
                           over_sampled=False,
                           label_group='origin'):
    if who_id == 4 and song_id == 4 and order_id in [3, 4]:
        raise ValueError('Corrupted EP')

    if label_group not in ['origin', 'single_stream', 'don_ka', 'big_small', 'balloon']:
        raise ValueError('label_group Error!')

    boolean_dict = {True: 'y', False: 'n'}

    filename = '%d-%d-%d-%s' % (who_id, song_id, order_id, boolean_dict[resampling])

    try:
        event_primitive_df = pd.read_csv('CSV/event_primitive/' + filename + '.csv')
    except FileNotFoundError:
        event_primitive_df = get_performance(who_id,
                                             song_id,
                                             order_id,
                                             scale=False,
                                             resample=resampling).event_primitive_df

        event_primitive_df.to_csv('CSV/event_primitive/' + filename + '.csv', index=False, float_format='%.4g')

    # set corresponding transform hit type function
    transform_hit_type_label = transform_hit_type_label_origin
    if label_group == 'single_stream':
        transform_hit_type_label = transform_hit_type_label_single_stream
    elif label_group == 'don_ka':
        transform_hit_type_label = transform_hit_type_label_don_ka
    elif label_group == 'big_small':
        transform_hit_type_label = transform_hit_type_label_big_small
    elif label_group == 'balloon':
        transform_hit_type_label = transform_hit_type_label_balloon

    columns = event_primitive_df.columns
    columns = [col for col in columns if re.match(HIT_TYPE_REGEX, col)]
    for col in columns:
        event_primitive_df.loc[:, col] = event_primitive_df[col].apply(transform_hit_type_label)

    event_primitive_df.dropna(inplace=True)
    event_primitive_df.reset_index(drop=True, inplace=True)

    if over_sampled:
        event_primitive_df = do_over_sampled(event_primitive_df)

    if scaling:
        event_primitive_df = do_scaling(event_primitive_df)

    if not acc:
        columns = [col for col in event_primitive_df.columns if re.match(ACC_REGEX, col)]
        event_primitive_df.drop(columns, axis=1, inplace=True)

    if not gyr:
        columns = [col for col in event_primitive_df.columns if re.match(GYR_REGEX, col)]
        event_primitive_df.drop(columns, axis=1, inplace=True)

    if not near:
        columns = [col for col in event_primitive_df.columns if re.match(NEAR_REGEX, col)]
        event_primitive_df.drop(columns, axis=1, inplace=True)

    return event_primitive_df


def transform_hit_type_label_origin(label):
    """
    Relabel the column.

    :param label: original label
    :return: transformed label
    """
    return label


def transform_hit_type_label_single_stream(label):
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


def transform_hit_type_label_don_ka(label):
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


def transform_hit_type_label_balloon(label):
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
