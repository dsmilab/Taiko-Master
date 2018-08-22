from .preprocessing.performance import *

import pandas as pd
import re

ACC_REGEX = '^\w*_A[XYZ]{0,2}_\w*$'
GYR_REGEX = '^\w*_G[XYZ]{0,2}_\w*$'
NEAR_REGEX = '^[LR]\d+$'

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

    return event_primitive_df
