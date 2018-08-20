from .preprocessing.performance import *

import pandas as pd


__all__ = ['get_event_primitive_df']


def get_event_primitive_df(who_id, song_id, order_id, scaling=True, resampling=True):
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

    return event_primitive_df
