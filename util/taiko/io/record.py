from ..config import *
from ..tools.timestamp import *

import pandas as pd
import numpy as np

__all__ = ['load_drummer_df', 'get_record']

TAILED_ADDITIONAL_TIME = 30


class _Record(object):
    """
    Handle all drummers' info.

    :protected attributes:
        drummer_df: dataframe about performance of drummers
    """

    def __init__(self):
        self._drummer_df = None

        self.__load_drummer_csv()

    def __load_drummer_csv(self):
        """
        Load CSV files which contain information about drummers' performance.

        :return: performance dataframe being in additional of timestamp pre-processing.
        """

        # read drummers' plays
        df = pd.read_csv(PLAY_TABLE_PATH)

        # read song's information and merge it
        tmp_df = pd.read_csv(SONG_TABLE_PATH, dtype={
            'song_length': np.int16
        })
        df = df.merge(tmp_df, how='left', left_on='song_id', right_on='song_id')

        # read drummers' personal information and merge it
        tmp_df = pd.read_csv(DRUMMER_TABLE_PATH)
        df = df.merge(tmp_df, how='left', left_on='drummer_id', right_on='id')

        # translate UTC timestamp into hardware timestamp
        df['hw_start_time'] = df['start_time'].apply(get_hwclock_time)
        df['hw_end_time'] = df['hw_start_time'] + df['song_length'] + TAILED_ADDITIONAL_TIME

        df.drop('id', axis=1, inplace=True)

        self._drummer_df = df

    @property
    def drummer_df(self):
        return self._drummer_df


def load_drummer_df():
    return _Record().drummer_df


def get_record(who_id, song_id, order_id):
    """
    Get the record from drummer info.

    :param who_id: # of drummer
    :param song_id: # of song
    :param order_id: # of performance repetitively
    :return: the desired unique record
    """
    df = load_drummer_df()
    df = df[(df['drummer_id'] == who_id) &
            (df['song_id'] == song_id) &
            (df['performance_order'] == order_id)]

    if len(df) == 0:
        raise KeyError('No matched performances.')
    elif len(df) > 1:
        raise KeyError('Duplicated matched performances.')

    # assume matched case is unique
    row = df.iloc[0]

    return row
