from ..config import *
from ..io import *

import pandas as pd
from pandas import datetime
import numpy as np
from scipy.stats import mode

RESAMPLE_RATE = '0.02S'

__all__ = ['get_play']


class _Play(object):

    def __init__(self, rec, is_zero_adjust=True):
        self._play_dict = {}
        self._start_time, self._end_time = None, None
        self._first_hit_time = None

        self.__set_hw_time(rec)
        self._play_dict['R'] = self.__build_play_df(RIGHT_HAND, is_zero_adjust, RESAMPLE_RATE)
        self._play_dict['L'] = self.__build_play_df(LEFT_HAND, is_zero_adjust, RESAMPLE_RATE)

    def __set_hw_time(self, rec):
        self._start_time = rec['hw_start_time']
        self._end_time = rec['hw_end_time']
        self._first_hit_time = rec['first_hit_time']

    def __build_play_df(self, handedness, is_zero_adjust, resample=None):
        """
        After setting duration of the song, build dataframe of a play.

        :param df: original dataframe
        :param modes: default is "None", adjust zero by this own case, otherwise will by this param
        :return: cropped and zero-adjusted dataframe, attributes' modes
        """

        df = load_arm_df(handedness)
        play_df = df[(df['timestamp'] >= self._start_time) &
                     (df['timestamp'] <= self._end_time)].copy()

        if resample is not None:
            play_df.loc[:, 'timestamp'] = pd.to_datetime(play_df['timestamp'], unit='s')
            play_df.loc[:, 'timestamp'] = play_df['timestamp'].apply(
                lambda x: x.tz_localize('UTC').tz_convert('Asia/Taipei'))
            play_df = play_df.set_index('timestamp').resample(resample).mean()
            play_df = play_df.interpolate(method='linear')
            play_df.reset_index(inplace=True)
            play_df.loc[:, 'timestamp'] = play_df['timestamp'].apply(lambda x: x.timestamp())
            play_df.fillna(method='ffill', inplace=True)

        if is_zero_adjust:
            modes_dict = {}
            copy_df = play_df.copy()
            for col in ZERO_ADJ_COL:
                mode_ = mode(copy_df[col])[0]
                modes_dict[col] = mode_

            # only considered attributes need zero adjust
            for col in ZERO_ADJ_COL:
                copy_df.loc[:, col] = copy_df[col] - modes_dict[col]

            play_df = copy_df

        return play_df

    @property
    def play_dict(self):
        return self._play_dict

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def first_hit_time(self):
        return self._first_hit_time


def get_play(who_id, song_id, order_id, is_adjust_zero=True):
    rec = get_record(who_id, song_id, order_id)
    return _Play(rec, is_adjust_zero)
