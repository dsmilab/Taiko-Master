from .config import *
import pandas as pd
import numpy as np
from glob import glob
from scipy.stats import mode

__all__ = ['Profile']


class Profile(object):
    _SAMPLING_RATE = '0.01S'

    _LABELS = [
        'right_don',
        'left_don',
        'right_ka',
        'left_ka',
        'big_don',
        'big_ka',
        'pause',
        'drumroll',
    ]

    _HANDEDNESS = {
        'left': 'L',
        'right': 'R',
    }

    def __init__(self, drummer_name):
        self._drummer_name = drummer_name
        self.__build()

    def __build(self):
        self._profile = {}

        for label_kwd in Profile._LABELS:
            self._profile[label_kwd] = {}
            for handedness_kwd, handedness_label in Profile._HANDEDNESS.items():
                self._profile[label_kwd][handedness_label] = []
                files = glob(posixpath.join(PROFILE_DIR_PATH, label_kwd, handedness_kwd, self._drummer_name + '_*.csv'))
                for file_ in files:
                    df = pd.read_csv(file_)
                    df = self.__build_play_df(df)
                    self._profile[label_kwd][handedness_label].append(df)

    @staticmethod
    def __build_play_df(raw_arm_df, calibrate=True, resample=True):

        play_df = raw_arm_df.copy()

        # resample for more samples
        if resample:
            play_df.loc[:, 'timestamp'] = pd.to_datetime(play_df['timestamp'], unit='s')
            play_df.loc[:, 'timestamp'] = play_df['timestamp'].apply(
                lambda x: x.tz_localize('UTC').tz_convert('Asia/Taipei'))
            play_df = play_df.set_index('timestamp').resample(Profile._SAMPLING_RATE).mean()
            play_df = play_df.interpolate(method='linear')
            play_df.reset_index(inplace=True)
            play_df.loc[:, 'timestamp'] = play_df['timestamp'].apply(lambda x: x.timestamp())
            play_df.fillna(method='ffill', inplace=True)

        # implement zero adjust for needed columns
        if calibrate:
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
    def profile(self):
        return self._profile
