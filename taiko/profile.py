from .config import *
from .primitive import *
import pandas as pd
import numpy as np
from glob import glob
from scipy.stats import mode

__all__ = ['Profile']


class Profile(object):
    _SAMPLING_RATE = '0.01S'

    _LABELS = {
        'right_don': 1,
        'left_don': 1,
        'right_ka': 2,
        'left_ka': 2,
        'big_don': 3,
        'big_ka': 4,
        'pause': 0,
        'drumroll': 5,
    }

    _HANDEDNESS = {
        'left': 'L',
        'right': 'R',
    }

    def __init__(self, drummer_name):
        self._drummer_name = drummer_name
        self.__build()
        self.__build_primitive_df()

    def __build(self):
        self._profile = {}

        for label_kwd in Profile._LABELS.keys():
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

    def __build_primitive_df(self):
        profile_primitive_df = pd.DataFrame()
        for label_kwd, label in Profile._LABELS.items():
            primitive_df = pd.DataFrame()
            for _, handedness_label in Profile._HANDEDNESS.items():
                tmp_primitive_mat = []
                for df in self._profile[label_kwd][handedness_label]:
                    start_time = df['timestamp'].iloc[0]
                    end_time = df['timestamp'].iloc[-1]
                    delta_t = 0.2
                    shift_t = 0.02

                    now_time = start_time
                    while now_time + delta_t <= end_time:
                        window_start_time = now_time
                        window_end_time = now_time + delta_t
                        window_df = df[(df['timestamp'] >= window_start_time) &
                                       (df['timestamp'] <= window_end_time)].copy()

                        feature_row = Primitive(window_df).features
                        tmp_primitive_mat.append(feature_row)

                        now_time += delta_t

                tmp_primitive_df = pd.DataFrame(data=tmp_primitive_mat,
                                                columns=[handedness_label + '_' + col for col in STAT_COLS])
                primitive_df = pd.concat([primitive_df, tmp_primitive_df], axis=1)

            primitive_df['hit_type'] = label
            profile_primitive_df = pd.concat([profile_primitive_df, primitive_df], ignore_index=True)

        self._profile_primitive_df = profile_primitive_df

    @property
    def profile(self):
        return self._profile

    @property
    def profile_primitive_df(self):
        return self._profile_primitive_df
