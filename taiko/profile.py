from .tools.config import *
from .tools.timestamp import *
from .tools.database import *
from .primitive import *
from .io import *

from collections import deque
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from glob import glob
from scipy.stats import mode
from sklearn import preprocessing

__all__ = ['create_profile',
           'plot_profile',
           'get_profile']


class _Profile(object):
    _SAMPLING_RATE = '0.01S'
    _WINDOW_T = 0.2

    _LABELS = {
        'right_don': 1,
        'left_don': 2,
        'right_ka': 3,
        'left_ka': 4,
        'big_don': 5,
        'big_ka': 6,
        'pause': 0,
        'drumroll': 7,
    }

    _HANDEDNESS = {
        'left': 'L',
        'right': 'R',
    }

    def __init__(self, drummer_name, window_size):
        self._drummer_name = drummer_name
        self._window_size = window_size
        self.__build()
        self.__build_primitive_df(window_size)

    def __build(self):
        self._profile = {}

        for label_kwd in _Profile._LABELS.keys():
            self._profile[label_kwd] = {}
            for handedness_kwd, handedness_label in _Profile._HANDEDNESS.items():
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
            play_df = resample_sensor_df(play_df)

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

    def __build_primitive_df(self, window_size):
        profile_primitive_df = pd.DataFrame()
        for label_kwd, label in _Profile._LABELS.items():
            primitive_df = pd.DataFrame()
            for _, handedness_label in _Profile._HANDEDNESS.items():
                window = deque()
                tmp_primitive_mat = []
                for df in self._profile[label_kwd][handedness_label]:
                    play_mat = df.values
                    play_id = 0
                    start_time = df['timestamp'].iloc[0] + _Profile._WINDOW_T
                    end_time = df['timestamp'].iloc[-1]

                    now_time = start_time
                    while now_time <= end_time:
                        window_start_time = now_time - _Profile._WINDOW_T
                        window_end_time = now_time

                        if len(window) == 0:
                            window.append(play_mat[play_id])
                            play_id += 1

                        while play_id < len(play_mat) and play_mat[play_id][0] < window_end_time:
                            window.append(play_mat[play_id])
                            play_id += 1

                        while window[0][0] < window_start_time:
                            window.popleft()

                        feature_row = get_features(window)
                        tmp_primitive_mat.append(feature_row)

                        now_time += window_size / 2.0

                tmp_primitive_df = pd.DataFrame(data=tmp_primitive_mat,
                                                columns=[handedness_label + '_' + col for col in STAT_COLS])
                primitive_df = pd.concat([primitive_df, tmp_primitive_df], axis=1)

            primitive_df['hit_type'] = label
            profile_primitive_df = pd.concat([profile_primitive_df, primitive_df], ignore_index=True)

        print(profile_primitive_df)
        self._profile_primitive_df = profile_primitive_df

    @property
    def profile(self):
        return self._profile

    @property
    def profile_primitive_df(self):
        return self._profile_primitive_df


def get_profile(drummer_name, window_size=0.2, forcibly=False):
    profile_csv_path = posixpath.join(PROFILE_EP_DIR_PATH, drummer_name + '_' + str(window_size) + '.csv')
    if not forcibly and os.path.isfile(profile_csv_path):
        profile_ep_df = pd.read_csv(profile_csv_path)
        return profile_ep_df
    else:
        profile_ep_df = _Profile(drummer_name, window_size).profile_primitive_df
        profile_csv_dir_path = os.path.dirname(profile_csv_path)
        os.makedirs(profile_csv_dir_path, exist_ok=True)
        profile_ep_df.to_csv(profile_csv_path, index=False, float_format='%.4f')

    return profile_ep_df


def create_profile(drummer_name):
    record_df = load_record_df(drummer_name=drummer_name,
                               song_id=99)

    def __create_key_act_profile(label_kwd, sensor_name):
        df = load_arm_df(drummer_name, sensor_name)
        df = resample_sensor_df(df)
        df = calibrate_sensor_df(df)
        df['handedness'] = label_kwd

        for e_id in range(0, len(REF_AVLINE[id_]), 2):
            start_timestamp = REF_AVLINE[id_][e_id]
            end_timestamp = REF_AVLINE[id_][e_id + 1]

            crop_df = df[(df.timestamp >= start_timestamp) & (df.timestamp <= end_timestamp)]

            filename = PREFIX[e_id // 2] + '_' + str(id_) + '.csv'
            filepath = posixpath.join(PROFILE_DIR_PATH, drummer_name, label_kwd, filename)
            dirname = os.path.dirname(filepath)
            os.makedirs(dirname, exist_ok=True)
            crop_df.to_csv(filepath, index=False, float_format='%.4f')

        return df

    for id_, row in record_df.iterrows():
        left_sensor_name = row['left_sensor_datetime']
        right_sensor_name = row['right_sensor_datetime']

        left_df = __create_key_act_profile('left', left_sensor_name)
        right_df = __create_key_act_profile('right', right_sensor_name)

        profile_df = pd.concat([left_df, right_df], ignore_index=True)
        profile_df.dropna(inplace=True)
        profile_filename = 'profile_' + str(id_) + '.csv'
        profile_filepath = posixpath.join(PROFILE_DIR_PATH, drummer_name, profile_filename)
        profile_df.to_csv(profile_filepath, index=False, float_format='%.4f')


def plot_profile(drummer_name):
    profiles = glob(posixpath.join(PROFILE_DIR_PATH, drummer_name, 'profile_*.csv'))
    for profile_ in profiles:
        profile_df = pd.read_csv(profile_)
        res = re.search('profile_\d+.csv', profile_)
        id_ = int(res.group(0)[8:-4])
        print(id_)
        for col in ZERO_ADJ_COL:
            plt.figure(figsize=(15, 8))
            sns.lineplot(x='timestamp', y=col, data=profile_df, hue='handedness')
            for e_id, x_ in enumerate(REF_AVLINE[id_]):
                color_ = 'black'
                if e_id // 2 % 2 == 0:
                    color_ = 'red'
                plt.axvline(x=x_, color=color_)
            plt.show()


def do_scaling(df):
    """
    Scale values of required features.

    :return: nothing
    """

    scaler = preprocessing.StandardScaler()
    columns = df.columns
    columns = [col for col in columns if not re.match(NO_SCALE_REGEX, col)]

    subset = df[columns]
    train_x = [tuple(x) for x in subset.values]
    train_x = scaler.fit_transform(train_x)
    new_df = pd.DataFrame(data=train_x, columns=columns)
    df.update(new_df)

    return df

