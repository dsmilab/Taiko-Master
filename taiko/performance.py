from .tools.config import *
from .tools.score import get_gained_score_multiplier
from .primitive import get_features
from .database import load_record_df, transform_hit_type
from .play import get_play
from .model import LGBM
from scipy.stats import mode

from collections import deque
from itertools import product

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE

__all__ = ['get_performance', 'do_over_sampled', 'do_scaling', 'get_pf_similarity']


class _Performance(object):
    def __init__(self, play, window_size, scale):
        self._event_primitive_df = None

        self._play = play

        self._events = self._play.events

        self.__build__primitive_df(window_size)

        if scale:
            self._performance_primitive_df = do_scaling(self._performance_primitive_df)

    def __build__primitive_df(self, window_size):
        labels = [label for label, _ in self._play.play_dict.items()]
        windows = [deque() for _ in range(len(self._play.play_dict))]
        play_ids = [0] * len(self._play.play_dict)
        play_mats = [play_df.values for _, play_df in self._play.play_dict.items()]

        tmp_primitive_mat = []

        start_time = self._play.start_time
        end_time = self._play.end_time

        now_time = start_time + window_size

        while now_time <= end_time:
            local_start_time = now_time - window_size
            local_end_time = now_time

            feature_row = [local_end_time]
            for i_ in range(len(self._play.play_dict)):
                # slide window
                if len(windows[i_]) == 0 and play_ids[i_] < len(play_mats[i_]):
                    windows[i_].append(play_mats[i_][play_ids[i_]])
                    play_ids[i_] += 1

                while play_ids[i_] < len(play_mats[i_]) and play_mats[i_][play_ids[i_]][0] < local_end_time:
                    windows[i_].append(play_mats[i_][play_ids[i_]])
                    play_ids[i_] += 1

                while len(windows[i_]) > 0 and windows[i_][0][0] < local_start_time:
                    windows[i_].popleft()

                feature_row += get_features(windows[i_])

            if not np.isnan(feature_row).any():
                tmp_primitive_mat.append(feature_row)

            now_time += window_size * (1.0 - OVERLAPPING_RATE)

        columns = ['timestamp'] + [a + '_' + b for a, b in product(labels, STAT_COLS)]
        performance_primitive_df = pd.DataFrame(data=tmp_primitive_mat,
                                                columns=columns)

        self._performance_primitive_df = performance_primitive_df

    @property
    def performance_primitive_df(self):
        return self._performance_primitive_df

    @property
    def play(self):
        return self._play


def get_performance(pid, window_size=WINDOW_T, scale=False):
    performance_csv_path = posixpath.join(PERFORMANCE_DIR_PATH, str(pid) + '_pf@' + str(window_size) + '.csv')

    if os.path.isfile(performance_csv_path):
        performance_ep_df = pd.read_csv(performance_csv_path)
    else:
        play = get_play(pid)
        performance_ep_df = _Performance(play, window_size, scale).performance_primitive_df
        performance_csv_dir_path = os.path.dirname(performance_csv_path)

        os.makedirs(performance_csv_dir_path, exist_ok=True)
        performance_ep_df.to_csv(performance_csv_path, index=False, float_format='%.4f')

    return performance_ep_df


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


def do_over_sampled(df):
    x_columns, y_columns = [], []
    for col in df.columns:
        if re.match('hit_type', col):
            y_columns.append(col)
        else:
            x_columns.append(col)

    x = df.drop(y_columns, axis=1)
    y = df[y_columns]
    y = np.ravel(y)

    x_resampled, y_resampled = SMOTE(k_neighbors=3, random_state=0).fit_sample(x, y)

    x_df = pd.DataFrame(columns=x_columns, data=x_resampled)
    y_df = pd.DataFrame(columns=y_columns, data=y_resampled)

    new_df = pd.concat([x_df, y_df], axis=1)
    new_df = new_df[df.columns]

    return new_df


def get_pf_similarity(pf1, pf2, mode_='pf_split'):
    # assert(pf1.columns == pf2.columns)
    assert(len(pf1) == len(pf2))
    if mode_ == 'pf_merge':
        pass
        # return __get_pf_similarity_with_raw_merge(pf1, pf2)
    elif mode_ == 'pf_split':
        return __get_pf_similarity_with_raw_split(pf1, pf2)
    elif mode_ == 'pf_merge_euc':
        pass
        # return __get_pf_similarity_with_raw_merge_euc(pf1, pf2)
    elif mode_ == 'pf_split_euc':
        return __get_pf_similarity_with_raw_split_euc(pf1, pf2)

    return None


def __get_pf_similarity_with_raw_split_euc(pf1, pf2):
    distances = []
    for col in pf1.columns:
        if col == 'timestamp':
            continue
        x = pf1[col].values
        y = pf2[col].values
        distance = np.linalg.norm(x - y)
        distances.append(distance)

    return distances


def __get_pf_similarity_with_raw_split(pf1, pf2):

    def __get_dtw(df1, df2):
        x = df1.values
        y = df2.values
        distance, _ = fastdtw(x, y, dist=euclidean)
        return distance

    dtws = []
    for col in pf1.columns:
        if col == 'timestamp':
            continue
        dtw = __get_dtw(pf1[col], pf2[col])
        dtws.append(dtw)

    return dtws
