from taiko.tools.config import *
from .primitive import *
from collections import deque

import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE

__all__ = ['get_performance', 'do_over_sampled', 'do_scaling']


class _Performance(object):
    """
    Handle the specific play and engineer features around hit events.

    :protected attributes:
        event_primitive_df: dataframe containing of primitives around events in this play.

        note_df: drum note dataframe of the particular song
        play: dataframe about particular arms of the record

        events: the 2D array which element (time, label) represents a note type "label" occurs at "time"
        time_unit: the minimum time interval between two notes depending on BPM of a song
        bar_unit: default is "time_unit x 8"
        delta_t: time interval we consider a local event
    """
    _WINDOW_T = 0.2

    def __init__(self, play, scale):
        self._event_primitive_df = None

        self._play = play

        self._events = self._play.events

        self.__build__primitive_df()

        if scale:
            self._performance_primitive_df = do_scaling(self._performance_primitive_df)

    def __build__primitive_df(self):
        """
        After setting play's dataframe, build dataframe of primitives around events in this play.

        :return: feature engineered dataframe of primitives
        """

        labels = [label for label, _ in self._play.play_dict.items()]
        windows = [deque() for _ in range(len(self._play.play_dict))]
        play_ids = [0] * len(self._play.play_dict)
        play_mats = [play_df.values for _, play_df in self._play.play_dict.items()]

        tmp_primitive_mat = []

        start_time = self._play.start_time
        end_time = self._play.end_time

        now_time = start_time + self._WINDOW_T

        while now_time <= end_time:
            local_start_time = now_time - self._WINDOW_T
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

            now_time += self._WINDOW_T

        columns = ['timestamp'] + [label + '_' + col for col in STAT_COLS for label in labels]
        performance_primitive_df = pd.DataFrame(data=tmp_primitive_mat,
                                                columns=columns)

        self._performance_primitive_df = performance_primitive_df

    @property
    def performance_primitive_df(self):
        return self._performance_primitive_df


def get_performance(play, scale=False):
    """
    Get the performance.

    :param play:
    :param scale: if "True", scale values of required features
    :return: the desired unique performance
    """

    return _Performance(play, scale).performance_primitive_df


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
