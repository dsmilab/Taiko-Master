from ..config import *
from ..io import *
from .play import *
from .primitive import *
from ..tools.timestamp import *
from collections import deque

import pandas as pd
from sklearn import preprocessing

__all__ = ['get_performance']

DELTA_T_DIVIDED_COUNT = 8


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

    def __init__(self, who_id, song_id, order_id, scale, resample):
        self._event_primitive_df = None

        self._note_df = load_note_df(who_id, song_id, order_id)
        self._play = get_play(who_id, song_id, order_id, resample=resample)

        tempe_row = get_best_score_row(song_id)
        tempe_who = int(tempe_row['drummer_id'])
        tempe_order = int(tempe_row['performance_order'])
        self._tempe_play = get_play(tempe_who, song_id, tempe_order, resample=resample)

        self._events = self.__retrieve_event()
        self._time_unit = self._note_df['time_unit'].min()
        self._bar_unit = self._time_unit * 8
        self._delta_t = self._bar_unit / DELTA_T_DIVIDED_COUNT

        self.__build_event_primitive_df()

        if scale:
            self.__scale()

    def __retrieve_event(self):
        """
        Retrieve event which means note occurs of the song.

        :return: 2D array
        """

        events = []

        first_hit_time = self._play.first_hit_time
        # spot vertical mark lines
        for _, row in self._note_df.iterrows():
            hit_type = int(row['label'])
            events.append((first_hit_time + row['timestamp'], hit_type))

        return events

    def __build_event_primitive_df(self):
        """
        After setting play's dataframe, build dataframe of primitives around events in this play.

        :return: feature engineered dataframe of primitives
        """

        event_primitive_df = pd.DataFrame(columns=['hit_type'])
        event_primitive_df['hit_type'] = [self._events[i][1] for i in range(len(self._events))]

        for label, play_df in self._play.play_dict.items():
            window = deque()
            tempe_window = deque()

            play_mat = play_df.values
            tempe_play_mat = self._tempe_play.play_dict[label].values
            print(tempe_play_mat)

            play_id = 0
            tempe_play_id = 0

            tmp_primitive_mat = []
            # split all event times with gap "delta_t"
            for id_, _ in enumerate(self._events):
                event_time = self._events[id_][0]
                local_start_time = event_time - self._delta_t / 2
                local_end_time = event_time + self._delta_t / 2

                window, play_mat, play_id = slide(window, play_mat, play_id, local_start_time, local_end_time)
                tempe_window, tempe_play_mat, tempe_play_id = slide(tempe_window, tempe_play_mat, tempe_play_id,
                                                                    local_start_time, local_end_time)

                tmp_primitive_mat.append(get_features(window, tempe_window))

            tmp_primitive_df = pd.DataFrame(data=tmp_primitive_mat,
                                            columns=[label + '_' + col for col in STAT_COLS])
            event_primitive_df = pd.concat([event_primitive_df, tmp_primitive_df], axis=1)

        near_df = self.__get_near_event_hit_type()
        event_primitive_df = pd.concat([event_primitive_df, near_df], axis=1)
        self._event_primitive_df = event_primitive_df

    def __get_near_event_hit_type(self, n_counts=2):
        """
        Get event hit type before and after the current hit type.

        :param n_counts: range to get hit types
        :return: the dataframe contains all hit types
        """

        mat = []

        for id_ in range(len(self._events)):
            near = []

            for i in range(n_counts):
                hit_type = 0
                if id_ - 1 - i >= 0:
                    hit_type = self._events[id_ - 1 - i][1]
                near.append(hit_type)

            for i in range(n_counts):
                hit_type = 0
                if id_ + 1 + i < len(self._events):
                    hit_type = self._events[id_ + 1 + i][1]
                near.append(hit_type)

            mat.append(near)

        near_df = pd.DataFrame(data=mat,
                               columns=['L' + str(i + 1) for i in range(n_counts)] +
                                       ['R' + str(i + 1) for i in range(n_counts)])

        return near_df

    def __scale(self):
        """
        Scale values of required features.

        :return: nothing
        """

        scaler = preprocessing.StandardScaler()
        columns = []
        for label, _ in self._play.play_dict.items():
            columns += [label + '_' + col for col in SCALE_COLUMNS]
        subset = self._event_primitive_df[columns]
        train_x = [tuple(x) for x in subset.values]
        train_x = scaler.fit_transform(train_x)
        df = pd.DataFrame(train_x, columns=columns)
        self._event_primitive_df.update(df)

    @property
    def event_primitive_df(self):
        return self._event_primitive_df


def get_performance(who_id, song_id, order_id, scale=True, resample=RESAMPLE_RATE):
    """
    Get the performance.

    :param who_id: # of drummer
    :param song_id: # of song
    :param order_id: # of performance repetitively
    :param scale: if "True", scale values of required features
    :param resample: if not "None", resample by this frequency
    :return: the desired unique performance
    """

    return _Performance(who_id, song_id, order_id, scale, resample)
