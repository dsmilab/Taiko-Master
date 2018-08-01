from ..config import *
from ..io.note import *
from ..io.record import *
from .play import *
from .primitive import *
from collections import deque

import pandas as pd
import numpy as np

__all__ = ['get_performance']

DELTA_T_DIVIDED_COUNT = 8


class _Performance(object):

    def __init__(self, who_id, song_id, order_id):
        self._event_primitive_df = None

        self._note_df = load_note_df(who_id, song_id, order_id)
        self._play = get_play(who_id, song_id, order_id)

        self._events = self.__retrieve_event()
        self._time_unit = self._note_df['time_unit'].min()
        self._bar_unit = self._time_unit * 8
        self._delta_t = self._bar_unit / DELTA_T_DIVIDED_COUNT

        self.__build_event_primitive_df()

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
        event_primitive_df = pd.DataFrame(columns=['hit_type'])
        event_primitive_df['hit_type'] = [self._events[i][1] for i in range(len(self._events))]

        for label, play_df in self._play.play_dict.items():
            window = deque()

            play_mat = play_df.values
            play_id = 0

            tmp_primitive_mat = []
            # split all event times with gap "delta_t"
            for id_, tm in enumerate(self._events):
                event_time = self._events[id_][0]
                local_start_time = event_time - self._delta_t / 2
                local_end_time = event_time + self._delta_t / 2

                if len(window) == 0:
                    window.append(play_mat[play_id])
                    play_id += 1

                while play_id < len(play_mat) and play_mat[play_id][0] < local_end_time:
                    window.append(play_mat[play_id])
                    play_id += 1

                while window[0][0] < local_start_time:
                    window.popleft()

                tmp_primitive_mat.append(get_features(window))

            tmp_primitive_df = pd.DataFrame(data=tmp_primitive_mat,
                                            columns=[label + '_' + col for col in STAT_COLS])
            event_primitive_df = pd.concat([event_primitive_df, tmp_primitive_df], axis=1)

        near_df = self.__get_near_event_hit_type()
        event_primitive_df = pd.concat([event_primitive_df, near_df], axis=1)

        self._event_primitive_df = event_primitive_df

    def __get_near_event_hit_type(self, n_counts=2):
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

    @property
    def event_primitive_df(self):
        return self._event_primitive_df


def get_performance(who_id, song_id, order_id):
    return _Performance(who_id, song_id, order_id)
