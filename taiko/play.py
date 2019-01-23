from .tools.config import *
from .tools.timestamp import *
from .io import *
from .image import *

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pandas as pd
import numpy as np
import posixpath
from scipy.stats import mode

__all__ = ['get_play',
           'get_similarity']


class _Play(object):

    def __init__(self, song_id, raw_arm_df_dict, play_start_time, calibrate, resample):
        # { filename: file_csv, }

        self._play_dict = {}
        self._start_time, self._end_time = None, None
        self._first_hit_time = None

        self._note_df = load_note_df(song_id)

        self.__set_hw_time(song_id, play_start_time)
        self._events = self.__retrieve_event(song_id)

        for filename, raw_arm_df in raw_arm_df_dict.items():
            position = filename[0]
            self._play_dict[position] = self.__build_play_df(raw_arm_df, calibrate, resample)

    def __set_hw_time(self, song_id, play_start_time):
        self._first_hit_time = play_start_time + INTRO_DUMMY_TIME_LENGTH
        self._start_time = play_start_time
        self._end_time = play_start_time + FIRST_HIT_ALIGN_DICT[song_id]

    def __build_play_df(self, raw_arm_df, calibrate, resample):

        # crop desired play time interval
        play_df = raw_arm_df[(raw_arm_df['timestamp'] >= self._start_time) &
                             (raw_arm_df['timestamp'] <= self._end_time)].copy()

        if calibrate:
            play_df = calibrate_sensor_df(play_df)

        if resample:
            play_df = resample_sensor_df(play_df)

        return play_df

    def __retrieve_event(self, song_id):
        """
        Retrieve event which means note occurs of the song.

        :return: 2D array
        """

        events = []
        note_df = load_note_df(song_id)
        # spot vertical mark lines
        for _, row in note_df.iterrows():
            hit_type = int(row['label'])
            events.append((np.float64(self._first_hit_time + row['timestamp']), hit_type))

        return events

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

    @property
    def events(self):
        return self._events


def get_play(record_row, calibrate=True, resample=True, from_tmp_dir=False):
    who_name = record_row['drummer_name']
    song_id = record_row['song_id']
    left_arm_filename = record_row['left_sensor_datetime']
    right_arm_filename = record_row['right_sensor_datetime']
    capture_dir_name = record_row['capture_datetime']

    left_arm_df = load_arm_df(who_name, left_arm_filename, from_tmp_dir)
    right_arm_df = load_arm_df(who_name, right_arm_filename, from_tmp_dir)

    capture_dir_path = get_capture_dir_path(who_name, capture_dir_name, from_tmp_dir)
    play_start_time = get_play_start_time(capture_dir_path, song_id)

    raw_arm_df_dict = {
        left_arm_filename[0]: left_arm_df,
        right_arm_filename[0]: right_arm_df,
    }

    return _Play(song_id, raw_arm_df_dict, play_start_time, calibrate, resample)


def get_similarity(play1, play2):
    def __retrieve_columns(left_df, right_df, columns):
        left_df = left_df[columns]
        right_df = right_df[columns]

        comb_df = pd.concat([left_df, right_df], axis=1, ignore_index=True)
        comb_df.dropna(inplace=True)
        return comb_df

    def __get_dtw(df1, df2):
        x = df1.values
        y = df2.values
        distance, _ = fastdtw(x, y, dist=euclidean)
        return distance

    acc_comb1_df = __retrieve_columns(play1.play_dict['L'], play1.play_dict['R'], ZERO_ADJ_COL[:3])
    acc_comb2_df = __retrieve_columns(play2.play_dict['L'], play2.play_dict['R'], ZERO_ADJ_COL[:3])
    acc_dist = __get_dtw(acc_comb1_df, acc_comb2_df)

    gyr_comb1_df = __retrieve_columns(play1.play_dict['L'], play1.play_dict['R'], ZERO_ADJ_COL[3:6])
    gyr_comb2_df = __retrieve_columns(play2.play_dict['L'], play2.play_dict['R'], ZERO_ADJ_COL[3:6])
    gyr_dist = __get_dtw(gyr_comb1_df, gyr_comb2_df)

    return acc_dist, gyr_dist
