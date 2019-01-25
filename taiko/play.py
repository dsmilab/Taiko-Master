from .tools.config import *
from .tools.timestamp import *
from .database import load_record_df
from .io import load_arm_df, load_note_df, get_capture_dir_path
from .image import *

from typing import Tuple, List
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
        self._end_time = play_start_time + INTRO_DUMMY_TIME_LENGTH + FIRST_HIT_ALIGN_DICT[song_id]

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
        events = []
        note_df = load_note_df(song_id)
        # spot vertical mark lines
        for _, row in note_df.iterrows():
            hit_type = int(row['label'])
            events.append((np.float64(self._first_hit_time + row['timestamp']), hit_type))

        return events

    @property
    def play_dict(self) -> dict:
        return self._play_dict

    @property
    def start_time(self) -> float:
        return self._start_time

    @property
    def end_time(self) -> float:
        return self._end_time

    @property
    def first_hit_time(self) -> float:
        return self._first_hit_time

    @property
    def events(self) -> List[Tuple[float, int]]:
        return self._events


def get_play(pid, calibrate=True, resample=True):
    record_df = load_record_df()
    record_row = record_df.loc[pid]
    drummer_name = record_row['drummer_name']
    song_id = record_row['song_id']
    left_arm_filename = record_row['left_sensor_datetime']
    right_arm_filename = record_row['right_sensor_datetime']

    left_arm_df = load_arm_df(drummer_name, left_arm_filename)
    right_arm_df = load_arm_df(drummer_name, right_arm_filename)

    play_start_time = __get_start_time_from_cache(pid, record_row)

    raw_arm_df_dict = {
        left_arm_filename[0]: left_arm_df,
        right_arm_filename[0]: right_arm_df,
    }

    return _Play(song_id, raw_arm_df_dict, play_start_time, calibrate, resample)


def __get_start_time_from_cache(pid, record_row):
    drummer_name = record_row['drummer_name']

    start_time_cache_path = posixpath.join(PLAY_DIR_PATH, drummer_name + '-' + str(pid) + '.tk')

    if os.path.isfile(start_time_cache_path):
        with open(start_time_cache_path) as f:
            play_start_time = float(f.readline())
    else:
        capture_dir_name = record_row['capture_datetime']
        song_id = record_row['song_id']
        capture_dir_path = get_capture_dir_path(drummer_name, capture_dir_name)
        play_start_time = get_play_start_time(capture_dir_path, song_id)

        start_time_cache_dir_path = os.path.dirname(start_time_cache_path)
        os.makedirs(start_time_cache_dir_path, exist_ok=True)
        with open(start_time_cache_path, 'w') as fw:
            fw.write(str(play_start_time))

    return play_start_time


def get_similarity(pf1, pf2):

    # def __get_l2_norms(df1, df2):
    #     l2_norms = []
    #     bins = np.linspace(0, len(df1), _K + 1)
    #     for i_ in range(len(bins) - 1):
    #         zmin = int(bins[i_])
    #         zmax = int(bins[i_ + 1])
    #         for col in columns:
    #             x = df1[col].values[zmin: zmax]
    #             y = df2[col].values[zmin: zmax]
    #
    #             if x.shape[0] < y.shape[0]:
    #                 y = y[:x.shape[0]]
    #             elif x.shape[0] > y.shape[0]:
    #                 x = x[:y.shape[0]]
    #
    #             distance = np.linalg.norm(x - y)
    #             l2_norms.append(distance)
    #
    #     return l2_norms

    assert(len(pf1) == len(pf2))
    x = pf1.values
    y = pf2.values
    sims = []
    for row in x - y:
        sims.extend(row)
    return sims
