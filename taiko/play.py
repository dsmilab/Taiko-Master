from .config import *
from .io import *
from .image import *
from .db import *

import pandas as pd
import numpy as np
from scipy.stats import mode

DELTA_T_DIVIDED_COUNT = 8
DUMMY_TIME_LENGTH = 15

__all__ = ['get_play',
           'get_play_score_auc',
           'estimate_remained_play_times']


class _Play(object):

    def __init__(self, song_id, raw_arm_df_dict, play_start_time, calibrate, resample):
        # { filename: file_csv, }

        self._play_dict = {}
        self._start_time, self._end_time = None, None
        self._first_hit_time = None

        self._note_df = load_note_df(song_id)
        # self._time_unit = mode(self._note_df['time_unit'])[0]
        self._resampling_rate = '0.01S'

        self.__set_hw_time(song_id, play_start_time)
        self._events = self.__retrieve_event(song_id)

        for filename, raw_arm_df in raw_arm_df_dict.items():
            position = filename[0]
            self._play_dict[position] = self.__build_play_df(raw_arm_df, calibrate, resample)

    def crop_near_raw_data(self, who_name, song_id, p_order, delta_t, position, prefix_name):
        for id_, _ in enumerate(self._events):
            event_time = self._events[id_][0]
            hit_type = self._events[id_][1]

            if hit_type < 1 or hit_type > 2:
                continue

            local_start_time = event_time - delta_t
            local_end_time = event_time + delta_t
            note_type = 'don' if hit_type == 1 else 'ka'
            filename = 'motif/%s/song%d/order%d/%s/%s/%s-%03d.csv' % \
                       (who_name, song_id, p_order, note_type, position, prefix_name, id_)

            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            df = self._play_dict[position]
            motif_df = df[(df['timestamp'] >= local_start_time) &
                          (df['timestamp'] <= local_end_time)].copy()

            motif_df.to_csv(filename, index=False)

    def __set_hw_time(self, song_id, play_start_time):
        play_time_length = SONG_LENGTH_DICT[song_id]
        intro_time_length = INTRO_LENGTH_DICT[song_id]

        self._start_time = play_start_time - DUMMY_TIME_LENGTH
        self._end_time = play_start_time + play_time_length + DUMMY_TIME_LENGTH
        self._first_hit_time = play_start_time + intro_time_length

    def __build_play_df(self, raw_arm_df, calibrate, resample):

        # crop desired play time interval
        play_df = raw_arm_df[(raw_arm_df['timestamp'] >= self._start_time) &
                             (raw_arm_df['timestamp'] <= self._end_time)].copy()

        # resample for more samples
        if resample:
            play_df.loc[:, 'timestamp'] = pd.to_datetime(play_df['timestamp'], unit='s')
            play_df.loc[:, 'timestamp'] = play_df['timestamp'].apply(
                lambda x: x.tz_localize('UTC').tz_convert('Asia/Taipei'))
            play_df = play_df.set_index('timestamp').resample(self._resampling_rate).mean()
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
            events.append((self._first_hit_time + row['timestamp'], hit_type))

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


def get_play(record_row, calibrate=True, resample=True):
    who_name = record_row['drummer_name']
    song_id = record_row['song_id']
    left_arm_filename = record_row['left_sensor_datetime']
    right_arm_filename = record_row['right_sensor_datetime']
    capture_dir_name = record_row['capture_datetime']

    left_arm_df = load_arm_df(who_name, left_arm_filename)
    right_arm_df = load_arm_df(who_name, right_arm_filename)

    capture_dir_path = get_capture_dir_path(who_name, capture_dir_name)
    play_start_time = get_play_start_time(capture_dir_path)

    raw_arm_df_dict = {
        left_arm_filename: left_arm_df,
        right_arm_filename: right_arm_df,
    }

    return _Play(song_id, raw_arm_df_dict, play_start_time, calibrate, resample)


def get_play_score_auc(capture_dir_path, song_id):
    area = 0.0
    timestamps, img_scores = read_score_board_info(capture_dir_path, song_id)
    for i_ in range(len(timestamps)):
        if i_ == 0:
            continue
        area += (timestamps[i_] - timestamps[i_ - 1]) * img_scores[i_ - 1]
    area /= len(timestamps)

    return area


def estimate_remained_play_times(capture_dir_path, song_id):
    auc = get_play_score_auc(capture_dir_path, song_id)
    stat = get_score_auc_stat(song_id)
    factor = (auc - stat['fc_mean_auc']) / stat['std_auc']
    times = 0
    if factor < 0:
        times = int(np.ceil(-factor * 2))
    return times
