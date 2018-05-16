import pandas as pd
import numpy as np
import tkconfig
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import time
import sys
import math
import gc

from scipy.stats import mode
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime, timedelta
from collections import deque
from tqdm import tqdm


class Sensor(object):
    TAILED_ADDITIONAL_TIME = 15

    def __init__(self, verbose=0):
        self._verbose = verbose
        self._left_df = None
        self._right_df = None
        self._drummer_df = None

        self.__setup()
        gc.collect()

    def __setup(self):
        self._left_df = self.__load_arm_csv(tkconfig.LEFT_PATH)
        self._right_df = self.__load_arm_csv(tkconfig.RIGHT_PATH)
        if self._verbose > 0:
            logging.info("load arm CSV.")

        self._drummer_df = self.__load_drummer_csv()
        if self._verbose > 0:
            logging.info("load drummer CSV.")

    @staticmethod
    def __load_arm_csv(arm_csv_path):
        files = next(os.walk(arm_csv_path))[2]
        script_df = [
            pd.read_csv(arm_csv_path + filename, dtype={
                'timestamp': np.float64
            }) for filename in files
        ]
        merged_df = pd.DataFrame(pd.concat(script_df, ignore_index=True))
        merged_df.drop('key', axis=1, inplace=True)
        return merged_df

    @staticmethod
    def __load_drummer_csv():
        df = pd.read_csv(tkconfig.TABLE_PATH + 'taiko_play.csv')
        tmp_df = pd.read_csv(tkconfig.TABLE_PATH + 'taiko_song.csv', dtype={
            'song_length': np.int16
        })
        df = df.merge(tmp_df, how='left', left_on='song_id', right_on='song_id')
        tmp_df = pd.read_csv(tkconfig.TABLE_PATH + 'taiko_drummer.csv')
        df = df.merge(tmp_df, how='left', left_on='drummer_id', right_on='id')
        df['hw_start_time'] = df['start_time'].apply(Sensor.get_hwclock_time)
        df['hw_end_time'] = df['hw_start_time'] + df['song_length'] + Sensor.TAILED_ADDITIONAL_TIME
        return df

    @staticmethod
    def get_hwclock_time(local_time, delta=0):
        d = datetime.strptime(local_time, "%m/%d/%Y %H:%M:%S")
        d = d + timedelta(seconds=int(delta))
        return time.mktime(d.timetuple())

    @property
    def left_df(self):
        return self._left_df

    @property
    def right_df(self):
        return self._right_df

    @property
    def drummer_df(self):
        return self._drummer_df


class Performance(object):
    DROPPED_COLUMNS = ['#', 'separator']
    RENAMED_COLUMNS = ['bar', 'bpm', 'time_unit', 'timestamp', 'label', 'continuous']

    def __init__(self, sensor, who_id, song_id, order_id):
        self._sensor = sensor
        self._who_id = who_id
        self._song_id = song_id
        self._order_id = order_id

        # params of a song
        self._time_unit = None
        self._bar_unit = None

        # importance dataframes as global sense
        self._song_df = None
        self._left_play_df, self._left_modes = None, None
        self._right_play_df, self._right_modes = None, None

        # duration of a song
        self._start_time, self._end_time = None, None

        self.__setup()

    def __setup(self):
        self._song_df = pd.read_csv(tkconfig.TABLE_PATH + 'taiko_song_' + str(self._song_id) + '_info.csv')
        self._song_df.drop(Performance.DROPPED_COLUMNS, axis=1, inplace=True)
        self._song_df.columns = Performance.RENAMED_COLUMNS

        self._start_time, self._end_time = self.__get_play_duration()

        self._left_play_df, self._left_modes = self.__build_play_df(self._sensor.left_df)
        self._right_play_df, self._right_modes = self.__build_play_df(self._sensor.right_df)

    def __get_play_duration(self):
        df = self._sensor.drummer_df
        df = df[(df['drummer_id'] == self._who_id) &
                (df['song_id'] == self._song_id) &
                (df['performance_order'] == self._order_id)]
        assert len(df) > 0, logging.error('No matched performances.')

        row = df.iloc[0]
        return row['hw_start_time'], row['hw_end_time']

    def __build_play_df(self, df):
        play_df = df[(df['timestamp'] >= self._start_time) & (df['timestamp'] <= self._end_time)]
        modes = self.__get_modes_dict(play_df)
        play_df = self.__adjust_zero(play_df, modes)
        return play_df, modes

    @staticmethod
    def __adjust_zero(df, modes_dict=None):
        copy_df = df.copy()
        for col in tkconfig.ZERO_ADJ_COL:
            mode_ = mode(copy_df[col])[0] if modes_dict is None else modes_dict[col]
            copy_df[col] = copy_df[col] - mode_
        return copy_df

    @staticmethod
    def __get_modes_dict(df):
        modes = {}
        copy_df = df.copy()
        for col in tkconfig.ZERO_ADJ_COL:
            mode_ = mode(copy_df[col])[0]
            modes[col] = mode_
        return modes

    def plot_global_event(self):
        # !!
        first_hit_time = self._start_time + 14
        for col in tkconfig.ALL_COLUMNS:
            if col != 'timestamp' and col != 'wall_time':
                plt.figure(figsize=(25, 8))

                # retrieve left arm info
                plt.plot(self._left_play_df['timestamp'], self._left_play_df[col], label='left')

                # retrieve right arm info
                plt.plot(self._right_play_df['timestamp'], self._right_play_df[col], label='right')

                # draw vertical mark line
                for i in range(len(self._song_df)):
                    row = self._song_df.iloc[i]
                    hit_type = int(row['label'])
                    if hit_type > 0:
                        plt.axvline(first_hit_time + row['timestamp'], color=tkconfig.COLORS[hit_type], lw=0.5)

                plt.legend()
                save_name = '%s who_id:%d song_id:%d order:%d' % (col, self._who_id, self._song_id, self._order_id)
                plt.title(save_name)

                plt.show()
                plt.close()


class Model(object):
    def __init__(self):
        pass
