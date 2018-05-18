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

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import itertools


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
    DELTA_T_DIVIDED_COUNT = 4
    TIME_UNIT_DIVIDED_COUNT = 8

    def __init__(self, sensor, who_id, song_id, order_id, left_modes=None, right_modes=None):
        self._sensor = sensor
        self._who_id = who_id
        self._song_id = song_id
        self._order_id = order_id

        # params of a song
        self._time_unit = None
        self._bar_unit = None

        # importance dataframes as global sense
        self._song_df = None
        self._primitive_df = None
        self._events = []
        self._left_play_df, self._left_modes = None, left_modes
        self._right_play_df, self._right_modes = None, right_modes

        # duration of a song
        self._start_time, self._end_time, self._first_hit_time = None, None, None

        # related primitive params
        self._delta_t = 0
        self._unit_time_interval = 0

        self.__setup()

    def __setup(self):
        self._song_df = pd.read_csv(tkconfig.TABLE_PATH + 'taiko_song_' + str(self._song_id) + '_info.csv')
        self._song_df.drop(Performance.DROPPED_COLUMNS, axis=1, inplace=True)
        self._song_df.columns = Performance.RENAMED_COLUMNS

        self._start_time, self._end_time, self._first_hit_time = self.__get_play_duration()

        self._events = self.__retrieve_event()
        self._left_play_df, self._left_modes = self.__build_play_df(self._sensor.left_df, self._left_modes)
        self._right_play_df, self._right_modes = self.__build_play_df(self._sensor.right_df, self._right_modes)

        self._time_unit = self._song_df['time_unit'].min()
        self._bar_unit = self._time_unit * 8
        self._delta_t = self._bar_unit / Performance.DELTA_T_DIVIDED_COUNT
        self._unit_time_interval = self._delta_t / Performance.TIME_UNIT_DIVIDED_COUNT

        self.__build_primitive_df()

    def __get_play_duration(self):
        df = self._sensor.drummer_df
        df = df[(df['drummer_id'] == self._who_id) &
                (df['song_id'] == self._song_id) &
                (df['performance_order'] == self._order_id)]
        assert len(df) > 0, logging.error('No matched performances.')

        row = df.iloc[0]
        return row['hw_start_time'], row['hw_end_time'], row['first_hit_time']

    def __build_play_df(self, df, modes=None):
        play_df = df[(df['timestamp'] >= self._start_time) & (df['timestamp'] <= self._end_time)]
        if modes is None:
            modes = self.__get_modes_dict(play_df)
        play_df = self.__adjust_zero(play_df, modes)
        return play_df, modes

    def __build_primitive_df(self):
        primitive_df = pd.DataFrame(columns=['hand_side'] + tkconfig.STAT_COLS)
        now_time = self._start_time
        id_ = 0
        while now_time + self._unit_time_interval <= self._end_time:
            local_start_time = now_time
            local_end_time = now_time + self._unit_time_interval

            # left arm
            left_features = self.get_statistical_features(self._left_play_df, local_start_time, local_end_time)
            primitive_df.loc[id_] = [tkconfig.LEFT_HAND] + left_features
            id_ += 1

            # right arm
            right_features = self.get_statistical_features(self._right_play_df, local_start_time, local_end_time)
            primitive_df.loc[id_] = [tkconfig.RIGHT_HAND] + right_features
            id_ += 1

            now_time += self._unit_time_interval
        self._primitive_df = primitive_df.dropna()

    @staticmethod
    def __do_fft(data):
        freqx = np.fft.fft(data) / math.sqrt(len(data))
        energy = np.sum(np.abs(freqx) ** 2)
        return energy

    @staticmethod
    def get_statistical_features(play_df, start_time, end_time):
        play_df = play_df[(play_df['timestamp'] >= start_time) & (play_df['timestamp'] <= end_time)]
        if len(play_df) == 0:
            return [np.nan] * len(tkconfig.STAT_COLS)

        rms_df = play_df[['timestamp', 'imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']].copy()

        # acceleration movement intensity
        rms_df['a_rms'] = (
            play_df['imu_ax'] * play_df['imu_ax'] +
            play_df['imu_ay'] * play_df['imu_ay'] +
            play_df['imu_az'] * play_df['imu_az']).apply(lambda x: math.sqrt(x))

        # gyroscope movement intensity
        rms_df['g_rms'] = (
            play_df['imu_gx'] * play_df['imu_gx'] +
            play_df['imu_gy'] * play_df['imu_gy'] +
            play_df['imu_gz'] * play_df['imu_gz']).apply(lambda x: math.sqrt(x))

        # average intensity
        ai = rms_df['a_rms'].sum() / len(rms_df)

        # variance intensity
        vi = 0
        for i in range(len(rms_df)):
            row = rms_df.iloc[i]
            mit = float(row['a_rms'])
            vi += (mit - ai) ** 2
        vi /= len(rms_df)

        # normalized signal magnitude area
        sma = (rms_df['imu_ax'].apply(lambda x: abs(x)).sum() +
               rms_df['imu_ay'].apply(lambda x: abs(x)).sum() +
               rms_df['imu_az'].apply(lambda x: abs(x)).sum()) / len(rms_df)

        # averaged acceleration energy
        aae = Performance.__do_fft(rms_df['a_rms']) / len(rms_df)

        # averaged rotation energy
        are = Performance.__do_fft(rms_df['g_rms']) / len(rms_df)

        return [ai, vi, sma, aae, are]

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

    def __retrieve_event(self):
        events = []

        # spot vertical mark lines
        for i in range(len(self._song_df)):
            row = self._song_df.iloc[i]
            hit_type = int(row['label'])
            events.append((self._first_hit_time + row['timestamp'], hit_type))

        return events

    def plot_global_event(self):
        for col in tkconfig.ALL_COLUMNS:
            if col != 'timestamp' and col != 'wall_time':
                plt.figure(figsize=(25, 8))

                # retrieve left arm info
                plt.plot(self._left_play_df['timestamp'], self._left_play_df[col], label='left')

                # retrieve right arm info
                plt.plot(self._right_play_df['timestamp'], self._right_play_df[col], label='right')

                # draw vertical mark line
                for tm, hit_type in self._events:
                    if hit_type > 0:
                        plt.axvline(tm, color=tkconfig.COLORS[hit_type], lw=0.5)

                plt.legend()
                save_name = '%s who_id:%d song_id:%d order:%d' % (col, self._who_id, self._song_id, self._order_id)
                plt.title(save_name)

                plt.show()
                plt.close()

    @property
    def primitive_df(self):
        return self._primitive_df

    @property
    def events(self):
        return self._events

    @property
    def delta_t(self):
        return self._delta_t

    @property
    def left_play_df(self):
        return self._left_play_df

    @property
    def right_play_df(self):
        return self._right_play_df

    @property
    def left_modes(self):
        return self._left_modes

    @property
    def right_modes(self):
        return self._right_modes

    @property
    def unit_time_interval(self):
        return self._unit_time_interval


class Model(object):
    """
    consider training "only one" performance to predict other performance with the same song_id
    """
    _K = 50
    _C = 1000

    def __init__(self, k_centroid, tolerance=1000):
        self._K = k_centroid
        self._C = tolerance

        self._left_vocab_kmeans, self._left_max_abs_scaler = None, None
        self._right_vocab_kmeans, self._right_max_abs_scaler = None, None
        self._left_clf = None
        self._right_clf = None

    def fit(self, performance):
        left_vocab_kmeans, left_max_abs_scaler = self.__build_vocabulary_kmeans(performance, tkconfig.LEFT_HAND)
        self._left_vocab_kmeans, self._left_max_abs_scaler = left_vocab_kmeans, left_max_abs_scaler

        right_vocab_kmeans, right_max_abs_scaler = self.__build_vocabulary_kmeans(performance, tkconfig.RIGHT_HAND)
        self._right_vocab_kmeans, self._right_max_abs_scaler = right_vocab_kmeans, right_max_abs_scaler

        left_x, right_x, y = self.retrieve_primitive_space(performance)
        self._left_clf = SVC(C=self._C)
        self._left_clf.fit(left_x, y)

        self._right_clf = SVC(C=self._C)
        self._right_clf.fit(right_x, y)

    def retrieve_primitive_space(self, performance):
        pf = performance
        left_x = []
        right_x = []
        y = []

        sys.stdout.flush()
        for id_, tm in tqdm(enumerate(pf.events), total=len(pf.events)):
            event_time = pf.events[id_][0]
            hit_type = pf.events[id_][1]
            local_start_time = event_time - pf.delta_t
            local_end_time = event_time + pf.delta_t

            # left arm
            hist = self.get_local_primitive_hist(pf.left_play_df, local_start_time, local_end_time,
                                                 self._left_vocab_kmeans, self._left_max_abs_scaler,
                                                 pf.unit_time_interval)
            left_x.append(hist)

            # right arm
            hist = self.get_local_primitive_hist(pf.right_play_df, local_start_time, local_end_time,
                                                 self._right_vocab_kmeans, self._right_max_abs_scaler,
                                                 pf.unit_time_interval)
            right_x.append(hist)

            y.append(hit_type)

        return left_x, right_x, y

    @staticmethod
    def __build_vocabulary_kmeans(performance, hand_side):
        pf = performance
        max_abs_scaler = preprocessing.MaxAbsScaler()
        subset = pf.primitive_df[pf.primitive_df['hand_side'] == hand_side][tkconfig.STAT_COLS]
        train_x = [tuple(x) for x in subset.values]
        train_x = max_abs_scaler.fit_transform(train_x)
        vocab_kmeans = KMeans(n_clusters=Model._K, random_state=0).fit(train_x)
        return vocab_kmeans, max_abs_scaler

    def predict(self, performance, true_label=True):
        left_x, right_x, y = self.retrieve_primitive_space(performance)
        left_pred_y = self._left_clf.predict(left_x)
        right_pred_y = self._right_clf.predict(right_x)
        if true_label:
            return left_pred_y, right_pred_y, y
        else:
            return left_pred_y, right_pred_y

    @staticmethod
    def get_local_primitive_hist(play_df, start_time, end_time, vocab_kmeans, max_abs_scaler, unit_time_interval):
        local_play_df = play_df[(play_df['timestamp'] >= start_time) & (play_df['timestamp'] <= end_time)]

        # unit_time_interval was set
        local_stat_df = pd.DataFrame(columns=tkconfig.STAT_COLS)
        now_time = start_time
        id_ = 0

        # feature extraction
        while now_time + unit_time_interval <= end_time:
            unit_start_time = now_time
            unit_end_time = now_time + unit_time_interval

            local_stat_df.loc[id_] = Performance.get_statistical_features(local_play_df, unit_start_time, unit_end_time)
            id_ += 1

            now_time += unit_time_interval

        local_stat_df.dropna(inplace=True)

        # build freq histogram
        vec = np.zeros(Model._K)
        subset = local_stat_df[tkconfig.STAT_COLS]
        train_x = [tuple(x) for x in subset.values]
        train_x = max_abs_scaler.transform(train_x)
        for group_num in vocab_kmeans.predict(train_x):
            vec[group_num] += 1

        return vec
