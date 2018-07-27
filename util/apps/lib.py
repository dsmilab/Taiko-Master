import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import apps.tkconfig as tkconfig
import logging
import os
import time
import sys
import math
import gc

from tqdm import tqdm
from scipy.stats import mode
from datetime import datetime, timedelta

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.svm import SVC


class Sensor(object):
    """
    Handle output from wearable devices.

    :protected attributes:
        verbose: default 0, otherwise will output loading message
        left_df: dataframe about left arm
        right_df: dataframe about right arm
        drummer_df: dataframe about performance of drummers
    """

    TAILED_ADDITIONAL_TIME = 30

    def __init__(self, verbose=0):
        self._verbose = verbose
        self._left_df = None
        self._right_df = None
        self._drummer_df = None

        self.__setup()
        gc.collect()

    def __setup(self):
        """
        Assist initial process.

        :return:
        """

        self._left_df = self.__load_arm_csv(tkconfig.LEFT_PATH)
        self._right_df = self.__load_arm_csv(tkconfig.RIGHT_PATH)
        if self._verbose > 0:
            logging.info("load arm CSV.")

        self._drummer_df = self.__load_drummer_csv()
        if self._verbose > 0:
            logging.info("load drummer CSV.")

    @staticmethod
    def __load_arm_csv(arm_csv_path):
        """
        Load CSV files which contain information about arms.

        :param arm_csv_path: the directory path of arm CSV stored
        :return: merged dataframe inside the directory
        """

        # trace the directory and read all of them
        files = next(os.walk(arm_csv_path))[2]
        script_df = [
            pd.read_csv(arm_csv_path + filename, dtype={
                'timestamp': np.float64
            }) for filename in files
        ]

        # merge them into a dataframe
        merged_df = pd.DataFrame(pd.concat(script_df, ignore_index=True))
        merged_df.drop('key', axis=1, inplace=True)

        return merged_df

    @staticmethod
    def __load_drummer_csv():
        """
        Load CSV files which contain information about drummers' performance.

        :return: performance dataframe being in additional of timestamp pre-processing.
        """

        # read drummers' plays
        df = pd.read_csv(tkconfig.TABLE_PATH + 'taiko_play.csv')

        # read song's information and merge it
        tmp_df = pd.read_csv(tkconfig.TABLE_PATH + 'taiko_song.csv', dtype={
            'song_length': np.int16
        })
        df = df.merge(tmp_df, how='left', left_on='song_id', right_on='song_id')

        # read drummers' personal information and merge it
        tmp_df = pd.read_csv(tkconfig.TABLE_PATH + 'taiko_drummer.csv')
        df = df.merge(tmp_df, how='left', left_on='drummer_id', right_on='id')

        # translate UTC timestamp into hardware timestamp
        df['hw_start_time'] = df['start_time'].apply(Sensor.get_hwclock_time)
        df['hw_end_time'] = df['hw_start_time'] + df['song_length'] + Sensor.TAILED_ADDITIONAL_TIME

        return df

    @staticmethod
    def get_hwclock_time(local_time, delta=0):
        """
        Given UTC time, translate it into hardware time.

        :param local_time: UTC time in specific string format
        :param delta: pass UTC time in # seconds
        :return: required hardware time.
        """

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
    """
    Handle the specific play.

    :protected attributes:
        sensor: the output of wearable device
        who_id: # of drummer
        song_id: # of song
        order_id: # of performance repetitively

        time_unit: the minimum time interval between two notes depending on BPM of a song
        bar_unit: default is "time_unit x 8"

        song_df: dataframe containing music of the song
        primitive_df: dataframe containing primitives of this play
        events: the 2D array which element (time, label) represents a note type "label" occurs at "time"

        left_play_df: dataframe about play of left arm
        left_modes: a list containing all attributes' mode of "left_play_df"
        right_play_df: dataframe about play of right arm
        right_modes: a list containing all attributes' mode of "right_play_df"

        start_time: timestamp where the song starts
        end_time: timestamp where the song ends
        first_hit_time: timestamp where the first note occurs
        delta_t: time interval we consider a local event
        unit_time_interval: time interval we consider a primitive
    """

    DROPPED_COLUMNS = ['#', 'separator', 'continuous']
    RENAMED_COLUMNS = ['bar', 'bpm', 'time_unit', 'timestamp', 'label']
    DELTA_T_DIVIDED_COUNT = 1
    TIME_UNIT_DIVIDED_COUNT = 2

    def __init__(self, sensor, who_id, song_id, order_id,
                 dt_dc_count=DELTA_T_DIVIDED_COUNT,
                 tu_dc_count=TIME_UNIT_DIVIDED_COUNT,
                 left_modes=None, right_modes=None):

        self._sensor = sensor
        self._who_id = who_id
        self._song_id = song_id
        self._order_id = order_id

        self.DELTA_T_DIVIDED_COUNT = dt_dc_count
        self.TIME_UNIT_DIVIDED_COUNT = tu_dc_count

        # params of a song
        self._time_unit = None
        self._bar_unit = None

        # importance dataframes as global sense
        self._song_df = None
        self._event_primitive_df = None
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
        """
        Assist initial process.

        :return:
        """

        difficulty = self.__get_play_difficulty()
        song_file_name = 'taiko_song_%d_%s_info.csv' % (self._song_id, difficulty)

        self._song_df = pd.read_csv(tkconfig.TABLE_PATH + song_file_name)
        self._song_df.drop(Performance.DROPPED_COLUMNS, axis=1, inplace=True)
        self._song_df.columns = Performance.RENAMED_COLUMNS
        self._song_df['label'] = self._song_df['label'].apply(self._transform_hit_type_label)

        self._start_time, self._end_time, self._first_hit_time = self.__get_play_duration()

        self._events = self.__retrieve_event()
        self._left_play_df, self._left_modes = self.__build_play_df(self._sensor.left_df, self._left_modes)
        self._right_play_df, self._right_modes = self.__build_play_df(self._sensor.right_df, self._right_modes)

        self._time_unit = self._song_df['time_unit'].min()
        self._bar_unit = self._time_unit * 8
        self._delta_t = self._bar_unit / self.DELTA_T_DIVIDED_COUNT
        self._unit_time_interval = self._delta_t / self.TIME_UNIT_DIVIDED_COUNT

        self.__build_event_primitive_df()

    @staticmethod
    def _transform_hit_type_label(label):
        if label in [1, 2, 3, 4]:
            return 1
        elif label in [5, 6]:
            return 2
        return 0

    def __get_play_difficulty(self):
        df = self._sensor.drummer_df
        df = df[(df['drummer_id'] == self._who_id) &
                (df['song_id'] == self._song_id) &
                (df['performance_order'] == self._order_id)]
        assert len(df) > 0, logging.error('No matched performances.')

        # assume matched case is unique
        row = df.iloc[0]

        return row['difficulty']

    def __get_play_duration(self):
        """
        Get duration of the song we interested.

        :return: "start_time", "end_time", "first_hit_time".
        """

        df = self._sensor.drummer_df
        df = df[(df['drummer_id'] == self._who_id) &
                (df['song_id'] == self._song_id) &
                (df['performance_order'] == self._order_id)]
        assert len(df) > 0, logging.error('No matched performances.')

        # assume matched case is unique
        row = df.iloc[0]

        return row['hw_start_time'], row['hw_end_time'], row['first_hit_time']

    def __build_play_df(self, df, modes=None):
        """
        After setting duration of the song, build dataframe of a play.

        :param df: original dataframe
        :param modes: default is "None", adjust zero by this own case, otherwise will by this param
        :return: cropped and zero-adjusted dataframe, attributes' modes
        """

        play_df = df[(df['timestamp'] >= self._start_time) & (df['timestamp'] <= self._end_time)]

        # "modes" param isn't set, then get this own modes
        if modes is None:
            modes = self.__get_modes_dict(play_df)

        play_df = self.__adjust_zero(play_df, modes)
        return play_df, modes

    def __build_event_primitive_df(self):
        event_primitive_df = pd.DataFrame(columns=['hit_type'] + tkconfig.L_STAT_COLS
                                                  + tkconfig.R_STAT_COLS
                                                  + tkconfig.COND_COLS)

        # split all event times with gap "unit_time_interval"
        for id_, tm in enumerate(self._events):
            event_time = self._events[id_][0]
            hit_type = self._events[id_][1]
            local_start_time = event_time - self._unit_time_interval / 2
            local_end_time = event_time + self._unit_time_interval / 2

            # left arm
            left_features = self.get_statistical_features(self._left_play_df, local_start_time, local_end_time)

            # right arm
            right_features = self.get_statistical_features(self._right_play_df, local_start_time, local_end_time)

            near = self.__get_near_event_hit_type(id_)

            event_primitive_df.loc[id_] = [hit_type] + left_features + right_features + near

        event_primitive_df['hit_type'] = event_primitive_df['hit_type'].astype(np.int8)
        event_primitive_df.dropna(inplace=True)

        self._event_primitive_df = event_primitive_df

    def __get_near_event_hit_type(self, now, n_counts=2):
        near = []
        for i in range(n_counts):
            hit_type = 0
            if now - 1 - i >= 0:
                hit_type = self._events[now - 1 - i][1]
            near.append(hit_type)

        for i in range(n_counts):
            hit_type = 0
            if now + 1 + i < len(self._events):
                hit_type = self._events[now + 1 + i][1]
            near.append(hit_type)

        return near

    def scale(self):
        max_abs_scaler = preprocessing.StandardScaler()
        subset = self._event_primitive_df[tkconfig.L_STAT_COLS + tkconfig.R_STAT_COLS]
        train_x = [tuple(x) for x in subset.values]
        train_x = max_abs_scaler.fit_transform(train_x)
        df = pd.DataFrame(train_x)
        df.columns = tkconfig.L_STAT_COLS + tkconfig.R_STAT_COLS
        self._event_primitive_df = self._event_primitive_df.drop(tkconfig.L_STAT_COLS +
                                                                 tkconfig.R_STAT_COLS, axis=1).join(df)

    @staticmethod
    def __do_fft(data):
        """
        Implement fast fourier transformation.

        :param data: a numeric series in time domain
        :return: the energy of this data in frequency domain
        """

        freqx = np.fft.fft(data) / math.sqrt(len(data))
        energy = np.sum(np.abs(freqx) ** 2)
        return energy

    @staticmethod
    def get_statistical_features(play_df, start_time, end_time):
        """
        Retrieve feature space in given time interval.

        :param play_df: dataframe of a play
        :param start_time: cropped feature space where to start
        :param end_time: cropped feature space where to end
        :return: feature space with 1D array
        """

        play_df = play_df[(play_df['timestamp'] >= start_time) & (play_df['timestamp'] <= end_time)]
        if len(play_df) <= 1:
            return [np.nan] * len(tkconfig.STAT_COLS)

        rms_df = play_df[['timestamp', 'imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']].copy()

        # acceleration movement intensity (MI)
        rms_df['a_rms'] = (
            play_df['imu_ax'] * play_df['imu_ax'] +
            play_df['imu_ay'] * play_df['imu_ay'] +
            play_df['imu_az'] * play_df['imu_az']).apply(lambda x: math.sqrt(x))

        # gyroscope movement intensity (MI)
        rms_df['g_rms'] = (
            play_df['imu_gx'] * play_df['imu_gx'] +
            play_df['imu_gy'] * play_df['imu_gy'] +
            play_df['imu_gz'] * play_df['imu_gz']).apply(lambda x: math.sqrt(x))

        # average intensity (xAI)
        aai = rms_df['a_rms'].sum() / len(rms_df)
        gai = rms_df['g_rms'].sum() / len(rms_df)

        # median of AMI
        mami = rms_df['a_rms'].median()
        mgmi = rms_df['g_rms'].median()

        # variance intensity (xVI)
        avi = 0
        for i in range(len(rms_df)):
            row = rms_df.iloc[i]
            mit = float(row['a_rms'])
            avi += (mit - aai) ** 2
        avi /= len(rms_df)

        gvi = 0
        for i in range(len(rms_df)):
            row = rms_df.iloc[i]
            mit = float(row['g_rms'])
            gvi += (mit - gai) ** 2
        gvi /= len(rms_df)

        # standard deviation intensity
        asdi = math.sqrt(avi)
        gsdi = math.sqrt(gvi)

        # normalized signal magnitude area (SMA)
        asma = (rms_df['imu_ax'].apply(lambda x: abs(x)).sum() +
                rms_df['imu_ay'].apply(lambda x: abs(x)).sum() +
                rms_df['imu_az'].apply(lambda x: abs(x)).sum()) / len(rms_df)

        gsma = (rms_df['imu_gx'].apply(lambda x: abs(x)).sum() +
                rms_df['imu_gy'].apply(lambda x: abs(x)).sum() +
                rms_df['imu_gz'].apply(lambda x: abs(x)).sum()) / len(rms_df)

        # averaged acceleration energy (AAE)
        aae = Performance.__do_fft(rms_df['a_rms']) / len(rms_df)

        # averaged rotation energy (ARE)
        are = Performance.__do_fft(rms_df['g_rms']) / len(rms_df)

        # skewness (ASS)
        # ass_child = 0
        # for i in range(len(rms_df)):
        #     row = rms_df.iloc[i]
        #     mit = float(row['a_rms'])
        #     ass_child += (mit - aai) ** 3
        # ass_child /= len(rms_df)
        # ass = ass_child / math.pow(avi, 3.0 / 2.0)

        # skewness (GSS)
        # gss_child = 0
        # for i in range(len(rms_df)):
        #     row = rms_df.iloc[i]
        #     mit = float(row['g_rms'])
        #     gss_child += (mit - gai) ** 3
        # gss_child /= len(rms_df)
        # gss = gss_child / math.pow(gvi, 3.0 / 2.0)

        # kurtosis (AKS)
        # aks_child = 0
        # for i in range(len(rms_df)):
        #     row = rms_df.iloc[i]
        #     mit = float(row['a_rms'])
        #     aks_child += (mit - aai) ** 4
        # aks_child /= len(rms_df)
        # aks = aks_child / (avi ** 3) - 3

        # kurtosis (GKS)
        # gks_child = 0
        # for i in range(len(rms_df)):
        #     row = rms_df.iloc[i]
        #     mit = float(row['g_rms'])
        #     gks_child += (mit - gai) ** 4
        # gks_child /= len(rms_df)
        # gks = gks_child / (gvi ** 3) - 3

        # interquartile range (AIR)
        air = rms_df['a_rms'].quantile(0.75) - rms_df['a_rms'].quantile(0.25)

        # interquartile range (GIR)
        gir = rms_df['g_rms'].quantile(0.75) - rms_df['g_rms'].quantile(0.25)

        # crossing rate
        a_zero_cross = 0
        g_zero_cross = 0
        a_mean_cross = 0
        g_mean_cross = 0
        for i in range(1, len(rms_df)):
            prev_row = rms_df.iloc[i - 1]
            now_row = rms_df.iloc[i]
            if (prev_row['a_rms'] <= mami) and (now_row['a_rms'] > mami):
                a_zero_cross += 1
            elif (prev_row['a_rms'] >= mami) and (now_row['a_rms'] < mami):
                a_zero_cross += 1

            if (prev_row['g_rms'] <= mgmi) and (now_row['g_rms'] > mgmi):
                g_zero_cross += 1
            elif (prev_row['g_rms'] >= mgmi) and (now_row['g_rms'] < mgmi):
                g_zero_cross += 1

            if (prev_row['a_rms'] >= aai) and (now_row['a_rms'] < aai):
                a_mean_cross += 1
            elif (prev_row['a_rms'] <= aai) and (now_row['a_rms'] > aai):
                a_mean_cross += 1

            if (prev_row['g_rms'] >= gai) and (now_row['g_rms'] < gai):
                g_mean_cross += 1
            elif (prev_row['g_rms'] <= gai) and (now_row['g_rms'] > gai):
                g_mean_cross += 1

        a_zero_cross /= len(rms_df)
        g_zero_cross /= len(rms_df)
        a_mean_cross /= len(rms_df)
        g_mean_cross /= len(rms_df)

        # correlation between two axes
        a_xy_corr = rms_df['imu_ax'].corr(rms_df['imu_ay'])
        a_yz_corr = rms_df['imu_ay'].corr(rms_df['imu_az'])
        a_zx_corr = rms_df['imu_az'].corr(rms_df['imu_ax'])
        g_xy_corr = rms_df['imu_gx'].corr(rms_df['imu_gy'])
        g_yz_corr = rms_df['imu_gy'].corr(rms_df['imu_gz'])
        g_zx_corr = rms_df['imu_gz'].corr(rms_df['imu_gx'])

        # eigenvalues of dominant directions (EVA)
        # w, v = np.linalg.eig(rms_df[['imu_ax', 'imu_ay', 'imu_az']].corr().as_matrix())
        # evas = w[np.argpartition(w, -2)[-2:]]
        #
        return [aai, avi, asma, gai, gvi, gsma, aae, are,
                mami, mgmi, asdi, gsdi, air, gir,
                a_zero_cross, g_zero_cross, a_mean_cross, g_mean_cross,
                a_xy_corr, a_yz_corr, a_zx_corr, g_xy_corr, g_yz_corr, g_zx_corr]

    @staticmethod
    def __adjust_zero(df, modes_dict):
        """
        Implement zero adjust.

        :param df: dataframe needed zero adjust
        :param modes_dict: dictionary containing all attributes in {"attribute": "mode"} sense
        :return: zero-adjusted dataframe
        """

        copy_df = df.copy()

        # only considered attributes need zero adjust
        for col in tkconfig.ZERO_ADJ_COL:
            copy_df[col] = copy_df[col] - modes_dict[col]

        return copy_df

    @staticmethod
    def __get_modes_dict(df):
        """
        Create mode dictionary of the play dataframe.

        :param df: considered dataframe
        :return: dictionary containing all attributes in {"attribute": "mode"} sense
        """

        modes = {}
        copy_df = df.copy()
        for col in tkconfig.ZERO_ADJ_COL:
            mode_ = mode(copy_df[col])[0]
            modes[col] = mode_
        return modes

    def __retrieve_event(self):
        """
        Retrieve event which means note occurs of the song.

        :return: 2D array
        """

        events = []

        # spot vertical mark lines
        for i in range(len(self._song_df)):
            row = self._song_df.iloc[i]
            hit_type = int(row['label'])
            events.append((self._first_hit_time + row['timestamp'], hit_type))

        return events

    def plot_global_event(self):
        """
        Plot time series of the song decorated vertical lines represents events with different colors.

        :return:
        """

        for col in tkconfig.ALL_COLUMNS:
            # skip these two columns with no sense
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
    def song_df(self):
        return self._song_df

    @property
    def event_primitive_df(self):
        return self._event_primitive_df

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

    @property
    def bar_unit(self):
        return self._bar_unit


class Model(object):
    """
    consider training "only one" performance to predict other performance with the same 'song_id'.

    :protected attributes:
        K: set # numbers of centroid in k-means algorithm
        C: set param 'C' in SVM classifier
        left_vocab_kmeans: vocabulary category model of left arm
        right_vocab_kmeans: vocabulary category model of right arm
        left_clf: classifier handles histogram coming from left arm
        right_clf: classifier handles histogram coming from right arm
    """

    _K = 150
    _C = 1000

    def __init__(self, k_centroid=_K, tolerance=_C):
        self._K = k_centroid
        self._C = tolerance

        self._vocab_kmeans, self._max_abs_scaler = None, None
        self._clf = None

    def fit(self, performance):
        """
        Given the performance, use it to train the model.

        :param performance: training set
        :return:
        """

        vocab_kmeans, max_abs_scaler = self.__build_vocabulary_kmeans(performance)
        self._vocab_kmeans, self._max_abs_scaler = vocab_kmeans, max_abs_scaler

        x, y = self.retrieve_primitive_space(performance)

        self._clf = SVC(C=self._C)
        self._clf.fit(x, y)

    def retrieve_primitive_space(self, performance):
        """
        Given performance, extract all primitives.

        :param performance: training set
        :return: left_x, right_x, y
            left_x: 2D array whose element (c1, c2, ..., cK) means frequency in left arm
            right_x: 2D array whose element (c1, c2, ..., cK) means frequency in right arm
            y: 1D array, true label
        """

        pf = performance
        x = []
        y = []

        sys.stdout.flush()
        for id_, tm in tqdm(enumerate(pf.events), total=len(pf.events)):
            event_time = pf.events[id_][0]
            hit_type = pf.events[id_][1]
            # if hit_type == 0 or hit_type == 3:
            #     continue
            local_start_time = event_time - pf.delta_t
            local_end_time = event_time + pf.delta_t

            hist = self.get_local_primitive_hist(pf.left_play_df, pf.right_play_df,
                                                 local_start_time, local_end_time,
                                                 self._vocab_kmeans, self._max_abs_scaler,
                                                 pf.unit_time_interval)

            x.append(hist)
            y.append(hit_type)

        return x, y

    def __build_vocabulary_kmeans(self, performance):
        """
        Implement k-means algorithm to build vocabulary.

        :param performance: training set
        :return:
            vocab_kmeans: k-means model
            max_abs_scaler: normalize numeric data
        """

        pf = performance
        max_abs_scaler = preprocessing.StandardScaler()
        subset = pf.primitive_df[tkconfig.L_STAT_COLS + tkconfig.R_STAT_COLS]
        train_x = [tuple(x) for x in subset.values]
        train_x = max_abs_scaler.fit_transform(train_x)
        print(train_x)
        vocab_kmeans = KMeans(n_clusters=self._K, random_state=0).fit(train_x)
        return vocab_kmeans, max_abs_scaler

    def predict(self, performance, true_label=True):
        """
        Given a test set, predict all events which they belongs to.

        :param performance: testing set
        :param true_label: if "True", return true label "y" in addition
        :return:
            left_pred_y: predicting result of left arm
            right_pred_y: predicting result of right arm
            y: true label
        """

        x, y = self.retrieve_primitive_space(performance)
        pred_y = self._clf.predict(x)

        if true_label:
            return pred_y, y
        else:
            return pred_y

    def get_local_primitive_hist(self, left_play_df, right_play_df, start_time, end_time,
                                 vocab_kmeans, max_abs_scaler, unit_time_interval):
        """
        Given a dataframe of a play with specific time interval, use trained model to label local event.

        :param play_df: dataframe of a play
        :param start_time: timestamp of the local event starts
        :param end_time: timestamp of the local event ends
        :param vocab_kmeans: trained vocabulary
        :param max_abs_scaler: trained scaler to normalize numeric
        :param unit_time_interval: time interval of a primitive considered
        :return: frequency vector represents histogram with a primitive
        """

        left_local_play_df = left_play_df[(left_play_df['timestamp'] >= start_time) &
                                          (left_play_df['timestamp'] <= end_time)]
        right_local_play_df = right_play_df[(right_play_df['timestamp'] >= start_time) &
                                            (right_play_df['timestamp'] <= end_time)]

        # unit_time_interval was set
        local_stat_df = pd.DataFrame(columns=['seq_id', 'hand_side'] + tkconfig.STAT_COLS)
        now_time = start_time
        id_ = 0
        seq_id = 0

        # feature extraction
        while now_time + unit_time_interval <= end_time:
            unit_start_time = now_time
            unit_end_time = now_time + unit_time_interval

            prefix = [seq_id, tkconfig.LEFT_HAND]
            local_stat_df.loc[id_] = prefix + Performance.get_statistical_features(left_local_play_df,
                                                                                   unit_start_time,
                                                                                   unit_end_time)
            id_ += 1

            prefix = [seq_id, tkconfig.RIGHT_HAND]
            local_stat_df.loc[id_] = prefix + Performance.get_statistical_features(right_local_play_df,
                                                                                   unit_start_time,
                                                                                   unit_end_time)
            id_ += 1

            now_time += unit_time_interval
            seq_id += 1

        # merge left and right hand depends on the same time
        left_local_df = local_stat_df[local_stat_df['hand_side'] == tkconfig.LEFT_HAND][['seq_id'] +
                                                                                        tkconfig.STAT_COLS]
        right_local_df = local_stat_df[local_stat_df['hand_side'] == tkconfig.RIGHT_HAND][['seq_id'] +
                                                                                          tkconfig.STAT_COLS]
        for col in tkconfig.STAT_COLS:
            left_local_df.rename(columns={col: 'L_' + col}, inplace=True)
            right_local_df.rename(columns={col: 'R_' + col}, inplace=True)

        local_stat_df = left_local_df.merge(right_local_df, on='seq_id').drop('seq_id', axis=1)
        local_stat_df.dropna(inplace=True)

        # build freq histogram
        vec = np.zeros(self._K)

        if len(local_stat_df) == 0:
            return vec

        subset = local_stat_df[tkconfig.L_STAT_COLS + tkconfig.R_STAT_COLS]
        train_x = [tuple(x) for x in subset.values]
        train_x = max_abs_scaler.transform(train_x)
        for group_num in vocab_kmeans.predict(train_x):
            vec[group_num] += 1

        return vec
