from .config import *
from .io import *
from .image import *

import pandas as pd
import numpy as np
import posixpath
from scipy.stats import mode
import glob as glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

import scipy
DELTA_T_DIVIDED_COUNT = 8
DUMMY_TIME_LENGTH = 15

__all__ = ['get_play']


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
            # self._play_dict[position] = self.__build_play_df(raw_arm_df, calibrate, resample)
            self._play_dict[position] = raw_arm_df

    def crop_near_raw_data(self, delta_t=0.12):
        for position in ['L', 'R']:
            for id_, _ in enumerate(self._events):
                event_time = self._events[id_][0]
                hit_type = self._events[id_][1]

                if hit_type < 1 or hit_type > 2:
                    continue

                local_start_time = event_time - delta_t
                local_end_time = event_time + delta_t
                note_type = 'don' if hit_type == 1 else 'ka'

                filename = posixpath.join(LOCAL_MOTIF_DIR_PATH, note_type, position, '%03d.csv' % id_)
                os.makedirs(os.path.dirname(filename), exist_ok=True)

                df = self._play_dict[position]
                motif_df = df[(df['timestamp'] >= local_start_time) &
                              (df['timestamp'] <= local_end_time)].copy()

                motif_df.to_csv(filename, index=False)

    def preprocess(self, df):

        df = np.asarray(df, dtype=np.float32)
        if len(df.shape) == 1:
            raise ValueError('Data must be a 2-D array')

    #     if np.any(sum(np.isnan(df)) != 0):
    #         print('Data contains null values. Will be replaced with 0')
    #         df = np.nan_to_num()

        #standardize data
        df = StandardScaler().fit_transform(df)
        #normalize data
        df = MinMaxScaler().fit_transform(df)
        return df

    def crop_input(self, Dir_sensData, Dir_captures, Dir_song_spectrum,order):
        Dir  = Dir_sensData
        mypath = Dir_captures
        path1 = posixpath.join(Dir, 'L*.csv')
        path2 = posixpath.join(Dir, 'R*.csv')

        startTime = get_play_start_time(mypath)
        print(startTime)
        filenames1 = glob.glob(path1)
        filenames2 = glob.glob(path2)
        print(filenames1)
        print(filenames2)
        print(Dir)
        df_left = pd.read_csv(filenames1[0])
        df_left = df_left[df_left['timestamp']>startTime]
        print(df_left)
        df_left = df_left.reset_index()
        df_left = df_left.drop(['index'],axis=1)
        df_left = df_left[:-1]
        df_right =  pd.read_csv(filenames2[0])
        print(df_right)
        df_right = df_right[df_right['timestamp']>startTime]
        df_right = df_right.reset_index()
        df_right = df_right.drop(['index'],axis=1)
        df_right = df_right[:-1]

        #timestamp convert to datetime
        df_left['timestamp'] = pd.to_datetime(df_left['timestamp'], unit='s')
        df_right['timestamp'] = pd.to_datetime(df_right['timestamp'], unit='s')

        #resampliong
        df_left = df_left.resample('0.008S', on='timestamp').sum()
        df_right = df_right.resample('0.008S', on='timestamp').sum()

        #combine two hands
        df = pd.concat([df_left,df_right],axis=1)

        columns=['L_imu_ax',
                 'L_imu_ay',
                 'L_imu_az',
                 'L_imu_gx',
                 'L_imu_gy',
                 'L_imu_gz',
                 'R_imu_ax',
                 'R_imu_ay',
                 'R_imu_az',
                 'R_imu_gx',
                 'R_imu_gy',
                 'R_imu_gz']

        df.columns=columns

        df = df.dropna(axis=0,how='any')

        #scaling
        df_scaled = self.preprocess(df)
        df_scaled = pd.DataFrame(df_scaled)
        df_scaled.columns=columns
        df_scaled = df_scaled.set_index(df.index)
        print( 'input finished with Scaling')

        timestamps = df_scaled.index-df_scaled.index[0]

        seconds = []
        for index in timestamps:
            second = pd.Timedelta.total_seconds(index)
            seconds.append(second)

        secondsIndex = pd.Index(seconds)
        df_secondsIndex = df_scaled.set_index(secondsIndex)

        df_spectrum = pd.read_csv(Dir_song_spectrum)


        #cut motifs
        motifs_don = [] # motifs of don
        motifs_non = [] # motifs of non
        motifs_continous = []
        flag_is_five = 0
        count_five = 0

        for i in range(len(df_spectrum)):
            if(df_spectrum['label'][i]==1):
                time_mid = df_spectrum['Rel_timestemp (s)'][i]+0.4
                location = df_secondsIndex.index.get_loc(time_mid, method='nearest')
                lower = location-12
                upper = location+12
                df_reset = df_secondsIndex.reset_index() #index重製
                motif_i = df_reset[lower:upper]
                motifs_don.append(motif_i)
                motif_i.to_csv(Dir+'/motifs/don/motifs_%d_order_%d.csv'%(i,order))


            elif(df_spectrum['label'][i]==0):
                time_mid = df_spectrum['Rel_timestemp (s)'][i]+0.4
                location = df_secondsIndex.index.get_loc(time_mid, method='nearest')
                lower = location-12
                upper = location+12
                df_reset = df_secondsIndex.reset_index() #index重製
                motif_i = df_reset[lower:upper]
                motifs_non.append(motif_i)
                motif_i.to_csv(Dir+'/motifs/non/motifs_%d_order_%d.csv'%(i,order))

            elif(df_spectrum['label'][i]==2):
                time_mid = df_spectrum['Rel_timestemp (s)'][i]+0.4
                location = df_secondsIndex.index.get_loc(time_mid, method='nearest')
                lower = location-12
                upper = location+12
                df_reset = df_secondsIndex.reset_index() #index重製
                motif_i = df_reset[lower:upper]
                motifs_non.append(motif_i)
                motif_i.to_csv(Dir+'/motifs/ka/motifs_%d_order_%d.csv'%(i,order))

            if(df_spectrum['label'][i]==5):
                if(count_five==0):
                    flag_is_five=1
                count_five = count_five+1
            else:
                if(flag_is_five!=0):
                    time_end = df_spectrum['Rel_timestemp (s)'][i-1]+0.4
                    location_end = df_secondsIndex.index.get_loc(time_end, method='nearest')
                    upper = location_end+12
                    time_start = df_spectrum['Rel_timestemp (s)'][i-count_five]+0.4
                    location_start = df_secondsIndex.index.get_loc(time_start, method='nearest')
                    lower = location_end-12
                    df_reset = df_secondsIndex.reset_index() #index重製
                    motif_i = df_reset[lower:upper]
                    motifs_continous.append(motif_i)
                    motif_i.to_csv(Dir+'/motifs/motifs_%d_order_%d.csv'%(i,order))
                    print( 'cutting motifs finished !')
        return df_scaled ,df_spectrum

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
    play_start_time = get_play_start_time(capture_dir_path)

    raw_arm_df_dict = {
        left_arm_filename[0]: left_arm_df,
        right_arm_filename[0]: right_arm_df,
    }

    return _Play(song_id, raw_arm_df_dict, play_start_time, calibrate, resample)
