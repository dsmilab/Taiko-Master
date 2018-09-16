from ..config import *
from ..io import *

import pandas as pd
from scipy.stats import mode

DELTA_T_DIVIDED_COUNT = 8

__all__ = ['get_play']


class _Play(object):
    """
    Handle particular arms of sensor data for the record.

    :protected attributes:
        play_dict: dictionary of {label: play_df}
        start_time: timestamp where the song starts
        end_time: timestamp where the song ends
        first_hit_time: timestamp where the first note occurs
    """

    def __init__(self, who_id, song_id, order_id, is_zero_adjust, resample):
        rec = get_record(who_id, song_id, order_id)

        self._play_dict = {}
        self._start_time, self._end_time = None, None
        self._first_hit_time = None

        self._note_df = load_note_df(who_id, song_id, order_id)
        self._time_unit = mode(self._note_df['time_unit'])[0]
        self._resampling_rate = '%.2fS' % (float(self._time_unit / DELTA_T_DIVIDED_COUNT))

        self.__set_hw_time(rec)
        self._events = self.__retrieve_event(who_id, song_id, order_id)
        self._play_dict['R'] = self.__build_play_df(RIGHT_HAND, is_zero_adjust, resample)
        self._play_dict['L'] = self.__build_play_df(LEFT_HAND, is_zero_adjust, resample)

    def __set_hw_time(self, rec):
        self._start_time = rec['hw_start_time']
        self._end_time = rec['hw_end_time']
        self._first_hit_time = rec['first_hit_time']

    def __build_play_df(self, handedness, calibrate, resample=False):
        """
        After setting duration of the song, build dataframe of a play.

        :param handedness: original dataframe
        :param calibrate: default is "None", adjust zero by this own case, otherwise will by this param
        :param resample: if not "None", resample by this frequency
        :return: cropped and zero-adjusted dataframe, attributes' modes
        """

        # crop desired play time interval
        df = load_arm_df(handedness)
        play_df = df[(df['timestamp'] >= self._start_time) &
                     (df['timestamp'] <= self._end_time)].copy()

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

    def __retrieve_event(self, who_id, song_id, order_id):
        """
        Retrieve event which means note occurs of the song.

        :return: 2D array
        """

        events = []
        note_df = load_note_df(who_id, song_id, order_id)
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


def get_play(who_id, song_id, order_id, calibrate=True, resample=True):
    """
    Get the particular play.

    :param who_id: # of drummer
    :param song_id: # of song
    :param order_id: # of performance repetitively
    :param calibrate: if "True", implement zero adjust
    :param resample: if not "None", resample by this frequency
    :return:
    """

    return _Play(who_id, song_id, order_id, calibrate, resample)
