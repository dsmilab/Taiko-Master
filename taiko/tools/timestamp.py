from .config import *
import time
import pandas as pd
from scipy.stats import mode
from datetime import datetime, timedelta

__all__ = ['resample_sensor_df',
           'calibrate_sensor_df',
           'get_hwclock_time',
           'get_hwclock_time_2',
           'get_datetime',
           'convert_datetime_format']


def resample_sensor_df(df):
    result_df = df.copy()

    # resample for more samples
    result_df.loc[:, 'timestamp'] = pd.to_datetime(result_df['timestamp'], unit='s')
    result_df.loc[:, 'timestamp'] = result_df['timestamp'].apply(
        lambda x: x.tz_localize('UTC').tz_convert('Asia/Taipei'))
    result_df = result_df.set_index('timestamp').resample(RESAMPLING_RATE).mean()
    result_df = result_df.interpolate(method='linear')
    result_df.reset_index(inplace=True)
    result_df.loc[:, 'timestamp'] = result_df['timestamp'].apply(lambda x: x.timestamp())
    result_df.fillna(method='ffill', inplace=True)

    return result_df


def calibrate_sensor_df(df):
    modes_dict = {}
    calibrated_df = df.copy()

    for col in ZERO_ADJ_COL:
        mode_ = mode(calibrated_df[col])[0]
        modes_dict[col] = mode_

    # only considered attributes need zero adjust
    for col in ZERO_ADJ_COL:
        calibrated_df.loc[:, col] = calibrated_df[col] - modes_dict[col]

    return calibrated_df


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


def get_hwclock_time_2(local_time, delta=0):
    """
    Given UTC time, translate it into hardware time.

    :param local_time: UTC time in specific string format
    :param delta: pass UTC time in # seconds
    :return: required hardware time.
    """

    d = datetime.strptime(local_time, "%Y_%m_%d_%H_%M_%S")
    d = d + timedelta(seconds=int(delta))

    return time.mktime(d.timetuple())


def get_datetime(utc0_time, delta=0):
    d = datetime.fromtimestamp(utc0_time) + timedelta(seconds=int(delta))
    utc8 = datetime.fromtimestamp(time.mktime(d.timetuple()))
    return utc8


def convert_datetime_format(ori_start_time):
    d = datetime.strptime(ori_start_time, '%Y_%m_%d_%H_%M_%S')
    return d.strftime('%m/%d/%Y %H:%M:%S')
