import time
from datetime import datetime, timedelta

__all__ = ['get_hwclock_time',
           'get_hwclock_time_2',
           'get_datetime',
           'convert_datetime_format']


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
