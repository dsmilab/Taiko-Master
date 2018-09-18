from datetime import datetime, timedelta
import time

__all__ = ['get_hwclock_time', 'get_datetime']


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


def get_datetime(utc0_time, delta=0):
    d = datetime.fromtimestamp(utc0_time) + timedelta(seconds=int(delta))
    utc8 = datetime.fromtimestamp(time.mktime(d.timetuple()))
    return utc8
