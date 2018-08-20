from datetime import datetime, timedelta
import time

__all__ = ['get_hwclock_time',
           'slide']


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


def slide(window, play_mat, play_id, local_start_time, local_end_time):
    if len(window) == 0 and play_id < len(play_mat):
        window.append(play_mat[play_id])
        play_id += 1

    while play_id < len(play_mat) and play_mat[play_id][0] < local_end_time:
        window.append(play_mat[play_id])
        play_id += 1

    while len(window) > 0 and window[0][0] < local_start_time:
        window.popleft()
    return window, play_mat, play_id
