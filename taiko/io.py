import pandas as pd
import numpy as np
from .config import *
from glob import glob

__all__ = ['load_arm_df',
           'load_note_df',
           'get_capture_dir_path']


class _Note(object):
    DICT = {
        1: 'easy',
        2: 'normal',
        3: 'easy',
        4: 'hard',
    }

    DROPPED_COLUMNS = ['#', 'separator', 'continuous']
    RENAMED_COLUMNS = ['bar', 'bpm', 'time_unit', 'timestamp', 'label']

    def __init__(self, song_id):
        self._song_id = song_id

        difficulty = _Note.DICT[song_id]
        note_file_name = 'taiko_song_%d_%s_info.csv' % (self._song_id, difficulty)

        self._note_df = pd.read_csv(posixpath.join(TABLE_PATH, note_file_name))
        self._note_df.drop(_Note.DROPPED_COLUMNS, axis=1, inplace=True)
        self._note_df.columns = _Note.RENAMED_COLUMNS

    @property
    def note_df(self):
        return self._note_df


def load_note_df(song_id):
    return _Note(song_id).note_df


class _ArmData(object):
    REMAINED_COLUMNS = ['timestamp', 'imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']

    def __init__(self, who_name, arm_filename, from_tmp_dir):
        self._arm_df = None

        self.__load_arm_csv(who_name, arm_filename, from_tmp_dir)

    def __load_arm_csv(self, who_name, arm_filename, from_tmp_dir):
        arm_path = None

        if from_tmp_dir:
            arm_path = posixpath.join(LOCAL_SENSOR_DIR_PATH, arm_filename)
        else:
            files = glob(posixpath.join(HOME_PATH, who_name, 'day[0-9]', 'sensor_data', arm_filename))

            if len(files) > 1:
                raise ValueError('More than one item matched')
            elif len(files) == 0:
                raise ValueError('No arm sensor data matched')

            arm_path = files[0]

        arm_df = pd.read_csv(arm_path, dtype={
            'timestamp': np.float64,
        })

        arm_df.drop(arm_df.tail(1).index, inplace=True)
        arm_df = arm_df[_ArmData.REMAINED_COLUMNS]
        for col in _ArmData.REMAINED_COLUMNS:
            arm_df[col] = arm_df[col].astype(np.float64)

        self._arm_df = arm_df

    @property
    def arm_df(self):
        return self._arm_df


def load_arm_df(who_name, arm_filename, from_tmp_dir=False):
    return _ArmData(who_name, arm_filename, from_tmp_dir).arm_df


def get_capture_dir_path(who_name, capture_dir_name, from_tmp_dir=False):
    capture_dir_path = None

    if from_tmp_dir:
        capture_dir_path = posixpath.join(LOCAL_SCREENSHOT_PATH, capture_dir_name)

    else:
        files = glob(posixpath.join(HOME_PATH, who_name, 'day[0-9]', 'bb_capture', capture_dir_name))

        if len(files) > 1:
            raise ValueError('More than one item matched')
        elif len(files) == 0:
            raise ValueError('No capture directory matched')

        capture_dir_path = files[0]

    return capture_dir_path
