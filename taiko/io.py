from .tools.config import *

import pandas as pd
import numpy as np
from glob import glob

__all__ = ['load_arm_df',
           'load_note_df',
           'get_capture_dir_path']

"""
Constants
"""

DICT = {
    1: 'easy',
    2: 'normal',
    3: 'easy',
    4: 'hard',
}

DROPPED_COLUMNS = ['#', 'separator', 'continuous']
RENAMED_COLUMNS = ['bar', 'bpm', 'time_unit', 'timestamp', 'label']
REMAINED_COLUMNS = ['timestamp', 'imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']

"""
Modules
"""


def load_note_df(song_id) -> pd.DataFrame:
    difficulty = DICT[song_id]
    note_file_name = 'taiko_song_%d_%s_info.csv' % (song_id, difficulty)

    note_df = pd.read_csv(posixpath.join(TABLE_PATH, note_file_name))
    note_df.drop(DROPPED_COLUMNS, axis=1, inplace=True)
    note_df.columns = RENAMED_COLUMNS

    return note_df


def load_arm_df(who_name, arm_filename) -> pd.DataFrame:
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
    arm_df = arm_df[REMAINED_COLUMNS]
    for col in REMAINED_COLUMNS:
        arm_df[col] = arm_df[col].astype(np.float64)

    return arm_df


def get_capture_dir_path(who_name, capture_dir_name) -> str:
    files = glob(posixpath.join(HOME_PATH, who_name, 'day[0-9]', 'bb_capture', capture_dir_name))

    if len(files) > 1:
        raise ValueError('More than one item matched')
    elif len(files) == 0:
        raise ValueError('No capture directory matched')

    capture_dir_path = files[0]

    return capture_dir_path
