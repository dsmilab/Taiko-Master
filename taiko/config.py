import logging
import os

# io.arm
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
LEFT_PATH = os.path.join(BASE_PATH, '../data/bb_left_forearm_csv/')
RIGHT_PATH = os.path.join(BASE_PATH, '../data/bb_right_forearm_csv/')

# io.record
TABLE_PATH = os.path.join(BASE_PATH, '../data/taiku_tables/')
PLAY_TABLE_PATH = os.path.join(TABLE_PATH, 'taiko_play.csv')
SONG_TABLE_PATH = os.path.join(TABLE_PATH, 'taiko_song.csv')
DRUMMER_TABLE_PATH = os.path.join(TABLE_PATH, 'taiko_drummer.csv')

ZERO_ADJ_COL = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz', 'msu_ax', 'msu_ay', 'msu_az']