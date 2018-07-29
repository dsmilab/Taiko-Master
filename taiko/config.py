import logging
import os

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
LEFT_PATH = os.path.join(BASE_PATH, '../data/bb_left_forearm_csv/')
RIGHT_PATH = os.path.join(BASE_PATH, '../data/bb_right_forearm_csv/')

ZERO_ADJ_COL = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz', 'msu_ax', 'msu_ay', 'msu_az']