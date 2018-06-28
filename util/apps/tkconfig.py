import logging

LOG_FORMAT = '%(asctime)s %(levelname)s << %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt='%H:%M:%S')

LEFT_PATH = '../data/bb_left_forearm_csv/'
RIGHT_PATH = '../data/bb_right_forearm_csv/'
TABLE_PATH = '../data/taiku_tables/'
PATH = '../data/bb_capture/'
OUTPUT_PATH = '../output/'
PROCESSED_PATH = '../data/'

LEFT_HAND = 'left'
RIGHT_HAND = 'right'

ALL_COLUMNS = ['timestamp', 'wall_time', 'imu_temp',
               'imu_ax', 'imu_ay', 'imu_az',
               'imu_gx', 'imu_gy', 'imu_gz',
               'msu_ax', 'msu_ay', 'msu_az',
               'baro_temp', 'baro']
ZERO_ADJ_COL = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz', 'msu_ax', 'msu_ay', 'msu_az']
STAT_COLS = ['AAI', 'AVI', 'ASMA', 'GAI', 'GVI', 'GSMA', 'AAE', 'ARE']
L_STAT_COLS = ['L_AAI', 'L_AVI', 'L_ASMA', 'L_GAI', 'L_GVI', 'L_GSMA', 'L_AAE', 'L_ARE']
R_STAT_COLS = ['R_AAI', 'R_AVI', 'R_ASMA', 'R_GAI', 'R_GVI', 'R_GSMA', 'R_AAE', 'R_ARE']
COLORS = ['black', 'red', 'blue', 'yellow', 'green', 'cyan', 'purple']