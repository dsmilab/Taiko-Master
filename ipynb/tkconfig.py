import logging

LOG_FORMAT = '%(asctime)s %(levelname)s << %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt='%H:%M:%S')

LEFT_PATH = '../data/bb_left_forearm_csv/'
RIGHT_PATH = '../data/bb_right_forearm_csv/'
TABLE_PATH = '../data/taiku_tables/'
PATH = '../data/bb_capture/'
OUTPUT_PATH = '../output/'
PROCESSED_PATH = '../data/'

ZERO_ADJ_COL = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz', 'msu_ax', 'msu_ay', 'msu_az']