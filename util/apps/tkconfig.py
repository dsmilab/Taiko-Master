import logging

# LOG_FORMAT = '%(asctime)s %(levelname)s << %(message)s'
# logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt='%H:%M:%S')

LEFT_PATH = '../data/bb_left_forearm_csv/'
RIGHT_PATH = '../data/bb_right_forearm_csv/'
TABLE_PATH = '../data/taiku_tables/'
PATH = '../data/bb_capture/'
OUTPUT_PATH = '../output/'
PROCESSED_PATH = '../data/'

BB_CAPTURE_PATH = '../bb_capture_output/'

LEFT_HAND = 'left'
RIGHT_HAND = 'right'

ALL_COLUMNS = ['timestamp', 'wall_time', 'imu_temp',
               'imu_ax', 'imu_ay', 'imu_az',
               'imu_gx', 'imu_gy', 'imu_gz',
               'msu_ax', 'msu_ay', 'msu_az',
               'baro_temp', 'baro']
ZERO_ADJ_COL = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz', 'msu_ax', 'msu_ay', 'msu_az']
STAT_COLS = ['AAI', 'AVI', 'ASMA', 'GAI', 'GVI', 'GSMA', 'AAE', 'ARE',
             'MAMI', 'MGMI', 'ASDI', 'GSDI', 'AIR', 'GIR',
             'AZCR', 'GZCR', 'AMCR', 'GMCR',
             'AXYCORR', 'AYZCORR', 'AZXCORR', 'GXYCORR', 'GYZCORR', 'GZXCORR']
L_STAT_COLS = ['L_AAI', 'L_AVI', 'L_ASMA', 'L_GAI', 'L_GVI', 'L_GSMA', 'L_AAE', 'L_ARE',
               'L_MAMI', 'L_MGMI', 'L_ASDI', 'L_GSDI', 'L_AIR', 'L_GIR',
               'L_AZCR', 'L_GZCR', 'L_AMCR', 'L_GMCR',
               'L_AXYCORR', 'L_AYZCORR', 'L_AZXCORR', 'L_GXYCORR', 'L_GYZCORR', 'L_GZXCORR']
R_STAT_COLS = ['R_AAI', 'R_AVI', 'R_ASMA', 'R_GAI', 'R_GVI', 'R_GSMA', 'R_AAE', 'R_ARE',
               'R_MAMI', 'R_MGMI', 'R_ASDI', 'R_GSDI', 'R_AIR', 'R_GIR',
               'R_AZCR', 'R_GZCR', 'R_AMCR', 'R_GMCR',
               'R_AXYCORR', 'R_AYZCORR', 'R_AZXCORR', 'R_GXYCORR', 'R_GYZCORR', 'R_GZXCORR']
COND_COLS = ['L2', 'L1', 'R1', 'R2']
COLORS = ['black', 'red', 'blue', 'yellow', 'green', 'cyan', 'purple']
