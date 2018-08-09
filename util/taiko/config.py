import os

# constant
RIGHT_HAND = 0
LEFT_HAND = 1
BASE_PATH = os.path.dirname(os.path.realpath(__file__))

# io.arm
LEFT_PATH = os.path.join(BASE_PATH, '../../data/bb_left_forearm_csv/')
RIGHT_PATH = os.path.join(BASE_PATH, '../../data/bb_right_forearm_csv/')

# io.record
TABLE_PATH = os.path.join(BASE_PATH, '../../data/taiko_tables/')
PLAY_TABLE_PATH = os.path.join(TABLE_PATH, 'taiko_play.csv')
SONG_TABLE_PATH = os.path.join(TABLE_PATH, 'taiko_song.csv')
DRUMMER_TABLE_PATH = os.path.join(TABLE_PATH, 'taiko_drummer.csv')

# image.scoreboard
BB_CAPTURE_PATH = os.path.join(BASE_PATH, '../../bb_capture_output/')
MNIST_MODEL_PATH = os.path.join(BASE_PATH, 'model/mnist_model.h5')

# preprocessing.primitive
STAT_COLS = ['AAI', 'AVI', 'ASMA', 'GAI', 'GVI', 'GSMA', 'AAE', 'ARE',
             'MAMI', 'MGMI', 'ASDI', 'GSDI', 'AIR', 'GIR',
             'AZCR', 'GZCR', 'AMCR', 'GMCR',
             'AXYCORR', 'AYZCORR', 'AZXCORR', 'GXYCORR', 'GYZCORR', 'GZXCORR']

ZERO_ADJ_COL = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz', 'msu_ax', 'msu_ay', 'msu_az']

ALL_COLUMNS = ['timestamp', 'wall_time', 'imu_temp',
               'imu_ax', 'imu_ay', 'imu_az',
               'imu_gx', 'imu_gy', 'imu_gz',
               'msu_ax', 'msu_ay', 'msu_az',
               'baro_temp', 'baro']

SENSOR_COLUMNS = [ALL_COLUMNS[0]] + ALL_COLUMNS[2:]
SCALE_COLUMNS = STAT_COLS[:-6]

# plot
COLORS = ['black', 'red', 'blue', 'yellow', 'green', 'cyan', 'purple']
