import os

# constant
RIGHT_HAND = 0
LEFT_HAND = 1
BASE_PATH = os.path.dirname(os.path.realpath(__file__))

# io.arm
LEFT_PATH = os.path.join(BASE_PATH, '../data/bb_left_forearm_csv/')
RIGHT_PATH = os.path.join(BASE_PATH, '../data/bb_right_forearm_csv/')

# io.record
TABLE_PATH = os.path.join(BASE_PATH, '../data/taiko_tables/')
PLAY_TABLE_PATH = os.path.join(TABLE_PATH, 'taiko_play.csv')
SONG_TABLE_PATH = os.path.join(TABLE_PATH, 'taiko_song.csv')
DRUMMER_TABLE_PATH = os.path.join(TABLE_PATH, 'taiko_drummer.csv')
RESULT_TABLE_PATH = os.path.join(TABLE_PATH, 'taiko_result.csv')

# image.scoreboard
BB_CAPTURE_PATH = os.path.join(BASE_PATH, '../bb_capture_output/')
MNIST_MODEL_PATH = os.path.join(BASE_PATH, 'image/mnist_model.h5')

# preprocessing.primitive
RMS_COLS = ['a_rms', 'g_rms', 'imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']

PREFIX_COLS = ['A', 'G', 'AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']
SUFFIX_COLS = ['AI', 'VI', 'MMI', 'SDI', 'IQR', 'FR', 'FFT_COEF', 'MDCR', 'MCR', 'ZCR', 'DTW']

STAT_COLS = [prefix + '_' + suffix for prefix, suffix in zip(PREFIX_COLS, [SUFFIX_COLS[0]] * 8)] +\
            [prefix + '_' + suffix for prefix, suffix in zip(PREFIX_COLS, [SUFFIX_COLS[1]] * 8)] +\
            [prefix + '_' + suffix for prefix, suffix in zip(PREFIX_COLS, [SUFFIX_COLS[2]] * 8)] +\
            ['A_SMA', 'G_SMA', 'A_AE', 'G_AE'] +\
            [prefix + '_' + suffix for prefix, suffix in zip(PREFIX_COLS, [SUFFIX_COLS[3]] * 8)] +\
            [prefix + '_' + suffix for prefix, suffix in zip(PREFIX_COLS, [SUFFIX_COLS[4]] * 8)] +\
            [prefix + '_' + suffix for prefix, suffix in zip(PREFIX_COLS, [SUFFIX_COLS[5]] * 8)] +\
            [prefix + '_' + suffix + '_' + str(i + 1)
             for prefix, suffix in zip(PREFIX_COLS, [SUFFIX_COLS[6]] * 8)
             for i in range(3)] +\
            [prefix + '_' + suffix for prefix, suffix in zip(PREFIX_COLS, [SUFFIX_COLS[7]] * 8)] +\
            [prefix + '_' + suffix for prefix, suffix in zip(PREFIX_COLS, [SUFFIX_COLS[8]] * 8)] +\
            [prefix + '_' + suffix for prefix, suffix in zip(PREFIX_COLS[2:], [SUFFIX_COLS[9]] * 6)] +\
            ['AXY_CORR', 'AYZ_CORR', 'AZX_CORR', 'GXY_CORR', 'GYZ_CORR', 'GZX_CORR']

KEYWORD_COLS = SUFFIX_COLS + ['SMA', 'AE', 'CORR']

ZERO_ADJ_COL = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz', 'msu_ax', 'msu_ay', 'msu_az']

ALL_COLUMNS = ['timestamp', 'wall_time', 'imu_temp',
               'imu_ax', 'imu_ay', 'imu_az',
               'imu_gx', 'imu_gy', 'imu_gz',
               'msu_ax', 'msu_ay', 'msu_az',
               'baro_temp', 'baro']

SENSOR_COLUMNS = [ALL_COLUMNS[0]] + ALL_COLUMNS[2:]
SCALE_COLUMNS = STAT_COLS[:-6]

# plot
COLORS = ['black', 'red', 'blue', 'yellow', 'green', 'cyan', 'purple', 'magenta']

# regex
ACC_REGEX = '^\w*_A[XYZ]{0,2}_\w*$'
GYR_REGEX = '^\w*_G[XYZ]{0,2}_\w*$'
NEAR_REGEX = '^[LR]\d+$'
HIT_TYPE_REGEX = '^(hit_type|[A-Z]+\d)$'
NO_SCALE_REGEX = '^(\w*_CORR|hit_type|[A-Z]+\d)$'
