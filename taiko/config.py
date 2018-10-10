import os
import posixpath
# constant
RIGHT_HAND = 0
LEFT_HAND = 1
BASE_PATH = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')


HOME_PATH = posixpath.join(BASE_PATH, '../data/alpha/')

PIC_DIR_PATH = posixpath.join(BASE_PATH, '../assets/')
TMP_DIR_PATH = posixpath.join(BASE_PATH, '../tmp/')

LOCAL_SENSOR_DIR_PATH = posixpath.join(TMP_DIR_PATH, 'sensor_data/')
LOCAL_RECORD_TABLE_PATH = posixpath.join(TMP_DIR_PATH, 'record_table.csv')


# io.record
TABLE_PATH = posixpath.join(BASE_PATH, '../data/taiko_tables/')
SONG_TABLE_PATH = posixpath.join(TABLE_PATH, 'taiko_song.csv')
PLAY_RESULT_TABLE_PATH = posixpath.join(TABLE_PATH, 'taiko_play_result.csv')

# network.client
REMOTE_BASE_PATH = 'Projects/beagle/'

SERVER_PROJECT_PATH = 'PyCharmPojects/Taiko-Master/'

SERVER_EXE_PATH = posixpath.join(SERVER_PROJECT_PATH, 'server_exe.py')

SERVER_TMP_DIR_PATH = posixpath.join(SERVER_PROJECT_PATH, 'tmp/')
SERVER_SCREENSHOT_PATH = posixpath.join(SERVER_TMP_DIR_PATH, 'uploaded_bb_capture/')

# capture.py
LOCAL_SCREENSHOT_PATH = posixpath.join(TMP_DIR_PATH, 'bb_capture/')

# image.scoreboard
BB_CAPTURE_PATH = posixpath.join(BASE_PATH, '../bb_capture_output/')

MNIST_MODEL_PATH = posixpath.join(BASE_PATH, 'external/mnist_model.h5')
DRUM_IMG_MODEL_PATH = posixpath.join(BASE_PATH, 'external/drum_img_model.h5')

# connect.ssh
SSH_CONFIG_PATH = posixpath.join(BASE_PATH, '../data/connect_host/')

# .database
ENTRY_SUCCESS = '@0@'

# preprocessing.primitive
RMS_COLS = ['a_rms', 'g_rms', 'imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']

ZERO_ADJ_COL = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']

ALL_COLUMNS = ['timestamp', 'wall_time', 'imu_temp',
               'imu_ax', 'imu_ay', 'imu_az',
               'imu_gx', 'imu_gy', 'imu_gz',
               'msu_ax', 'msu_ay', 'msu_az',
               'baro_temp', 'baro']


# plot
COLORS = ['black', 'red', 'blue', 'yellow', 'green', 'cyan', 'purple', 'magenta']

# regex
ACC_REGEX = '^\w*_A[XYZ]{0,2}_\w*$'
GYR_REGEX = '^\w*_G[XYZ]{0,2}_\w*$'
NEAR_REGEX = '^[LR]\d+$'
HIT_TYPE_REGEX = '^(hit_type|[A-Z]+\d)$'
NO_SCALE_REGEX = '^(\w*_CORR|hit_type|[A-Z]+\d)$'

SONG_LENGTH_DICT = {
    1: 89,
    2: 109,
    3: 134,
    4: 134,
    99: 500,
}

INTRO_LENGTH_DICT = {
    1: 2.18 - 1.8,
    2: 2.00 - 1.8,
    3: 1.82 - 1.8,
    4: 1.94 - 1.8,
}

RESULT_BOARD_INFO_COLUMNS = ['score', 'good', 'ok', 'bad', 'max_combo', 'drumroll']
