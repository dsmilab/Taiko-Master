import os
import posixpath
# constant
RESAMPLING_RATE = '0.01S'

RIGHT_HAND = 0
LEFT_HAND = 1

INTRO_DUMMY_TIME_LENGTH = 2
PLAY_ENDS_DUMMY_TIME_LENGTH = 3
SYNC_TIME_SHIFT = 1.8


BASE_PATH = posixpath.join(os.path.dirname(os.path.realpath(__file__)), '..').replace('\\', '/')

PIC_DIR_PATH = posixpath.join(BASE_PATH, '../assets/')

TMP_DIR_PATH = posixpath.join(BASE_PATH, '../tmp/')
LOCAL_SCREENSHOT_PATH = posixpath.join(TMP_DIR_PATH, 'bb_capture/')
LOCAL_SENSOR_DIR_PATH = posixpath.join(TMP_DIR_PATH, 'sensor_data/')
LOCAL_MOTIF_DIR_PATH = posixpath.join(TMP_DIR_PATH, 'motif/')
LOCAL_RECORD_TABLE_PATH = posixpath.join(TMP_DIR_PATH, 'record_table.csv')

DATA_DIR_PATH = posixpath.join(BASE_PATH, '../data/')
HOME_PATH = posixpath.join(DATA_DIR_PATH, 'alpha/')
CONNECT_HOST_DIR_PATH = posixpath.join(DATA_DIR_PATH, 'connect_host/')
MOTIF_DIR_PATH = posixpath.join(DATA_DIR_PATH, 'motif/')
PROFILE_DIR_PATH = posixpath.join(DATA_DIR_PATH, 'alpha_profile/')
PERFORMANCE_DIR_PATH = posixpath.join(DATA_DIR_PATH, 'alpha_performance/')
PROFILE_EP_DIR_PATH = posixpath.join(DATA_DIR_PATH, 'alpha_profile_ep/')

# io.record
TABLE_PATH = posixpath.join(BASE_PATH, '../data/taiko_tables/')
SONG_TABLE_PATH = posixpath.join(TABLE_PATH, 'taiko_song.csv')
PLAY_RESULT_TABLE_PATH = posixpath.join(TABLE_PATH, 'taiko_play_result.csv')

# network.client
REMOTE_BASE_PATH = 'beagle/'

SERVER_PROJECT_PATH = 'PyCharmPojects/Taiko-Master/'

SERVER_EXE_PATH = posixpath.join(SERVER_PROJECT_PATH, 'server_exe.py')

SERVER_TMP_DIR_PATH = posixpath.join(SERVER_PROJECT_PATH, 'tmp/')
SERVER_SCREENSHOT_PATH = posixpath.join(SERVER_TMP_DIR_PATH, 'uploaded_bb_capture/')

# image.scoreboard
BB_CAPTURE_PATH = posixpath.join(BASE_PATH, '../bb_capture_output/')

EXTERNAL_PATH = posixpath.join(BASE_PATH, 'external/')
MNIST_MODEL_PATH = posixpath.join(BASE_PATH, 'external/mnist_model.h5')
DRUM_IMG_MODEL_PATH = posixpath.join(BASE_PATH, 'external/drum_img_model.h5')
SPIRIT_IMG_MODEL_PATH = posixpath.join(BASE_PATH, 'external/spirit_img_model.h5')
ENCODER_MODEL_PATH = posixpath.join(BASE_PATH, 'external/encoder.h5')
VAE_MODEL_PATH = posixpath.join(BASE_PATH, 'external/vae.h5')

# connect.ssh
SSH_CONFIG_PATH = posixpath.join(BASE_PATH, '../data/connect_host/')

# .database
ENTRY_SUCCESS = '@0@'

# preprocessing.primitive
RMS_COLS = ['a_rms', 'g_rms', 'imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']

PREFIX_COLS = ['A', 'G', 'AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']
SUFFIX_COLS = ['AI', 'MMI', 'FR']

STAT_COLS = [prefix + '_' + suffix for prefix, suffix in zip(PREFIX_COLS, [SUFFIX_COLS[0]] * 8)] +\
            [prefix + '_' + suffix for prefix, suffix in zip(PREFIX_COLS, [SUFFIX_COLS[1]] * 8)] +\
            ['A_SMA', 'G_SMA'] +\
            [prefix + '_' + suffix for prefix, suffix in zip(PREFIX_COLS, [SUFFIX_COLS[2]] * 8)] +\
            ['AXY_CORR', 'AYZ_CORR', 'AZX_CORR', 'GXY_CORR', 'GYZ_CORR', 'GZX_CORR']

ZERO_ADJ_COL = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']

ALL_COLUMNS = ['timestamp', 'wall_time', 'imu_temp',
               'imu_ax', 'imu_ay', 'imu_az',
               'imu_gx', 'imu_gy', 'imu_gz',
               'msu_ax', 'msu_ay', 'msu_az',
               'baro_temp', 'baro']

SENSOR_COLUMNS = [ALL_COLUMNS[0]] + ALL_COLUMNS[3:9]

# plot
COLORS = ['black', 'red', 'blue', 'yellow', 'green', 'cyan', 'purple', 'magenta']

# regex
ACC_REGEX = '^\\w*_A[XYZ]{0,2}_\\w*$'
GYR_REGEX = '^\\w*_G[XYZ]{0,2}_\\w*$'
NEAR_REGEX = '^[LR]\\d+$'
HIT_TYPE_REGEX = '^(hit_type|[A-Z]+\\d)$'
NO_SCALE_REGEX = '^(\\w*_CORR|hit_type|[A-Z]+\\d|timestamp)$'

FIRST_HIT_ALIGN_DICT = {
    1: 90.25,  # doraemon
    2: 105.95,  # RPG
    3: 132.20,  # sakura
    4: 153.01,  # let it go
}

SCORE_UNIT_DICT = {
    1: 10490,
    2: 7180,
    3: 3900,
    4: 2940,
}

RESULT_BOARD_INFO_COLUMNS = ['score', 'good', 'ok', 'bad', 'max_combo', 'drumroll']

PREFIX = ['right_don', 'left_don', 'right_ka', 'left_ka', 'big_don', 'big_ka', 'pause', 'drumroll']
# id: avline
REF_AVLINE = {
    0: [1538104303.5, 1538104306.5,
        1538104306.8, 1538104309.2,
        1538104310.3, 1538104312.6,
        1538104313.3, 1538104315.6,
        1538104318.6, 1538104321.0,
        1538104321.9, 1538104324.5,
        1538104325.5, 1538104329.0,
        1538104329.5, 1538104333.5],

    24: [1538034458.0, 1538034461.8,
         1538034462.6, 1538034466.0,
         1538034467.0, 1538034470.8,
         1538034471.5, 1538034475.0,
         1538034477.5, 1538034481.2,
         1538034482.5, 1538034486.2,
         1538034487.0, 1538034490.0,
         1538034491.0, 1538034495.0],

    41: [1537861727.5, 1537861731.5,
         1537861731.6, 1537861736.1,
         1537861736.2, 1537861740.8,
         1537861740.9, 1537861745.5,
         1537861745.6, 1537861749.9,
         1537861750.0, 1537861754.6,
         1537861755.2, 1537861757.5,
         1537861758.1, 1537861764.1],

    58: [1537933057.5, 1537933061.5,
         1537933061.6, 1537933065.5,
         1537933065.6, 1537933070.0,
         1537933070.1, 1537933074.0,
         1537933074.1, 1537933078.5,
         1537933078.6, 1537933082.8,
         1537933082.9, 1537933083.6,
         1537933083.8, 1537933088.6],

    75: [1537963275.3, 1537963280.2,
         1537963281.1, 1537963286.0,
         1537963287.0, 1537963291.8,
         1537963292.5, 1537963297.5,
         1537963299.0, 1537963303.8,
         1537963305.0, 1537963310.0,
         1537963311.0, 1537963314.0,
         1537963314.8, 1537963319.8],

    92: [1537953146.3, 1537953149.5,
         1537953149.6, 1537953153.2,
         1537953153.3, 1537953157.0,
         1537953157.1, 1537953161.0,
         1537953161.2, 1537953165.0,
         1537953166.8, 1537953170.8,
         1537953171.7, 1537953175.0,
         1537953175.2, 1537953179.6],

    109: [1538122691.5, 1538122696.1,
          1538122696.4, 1538122701.0,
          1538122702.0, 1538122707.0,
          1538122707.5, 1538122712.8,
          1538122715.5, 1538122720.8,
          1538122722.0, 1538122727.0,
          1538122731.6, 1538122735.0,
          1538122735.1, 1538122739.7],

    126: [1537958367.5, 1537958372.3,
          1537958372.8, 1537958377.5,
          1537958378.8, 1537958383.7,
          1537958384.2, 1537958388.8,
          1537958391.0, 1537958396.0,
          1537958396.8, 1537958401.8,
          1537958402.1, 1537958407.2,
          1537958407.7, 1537958412.5],

    144: [1538118684.8, 1538118688.7,
          1538118690.5, 1538118694.3,
          1538118694.4, 1538118698.1,
          1538118698.8, 1538118702.6,
          1538118704.3, 1538118708.2,
          1538118708.8, 1538118712.8,
          1538118713.1, 1538118716.8,
          1538118717.1, 1538118720.8],

    161: [1537939766.5, 1537939770.5,
          1537939770.6, 1537939774.5,
          1537939774.8, 1537939778.5,
          1537939778.8, 1537939782.5,
          1537939782.7, 1537939786.0,
          1537939786.5, 1537939790.0,
          1537939791.0, 1537939794.0,
          1537939794.5, 1537939798.8],

    178: [1538392551.3, 1538392554.8,
          1538392555.0, 1538392558.8,
          1538392559.3, 1538392563.1,
          1538392563.3, 1538392567.2,
          1538392567.5, 1538392571.5,
          1538392571.7, 1538392575.5,
          1538392576.1, 1538392578.9,
          1538392579.2, 1538392583.4],

    195: [1538017589.4, 1538017593.6,
          1538017594.2, 1538017598.3,
          1538017599.3, 1538017603.5,
          1538017604.4, 1538017608.4,
          1538017609.1, 1538017613.0,
          1538017614.0, 1538017617.9,
          1538017618.3, 1538017623.8,
          1538017624.3, 1538017629.2],

    213: [1538199875.4, 1538199879.2,
          1538199880.0, 1538199884.0,
          1538199884.8, 1538199888.6,
          1538199889.6, 1538199893.2,
          1538199894.4, 1538199898.0,
          1538199899.1, 1538199902.8,
          1538199903.5, 1538199907.5,
          1538199908.0, 1538199912.2],

    248: [1537596430.3, 1537596437.6,
          1537596437.7, 1537596445.7,
          1537596446.1, 1537596454.0,
          1537596454.1, 1537596461.9,
          1537596462.0, 1537596470.0,
          1537596470.1, 1537596478.0,
          1537596484.6, 1537596487.0,
          1537596478.5, 1537596483.0],

    253: [1538220035.6, 1538220042.2,
          1538220042.3, 1538220049.0,
          1538220049.1, 1538220055.7,
          1538220055.8, 1538220062.7,
          1538220062.8, 1538220069.8,
          1538220069.9, 1538220077.0,
          1538220077.5, 1538220079.5,
          1538220079.9, 1538220083.8],

    270: [1537596860.8, 1537596869.0,
          1537596869.5, 1537596876.7,
          1537596877.1, 1537596884.2,
          1537596884.3, 1537596891.5,
          1537596891.8, 1537596899.0,
          1537596899.1, 1537596906.5,
          1537596910.4, 1537596911.2,
          1537596907.2, 1537596909.0],

    289: [1537855725.5, 1537855730.6,
          1537855731.1, 1537855736.6,
          1537855737.4, 1537855742.9,
          1537855743.2, 1537855748.6,
          1537855749.4, 1537855755.3,
          1537855755.7, 1537855761.0,
          1537855762.0, 1537855766.2,
          1537855767.2, 1537855771.0],

    306: [1538138331.0, 1538138337.0,
          1538138337.1, 1538138343.0,
          1538138343.8, 1538138349.7,
          1538138349.8, 1538138355.7,
          1538138357.5, 1538138363.5,
          1538138364.2, 1538138370.3,
          1538138371.0, 1538138372.7,
          1538138373.0, 1538138375.4],

    323: [1537881775.5, 1537881781.0,
          1537881781.1, 1537881786.3,
          1537881786.5, 1537881791.8,
          1537881792.0, 1537881797.2,
          1537881797.5, 1537881802.7,
          1537881802.8, 1537881807.8,
          1537881808.2, 1537881810.5,
          1537881810.7, 1537881815.0],

    340: [1538134456.7, 1538134461.6,
          1538134461.8, 1538134466.7,
          1538134466.8, 1538134471.6,
          1538134471.7, 1538134476.5,
          1538134476.8, 1538134481.5,
          1538134481.8, 1538134486.5,
          1538134487.0, 1538134489.0,
          1538134489.6, 1538134493.2],
}
