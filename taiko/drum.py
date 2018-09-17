from config import *
import os
import sys
import pandas as pd
import numpy as np

from datetime import datetime
import time

from keras.models import load_model
from skimage.io import imread
from skimage.color import rgb2grey

__all__ = ['get_play_start_time']


IMG_ROWS = 65
IMG_COLS = 65
X_ANCHOR = 95
Y_ANCHOR = 85


def get_play_start_time(img_dir):
    model = load_model(DRUM_IMG_MODEL_PATH)
    files = sorted(next(os.walk(img_dir))[2])

    for file_id, filename in enumerate(files):
        img = imread(img_dir + '/' + filename)
        img = img[X_ANCHOR:X_ANCHOR + IMG_ROWS, Y_ANCHOR:Y_ANCHOR + IMG_COLS]
        img = rgb2grey(img)
        x_train = [img]
        x_train = np.asarray(x_train)
        x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
        x = model.predict_classes(x_train)[0]
        if x == 1:
            cap = float(filename[5:-4])
            return cap
    return None


def convert_datetime_format(ori_start_time):
    d = datetime.strptime(ori_start_time, '%Y_%m_%d_%H_%M_%S')
    return d.strftime('%m/%d/%Y %H:%M:%S')


def main(argv):
    if len(argv) < 4:
        return 1
    who_name = argv[0]
    gender = argv[1]
    song_id = int(argv[2])
    record_start_time = argv[3]
    record_start_time = convert_datetime_format(record_start_time)
    print(who_name, gender, song_id, record_start_time)
    drummer_df = pd.read_csv(DRUMMER_TABLE_PATH)
    if not (who_name in list(drummer_df.name)):
        last_id = drummer_df['id'].iloc[-1]
        index_ = drummer_df.index[-1] + 1
        drummer_df.loc[index_] = [last_id + 1, who_name, 0, gender, 0]
        drummer_df.to_csv(DRUMMER_TABLE_PATH, index=False)

    row = drummer_df[drummer_df.name == who_name]

    who_id = int(row['id'].iloc[0])

    song_df = pd.read_csv(SONG_TABLE_PATH)
    row = song_df[song_df.song_id == song_id]
    intro_length = float(row['intro_length'].iloc[0])
    play_start_time = get_play_start_time('tmp_capture/')

    first_hit_time = play_start_time + intro_length

    play_df = pd.read_csv(PLAY_TABLE_PATH)
    sel_df = play_df[(play_df.drummer_id == who_id) & (play_df.song_id == song_id)].copy()
    performance_order = 1
    if len(sel_df.performance_order) > 0:
        performance_order = max(sel_df.performance_order) + 1

    pid = 1
    if len(play_df.pid) > 0:
        pid = max(play_df.pid) + 1

    index_ = play_df.index[-1] + 1
    play_df.loc[index_] = [pid, who_id, song_id, performance_order, 'easy', record_start_time, first_hit_time]
    play_df.to_csv(PLAY_TABLE_PATH, index=False)

    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
