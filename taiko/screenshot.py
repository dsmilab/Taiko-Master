from config import *
import numpy as np
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
            timestamp = float(filename[5:-4])
            return timestamp
    return None
