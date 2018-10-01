from .config import *
from .tools.score import *
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import posixpath
from keras.models import load_model
import re

from skimage.io import imshow, imsave, imread
from skimage.transform import resize
from skimage.color import rgb2grey

__all__ = ['read_result_board_info', 'get_play_start_time']


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ResultProcessor(metaclass=Singleton):
    COLUMNS = ['score', 'good', 'ok', 'bad', 'max_combo', 'drumroll']

    X_ANCHOR = [275, 258, 279, 300, 258, 279]
    Y_ANCHOR = [279, 438, 438, 438, 561, 561]

    IMG_ROWS = [20, 15, 15, 15, 15, 15]
    IMG_COLS = [15, 10, 10, 10, 10, 10]

    DIGIT_COUNTS = [6, 4, 4, 4, 4, 4]

    TARGET_IMG_ROWS = 15
    TARGET_IMG_COLS = 10

    def __init__(self):
        self._model = load_model(MNIST_MODEL_PATH)

    def process(self, pic_path):
        rp = ResultProcessor

        img = imread(pic_path)
        all_digits = []
        for pos in range(len(rp.DIGIT_COUNTS)):
            digits = []
            for digit in range(rp.DIGIT_COUNTS[pos]):
                cropped = img[rp.X_ANCHOR[pos]:rp.X_ANCHOR[pos] + rp.IMG_ROWS[pos],
                              rp.Y_ANCHOR[pos] + digit * rp.IMG_COLS[pos]:rp.Y_ANCHOR[pos] + (digit + 1) * rp.IMG_COLS[pos]]
                cropped = resize(cropped, (rp.TARGET_IMG_ROWS, rp.TARGET_IMG_COLS),
                                 mode='constant', preserve_range=False)
                cropped = rgb2grey(cropped)
                digits.append(cropped)

            all_digits.extend(digits)

        all_digits = np.asarray(all_digits)
        all_digits = all_digits.reshape(all_digits.shape[0], rp.TARGET_IMG_ROWS, rp.TARGET_IMG_COLS, 1)
        all_scores = self._model.predict_classes(all_digits)

        processed_result = self.__process_scores(all_scores)
        if processed_result:
            return self._result
        else:
            return None

    def __process_scores(self, all_scores):
        rp = ResultProcessor

        result_dict = {}
        front = 0
        for pos in range(len(rp.DIGIT_COUNTS)):
            score = get_processing_score(all_scores[front: front + rp.DIGIT_COUNTS[pos]])
            if score is None:
                return False
            front += rp.DIGIT_COUNTS[pos]

            result_dict[ResultProcessor.COLUMNS[pos]] = score

        self._result = result_dict
        return True

    @property
    def result(self):
        return self._result


class DrumProcessor(metaclass=Singleton):
    X_ANCHOR = 95
    Y_ANCHOR = 85

    IMG_ROWS = 65
    IMG_COLS = 65

    def __init__(self):
        self._model = load_model(DRUM_IMG_MODEL_PATH)

    def process(self, pic_path):
        dp = DrumProcessor

        img = imread(pic_path)
        cropped = img[dp.X_ANCHOR:dp.X_ANCHOR + dp.IMG_ROWS, dp.Y_ANCHOR:dp.Y_ANCHOR + dp.IMG_COLS]
        cropped = rgb2grey(cropped)
        x_train = [cropped]
        x_train = np.asarray(x_train)
        x_train = x_train.reshape(x_train.shape[0], dp.IMG_ROWS, dp.IMG_COLS, 1)
        x = self._model.predict_classes(x_train)[0]

        return True if x == 1 else False


def get_play_start_time(capture_dir_path):
    files = glob(posixpath.join(capture_dir_path, '*'))

    for pic_path in sorted(files):
        is_drum = DrumProcessor().process(pic_path)
        if is_drum:
            res = re.search('(\d){4}-(\d)+.(\d)+.png', pic_path)
            filename = res.group(0)
            timestamp = float(filename[5:-4])
            return timestamp

    raise Exception('unknown drum detected')


def read_result_board_info(capture_dir_path):
    files = glob(posixpath.join(capture_dir_path, '*'))

    for pic_path in reversed(sorted(files)):
        result = ResultProcessor().process(pic_path)
        if result is not None:
            return result

    raise Exception('unknown info detected')
