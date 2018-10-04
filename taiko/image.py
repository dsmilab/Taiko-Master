from .config import *
from .tools.score import *
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from abc import abstractmethod
from glob import glob
import posixpath
from keras.models import load_model
import re

from skimage.io import imshow, imsave, imread
from skimage.transform import resize
from skimage.color import rgb2grey

__all__ = ['read_result_board_info',
           'read_score_board_info',
           'get_play_start_time']


class _Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class _Processor(object):

    def __init__(self):
        pass

    @abstractmethod
    def process(self, pic_path):
        raise NotImplementedError("Please Implement this method")


class _ScoreProcessor(metaclass=_Singleton):
    X_ANCHOR = 178
    Y_ANCHOR = 40

    IMG_ROW = 15
    IMG_COL = 10

    DIGIT_COUNT = 6

    def __init__(self):
        self._model = load_model(MNIST_MODEL_PATH)

    def process(self, pic_path):
        sp = _ScoreProcessor

        img = imread(pic_path)
        all_digits = []
        digits = []
        for digit in range(sp.DIGIT_COUNT):
            cropped = img[sp.X_ANCHOR: sp.X_ANCHOR + sp.IMG_ROW,
                          sp.Y_ANCHOR + sp.IMG_COL * digit: sp.Y_ANCHOR + sp.IMG_COL * (digit + 1)]
            cropped = rgb2grey(cropped)
            digits.append(cropped)
        all_digits.extend(digits)

        all_digits = np.asarray(all_digits)
        all_digits = all_digits.reshape(all_digits.shape[0], sp.IMG_ROW, sp.IMG_COL, 1)
        all_scores = self._model.predict_classes(all_digits, verbose=0)

        processed_result = self.__process_scores(all_scores)
        return processed_result

    @staticmethod
    def __process_scores(all_scores):
        img_score = get_processing_score(all_scores)
        return img_score


class _ResultProcessor(_Processor, metaclass=_Singleton):
    COLUMNS = ['score', 'good', 'ok', 'bad', 'max_combo', 'drumroll']

    X_ANCHOR = [275, 258, 279, 300, 258, 279]
    Y_ANCHOR = [279, 438, 438, 438, 561, 561]

    IMG_ROWS = [20, 15, 15, 15, 15, 15]
    IMG_COLS = [15, 10, 10, 10, 10, 10]

    DIGIT_COUNTS = [6, 4, 4, 4, 4, 4]

    TARGET_IMG_ROWS = 15
    TARGET_IMG_COLS = 10

    def __init__(self):
        _Processor.__init__(self)
        self._model = load_model(MNIST_MODEL_PATH)

    def process(self, pic_path):
        rp = _ResultProcessor

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
        all_scores = self._model.predict_classes(all_digits, verbose=0)

        processed_result = self.__process_scores(all_scores)
        return processed_result

    @staticmethod
    def __process_scores(all_scores):
        rp = _ResultProcessor

        result_dict = {}
        front = 0
        for pos in range(len(rp.DIGIT_COUNTS)):
            score = get_processing_score(all_scores[front: front + rp.DIGIT_COUNTS[pos]])
            if score is None:
                return None

            front += rp.DIGIT_COUNTS[pos]
            result_dict[_ResultProcessor.COLUMNS[pos]] = score

        return result_dict


class _DrumProcessor(_Processor, metaclass=_Singleton):
    X_ANCHOR = 95
    Y_ANCHOR = 85

    IMG_ROW = 65
    IMG_COL = 65

    def __init__(self):
        _Processor.__init__(self)
        self._model = load_model(DRUM_IMG_MODEL_PATH)

    def process(self, pic_path):
        dp = _DrumProcessor

        img = imread(pic_path)
        cropped = img[dp.X_ANCHOR:dp.X_ANCHOR + dp.IMG_ROW, dp.Y_ANCHOR:dp.Y_ANCHOR + dp.IMG_COL]
        cropped = rgb2grey(cropped)
        x_train = [cropped]
        x_train = np.asarray(x_train)
        x_train = x_train.reshape(x_train.shape[0], dp.IMG_ROW, dp.IMG_COL, 1)
        x = self._model.predict_classes(x_train, verbose=0)[0]

        return True if x == 1 else False


def get_play_start_time(capture_dir_path):
    files = glob(posixpath.join(capture_dir_path, '*'))

    for pic_path in sorted(files):
        is_drum = _DrumProcessor().process(pic_path)
        if is_drum:
            res = re.search('(\d){4}-(\d)+.(\d)+.png', pic_path)
            filename = res.group(0)
            timestamp = float(filename[5:-4])
            return timestamp

    raise Exception('unknown drum detected')


def read_result_board_info(capture_dir_path):
    files = glob(posixpath.join(capture_dir_path, '*'))

    for pic_path in reversed(sorted(files)):
        result = _ResultProcessor().process(pic_path)
        if result is not None:
            return result

    raise RuntimeError('unknown result info detected')


def read_score_board_info(capture_dir_path, song_id, timestamp_calibrate=True):
    file_paths = glob(posixpath.join(capture_dir_path, '*'))
    files = [re.search('(\d){4}-(\d)+.(\d)+.png', file_path).group(0) for file_path in file_paths]

    play_start_time = get_play_start_time(capture_dir_path)
    play_end_time = play_start_time + SONG_LENGTH_DICT[song_id]

    play_start_frame = -1
    play_end_frame = -1

    for id_, filename in enumerate(sorted(files)):
        frame_time = float(filename[5: -4])
        if frame_time <= play_start_time:
            play_start_frame = id_
        if frame_time <= play_end_time:
            play_end_frame = id_

    timestamps = []
    for i_ in range(play_start_frame, play_end_frame):
        timestamps.append(float(files[i_][5: -4]))

    img_scores = [0]

    for pic_path in sorted(file_paths)[play_start_frame: play_end_frame]:
        score = _ScoreProcessor().process(pic_path)

        if score is None or score < img_scores[-1]:
            score = img_scores[-1]
            # raise RuntimeError('unknown score info detected')
        img_scores.append(score)

    del img_scores[0]
    if timestamp_calibrate:
        timestamps = [timestamp - play_start_time for timestamp in timestamps]

    return timestamps, img_scores
