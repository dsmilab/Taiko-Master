from taiko.tools.config import *
from .tools.score import *
from .tools.singleton import *

import numpy as np

import matplotlib.pyplot as plt
from abc import abstractmethod
from glob import glob
import posixpath
from keras.models import load_model
import re

from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from skimage.color import rgb2grey

__all__ = ['read_result_board_info',
           'read_score_board_info',
           'get_play_start_time']


class _Processor(object):

    def __init__(self):
        pass

    @abstractmethod
    def process(self, pic_path):
        raise NotImplementedError("Please Implement this method")


class _ScoreProcessor(_Processor, metaclass=_Singleton):
    X_ANCHOR = 178
    Y_ANCHOR = 40

    IMG_ROW = 15
    IMG_COL = 10

    DIGIT_COUNT = 6

    def __init__(self):
        super(_ScoreProcessor, self).__init__()
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
    X_ANCHOR = [275, 258, 279, 300, 258, 279]
    Y_ANCHOR = [279, 438, 438, 438, 561, 561]

    IMG_ROWS = [20, 15, 15, 15, 15, 15]
    IMG_COLS = [15, 10, 10, 10, 10, 10]

    DIGIT_COUNTS = [6, 4, 4, 4, 4, 4]

    TARGET_IMG_ROWS = 15
    TARGET_IMG_COLS = 10

    def __init__(self):
        super(_ResultProcessor, self).__init__()
        self._model = load_model(MNIST_MODEL_PATH)

    def process(self, pic_path):
        rp = _ResultProcessor
        try:
            img = imread(pic_path)
        except ValueError:
            return None

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
            result_dict[RESULT_BOARD_INFO_COLUMNS[pos]] = score

        return result_dict


class _SpiritProcessor(_Processor, metaclass=_Singleton):
    X_ANCHOR = 65
    Y_ANCHOR = 567

    IMG_ROW = 45
    IMG_COL = 45

    def __init__(self):
        super(_SpiritProcessor, self).__init__()
        self._model = load_model(SPIRIT_IMG_MODEL_PATH)

    def process(self, pic_path):
        sp = _SpiritProcessor

        try:
            img = imread(pic_path)
        except ValueError:
            return True
        cropped = img[sp.X_ANCHOR:sp.X_ANCHOR + sp.IMG_ROW, sp.Y_ANCHOR:sp.Y_ANCHOR + sp.IMG_COL]
        cropped = rgb2grey(cropped)
        x_train = [cropped]
        x_train = np.asarray(x_train)
        x_train = x_train.reshape(x_train.shape[0], sp.IMG_ROW, sp.IMG_COL, 1)
        x = self._model.predict_classes(x_train, verbose=0)[0]

        return True if x == 1 else False


def get_play_start_time(capture_dir_path, song_id):
    files = glob(posixpath.join(capture_dir_path, '*'))

    # handle sync case between date 2018-09-24 and date 2018-09-29, inclusively.
    res = re.search('\\d{4}_\\d{2}_\\d{2}_\\d{2}_\\d{2}_\\d{2}', capture_dir_path)
    date = res.group(0)[:10]
    sync_offset = 0
    if date == '2018_09_21':
        sync_offset = -1.5
    elif date == '2018_09_25':
        sync_offset = -2.3
    elif date == '2018_09_26':
        sync_offset = -0.8
    elif date == '2018_09_27':
        sync_offset = -1.1
    elif date == '2018_09_28':
        sync_offset = -1.4
    elif date == '2018_09_29':
        sync_offset = -1.8
    elif date == '2018_10_01':
        sync_offset = -0.2

    for pic_path in reversed(sorted(files)):
        is_spirit = _SpiritProcessor().process(pic_path)
        if not is_spirit:
            res = re.search('(\\d){4}-(\\d)+.(\\d)+.png', pic_path)
            filename = res.group(0)
            timestamp = float(filename[5:-4]) + sync_offset
            return timestamp - FIRST_HIT_ALIGN_DICT[song_id] - INTRO_DUMMY_TIME_LENGTH

    raise Exception('unknown spirit detected')


def read_result_board_info(capture_dir_path):
    files = glob(posixpath.join(capture_dir_path, '*.png'))

    for pic_path in reversed(sorted(files)):
        result = _ResultProcessor().process(pic_path)
        if result is not None:
            return result

    raise RuntimeError('unknown result info detected')


def read_score_board_info(capture_dir_path, song_id, timestamp_calibrate=True, raise_exception=False):
    file_paths = glob(posixpath.join(capture_dir_path, '*.png'))
    files = sorted([re.search('(\\d){4}-(\\d)+.(\\d)+.png', file_path).group(0) for file_path in file_paths])

    play_start_time = get_play_start_time(capture_dir_path, song_id)
    play_end_time = play_start_time + FIRST_HIT_ALIGN_DICT[song_id]

    play_start_frame = -1
    play_end_frame = -1

    for id_, filename in enumerate(files):
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

        if score is None or score < img_scores[-1] or score > img_scores[-1] + 25000:
            if raise_exception:
                raise RuntimeError('unknown score info detected')
            else:
                score = img_scores[-1]
                
        img_scores.append(score)

    del img_scores[0]
    if timestamp_calibrate:
        timestamps = [timestamp - play_start_time for timestamp in timestamps]

    return timestamps, img_scores
