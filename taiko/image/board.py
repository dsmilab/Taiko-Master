from ..config import *
from ..tools.score import *
from ..io.record import *
from abc import abstractmethod
import os
import numpy as np

from datetime import datetime

from keras.models import load_model

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2grey

__all__ = ['get_scores_with_timestamps', 'get_result_info']

IMG_ROWS = [15, 20, 15, 15, 15, 15, 15]
IMG_COLS = [10, 15, 10, 10, 10, 10, 10]
TARGET_IMG_ROWS = 15
TARGET_IMG_COLS = 10
X_ANCHOR = [180, 280, 262, 283, 304, 262, 283]
Y_ANCHOR = [40, 278, 437, 437, 437, 559, 559]
DIGIT_COUNTS = [6, 6, 4, 4, 4, 4, 4]


class _Board(object):

    def __init__(self, row):
        self._model = load_model(MNIST_MODEL_PATH)

        d = datetime.strptime(row['start_time'], '%m/%d/%Y %H:%M:%S')
        directory = d.strftime('bb_capture.capture_%Y_%m_%d_%H_%M_%S')

        self._workspace = posixpath.join(BB_CAPTURE_PATH, directory)
        self._files = sorted(next(os.walk(self._workspace))[2])

    @abstractmethod
    def _process_images(self):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def _process_scores(self, all_scores):
        raise NotImplementedError("Please Implement this method")


class _ResultBoard(_Board):

    def __init__(self, row):
        super(_ResultBoard, self).__init__(row)
        self._img_screenshot = []
        self._process_images()

    def _process_images(self):
        for filename in reversed(self._files):
            img = imread(posixpath.join(self._workspace, filename))

            all_digits = []
            for pos in range(1, len(DIGIT_COUNTS)):
                digits = []
                for digit in range(DIGIT_COUNTS[pos]):
                    cropped = img[X_ANCHOR[pos]:X_ANCHOR[pos] + IMG_ROWS[pos],
                                  Y_ANCHOR[pos] + digit * IMG_COLS[pos]:Y_ANCHOR[pos] + (digit + 1) * IMG_COLS[pos]]
                    cropped = resize(cropped, (TARGET_IMG_ROWS, TARGET_IMG_COLS),
                                     mode='constant', preserve_range=False)
                    cropped = rgb2grey(cropped)
                    digits.append(cropped)

                all_digits.extend(digits)

            all_digits = np.asarray(all_digits)
            all_digits = all_digits.reshape(all_digits.shape[0], TARGET_IMG_ROWS, TARGET_IMG_COLS, 1)
            all_scores = self._model.predict_classes(all_digits)

            processed_result = self._process_scores(all_scores)
            if processed_result:
                return

    def _process_scores(self, all_scores):
        img_screenshot = []
        front = 0
        for pos in range(1, len(DIGIT_COUNTS)):
            score = get_processing_score(all_scores[front: front + DIGIT_COUNTS[pos]])
            if score is None:
                return False
            front += DIGIT_COUNTS[pos]

            img_screenshot.append(score)

        self._img_screenshot = img_screenshot
        return True

    @property
    def img_screenshot(self):
        return self._img_screenshot


class _ScoreBoard(_Board):

    def __init__(self, row, timestamp_zero_adjust):
        super(_ScoreBoard, self).__init__(row)
        self._timestamps = []
        self._img_scores = []

        first_hit_time = float(row['first_hit_time'])
        score_start_time = first_hit_time - 1
        score_end_time = score_start_time + row['song_length'] + 3

        self._crop_frames(score_start_time, score_end_time)
        self._process_images()

        if timestamp_zero_adjust:
            self._timestamps = [timestamp - first_hit_time for timestamp in self._timestamps]

    def _crop_frames(self, score_start_time, score_end_time):
        # retrieve first and last
        self._score_start_frame = -1
        self._score_end_frame = -1

        for id_, filename in enumerate(self._files):
            frame_time = float(filename[5: -4])
            if frame_time <= score_start_time:
                self._score_start_frame = id_
            if frame_time <= score_end_time:
                self._score_end_frame = id_

        for i in range(self._score_start_frame, self._score_end_frame):
            self._timestamps.append(float(self._files[i][5: -4]))

    def _process_images(self):
        all_digits = []

        for filename in self._files[self._score_start_frame: self._score_end_frame]:
            img = imread(posixpath.join(self._workspace, filename))
            digits = []
            for digit in range(DIGIT_COUNTS[0]):
                cropped = img[X_ANCHOR[0]: X_ANCHOR[0] + IMG_ROWS[0],
                              Y_ANCHOR[0] + IMG_COLS[0] * digit: Y_ANCHOR[0] + IMG_COLS[0] * (digit + 1)]
                cropped = rgb2grey(cropped)
                digits.append(cropped)
            all_digits.extend(digits)

        all_digits = np.asarray(all_digits)
        all_digits = all_digits.reshape(all_digits.shape[0], TARGET_IMG_ROWS, TARGET_IMG_COLS, 1)
        all_scores = self._model.predict_classes(all_digits)
        self._process_scores(all_scores)

    def _process_scores(self, all_scores):
        img_scores = [0]

        for k in range(0, len(all_scores), DIGIT_COUNTS[0]):
            score = get_processing_score(all_scores[k: k + DIGIT_COUNTS[0]])
            if score is None or score < img_scores[-1]:
                score = img_scores[-1]

            img_scores.append(score)

        del img_scores[0]
        self._img_scores = img_scores

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def img_scores(self):
        return self._img_scores


def get_scores_with_timestamps(who_id, song_id, order_id, timestamp_zero_adjust=True):
    """
    Get score and timestamp info.

    :param who_id: # of drummer
    :param song_id: # of song
    :param order_id: # of performance repetitively
    :param timestamp_zero_adjust: shift all timestamps to leftmost as 0
    :return: list of scores, list of corresponding timestamps
    """

    row = get_record(who_id, song_id, order_id)
    scoreboard = _ScoreBoard(row, timestamp_zero_adjust)
    return scoreboard.img_scores, scoreboard.timestamps


def get_result_info(who_id, song_id, order_id):
    row = get_record(who_id, song_id, order_id)
    return _ResultBoard(row).img_screenshot
