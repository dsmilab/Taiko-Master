from ..config import *
from ..io.record import *

import os
import numpy as np
from datetime import datetime

from keras.models import load_model

from skimage.io import imread
from skimage.color import rgb2grey

IMG_ROWS, IMG_COLS = 15, 10
X_ANCHOR = 180
Y_ANCHOR = 40
DIGIT_COUNTS = 6

__all__ = ['get_scores_with_timestamps']


class _ScoreBoard(object):

    def __init__(self, who_id, song_id, order_id, zero_adjust=True):

        self._model = load_model(MNIST_MODEL_PATH)

        self._score_start_frame = None
        self._score_end_frame = None
        self._timestamps = []
        row = get_record(who_id, song_id, order_id)

        d = datetime.strptime(row['start_time'], '%m/%d/%Y %H:%M:%S')
        directory = d.strftime('bb_capture.capture_%Y_%m_%d_%H_%M_%S')

        first_hit_time = float(row['first_hit_time'])
        score_start_time = first_hit_time - 1
        score_end_time = score_start_time + row['song_length'] + 3

        workspace = BB_CAPTURE_PATH + directory + '/'
        files = sorted(next(os.walk(workspace))[2])

        self._crop_frames(files, score_start_time, score_end_time)
        self._process_images(workspace, files)

        if zero_adjust:
            self._timestamps = [timestamp - first_hit_time for timestamp in self._timestamps]

    def _crop_frames(self, files, score_start_time, score_end_time):
        # retrieve first and last
        self._score_start_frame = -1
        self._score_end_frame = -1
        for id_, filename in enumerate(files):
            frame_time = float(filename[5: -4])
            if frame_time <= score_start_time:
                self._score_start_frame = id_
            if frame_time <= score_end_time:
                self._score_end_frame = id_

        self._timestamps = []
        for i in range(self._score_start_frame, self._score_end_frame):
            self._timestamps.append(float(files[i][5: -4]))

    def _process_images(self, workspace, files):
        all_digits = []

        for filename in files[self._score_start_frame: self._score_end_frame]:
            img = imread(workspace + filename)
            digits = []
            for digit in range(DIGIT_COUNTS):
                cropped = img[X_ANCHOR:X_ANCHOR + IMG_ROWS, Y_ANCHOR +
                              IMG_COLS * digit:Y_ANCHOR + IMG_COLS * (digit + 1)]
                cropped = rgb2grey(cropped)
                digits.append(cropped)

            all_digits.extend(digits)

        all_digits = np.asarray(all_digits)
        all_digits = all_digits.reshape(all_digits.shape[0], IMG_ROWS, IMG_COLS, 1)
        all_scores = self._model.predict_classes(all_digits)

        return self._process_scores(all_scores)

    def _process_scores(self, all_scores):
        img_scores = [0]
        for k in range(0, len(all_scores), DIGIT_COUNTS):
            score = 0

            broken = False
            leading_space = True
            for d in all_scores[k: k + DIGIT_COUNTS]:
                if d == 10:
                    d = 0
                    if not leading_space:
                        broken = True
                else:
                    leading_space = False
                score = score * 10 + d

            if score < img_scores[-1] or broken:
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


def get_scores_with_timestamps(who_id, song_id, order_id, zero_adjust=True):
    scoreboard = _ScoreBoard(who_id, song_id, order_id, zero_adjust)
    return scoreboard.timestamps, scoreboard.timestamps
