import apps.tkconfig as tkconfig

import os
import numpy as np
from datetime import datetime

from keras.models import load_model

from skimage.io import imshow, imread
from skimage.color import rgb2grey

IMG_ROWS, IMG_COLS = 15, 10
X_ANCHOR = 180
Y_ANCHOR = 40
DIGIT_COUNTS = 6


class ScoreBoard(object):

    def __init__(self, drummer_df):
        self._drummer_df = drummer_df
        self._model = load_model('model/mnist_model.h5')

    def get_scores(self, who_id, song_id, performance_order):
        row = self._drummer_df[(self._drummer_df['drummer_id'] == who_id) &
                               (self._drummer_df['song_id'] == song_id) &
                               (self._drummer_df['performance_order'] == performance_order)]
        row = row.iloc[0]

        d = datetime.strptime(row['start_time'], '%m/%d/%Y %H:%M:%S')
        directory = d.strftime('bb_capture.capture_%Y_%m_%d_%H_%M_%S')

        score_start_time = float(row['first_hit_time']) - 1
        score_end_time = score_start_time + row['song_length'] + 3

        workspace = tkconfig.BB_CAPTURE_PATH + directory + '/'
        files = next(os.walk(workspace))[2]
        files.sort()

        score_start_frame, score_end_frame = self._crop_frames(files, score_start_time, score_end_time)
        img_scores = self._process_images(workspace, files, score_start_frame, score_end_frame)

        return img_scores

    @staticmethod
    def _crop_frames(files, score_start_time, score_end_time):
        # retrieve first and last
        score_start_frame = -1
        score_end_frame = -1
        for id_, filename in enumerate(files):
            frame_time = float(filename[5: -4])
            if frame_time <= score_start_time:
                score_start_frame = id_
            if frame_time <= score_end_time:
                score_end_frame = id_

        return score_start_frame, score_end_frame

    def _process_images(self, workspace, files, score_start_frame, score_end_frame):
        all_digits = []

        for filename in files[score_start_frame: score_end_frame]:
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

    @staticmethod
    def _process_scores(all_scores):
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

        return img_scores
