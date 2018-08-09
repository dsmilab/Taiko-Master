from ..config import *
from ..io.record import *
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime

from keras.models import load_model

from skimage.io import imshow, imread
from skimage.transform import resize
from skimage.color import rgb2grey

IMG_ROWS = [20, 15, 15, 15, 15, 15]
IMG_COLS = [15, 10, 10, 10, 10, 10]
TARGET_IMG_ROWS = 15
TARGET_IMG_COLS = 10
X_ANCHOR = [280, 262, 283, 304, 262, 283]
Y_ANCHOR = [278, 437, 437, 437, 559, 559]
DIGIT_COUNTS = [6, 4, 4, 4, 4, 4]

__all__ = ['result_board']


class _ResultBoard(object):
    """

    """

    def __init__(self, who_id, song_id, order_id):

        self._model = load_model(MNIST_MODEL_PATH)

        self._score_start_frame = None
        self._score_end_frame = None
        self._timestamps = []
        row = get_record(who_id, song_id, order_id)

        d = datetime.strptime(row['start_time'], '%m/%d/%Y %H:%M:%S')
        directory = d.strftime('bb_capture.capture_%Y_%m_%d_%H_%M_%S')

        workspace = os.path.join(BB_CAPTURE_PATH, directory)
        files = sorted(next(os.walk(workspace))[2])

        # self._crop_frames(files)
        self._process_images(workspace, files)
        print(self._img_scores)

    def _process_images(self, workspace, files):

        for filename in reversed(files):
            img = imread(os.path.join(workspace, filename))
            imshow(img)
            plt.show()
            all_ok = True
            for pos in range(len(DIGIT_COUNTS)):
                digits = []
                for digit in range(DIGIT_COUNTS[pos]):
                    cropped = img[X_ANCHOR[pos]:X_ANCHOR[pos] + IMG_ROWS[pos],
                                  Y_ANCHOR[pos] + digit * IMG_COLS[pos]:Y_ANCHOR[pos] + (digit + 1) * IMG_COLS[pos]]
                    cropped = rgb2grey(cropped)
                    cropped = resize(cropped, (TARGET_IMG_ROWS, TARGET_IMG_COLS),
                                     mode='constant', preserve_range=True)
                    imshow(cropped)
                    plt.show()
                    digits.append(cropped)

                all_digits = []
                all_digits.extend(digits)

                all_digits = np.asarray(all_digits)
                all_digits = all_digits.reshape(all_digits.shape[0], TARGET_IMG_ROWS, TARGET_IMG_COLS, 1)
                all_scores = self._model.predict_classes(all_digits)

                processed_result = self._process_scores(all_scores, DIGIT_COUNTS[pos])
                all_ok = all_ok & processed_result

            if all_ok:
                return

    def _process_scores(self, all_scores, digit_counts):
        img_scores = [0]

        for k in range(0, len(all_scores), digit_counts):
            score = 0

            broken = False
            leading_space = True
            for d in all_scores[k: k + digit_counts]:
                if d == 10:
                    d = 0
                    if not leading_space:
                        broken = True
                else:
                    leading_space = False
                score = score * 10 + d

            if score < img_scores[-1] or broken:
                return False

            img_scores.append(score)

        del img_scores[0]
        self._img_scores = img_scores
        return True


def result_board(who_id, song_id, order_id):
    return _ResultBoard(who_id, song_id, order_id)
