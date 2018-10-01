from .config import *
from .tools.score import *
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import posixpath
from keras.models import load_model

from skimage.io import imshow, imsave, imread
from skimage.transform import resize
from skimage.color import rgb2grey

__all__ = ['read_result_board_info']

X_ANCHOR = [275, 258, 279, 300, 258, 279]
Y_ANCHOR = [279, 438, 438, 438, 561, 561]

IMG_ROWS = [20, 15, 15, 15, 15, 15]
IMG_COLS = [15, 10, 10, 10, 10, 10]

TARGET_IMG_ROWS = 15
TARGET_IMG_COLS = 10
DIGIT_COUNTS = [6, 4, 4, 4, 4, 4]

COLUMNS = ['score', 'good', 'ok', 'bad', 'max_combo', 'drumroll']


class ResultProcessor(object):

    def __init__(self):
        self._model = load_model(MNIST_MODEL_PATH)

    def process(self, pic_path):
        img = imread(pic_path)
        imshow(img)
        plt.show()
        all_digits = []
        for pos in range(len(DIGIT_COUNTS)):
            digits = []
            for digit in range(DIGIT_COUNTS[pos]):
                cropped = img[X_ANCHOR[pos]:X_ANCHOR[pos] + IMG_ROWS[pos],
                              Y_ANCHOR[pos] + digit * IMG_COLS[pos]:Y_ANCHOR[pos] + (digit + 1) * IMG_COLS[pos]]
                cropped = resize(cropped, (TARGET_IMG_ROWS, TARGET_IMG_COLS),
                                 mode='constant', preserve_range=False)
                imshow(cropped)
                plt.show()
                cropped = rgb2grey(cropped)
                digits.append(cropped)

            all_digits.extend(digits)

        all_digits = np.asarray(all_digits)
        all_digits = all_digits.reshape(all_digits.shape[0], TARGET_IMG_ROWS, TARGET_IMG_COLS, 1)
        all_scores = self._model.predict_classes(all_digits)

        processed_result = self._process_scores(all_scores)
        if processed_result:
            return self._result
        else:
            return None

    def _process_scores(self, all_scores):
        result_dict = {}
        front = 0
        for pos in range(len(DIGIT_COUNTS)):
            score = get_processing_score(all_scores[front: front + DIGIT_COUNTS[pos]])
            if score is None:
                return False
            front += DIGIT_COUNTS[pos]

            result_dict[COLUMNS[pos]] = score

        self._result = result_dict
        return True

    @property
    def result(self):
        return self._result


proc = ResultProcessor()


def read_result_board_info(capture_dir_path):
    files = glob(posixpath.join(capture_dir_path, '*'))

    for pic_path in reversed(sorted(files)):
        result = proc.process(pic_path)
        if result is not None:
            return result

    raise Exception('unknown info detected')
