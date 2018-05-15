import pandas as pd
import numpy as np
import tkconfig
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import time
import sys
import math

from scipy.stats import mode
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime, timedelta
from collections import deque
from tqdm import tqdm


class Model(object):
    def __init__(self, verbose=0):
        self._verbose = verbose
        self._left_df = pd.DataFrame()
        self._right_df = pd.DataFrame()
        self.__setup()

    def __setup(self):
        self._left_df = self.__load_arm_csv(tkconfig.LEFT_PATH)
        self._right_df = self.__load_arm_csv(tkconfig.RIGHT_PATH)
        if self._verbose > 0:
            logging.info("load arm CSV.")

    @staticmethod
    def __load_arm_csv(arm_csv_path):
        files = next(os.walk(arm_csv_path))[2]
        script_df = [
            pd.read_csv(arm_csv_path + filename, dtype={
                'timestamp': np.float64
            }) for filename in files
        ]
        merged_df = pd.DataFrame(pd.concat(script_df, ignore_index=True))
        merged_df.drop('key', axis=1, inplace=True)
        return merged_df

    @property
    def left_df(self):
        return self._left_df

    @property
    def right_df(self):
        return self._right_df


model = Model(verbose=1)