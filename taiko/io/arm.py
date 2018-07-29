from ..config import *

import os

import pandas as pd
import numpy as np
from scipy.stats import mode

__all__ = ['load_arm_df']


class _ArmData(object):

    def __init__(self, arm_csv_path):
        self._arm_df = None

        self.__load_arm_csv(arm_csv_path)

    def __load_arm_csv(self, arm_csv_path):
        """
        Load CSV files which contain information about arms.

        :param arm_csv_path: the directory path of arm CSV stored
        :return: merged dataframe inside the directory
        """

        # trace the directory and read all of them
        files = next(os.walk(arm_csv_path))[2]
        script_df = [
            pd.read_csv(os.path.join(arm_csv_path, filename), dtype={
                'timestamp': np.float64
            }) for filename in files
        ]

        # merge them into a dataframe
        merged_df = pd.DataFrame(pd.concat(script_df, ignore_index=True))
        merged_df.drop('key', axis=1, inplace=True)

        self._arm_df = merged_df

    @property
    def arm_df(self):
        return self._arm_df


def load_arm_df(handedness):
    if handedness == 0:
        return _ArmData(RIGHT_PATH).arm_df
    elif handedness == 1:
        return _ArmData(LEFT_PATH).arm_df

    raise ValueError('handedness must be 0 or 1.')