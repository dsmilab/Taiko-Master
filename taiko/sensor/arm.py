from ..config import *

import os

import pandas as pd
import numpy as np
from scipy.stats import mode

__all__ = ['load_arm_data']


class _ArmData(object):

    def __init__(self, arm_csv_path, is_zero_adjust=True):
        # declare
        self._arm_df = None
        self._modes = None

        # define
        self.__load_arm_csv(arm_csv_path)

        if is_zero_adjust:
            self.__set_modes_dict()
            self.__adjust_zero()
        print(self._arm_df)

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

    def __adjust_zero(self):
        """
        Implement zero adjust.

        :return: zero-adjusted dataframe
        """

        copy_df = self._arm_df.copy()
        # only considered attributes need zero adjust
        for col in ZERO_ADJ_COL:
            copy_df[col] = copy_df[col] - self._modes[col]

        self._arm_df = copy_df

    def __set_modes_dict(self):
        """
        Create mode dictionary of the play dataframe.

        :return: dictionary containing all attributes in {"attribute": "mode"} sense
        """

        modes = {}
        copy_df = self._arm_df.copy()
        for col in ZERO_ADJ_COL:
            mode_ = mode(copy_df[col])[0]
            modes[col] = mode_
        self._modes = modes

    @property
    def arm_df(self):
        return self._arm_df

    @property
    def modes(self):
        return self._modes

def load_arm_data(arm_csv_path, is_zero_adjust):
    return _ArmData(arm_csv_path, is_zero_adjust)