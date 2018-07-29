from ..config import *
from .arm import *

import pandas as pd
import numpy as np

__all__ = ['load']

class _Sensor(object):
    """
    Handle output from wearable devices.

    :protected attributes:
        verbose: default 0, otherwise will output loading message
        left_df: dataframe about left arm
        right_df: dataframe about right arm
    """

    TAILED_ADDITIONAL_TIME = 30

    def __init__(self, verbose=0, is_zero_adjust=True, left_path=LEFT_PATH, right_path=RIGHT_PATH):
        self._left_df, self._left_modes = None, None
        self._right_df, self._right_modes = None, None

        left_arm = load_arm_data(left_path, is_zero_adjust)
        self._left_df = left_arm.arm_df
        self._left_modes = left_arm.modes

        right_arm = load_arm_data(right_path, is_zero_adjust)
        self._right_df = right_arm.arm_df
        self._right_modes = right_arm.modes

        if verbose > 0:
            logging.info("load arm CSV.")

def load(verbose=0):
    return _Sensor(verbose=verbose)
