from ..config import *

import pandas as pd
import numpy as np

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import math

__all__ = ['get_features']


class _Primitive(object):
    """
    Transform window information into primitive and brings useful features.

    :protected attributes:
        rms_df: important dataframe contains three axes acceleration and gyroscope
        features: engineered features from cropped primitive
    """

    def __init__(self, window, tempe_window):
        self._rms_df = self.__convert_to_df(window)
        self._tempe_rms_df = self.__convert_to_df(tempe_window)
        self._features = self.__retrieve_features()

    @staticmethod
    def __convert_to_df(window):
        mat = []
        while len(window) > 0:
            mat.append(window.popleft())
        df = pd.DataFrame(mat, columns=SENSOR_COLUMNS)
        rms_df = df[RMS_COLS[2:]].copy()

        # acceleration movement intensity (AMI)
        rms_df[RMS_COLS[0]] = (rms_df[RMS_COLS[2]] ** 2 +
                               rms_df[RMS_COLS[3]] ** 2 +
                               rms_df[RMS_COLS[4]] ** 2).apply(math.sqrt)

        # gyroscope movement intensity (GMI)
        rms_df[RMS_COLS[1]] = (rms_df[RMS_COLS[5]] ** 2 +
                               rms_df[RMS_COLS[6]] ** 2 +
                               rms_df[RMS_COLS[7]] ** 2).apply(math.sqrt)

        return rms_df

    def __retrieve_features(self):

        ai_s = [self.__get_ai(col) for col in RMS_COLS]
        vi_s = [self.__get_vi(col, ai) for col, ai in zip(RMS_COLS, ai_s)]
        mmi_s = [self.__get_mdmi(col)for col in RMS_COLS]

        asma = self.__get_sma(RMS_COLS[2:5])
        gsma = self.__get_sma(RMS_COLS[5:8])

        aae = self.__get_ae(RMS_COLS[0])
        are = self.__get_ae(RMS_COLS[1])

        # standard deviation intensity
        sdi_s = [math.sqrt(sdi) for sdi in vi_s]

        iqr_s = [self.__get_iqr(col) for col in RMS_COLS]
        fr_s = [self.__get_full_range(col) for col in RMS_COLS]

        freqx_s_s = np.array([self.__get_fft_coef(col, 3) for col in RMS_COLS])
        freqx_s = freqx_s_s.reshape(len(RMS_COLS) * 3).tolist()

        # ass = self.__get_ss(RMS_COLS[0], aai, avi)
        # gss = self.__get_ss(RMS_COLS[1], gai, gvi)
        #
        # aks = self.__get_ks(RMS_COLS[0], aai, avi)
        # gks = self.__get_ks(RMS_COLS[1], gai, gvi)

        median_cross_s = [self.__get_median_cross(col, mmi) for col, mmi in zip(RMS_COLS, mmi_s)]
        mean_cross_s = [self.__get_mean_cross(col, ai) for col, ai in zip(RMS_COLS, ai_s)]
        zero_cross_s = [self.__get_zero_cross(col) for col in RMS_COLS[2:]]

        dtw_s = [self.__get_dtw_distance(col) for col in RMS_COLS]

        a_xy_corr = self.__get_corr(RMS_COLS[2], RMS_COLS[3])
        a_yz_corr = self.__get_corr(RMS_COLS[3], RMS_COLS[4])
        a_zx_corr = self.__get_corr(RMS_COLS[4], RMS_COLS[2])

        g_xy_corr = self.__get_corr(RMS_COLS[5], RMS_COLS[6])
        g_yz_corr = self.__get_corr(RMS_COLS[6], RMS_COLS[7])
        g_zx_corr = self.__get_corr(RMS_COLS[7], RMS_COLS[5])

        return ai_s +\
            vi_s +\
            mmi_s +\
            [asma, gsma, aae, are] +\
            sdi_s +\
            iqr_s +\
            fr_s +\
            freqx_s +\
            median_cross_s +\
            mean_cross_s +\
            zero_cross_s +\
            dtw_s +\
            [a_xy_corr, a_yz_corr, a_zx_corr,
                g_xy_corr, g_yz_corr, g_zx_corr]

    def __get_ai(self, col):
        #  average intensity (AI)
        ai = self._rms_df[col].sum() / len(self._rms_df)
        return ai

    def __get_mdmi(self, col):
        # median of AMI
        mdmi = self._rms_df[col].median()
        return mdmi

    def __get_vi(self, col, ai):
        # variance intensity (VI)
        vi = 0
        for i in range(len(self._rms_df)):
            row = self._rms_df.iloc[i]
            mit = float(row[col])
            vi += (mit - ai) ** 2
        vi /= len(self._rms_df)
        return vi

    def __get_sma(self, cols):
        # normalized signal magnitude area (SMA)
        sma = (self._rms_df[cols[0]].apply(abs).sum() +
               self._rms_df[cols[1]].apply(abs).sum() +
               self._rms_df[cols[2]].apply(abs).sum()) / len(self._rms_df)
        return sma

    def __get_ae(self, col):
        # averaged energy (AE)
        aae = do_fft(self._rms_df[col]) / len(self._rms_df)
        return aae

    def __get_ss(self, col, ai, vi):
        # skewness which is the degree of asymmetry of the sensor signal distribution (SS)
        ss_child = 0
        for i in range(len(self._rms_df)):
            row = self._rms_df.iloc[i]
            mit = float(row[col])
            ss_child += (mit - ai) ** 3
        ss_child /= len(self._rms_df)
        ss = ss_child / math.pow(vi, 3.0 / 2.0)
        return ss

    def __get_ks(self, col, ai, vi):
        # kurtosis which is the degree of peakedness of the sensor signal distribution (AKS)
        ks_child = 0
        for i in range(len(self._rms_df)):
            row = self._rms_df.iloc[i]
            mit = float(row[col])
            ks_child += (mit - ai) ** 4
        ks_child /= len(self._rms_df)
        ks = ks_child / (vi ** 3) - 3
        return ks

    def __get_full_range(self, col):
        full_range = max(self._rms_df[col]) - min(self._rms_df[col])
        return full_range

    def __get_iqr(self, col):
        # interquartile range (IQR)
        iqr = self._rms_df[col].quantile(0.75) - self._rms_df[col].quantile(0.25)
        return iqr

    def __get_zero_cross(self, col):
        # the total number of times the signal changes form positive to negative or negative or vice versa
        zero_cross = 0
        for i in range(1, len(self._rms_df)):
            prev_row = self._rms_df.iloc[i - 1]
            now_row = self._rms_df.iloc[i]
            if (prev_row[col] < 0) and (now_row[col] >= 0):
                zero_cross += 1
            elif (prev_row[col] > 0) and (now_row[col] <= 0):
                zero_cross += 1
        zero_cross /= len(self._rms_df)
        return zero_cross

    def __get_median_cross(self, col, mdmi):
        median_cross = 0
        for i in range(1, len(self._rms_df)):
            prev_row = self._rms_df.iloc[i - 1]
            now_row = self._rms_df.iloc[i]
            if (prev_row[col] < mdmi) and (now_row[col] >= mdmi):
                median_cross += 1
            elif (prev_row[col] > mdmi) and (now_row[col] <= mdmi):
                median_cross += 1
        median_cross /= len(self._rms_df)
        return median_cross

    def __get_mean_cross(self, col, ai):
        # the total number of times the acceleration signal changes form below average to above average or vice versa
        mean_cross = 0
        for i in range(1, len(self._rms_df)):
            prev_row = self._rms_df.iloc[i - 1]
            now_row = self._rms_df.iloc[i]
            if (prev_row[col] > ai) and (now_row[col] <= ai):
                mean_cross += 1
            elif (prev_row[col] < ai) and (now_row[col] >= ai):
                mean_cross += 1
        mean_cross /= len(self._rms_df)
        return mean_cross

    def __get_dtw_distance(self, col):
        x = np.array(self._rms_df[col])
        y = np.array(self._tempe_rms_df[col])
        distance, path = fastdtw(x, y, dist=euclidean)
        return distance

    def __get_fft_coef(self, col, max_n_counts):
        freqx = get_fft_coef(self._rms_df[col])
        return freqx[-max_n_counts:]

    def __get_corr(self, col1, col2):
        # correlation between two columns
        corr = self._rms_df[col1].corr(self._rms_df[col2])
        return corr

    def __get_evas(self):
        # eigenvalues of dominant directions (EVA)
        w = np.linalg.eig(self._rms_df[['imu_ax', 'imu_ay', 'imu_az']].corr().values)
        evas = w[np.argpartition(w, -2)[-2:]]
        return evas

    @property
    def features(self):
        return self._features


def do_fft(data):
    """
    Implement fast fourier transformation.

    :param data: a numeric series in time domain
    :return: the energy of this data in frequency domain
    """

    freqx = np.fft.fft(data) / math.sqrt(len(data))
    energy = np.sum(np.abs(freqx) ** 2)
    return energy


def get_fft_coef(data):
    freqx = np.fft.fft(data) / math.sqrt(len(data))
    freqx = sorted(np.abs(freqx))

    return freqx


def get_features(window, tempe_window):
    """
    Get all engineered features from the window.

    :param window: the range of sensor data we care about
    :param tempe_window: the range of sensor data we care about
    :return: engineered features
    """

    return _Primitive(window, tempe_window).features
