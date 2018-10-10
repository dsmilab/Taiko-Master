import numpy as np
import math

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

__all__ = ['Motif']

RMS_COLS = ['a_rms', 'g_rms', 'imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']


class Motif(object):

    def __init__(self, df):
        rms_df = df[RMS_COLS[2:]].copy()
        # acceleration movement intensity (AMI)
        rms_df[RMS_COLS[0]] = (rms_df[RMS_COLS[2]] ** 2 +
                               rms_df[RMS_COLS[3]] ** 2 +
                               rms_df[RMS_COLS[4]] ** 2).apply(math.sqrt)

        # gyroscope movement intensity (GMI)

        rms_df[RMS_COLS[1]] = (rms_df[RMS_COLS[5]] ** 2 +
                               rms_df[RMS_COLS[6]] ** 2 +
                               rms_df[RMS_COLS[7]] ** 2).apply(math.sqrt)

        self._rms_df = rms_df

    def get_ai(self, col):
        #  average intensity (AI)
        ai = self._rms_df[col].sum() / len(self._rms_df)
        return ai

    def get_mdmi(self, col):
        # median of AMI
        mdmi = self._rms_df[col].median()
        return mdmi

    def get_vi(self, col, ai):
        # variance intensity (VI)
        vi = 0
        for i in range(len(self._rms_df)):
            row = self._rms_df.iloc[i]
            mit = float(row[col])
            vi += (mit - ai) ** 2
        vi /= len(self._rms_df)
        return vi

    def get_sma(self, cols):
        # normalized signal magnitude area (SMA)
        sma = (self._rms_df[cols[0]].apply(abs).sum() +
               self._rms_df[cols[1]].apply(abs).sum() +
               self._rms_df[cols[2]].apply(abs).sum()) / len(self._rms_df)
        return sma

    def __get_ae(self, col):
        # averaged energy (AE)
        aae = do_fft(self._rms_df[col]) / len(self._rms_df)
        return aae

    def get_ss(self, col, ai, vi):
        # skewness which is the degree of asymmetry of the sensor signal distribution (SS)
        ss_child = 0
        for i in range(len(self._rms_df)):
            row = self._rms_df.iloc[i]
            mit = float(row[col])
            ss_child += (mit - ai) ** 3
        ss_child /= len(self._rms_df)
        ss = ss_child / math.pow(vi, 3.0 / 2.0)
        return ss

    def get_ks(self, col, ai, vi):
        # kurtosis which is the degree of peakedness of the sensor signal distribution (AKS)
        ks_child = 0
        for i in range(len(self._rms_df)):
            row = self._rms_df.iloc[i]
            mit = float(row[col])
            ks_child += (mit - ai) ** 4
        ks_child /= len(self._rms_df)
        ks = ks_child / (vi ** 3) - 3
        return ks

    def get_full_range(self, col):
        full_range = max(self._rms_df[col]) - min(self._rms_df[col])
        return full_range

    def get_iqr(self, col):
        # interquartile range (IQR)
        iqr = self._rms_df[col].quantile(0.75) - self._rms_df[col].quantile(0.25)
        return iqr

    def get_zero_cross(self, col):
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

    def get_median_cross(self, col, mdmi):
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

    def get_mean_cross(self, col, ai):
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

    def get_fft_coef(self, col, max_n_counts):
        freqx = get_fft_coef(self._rms_df[col])
        return freqx[-max_n_counts:]

    def get_corr(self, col1, col2):
        # correlation between two columns
        corr = self._rms_df[col1].corr(self._rms_df[col2])
        return corr

    def get_evas(self):
        # eigenvalues of dominant directions (EVA)
        w = np.linalg.eig(self._rms_df[['imu_ax', 'imu_ay', 'imu_az']].corr().values)
        evas = w[np.argpartition(w, -2)[-2:]]
        return evas


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


def get_dtw_distance(rms_df1, rms_df2, col):
    x = np.array(rms_df1[col])
    y = np.array(rms_df2[col])
    distance, _ = fastdtw(x, y, dist=euclidean)
    return distance
