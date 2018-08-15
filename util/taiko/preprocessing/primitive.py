from ..config import *

import pandas as pd
import numpy as np
import math

__all__ = ['get_features']


class _Primitive(object):
    """
    Transform window information into primitive and brings useful features.

    :protected attributes:
        rms_df: important dataframe contains three axes acceleration and gyroscope
        features: engineered features from cropped primitive
    """

    def __init__(self, window):
        self._rms_df = None
        self.__convert_to_df(window)
        self._features = self.__retrieve_features()

    def __convert_to_df(self, window):
        mat = []
        while len(window) > 0:
            mat.append(window.popleft())
        df = pd.DataFrame(mat, columns=SENSOR_COLUMNS)
        rms_df = df[['timestamp', 'imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']].copy()

        # acceleration movement intensity (AMI)
        rms_df['a_rms'] = (rms_df['imu_ax'] ** 2 + rms_df['imu_ay'] ** 2 + rms_df['imu_az'] ** 2).apply(math.sqrt)

        # gyroscope movement intensity (GMI)
        rms_df['g_rms'] = (rms_df['imu_gx'] ** 2 + rms_df['imu_gy'] ** 2 + rms_df['imu_gz'] ** 2).apply(math.sqrt)

        self._rms_df = rms_df

    def __retrieve_features(self):
        aai = self.__get_aai()
        gai = self.__get_gai()

        mami = self.__get_mami()
        mgmi = self.__get_mgmi()

        avi = self.__get_avi(aai)
        gvi = self.__get_gvi(gai)

        asma = self.__get_asma()
        gsma = self.__get_gsma()

        aae = self.__get_aae()
        are = self.__get_are()

        # acceleration standard deviation intensity
        asdi = math.sqrt(avi)
        # gyroscope standard deviation intensity
        gsdi = math.sqrt(gvi)

        air = self.__get_air()
        gir = self.__get_gir()

        # ass = self.__get_ass(aai, avi)
        # gss = self.__get_gss(gai, gvi)
        #
        # aks = self.__get_aks(aai, avi)
        # gks = self.__get_gks(gai, gvi)

        a_zero_cross = self.__get_a_zero_cross(mami)
        g_zero_cross = self.__get_g_zero_cross(mgmi)

        a_mean_cross = self.__get_a_mean_cross(aai)
        g_mean_cross = self.__get_g_mean_cross(gai)

        a_xy_corr = self.__get_a_xy_corr()
        a_yz_corr = self.__get_a_yz_corr()
        a_zx_corr = self.__get_a_zx_corr()

        g_xy_corr = self.__get_g_xy_corr()
        g_yz_corr = self.__get_g_yz_corr()
        g_zx_corr = self.__get_g_zx_corr()

        return [aai,
                avi,
                asma,

                gai,
                gvi,

                aae,
                are,

                mami,
                mgmi,

                asdi,
                gsdi,

                air,
                gir,

                # ass,
                # gss,
                # aks,
                # gks,

                a_zero_cross,
                g_zero_cross,

                a_mean_cross,
                g_mean_cross,

                a_xy_corr,
                a_yz_corr,
                a_zx_corr,
                g_xy_corr,
                g_yz_corr,
                g_zx_corr]

    def __get_aai(self):
        # acceleration average intensity (AAI)
        aai = self._rms_df['a_rms'].sum() / len(self._rms_df)
        return aai

    def __get_gai(self):
        # gyroscope average intensity (GAI)
        gai = self._rms_df['g_rms'].sum() / len(self._rms_df)
        return gai

    def __get_mami(self):
        # median of AMI
        mami = self._rms_df['a_rms'].median()
        return mami

    def __get_mgmi(self):
        # median of GMI
        mgmi = self._rms_df['g_rms'].median()
        return mgmi

    def __get_avi(self, aai):
        # acceleration variance intensity (AVI)
        avi = 0
        for i in range(len(self._rms_df)):
            row = self._rms_df.iloc[i]
            mit = float(row['a_rms'])
            avi += (mit - aai) ** 2
        avi /= len(self._rms_df)
        return avi

    def __get_gvi(self, gai):
        # gyroscope variance intensity (GVI)
        gvi = 0
        for i in range(len(self._rms_df)):
            row = self._rms_df.iloc[i]
            mit = float(row['g_rms'])
            gvi += (mit - gai) ** 2
        gvi /= len(self._rms_df)
        return gvi

    def __get_asma(self):
        # acceleration normalized signal magnitude area (ASMA)
        asma = (self._rms_df['imu_ax'].apply(abs).sum() +
                self._rms_df['imu_ay'].apply(abs).sum() +
                self._rms_df['imu_az'].apply(abs).sum()) / len(self._rms_df)
        return asma

    def __get_gsma(self):
        # gyroscope normalized signal magnitude area (GSMA)
        gsma = (self._rms_df['imu_gx'].apply(abs).sum() +
                self._rms_df['imu_gy'].apply(abs).sum() +
                self._rms_df['imu_gz'].apply(abs).sum()) / len(self._rms_df)
        return gsma

    def __get_aae(self):
        # averaged acceleration energy (AAE)
        aae = do_fft(self._rms_df['a_rms']) / len(self._rms_df)
        return aae

    def __get_are(self):
        # averaged rotation energy (ARE)
        are = do_fft(self._rms_df['g_rms']) / len(self._rms_df)
        return are

    def __get_ass(self, aai, avi):
        # acceleration skewness which is the degree of asymmetry of the sensor signal distribution (ASS)
        ass_child = 0
        for i in range(len(self._rms_df)):
            row = self._rms_df.iloc[i]
            mit = float(row['a_rms'])
            ass_child += (mit - aai) ** 3
        ass_child /= len(self._rms_df)
        ass = ass_child / math.pow(avi, 3.0 / 2.0)
        return ass

    def __get_gss(self, gai, gvi):
        # gyroscope skewness which is the degree of asymmetry of the sensor signal distribution (GSS)
        gss_child = 0
        for i in range(len(self._rms_df)):
            row = self._rms_df.iloc[i]
            mit = float(row['g_rms'])
            gss_child += (mit - gai) ** 3
        gss_child /= len(self._rms_df)
        gss = gss_child / math.pow(gvi, 3.0 / 2.0)
        return gss

    def __get_aks(self, aai, avi):
        # acceleration kurtosis which is the degree of peakedness of the sensor signal distribution (AKS)
        aks_child = 0
        for i in range(len(self._rms_df)):
            row = self._rms_df.iloc[i]
            mit = float(row['a_rms'])
            aks_child += (mit - aai) ** 4
        aks_child /= len(self._rms_df)
        aks = aks_child / (avi ** 3) - 3
        return aks

    def __get_gks(self, gai, gvi):
        # gyroscope kurtosis which is the degree of peakedness of the sensor signal distribution (GKS)
        gks_child = 0
        for i in range(len(self._rms_df)):
            row = self._rms_df.iloc[i]
            mit = float(row['g_rms'])
            gks_child += (mit - gai) ** 4
        gks_child /= len(self._rms_df)
        gks = gks_child / (gvi ** 3) - 3
        return gks

    def __get_air(self):
        # acceleration interquartile range (AIR)
        air = self._rms_df['a_rms'].quantile(0.75) - self._rms_df['a_rms'].quantile(0.25)
        return air

    def __get_gir(self):
        # gyroscope interquartile range (GIR)
        gir = self._rms_df['g_rms'].quantile(0.75) - self._rms_df['g_rms'].quantile(0.25)
        return gir

    def __get_a_zero_cross(self, mami):
        # the total number of times the acceleration signal changes form positive to negative or negative or vice versa
        a_zero_cross = 0
        for i in range(1, len(self._rms_df)):
            prev_row = self._rms_df.iloc[i - 1]
            now_row = self._rms_df.iloc[i]
            if (prev_row['a_rms'] <= mami) and (now_row['a_rms'] > mami):
                a_zero_cross += 1
            elif (prev_row['a_rms'] >= mami) and (now_row['a_rms'] < mami):
                a_zero_cross += 1
        a_zero_cross /= len(self._rms_df)
        return a_zero_cross

    def __get_g_zero_cross(self, mgmi):
        # the total number of times the gyroscope signal changes form positive to negative or negative or vice versa
        g_zero_cross = 0
        for i in range(1, len(self._rms_df)):
            prev_row = self._rms_df.iloc[i - 1]
            now_row = self._rms_df.iloc[i]
            if (prev_row['g_rms'] <= mgmi) and (now_row['g_rms'] > mgmi):
                g_zero_cross += 1
            elif (prev_row['g_rms'] >= mgmi) and (now_row['g_rms'] < mgmi):
                g_zero_cross += 1
        g_zero_cross /= len(self._rms_df)
        return g_zero_cross

    def __get_a_mean_cross(self, aai):
        # the total number of times the acceleration signal changes form below average to above average or vice versa
        a_mean_cross = 0
        for i in range(1, len(self._rms_df)):
            prev_row = self._rms_df.iloc[i - 1]
            now_row = self._rms_df.iloc[i]
            if (prev_row['a_rms'] >= aai) and (now_row['a_rms'] < aai):
                a_mean_cross += 1
            elif (prev_row['a_rms'] <= aai) and (now_row['a_rms'] > aai):
                a_mean_cross += 1
        a_mean_cross /= len(self._rms_df)
        return a_mean_cross

    def __get_g_mean_cross(self, gai):
        # the total number of times the gyroscope signal changes form below average to above average or vice versa
        g_mean_cross = 0
        for i in range(1, len(self._rms_df)):
            prev_row = self._rms_df.iloc[i - 1]
            now_row = self._rms_df.iloc[i]
            if (prev_row['g_rms'] >= gai) and (now_row['g_rms'] < gai):
                g_mean_cross += 1
            elif (prev_row['g_rms'] <= gai) and (now_row['g_rms'] > gai):
                g_mean_cross += 1
        g_mean_cross /= len(self._rms_df)
        return g_mean_cross

    def __get_a_xy_corr(self):
        # correlation between two acceleration X, Y axes
        a_xy_corr = self._rms_df['imu_ax'].corr(self._rms_df['imu_ay'])
        return a_xy_corr

    def __get_a_yz_corr(self):
        # correlation between two acceleration Y, Z axes
        a_yz_corr = self._rms_df['imu_ay'].corr(self._rms_df['imu_az'])
        return a_yz_corr

    def __get_a_zx_corr(self):
        # correlation between two acceleration X, Z axes
        a_zx_corr = self._rms_df['imu_az'].corr(self._rms_df['imu_ax'])
        return a_zx_corr

    def __get_g_xy_corr(self):
        # correlation between two gyroscope X, Y axes
        g_xy_corr = self._rms_df['imu_gx'].corr(self._rms_df['imu_gy'])
        return g_xy_corr

    def __get_g_yz_corr(self):
        # correlation between two gyroscope Y, Z axes
        g_yz_corr = self._rms_df['imu_gy'].corr(self._rms_df['imu_gz'])
        return g_yz_corr

    def __get_g_zx_corr(self):
        # correlation between two gyroscope X, Z axes
        g_zx_corr = self._rms_df['imu_gz'].corr(self._rms_df['imu_gx'])
        return g_zx_corr

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


def get_features(window):
    """
    Get all engineered features from the window.

    :param window: the range of sensor data we care about
    :return: engineered features
    """

    return _Primitive(window).features
