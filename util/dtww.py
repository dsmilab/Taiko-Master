import sys
sys.path.append('..')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import taiko as tk
from taiko.play import *
from taiko.performance import *
from taiko.model import *
from taiko.tools.database import *

from sklearn.model_selection import train_test_split
import lightgbm as lgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import posixpath
from skimage.io import imshow, imsave, imread

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

sns.set(font_scale=1.5)

record_df = load_record_df(song_id=1)
record_df

lgbm = LGBM()
pfs = {}
for id_, row in record_df.iterrows():
    try:
        pf = get_performance(id_=id_)
        x = pf.drop('timestamp', axis=1)
        ts = pf['timestamp']
        pred_df = pd.DataFrame(data={
            'timestamp': ts,
            'hit_type': lgbm.predict(x)
        })
        pfs[id_] = pred_df
    except Exception as e:
        pass

tmp_mat = []
for i_ in pfs.keys():
    for j_ in pfs.keys():
        if i_ >= j_:
            continue
        a_df = pfs[i_]
        b_df = pfs[j_]
        
        lst = [i_, j_]
        for label in range(6):
            x = np.array(a_df[a_df['hit_type'] == label].timestamp)
            if(len(x) == 0):
                x = np.array([0])
            x -= x[0]
            y = np.array(b_df[b_df['hit_type'] == label].timestamp)
            if(len(y) == 0):
                y = np.array([0])
            y -= y[0]
            distance, _ = fastdtw(x, y, dist=euclidean)
            lst += [distance]
        tmp_mat.append(lst)

final_df = pd.DataFrame(data=tmp_mat)
final_df.to_csv('final.csv', index=False, float_format='%.4f')
