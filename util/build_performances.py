import sys
sys.path.append('..')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import taiko as tk
from taiko.model import *
from taiko.play import *
from taiko.performance import *

import lightgbm as lgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import posixpath

sns.set(font_scale=1.5)

record_files = glob('../data/alpha/*/*/record_table.csv')
record_dfs = []
for record_file_path in record_files:
    record_df = pd.read_csv(record_file_path)
    record_dfs.append(record_df)
record_df = pd.concat(record_dfs, ignore_index=True)

record_df = record_df[(record_df['song_id'] >= 1) & (record_df['song_id'] <= 4)]

for id_, row in record_df.iterrows():
    try:
        record_row = record_df.loc[id_]
        play = get_play(record_row)
        pf = get_performance(play, id_=id_)
    except Exception as e:
        print(id_)
    