from .image import *

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_play_score']


def plot_play_score(capture_dir_path, song_id):
    timestamps, img_scores = read_score_board_info(capture_dir_path, song_id)
    score_df = pd.DataFrame(data={
        'timestamp': timestamps,
        'score': img_scores,
    })
    plt.figure(figsize=(15, 5))
    sns.lineplot(x='timestamp', y='score', data=score_df)
    plt.show()
