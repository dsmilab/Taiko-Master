import matplotlib
matplotlib.use('Agg')

from .config import *
from .image import *
from .db import *
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_play_score']


def plot_play_score(capture_dir_path, song_id, compare_veteran=False, save_image=False):
    timestamps, img_scores = read_score_board_info(capture_dir_path, song_id)
    whos = ['you'] * len(timestamps)

    if compare_veteran:
        veteran_timestamps, veteran_img_scores = get_best_score_board_info(song_id)
        timestamps.extend(veteran_timestamps)
        img_scores.extend(veteran_img_scores)
        whos.extend(['veteran'] * len(veteran_timestamps))

    score_df = pd.DataFrame(data={
        'timestamp': timestamps,
        'score': img_scores,
        'who': whos,
    })

    plt.figure(figsize=(20, 8))
    sns.lineplot(x='timestamp', y='score', data=score_df, hue='who')

    if save_image:
        auc = get_play_score_auc(timestamps=timestamps, img_scores=img_scores)
        times = estimate_remained_play_times(song_id, auc=auc)
        filename = posixpath.join(PIC_DIR_PATH, 'curve_%d.png' % times)
        plt.savefig(filename)
    else:
        plt.show()
