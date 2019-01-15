from .tools.config import *
from .image import *
from .db import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


__all__ = ['plot_play_score',
           'plot_raw_acc_signal',
           'plot_raw_gyr_signal']


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
    sns.set(font_scale=2)
    sns.lineplot(x='timestamp', y='score', data=score_df, hue='who')

    if save_image:
        auc = get_play_score_auc(timestamps=timestamps, img_scores=img_scores)
        times = estimate_remained_play_times(song_id, auc=auc)
        filename = posixpath.join(TMP_DIR_PATH, 'curve_%d.png' % times)
        plt.savefig(filename)
    else:
        plt.show()


def plot_raw_acc_signal(left_df, right_df, marks=None, title=None):
    show_cols = ZERO_ADJ_COL[:3]
    fig, axes = plt.subplots(nrows=len(show_cols), ncols=1, sharex='all', sharey='all', figsize=(30, 30))

    if title is not None:
        axes[0].set_title('accelerometer (' + title + ')', fontsize=45)

    for i_, col in enumerate(show_cols):
        sns.lineplot(data=left_df, x='timestamp', y=col, label='left', ax=axes[i_])
        sns.lineplot(data=right_df, x='timestamp', y=col, label='right', ax=axes[i_])
        axes[i_].set_ylabel(col + ' $(9.8\\ m/s^2)$')
        for x_, color_ in marks:
            axes[i_].axvline(x_, color=color_, alpha=1, lw=0.2)
    plt.xlabel('timestamp $(s)$')
    plt.show()


def plot_raw_gyr_signal(left_df, right_df, marks=None, title=None):
    show_cols = ZERO_ADJ_COL[3:6]
    fig, axes = plt.subplots(nrows=len(show_cols), ncols=1, sharex='all', sharey='all', figsize=(30, 30))

    if title is not None:
        axes[0].set_title('gyroscope (' + title + ')', fontsize=45)

    for i_, col in enumerate(show_cols):
        sns.lineplot(data=left_df, x='timestamp', y=col, label='left', ax=axes[i_])
        sns.lineplot(data=right_df, x='timestamp', y=col, label='right', ax=axes[i_])
        axes[i_].set_ylabel(col + ' $(degree\\/s)$')
        for x_, color_ in marks:
            axes[i_].axvline(x_, color=color_, alpha=1, lw=0.2)
    plt.xlabel('timestamp $(s)$')
    plt.show()
