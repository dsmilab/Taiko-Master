from .tools.config import *
from .image import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import posixpath


__all__ = ['plot_raw_acc_signal',
           'plot_raw_gyr_signal']


def plot_raw_acc_signal(left_df, right_df, marks=None, title=None, figname=None):
    show_cols = ZERO_ADJ_COL[:3]
    fig, axes = plt.subplots(nrows=len(show_cols), ncols=1, sharex='all', sharey='all', figsize=(30, 30))

    if title is not None:
        axes[0].set_title('accelerometer (' + title + ')', fontsize=45)

    for i_, col in enumerate(show_cols):
        sns.lineplot(data=left_df, x='timestamp', y=col, label='left', ax=axes[i_])
        sns.lineplot(data=right_df, x='timestamp', y=col, label='right', ax=axes[i_])
        axes[i_].set_ylabel(col + ' $(9.8\\ m/s^2)$')
        for x_, color_ in marks:
            axes[i_].axvline(x_, color=color_, alpha=1, lw=0.5)
    plt.xlabel('timestamp $(s)$')

    if figname is None:
        plt.show()
    else:
        figname_path = posixpath.join(RAW_PLAY_FIG_DIR_PATH, figname)
        figname_dir_path = os.path.dirname(figname_path)
        os.makedirs(figname_dir_path, exist_ok=True)
        plt.savefig(figname_path)
    plt.close()


def plot_raw_gyr_signal(left_df, right_df, marks=None, title=None, figname=None):
    show_cols = ZERO_ADJ_COL[3:6]
    fig, axes = plt.subplots(nrows=len(show_cols), ncols=1, sharex='all', sharey='all', figsize=(30, 30))

    if title is not None:
        axes[0].set_title('gyroscope (' + title + ')', fontsize=45)

    for i_, col in enumerate(show_cols):
        sns.lineplot(data=left_df, x='timestamp', y=col, label='left', ax=axes[i_])
        sns.lineplot(data=right_df, x='timestamp', y=col, label='right', ax=axes[i_])
        axes[i_].set_ylabel(col + ' $(degree\\/s)$')
        for x_, color_ in marks:
            axes[i_].axvline(x_, color=color_, alpha=1, lw=0.5)
    plt.xlabel('timestamp $(s)$')

    if figname is None:
        plt.show()
    else:
        figname_path = posixpath.join(RAW_PLAY_FIG_DIR_PATH, figname)
        figname_dir_path = os.path.dirname(figname_path)
        os.makedirs(figname_dir_path, exist_ok=True)
        plt.savefig(figname_path)
    plt.close()
