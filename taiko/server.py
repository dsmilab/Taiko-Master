from .visualize import *

from glob import glob
import posixpath
import sys
import os


def process_captures(capture_dir_path, song_id):
    plot_play_score(capture_dir_path, song_id, True, True)
