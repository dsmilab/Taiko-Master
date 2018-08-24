from ..config import *
from .record import *

import pandas as pd
import os

__all__ = ['load_note_df']

DROPPED_COLUMNS = ['#', 'separator', 'continuous']
RENAMED_COLUMNS = ['bar', 'bpm', 'time_unit', 'timestamp', 'label']
NOTE_MAGIC_STR = 'taiko_song_%d_%s_info.csv'


class _Note(object):
    """
    Drum note

    :protected attributes:
        drummer_df: dataframe about performance of drummers
        who_id: # of drummer
        song_id: # of song
        order_id: # of performance repetitively
        note_df: drum note dataframe of the particular song
    """

    def __init__(self, who_id, song_id, order_id):
        self._drummer_df = load_drummer_df()
        self._who_id = who_id
        self._song_id = song_id
        self._order_id = order_id

        difficulty = get_record(who_id, song_id, order_id)['difficulty']
        note_file_name = NOTE_MAGIC_STR % (self._song_id, difficulty)

        self._note_df = pd.read_csv(os.path.join(TABLE_PATH, note_file_name))
        self._note_df.drop(DROPPED_COLUMNS, axis=1, inplace=True)
        self._note_df.columns = RENAMED_COLUMNS

    @property
    def note_df(self):
        return self._note_df


def load_note_df(who_id, song_id, order_id):
    """
    Load particular drum note dataframe.

    :param who_id: # of drummer
    :param song_id: # of song
    :param order_id: # of performance repetitively
    :return: drum note dataframe of the particular song
    """
    return _Note(who_id, song_id, order_id).note_df
