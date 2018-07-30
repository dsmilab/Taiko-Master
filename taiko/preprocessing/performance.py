from ..io.note import *
from ..io.record import *

__all__ = ['get_performance']


class _Performance(object):

    def __init__(self, who_id, song_id, order_id):
        self._note_df = load_note_df(who_id, song_id, order_id)
        self._first_hit_time = get_record(who_id, song_id, order_id)['first_hit_time']

        print(self.__retrieve_event())

    def __retrieve_event(self):
        """
        Retrieve event which means note occurs of the song.

        :return: 2D array
        """

        events = []

        # spot vertical mark lines
        for _, row in self._note_df.iterrows():
            hit_type = int(row['label'])
            events.append((self._first_hit_time + row['timestamp'], hit_type))

        return events


def get_performance(who_id, song_id, order_id):
    return _Performance(who_id, song_id, order_id)
