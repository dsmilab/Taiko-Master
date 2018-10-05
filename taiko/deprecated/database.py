from taiko.config import *
from taiko.tools.timestamp import *
from screenshot import *
import pandas as pd

__all__ = ['Database']


class Database(object):
    def __init__(self):
        pass

    @staticmethod
    def insert_play(who_name, gender, song_id, difficulty, ori_record_start_time):
        capture_dir = 'capture_' + ori_record_start_time
        img_dir = 'bb_capture_output/' + capture_dir
        play_start_time = get_play_start_time(img_dir)
        if play_start_time is None:
            raise RuntimeError('cannot processing %s image folder' % img_dir)

        record_start_time = convert_datetime_format(ori_record_start_time)
        intro_length = Database.__query_intro_length(song_id)
        first_hit_time = play_start_time + intro_length

        who_id = Database.__query_drummer_id(who_name, gender)

        performance_order = Database.__query_performance_order(who_id, song_id)

        play_df = pd.read_csv(PLAY_TABLE_PATH)

        pid = max(play_df.pid) + 1 if len(play_df.pid) > 0 else 1

        index_ = play_df.index[-1] + 1
        play_df.loc[index_] = [pid, who_id, song_id, performance_order, difficulty, record_start_time, first_hit_time]
        play_df.to_csv(PLAY_TABLE_PATH, index=False)

    @staticmethod
    def __query_drummer_id(who_name, gender):
        drummer_df = pd.read_csv(DRUMMER_TABLE_PATH)
        if not (who_name in list(drummer_df.name)):
            last_id = drummer_df['id'].iloc[-1]
            index_ = drummer_df.index[-1] + 1
            drummer_df.loc[index_] = [last_id + 1, who_name, 0, gender, 0]
            drummer_df.to_csv(DRUMMER_TABLE_PATH, index=False)

        row = drummer_df[drummer_df.name == who_name]
        who_id = int(row['id'].iloc[0])

        return who_id

    @staticmethod
    def __query_intro_length(song_id):
        song_df = pd.read_csv(SONG_TABLE_PATH)
        row = song_df[song_df.song_id == song_id]
        intro_length = float(row['intro_length'].iloc[0])

        return intro_length

    @staticmethod
    def __query_performance_order(who_id, song_id):
        play_df = pd.read_csv(PLAY_TABLE_PATH)
        sel_df = play_df[(play_df.drummer_id == who_id) & (play_df.song_id == song_id)].copy()

        performance_order = max(sel_df.performance_order) + 1 if len(sel_df.performance_order) > 0 else 1
        return performance_order
