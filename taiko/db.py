from .tools.singleton import *
from .image import *
from .play import *
from .config import *

import pandas as pd
import numpy as np
from glob import glob

__all__ = ['update_play_result',
           'update_play_score_auc',
           'get_score_auc_stat']

SONGS = 4


class Database(metaclass=_Singleton):

    def __init__(self):
        self.__load_record_df()

    def __load_record_df(self):
        record_files = glob('../data/alpha/*/*/record_table.csv')
        record_dfs = []
        for record_file_path in record_files:
            record_df = pd.read_csv(record_file_path)
            record_dfs.append(record_df)
        self._record_df = pd.concat(record_dfs, ignore_index=True)

    @property
    def record_df(self):
        return self._record_df


def update_play_result(verbose=1):
    record_df = Database().record_df

    # only 4 songs we considered
    record_df = record_df[(record_df['song_id'] >= 1) & (record_df['song_id'] <= SONGS)]

    aggregate_dict = {}
    columns = ['drummer_name', 'song_id', 'p_order', 'capture_datetime'] + RESULT_BOARD_INFO_COLUMNS
    data = {}
    for col in columns:
        data[col] = []

    for id_, row in record_df.iterrows():
        try:
            capture_dir = row['capture_datetime']
            who_name = row['drummer_name']
            song_id = row['song_id']
            dirs = glob('../data/alpha/' + who_name + '/*/bb_capture/' + capture_dir)
            capture_dir_path = dirs[0]
            result = read_result_board_info(capture_dir_path)

            # increment p_order
            try:
                aggregate_dict[(who_name, song_id)]
            except KeyError:
                aggregate_dict[(who_name, song_id)] = 0

            p_order = aggregate_dict[(who_name, song_id)] + 1
            aggregate_dict[(who_name, song_id)] = p_order

            if verbose > 0:
                print(who_name, capture_dir)
                print('who = %s, id = %d, p_order = %d' % (who_name, id_, p_order))
                print(result)

            data['drummer_name'].append(who_name)
            data['song_id'].append(song_id)
            data['p_order'].append(p_order)
            data['capture_datetime'].append(capture_dir)
            for col in result.keys():
                data[col].append(result[col])

        except Exception as e:
            print(e)

    play_result_df = pd.DataFrame(data=data)
    play_result_df.to_csv('../data/taiko_tables/taiko_play_result.csv', index=False)


def update_play_score_auc(verbose=1):
    record_df = Database().record_df

    # only 4 songs we considered
    record_df = record_df[(record_df['song_id'] >= 1) & (record_df['song_id'] <= SONGS)]

    columns = ['drummer_name', 'song_id', 'p_order', 'capture_datetime', 'auc']
    data = {}
    for col in columns:
        data[col] = []

    aggregate_dict = {}
    for id_, row in record_df.iterrows():
        try:
            capture_dir = row['capture_datetime']
            who_name = row['drummer_name']
            song_id = row['song_id']
            dirs = glob('../data/alpha/' + who_name + '/*/bb_capture/' + capture_dir)
            capture_dir_path = dirs[0]

            # increment p_order
            try:
                aggregate_dict[(who_name, song_id)]
            except KeyError:
                aggregate_dict[(who_name, song_id)] = 0

            p_order = aggregate_dict[(who_name, song_id)] + 1
            aggregate_dict[(who_name, song_id)] = p_order

            auc = get_play_score_auc(capture_dir_path, song_id)

            data['drummer_name'].append(who_name)
            data['song_id'].append(song_id)
            data['p_order'].append(p_order)
            data['capture_datetime'].append(capture_dir)
            data['auc'].append(auc)

            if verbose > 0:
                print(who_name, capture_dir, auc)

        except Exception as e:
            print(e)

    play_score_auc_df = pd.DataFrame(data=data)
    play_score_auc_df.to_csv('../data/taiko_tables/taiko_play_score_auc.csv', index=False, float_format='%.4f')


def get_score_auc_stat(song_id):
    play_score_auc_df = pd.read_csv('../data/taiko_tables/taiko_play_score_auc.csv')
    play_result_df = pd.read_csv('../data/taiko_tables/taiko_play_result.csv')
    df = play_score_auc_df.merge(play_result_df,
                                 on=['drummer_name', 'song_id', 'p_order', 'capture_datetime'],
                                 how='left')
    df = df[(df['song_id'] == song_id)]
    full_combo_df = df[df['bad'] == 0]
    fc_mean_auc = np.mean(full_combo_df['auc'])

    dif_auc = []
    for id_, row in df.iterrows():
        auc = row['auc']
        dif_auc.append(auc - fc_mean_auc)
    std_auc = np.std(dif_auc, ddof=1)

    return {'fc_mean_auc': fc_mean_auc,
            'std_auc': std_auc}
