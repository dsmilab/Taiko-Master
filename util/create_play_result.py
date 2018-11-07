import sys
sys.path.append('..')

from taiko.db import Database
from taiko.db import *
from taiko.play import *
from taiko.image import *
from taiko.config import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import posixpath

SONGS = 4
def create_play_result(verbose=1):
    record_df = Database().record_df

    # only 4 songs we considereda
    record_df = record_df[(record_df['song_id'] >= 1) & (record_df['song_id'] <= SONGS)]

    aggregate_dict = {}
    columns = ['drummer_name', 'song_id', 'p_order', 'capture_datetime', 'auc'] + RESULT_BOARD_INFO_COLUMNS
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

            auc = get_play_score_auc(capture_dir_path, song_id)

            data['drummer_name'].append(who_name)
            data['song_id'].append(song_id)
            data['p_order'].append(p_order)
            data['capture_datetime'].append(capture_dir)
            data['auc'].append(auc)

            for col in result.keys():
                data[col].append(result[col])
            
            if verbose > 0:
                message = 'who_name = %s, song_id = %d, p_order = %d, %s, auc = %.4f\n' \
                          'result = %s' % \
                          (who_name, song_id, p_order, capture_dir, auc, str(result))
                sys.stdout.write(message)
                sys.stdout.flush()
            
            if result['bad'] > 0:
                continue
                
            play = get_play(row)
            for position in ['L', 'R']:
                play.crop_near_raw_data(who_name, song_id, p_order, 0.1, position, id_)
                
        except Exception as e:
            print(e)

    play_result_df = pd.DataFrame(data=data)
    play_result_df.to_csv(PLAY_RESULT_TABLE_PATH, index=False, float_format='%.4f')

    
create_play_result()