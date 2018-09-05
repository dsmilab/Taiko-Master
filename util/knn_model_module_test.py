import taiko as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

LABEL_GROUP = 'single_stream'
OVER_SAMPLED = False


def main():
    result_df = pd.DataFrame(
        columns=['test_who', 'song_id', 'test_order', 'f1_score', 'feat', 'mode'])
    MODE = ['one-to-one', 'rest-to-one', 'all-to-one']
    MAPP = {(True, True, False): 'acc + gyr',
            (True, False, False): 'acc',
            (False, True, False): 'gyr',
           }
    id_ = 0

    for song in range(4):
        for acc in [True, False]:
            for gyr in [True, False]:
                for near in [True, False]:
                    if (acc, gyr, near) not in MAPP.keys():
                        continue

                    if OVER_SAMPLED and near:
                        continue

                    model = tk.KNN(song + 1, acc, gyr, near, over_sampled=OVER_SAMPLED, label_group=LABEL_GROUP)
                    for who in tqdm(range(8)):
                        for mode in MODE:
                            res = model.run(who + 1, mode=mode)
                            for key, val in res.items():
                                result_df.loc[id_] = [who + 1,
                                                      song + 1,
                                                      key,
                                                      val,
                                                      MAPP[(acc, gyr, near)],
                                                      mode]
                                id_ += 1

    result_df.to_csv('CSV/KNN_result_' + LABEL_GROUP + '_' + str(OVER_SAMPLED) + '.csv', index=False)


if __name__ == '__main__':
    LABEL_GROUP = sys.argv[1]
    if sys.argv[2] == 'true':
        OVER_SAMPLED = True
    main()
