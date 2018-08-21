from ..cache import *
from ..io.record import *
from ..tools.score import *
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
from sklearn.metrics import classification_report
import lightgbm as lgb

__all__ = ['LGBM']


class _Model(object):
    def __init__(self, song_id):
        self._ep_dict = {}
        self._load_event_primitive(song_id)

    def _load_event_primitive(self, song_id):
        df = load_drummer_df()
        df = df[df['song_id'] == song_id]
        df = df[['drummer_id', 'performance_order']]

        for _, row in df.iterrows():
            who_id = int(row['drummer_id'])
            order_id = int(row['performance_order'])

            try:
                ep_df = get_event_primitive_df(who_id, song_id, order_id)
            except ValueError:
                continue

            if who_id not in self._ep_dict:
                self._ep_dict[who_id] = {}
            self._ep_dict[who_id][order_id] = ep_df

    @property
    def ep_dict(self):
        return self._ep_dict


class LGBM(_Model):
    def __init__(self, song_id):
        super(LGBM, self).__init__(song_id)
        self._params = dict({
            'learning_rate': 0.1,
            'application': 'multiclass',
            'max_depth': 4,
            'num_leaves': 2 ** 4,
            'verbosity': 0
        })

    def run(self, test_who, num_boost_round=200, verbose_eval=50, early_stopping_round=100,
            params=None, mode='one-to-one'):
        dc = self._ep_dict[test_who]
        order_ids = []
        train_dfs = []
        for key, value in dc.items():
            order_ids.append(key)
            train_dfs.append(value)

        scores = {}
        for i_ in range(len(order_ids)):
            order_id = order_ids[i_]
            train_df = pd.DataFrame(pd.concat(train_dfs[0:i_] + train_dfs[i_ + 1: len(train_dfs)], ignore_index=True))
            test_df = train_dfs[i_]
            f1_score = self._run(train_df, test_df, num_boost_round, verbose_eval, early_stopping_round, params)
            scores[order_id] = f1_score

        return scores

    def _run(self, train_df, test_df, num_boost_round, verbose_eval, early_stopping_round, params):
        x = train_df.drop(['hit_type'], axis=1)
        y = train_df['hit_type']
        train_set = lgb.Dataset(x, y)
        valid_set = lgb.Dataset(x, y, free_raw_data=False)
        watchlist = [valid_set]

        my_params = self._params
        my_params['num_classes'] = len(y.unique())
        if params is not None:
            my_params.update(params)

        model = lgb.train(my_params,
                          train_set=train_set,
                          valid_sets=watchlist,
                          num_boost_round=num_boost_round,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stopping_round,
                          feval=my_f1_score)

        x_test = test_df.drop(['hit_type'], axis=1)
        y_true = test_df['hit_type']
        y_pred = model.predict(x_test, num_iteration=model.best_iteration)
        y_test = pd.Series(data=[np.argmax(xx) for xx in y_pred])

        f1_score = round(metrics.f1_score(y_test, y_true, average='weighted'), 4)

        print(classification_report(y_true, y_test))

        return f1_score
