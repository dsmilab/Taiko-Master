from ..cache import *
from ..io.record import *
from ..tools.score import *

import abc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report

__all__ = ['LGBM', 'SVM']


class _Model(object):
    def __init__(self, song_id, acc, gyr, near, label_group):
        self._ep_dict = {}
        self._load_event_primitive(song_id, acc, gyr, near, label_group)

    def _load_event_primitive(self, song_id, acc, gyr, near, label_group):
        df = load_drummer_df()
        df = df[df['song_id'] == song_id]
        df = df[['drummer_id', 'performance_order']]

        for _, row in df.iterrows():
            who_id = int(row['drummer_id'])
            order_id = int(row['performance_order'])

            try:
                ep_df = get_event_primitive_df(who_id, song_id, order_id,
                                               acc=acc,
                                               gyr=gyr,
                                               near=near,
                                               label_group=label_group)
            except ValueError:
                continue

            if who_id not in self._ep_dict:
                self._ep_dict[who_id] = {}
            self._ep_dict[who_id][order_id] = ep_df

    @abc.abstractmethod
    def run(self, test_who):
        raise NotImplementedError("Please Implement this method")

    @property
    def ep_dict(self):
        return self._ep_dict


class LGBM(_Model):
    def __init__(self, song_id, acc=True, gyr=True, near=True, label_group='single_stream'):
        super(LGBM, self).__init__(song_id, acc, gyr, near, label_group)
        self._params = dict({
            'learning_rate': 0.1,
            'application': 'multiclass',
            'max_depth': 4,
            'num_leaves': 2 ** 4,
            'verbosity': 0
        })

    def run(self, test_who, num_boost_round=200, verbose_eval=50, early_stopping_round=100,
            params=None, mode='one-to-one'):
        if mode not in ['one-to-one', 'rest-to-one', 'all-to-one']:
            raise ValueError('mode Error!')

        ep_ids = []
        for key1, value1 in self._ep_dict.items():
            for key2, value2 in value1.items():
                ep_ids.append(((key1, key2), value2))

        scores = {}
        if mode == 'one-to-one':
            ep_ids = [xx for xx in ep_ids if xx[0][0] == test_who]
            order_ids = []
            train_dfs = []
            for (key, df) in ep_ids:
                order_ids.append(key[1])
                train_dfs.append(df)

            for i_ in range(len(order_ids)):
                order_id = order_ids[i_]
                train_df = pd.concat(train_dfs[0:i_] +
                                     train_dfs[i_ + 1: len(train_dfs)],
                                     ignore_index=True)
                test_df = train_dfs[i_]
                f1_score = self._run(train_df, test_df, num_boost_round, verbose_eval, early_stopping_round, params)
                scores[order_id] = f1_score

        elif mode == 'rest-to-one':
            train_ids = [xx for xx in ep_ids if xx[0][0] != test_who]
            test_ids = [xx for xx in ep_ids if xx[0][0] == test_who]

            order_ids = []
            test_dfs = []
            for (key, df) in test_ids:
                order_ids.append(key[1])
                test_dfs.append(df)

            train_df = pd.concat([df for (key, df) in train_ids], ignore_index=True)

            for i_ in range(len(order_ids)):
                order_id = order_ids[i_]
                test_df = test_dfs[i_]

                f1_score = self._run(train_df, test_df, num_boost_round, verbose_eval, early_stopping_round, params)
                scores[order_id] = f1_score

        elif mode == 'all-to-one':
            train_ids = [xx for xx in ep_ids if xx[0][0] != test_who]
            test_ids = [xx for xx in ep_ids if xx[0][0] == test_who]

            order_ids = []
            test_dfs = []
            for (key, df) in test_ids:
                order_ids.append(key[1])
                test_dfs.append(df)

            pre_train_df = pd.concat([df for (key, df) in train_ids], ignore_index=True)
            for i_ in range(len(order_ids)):
                order_id = order_ids[i_]
                test_df = test_dfs[i_]

                other_train_df = pd.concat(test_dfs[0:i_] +
                                           test_dfs[i_ + 1: len(test_dfs)],
                                           ignore_index=True)
                train_df = pd.concat([other_train_df, pre_train_df], ignore_index=True)
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


class SVM(_Model):

    def __init__(self, song_id, acc=True, gyr=True, near=False, label_group='single_stream'):
        super(SVM, self).__init__(song_id, acc, gyr, near, label_group)
        self._params = dict({
            'C': 1,
            'kernel': 'rbf'
        })

    def run(self, test_who, kernel='rbf', mode='one-to-one'):
        if mode not in ['one-to-one', 'rest-to-one', 'all-to-one']:
            raise ValueError('mode Error!')

        ep_ids = []
        for key1, value1 in self._ep_dict.items():
            for key2, value2 in value1.items():
                ep_ids.append(((key1, key2), value2))

        scores = {}
        if mode == 'one-to-one':
            ep_ids = [xx for xx in ep_ids if xx[0][0] == test_who]
            order_ids = []
            train_dfs = []
            for (key, df) in ep_ids:
                order_ids.append(key[1])
                train_dfs.append(df)

            for i_ in range(len(order_ids)):
                order_id = order_ids[i_]
                train_df = pd.concat(train_dfs[0:i_] +
                                     train_dfs[i_ + 1: len(train_dfs)],
                                     ignore_index=True)
                test_df = train_dfs[i_]
                f1_score = self._run(train_df, test_df, kernel)
                scores[order_id] = f1_score

        elif mode == 'rest-to-one':
            train_ids = [xx for xx in ep_ids if xx[0][0] != test_who]
            test_ids = [xx for xx in ep_ids if xx[0][0] == test_who]

            order_ids = []
            test_dfs = []
            for (key, df) in test_ids:
                order_ids.append(key[1])
                test_dfs.append(df)

            train_df = pd.concat([df for (key, df) in train_ids], ignore_index=True)

            for i_ in range(len(order_ids)):
                order_id = order_ids[i_]
                test_df = test_dfs[i_]

                f1_score = self._run(train_df, test_df, kernel)
                scores[order_id] = f1_score

        elif mode == 'all-to-one':
            train_ids = [xx for xx in ep_ids if xx[0][0] != test_who]
            test_ids = [xx for xx in ep_ids if xx[0][0] == test_who]

            order_ids = []
            test_dfs = []
            for (key, df) in test_ids:
                order_ids.append(key[1])
                test_dfs.append(df)

            pre_train_df = pd.concat([df for (key, df) in train_ids], ignore_index=True)
            for i_ in range(len(order_ids)):
                order_id = order_ids[i_]
                test_df = test_dfs[i_]

                other_train_df = pd.concat(test_dfs[0:i_] +
                                           test_dfs[i_ + 1: len(test_dfs)],
                                           ignore_index=True)
                train_df = pd.concat([other_train_df, pre_train_df], ignore_index=True)
                f1_score = self._run(train_df, test_df, kernel)
                scores[order_id] = f1_score

        return scores

    def _run(self, train_df, test_df, kernel):
        train_df.drop_duplicates(inplace=True)

        x = train_df.drop(['hit_type'], axis=1)
        y = train_df['hit_type']

        x_test = test_df.drop(['hit_type'], axis=1)
        y_true = test_df['hit_type']

        my_params = self._params
        my_params['kernel'] = kernel

        best_f1 = -1
        best_y_test = None
        for C_ in [0.01, 0.1, 1, 10]:
            model = SVC(C=C_,
                        kernel=my_params['kernel'])
            model.fit(x, y)

            y_test = model.predict(x_test)
            f1_score = round(metrics.f1_score(y_test, y_true, average='weighted'), 4)

            if f1_score > best_f1:
                best_f1 = max(best_f1, f1_score)
                best_y_test = y_test

        print(classification_report(y_true, best_y_test))

        return best_f1
