from .profile import get_profile
from .database import get_all_drummers
from .tools.score import my_f1_score
from .tools.config import *

from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import lightgbm as lgb
import posixpath
import multiprocessing

__all__ = ['LGBM', 'KNN']


class _Model(object):
    def __init__(self, verbose=0):
        self._load_profiles(verbose)

    def _load_profiles(self, verbose) -> None:
        pfs = []
        with multiprocessing.Pool() as p:
            drummers = get_all_drummers()
            if verbose > 0:
                for id_, pf in tqdm(enumerate(p.imap(get_profile, drummers)), total=len(drummers)):
                    pf['who'] = id_
                    pfs.append(pf)
            else:
                for id_, pf in enumerate(p.imap(get_profile, drummers)):
                    pf['who'] = id_
                    pfs.append(pf)

        self._pf = pd.concat(pfs, ignore_index=True)


class LGBM(_Model):
    def __init__(self):
        super(LGBM, self).__init__()
        self._params = dict({
            'learning_rate': 0.2,
            'application': 'multiclass',
            'max_depth': 5,
            'num_leaves': 2 ** 5,
            'verbosity': 0,
            'metric': 'None',
        })
        self._model_name = 'lgbm.h5'

    def pre_train(self, params=None) -> None:
        best_score, best_model = -1, None
        for seed_ in range(10):
            seed = 1000 * seed_ + 997

            x = self._pf.drop(['hit_type', 'who'], axis=1)
            y = self._pf['hit_type']

            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed, test_size=0.5, stratify=y)
            train_set = lgb.Dataset(x_train, y_train)
            valid_set = lgb.Dataset(x_test, y_test, free_raw_data=False)
            watchlist = [valid_set]

            my_params = self._params
            my_params['num_classes'] = len(y.unique())
            if params is not None:
                my_params.update(params)
            model = lgb.train(my_params,
                              train_set=train_set,
                              valid_sets=watchlist,
                              num_boost_round=500,
                              verbose_eval=5,
                              early_stopping_rounds=100,
                              feval=my_f1_score)

            y_pred = model.predict(x_test, num_iteration=model.best_iteration)
            y_pred = pd.Series(data=[np.argmax(xx) for xx in y_pred])
            f1_score = metrics.f1_score(y_test, y_pred, average='macro')
            if f1_score > best_score:
                best_score = f1_score
                best_model = model
            print(classification_report(y_test, y_pred))

        best_model.save_model(posixpath.join(EXTERNAL_PATH, self._model_name))

    def predict(self, x) -> pd.Series:
        model = lgb.Booster(model_file=posixpath.join(EXTERNAL_PATH, self._model_name))
        y_pred = model.predict(x, num_iteration=model.best_iteration)
        predictions = pd.Series(data=[np.argmax(xx) for xx in y_pred])

        return predictions


class KNN(_Model):
    def __init__(self):
        super(KNN, self).__init__()

    def pre_train(self):
        for seed_ in range(10):
            seed = 1000 * seed_ + 997

            x = self._pf.drop(['hit_type'], axis=1)
            y = self._pf['hit_type']

            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed, test_size=0.5, stratify=y)
            best_f1 = -1
            best_y_pred = None
            for K_ in [5, 10, 20, 50]:
                model = KNeighborsClassifier(n_neighbors=K_,
                                             n_jobs=-1)
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                f1_score = round(metrics.f1_score(y_test, y_pred, average='macro'), 4)

                if f1_score > best_f1:
                    best_f1 = max(best_f1, f1_score)
                    best_y_pred = y_pred

            print(classification_report(y_test, best_y_pred))

