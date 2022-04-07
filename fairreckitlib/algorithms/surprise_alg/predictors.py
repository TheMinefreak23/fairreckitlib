"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time

import numpy as np
import pandas as pd
from surprise.dataset import Dataset
from surprise.reader import Reader
from surprise.prediction_algorithms import BaselineOnly
from surprise.prediction_algorithms import CoClustering
from surprise.prediction_algorithms import NMF
from surprise.prediction_algorithms import NormalPredictor
from surprise.prediction_algorithms import SlopeOne
from surprise.prediction_algorithms import SVD
from surprise.prediction_algorithms import SVDpp

from ..predictor import PredictorAlgorithm


class PredictorSurprise(PredictorAlgorithm):

    def train(self, train_set):
        reader = Reader(rating_scale=self.rating_scale)
        dataset = Dataset.load_from_df(train_set, reader)
        self.algo.fit(dataset.build_full_trainset())

    def predict(self, user, item):
        prediction = self.algo.predict(user, item, clip=False)
        return prediction.est

    def predict_batch(self, user_item_pairs):
        df = user_item_pairs[['user', 'item']]
        df['prediction'] = np.zeros(len(df))
        for i, row in df.iterrows():
            df.at[i, 'prediction'] = self.predict(
                row['user'],
                row['item']
            )

        return df


def create_predictor_baseline_only_als(params, **kwargs):
    bsl_options = {
        'method': 'als',
        'reg_i': params['reg_i'],
        'reg_u': params['reg_u'],
        'n_epochs': params['epochs']
    }

    return PredictorSurprise(BaselineOnly(
        bsl_options=bsl_options,
        verbose=False
    ), params, **kwargs)


def create_predictor_baseline_only_sgd(params, **kwargs):
    bsl_options = {
        'method': 'sgd',
        'reg': params['regularization'],
        'learning_rate': params['learning_rate'],
        'n_epochs': params['epochs']
     }

    return PredictorSurprise(BaselineOnly(
        bsl_options=bsl_options,
        verbose=False
    ), params, **kwargs)


def create_predictor_co_clustering(params, **kwargs):
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    return PredictorSurprise(CoClustering(
        n_cltr_u=params['user_clusters'],
        n_cltr_i=params['item_clusters'],
        n_epochs=params['epochs'],
        random_state=params['random_seed'],
        verbose=False
    ), params, **kwargs)


def create_predictor_nmf(params, **kwargs):
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    return PredictorSurprise(NMF(
        n_factors=params['factors'],
        n_epochs=params['epochs'],
        biased=params['biased'],
        reg_pu=params['reg_pu'],
        reg_qi=params['reg_qi'],
        reg_bu=params['reg_bu'],
        reg_bi=params['reg_bi'],
        lr_bu=params['lr_bu'],
        lr_bi=params['lr_bi'],
        init_low=params['init_low'],
        init_high=params['init_high'],
        random_state=params['random_seed'],
        verbose=False
    ), params, **kwargs)


def create_predictor_normal_predictor(params, **kwargs):
    return PredictorSurprise(NormalPredictor(), params, **kwargs)


def create_predictor_slope_one(params, **kwargs):
    return PredictorSurprise(SlopeOne(), params, **kwargs)


def create_predictor_svd(params, **kwargs):
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    return PredictorSurprise(SVD(
        n_factors=params['factors'],
        n_epochs=params['epochs'],
        biased=params['biased'],
        init_mean=params['init_mean'],
        init_std_dev=params['init_std_dev'],
        lr_all=params['learning_rate'],
        reg_all=params['regularization'],
        lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
        reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
        random_state=params['random_seed'],
        verbose=False
    ), params, **kwargs)


def create_predictor_svd_pp(params, **kwargs):
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    return PredictorSurprise(SVDpp(
        n_factors=params['factors'],
        n_epochs=params['epochs'],
        init_mean=params['init_mean'],
        init_std_dev=params['init_std_dev'],
        lr_all=params['learning_rate'],
        reg_all=params['regularization'],
        lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None, lr_yj=None,
        reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, reg_yj=None,
        random_state=params['random_seed'],
        verbose=False
    ), params, **kwargs)
