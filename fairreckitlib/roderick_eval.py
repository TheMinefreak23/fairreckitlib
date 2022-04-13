from datetime import datetime
import os
import time

import yaml

from fairreckitlib.data.set import DATASET_LFM_360K, DATASET_LFM_1B, DATASET_LFM_2B
from fairreckitlib.data.set import DATASET_ML_100K, DATASET_ML_25M
from fairreckitlib.data.split.factory import SPLIT_RANDOM, SPLIT_TEMPORAL
from fairreckitlib.experiment.common import *
from fairreckitlib.experiment.config import *
from fairreckitlib.recommender_system import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def print_available_datasets(recommender_system):
    print('\nAVAILABLE DATASETS:')
    datasets = recommender_system.get_available_datasets()
    for dataset_name in datasets:
        print('')
        print(dataset_name)
        print('----------')
        for key in datasets[dataset_name]:
            print(key, '=>', datasets[dataset_name][key])
        print('----------')


def print_available_predictors(recommender_system, params=False):
    print('\nAVAILABLE PREDICTORS:')
    predictors = recommender_system.get_available_predictors()
    for algo_api in predictors:
        print('')
        print(algo_api)
        print('----------')
        for algo in predictors[algo_api]:
            if params:
                print(algo[EXP_KEY_MODEL_NAME], algo[EXP_KEY_MODEL_PARAMS])
            else:
                print(algo[EXP_KEY_MODEL_NAME])
        print('----------')


def print_available_recommenders(recommender_system, params=False):
    print('\nAVAILABLE RECOMMENDERS:')
    recommenders = recommender_system.get_available_recommenders()
    for algo_api in recommenders:
        print('')
        print(algo_api)
        print('----------')
        for algo in recommenders[algo_api]:
            if params:
                print(algo[EXP_KEY_MODEL_NAME], algo[EXP_KEY_MODEL_PARAMS])
            else:
                print(algo[EXP_KEY_MODEL_NAME])
        print('----------')


def create_prediction_experiment_config(experiment_name):
    datasets = [
        create_config_dataset(DATASET_ML_100K, 0.2, SPLIT_RANDOM),
        # create_config_dataset(DATASET_ML_100K, 0.3, SPLIT_RANDOM)
    ]
    models = create_config_all_predictor_models(
        lenskit=True,
        surprise=True
    )
    evaluation = {}

    return {
        EXP_KEY_NAME: experiment_name,
        EXP_KEY_TYPE: EXP_TYPE_PREDICTION,
        EXP_KEY_DATASETS: datasets,
        EXP_KEY_MODELS: models,
        EXP_KEY_EVALUATION: evaluation
    }


def run_prediction_experiment(recommender_system, num_threads):
    stamp = str(int(datetime.timestamp(datetime.now())))
    config = create_prediction_experiment_config(stamp + '_HelloFRK')

    """
    with open(os.path.join('..', 'predictors.yml'), 'w') as file:
        yaml.dump(config, file)
    """

    recommender_system.run_experiment(config, num_threads=num_threads)


def create_recommender_experiment_config(experiment_name):
    datasets = [
        create_config_dataset(DATASET_ML_100K, 0.2, SPLIT_RANDOM),
        # create_config_dataset(DATASET_ML_100K, 0.3, SPLIT_RANDOM)
    ]
    models = create_config_all_recommender_models(
        elliot=False,
        implicit=True,
        lenskit=False
    )

    from fairreckitlib.metrics.metrics2 import Metric
    from fairreckitlib.metrics.filter import Filter
    metrics = [Metric.precision, Metric.recall, Metric.mrr, Metric.rmse, Metric.item_coverage, Metric.user_coverage]
    gender_filter = {'type': Filter.Equals.value, 'name': 'gender', 'value': 'male'}
    country_filter = {'type': Filter.Equals.value, 'name': 'country', 'value': 'Mexico'}
    age_filter = {'type': Filter.Clamp.value, 'name': 'age', 'value': {'min': 15, 'max': 25}}
    filters = [
        [gender_filter],
        [country_filter, age_filter]  # Multiple filter 'passes'
    ]
    evaluation = {
        'metrics': metrics,
        'filters': filters
    }

    return {
        EXP_KEY_NAME: experiment_name,
        EXP_KEY_TYPE: EXP_TYPE_RECOMMENDATION,
        EXP_KEY_TOP_K: 10,
        EXP_KEY_DATASETS: datasets,
        EXP_KEY_MODELS: models,
        EXP_KEY_EVALUATION: evaluation
    }


def run_recommender_experiment(recommender_system, num_threads):
    stamp = str(int(datetime.timestamp(datetime.now())))
    config = create_recommender_experiment_config(stamp + '_HelloFRK')

    """
    with open(os.path.join('..', 'recommenders.yml'), 'w') as file:
        yaml.dump(config, file)
    """

    recommender_system.run_experiment(config, num_threads=num_threads)


import pandas as pd
from scipy import sparse
import numpy as np


def test_360k():
    df = pd.read_csv('../LFM-360K_New.tsv', sep='\t', header=None, names=['user_id', 'artist_id', 'plays'])
    print(df)

    df["user_id"] = df["user_id"].astype("category")
    df["artist_id"] = df["artist_id"].astype("category")
    print(len(df["artist_id"]))
    artist_list = list(df["user_id"].cat.categories)
    print(len(artist_list))

    plays = sparse.coo_matrix(
        (
            df["plays"].astype(np.float32),
            (df["user_id"].cat.codes.copy(), df["artist_id"].cat.codes.copy()),
        )
    )
    print(plays.shape)
    print(plays)

    plays = plays.tocsr()
    print(plays.shape)
    print(plays)
    raise NotImplementedError()


import zlib
from pandas.api.types import CategoricalDtype


def test_lfm():
    # df = self.data_frames["lfm_360k_usersha1-artmbid-artname-plays"].copy()
    df = pd.read_csv('../usersha1-artmbid-artname-plays.tsv', sep='\t', header=None,
                     names=['user-mboxsha1', 'artist_mbid', 'artist-name', 'plays'])

    df['user_id'] = df['user-mboxsha1'].map(lambda x: zlib.adler32(str(x).encode('utf-8', errors='ignore')))
    df['artist_id'] = df['artist-name'].map(lambda x: zlib.adler32(str(x).encode('utf-8', errors='ignore')))

    users = df["user_id"].unique()
    items = df["artist_id"].unique()
    shape = (len(users), len(items))

    # Create indices for users and items
    user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
    item_cat = CategoricalDtype(categories=sorted(items), ordered=True)
    user_index = df["user_id"].astype(user_cat).cat.codes
    item_index = df["artist_id"].astype(item_cat).cat.codes

    print(len(user_index), user_index)
    print(len(item_index), item_index)
    print(len(df['plays'].astype(np.float32)), df['plays'].astype(np.float32))

    df['user'] = user_index
    df['item'] = item_index
    df['rating'] = df['plays'].astype(np.float32)
    df = df[['user', 'item', 'rating']]
    df.to_csv('../LFM_360K.tsv', sep='\t', header=False, index=False)


if __name__ == '__main__':
    rs = RecommenderSystem(
        os.path.join('..', 'datasets'),
        os.path.join('..', 'results')
    )

    # print_available_datasets(rs)
    # print_available_predictors(rs)
    # print_available_recommenders(rs)

    max_threads = 1

    #run_prediction_experiment(rs, max_threads)
    run_recommender_experiment(rs, max_threads)
