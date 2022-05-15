"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import math
import pytest

import numpy as np
import pandas as pd

from src.fairreckitlib.core.apis import ELLIOT_API
from src.fairreckitlib.core.config_constants import KEY_RATED_ITEMS_FILTER
from src.fairreckitlib.core.config_constants import TYPE_PREDICTION, TYPE_RECOMMENDATION
from src.fairreckitlib.core.factories import Factory, GroupFactory
from src.fairreckitlib.data.set.dataset_registry import DataRegistry
from src.fairreckitlib.model.algorithms.base_algorithm import BaseAlgorithm
from src.fairreckitlib.model.algorithms.base_predictor import BasePredictor
from src.fairreckitlib.model.algorithms.base_recommender import BaseRecommender
from src.fairreckitlib.model.algorithms.lenskit import lenskit_algorithms
from src.fairreckitlib.model.algorithms.surprise import surprise_algorithms
from src.fairreckitlib.model.model_factory import create_model_factory

data_registry = DataRegistry('tests/datasets')
dataset = data_registry.get_set('ML-100K')
sub_set = dataset.load_matrix_df().sort_values(by=['user'], axis=0)[:1000]

model_factory = create_model_factory()
NUM_THREADS = 1
non_deterministic_algos = [lenskit_algorithms.RANDOM, surprise_algorithms.NORMAL_PREDICTOR]
top_k = [1, 5, 10]

algo_kwargs = {
    'rating_scale': (dataset.get_matrix_info('rating_min'), dataset.get_matrix_info('rating_max')),
    KEY_RATED_ITEMS_FILTER: True, # recommenders only
    'num_threads': NUM_THREADS
}


def test_model_factory():
    """Test algorithms and factories in the factory to be derived from the correct base class."""
    assert isinstance(model_factory, GroupFactory), 'expected model group factory.'

    assert bool(model_factory.get_factory(TYPE_PREDICTION)), 'missing prediction models.'
    assert bool(model_factory.get_factory(TYPE_RECOMMENDATION)), 'missing recommender models.'

    for _, model_type in enumerate(model_factory.get_available_names()):
        model_type_factory = model_factory.get_factory(model_type)
        assert isinstance(model_type_factory, GroupFactory), 'expected API group factory.'

        for _, api_name in enumerate(model_type_factory.get_available_names()):
            algo_api_factory = model_type_factory.get_factory(api_name)
            assert isinstance(algo_api_factory, Factory), 'expected algorithm factory.'

            for _, algo_name in enumerate(algo_api_factory.get_available_names()):
                algo = algo_api_factory.create(algo_name, None, **algo_kwargs)

                assert isinstance(algo, BaseAlgorithm), 'expected base algorithm.'

                if model_type == TYPE_PREDICTION:
                    assert isinstance(algo, BasePredictor), 'expected base predictor.'
                elif model_type == TYPE_RECOMMENDATION:
                    assert isinstance(algo, BaseRecommender), 'expected base recommender.'
                else:
                    raise NotImplementedError('Unknown model type: ' + model_type)


def test_algorithm_creation():
    """Test creation and parameters creation to match for all algorithms."""
    for _, model_type in enumerate(model_factory.get_available_names()):
        model_type_factory = model_factory.get_factory(model_type)

        for _, algo_api_name in enumerate(model_type_factory.get_available_names()):
            algo_api_factory = model_type_factory.get_factory(algo_api_name)

            for _, algo_name in enumerate(algo_api_factory.get_available_names()):
                algo_param_defaults = algo_api_factory.create_params(algo_name).get_defaults()

                # check name / num threads
                algo = algo_api_factory.create(algo_name, algo_param_defaults, **algo_kwargs)
                assert algo.get_name() == algo_name
                assert algo.get_num_threads() == NUM_THREADS

                # check if all parameters are used by the algorithm on creation
                for param_name, _ in algo_param_defaults.items():
                    # (random) seeds are allowed to be None
                    if param_name == 'seed':
                        continue

                    # copy defaults and remove param for key error evaluation
                    algo_params = dict(algo_param_defaults)
                    del algo_params[param_name]

                    # if the parameter is in use it should raise a key error
                    with pytest.raises(KeyError):
                        algo_api_factory.create(algo_name, algo_params, **algo_kwargs)
                        print(model_type + ' algorithm \'' + algo_api_name + '.' + algo_name +
                              '\' has an unused parameter: \'' + param_name + '\'')


def test_predictors():
    """Test all predictors to obey the BasePredictor interface."""
    print('\nTesting predictor interface for all available predictors:\n')
    predictor_factory = model_factory.get_factory(TYPE_PREDICTION)

    for _, algo_api_name in enumerate(predictor_factory.get_available_names()):
        algo_api_factory = predictor_factory.get_factory(algo_api_name)

        for _, predictor_name in enumerate(algo_api_factory.get_available_names()):
            predictor = algo_api_factory.create(predictor_name, None, **algo_kwargs)

            assert_predictor_interface(algo_api_name, predictor, sub_set)


def test_recommenders():
    """Test all recommenders to obey the BaseRecommender interface."""
    print('\nTesting recommender interface for all available recommenders:\n')
    recommender_factory = model_factory.get_factory(TYPE_RECOMMENDATION)

    for _, algo_api_name in enumerate(recommender_factory.get_available_names()):
        # testing Elliot is impossible, because it is not a procedural API
        if algo_api_name == ELLIOT_API:
            continue

        algo_api_factory = recommender_factory.get_factory(algo_api_name)

        for _, recommender_name in enumerate(algo_api_factory.get_available_names()):
            recommender = algo_api_factory.create(recommender_name, None, **algo_kwargs)

            assert_recommender_interface(algo_api_name, recommender, sub_set)


def assert_algorithm_training(algorithm, train_set):
    """Assert the algorithm to obey the BaseAlgorithm training interface."""
    assert not bool(algorithm.get_users()), 'did not expect users for an untrained algorithm.'
    assert not bool(algorithm.get_items()), 'did not expect items for an untrained algorithm.'

    algorithm.train(train_set)

    assert len(algorithm.get_users()) == len(train_set['user'].unique()), \
        'expected algorithm\'s unique users to be the same as in the train set.'
    assert len(algorithm.get_items()) == len(train_set['item'].unique()), \
        'expected algorithm\'s unique items to be the same as in the train set.'


def assert_predictor_interface(api_name, predictor, train_set):
    """Assert the predictor to obey the BasePredictor interface."""
    print('Training predictor \'' + api_name + '.' + predictor.get_name() + '\'')

    assert_algorithm_training(predictor, train_set)

    print('Testing predictor \'' + api_name + '.' + predictor.get_name() + '\'')

    min_user = predictor.get_users().min()
    max_user = predictor.get_users().max()

    min_item = predictor.get_items().min()
    max_item = predictor.get_items().max()

    # test user edge cases for all items
    for _, item in enumerate(predictor.get_items()):
        # test (singular) min user edge case
        prediction = predictor.predict(min_user - 1, item)
        assert math.isnan(prediction), 'expected failure because user does not exist.'

        # test (singular) max user edge case
        prediction = predictor.predict(max_user + 1, item)
        assert math.isnan(prediction), 'expected failure because user does not exist.'

        # test (batching) min and max user edge cases
        pairs = pd.DataFrame({'user': [min_user - 1, max_user + 1], 'item': [item, item]})
        pairs = predictor.predict_batch(pairs)
        assert len(pairs) == 0, 'expected an empty prediction dataframe.'

    # test item edge cases for all users
    for _, user in enumerate(predictor.get_users()):
        # test (singular) min item edge case
        prediction = predictor.predict(user, min_item - 1)
        assert math.isnan(prediction), 'expected failure because item does not exist.'

        # test (singular) max item edge case
        prediction = predictor.predict(user, max_item + 1)
        assert math.isnan(prediction), 'expected failure because item does not exist.'

        # test (batching) min and max item edge cases
        pairs = pd.DataFrame({'user': [user, user], 'item': [min_item - 1, max_item + 1]})
        pairs = predictor.predict_batch(pairs)
        assert len(pairs) == 0, 'expected an empty prediction dataframe.'

    # test all user / item combinations
    for _, user in enumerate(predictor.get_users()):
        # test (singular) items per user
        for _, item in enumerate(predictor.get_items()):
            prediction = predictor.predict(user, item)
            assert isinstance(prediction, float), \
                'expected predicted rating to be a float or NaN.'

            # skip item when the predictor is non-deterministic
            if predictor.get_name() in non_deterministic_algos:
                continue

            # test (singular) if prediction is deterministic
            second_prediction = predictor.predict(user, item)
            if math.isnan(prediction):
                assert (math.isnan(second_prediction)), \
                    'expected the same prediction to be NaN again.'
            else:
                assert prediction == second_prediction, \
                    'expected the first and second prediction to be the same.'

        # test (batching) items per user
        num_items = len(predictor.get_items())
        user_item_pairs = pd.DataFrame({
            'user': np.full(num_items, user),
            'item': predictor.get_items()
        })

        pairs = predictor.predict_batch(user_item_pairs)
        assert pd.api.types.is_numeric_dtype(pairs['prediction']), \
            'expected predicted ratings to be numeric (float) or NaN.'
        assert len(pairs) == num_items, \
            'expected prediction for every item.'

        # test (batching) if prediction is deterministic
        if predictor.get_name() not in non_deterministic_algos:
            second_pairs = predictor.predict_batch(user_item_pairs)

            assert len(second_pairs) == num_items, \
                'expected second prediction for every item.'

            second_pairs.rename(columns={'prediction': 'second_prediction'}, inplace=True)
            merged_pairs = pd.merge(pairs, second_pairs, how='left', on=['user', 'item'])

            assert len(merged_pairs) == num_items, \
                'expected merge of the first and second pairs to be the same.'
            assert merged_pairs['prediction'].equals(merged_pairs['second_prediction']), \
                'expected first and second prediction to be the same.'


def assert_recommender_interface(api_name, recommender, train_set):
    """Assert the recommender to obey the BaseRecommender interface."""
    print('Training recommender \'' + api_name + '.' + recommender.get_name() + '\'')

    assert_algorithm_training(recommender, train_set)

    print('Testing recommender \'' + api_name + '.' + recommender.get_name() + '\'')

    min_user = recommender.get_users().min()
    max_user = recommender.get_users().max()

    for _, num_items in enumerate(top_k):
        # test (singular) min and max user edge case
        assert len(recommender.recommend(min_user - 1, num_items=num_items)) == 0, \
            'expected no recommendations because user does not exist.'
        assert len(recommender.recommend(max_user + 1, num_items=num_items)) == 0, \
            'expected no recommendations because user does not exist.'

        # test (singular) user
        for _, user in enumerate(recommender.get_users()):
            recs = recommender.recommend(user, num_items=num_items)

            assert_single_user_recs(recs, num_items)

            # test (singular) user to be deterministic
            if recommender.get_name() not in non_deterministic_algos:
                second_recs = recommender.recommend(user, num_items=num_items)

                assert_recs_are_deterministic(recs, second_recs)

        # test (batch) min and max user edge case
        users = [min_user - 1, max_user + 1]
        assert len(recommender.recommend_batch(users, num_items=num_items)) == 0, \
            'expected no recommendations because users do not exist.'

        # test (batch) users but with singular user
        for _, user in enumerate(recommender.get_users()):
            recs = recommender.recommend_batch([user], num_items=num_items)

            assert_single_user_recs(recs, num_items)
            assert len(recs['user'].unique()) == 1, \
                'expected recommendations for only one user.'
            assert recs['rank'].equals(pd.Series(np.arange(1, 1 + num_items))), \
                'expected rankings to be in ascending order from [1..num_items].'

            # test (batch) users but with singular user to be deterministic
            if recommender.get_name() not in non_deterministic_algos:
                second_recs = recommender.recommend(user, num_items=num_items)

                assert_recs_are_deterministic(recs, second_recs)

        # test (batch) users
        users = recommender.get_users()
        num_users = len(users)

        recs = recommender.recommend_batch(users, num_items=num_items)
        assert len(recs) == num_users * num_items, \
            'expected (num_users * num_items) recommendations.'
        assert len(recs['user'].unique()) == num_users, \
            'expected item recommendations for all users.'
        assert len(recs['rank'].unique()) == num_items, \
            'expected num_items rankings for all users.'

        # test (batch) users to be deterministic
        if recommender.get_name() not in non_deterministic_algos:
            second_recs = recommender.recommend_batch(users, num_items=num_items)
            assert len(recs) == len(second_recs), \
                'expected the same amount of item recommendations.'
            assert_recs_are_deterministic(recs, second_recs)


def assert_single_user_recs(recs, num_items):
    """Assert recommendations dataframe for a single user."""
    assert len(recs) == num_items, \
        'expected num_items recommendations.'
    assert len(recs['item'].unique()) == num_items, \
        'expected unique item recommendations.'
    assert recs['score'].is_monotonic_decreasing, \
        'expected recommendations to be in decreasing order.'


def assert_recs_are_deterministic(recs, second_recs):
    """Assert recommendations dataframes are deterministic."""
    assert recs['item'].equals(second_recs['item']), \
        'expected the first and second recommendation items to be the same.'
    assert recs['score'].equals(second_recs['score']), \
        'expected the first and second recommendation scores to be the same.'
