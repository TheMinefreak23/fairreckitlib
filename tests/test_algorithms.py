"""This module tests the algorithm interface of predictors/recommenders.

Functions:

    test_algorithm_subset: test if subset is sufficient for testing.
    test_model_factory: test if created factories are correct.
    test_algorithm_creation: test if algorithms and params are created correctly.
    test_predictors: test predictor algorithms.
    test_recommenders: test recommender algorithms.
    assert_algorithm_training: assert train set.
    assert_frame_headers: assert headers.
    assert_predictor_interface: assert predictor.
    assert_predictor_edge_cases: assert edge case returns.
    assert_predictor_singular_user: assert prediction for single user.
    assert_predictor_multi_users: assert prediction for multiple users.
    assert_recommender_interface: assert recommender.
    assert_recommender_edge_cases: assert edge case returns.
    assert_recommender_singular_user: assert recommender for single user.
    assert_recommender_multi_users: assert recommender for multiple users.
    assert_single_user_recs: assert recommendations for single user.
    assert_recs_are_deterministic: assert recommendations are deterministic.

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
PREDICTION_FRAME_HEADERS = ['user', 'item', 'prediction']
REC_FRAME_HEADER_SINGLE = ['item', 'score']
REC_FRAME_HEADER_BATCH = ['rank', 'user'] + REC_FRAME_HEADER_SINGLE

non_deterministic_algos = [lenskit_algorithms.RANDOM, surprise_algorithms.NORMAL_PREDICTOR]
top_k = [1, 5, 10]

algo_kwargs = {
    'rating_scale': (dataset.get_matrix_info('rating_min'), dataset.get_matrix_info('rating_max')),
    KEY_RATED_ITEMS_FILTER: True, # recommenders only
    'num_threads': NUM_THREADS
}


def test_algorithm_subset():
    """Test if the generated subset is sufficient to verify the other algorithm tests."""
    assert len(sub_set['user'].unique()) > 1, 'expected more than one user for testing algorithms.'
    assert len(sub_set['item'].unique()) > 1, 'expected more than one item for testing algorithms.'


def test_model_factory():
    """Test algorithms and factories in the factory to be derived from the correct base class."""
    assert isinstance(model_factory, GroupFactory), 'expected model group factory.'

    assert bool(model_factory.get_factory(TYPE_PREDICTION)), 'missing prediction models.'
    assert bool(model_factory.get_factory(TYPE_RECOMMENDATION)), 'missing recommender models.'

    for model_type in model_factory.get_available_names():
        model_type_factory = model_factory.get_factory(model_type)
        assert isinstance(model_type_factory, GroupFactory), 'expected API group factory.'

        for algo_api_name in model_type_factory.get_available_names():
            algo_api_factory = model_type_factory.get_factory(algo_api_name)
            assert isinstance(algo_api_factory, Factory), 'expected algorithm factory.'

            for algo_name in algo_api_factory.get_available_names():
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
    for model_type in model_factory.get_available_names():
        model_type_factory = model_factory.get_factory(model_type)

        for algo_api_name in model_type_factory.get_available_names():
            algo_api_factory = model_type_factory.get_factory(algo_api_name)

            for algo_name in algo_api_factory.get_available_names():
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

    for algo_api_name in predictor_factory.get_available_names():
        algo_api_factory = predictor_factory.get_factory(algo_api_name)

        for predictor_name in algo_api_factory.get_available_names():
            predictor = algo_api_factory.create(predictor_name, None, **algo_kwargs)

            assert_predictor_interface(algo_api_name, predictor, sub_set)


def test_recommenders():
    """Test all recommenders to obey the BaseRecommender interface."""
    print('\nTesting recommender interface for all available recommenders:\n')
    recommender_factory = model_factory.get_factory(TYPE_RECOMMENDATION)

    for algo_api_name in recommender_factory.get_available_names():
        # testing Elliot is impossible, because it is not a procedural API
        if algo_api_name == ELLIOT_API:
            continue

        algo_api_factory = recommender_factory.get_factory(algo_api_name)

        for recommender_name in algo_api_factory.get_available_names():
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


def assert_frame_headers(dataframe, expected_header):
    """Assert the dataframe to have the specified expected header."""
    for column_name in dataframe.columns.values:
        assert column_name in expected_header, 'expected one of ' + \
            str(expected_header) + ' got: ' + str(column_name)


def assert_predictor_interface(api_name, predictor, train_set):
    """Assert the predictor to obey the BasePredictor interface."""
    print('Training predictor \'' + api_name + '.' + predictor.get_name() + '\'')

    assert_algorithm_training(predictor, train_set)

    print('Testing predictor \'' + api_name + '.' + predictor.get_name() + '\'')

    assert_predictor_edge_cases(predictor)
    assert_predictor_singular_user(predictor)
    assert_predictor_multi_users(predictor)


def assert_predictor_edge_cases(predictor):
    """Assert the predictor to return impossible predictions for unknown users/items."""
    min_user = predictor.get_users().min()
    max_user = predictor.get_users().max()

    min_item = predictor.get_items().min()
    max_item = predictor.get_items().max()

    # test user edge cases for all items
    for item in predictor.get_items():
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
    for user in predictor.get_users():
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


def assert_predictor_singular_user(predictor):
    """Assert the predictor's (batch) predictions for a single user."""
    # test all user / item combinations
    for user in predictor.get_users():
        # test (singular) items per user
        for item in predictor.get_items():
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
        assert_frame_headers(pairs, PREDICTION_FRAME_HEADERS)
        assert pd.api.types.is_numeric_dtype(pairs['prediction']), \
            'expected predicted ratings to be numeric (float) or NaN.'
        assert len(pairs) == num_items, \
            'expected prediction for every item.'

        # test (batching) if prediction is deterministic
        if predictor.get_name() not in non_deterministic_algos:
            second_pairs = predictor.predict_batch(user_item_pairs)
            assert_frame_headers(second_pairs, PREDICTION_FRAME_HEADERS)
            assert len(second_pairs) == num_items, \
                'expected second prediction for every item.'

            second_pairs.rename(columns={'prediction': 'second_prediction'}, inplace=True)
            merged_pairs = pd.merge(pairs, second_pairs, how='left', on=['user', 'item'])

            assert len(merged_pairs) == num_items, \
                'expected merge of the first and second pairs to be the same.'
            assert merged_pairs['prediction'].equals(merged_pairs['second_prediction']), \
                'expected first and second prediction to be the same.'


def assert_predictor_multi_users(predictor):
    """Assert the predictor's batch predictions for multiple users."""
    user_item_pairs = pd.DataFrame(columns=['user', 'item'])
    num_items = len(predictor.get_items())
    num_users = len(predictor.get_users())

    # test (batch) users
    for user in predictor.get_users():
        user_item_pairs = user_item_pairs.append(pd.DataFrame({
            'user': np.full(num_items, user),
            'item': predictor.get_items()
        }), ignore_index=True)

    pairs = predictor.predict_batch(user_item_pairs)
    assert_frame_headers(pairs, PREDICTION_FRAME_HEADERS)
    assert len(pairs) == num_users * num_items, \
        'expected predictions for all user/item combinations.'
    assert pd.api.types.is_numeric_dtype(pairs['prediction']), \
        'expected predictions to be numeric (float) or NaN.'

    # test (batch) users to be deterministic
    if predictor.get_name() in non_deterministic_algos:
        return

    second_pairs = predictor.predict_batch(user_item_pairs)
    assert_frame_headers(second_pairs, PREDICTION_FRAME_HEADERS)
    assert len(pairs) == len(second_pairs), \
        'expected the same amount of predictions as the first time.'
    assert pairs['prediction'].equals(second_pairs['prediction']), \
        'expected the first and second predictions to be the same.'


def assert_recommender_interface(api_name, recommender, train_set):
    """Assert the recommender to obey the BaseRecommender interface."""
    print('Testing recommender \'' + api_name + '.' + recommender.get_name() + '\'')

    assert_algorithm_training(recommender, train_set)

    for num_items in top_k:
        assert_recommender_edge_cases(recommender, num_items)
        assert_recommender_singular_user(recommender, num_items)
        assert_recommender_multi_users(recommender, num_items)


def assert_recommender_edge_cases(recommender, num_items):
    """Assert the recommender to produce no recommendations for unknown users."""
    min_user = recommender.get_users().min()
    max_user = recommender.get_users().max()

    # test (singular) min and max user edge case
    recs = recommender.recommend(min_user - 1, num_items=num_items)
    assert_frame_headers(recs, REC_FRAME_HEADER_SINGLE)
    assert len(recs) == 0,  'expected no recommendations because user does not exist.'
    recs = recommender.recommend(max_user + 1, num_items=num_items)
    assert_frame_headers(recs, REC_FRAME_HEADER_SINGLE)
    assert len(recs) == 0, 'expected no recommendations because user does not exist.'

    # test (batch) min and max user edge case
    users = [min_user - 1, max_user + 1]
    recs = recommender.recommend_batch(users, num_items=num_items)
    assert_frame_headers(recs, REC_FRAME_HEADER_BATCH)
    assert len(recs) == 0, 'expected no recommendations because users do not exist.'


def assert_recommender_singular_user(recommender, num_items):
    """Assert the recommender's (batch) item recommendations for a single user."""
    # test (singular) user
    for user in recommender.get_users():
        recs = recommender.recommend(user, num_items=num_items)

        assert_single_user_recs(recs, num_items, REC_FRAME_HEADER_SINGLE)

        # test (singular) user to be deterministic
        if recommender.get_name() not in non_deterministic_algos:
            second_recs = recommender.recommend(user, num_items=num_items)

            assert_recs_are_deterministic(recs, second_recs)

    # test (batch) users but with singular user
    for user in recommender.get_users():
        recs = recommender.recommend_batch([user], num_items=num_items)

        assert_single_user_recs(recs, num_items, REC_FRAME_HEADER_BATCH)
        assert len(recs['user'].unique()) == 1, \
            'expected recommendations for only one user.'
        assert recs['rank'].equals(pd.Series(np.arange(1, 1 + num_items))), \
            'expected rankings to be in ascending order from [1..num_items].'

        # test (batch) users but with singular user to be deterministic
        if recommender.get_name() not in non_deterministic_algos:
            second_recs = recommender.recommend(user, num_items=num_items)

            assert_recs_are_deterministic(recs, second_recs)


def assert_recommender_multi_users(recommender, num_items):
    """Assert the recommender's batch item recommendations for multiple users."""
    # test (batch) users
    users = recommender.get_users()
    num_users = len(users)

    recs = recommender.recommend_batch(users, num_items=num_items)
    assert_frame_headers(recs, REC_FRAME_HEADER_BATCH)
    assert len(recs) == num_users * num_items, \
        'expected (num_users * num_items) recommendations.'
    assert len(recs['user'].unique()) == num_users, \
        'expected item recommendations for all users.'
    assert len(recs['rank'].unique()) == num_items, \
        'expected num_items rankings for all users.'

    # test (batch) users to be deterministic
    if recommender.get_name() not in non_deterministic_algos:
        second_recs = recommender.recommend_batch(users, num_items=num_items)
        assert_frame_headers(second_recs, REC_FRAME_HEADER_BATCH)
        assert len(recs) == len(second_recs), \
            'expected the same amount of item recommendations.'
        assert_recs_are_deterministic(recs, second_recs)


def assert_single_user_recs(recs, num_items, header):
    """Assert recommendations dataframe for a single user."""
    assert_frame_headers(recs, header)
    assert len(recs) == num_items, \
        'expected num_items recommendations.'
    assert len(recs['item'].unique()) == num_items, \
        'expected unique item recommendations.'
    assert pd.api.types.is_numeric_dtype(recs['score']), \
        'expected recommendations to be numeric (float) or NaN.'
    assert recs['score'].is_monotonic_decreasing, \
        'expected recommendations to be in decreasing order.'


def assert_recs_are_deterministic(recs, second_recs):
    """Assert recommendations dataframes are deterministic."""
    assert recs['item'].equals(second_recs['item']), \
        'expected the first and second recommendation items to be the same.'
    assert recs['score'].equals(second_recs['score']), \
        'expected the first and second recommendation scores to be the same.'
