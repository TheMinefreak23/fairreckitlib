"""This module tests the algorithm interface of predictors/recommenders.

Classes:

    DummyPredictor: dummy predictor implementation to test various errors.
    DummyRecommender: dummy recommender implementation to test various errors.

Functions:

    test_algorithm_creation: test if algorithms and params are created correctly.
    test_predictor_interface_errors: test interface errors for not implemented functions.
    test_predictors: test all available predictor algorithms.
    test_recommender_interface_errors: test interface errors for not implemented functions.
    test_recommenders: test all available recommender algorithms.
    assert_algorithm_training: assert algorithm training functionality.
    assert_frame_headers: assert frame headers of returned rating dataframes.
    assert_predictor_interface: assert base interface of a predictor.
    assert_predictor_edge_cases: assert edge case predictions.
    assert_predictor_singular_user: assert prediction for single user.
    assert_predictor_multi_users: assert prediction for a batch of users.
    assert_recommender_interface: assert base interface of a recommender.
    assert_recommender_edge_cases: assert edge case recommendations.
    assert_recommender_singular_user: assert recommender for single user.
    assert_recommender_multi_users: assert recommender for a batch of users.
    assert_single_user_recs: assert item recommendations of a single user.
    assert_recs_are_deterministic: assert if item recommendations are deterministic.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from src.fairreckitlib.core.core_constants import KEY_RATED_ITEMS_FILTER
from src.fairreckitlib.core.core_constants import TYPE_PREDICTION, TYPE_RECOMMENDATION
from src.fairreckitlib.data.set.dataset_config import DATASET_RATINGS_EXPLICIT
from src.fairreckitlib.model.algorithms.base_algorithm import BaseAlgorithm
from src.fairreckitlib.model.algorithms.base_predictor import BasePredictor, Predictor
from src.fairreckitlib.model.algorithms.base_recommender import BaseRecommender, Recommender
from src.fairreckitlib.model.algorithms.lenskit import lenskit_algorithms
from src.fairreckitlib.model.algorithms.surprise import surprise_algorithms
from src.fairreckitlib.model.model_factory import create_model_factory
from .conftest import NUM_THREADS
from .test_model_algorithm_matrices import DummyMatrix, create_algo_matrix

PREDICTION_FRAME_HEADERS = ['user', 'item', 'prediction']
REC_FRAME_HEADER_SINGLE = ['item', 'score']
REC_FRAME_HEADER_BATCH = ['rank', 'user'] + REC_FRAME_HEADER_SINGLE

model_factory = create_model_factory()
non_deterministic_algos = [lenskit_algorithms.RANDOM, surprise_algorithms.NORMAL_PREDICTOR]
top_k = [1, 5, 10]

algo_kwargs = {
    KEY_RATED_ITEMS_FILTER: True, # recommenders only
    'num_threads': NUM_THREADS,
    'rating_type': DATASET_RATINGS_EXPLICIT # used by lenskit KNN algorithms
}


class DummyPredictor(Predictor):
    """Dummy predictor to test various errors."""

    def __init__(self, name: str, params: Dict[str, Any], **kwargs):
        """Construct dummy predictor."""
        Predictor.__init__(self, name, params, 1)
        self.train_error = kwargs.get('train_error')
        self.test_error = kwargs.get('test_error')
        if kwargs.get('const_error', False):
            raise kwargs['const_error']()

    def on_predict(self, user: int, item: int) -> float:
        """Raise error on predicting."""
        if self.test_error is None:
            return Predictor.on_predict(self, user, item)

        raise self.test_error()

    def on_train(self, train_set: Any) -> None:
        """Raise error or fake training."""
        if self.train_error is None:
            if self.test_error is not None:
                return # succeed training

            Predictor.on_train(self, train_set)

        raise self.train_error()


class DummyRecommender(Recommender):
    """Dummy recommender to test not implemented and construction errors."""

    def __init__(self, name: str, params: Dict[str, Any], **kwargs):
        """Construct dummy recommender."""
        Recommender.__init__(self, name, params, 1, True)
        self.train_error = kwargs.get('train_error')
        self.test_error = kwargs.get('test_error')
        if kwargs.get('const_error', False):
            raise kwargs['const_error']()

    def on_recommend(self, user: int, num_items: int) -> pd.DataFrame:
        """Raise error on recommending."""
        if self.test_error is None:
            return Recommender.on_recommend(self, user, num_items)

        raise self.test_error()

    def on_train(self, train_set: Any) -> None:
        """Raise error or fake training."""
        if self.train_error is None:
            if self.test_error is not None:
                return # succeed training

            Recommender.on_train(self, train_set)

        raise self.train_error()


def test_algorithm_creation() -> None:
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


def test_predictor_interface_errors() -> None:
    """Test predictor interface errors for not implemented functions."""
    predictor = DummyPredictor('predictor', {}, **algo_kwargs)

    pytest.raises(NotImplementedError, predictor.on_train, None)
    pytest.raises(NotImplementedError, predictor.on_predict, 0, 0)


def test_predictors() -> None:
    """Test all predictors to obey the BasePredictor interface."""
    print('\nTesting predictor interface for all available predictors:\n')
    predictor_factory = model_factory.get_factory(TYPE_PREDICTION)

    for algo_api_name in predictor_factory.get_available_names():
        algo_api_factory = predictor_factory.get_factory(algo_api_name)

        for predictor_name in algo_api_factory.get_available_names():
            predictor = algo_api_factory.create(predictor_name, None, **algo_kwargs)

            assert_predictor_interface(algo_api_name, predictor)


def test_recommender_interface_errors() -> None:
    """Test recommender interface errors for not implemented functions."""
    recommender = DummyRecommender('recommender', {}, **algo_kwargs)

    pytest.raises(NotImplementedError, recommender.on_train, None)
    pytest.raises(NotImplementedError, recommender.on_recommend, 0, 0)


def test_recommenders() -> None:
    """Test all recommenders to obey the BaseRecommender interface."""
    print('\nTesting recommender interface for all available recommenders:\n')
    recommender_factory = model_factory.get_factory(TYPE_RECOMMENDATION)

    for algo_api_name in recommender_factory.get_available_names():
        algo_api_factory = recommender_factory.get_factory(algo_api_name)

        for recommender_name in algo_api_factory.get_available_names():
            recommender = algo_api_factory.create(recommender_name, None, **algo_kwargs)

            assert_recommender_interface(algo_api_name, recommender)


def assert_algorithm_training(api_name: str, algorithm: BaseAlgorithm) -> None:
    """Assert the algorithm to obey the BaseAlgorithm training interface."""
    assert not bool(algorithm.get_train_set()), \
        'did not expect train set for an untrained algorithm.'

    # test failure for incorrect matrix training
    pytest.raises(TypeError, algorithm.train, DummyMatrix())

    train_set = create_algo_matrix(api_name)
    algorithm.train(train_set)

    assert algorithm.get_train_set() == train_set, \
        'expected algorithm train set to be the same as the input train set.'


def assert_frame_headers(dataframe: pd.DataFrame, expected_header: List[str]) -> None:
    """Assert the dataframe to have the specified expected header."""
    for column_name in dataframe.columns.values:
        assert column_name in expected_header, 'expected one of ' + \
            str(expected_header) + ' got: ' + str(column_name)


def assert_predictor_interface(api_name: str, predictor: BasePredictor) -> None:
    """Assert the predictor to obey the BasePredictor interface."""
    print('Testing predictor \'' + api_name + '.' + predictor.get_name() + '\'')

    # test failure on predictions when untrained
    pytest.raises(RuntimeError, predictor.predict, 0, 0)
    pytest.raises(RuntimeError, predictor.predict_batch, pd.DataFrame())

    assert_algorithm_training(api_name, predictor)

    assert_predictor_edge_cases(predictor)
    assert_predictor_singular_user(predictor)
    assert_predictor_multi_users(predictor)


def assert_predictor_edge_cases(predictor: BasePredictor) -> None:
    """Assert the predictor to return impossible predictions for unknown users/items."""
    min_user = predictor.get_train_set().get_users().min()
    max_user = predictor.get_train_set().get_users().max()

    min_item = predictor.get_train_set().get_items().min()
    max_item = predictor.get_train_set().get_items().max()

    # test user edge cases for all items
    for item in predictor.get_train_set().get_items():
        # test (singular) min user edge case
        prediction = predictor.predict(min_user - 1, item)
        assert math.isnan(prediction), 'expected failure because user does not exist.'

        # test (singular) max user edge case
        prediction = predictor.predict(max_user + 1, item)
        assert math.isnan(prediction), 'expected failure because user does not exist.'

        # test (batching) min and max user edge cases
        pairs = pd.DataFrame({'user': [min_user - 1, max_user + 1], 'item': [item, item]})
        pairs = predictor.predict_batch(pairs)
        assert pairs['prediction'].isna().all(), \
            'expected prediction frame with NaN values for unknown users'

    # test item edge cases for all users
    for user in predictor.get_train_set().get_users():
        # test (singular) min item edge case
        prediction = predictor.predict(user, min_item - 1)
        assert math.isnan(prediction), 'expected failure because item does not exist.'

        # test (singular) max item edge case
        prediction = predictor.predict(user, max_item + 1)
        assert math.isnan(prediction), 'expected failure because item does not exist.'

        # test (batching) min and max item edge cases
        pairs = pd.DataFrame({'user': [user, user], 'item': [min_item - 1, max_item + 1]})
        pairs = predictor.predict_batch(pairs)
        assert pairs['prediction'].isna().all(), \
            'expected prediction frame with NaN values for unknown items'


def assert_predictor_singular_user(predictor: BasePredictor) -> None:
    """Assert the predictor's (batch) predictions for a single user."""
    # test all user / item combinations
    for user in predictor.get_train_set().get_users():
        # test (singular) items per user
        for item in predictor.get_train_set().get_items():
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
        num_items = len(predictor.get_train_set().get_items())
        user_item_pairs = pd.DataFrame({
            'user': np.full(num_items, user),
            'item': predictor.get_train_set().get_items()
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


def assert_predictor_multi_users(predictor: BasePredictor) -> None:
    """Assert the predictor's batch predictions for multiple users."""
    user_item_pairs = pd.DataFrame(columns=['user', 'item'])
    num_items = len(predictor.get_train_set().get_items())
    num_users = len(predictor.get_train_set().get_users())

    # test (batch) users
    for user in predictor.get_train_set().get_users():
        user_item_pairs = user_item_pairs.append(pd.DataFrame({
            'user': np.full(num_items, user),
            'item': predictor.get_train_set().get_items()
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


def assert_recommender_interface(api_name: str, recommender: BaseRecommender) -> None:
    """Assert the recommender to obey the BaseRecommender interface."""
    print('Testing recommender \'' + api_name + '.' + recommender.get_name() + '\'')

    # test failure on item recommendations when untrained
    pytest.raises(RuntimeError, recommender.recommend, 0)
    pytest.raises(RuntimeError, recommender.recommend_batch, [0])

    assert_algorithm_training(api_name, recommender)

    for num_items in top_k:
        assert_recommender_edge_cases(recommender, num_items)
        assert_recommender_singular_user(recommender, num_items)
        assert_recommender_multi_users(recommender, num_items)


def assert_recommender_edge_cases(recommender: BaseRecommender, num_items: int) -> None:
    """Assert the recommender to produce no recommendations for unknown users."""
    min_user = recommender.get_train_set().get_users().min()
    max_user = recommender.get_train_set().get_users().max()

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


def assert_recommender_singular_user(recommender: BaseRecommender, num_items: int) -> None:
    """Assert the recommender's (batch) item recommendations for a single user."""
    # test (singular) user
    for user in recommender.get_train_set().get_users():
        recs = recommender.recommend(user, num_items=num_items)

        assert_single_user_recs(recs, num_items, REC_FRAME_HEADER_SINGLE)

        # test (singular) user to be deterministic
        if recommender.get_name() not in non_deterministic_algos:
            second_recs = recommender.recommend(user, num_items=num_items)

            assert_recs_are_deterministic(recs, second_recs)

    # test (batch) users but with singular user
    for user in recommender.get_train_set().get_users():
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


def assert_recommender_multi_users(recommender: BaseRecommender, num_items: int) -> None:
    """Assert the recommender's batch item recommendations for multiple users."""
    # test (batch) users
    users = recommender.get_train_set().get_users()
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


def assert_single_user_recs(recs: pd.DataFrame, num_items: int, header: List[str]) -> None:
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


def assert_recs_are_deterministic(recs: pd.DataFrame, second_recs: pd.DataFrame) -> None:
    """Assert recommendations dataframes are deterministic."""
    assert recs['item'].equals(second_recs['item']), \
        'expected the first and second recommendation items to be the same.'
    assert recs['score'].equals(second_recs['score']), \
        'expected the first and second recommendation scores to be the same.'
