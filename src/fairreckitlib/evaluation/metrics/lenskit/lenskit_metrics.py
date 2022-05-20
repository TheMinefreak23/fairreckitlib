"""This module contains name constants and creation wrappers for implemented lenskit algorithms.

Functions:

    create_biased_mf: create lenskit BiasedMF algorithm.
    create_implicit_mf: create lenskit ImplicitMF algorithm.
    create_item_item: create lenskit ItemItem algorithm.
    create_pop_score: create lenskit PopScore algorithm.
    create_random: create lenskit Random algorithm.
    create_user_user: create lenskit UserUser algorithm.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time
from typing import Any, Dict

from lenskit.algorithms import CandidateSelector
from lenskit.algorithms.als import BiasedMF, ImplicitMF
from lenskit.algorithms.basic import PopScore, Random
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms.user_knn import UserUser
from seedbank import numpy_rng

BIASED_MF = 'BiasedMF'
IMPLICIT_MF = 'ImplicitMF'
ITEM_ITEM = 'ItemItem'
POP_SCORE = 'PopScore'
RANDOM = 'Random'
USER_USER = 'UserUser'


def create_biased_mf(params: Dict[str, Any]) -> BiasedMF:
    """Create the lenskit BiasedMF algorithm.

    Args:
        params: containing the following name-value pairs:
            features(int): the number of features to train.
            iterations(int): the number of iterations to train.
            user_reg(float): the regularization factor for users.
            item_reg(float): the regularization factor for items.
            damping(float): damping factor for the underlying bias.
            method(str): the solver to use ('cd' or 'lu').
            random_seed(int): the random seed or None for the current time as seed.

    Returns:
        the lenskit BiasedMF algorithm.
    """
    if params['seed'] is None:
        params['seed'] = int(time.time())

    return BiasedMF(
        params['features'],
        iterations=params['iterations'],
        reg=(params['user_reg'], params['item_reg']),
        damping=params['damping'],
        bias=True,
        method=params['method'],
        rng_spec=numpy_rng(spec=params['seed']),
        progress=None,
        save_user_features=True
    )


def create_implicit_mf(params: Dict[str, Any]) -> ImplicitMF:
    """Create the lenskit ImplicitMF algorithm.

    Args:
        params: containing the following name-value pairs:
            features(int): the number of features to train.
            iterations(int): the number of iterations to train.
            reg(float): the regularization factor.
            weight(flot): the scaling weight for positive samples.
            use_ratings(bool): whether to use the rating column or treat
                every rated user-item pair as having a rating of 1.
            method(str): the training method ('cg' or 'lu').
            random_seed(int): the random seed or None for the current time as seed.

    Returns:
        the lenskit ImplicitMF algorithm.
    """
    if params['seed'] is None:
        params['seed'] = int(time.time())

    return ImplicitMF(
        params['features'],
        iterations=params['iterations'],
        reg=params['reg'],
        weight=params['weight'],
        use_ratings=params['use_ratings'],
        method=params['method'],
        rng_spec=numpy_rng(spec=params['seed']),
        progress=None,
        save_user_features=True
    )


def create_item_item(params: Dict[str, Any]) -> ItemItem:
    """Create the lenskit ItemItem algorithm.

    Args:
        params: containing the following name-value pairs:
            max_neighbors(int): the maximum number of neighbors for scoring each item.
            min_neighbors(int): the minimum number of neighbors for scoring each item.
            min_similarity(float): minimum similarity threshold for considering a neighbor.
            feedback(str): control how feedback should be interpreted ('explicit' or 'implicit').

    Returns:
        the lenskit ItemItem algorithm.
    """
    return ItemItem(
        params['max_neighbors'],
        min_nbrs=params['min_neighbors'],
        min_sim=params['min_similarity'],
        save_nbrs=None,
        feedback=params['feedback']
    )


def create_pop_score(params: Dict[str, Any]) -> PopScore:
    """Create the lenskit PopScore algorithm.

    Args:
        params: containing the following name-value pairs:
            score_method(str): for computing popularity scores ('quantile', 'rank' or 'count').

    Returns:
        the lenskit PopScore algorithm.
    """
    return PopScore(
        score_method=params['score_method']
    )


def create_random(params: Dict[str, Any], selector: CandidateSelector) -> Random:
    """Create the lenskit Random algorithm.

    Args:
        params: containing the following name-value pairs:
            random_seed(int): the random seed or None for the current time as seed.
        selector: that selects candidate items for recommendations.

    Returns:
        the lenskit Random algorithm.
    """
    if params['seed'] is None:
        params['seed'] = int(time.time())

    return Random(
        selector=selector,
        rng_spec=numpy_rng(spec=params['seed'])
    )


def create_user_user(params: Dict[str, Any]) -> UserUser:
    """Create the lenskit UserUser algorithm.

    Args:
        params: containing the following name-value pairs:
            max_neighbors(int): the maximum number of neighbors for scoring each item.
            min_neighbors(int): the minimum number of neighbors for scoring each item.
            min_similarity(float): minimum similarity threshold for considering a neighbor.
            feedback(str): control how feedback should be interpreted ('explicit' or 'implicit').

    Returns:
        the lenskit UserUser algorithm.
    """
    return UserUser(
        params['max_neighbors'],
        min_nbrs=params['min_neighbors'],
        min_sim=params['min_similarity'],
        feedback=params['feedback']
    )
