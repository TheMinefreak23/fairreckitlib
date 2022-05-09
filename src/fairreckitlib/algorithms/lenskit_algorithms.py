"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

import time

from lenskit.algorithms.als import BiasedMF
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms.basic import PopScore
from lenskit.algorithms.basic import Random
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms.user_knn import UserUser
from seedbank import numpy_rng


def create_biased_mf(params):
    """Creates the BiasedMF algorithm.

    Args:
        params(dict): with the entries:
            features,
            iterations,
            user_reg,
            item_reg,
            damping,
            method,
            random_seed

    Returns:
        (lenskit.BiasedMF) algorithm.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    return BiasedMF(
        params['features'],
        iterations=params['iterations'],
        reg=(params['user_reg'], params['item_reg']),
        damping=params['damping'],
        bias=True,
        method=params['method'],
        rng_spec=numpy_rng(spec=params['random_seed']),
        progress=None,
        save_user_features=True
    )


def create_implicit_mf(params):
    """Creates the ImplicitMF algorithm.

    Args:
        params(dict): with the entries:
            features,
            iterations,
            reg,
            weight,
            use_ratings,
            method,
            random_seed

    Returns:
        (lenskit.ImplicitMF) algorithm.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    return ImplicitMF(
        params['features'],
        iterations=params['iterations'],
        reg=params['reg'],
        weight=params['weight'],
        use_ratings=params['use_ratings'],
        method=params['method'],
        rng_spec=numpy_rng(spec=params['random_seed']),
        progress=None,
        save_user_features=True
    )


def create_item_item(params, feedback):
    """Creates the ItemItem algorithm.

    Args:
        params(dict): with the entries:
            max_nnbrs,
            min_nbrs,
            min_sim
        feedback(str): one of:
            explicit,
            implicit

    Returns:
        (lenskit.ItemItem) algorithm.
    """
    return ItemItem(
        params['max_nnbrs'],
        min_nbrs=params['min_nbrs'],
        min_sim=params['min_sim'],
        save_nbrs=None,
        feedback=feedback
    )


def create_pop_score(params):
    """Creates the PopScore algorithm.

    Args:
        params(dict): with the entries:
            score_method

    Returns:
        (lenskit.PopScore) algorithm.
    """
    return PopScore(
        score_method=params['score_method']
    )


def create_random(params, selector):
    """Creates the Random algorithm.

    Args:
        params(dict): with the entries:
            random_seed
        selector(lenskit.CandidateSelector): selects candidate items for recommendations.

    Returns:
        (lenskit.Random) algorithm.
    """
    if params['random_seed'] is None:
        params['random_seed'] = int(time.time())

    return Random(
        selector=selector,
        rng_spec=numpy_rng(spec=params['random_seed'])
    )


def create_user_user(params, feedback):
    """Creates the UserUser algorithm.

    Args:
        params(dict): with the entries:
            max_nnbrs,
            min_nbrs,
            min_sim
        feedback(str): one of:
            explicit,
            implicit

    Returns:
        (lenskit.UserUser) algorithm.
    """
    return UserUser(
        params['max_nnbrs'],
        min_nbrs=params['min_nbrs'],
        min_sim=params['min_sim'],
        feedback=feedback
    )
