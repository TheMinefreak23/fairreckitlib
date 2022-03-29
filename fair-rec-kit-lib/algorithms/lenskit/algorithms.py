""""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from lenskit.algorithms.als import BiasedMF, ImplicitMF
from lenskit.algorithms.basic import PopScore, Random
from lenskit.algorithms.bias import Bias
from lenskit.algorithms.funksvd import FunkSVD
from lenskit.algorithms.item_knn import ItemItem
from lenskit.algorithms.user_knn import UserUser
from seedbank import numpy_rng


def create_biased_mf(params):
    return BiasedMF(
        params['features'],
        iterations=params['iterations'],
        reg=params['reg'],
        damping=params['damping'],
        bias=params['bias'],
        method=params['method'],
        rng_spec=numpy_rng(spec=params['random_seed']),
        progress=None,
        save_user_features=True
    )


def create_implicit_mf(params):
    return ImplicitMF(
        params['features'],
        iterations=params['features'],
        reg=params['reg'],
        weight=params['weight'],
        use_ratings=params['use_ratings'],
        method=params['method'],
        rng_spec=numpy_rng(spec=params['random_seed']),
        progress=None,
        save_user_features=True
    )


def create_pop_score(params):
    return PopScore(
        score_method=params['score_method']
    )


def create_random(params):
    return Random(
        selector=None,
        rng_spec=numpy_rng(spec=params['random_seed'])
    )


def create_bias(params):
    return Bias(
        items=params['items'],
        users=params['users'],
        damping=params['damping'],
    )


def create_funk_svd(params):
    return FunkSVD(
        params['features'],
        iterations=params['iterations'],
        lrate=params['lrate'],
        reg=params['reg'],
        damping=params['damping'],
        range=params['range'],
        bias=params['bias'],
        random_state=None
    )


def create_item_item(params):
    return ItemItem(
        params['max_nnbrs'],
        min_nbrs=params['min_nbrs'],
        min_sim=params['min_sim'],
        save_nbrs=params['save_nbrs'],
        feedback=params['feedback']
    )


def create_user_user(params):
    return UserUser(
        params['max_nnbrs'],
        min_nbrs=params['min_nbrs'],
        min_sim=params['min_sim'],
        feedback=params['feedback']
    )
