"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..factory import create_algorithm_factory_from_list
from .algorithms import create_biased_mf
from .algorithms import create_implicit_mf
from .algorithms import create_item_item
from .algorithms import create_pop_score
from .algorithms import create_random
from .algorithms import create_user_user
from .params import get_lenskit_params_biased_mf
from .params import get_lenskit_params_implicit_mf
from .params import get_lenskit_params_item_item
from .params import get_lenskit_params_pop_score
from .params import get_lenskit_params_random
from .params import get_lenskit_params_user_user
from .predictor import LensKitPredictor
from .recommender import LensKitRecommender

LENSKIT_API = 'LensKit'

LENSKIT_BIASED_MF = 'BiasedMF'
LENSKIT_IMPLICIT_MF = 'ImplicitMF'
LENSKIT_ITEM_ITEM = 'ItemItem'
LENSKIT_POP_SCORE = 'PopScore'
LENSKIT_RANDOM = 'Random'
LENSKIT_USER_USER = 'UserUser'


def get_lenskit_predictor_factory():
    """Gets the algorithm factory with LensKit predictors.

    Returns:
        (AlgorithmFactory) with available predictors.
    """
    return create_algorithm_factory_from_list(LENSKIT_API, [
        (LENSKIT_BIASED_MF,
         _create_predictor_biased_mf,
         get_lenskit_params_biased_mf
         ),
        (LENSKIT_IMPLICIT_MF,
         _create_predictor_implicit_mf,
         get_lenskit_params_implicit_mf
         ),
        (LENSKIT_ITEM_ITEM,
         _create_predictor_item_item,
         get_lenskit_params_item_item
         ),
        (LENSKIT_POP_SCORE,
         _create_predictor_pop_score,
         get_lenskit_params_pop_score
         ),
        (LENSKIT_USER_USER,
         _create_predictor_user_user,
         get_lenskit_params_user_user
        )
    ])


def get_lenskit_recommender_factory():
    """Gets the algorithm factory with LensKit recommenders.

    Returns:
        (AlgorithmFactory) with available recommenders.
    """
    return create_algorithm_factory_from_list(LENSKIT_API, [
        (LENSKIT_BIASED_MF,
         _create_recommender_biased_mf,
         get_lenskit_params_biased_mf
         ),
        (LENSKIT_IMPLICIT_MF,
         _create_recommender_implicit_mf,
         get_lenskit_params_implicit_mf
         ),
        (LENSKIT_ITEM_ITEM,
         _create_recommender_item_item,
         get_lenskit_params_item_item
         ),
        (LENSKIT_POP_SCORE,
         _create_recommender_pop_score,
         get_lenskit_params_pop_score
         ),
        (LENSKIT_RANDOM,
         _create_recommender_random,
         get_lenskit_params_random
         ),
        (LENSKIT_USER_USER,
         _create_recommender_user_user,
         get_lenskit_params_user_user
         )
    ])


def _create_predictor_biased_mf(params, **kwargs):
    """Creates the BiasedMF predictor.

    Args:
        params(dict): with the entries:
            features,
            iterations,
            user_reg,
            item_reg,
            damping,
            method,
            random_seed

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        (LensKitPredictor) wrapper of BiasedMF.
    """
    return LensKitPredictor(create_biased_mf(params), params, **kwargs)


def _create_predictor_implicit_mf(params, **kwargs):
    """Creates the ImplicitMF predictor.

    Args:
        params(dict): with the entries:
            features,
            iterations,
            reg,
            weight,
            use_ratings,
            method,
            random_seed

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        (LensKitPredictor) wrapper of BiasedMF.
    """
    return LensKitPredictor(create_implicit_mf(params), params, **kwargs)


def _create_predictor_item_item(params, **kwargs):
    """Creates the ItemItem predictor.

    Args:
        params(dict): with the entries:
            max_nnbrs,
            min_nbrs,
            min_sim

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rating_type(str): one of:
            explicit,
            implicit

    Returns:
        (LensKitPredictor) wrapper of BiasedMF.
    """
    return LensKitPredictor(create_item_item(params, kwargs['rating_type']), params, **kwargs)


def _create_predictor_pop_score(params, **kwargs):
    """Creates the PopScore predictor.

    Args:
        params(dict): with the entries:
            score_method

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        (LensKitPredictor) wrapper of PopScore.
    """
    return LensKitPredictor(create_pop_score(params), params, **kwargs)


def _create_predictor_user_user(params, **kwargs):
    """Creates the UserUser predictor.

    Args:
        params(dict): with the entries:
            max_nnbrs,
            min_nbrs,
            min_sim

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rating_type(str): one of:
            explicit,
            implicit

    Returns:
        (LensKitPredictor) wrapper of UserUser.
    """
    return LensKitPredictor(create_user_user(params, kwargs['rating_type']), params, **kwargs)


def _create_recommender_biased_mf(params, **kwargs):
    """Creates the BiasedMF recommender.

    Args:
        params(dict): with the entries:
            features,
            iterations,
            user_reg,
            item_reg,
            damping,
            method,
            random_seed

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        (LensKitRecommender) wrapper of BiasedMF.
    """
    return LensKitRecommender(create_biased_mf(params), params, **kwargs)


def _create_recommender_implicit_mf(params, **kwargs):
    """Creates the ImplicitMF recommender.

    Args:
        params(dict): with the entries:
            features,
            iterations,
            reg,
            weight,
            use_ratings,
            method,
            random_seed

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        (LensKitRecommender) wrapper of ImplicitMF.
    """
    return LensKitRecommender(create_implicit_mf(params), params, **kwargs)


def _create_recommender_item_item(params, **kwargs):
    """Creates the ItemItem recommender.

    Args:
        params(dict): with the entries:
            max_nnbrs,
            min_nbrs,
            min_sim

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rating_type(str): one of:
            explicit,
            implicit

    Returns:
        (LensKitRecommender) wrapper of ItemItem.
    """
    return LensKitRecommender(create_item_item(params, kwargs['rating_type']), params, **kwargs)


def _create_recommender_pop_score(params, **kwargs):
    """Creates the PopScore recommender.

    Args:
        params(dict): with the entries:
            score_method

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        (LensKitRecommender) wrapper of PopScore.
    """
    return LensKitRecommender(create_pop_score(params), params, **kwargs)


def _create_recommender_random(params, **kwargs):
    """Creates the Random recommender.

    Args:
        params(dict): with the entries:
            random_seed

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        (LensKitRecommender) wrapper of Random.
    """
    return LensKitRecommender(create_random(params), params, **kwargs)


def _create_recommender_user_user(params, **kwargs):
    """Creates the UserUser recommender.

    Args:
        params(dict): with the entries:
            max_nnbrs,
            min_nbrs,
            min_sim

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rating_type(str): one of:
            explicit,
            implicit

    Returns:
        (LensKitRecommender) wrapper of UserUser.
    """
    return LensKitRecommender(create_user_user(params, kwargs['rating_type']), params, **kwargs)
