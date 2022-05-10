"""
This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from lenskit.algorithms.basic import AllItemsCandidateSelector
from lenskit.algorithms.basic import UnratedItemCandidateSelector
from lenskit.algorithms.ranking import TopN

from ....core.apis import LENSKIT_API
from ....core.factories import create_factory_from_list
from .lenskit_algorithms import create_biased_mf
from .lenskit_algorithms import create_implicit_mf
from .lenskit_algorithms import create_item_item
from .lenskit_algorithms import create_pop_score
from .lenskit_algorithms import create_random
from .lenskit_algorithms import create_user_user
from .lenskit_params import create_lenskit_params_biased_mf
from .lenskit_params import create_lenskit_params_implicit_mf
from .lenskit_params import create_lenskit_params_item_item
from .lenskit_params import create_lenskit_params_pop_score
from .lenskit_params import create_lenskit_params_random
from .lenskit_params import create_lenskit_params_user_user
from .lenskit_predictor import LensKitPredictor
from .lenskit_recommender import LensKitRecommender

LENSKIT_BIASED_MF = 'BiasedMF'
LENSKIT_IMPLICIT_MF = 'ImplicitMF'
LENSKIT_ITEM_ITEM = 'ItemItem'
LENSKIT_POP_SCORE = 'PopScore'
LENSKIT_RANDOM = 'Random'
LENSKIT_USER_USER = 'UserUser'


def create_lenskit_predictor_factory():
    """Creates the algorithm factory with LensKit predictors.

    Returns:
        (Factory) with available predictors.
    """
    return create_factory_from_list(LENSKIT_API, [
        (LENSKIT_BIASED_MF,
         _create_predictor_biased_mf,
         create_lenskit_params_biased_mf
         ),
        (LENSKIT_IMPLICIT_MF,
         _create_predictor_implicit_mf,
         create_lenskit_params_implicit_mf
         ),
        (LENSKIT_ITEM_ITEM,
         _create_predictor_item_item,
         create_lenskit_params_item_item
         ),
        (LENSKIT_POP_SCORE,
         _create_predictor_pop_score,
         create_lenskit_params_pop_score
         ),
        (LENSKIT_USER_USER,
         _create_predictor_user_user,
         create_lenskit_params_user_user
        )
    ])


def create_lenskit_recommender_factory():
    """Creates the algorithm factory with LensKit recommenders.

    Returns:
        (Factory) with available recommenders.
    """
    return create_factory_from_list(LENSKIT_API, [
        (LENSKIT_BIASED_MF,
         _create_recommender_biased_mf,
         create_lenskit_params_biased_mf
         ),
        (LENSKIT_IMPLICIT_MF,
         _create_recommender_implicit_mf,
         create_lenskit_params_implicit_mf
         ),
        (LENSKIT_ITEM_ITEM,
         _create_recommender_item_item,
         create_lenskit_params_item_item
         ),
        (LENSKIT_POP_SCORE,
         _create_recommender_pop_score,
         create_lenskit_params_pop_score
         ),
        (LENSKIT_RANDOM,
         _create_recommender_random,
         create_lenskit_params_random
         ),
        (LENSKIT_USER_USER,
         _create_recommender_user_user,
         create_lenskit_params_user_user
         )
    ])


def _create_predictor_biased_mf(name, params, **kwargs):
    """Creates the BiasedMF predictor.

    Args:
        name(str): the name of the algorithm.
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
    algo = create_biased_mf(params)

    return LensKitPredictor(algo, name, params, **kwargs)


def _create_predictor_implicit_mf(name, params, **kwargs):
    """Creates the ImplicitMF predictor.

    Args:
        name(str): the name of the algorithm.
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
    algo = create_implicit_mf(params)

    return LensKitPredictor(algo, name, params, **kwargs)


def _create_predictor_item_item(name, params, **kwargs):
    """Creates the ItemItem predictor.

    Args:
        name(str): the name of the algorithm.
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
    algo = create_item_item(params, kwargs['rating_type'])

    return LensKitPredictor(algo, name, params, **kwargs)


def _create_predictor_pop_score(name, params, **kwargs):
    """Creates the PopScore predictor.

    Args:
        name(str): the name of the algorithm.
        params(dict): with the entries:
            score_method

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.

    Returns:
        (LensKitPredictor) wrapper of PopScore.
    """
    algo = create_pop_score(params)

    return LensKitPredictor(algo, name, params, **kwargs)


def _create_predictor_user_user(name, params, **kwargs):
    """Creates the UserUser predictor.

    Args:
        name(str): the name of the algorithm.
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
    algo = create_user_user(params, kwargs['rating_type'])

    return LensKitPredictor(algo, name, params, **kwargs)


def _create_candidate_selector(rated_items_filter):
    """Creates a candidate selector for the specified filter.

    Args:
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.

    Returns:
        (lenskit.CandidateSelector): the corresponding selector.
    """
    return UnratedItemCandidateSelector() if rated_items_filter else AllItemsCandidateSelector()


def _create_recommender_biased_mf(name, params, **kwargs):
    """Creates the BiasedMF recommender.

    Args:
        name(str): the name of the algorithm.
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
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.

    Returns:
        (LensKitRecommender) wrapper of BiasedMF.
    """
    algo = TopN(
        create_biased_mf(params),
        _create_candidate_selector(kwargs['rated_items_filter'])
    )

    return LensKitRecommender(algo, name, params, **kwargs)


def _create_recommender_implicit_mf(name, params, **kwargs):
    """Creates the ImplicitMF recommender.

    Args:
        name(str): the name of the algorithm.
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
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.

    Returns:
        (LensKitRecommender) wrapper of ImplicitMF.
    """
    algo = TopN(
        create_implicit_mf(params),
        _create_candidate_selector(kwargs['rated_items_filter'])
    )

    return LensKitRecommender(algo, name, params, **kwargs)


def _create_recommender_item_item(name, params, **kwargs):
    """Creates the ItemItem recommender.

    Args:
        name(str): the name of the algorithm.
        params(dict): with the entries:
            max_nnbrs,
            min_nbrs,
            min_sim

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.
        rating_type(str): one of:
            explicit,
            implicit

    Returns:
        (LensKitRecommender) wrapper of ItemItem.
    """
    algo = TopN(
        create_item_item(params, kwargs['rating_type']),
        _create_candidate_selector(kwargs['rated_items_filter'])
    )

    return LensKitRecommender(algo, name, params, **kwargs)


def _create_recommender_pop_score(name, params, **kwargs):
    """Creates the PopScore recommender.

    Args:
        name(str): the name of the algorithm.
        params(dict): with the entries:
            score_method

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.

    Returns:
        (LensKitRecommender) wrapper of PopScore.
    """
    algo = TopN(
        create_pop_score(params),
        _create_candidate_selector(kwargs['rated_items_filter'])
    )

    return LensKitRecommender(algo, name, params, **kwargs)


def _create_recommender_random(name, params, **kwargs):
    """Creates the Random recommender.

    Args:
        name(str): the name of the algorithm.
        params(dict): with the entries:
            random_seed

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.

    Returns:
        (LensKitRecommender) wrapper of Random.
    """
    algo = create_random(
        params,
        _create_candidate_selector(kwargs['rated_items_filter'])
    )

    return LensKitRecommender(algo, name, params, **kwargs)


def _create_recommender_user_user(name, params, **kwargs):
    """Creates the UserUser recommender.

    Args:
        name(str): the name of the algorithm.
        params(dict): with the entries:
            max_nnbrs,
            min_nbrs,
            min_sim

    Keyword Args:
        num_threads(int): the max number of threads the algorithm can use.
        rated_items_filter(bool): whether to filter already rated items when
            producing item recommendations.
        rating_type(str): one of:
            explicit,
            implicit

    Returns:
        (LensKitRecommender) wrapper of UserUser.
    """
    algo = TopN(
        create_user_user(params, kwargs['rating_type']),
        _create_candidate_selector(kwargs['rated_items_filter'])
    )

    return LensKitRecommender(algo, name, params, **kwargs)
