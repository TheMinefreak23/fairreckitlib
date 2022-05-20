"""This module contains functionality to create the lenskit predictor/recommender factory.

Functions:

    create_predictor_factory: create factory with lenskit predictors.
    create_recommender_factory: create factory with lenskit recommenders.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ....core.apis import LENSKIT_API
from ....core.config_params import create_params_random_seed
from ....core.factories import Factory, create_factory_from_list
from . import lenskit_algorithms
from . import lenskit_params
from . import lenskit_predictor
from . import lenskit_recommender


def create_predictor_factory() -> Factory:
    """Create the factory with LensKit predictors.

    Returns:
        the factory with all available predictors.
    """
    return create_factory_from_list(LENSKIT_API, [
        (lenskit_algorithms.BIASED_MF,
         lenskit_predictor.create_biased_mf,
         lenskit_params.create_params_biased_mf
         ),
        (lenskit_algorithms.IMPLICIT_MF,
         lenskit_predictor.create_implicit_mf,
         lenskit_params.create_params_implicit_mf
         ),
        (lenskit_algorithms.ITEM_ITEM,
         lenskit_predictor.create_item_item,
         lenskit_params.create_params_knn
         ),
        (lenskit_algorithms.POP_SCORE,
         lenskit_predictor.create_pop_score,
         lenskit_params.create_params_pop_score
         ),
        (lenskit_algorithms.USER_USER,
         lenskit_predictor.create_user_user,
         lenskit_params.create_params_knn
        )
    ])


def create_recommender_factory() -> Factory:
    """Create the factory with LensKit recommenders.

    Returns:
        the factory with all available recommenders.
    """
    return create_factory_from_list(LENSKIT_API, [
        (lenskit_algorithms.BIASED_MF,
         lenskit_recommender.create_biased_mf,
         lenskit_params.create_params_biased_mf
         ),
        (lenskit_algorithms.IMPLICIT_MF,
         lenskit_recommender.create_implicit_mf,
         lenskit_params.create_params_implicit_mf
         ),
        (lenskit_algorithms.ITEM_ITEM,
         lenskit_recommender.create_item_item,
         lenskit_params.create_params_knn
         ),
        (lenskit_algorithms.POP_SCORE,
         lenskit_recommender.create_pop_score,
         lenskit_params.create_params_pop_score
         ),
        (lenskit_algorithms.RANDOM,
         lenskit_recommender.create_random,
         create_params_random_seed
         ),
        (lenskit_algorithms.USER_USER,
         lenskit_recommender.create_user_user,
         lenskit_params.create_params_knn
         )
    ])
