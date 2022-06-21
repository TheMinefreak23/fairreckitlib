"""This module contains functionality to create the elliot recommender factory.

Functions:

    create_recommender_factory: create factory with elliot recommenders.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ....core.config.config_factories import Factory, create_factory_from_list
from ....core.config.config_parameters import create_params_random_seed
from ....core.core_constants import ELLIOT_API
from . import elliot_algorithms
from . import elliot_params
from . import elliot_recommender


def create_recommender_factory() -> Factory:
    """Create the factory with Elliot recommenders.

    Returns:
        the factory with all available recommenders.
    """
    return create_factory_from_list(ELLIOT_API, [
        (elliot_algorithms.FUNK_SVD,
         elliot_recommender.create_funk_svd,
         elliot_params.create_params_funk_svd
         ),
        (elliot_algorithms.ITEM_KNN,
         elliot_recommender.create_item_knn,
         elliot_params.create_params_knn
         ),
        (elliot_algorithms.MOST_POP,
         elliot_recommender.create_most_pop,
         None
         ),
        (elliot_algorithms.MULTI_VAE,
         elliot_recommender.create_multi_vae,
         elliot_params.create_params_multi_vae
         ),
        (elliot_algorithms.PURE_SVD,
         elliot_recommender.create_pure_svd,
         elliot_params.create_params_pure_svd
         ),
        (elliot_algorithms.RANDOM,
         elliot_recommender.create_random,
         create_params_random_seed
         ),
        (elliot_algorithms.SVD_PP,
         elliot_recommender.create_svd_pp,
         elliot_params.create_params_svd_pp
         ),
        (elliot_algorithms.USER_KNN,
         elliot_recommender.create_user_knn,
         elliot_params.create_params_knn
         )
    ])
