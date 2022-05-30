"""This module contains functionality to create the implicit recommender factory.

Functions:

    create_recommender_factory: create factory with implicit recommenders.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""


from ....core.config.config_factories import Factory, create_factory_from_list
from ....core.core_constants import IMPLICIT_API
from . import implicit_algorithms
from . import implicit_recommender
from . import implicit_params


def create_recommender_factory() -> Factory:
    """Create the factory with Implicit recommenders.

    Returns:
        the factory with all available recommenders.
    """
    return create_factory_from_list(IMPLICIT_API, [
        (implicit_algorithms.ALTERNATING_LEAST_SQUARES,
         implicit_recommender.create_als,
         implicit_params.create_params_als
         ),
        (implicit_algorithms.BAYESIAN_PERSONALIZED_RANKING,
         implicit_recommender.create_bpr,
         implicit_params.create_params_bpr
         ),
        (implicit_algorithms.LOGISTIC_MATRIX_FACTORIZATION,
         implicit_recommender.create_lmf,
         implicit_params.create_params_lmf
         )
    ])
