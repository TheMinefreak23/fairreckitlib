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
from . import lenskit_metrics, lenskit_predictor


def create_predictor_factory() -> Factory:
    """Create the factory with LensKit predictors.

    Returns:
        the factory with all available predictors.
    """



def create_recommender_factory() -> Factory:
    """Create the factory with LensKit recommenders.

    Returns:
        the factory with all available recommenders.
    """