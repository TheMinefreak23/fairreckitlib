"""This module contains functionality to create a splitters factory.

Functions:

    create_split_factory: create a factory with data splitters.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ...core.params.config_parameters import create_params_random_seed
from ...core.factories import Factory, create_factory_from_list
from .random_splitter import create_random_splitter
from .split_constants import KEY_SPLITTING, SPLIT_RANDOM, SPLIT_TEMPORAL
from .temporal_splitter import create_temporal_splitter


def create_split_factory() -> Factory:
    """Create a Factory with all data splitters.

    Returns:
        the factory with all available splitters.
    """
    return create_factory_from_list(KEY_SPLITTING, [
        (SPLIT_RANDOM,
         create_random_splitter,
         create_params_random_seed
         ),
        (SPLIT_TEMPORAL,
         create_temporal_splitter,
         None
         )
    ])
