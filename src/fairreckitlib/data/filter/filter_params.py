"""This module contains the parameter creation functions for filters.

Functions:

    create_params_funk_svd: create FunkSVD config parameters.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from .filter_constants import FILTER_NUMERICAL, FILTER_CATEGORICAL, FILTER_COUNT
from fairreckitlib.core.config.config_parameters import ConfigParameters

def create_params_numerical() -> ConfigParameters:
    """Create the parameters of the FunkSVD algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_single_option('name', str, '', ['user_age', 'rating', 'timestamp'])
    params.add_number('min', float, 1, (0, 10000000000))
    params.add_number('max', float, 100, (0, 10000000000))
    return params

def create_params_categorical() -> ConfigParameters:
    """Create the parameters of the FunkSVD algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_single_option('name', str, '', ['user_gender', 'user_country', 'user_occupation',
                                                      'artist_gender', 'artist_country', 'artist_genres',
                                                      'movie_genres'])
    params.add_multi_option('values', [], []) #optoins cant know..
    return params

def create_params_count() -> ConfigParameters:
    """Create the parameters of the FunkSVD algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_single_option('name', str, '', ['user_country', 'artist_country', 'artist_genres', 'movie_genres'])
    params.add_number('threshold', int, 100, (1, 10000000000))
    return params
