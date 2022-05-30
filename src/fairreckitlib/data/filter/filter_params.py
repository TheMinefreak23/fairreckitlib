"""This module contains the parameter creation functions for filters.

Functions:

    create_params_funk_svd: create FunkSVD config parameters.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ...core.params.config_parameters import ConfigParameters

# first refactor filter !


def create_params_() -> ConfigParameters:
    """Create the parameters of the FunkSVD algorithm.

    Returns:
        the configuration parameters of the algorithm.
    """
    params = ConfigParameters()
    params.add_number('iterations', int, 10, (1, 50))
    params.add_number('factors', int, 10, (1, 100))
    params.add_number('learning_rate', float, 0.001, (0.0001, 1.0))
    params.add_number('regularization_factors', float, 0.1, (0.0001, 1.0))
    params.add_number('regularization_bias', float, 0.001, (0.0001, 1.0))
    params.add_random_seed('seed')
    return params
