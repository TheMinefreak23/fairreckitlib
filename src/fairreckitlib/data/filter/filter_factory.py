"""This module combines all three types of filters into a factory.

Functions:
    create_filter_factory: Creates a factory of three filter objects.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from ..filter.filter_params import (
    create_params_categorical, create_params_numerical, create_params_count)
from ...core.config.config_factories import Factory, create_factory_from_list
from .filter_constants import KEY_DATA_FILTERS, FILTER_NUMERICAL, FILTER_CATEGORICAL, FILTER_COUNT
from .numerical_filter import create_numerical_filter
from .categorical_filter import create_categorical_filter
from .count_filter import create_count_filter

def create_filter_factory() -> Factory:
    """Create a Factory with the following data filters.

    Numerical: A filter using a min and max value.
    Categorical: A filter using a list of column values.
    Count: A filter using a threshold.

    Returns:
        The factory with all available filters.
    """
    return create_factory_from_list(KEY_DATA_FILTERS, [
        (FILTER_NUMERICAL,
         create_numerical_filter,
         create_params_numerical
         ),
        (FILTER_CATEGORICAL,
         create_categorical_filter,
         create_params_categorical
         ),
        (FILTER_COUNT,
         create_count_filter, # count categorical params (country)
         create_params_count
         )
    ])
