"""This module contains filtering constants that are used in other modules.

Constants:

    KEY_DATA_FILTERS: key that is used to identify data filters.

This program has been developed by students from the bachelor Computer Science at
Utrecht University within the Software Project course.
© Copyright Utrecht University (Department of Information and Computing Sciences)
"""

from typing import Dict, List, Any


FILTER_NUMERICAL = 'numerical'
FILTER_CATEGORICAL = 'categorical'
FILTER_COUNT = 'count'

KEY_DATA_FILTERS = 'filters'


def deduce_filter_type(params: Dict[str, Any]) -> str:
    """Get filter type ('numerical', 'categorical', 'count') from Configuration params.
    
    Args:
        params: Configuration parameters.
    
    Return:
        Either 'numerical', 'categorical' or 'count'. Default 'categorical'
    """
    keys = params.keys
    if 'min' in keys and 'max' in keys:
        return FILTER_NUMERICAL
    elif 'values' in keys:
        return FILTER_CATEGORICAL
    elif 'threshold' in keys:
        return FILTER_COUNT
    return FILTER_CATEGORICAL
